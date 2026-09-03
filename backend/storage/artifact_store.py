"""Artifact storage abstraction and the local filesystem implementation.

The application stores only logical ``storage_key`` values.  The local backend
keeps those keys below one configured root, writes through a staging directory,
and moves deleted objects to a recoverable trash area.  Callers must not join
user-controlled paths themselves.
"""

from __future__ import annotations

import hashlib
import io
import errno
import logging
import mimetypes
import os
import re
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping, Optional, Union

from backend.config.config import config

logger = logging.getLogger(__name__)

FileData = Union[bytes, bytearray, memoryview, BinaryIO]


class ArtifactStoreError(RuntimeError):
    """Base error for artifact storage operations."""


class InvalidStorageKeyError(ArtifactStoreError, ValueError):
    """Raised when a logical key could escape the artifact root."""


class ArtifactNotFoundError(ArtifactStoreError, FileNotFoundError):
    """Raised when an artifact key does not exist."""


@dataclass(frozen=True)
class ArtifactMetadata:
    storage_key: str
    size_bytes: int
    sha256: str
    content_type: str = "application/octet-stream"
    original_name: str = ""
    modified_at: Optional[datetime] = None


class ArtifactStore(ABC):
    """Backend-neutral artifact operations used by parsers and API routes."""

    @abstractmethod
    def upload_file(
        self,
        storage_key: str,
        file_data: FileData,
        *,
        content_type: Optional[str] = None,
        original_name: Optional[str] = None,
    ) -> ArtifactMetadata:
        raise NotImplementedError

    @abstractmethod
    def get_file(self, storage_key: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def open_file(self, storage_key: str, mode: str = "rb"):
        raise NotImplementedError

    @abstractmethod
    def read_range(self, storage_key: str, start: int = 0, length: Optional[int] = None) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def delete_file(self, storage_key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        """Delete regular files below one safe logical prefix."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, storage_key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def stat(self, storage_key: str) -> ArtifactMetadata:
        raise NotImplementedError

    @abstractmethod
    def materialize(self, storage_key: str, suffix: str = "") -> Iterator[str]:
        """Yield a local read-only path for a stored artifact."""
        raise NotImplementedError


class LocalFSArtifactStore(ArtifactStore):
    """Atomic, path-confined artifact storage on a persistent POSIX directory."""

    _INTERNAL_DIRS = {".staging", ".trash"}

    def __init__(
        self,
        root_dir: Union[str, os.PathLike],
        *,
        trash_enabled: bool = True,
        file_mode: int = 0o640,
        dir_mode: int = 0o750,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.trash_enabled = trash_enabled
        self.file_mode = file_mode
        self.dir_mode = dir_mode
        self.staging_dir = self.root_dir / ".staging"
        self.trash_dir = self.root_dir / ".trash"
        self._mkdir(self.root_dir)
        self._mkdir(self.staging_dir)
        if self.trash_enabled:
            self._mkdir(self.trash_dir)

    def _mkdir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(self.dir_mode)
        except OSError:
            logger.debug("Unable to apply directory mode to %s", path, exc_info=True)

    @staticmethod
    def normalize_key(storage_key: str) -> str:
        if not isinstance(storage_key, str):
            raise InvalidStorageKeyError("storage_key must be a string")
        # Logical keys are exact identifiers.  Silently trimming a legacy key
        # could make one database row read or delete a different object.
        if storage_key != storage_key.strip():
            raise InvalidStorageKeyError("storage_key has leading or trailing whitespace")
        value = storage_key
        if not value or "\x00" in value or "\\" in value:
            raise InvalidStorageKeyError("storage_key is empty or contains invalid characters")
        raw_parts = value.split("/")
        key = PurePosixPath(value)
        if key.is_absolute() or any(part in ("", ".", "..") for part in raw_parts):
            raise InvalidStorageKeyError(f"unsafe storage_key: {storage_key!r}")
        if key.parts[0] in LocalFSArtifactStore._INTERNAL_DIRS:
            raise InvalidStorageKeyError("storage_key uses a reserved namespace")
        return key.as_posix()

    def _ensure_parent(self, key: str) -> Path:
        current = self.root_dir
        for part in PurePosixPath(key).parts[:-1]:
            current = current / part
            # ``exist_ok`` closes the normal race where two uploads create the
            # same new document directory.  Validate after mkdir as well: an
            # existing symlink must never be accepted as a directory.
            try:
                current.mkdir(mode=self.dir_mode)
                self._fsync_dir(current.parent)
            except FileExistsError as exc:
                if current.is_symlink() or not current.is_dir():
                    raise InvalidStorageKeyError(
                        f"unsafe parent in storage_key: {key!r}"
                    ) from exc
            except OSError as exc:
                if current.is_symlink() or not current.is_dir():
                    raise InvalidStorageKeyError(
                        f"unsafe parent in storage_key: {key!r}"
                    ) from exc
                raise
            if current.is_symlink() or not current.is_dir():
                raise InvalidStorageKeyError(f"unsafe parent in storage_key: {key!r}")
        return current

    def _path_for(self, storage_key: str, *, create_parent: bool = False) -> Path:
        key = self.normalize_key(storage_key)
        candidate = self.root_dir.joinpath(*PurePosixPath(key).parts)
        if create_parent:
            self._ensure_parent(key)
        try:
            resolved_parent = candidate.parent.resolve(strict=True)
        except FileNotFoundError:
            resolved_parent = candidate.parent.resolve(strict=False)
        if os.path.commonpath((str(self.root_dir), str(resolved_parent))) != str(self.root_dir):
            raise InvalidStorageKeyError(f"storage_key escapes artifact root: {storage_key!r}")
        if candidate.is_symlink():
            raise InvalidStorageKeyError(f"storage_key resolves to a symlink: {storage_key!r}")
        return candidate

    @staticmethod
    def _stream(file_data: FileData) -> BinaryIO:
        if isinstance(file_data, (bytes, bytearray, memoryview)):
            return io.BytesIO(bytes(file_data))
        if not hasattr(file_data, "read"):
            raise TypeError("file_data must be bytes or a binary file-like object")
        try:
            file_data.seek(0)
        except (AttributeError, OSError, io.UnsupportedOperation):
            pass
        return file_data

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            unsupported = {
                errno.EBADF,
                errno.EINVAL,
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }
            if exc.errno in unsupported:
                logger.debug("Directory fsync is not available for %s", path, exc_info=True)
                return
            raise

    def upload_file(
        self,
        storage_key: str,
        file_data: FileData,
        *,
        content_type: Optional[str] = None,
        original_name: Optional[str] = None,
    ) -> ArtifactMetadata:
        key = self.normalize_key(storage_key)
        target = self._path_for(key, create_parent=True)
        stream = self._stream(file_data)
        digest = hashlib.sha256()
        size = 0
        temp_path: Optional[Path] = None
        try:
            fd, raw_temp = tempfile.mkstemp(prefix="artifact-", suffix=".part", dir=self.staging_dir)
            temp_path = Path(raw_temp)
            with os.fdopen(fd, "wb") as output:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    if isinstance(chunk, str):
                        raise TypeError("file_data must be opened in binary mode")
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            try:
                temp_path.chmod(self.file_mode)
            except OSError:
                logger.debug("Unable to apply file mode to %s", temp_path, exc_info=True)
            os.replace(temp_path, target)
            self._fsync_dir(target.parent)
        except Exception as exc:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            if isinstance(exc, (ArtifactStoreError, TypeError, ValueError)):
                raise
            raise ArtifactStoreError(f"failed to write artifact {key}: {exc}") from exc

        guessed_type = content_type or mimetypes.guess_type(original_name or key)[0]
        return ArtifactMetadata(
            storage_key=key,
            size_bytes=size,
            sha256=digest.hexdigest(),
            content_type=guessed_type or "application/octet-stream",
            original_name=original_name or PurePosixPath(key).name,
            modified_at=datetime.fromtimestamp(target.stat().st_mtime, tz=timezone.utc),
        )

    def get_file(self, storage_key: str) -> bytes:
        path = self._path_for(storage_key)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise ArtifactNotFoundError(f"artifact not found: {storage_key}") from exc
        except OSError as exc:
            raise ArtifactStoreError(f"failed to read artifact {storage_key}: {exc}") from exc

    def open_file(self, storage_key: str, mode: str = "rb"):
        if mode not in ("rb", "r"):
            raise ValueError("artifacts can only be opened read-only")
        path = self._path_for(storage_key)
        if not path.is_file():
            raise ArtifactNotFoundError(f"artifact not found: {storage_key}")
        return path.open(mode)

    def read_range(self, storage_key: str, start: int = 0, length: Optional[int] = None) -> bytes:
        if start < 0:
            raise ValueError("start must be non-negative")
        if length is not None and length < 0:
            raise ValueError("length must be non-negative")
        with self.open_file(storage_key, "rb") as source:
            source.seek(start)
            return source.read() if length is None else source.read(length)

    def delete_file(self, storage_key: str) -> bool:
        key = self.normalize_key(storage_key)
        source = self._path_for(key)
        if not source.exists():
            return False
        if not source.is_file():
            raise ArtifactStoreError(f"artifact key is not a regular file: {key}")
        try:
            if self.trash_enabled:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                destination = self.trash_dir / stamp / uuid.uuid4().hex
                destination = destination.joinpath(*PurePosixPath(key).parts)
                self._mkdir(destination.parent)
                os.replace(source, destination)
                self._fsync_dir(destination.parent)
            else:
                source.unlink()
            self._fsync_dir(source.parent)
            return True
        except FileNotFoundError:
            # Another worker may have completed the same idempotent delete
            # after our existence check.
            return False
        except OSError as exc:
            raise ArtifactStoreError(f"failed to delete artifact {key}: {exc}") from exc

    def delete_prefix(self, prefix: str) -> int:
        """Move ordinary files below ``prefix`` into trash one by one.

        The prefix is a logical key, so an empty prefix and internal storage
        namespaces are rejected by ``normalize_key``.  Directory symlinks are
        not followed and symlink files are not treated as artifacts.
        """

        key = self.normalize_key(prefix)
        start = self._path_for(key)
        if not start.exists():
            return 0
        if start.is_symlink():
            raise InvalidStorageKeyError(f"prefix resolves to a symlink: {prefix!r}")
        if start.is_file():
            return int(self.delete_file(key))
        if not start.is_dir():
            return 0

        storage_keys = []
        for directory, dir_names, file_names in os.walk(start, followlinks=False):
            directory_path = Path(directory)
            # Do not even retain symlinked directories in os.walk's pending
            # list.  ``followlinks=False`` is kept as a second line of defence.
            dir_names[:] = [
                name for name in dir_names
                if not (directory_path / name).is_symlink()
            ]
            for file_name in file_names:
                path = directory_path / file_name
                if path.is_symlink() or not path.is_file():
                    continue
                storage_keys.append(path.relative_to(self.root_dir).as_posix())

        deleted = 0
        for storage_key in sorted(storage_keys):
            if self.delete_file(storage_key):
                deleted += 1
        return deleted

    def exists(self, storage_key: str) -> bool:
        return self._path_for(storage_key).is_file()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def stat(self, storage_key: str) -> ArtifactMetadata:
        key = self.normalize_key(storage_key)
        path = self._path_for(key)
        try:
            info = path.stat()
        except FileNotFoundError as exc:
            raise ArtifactNotFoundError(f"artifact not found: {key}") from exc
        if not path.is_file():
            raise ArtifactNotFoundError(f"artifact not found: {key}")
        return ArtifactMetadata(
            storage_key=key,
            size_bytes=info.st_size,
            sha256=self._sha256(path),
            content_type=mimetypes.guess_type(key)[0] or "application/octet-stream",
            original_name=PurePosixPath(key).name,
            modified_at=datetime.fromtimestamp(info.st_mtime, tz=timezone.utc),
        )

    @contextmanager
    def materialize(self, storage_key: str, suffix: str = "") -> Iterator[str]:
        path = self._path_for(storage_key)
        if not path.is_file():
            raise ArtifactNotFoundError(f"artifact not found: {storage_key}")
        # Always materialize a private snapshot.  Yielding the persistent path
        # lets a concurrent recoverable delete move it away while a parser is
        # still opening related files.
        materialized_suffix = suffix or path.suffix
        fd, raw_temp = tempfile.mkstemp(
            prefix="materialized-", suffix=materialized_suffix, dir=self.staging_dir
        )
        temp_path = Path(raw_temp)
        try:
            with os.fdopen(fd, "wb") as output, path.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            yield str(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def iter_keys(self, prefix: str = "") -> Iterator[str]:
        if prefix:
            normalized_prefix = self.normalize_key(prefix)
            start = self._path_for(normalized_prefix)
        else:
            start = self.root_dir
        if start.is_file():
            yield start.relative_to(self.root_dir).as_posix()
            return
        if not start.exists():
            return
        for path in start.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root_dir)
            if relative.parts and relative.parts[0] in self._INTERNAL_DIRS:
                continue
            yield relative.as_posix()


def create_artifact_store() -> ArtifactStore:
    backend = config.ARTIFACT_BACKEND.lower().strip()
    if backend != "local":
        raise ArtifactStoreError(f"unsupported ARTIFACT_BACKEND: {backend}")
    return LocalFSArtifactStore(
        config.ARTIFACT_STORAGE_DIR,
        trash_enabled=config.ARTIFACT_TRASH_ENABLED,
    )


artifact_store = create_artifact_store()


def storage_key_from(record: Mapping[str, Any]) -> str:
    """Read the neutral key while accepting the legacy ``minio_path`` column.

    The database column is intentionally retained during the compatibility
    phase so existing deployments do not require an unsafe in-place rewrite.
    New application code must treat its value as an ArtifactStore key.
    """

    key = str(record.get("storage_key") or record.get("minio_path") or "")
    if not key:
        raise ArtifactNotFoundError("record has no artifact storage_key")
    return LocalFSArtifactStore.normalize_key(key)


def safe_original_name(file_name: Optional[str], fallback: str = "document") -> str:
    """Return a display-only basename; it is never used as a physical path."""

    value = str(file_name or "").replace("\\", "/").split("/")[-1].strip()
    value = "".join(char for char in value if ord(char) >= 32 and char != "\x7f")
    if value in ("", ".", ".."):
        value = fallback
    return value[:255]


def source_artifact_key(prefix: str, doc_id: str, original_name: Optional[str]) -> str:
    """Build a generated key that never embeds a user-controlled filename."""

    name = safe_original_name(original_name)
    suffix = Path(name).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,15}", suffix):
        suffix = ""
    return LocalFSArtifactStore.normalize_key(f"{prefix}/{doc_id}/source{suffix}")
