#!/usr/bin/env python3
"""Migrate legacy MinIO objects and local evidence images to LocalFS.

The migration deliberately reads MinIO through its API.  Files below a MinIO
data volume (for example ``xl.meta``) are an implementation detail and must not
be copied as application objects.

The destination keeps every logical object key unchanged so the legacy
``minio_path`` database columns continue to work as neutral storage keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Dict, Iterable, Iterator, Optional, Tuple
from urllib.parse import urlsplit


CHUNK_SIZE = 1024 * 1024
SUCCESS_STATUSES = {"migrated", "already_present", "verified", "resumed", "already_in_place"}
RESERVED_NAMESPACES = {".staging", ".trash"}


class MigrationError(RuntimeError):
    """A source object could not be migrated or verified safely."""


@dataclass(frozen=True)
class SourceObject:
    source: str
    storage_key: str
    size_bytes: int
    source_etag: str = ""
    bucket: str = ""
    local_path: Optional[Path] = None

    @property
    def identity(self) -> str:
        return f"{self.source}\0{self.bucket}\0{self.storage_key}"


class JsonlManifest:
    """Append-only manifest with fsync so interrupted runs are resumable."""

    def __init__(self, path: Path):
        self.path = path
        self.previous: Dict[str, dict] = {}
        self._handle: Optional[BinaryIO] = None

    def __enter__(self) -> "JsonlManifest":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as source:
                for line_number, raw in enumerate(source, start=1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        print(
                            f"[WARN] manifest line {line_number} is invalid and ignored: {exc}",
                            file=sys.stderr,
                        )
                        continue
                    identity = record.get("identity")
                    if identity:
                        self.previous[str(identity)] = record
        self._handle = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def append(self, item: SourceObject, status: str, **details) -> dict:
        if self._handle is None:
            raise RuntimeError("manifest is not open")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "identity": item.identity,
            "source": item.source,
            "bucket": item.bucket or None,
            "storage_key": item.storage_key,
            "source_etag": item.source_etag or None,
            "expected_size_bytes": item.size_bytes,
            "status": status,
            **details,
        }
        self._handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.previous[item.identity] = record
        return record


def normalize_storage_key(raw_key: str) -> str:
    if not isinstance(raw_key, str):
        raise MigrationError("storage key must be a string")
    if not raw_key or raw_key != raw_key.strip() or "\x00" in raw_key or "\\" in raw_key:
        raise MigrationError(f"unsafe storage key: {raw_key!r}")
    parts = raw_key.split("/")
    key = PurePosixPath(raw_key)
    if key.is_absolute() or any(part in ("", ".", "..") for part in parts):
        raise MigrationError(f"unsafe storage key: {raw_key!r}")
    if key.parts[0] in RESERVED_NAMESPACES:
        raise MigrationError(f"reserved storage namespace: {raw_key!r}")
    return key.as_posix()


def ensure_directory(path: Path, root: Path) -> None:
    """Create a path component-by-component and reject symlink parents."""

    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise MigrationError(f"destination escapes artifact root: {path}") from exc

    current = root
    for part in relative.parts:
        current = current / part
        try:
            current.mkdir(mode=0o750)
        except FileExistsError:
            pass
        if current.is_symlink() or not current.is_dir():
            raise MigrationError(f"unsafe destination directory: {current}")


def target_path(root: Path, storage_key: str, *, create_parent: bool) -> Path:
    key = normalize_storage_key(storage_key)
    candidate = root.joinpath(*PurePosixPath(key).parts)
    if create_parent:
        ensure_directory(candidate.parent, root)
    try:
        resolved_parent = candidate.parent.resolve(strict=True)
    except FileNotFoundError:
        resolved_parent = candidate.parent.resolve(strict=False)
    if os.path.commonpath((str(root), str(resolved_parent))) != str(root):
        raise MigrationError(f"destination escapes artifact root: {storage_key!r}")
    if candidate.is_symlink():
        raise MigrationError(f"destination is a symlink: {storage_key!r}")
    return candidate


def fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Some filesystems do not support directory fsync.  File fsync still ran.
        pass


def digest_stream(chunks: Iterable[bytes]) -> Tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for chunk in chunks:
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise MigrationError("source returned non-binary data")
        if not chunk:
            continue
        digest.update(chunk)
        size += len(chunk)
    return size, digest.hexdigest()


def digest_file(path: Path) -> Tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while True:
            chunk = source.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def iter_file_chunks(source: BinaryIO) -> Iterator[bytes]:
    while True:
        chunk = source.read(CHUNK_SIZE)
        if not chunk:
            return
        yield chunk


def write_atomically(
    root: Path,
    storage_key: str,
    chunks: Iterable[bytes],
    expected_size: int,
) -> Tuple[str, int, str]:
    """Write without replacing a conflicting destination object."""

    staging = root / ".staging"
    ensure_directory(staging, root)
    destination = target_path(root, storage_key, create_parent=True)
    fd, raw_temp = tempfile.mkstemp(prefix="migration-", suffix=".part", dir=staging)
    temporary = Path(raw_temp)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, "wb") as output:
            for chunk in chunks:
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise MigrationError("source returned non-binary data")
                if not chunk:
                    continue
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o640)

        if size != expected_size:
            raise MigrationError(f"source size changed while reading: expected {expected_size}, got {size}")
        sha256 = digest.hexdigest()

        # A hard link publishes the completed inode atomically and fails if the
        # destination appeared concurrently.  This avoids os.replace overwriting
        # a different object during a migration rerun.
        try:
            os.link(temporary, destination)
            status = "migrated"
            fsync_directory(destination.parent)
        except FileExistsError:
            existing_size, existing_sha256 = digest_file(destination)
            if existing_size != size or existing_sha256 != sha256:
                raise MigrationError(
                    "destination conflict: existing object has different size or SHA-256"
                )
            status = "already_present"
        return status, size, sha256
    finally:
        temporary.unlink(missing_ok=True)


def verify_destination(
    root: Path,
    item: SourceObject,
    chunks: Iterable[bytes],
) -> Tuple[int, str]:
    source_size, source_sha256 = digest_stream(chunks)
    if source_size != item.size_bytes:
        raise MigrationError(
            f"source size changed while reading: expected {item.size_bytes}, got {source_size}"
        )
    destination = target_path(root, item.storage_key, create_parent=False)
    if not destination.is_file():
        raise MigrationError("destination object is missing")
    destination_size, destination_sha256 = digest_file(destination)
    if destination_size != source_size or destination_sha256 != source_sha256:
        raise MigrationError("destination size or SHA-256 does not match source")
    return source_size, source_sha256


def resume_is_valid(root: Path, item: SourceObject, previous: Optional[dict]) -> bool:
    if not previous or previous.get("status") not in SUCCESS_STATUSES:
        return False
    if int(previous.get("size_bytes", -1)) != item.size_bytes:
        return False
    previous_etag = str(previous.get("source_etag") or "")
    if item.source_etag and previous_etag != item.source_etag:
        return False
    previous_sha256 = str(previous.get("sha256") or "")
    if len(previous_sha256) != 64:
        return False
    destination = target_path(root, item.storage_key, create_parent=False)
    if not destination.is_file():
        return False
    size, sha256 = digest_file(destination)
    return size == item.size_bytes and sha256 == previous_sha256


def parse_endpoint(endpoint: str, explicit_secure: Optional[bool]) -> Tuple[str, bool]:
    value = endpoint.strip()
    if not value:
        raise MigrationError("MinIO endpoint is empty")
    if "://" in value:
        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path not in ("", "/"):
            raise MigrationError("MinIO endpoint must be host:port or an HTTP(S) origin")
        secure = explicit_secure if explicit_secure is not None else parsed.scheme == "https"
        return parsed.netloc, secure
    return value.rstrip("/"), bool(explicit_secure)


def env_bool(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise MigrationError(f"{name} must be true or false")


def create_minio_client(args):
    try:
        from minio import Minio
    except ImportError as exc:
        raise MigrationError(
            "MinIO SDK is not installed; run: pip install -r requirements-migration.txt"
        ) from exc

    endpoint, secure = parse_endpoint(args.endpoint, args.secure)
    return Minio(
        endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        secure=secure,
        cert_check=not args.no_cert_check,
    )


def list_minio_objects(client, bucket: str, prefix: str) -> Iterator[SourceObject]:
    if not client.bucket_exists(bucket):
        raise MigrationError(f"MinIO bucket does not exist: {bucket}")
    for obj in client.list_objects(bucket, prefix=prefix or None, recursive=True):
        if getattr(obj, "is_dir", False):
            continue
        storage_key = normalize_storage_key(obj.object_name)
        yield SourceObject(
            source="minio",
            bucket=bucket,
            storage_key=storage_key,
            size_bytes=int(obj.size or 0),
            source_etag=str(obj.etag or "").strip('"'),
        )


@contextmanager
def open_minio_chunks(client, item: SourceObject) -> Iterator[Iterable[bytes]]:
    response = client.get_object(item.bucket, item.storage_key)
    try:
        yield iter_file_chunks(response)
    finally:
        response.close()
        response.release_conn()


def list_legacy_images(source_root: Path) -> Iterator[SourceObject]:
    if not source_root.exists():
        print(f"[INFO] legacy image directory does not exist, skipping: {source_root}")
        return
    if source_root.is_symlink() or not source_root.is_dir():
        raise MigrationError(f"legacy image source is not a safe directory: {source_root}")

    for source_path in sorted(source_root.rglob("*")):
        if source_path.is_symlink():
            raise MigrationError(f"legacy image source contains a symlink: {source_path}")
        if not source_path.is_file():
            continue
        relative = source_path.relative_to(source_root).as_posix()
        storage_key = normalize_storage_key(f"images/{relative}")
        info = source_path.stat()
        yield SourceObject(
            source="legacy_images",
            storage_key=storage_key,
            size_bytes=info.st_size,
            source_etag=f"{info.st_mtime_ns}:{info.st_size}",
            local_path=source_path,
        )


@contextmanager
def open_local_chunks(item: SourceObject) -> Iterator[Iterable[bytes]]:
    if item.local_path is None:
        raise MigrationError("legacy image source has no local path")
    with item.local_path.open("rb") as source:
        yield iter_file_chunks(source)


def process_item(
    item: SourceObject,
    root: Path,
    manifest: JsonlManifest,
    summary: Counter,
    open_chunks,
    *,
    dry_run: bool,
    verify_only: bool,
    resume: bool,
) -> None:
    label = f"{item.source}:{item.bucket + '/' if item.bucket else ''}{item.storage_key}"
    try:
        if dry_run:
            manifest.append(item, "dry_run", size_bytes=item.size_bytes)
            summary["dry_run"] += 1
            print(f"[DRY-RUN] {label} ({item.size_bytes} bytes)")
            return

        if resume and not verify_only and resume_is_valid(root, item, manifest.previous.get(item.identity)):
            previous = manifest.previous[item.identity]
            manifest.append(
                item,
                "resumed",
                size_bytes=item.size_bytes,
                sha256=previous["sha256"],
            )
            summary["resumed"] += 1
            print(f"[RESUME] {label}")
            return

        destination = target_path(root, item.storage_key, create_parent=not verify_only)
        if item.local_path is not None and item.local_path.resolve() == destination.resolve():
            size, sha256 = digest_file(destination)
            if size != item.size_bytes:
                raise MigrationError("in-place image size changed while scanning")
            status = "verified" if verify_only else "already_in_place"
        else:
            with open_chunks(item) as chunks:
                if verify_only:
                    size, sha256 = verify_destination(root, item, chunks)
                    status = "verified"
                else:
                    status, size, sha256 = write_atomically(
                        root, item.storage_key, chunks, item.size_bytes
                    )

        manifest.append(item, status, size_bytes=size, sha256=sha256)
        summary[status] += 1
        print(f"[{status.upper()}] {label} sha256={sha256}")
    except Exception as exc:
        manifest.append(item, "error", size_bytes=item.size_bytes, error=str(exc))
        summary["error"] += 1
        print(f"[ERROR] {label}: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Stream legacy MinIO objects and app/storage/images into LocalFS ArtifactStore."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=repo_root / "volumes" / "artifacts",
        help="LocalFS ArtifactStore root (default: %(default)s)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "volumes" / "migration" / "minio-to-localfs.jsonl",
        help="append-only JSONL manifest (default: %(default)s)",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("MIGRATION_MINIO_ENDPOINT", os.getenv("MINIO_ENDPOINT", "localhost:9002")),
        help="legacy MinIO endpoint, without a path",
    )
    parser.add_argument(
        "--access-key",
        default=os.getenv("MIGRATION_MINIO_ACCESS_KEY", os.getenv("MINIO_ACCESS_KEY", "admin")),
    )
    parser.add_argument(
        "--secret-key",
        default=os.getenv("MIGRATION_MINIO_SECRET_KEY", os.getenv("MINIO_SECRET_KEY", "password")),
    )
    parser.add_argument(
        "--secure",
        action=argparse.BooleanOptionalAction,
        default=env_bool("MIGRATION_MINIO_SECURE"),
        help="use HTTPS (a URL scheme also supplies the default)",
    )
    parser.add_argument("--no-cert-check", action="store_true", help="disable TLS certificate checks")
    parser.add_argument("--bucket", default="documents", help="legacy MinIO bucket")
    parser.add_argument("--prefix", default="", help="optional MinIO object prefix")
    parser.add_argument("--skip-minio", action="store_true", help="only migrate legacy local images")
    parser.add_argument(
        "--legacy-images-dir",
        type=Path,
        default=repo_root / "app" / "storage" / "images",
        help="old IMAGE_STORAGE_DIR to migrate under the images/ namespace",
    )
    parser.add_argument(
        "--skip-legacy-images", action="store_true", help="do not migrate app/storage/images"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="list work without reading object bodies")
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="stream and hash both sources and destinations without writing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse successful manifest entries after rechecking destination SHA-256",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.skip_minio and args.skip_legacy_images:
        parser.error("both sources are disabled")

    root_argument = args.artifact_root.expanduser()
    if root_argument.is_symlink():
        raise MigrationError(f"artifact root must not be a symlink: {root_argument}")
    root = root_argument.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    if not root.is_dir():
        raise MigrationError(f"artifact root is not a safe directory: {root}")
    ensure_directory(root / ".staging", root)

    summary: Counter = Counter()
    with JsonlManifest(args.manifest.expanduser().resolve()) as manifest:
        if not args.skip_minio:
            try:
                client = create_minio_client(args)
                for item in list_minio_objects(client, args.bucket, args.prefix):
                    process_item(
                        item,
                        root,
                        manifest,
                        summary,
                        lambda current: open_minio_chunks(client, current),
                        dry_run=args.dry_run,
                        verify_only=args.verify_only,
                        resume=args.resume,
                    )
            except Exception as exc:
                summary["error"] += 1
                print(f"[ERROR] MinIO inventory failed: {exc}", file=sys.stderr)

        if not args.skip_legacy_images:
            try:
                for item in list_legacy_images(args.legacy_images_dir.expanduser().resolve()):
                    process_item(
                        item,
                        root,
                        manifest,
                        summary,
                        open_local_chunks,
                        dry_run=args.dry_run,
                        verify_only=args.verify_only,
                        resume=args.resume,
                    )
            except Exception as exc:
                summary["error"] += 1
                print(f"[ERROR] legacy image inventory failed: {exc}", file=sys.stderr)

    ordered = ", ".join(f"{key}={value}" for key, value in sorted(summary.items())) or "no objects"
    print(f"[SUMMARY] {ordered}")
    print(f"[MANIFEST] {args.manifest.expanduser().resolve()}")
    return 1 if summary["error"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[ERROR] migration interrupted", file=sys.stderr)
        raise SystemExit(130)
    except MigrationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
