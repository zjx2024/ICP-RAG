#!/usr/bin/env python3
"""Read-only PostgreSQL <-> LocalFS ArtifactStore consistency audit.

The audit intentionally does not import ``backend.storage.artifact_store``:
that module creates the configured LocalFS directories during import.  A
deployment acceptance check must be observational only, including when the
configured root is absent or incorrectly mounted.
"""

from __future__ import annotations

import argparse
import json
import os
import stat as stat_module
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


EXIT_OK = 0
EXIT_INCONSISTENT = 1
EXIT_AUDIT_ERROR = 2
INTERNAL_NAMESPACES = {".staging", ".trash"}


class AuditError(RuntimeError):
    """The audit could not complete and therefore cannot attest consistency."""


class InvalidStorageKeyError(ValueError):
    """A database value is not a confined ArtifactStore logical key."""


@dataclass(frozen=True)
class ReferenceSource:
    table: str
    owner_column: str


REFERENCE_SOURCES: Tuple[ReferenceSource, ...] = (
    ReferenceSource("documents", "doc_id"),
    ReferenceSource("icp_documents", "doc_id"),
    ReferenceSource("document_images", "image_id"),
)


@dataclass(frozen=True)
class ArtifactReference:
    table: str
    owner_column: str
    owner_id: str
    storage_key: str


@dataclass(frozen=True)
class ReferenceIssue:
    table: str
    owner_column: str
    owner_id: str
    storage_key: str
    reason: str


@dataclass
class AuditResult:
    artifact_root: str
    references_scanned: int = 0
    unique_storage_keys: int = 0
    reference_issues: List[ReferenceIssue] = field(default_factory=list)
    orphan_scan_enabled: bool = False
    orphan_keys: List[str] = field(default_factory=list)

    @property
    def is_consistent(self) -> bool:
        return not self.reference_issues and not self.orphan_keys

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_root": self.artifact_root,
            "consistent": self.is_consistent,
            "references_scanned": self.references_scanned,
            "unique_storage_keys": self.unique_storage_keys,
            "reference_issues": [asdict(issue) for issue in self.reference_issues],
            "orphan_scan_enabled": self.orphan_scan_enabled,
            "orphan_keys": list(self.orphan_keys),
            "summary": {
                "reference_issues": len(self.reference_issues),
                "orphans": len(self.orphan_keys),
            },
        }


def normalize_storage_key(storage_key: str) -> str:
    """Apply the same path contract as LocalFSArtifactStore without writes."""

    if not isinstance(storage_key, str):
        raise InvalidStorageKeyError("storage_key must be a string")
    if storage_key != storage_key.strip():
        raise InvalidStorageKeyError("storage_key must not contain surrounding whitespace")
    value = storage_key
    if not value or "\x00" in value or "\\" in value:
        raise InvalidStorageKeyError("storage_key is empty or contains invalid characters")
    raw_parts = value.split("/")
    key = PurePosixPath(value)
    if key.is_absolute() or any(part in ("", ".", "..") for part in raw_parts):
        raise InvalidStorageKeyError("storage_key is not a safe relative path")
    if key.parts[0] in INTERNAL_NAMESPACES:
        raise InvalidStorageKeyError("storage_key uses an internal namespace")
    return key.as_posix()


def validate_artifact_root(root_argument: Path) -> Path:
    """Resolve an existing ArtifactStore root without creating or changing it."""

    raw_root = root_argument.expanduser()
    if raw_root.is_symlink():
        raise AuditError(f"artifact root must not be a symlink: {raw_root}")
    try:
        root = raw_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise AuditError(f"artifact root does not exist: {raw_root}") from exc
    except OSError as exc:
        raise AuditError(f"cannot resolve artifact root {raw_root}: {exc}") from exc
    if not root.is_dir():
        raise AuditError(f"artifact root is not a directory: {root}")
    return root


def _reference_status(root: Path, storage_key: str) -> Optional[str]:
    """Return ``None`` for a usable regular file, otherwise a stable reason."""

    current = root
    parts = PurePosixPath(storage_key).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            metadata = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError):
            return "not_found"
        except OSError as exc:
            raise AuditError(f"cannot inspect artifact {storage_key!r}: {exc}") from exc
        if stat_module.S_ISLNK(metadata.st_mode):
            return "symlink_not_allowed"
        is_last = index == len(parts) - 1
        if is_last:
            if not stat_module.S_ISREG(metadata.st_mode):
                return "not_a_regular_file"
        elif not stat_module.S_ISDIR(metadata.st_mode):
            return "not_found"
    return None


def read_references(connection: Any) -> List[ArtifactReference]:
    """Read all non-empty legacy columns in a database read-only transaction."""

    try:
        connection.set_session(readonly=True, autocommit=False)
    except Exception as exc:
        raise AuditError(f"cannot enable PostgreSQL read-only transaction: {exc}") from exc

    references: List[ArtifactReference] = []
    try:
        with connection.cursor() as cursor:
            for source in REFERENCE_SOURCES:
                # Table and column identifiers are fixed constants above, never user input.
                cursor.execute(
                    f'SELECT "{source.owner_column}"::text, "minio_path" '
                    f'FROM "{source.table}" '
                    'WHERE "minio_path" IS NOT NULL AND BTRIM("minio_path") <> \'\''
                )
                for owner_id, storage_key in cursor:
                    references.append(
                        ArtifactReference(
                            table=source.table,
                            owner_column=source.owner_column,
                            owner_id=str(owner_id),
                            storage_key=str(storage_key),
                        )
                    )
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError(f"cannot read PostgreSQL artifact references: {exc}") from exc
    return references


def iter_local_artifacts(root: Path) -> Iterator[str]:
    """Yield regular local artifacts, never following symlinks or internal dirs."""

    def raise_walk_error(error: OSError) -> None:
        raise AuditError(f"cannot scan ArtifactStore: {error}")

    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False, onerror=raise_walk_error
    ):
        directory_path = Path(directory)
        if directory_path == root:
            directory_names[:] = [
                name
                for name in directory_names
                if name not in INTERNAL_NAMESPACES
                and not (directory_path / name).is_symlink()
            ]
        else:
            directory_names[:] = [
                name for name in directory_names if not (directory_path / name).is_symlink()
            ]

        for file_name in file_names:
            path = directory_path / file_name
            try:
                metadata = os.lstat(path)
            except OSError as exc:
                raise AuditError(f"cannot inspect local artifact {path}: {exc}") from exc
            if not stat_module.S_ISREG(metadata.st_mode):
                continue
            yield path.relative_to(root).as_posix()


def audit_references(
    connection: Any,
    artifact_root: Path,
    *,
    check_orphans: bool = False,
) -> AuditResult:
    """Run the audit using an existing connection and an existing root."""

    root = validate_artifact_root(artifact_root)
    references = read_references(connection)
    valid_keys = set()
    issues: List[ReferenceIssue] = []

    for reference in references:
        try:
            storage_key = normalize_storage_key(reference.storage_key)
        except InvalidStorageKeyError:
            issues.append(
                ReferenceIssue(
                    table=reference.table,
                    owner_column=reference.owner_column,
                    owner_id=reference.owner_id,
                    storage_key=reference.storage_key,
                    reason="invalid_storage_key",
                )
            )
            continue

        valid_keys.add(storage_key)
        reason = _reference_status(root, storage_key)
        if reason:
            issues.append(
                ReferenceIssue(
                    table=reference.table,
                    owner_column=reference.owner_column,
                    owner_id=reference.owner_id,
                    storage_key=storage_key,
                    reason=reason,
                )
            )

    orphan_keys: List[str] = []
    if check_orphans:
        orphan_keys = sorted(set(iter_local_artifacts(root)) - valid_keys)

    return AuditResult(
        artifact_root=str(root),
        references_scanned=len(references),
        unique_storage_keys=len(valid_keys),
        reference_issues=sorted(
            issues,
            key=lambda issue: (issue.table, issue.owner_id, issue.storage_key, issue.reason),
        ),
        orphan_scan_enabled=check_orphans,
        orphan_keys=orphan_keys,
    )


def database_settings_from_env() -> Dict[str, Any]:
    """Build psycopg2 settings from the application's PostgreSQL environment."""

    try:
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        connect_timeout = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "10"))
    except ValueError as exc:
        raise AuditError("POSTGRES_PORT and POSTGRES_CONNECT_TIMEOUT must be integers") from exc
    if port <= 0 or connect_timeout <= 0:
        raise AuditError("POSTGRES_PORT and POSTGRES_CONNECT_TIMEOUT must be positive")

    settings: Dict[str, Any] = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": port,
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "password"),
        "dbname": os.getenv("POSTGRES_DB", "knowledge_graph"),
        "connect_timeout": connect_timeout,
        "application_name": "icp-rag-artifact-audit",
    }
    sslmode = os.getenv("POSTGRES_SSLMODE", "").strip()
    if sslmode:
        settings["sslmode"] = sslmode
    return settings


def connect_database(settings: Dict[str, Any]) -> Any:
    try:
        import psycopg2
    except ImportError as exc:
        raise AuditError("psycopg2 is required; install the project requirements") from exc
    try:
        return psycopg2.connect(**settings)
    except Exception as exc:
        raise AuditError(f"cannot connect to PostgreSQL: {exc}") from exc


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def print_text_result(result: AuditResult) -> None:
    for issue in result.reference_issues:
        print(
            "[REFERENCE-ISSUE] "
            f"{issue.table}.{issue.owner_column}={_quoted(issue.owner_id)} "
            f"key={_quoted(issue.storage_key)} reason={issue.reason}"
        )
    for storage_key in result.orphan_keys:
        print(f"[ORPHAN] key={_quoted(storage_key)}")
    print(
        "[SUMMARY] "
        f"consistent={str(result.is_consistent).lower()} "
        f"references={result.references_scanned} "
        f"unique_keys={result.unique_storage_keys} "
        f"reference_issues={len(result.reference_issues)} "
        f"orphan_scan={str(result.orphan_scan_enabled).lower()} "
        f"orphans={len(result.orphan_keys)}"
    )


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    default_root = Path(
        os.getenv("ARTIFACT_STORAGE_DIR", str(repo_root / "volumes" / "artifacts"))
    )
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of PostgreSQL artifact references and LocalFS ArtifactStore."
        )
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=default_root,
        help="existing LocalFS ArtifactStore root (default: ARTIFACT_STORAGE_DIR or %(default)s)",
    )
    parser.add_argument(
        "--check-orphans",
        "--report-orphans",
        dest="check_orphans",
        action="store_true",
        help="also report regular files not referenced by the three PostgreSQL columns",
    )
    parser.add_argument("--json", action="store_true", help="emit one JSON result")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    args = build_parser().parse_args(argv)
    connection = None
    try:
        # Validate the mount before opening PostgreSQL.  This operation never
        # creates the missing root, which catches broken container mounts.
        root = validate_artifact_root(args.artifact_root)
        connection = connect_database(database_settings_from_env())
        result = audit_references(
            connection,
            root,
            check_orphans=args.check_orphans,
        )
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
        else:
            print_text_result(result)
        return EXIT_OK if result.is_consistent else EXIT_INCONSISTENT
    except AuditError as exc:
        if args.json:
            print(
                json.dumps(
                    {"consistent": False, "audit_error": str(exc)},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        else:
            print(f"[AUDIT-ERROR] {exc}", file=sys.stderr)
        return EXIT_AUDIT_ERROR
    finally:
        if connection is not None:
            try:
                connection.rollback()
            except Exception:
                pass
            try:
                connection.close()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[AUDIT-ERROR] interrupted", file=sys.stderr)
        raise SystemExit(130)
