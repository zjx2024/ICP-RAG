"""Consistency helper for registering an artifact in PostgreSQL."""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional

from backend.storage.artifact_store import artifact_store

logger = logging.getLogger(__name__)


class ArtifactRegistrationUncertainError(RuntimeError):
    """The database outcome could not be verified; the artifact was retained."""


def confirm_registration_or_compensate(
    *,
    doc_id: str,
    storage_key: str,
    lookup: Callable[[str], Optional[Mapping[str, Any]]],
    rollback: Callable[[], None],
) -> bool:
    """Resolve an ambiguous PostgreSQL commit before compensating a file.

    Returns ``True`` when the row is present (the commit succeeded despite the
    raised exception) and ``False`` after a confirmed-absent row is compensated.
    If the read-back itself fails, retain the artifact and raise: an orphan is
    repairable, while deleting a committed row's original is data loss.
    """

    try:
        rollback()
    except Exception:
        logger.warning("PostgreSQL rollback failed for %s", doc_id, exc_info=True)

    try:
        persisted = lookup(doc_id)
    except Exception as exc:
        raise ArtifactRegistrationUncertainError(
            f"文档 {doc_id} 的数据库提交结果无法确认；原件已保留，请运行一致性审计"
        ) from exc

    if persisted:
        logger.warning(
            "PostgreSQL insert for %s raised after the row became visible; retaining artifact",
            doc_id,
        )
        return True

    artifact_store.delete_file(storage_key)
    return False
