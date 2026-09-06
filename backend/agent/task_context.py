"""Read-only current task shared by retrieval, relevance and answer stages."""
from __future__ import annotations

from typing import Any, Mapping


def current_task(state: Mapping[str, Any]) -> dict[str, Any]:
    """Planning owns resolution; rewrites, fields and old answers cannot replace it."""
    query = str(state.get("query") or "").strip()
    dialogue = state.get("dialogue_context")
    dialogue = dialogue if isinstance(dialogue, Mapping) else {}
    status = str(dialogue.get("status") or "standalone")
    resolved = str(dialogue.get("contextualized_query") or "").strip()
    usable = status in {"standalone", "resolved"}
    return {
        "original_question": query,
        "resolved_question": resolved if usable and resolved else query,
        "information_goals": list(dialogue.get("requested_facets") or []) if usable else [],
        "subjects": list(dialogue.get("subjects") or []) if usable else [],
        "context_status": status,
    }


def task_query(state: Mapping[str, Any]) -> str:
    """A relevance query, not an identity or calculation authorization."""
    task = current_task(state)
    return "\n".join(dict.fromkeys(
        str(value).strip() for value in (
            task["resolved_question"], task["original_question"],
            *task["subjects"], *task["information_goals"],
        ) if str(value).strip()
    ))
