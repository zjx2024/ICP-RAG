"""Request-local provenance for statements about the conversation itself.

History can prove what was said, never that an old engineering conclusion is
true. The model classifies that distinction; code checks the closed protocol
and verbatim provenance. No user-wording allow-list is used.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


def conversation_review_context(
    state: Mapping[str, Any], *, max_messages: int = 16
) -> dict[str, Any]:
    messages = state.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    offered: list[dict[str, Any]] = []
    remaining = 18000
    count = min(16, max(0, max_messages))
    for index in range(len(messages) - 1, max(-1, len(messages) - count - 1), -1):
        message = messages[index]
        if isinstance(message, Mapping):
            role, content = message.get("role"), message.get("content")
        else:
            role, content = (
                getattr(message, "type", ""),
                getattr(message, "content", ""),
            )
        role = {"ai": "assistant", "human": "user"}.get(role, role)
        if role not in {"assistant", "user"} or not isinstance(content, str):
            continue
        limit = min(6000, remaining)
        if limit <= 0:
            break
        offered.append(
            {
                "message_id": f"h{index}",
                "role": role,
                "content": content[:limit],
                "truncated": len(content) > limit,
            }
        )
        if role == "assistant" and isinstance(message, Mapping):
            # Progress describes prior delivery, not current source truth.
            # The original answer remains the only basis for history quotes;
            # a delivery label never suppresses a new question in that area.
            progress = []
            for entry in (message.get("answer_delivery_manifest") or [])[:16]:
                if not isinstance(entry, Mapping) or entry.get("kind") != "goal_delivery":
                    continue
                if entry.get("status") not in {"answered", "partial", "unanswered"}:
                    continue
                progress.append({"goal": str(entry.get("goal") or "")[:600],
                                 "status": entry["status"],
                                 "evidence_ids": [str(value)[:240] for value in (entry.get("evidence_ids") or [])[:16]]})
            if progress:
                offered[-1]["prior_delivery"] = progress
        remaining -= len(content[:limit])
    return {"query": str(state.get("query") or ""), "messages": list(reversed(offered))}


def clarification_review_observations(
    contract: Mapping[str, Any],
    *,
    categories: tuple[str, ...] = (
        "missing_input_requirements",
        "unresolved_identity_slots",
        "clarification_reasons",
        "clarification_fallback",
    ),
) -> list[dict[str, Any]]:
    """Offer only validated current gap records, not raw tools or old answers."""
    observations: list[dict[str, Any]] = []
    for category in categories:
        values = contract.get(category) or []
        if category == "clarification_fallback":
            values = (
                [values]
                if isinstance(values, Mapping) and values.get("required") is True
                else []
            )
        if not isinstance(values, list):
            continue
        for record in values:
            if not isinstance(record, Mapping) or len(observations) >= 48:
                continue
            fields: dict[str, str] = {}

            def flatten(value: Any, path: str, depth: int = 0) -> None:
                if depth > 6 or len(fields) >= 100:
                    return
                if isinstance(value, Mapping):
                    for key, child in value.items():
                        flatten(
                            child,
                            f"{path}/{str(key).replace('~', '~0').replace('/', '~1')}",
                            depth + 1,
                        )
                elif isinstance(value, list):
                    for index, child in enumerate(value):
                        flatten(child, f"{path}/{index}", depth + 1)
                elif isinstance(value, (str, int, float, bool)):
                    text = value if isinstance(value, str) else json.dumps(value)
                    if text and len(text) <= 1000:
                        fields[path] = text

            flatten(record, "")
            observations.append(
                {
                    "observation_id": f"g{len(observations) + 1}",
                    "category": category,
                    "fields": fields,
                }
            )
    return observations
