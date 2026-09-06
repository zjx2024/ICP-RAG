"""Model-selected prior USER input, never authoritative evidence or permission.

Inputs without a database hit must remain referable. Rebuild their provenance
from bounded message history each turn; do not persist fabricated citations or
carry old calculator permissions alongside them.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from backend.agent.answer_context_binding import conversation_review_context
from backend.agent.query_understanding import extract_business_codes
from backend.agent.hmc_identity import extract_hmc_components


def offered_user_inputs(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages = conversation_review_context(state)["messages"]
    if messages and messages[-1]["role"] == "user" and messages[-1]["content"] == str(state.get("query") or ""):
        messages = messages[:-1]
    return [{**message, "authority": "unverified_user_input"} for message in messages if message["role"] == "user"]


def validate_user_context_bindings(value: Any, offered: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) > 8 or not isinstance(offered, list):
        return []
    messages = {row["message_id"]: row for row in offered if isinstance(row, Mapping) and row.get("role") == "user"}
    bindings: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            return []
        message_id, quote = item.get("message_id"), item.get("quote")
        if not isinstance(message_id, str) or not isinstance(quote, str) or not quote.strip() or len(quote) > 1200:
            return []
        if message_id not in messages or quote not in messages[message_id]["content"]:
            return []
        binding = {"message_id": message_id, "quote": quote, "authority": "unverified_user_input"}
        if binding not in bindings:
            bindings.append(binding)
    return bindings


def selected_user_input_quotes(state: Mapping[str, Any]) -> list[str]:
    frame = state.get("query_frame")
    context = state.get("dialogue_context")
    if not isinstance(frame, Mapping) or not isinstance(context, Mapping):
        return []
    if context.get("context_used") is not True or context.get("status") != "resolved":
        return []
    if frame.get("user_input_query") != state.get("query"):
        return []
    return [item["quote"] for item in validate_user_context_bindings(frame.get("user_context_bindings"), offered_user_inputs(state))]


def selected_user_hmc_components(state: Mapping[str, Any]) -> dict[str, str]:
    query = str(state.get("query") or "")
    # A current identity, even a partial correction, supersedes past input.
    # These are code-field delimiters, not natural-language intent keywords.
    if extract_business_codes(query) or re.search(r"\b(?:LCN|HAMC|HSC)\b", query, re.I):
        return {}
    candidates = [extract_hmc_components(quote) for quote in selected_user_input_quotes(state)]
    candidates = [candidate for candidate in candidates if candidate]
    if not candidates or any(candidate != candidates[0] for candidate in candidates[1:]):
        return {}
    return candidates[0]
