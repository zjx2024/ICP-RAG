"""Bounded LLM transport for current-read engineering-model selection."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Mapping, Sequence

from backend.agent.current_model_selection import (
    SELECTION_PROMPT, model_candidates, validate_selection_with_reason,
)
from backend.agent.model_role_service import agent_model_role_service
from backend.agent.model_runtime_policy import bounded_model_transport, remaining_seconds
from backend.core.llm_service import llm_service

logger = logging.getLogger(__name__)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _selection_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Reserve the bounded output for the selection contract, not hidden thought."""

    prepared = {**model, "temperature": 0, "stream": False}
    if str(prepared.get("provider") or "").casefold() == "ollama":
        prepared.update(reasoning=False, format="json")
    return bounded_model_transport(prepared)


def _selection_failed(doc_type: str, reason: str) -> None:
    logger.warning("Current %s model selection unavailable: %s", doc_type, reason)
    try:
        from backend.agent.tracing import record_degradation

        record_degradation("model_selection", detail=f"{doc_type}:{reason}")
    except Exception as exc:
        logger.debug("Unable to record model-selection degradation: %s", type(exc).__name__)


async def select_current_models(
    state: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, doc_type: str,
) -> dict[str, Any]:
    candidates = model_candidates(rows, doc_type)
    query = _text(state.get("query"))
    if not candidates or not query:
        return {}
    model_id = _text(state.get("chat_model_id"))
    payload = {"current_question": query, "retrieval_context": _text(state.get("rewritten_query")), "candidates": candidates}
    try:
        if model_id:
            from backend.storage.postgres_client import postgres_client
            model = postgres_client.get_model(model_id)
        elif state.get("agent_id"):
            model = agent_model_role_service.get_model_by_role(_text(state.get("agent_id")), "plan")
        else:
            return {}
        model = agent_model_role_service.apply_agent_llm_params(dict(state), model)
        if not model:
            return {}
        model = _selection_model(model)
        deadline = time.monotonic() + remaining_seconds()
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            raw = await asyncio.wait_for(asyncio.to_thread(
                llm_service.call_model,
                [{"role": "system", "content": SELECTION_PROMPT},
                 {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                bounded_model_transport(model, remaining), template_name="current_model_selection",
            ), timeout=remaining)
            # An unavailable transport is not a semantic decision to retry.
            # Only a nonempty structured proposal gets one bounded correction.
            proposal = json.loads(raw)
            result, rejection = validate_selection_with_reason(proposal, candidates, query=query)
            if result:
                return result
            _selection_failed(doc_type, f"invalid_contract:{rejection}")
            if attempt or rejection == "invalid_offered_identities":
                return {}
            payload = {**payload, "contract_correction": {
                "error": rejection,
                "instruction": "重新理解同一个原始问题，只从同一候选中选择。无需输出 question_basis，当前问题身份由程序绑定；模型身份和标题必须逐字来自候选，不得修造ID。",
            }}
        return {}
    except Exception as exc:
        _selection_failed(doc_type, type(exc).__name__)
        return {}
