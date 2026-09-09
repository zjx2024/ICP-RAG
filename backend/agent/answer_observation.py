"""Opt-in, process-local answer observations for controlled offline experiments.

No API parameter or public trace exposes this collector. The diagnostic runner
owns full observations; failed review exchanges also get a private local file.
Observers never decide whether text is correct and never modify the answer.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any, Iterator, Mapping


_STATE_KEYS = (
    "query", "rewritten_query", "messages", "system_prompt", "agent_id", "chat_model_id", "temperature", "top_p",
    "max_tokens", "intent", "query_frame", "dialogue_context", "candidate_faults",
    "conversation_anchors", "selected_conversation_anchors", "evidence_registry",
    "context_evidence_ids", "safety_decision", "tool_results", "calculation_results",
    "missing_input_requirements", "collection_resolution",
    "calculation_requests", "calculation_authorizations", "allowed_tools", "requested_source_tools", "hmc_mfl_binding", "answer_mode",
    "structured_entity_binding", "structured_record_binding",
    "rbd_query_binding", "fta_query_binding",
    "clarification_question", "diagnostic_context", "domain_results", "kg_summary",
    "query_targets", "rerank_model_id", "sql_results", "tool_calls", "tool_queries",
    "adaptive_decision", "history_parent_binding",
    "topology_dependency", "vector_results",
)
_MODEL_KEYS = (
    "id", "model_id", "model_name", "provider", "temperature", "top_p", "max_tokens",
    "num_predict", "reasoning", "num_ctx", "timeout", "seed", "stop",
)
_SECRET_KEYS = frozenset({
    "password", "passwd", "api_key", "apikey", "authorization", "cookie",
    "access_token", "refresh_token", "secret", "client_secret", "headers",
    "base_url", "api_url", "endpoint",
})


def detached(value: Any) -> Any:
    """Copy JSON data without retaining mutable state or serializing objects."""
    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>" if str(key).casefold() in _SECRET_KEYS else detached(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [detached(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return f"<unsupported:{type(value).__name__}>"


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def public_model(model: Mapping[str, Any] | None) -> dict[str, Any]:
    return {key: detached(model[key]) for key in _MODEL_KEYS if model and key in model}


@dataclass
class AnswerObservation:
    started: float = field(default_factory=time.monotonic)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    reviews: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return detached({"inputs": self.inputs, "stages": self.stages,
                         "model_calls": self.model_calls, "reviews": self.reviews, "errors": self.errors})


_observer: ContextVar[AnswerObservation | None] = ContextVar("answer_observation", default=None)


def answer_observation_enabled() -> bool:
    return _observer.get() is not None


def observe_review_exchange(*, raw, prepared, ledger, safety, calculation_requests,
                            salvage=False, result=None, error=None, transport=None) -> None:
    """Preserve exact validator inputs, not hidden model reasoning.

    Public traces retain only codes. These private snapshots permit zero-LLM
    replay of rejected bindings, including source values and full feedback.
    Recording is observational: an I/O failure cannot change the verdict.
    """
    from pathlib import Path
    import logging
    import os
    import uuid
    from backend.agent.tracing import peek
    from backend.config.config import config

    observation, trace = _observer.get(), peek()
    if observation is None and (error is None or trace is None):
        return
    try:
        files = (Path(__file__), Path(__file__).with_name("answer_review.py"),
                 Path(__file__).with_name("citation_binding.py"))
        snapshot = detached({
            "schema_version": "icp-review-exchange/v1",
            "trace_id": trace.trace_id if trace else None,
            "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "raw": raw, "prepared": prepared, "ledger": ledger, "safety": safety,
            "calculation_requests": calculation_requests, "salvage": salvage,
            "result": result, "error": {"code": error.code, "feedback": error.feedback} if error else None,
        })
        if transport is not None:
            snapshot["transport"] = detached(transport)
        snapshot["sha256"] = fingerprint(snapshot)
        if observation is not None:
            observation.reviews.append(snapshot)
        if error is not None and trace is not None:
            directory = Path(getattr(config, "TRACE_DIR", None) or "data/traces").parent / "private_answer_reviews"
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = directory / f"{trace.trace_id}-{uuid.uuid4().hex}.json"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, ensure_ascii=False, allow_nan=False)
    except Exception as exc:
        if observation is not None:
            observation.errors.append(f"review_exchange:{type(exc).__name__}")
        logging.getLogger(__name__).warning("Private review capture failed: %s", type(exc).__name__)


@contextmanager
def observe_answers() -> Iterator[AnswerObservation]:
    observation = AnswerObservation()
    token = _observer.set(observation)
    try:
        yield observation
    finally:
        _observer.reset(token)


def observe_answer_input(
    state: Mapping[str, Any], *, stage: str = "entry", prompt: str = "",
    context: str = "", citations: Any = None, model: Mapping[str, Any] | None = None,
) -> None:
    observation = _observer.get()
    if observation is None:
        return
    try:
        observation.inputs.append({
            "stage": stage,
            "state": detached({key: state[key] for key in _STATE_KEYS if key in state}),
            "prompt": prompt, "context": context,
            "citations": detached(citations or []), "model": public_model(model),
        })
    except Exception as exc:
        # An observational failure cannot alter published behavior. The runner
        # treats an incomplete observation as an experiment error, not success.
        observation.errors.append(f"input:{stage}:{type(exc).__name__}")


def observe_answer_stage(stage: str, answer: str, *, citations: Any = None) -> None:
    observation = _observer.get()
    if observation is None:
        return
    try:
        previous = observation.stages[-1]["answer"] if observation.stages else None
        observation.stages.append({
            "stage": stage, "answer": answer, "sha256": fingerprint(answer),
            "chars": len(answer), "changed": previous is not None and answer != previous,
            "elapsed_ms": round((time.monotonic() - observation.started) * 1000, 3),
            "citations": detached(citations) if citations is not None else None,
        })
    except Exception as exc:
        observation.errors.append(f"stage:{stage}:{type(exc).__name__}")


def observe_model_call(fields: Mapping[str, Any]) -> None:
    observation = _observer.get()
    if observation is not None:
        try:
            observation.model_calls.append(detached({
                key: fields[key] for key in ("template_name", "duration_ms", "ok", "model_name", "response_chars", "token_usage", "finish_reason", "resource_budget")
                if key in fields
            }))
        except Exception as exc:
            observation.errors.append(f"model_call:{type(exc).__name__}")
