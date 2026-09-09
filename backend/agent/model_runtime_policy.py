"""Provider-neutral output/transport bounds for the existing role policies."""
from __future__ import annotations

import math
import json
import os
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Mapping


def _configured_seconds() -> float:
    try:
        value = float(os.getenv("ICP_AGENT_TOTAL_TIMEOUT_SECONDS", "240"))
        return value if math.isfinite(value) and value > 0 else 240.0
    except ValueError:
        return 240.0


TURN_TIMEOUT_SECONDS = _configured_seconds()
_deadline: ContextVar[float | None] = ContextVar("icp_turn_deadline", default=None)
_capacities: ContextVar[dict | None] = ContextVar("icp_context_capacities", default=None)


def remaining_seconds(limit: float | None = None) -> float:
    deadline = _deadline.get()
    seconds = deadline - time.monotonic() if deadline is not None else TURN_TIMEOUT_SECONDS
    if limit is not None:
        seconds = min(seconds, limit)
    if seconds <= 0:
        raise TimeoutError("ICP 整轮时间预算已耗尽")
    return seconds


@contextmanager
def turn_budget(seconds: float | None = None):
    """One deadline, inherited by async tasks and to_thread model calls."""
    if _deadline.get() is not None:
        yield
        return
    token = _deadline.set(time.monotonic() + (seconds or TURN_TIMEOUT_SECONDS))
    capacities = _capacities.set({})
    try:
        yield
    finally:
        _capacities.reset(capacities)
        _deadline.reset(token)


def bounded_model_transport(model: Mapping[str, Any] | None, seconds: float | None = None) -> dict[str, Any] | None:
    """Align socket timeout and both provider output keys with node budgets.

    wait_for cannot kill a worker thread. A transport read timeout prevents an
    abandoned synchronous request retaining the configured multi-minute socket
    timeout. It does not promise remote inference cancellation.
    """
    if not model:
        return model
    seconds = remaining_seconds(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("model transport budget must be finite and positive")
    prepared = dict(model)
    try:
        configured = float(prepared.get("timeout"))
    except (TypeError, ValueError):
        configured = seconds
    prepared["timeout"] = min(configured, seconds) if math.isfinite(configured) and configured > 0 else seconds
    # Agent parameters have already overridden the registered model defaults.
    # Both providers now consume that one value; node constants cannot replace it.
    output = prepared.get("max_tokens", prepared.get("num_predict"))
    if output is not None:
        if isinstance(output, bool) or int(output) <= 0:
            raise ValueError("模型输出预算必须为正整数")
        prepared["max_tokens"] = prepared["num_predict"] = int(output)
    prepared["_icp_runtime"] = True
    return prepared


class ModelBudgetError(RuntimeError):
    """A resource limit is an execution result, never evidence of absence."""


def prepare_model_request(model: Mapping[str, Any] | None, messages: list) -> dict | None:
    """Check a complete prompt, without silently cropping records or history.

    The provider owns tokenization and rejects overflow without truncation.
    Serialized bytes describe transport size only, never a token threshold.
    """
    if not model or not model.get("_icp_runtime"):
        return dict(model) if model else model
    prepared = bounded_model_transport(model)
    if prepared.get("provider") == "ollama":
        import requests
        cache = _capacities.get()
        version_key = ("no_truncation", prepared.get("base_url"))
        if cache is None or version_key not in cache:
            try:
                response = requests.get(str(prepared.get("base_url")).rstrip("/") + "/api/version",
                                        timeout=remaining_seconds())
                response.raise_for_status()
                version = str(response.json().get("version") or "")
                parts = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
                if not parts or tuple(map(int, parts.groups())) < (0, 33, 3):
                    raise ModelBudgetError("当前 Ollama 版本未通过无截断协议兼容验证，需要 0.33.3 或更新版本")
                if cache is not None:
                    cache[version_key] = version
            except (requests.RequestException, ValueError, TypeError) as exc:
                raise ModelBudgetError("无法核验模型服务的无截断协议，本轮未静默删除输入") from exc
    capacity = prepared.get("num_ctx") or prepared.get("context_length") or prepared.get("context_window")
    source = "model_configuration" if capacity else "unknown"
    if not capacity and prepared.get("provider") == "ollama":
        import requests

        key = (prepared.get("base_url"), prepared.get("model_name"))
        cache = _capacities.get()
        if cache is not None and key in cache:
            capacity = cache[key]
        else:
            try:
                response = requests.get(str(key[0]).rstrip("/") + "/api/ps", timeout=remaining_seconds())
                response.raise_for_status()
                capacity = next((row.get("context_length") for row in response.json().get("models", [])
                                 if (row.get("name") or row.get("model")) == key[1]), None)
                if not capacity:
                    response = requests.post(str(key[0]).rstrip("/") + "/api/show",
                                             json={"model": key[1]}, timeout=remaining_seconds())
                    response.raise_for_status()
                    info = response.json().get("model_info") or {}
                    capacity = next((v for k, v in info.items() if k.endswith(".context_length")), None)
                if cache is not None and capacity:
                    cache[key] = capacity
            except (requests.RequestException, ValueError, TypeError) as exc:
                raise ModelBudgetError("无法读取模型上下文容量，本轮未静默截断输入") from exc
        source = "ollama_capacity"
        if not capacity:
            raise ModelBudgetError("模型未提供上下文容量，无法验证本轮输入预算")
    output = prepared.get("max_tokens")
    if output is None:
        from backend.config.config import config
        output = config.MODEL_MAX_TOKENS
        prepared["max_tokens"] = prepared["num_predict"] = int(output)
    input_bytes = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
    prepared["_runtime_budget"] = {
        "context_tokens": int(capacity) if capacity else None,
        "capacity_source": source,
        "output_tokens": int(output),
        "input_bytes": input_bytes,
        "input_tokens": None,
        "context_enforcement": "provider_tokenizer_no_truncation" if prepared.get("provider") == "ollama" else "provider",
        "remaining_seconds": round(remaining_seconds(), 3),
    }
    if capacity:
        capacity = int(capacity)
        if int(output) >= capacity:
            raise ModelBudgetError("Agent 输出预算占满模型上下文，没有为输入留出空间；请调整输出设置")
        if prepared.get("provider") == "ollama":
            prepared["num_ctx"] = capacity
    return prepared
