"""Provider-neutral output/transport bounds for the existing role policies."""
from __future__ import annotations

import math
from typing import Any, Mapping


def bounded_model_transport(model: Mapping[str, Any] | None, seconds: float) -> dict[str, Any] | None:
    """Align socket timeout and both provider output keys with node budgets.

    wait_for cannot kill a worker thread. A transport read timeout prevents an
    abandoned synchronous request retaining the configured multi-minute socket
    timeout. It does not promise remote inference cancellation.
    """
    if not model:
        return model
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("model transport budget must be finite and positive")
    prepared = dict(model)
    try:
        configured = float(prepared.get("timeout"))
    except (TypeError, ValueError):
        configured = seconds
    prepared["timeout"] = min(configured, seconds) if math.isfinite(configured) and configured > 0 else seconds
    # Role-specific preparation already owns num_predict. The qwen transport
    # reads max_tokens instead, so leaving its old value bypasses that budget.
    if "num_predict" in prepared:
        prepared["max_tokens"] = prepared["num_predict"]
    return prepared
