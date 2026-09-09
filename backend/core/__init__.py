"""Core package with dependency-safe lazy compatibility exports.

Importing a focused module such as :mod:`backend.core.embedding` must not open a
PostgreSQL connection or load every parser.  The project historically re-exported
the names below from this package, so they remain available through PEP 562 lazy
attribute loading without introducing import-time infrastructure side effects.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "config": ("backend.config.config", "config"),
    "TaskManager": ("backend.core.task_manager", "TaskManager"),
    "task_manager": ("backend.core.task_manager", "task_manager"),
    "llm_service": ("backend.core.llm_service", "llm_service"),
    "model_config_service": ("backend.core.model_config", "model_config_service"),
    "embedding_service": ("backend.core.embedding", "embedding_service"),
    "rerank_service": ("backend.core.rerank", "rerank_service"),
    "knowledge_retriever": ("backend.core.retrieval", "knowledge_retriever"),
    "BusinessError": ("backend.core.exceptions", "BusinessError"),
    "ok": ("backend.core.response", "ok"),
    "fail": ("backend.core.response", "fail"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
