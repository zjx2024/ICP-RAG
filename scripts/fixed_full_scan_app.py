"""A/B-only FastAPI entry point that reproduces the former fixed full scan.

Run with uvicorn on a second local port.  The application, evidence gate and
answer implementation remain identical to the normal service; only the query
tool selection is replaced with the former unconditional seven-source scan.
This module is not imported by production startup scripts.
"""
from __future__ import annotations

from typing import Any, Dict

from backend.agent.nodes import plan as plan_module


_adaptive_resolve_tool_policy = plan_module.resolve_tool_policy
_QUERY_ORDER = (
    "function_locator",
    "hmc_query",
    "fmeca_query",
    "fta_query",
    "rbd_query",
    "mfl_query",
    "fault_history_query",
    "hybrid_search",
)
_DOC_TYPES = (
    "function_locator",
    "hmc",
    "fmeca",
    "fta",
    "rbd",
    "mfl",
    "fault_history",
)


def _fixed_full_scan_policy(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    decision = dict(_adaptive_resolve_tool_policy(*args, **kwargs))
    existing_tools = list(decision.get("allowed_tools") or [])
    calculators = [
        tool for tool in ("fta_analyzer", "rbd_calculator")
        if tool in existing_tools
    ]
    selected = [*_QUERY_ORDER, *calculators, "evidence_synthesis"]
    existing_plan = {
        str(item.get("tool") or ""): dict(item)
        for item in (decision.get("tools_plan") or [])
        if isinstance(item, dict) and item.get("tool")
    }
    decision.update(
        {
            "authorized_tools": selected,
            "allowed_tools": selected,
            "authorized_doc_types": list(_DOC_TYPES),
            "hybrid_doc_types": list(_DOC_TYPES),
            "tools_plan": [
                existing_plan.get(tool)
                or {
                    "tool": tool,
                    "reason": "A/B 基准：复现旧版固定七源检索",
                    "targets": [
                        str(item.get("intent") or "")
                        for item in (decision.get("query_targets") or [])
                        if isinstance(item, dict) and item.get("intent")
                    ],
                }
                for tool in selected
            ],
            "notes": [
                *(decision.get("notes") or []),
                "A/B 基准已固定执行七类查询；不代表生产推荐配置",
            ],
        }
    )
    return decision


plan_module.resolve_tool_policy = _fixed_full_scan_policy

# Import only after patching the planner module.  ``backend.main`` then builds
# the same graph and API with the A/B policy above.
from backend.main import app  # noqa: E402,F401
