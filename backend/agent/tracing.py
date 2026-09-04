"""全链路 tracing（P0-B / B1）。

以「会话内一次提问」为 trace 单元：chat_router 进入时 ``begin_turn``
生成 ``trace_id`` 并写入 ContextVar，节点与 LLM 调用通过上下文自动关联；
回答流结束后 ``end_turn`` 将事件以 JSON-lines 落盘（目录可配置），同时
更新进程级聚合计数。查询接口按 ``trace_id`` 回放或取聚合统计。

降级计数只统计三类系统性降级，对应路线图可量化验收项：
- ``planner_json``：规划模型 JSON 解析失败退回宽召回
- ``adaptive_fallback``：adaptive_tools 路由输出不可用退回安全候选集
- ``retrieval_projection``：检索管线出错（含投影失败）触发降级

LLM 调用统一挂在 ``llm_service.call_model``（P0-7 收敛后的公开入口），
工具调用事件由各工具节点/检索映射处主动上报。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from backend.config.config import config

logger = logging.getLogger(__name__)

DEGRADATION_KINDS = (
    "planner_json",
    "adaptive_fallback",
    "retrieval_projection",
)

_current_trace: ContextVar[Optional["TraceRecorder"]] = ContextVar(
    "icp_trace_recorder", default=None
)

_AGG_LOCK = threading.Lock()
_AGG: Dict[str, int] = {
    "turns": 0,
    "llm_calls": 0,
    "llm_failures": 0,
    "tool_events": 0,
    "deg_planner_json": 0,
    "deg_adaptive_fallback": 0,
    "deg_retrieval_projection": 0,
}


def _trace_dir() -> str:
    directory = getattr(config, "TRACE_DIR", None) or "data/traces"
    os.makedirs(directory, exist_ok=True)
    return directory


class TraceRecorder:
    """一次提问的 trace 记录器；事件按发生顺序保存，落盘为 JSON-lines。"""

    def __init__(self, session_id: str, query: str) -> None:
        self.trace_id = uuid.uuid4().hex
        self.session_id = str(session_id or "")
        self.query = str(query or "")[:500]
        self.started_at = time.time()
        self.events: List[Dict[str, Any]] = []
        self._finished = False

    def _record(self, event_type: str, **fields: Any) -> None:
        if self._finished:
            return
        self.events.append(
            {
                "ts_ms": int((time.time() - self.started_at) * 1000),
                "type": event_type,
                **fields,
            }
        )

    def record_llm(
        self,
        *,
        template_name: str,
        duration_ms: float,
        ok: bool,
        model_name: str = "",
        response_chars: int = 0,
    ) -> None:
        self._record(
            "llm",
            template_name=template_name or "<default>",
            duration_ms=round(duration_ms, 1),
            ok=bool(ok),
            model_name=model_name or "",
            response_chars=response_chars,
        )

    def record_tool(
        self,
        *,
        tool_name: str,
        authorized: Optional[str] = None,
        evidence_count: Optional[int] = None,
        duration_ms: Optional[float] = None,
        ok: bool = True,
    ) -> None:
        self._record(
            "tool",
            tool_name=tool_name,
            authorized=authorized,
            evidence_count=evidence_count,
            duration_ms=round(duration_ms, 1) if duration_ms is not None else None,
            ok=bool(ok),
        )

    def record_degradation(self, kind: str, detail: str = "") -> None:
        if kind not in DEGRADATION_KINDS:
            kind = "other"
        self._record(
            "degradation",
            kind=kind,
            detail=str(detail or "")[:300],
        )

    def summary(self) -> Dict[str, Any]:
        llm_events = [event for event in self.events if event["type"] == "llm"]
        tool_events = [event for event in self.events if event["type"] == "tool"]
        degradation_events = [
            event for event in self.events if event["type"] == "degradation"
        ]
        deg_counts: Dict[str, int] = {}
        for kind in DEGRADATION_KINDS:
            deg_counts[kind] = sum(
                1 for event in degradation_events if event["kind"] == kind
            )
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "query": self.query,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started_at)),
            "duration_ms": round((time.time() - self.started_at) * 1000, 1),
            "llm_calls": len(llm_events),
            "llm_failures": sum(1 for event in llm_events if not event["ok"]),
            "tool_events": len(tool_events),
            "degradations": deg_counts,
        }

    def write(self) -> Dict[str, Any]:
        """落盘 JSON-lines 并返回 summary；重复调用幂等。"""
        summary = self.summary()
        if self._finished:
            return summary
        self._finished = True
        directory = _trace_dir()
        path = os.path.join(directory, f"{self.trace_id}.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
            for event in self.events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        with _AGG_LOCK:
            # record_* 已在调用时刻累加 llm/tool/degradation 计数，这里只补 turns。
            _AGG["turns"] += 1
        return summary


def begin_turn(session_id: str, query: str) -> str:
    """开始一次提问 trace：生成 trace_id 并绑定到当前上下文。"""
    recorder = TraceRecorder(session_id=session_id, query=query)
    _current_trace.set(recorder)
    return recorder.trace_id


def end_turn() -> Optional[Dict[str, Any]]:
    """结束当前上下文 trace 并落盘；无活跃 trace 时返回 None。"""
    recorder = _current_trace.get()
    if recorder is None:
        return None
    _current_trace.set(None)
    return recorder.write()


def peek() -> Optional[TraceRecorder]:
    """当前上下文活跃的 trace（未开始/已结束为 None）。"""
    return _current_trace.get()


def record_llm(**fields: Any) -> None:
    """LLM 调用留痕：有活跃 trace 进事件；聚合计数始终累加。"""
    with _AGG_LOCK:
        _AGG["llm_calls"] += 1
        if not fields.get("ok", True):
            _AGG["llm_failures"] += 1
    recorder = peek()
    if recorder is not None:
        recorder.record_llm(**fields)


def record_tool(**fields: Any) -> None:
    """工具调用事件留痕（工具名 / 授权来源 / 证据数 / 耗时）。"""
    with _AGG_LOCK:
        _AGG["tool_events"] += 1
    recorder = peek()
    if recorder is not None:
        recorder.record_tool(**fields)


def record_degradation(kind: str, detail: str = "") -> None:
    """三类系统性降级计数（即使无活跃 trace 也计进程级聚合）。

    未知 kind 不落计数（避免类型漂移污染降级率统计），但活跃 trace 下
    仍记录 detail 便于排查。
    """
    valid = kind in DEGRADATION_KINDS
    if valid:
        with _AGG_LOCK:
            _AGG[f"deg_{kind}"] += 1
    recorder = peek()
    if recorder is not None:
        recorder.record_degradation(kind, detail=detail)


def aggregate_stats() -> Dict[str, Any]:
    """进程级聚合统计（可回答“平均几次模型调用 / 降级率”）。"""
    with _AGG_LOCK:
        snapshot = dict(_AGG)
    turns = snapshot["turns"] or 1
    llm_calls = snapshot["llm_calls"]
    degradations = (
        snapshot["deg_planner_json"]
        + snapshot["deg_adaptive_fallback"]
        + snapshot["deg_retrieval_projection"]
    )
    return {
        **snapshot,
        "avg_llm_calls_per_turn": round(llm_calls / turns, 2),
        "llm_failure_rate": round(snapshot["llm_failures"] / max(llm_calls, 1), 4),
        "degradation_rate_per_turn": round(degradations / turns, 2),
    }


def get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    """按 trace_id 回放：首行 summary + 事件列表。"""
    path = os.path.join(_trace_dir(), f"{trace_id}.jsonl")
    if not os.path.exists(path):
        return None
    events: List[Dict[str, Any]] = []
    summary: Optional[Dict[str, Any]] = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("type") is None and "trace_id" in payload:
                summary = payload
            else:
                events.append(payload)
    if summary is None:
        summary = {"trace_id": trace_id}
    summary["events"] = events
    return summary


def list_traces(limit: int = 20) -> List[Dict[str, Any]]:
    """列出最近 trace（按文件修改时间倒序），仅含 summary 首行。"""
    directory = _trace_dir()
    if not os.path.isdir(directory):
        return []
    entries = []
    for name in os.listdir(directory):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(directory, name)
        try:
            with open(path, encoding="utf-8") as handle:
                first = handle.readline().strip()
            if not first:
                continue
            payload = json.loads(first)
            entries.append(
                (
                    payload.get("started_at") or "",
                    os.path.getmtime(path),
                    payload,
                )
            )
        except (OSError, ValueError):
            continue
    entries.sort(key=lambda item: item[1], reverse=True)
    return [entry[2] for entry in entries[: max(1, int(limit))]]