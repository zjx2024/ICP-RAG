#!/usr/bin/env python3
"""Run repeatable paraphrase evaluations against the local ICP agent.

The runner intentionally talks to the same public session/chat APIs as the UI.
It does not import backend modules or read the database directly, so every
captured plan, tool call, graph traversal and evidence item reflects the real
HTTP/SSE contract.

Minimal input (UTF-8 JSON)::

    {
      "cases": [
        {
          "id": "restart-cause",
          "question": "系统运行时反复重启的原因是什么？",
          "paraphrases": [
            "设备正常工作时总是自动复位，可能是什么原因？",
            {"id": "colloquial", "question": "机器跑着跑着一直重启，咋回事？"}
          ],
          "assertions": {
            "any_tools": ["fta_query", "graph_traversal"],
            "any_doc_types": ["fta"],
            "required_used_doc_types": ["fta"],
            "any_used_record_ids": ["FTA-SYN-003", "FM-SYN-040"],
            "required_answer_facets": ["cause", "impact", "resolution"],
            "min_evidence": 1,
            "min_citations": 1,
            "answer_not_contains": ["唯一根因已经确定"]
          }
        }
      ]
    }

Top-level JSON may also be a case array.  A case may use ``questions`` instead
of ``question`` + ``paraphrases``.  Question-list entries can be strings or
objects with ``id``, ``question``, ``assertions`` and ``metadata``.  A
multi-turn case uses ``turns`` instead; every item is sent through the same
temporary session in order, so pronouns and follow-up questions exercise the
real conversation state.  Supported assertions are printed in ``--help`` and
unknown names are rejected so a typo cannot silently weaken an evaluation.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import ProxyHandler, Request, build_opener, urlopen


SCHEMA_VERSION = "icp-paraphrase-evaluation/v1"
DEFAULT_BASE_URL = "http://127.0.0.1:19999"
DEFAULT_AGENT_NAME = "ICP故障诊断智能体"

ASSERTION_LIST_FIELDS = {
    "required_tools",
    "any_tools",
    "forbidden_tools",
    "required_doc_types",
    "any_doc_types",
    "required_used_doc_types",
    "any_used_doc_types",
    "forbidden_used_doc_types",
    "required_used_record_ids",
    "any_used_record_ids",
    "forbidden_used_record_ids",
    "required_answer_facets",
    "answer_contains",
    "answer_not_contains",
    "answer_regex",
    "answer_not_repeat_prior",
    "required_event_types",
    "required_scope_routes",
    "forbidden_scope_routes",
}
ASSERTION_INTEGER_FIELDS = {
    "min_evidence",
    "min_used_evidence",
    "min_citations",
}
ASSERTION_NUMBER_FIELDS = {"max_duration_seconds"}
SUPPORTED_ASSERTIONS = (
    ASSERTION_LIST_FIELDS | ASSERTION_INTEGER_FIELDS | ASSERTION_NUMBER_FIELDS
)

# These facets intentionally describe answer semantics rather than fixed
# headings.  They let a case require a complete fault response while leaving
# the LLM free to use natural Chinese prose and a user-friendly layout.
SUPPORTED_ANSWER_FACETS = {
    "cause",
    "impact",
    "resolution",
    "history_time",
    "history_action",
    "history_result",
}

ANSWER_FACET_PATTERNS: Mapping[str, Tuple[str, ...]] = {
    "cause": (
        r"(?:原因|根因|故障机理|诱因|致因)\s*(?:是|为|包括|可能|：|:)",
        r"(?:原因|根因).{0,16}(?:指向|定位于|在于)",
        r"确认(?:故障|问题)(?:位于|来自|源于)",
        r"(?:可能|主要|潜在)(?:的)?原因",
        r"(?:核心|作用|故障)?机制\s*(?:是|为|在于|：|:)",
        r".{0,80}(?:导致|引起|造成|所致)的.{0,50}(?:故障|失效|异常)",
        r"(?:由于|源于|由|(?<!此)因).{0,100}(?:引起|导致|造成|所致)",
        r"(?:不足|拥塞|老化|退化|损坏|失效|异常|错误|松动|开路|短路).{0,45}(?:会|可)?(?:直接)?(?:引起|导致|造成)",
    ),
    "impact": (
        r"(?:局部影响|最终影响|影响|后果|风险|隐患)\s*(?:是|为|包括|：|:)",
        r"(?:会|可能|可)(?:进一步)?(?:导致|造成|引起|使得).{0,100}"
        r"(?:中断|异常|失效|降级|复位|重构|停机|离线|切换|丢失|丧失|缺失|损坏|错误|降低|下降|升高|延迟|无法|不可用|不正确|拒绝|超时)",
        r"(?:导致|造成|引起|使得).{0,100}"
        r"(?:中断|异常|失效|降级|复位|重构|停机|离线|切换|丢失|丧失|缺失|损坏|错误|降低|下降|升高|延迟|无法|不可用|不正确|拒绝|超时)",
    ),
    "resolution": (
        r"(?:处置|处理|修复|维修|排查)(?:方法|措施|步骤|建议|方案)?"
        r"\s*(?:是|为|包括|：|:)",
        r"(?:最终方案|建议操作|关键操作)\s*(?:是|为|包括|：|:)",
        r"(?:更换|校准|回滚|复测|切换|隔离|锁止|压接|重新配置|恢复参数)",
    ),
    "history_time": (
        r"(?<!\d)20\d{2}[-/.]\d{1,2}(?:[-/.]\d{1,2})?(?!\d)",
        r"20\d{2}年\d{1,2}月(?:\d{1,2}日)?",
    ),
    "history_action": (
        r"(?:排故方法|处置方法|处理过程|处理步骤|处理方案|修复方案|操作步骤|"
        r"关键操作与最终处理|关键操作|最终处理|最终方案|处置与验证)"
        r"\s*(?:是|为|包括|：|:)",
        r"(?:排故|处置|处理|修复)时.{0,180}(?:检查|确认|更换|修复|校准|回滚|切换)",
        r"(?:当时|该案例|历史案例).{0,80}(?:通过|采用|执行).{0,180}(?:检查|确认|更换|修复|校准|回滚|切换|压接)",
    ),
    "history_result": (
        r"(?:验证结果|复测结果|处置结果|处理结果)\s*(?:是|为|显示|：|:)",
        r"(?:验证结果|复测结果|处置结果|处理结果).{0,120}(?:恢复|正常|无|通过|低于|小于|满足|稳定)",
        r"验证.{0,120}(?:恢复|正常|无|通过|低于|小于|满足|稳定)",
        r"(?:修复|处置|处理|更换|回滚)后.{0,160}(?:恢复|正常|无|通过|低于|小于|满足)",
    ),
}

# The current chat_router names are already canonical.  These aliases only
# make reports readable across older/local experimental builds.
EVENT_TYPE_ALIASES = {
    "planning": "plan",
    "tool": "tool_call",
    "tool_result": "tool_call",
    "reflection": "reflect",
    "evidence_review": "stage",
    "final": "answer",
    "final_answer": "answer",
    "citation": "citations",
    "complete": "done",
    "completed": "done",
}


class EvaluationInputError(ValueError):
    """The case file or command configuration is invalid."""


class ApiError(RuntimeError):
    """The local agent API returned an invalid or unsuccessful response."""


@dataclass(frozen=True)
class EvaluationTurn:
    """One user turn within an isolated evaluation session."""

    index: int
    turn_id: str
    question: str
    assertions: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationCase:
    index: int
    case_id: str
    group_id: str
    variant_id: str
    question: str
    assertions: Mapping[str, Any]
    metadata: Mapping[str, Any]
    turns: Tuple[EvaluationTurn, ...] = ()


@dataclass(frozen=True)
class RunnerConfig:
    base_url: str
    agent_id: str
    user_id: str
    timeout_seconds: float
    keep_sessions: bool
    include_raw_events: bool
    chat_model_id: str
    embedding_model_id: str
    rerank_model_id: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _api_url(base_url: str, path: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        raise EvaluationInputError("base URL 不能为空")
    normalized_path = "/" + path.lstrip("/")
    # Accept both http://host:19999 and http://host:19999/api as a convenience.
    if base.endswith("/api") and normalized_path.startswith("/api/"):
        normalized_path = normalized_path[4:]
    return base + normalized_path


def _decode_error_body(exc: HTTPError) -> str:
    try:
        raw = exc.read()
    except Exception:
        raw = b""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return str(exc.reason or "HTTP error")
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return text[:1000]
    if isinstance(payload, Mapping):
        return str(payload.get("detail") or payload.get("message") or payload)[:1000]
    return str(payload)[:1000]


def _open_url(request: Request, timeout: float):
    """Bypass environment proxies for loopback APIs.

    Local development shells commonly export HTTP_PROXY.  urllib may send a
    localhost request through that proxy even though curl/no_proxy works,
    producing misleading 502 evaluation failures.
    """

    hostname = (urlsplit(request.full_url).hostname or "").lower()
    is_loopback = hostname == "localhost"
    if hostname:
        try:
            is_loopback = is_loopback or ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            pass
    if is_loopback:
        return build_opener(ProxyHandler({})).open(request, timeout=timeout)
    return urlopen(request, timeout=timeout)


def _request_json(
    method: str,
    url: str,
    *,
    payload: Optional[Mapping[str, Any]] = None,
    timeout: float,
) -> Any:
    body = None
    headers = {"Accept": "application/json", "Connection": "close"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method.upper())
    try:
        with _open_url(request, timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        raise ApiError(
            f"{method.upper()} {url} -> HTTP {exc.code}: {_decode_error_body(exc)}"
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ApiError(f"{method.upper()} {url} 失败: {exc}") from exc
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ApiError(f"{method.upper()} {url} 返回了无效 JSON") from exc


def _dispatch_sse_event(event_name: str, data_lines: Sequence[str]) -> Optional[Dict[str, Any]]:
    if not data_lines:
        return None
    raw_data = "\n".join(data_lines).strip()
    if not raw_data:
        return None
    if raw_data == "[DONE]":
        return {"type": "done"}
    try:
        payload = json.loads(raw_data)
    except ValueError as exc:
        raise ApiError(f"SSE data 不是合法 JSON: {raw_data[:300]}") from exc
    if not isinstance(payload, Mapping):
        raise ApiError(f"SSE data 必须是 JSON 对象，实际为 {type(payload).__name__}")
    event = dict(payload)
    if not str(event.get("type") or "").strip() and event_name:
        event["type"] = event_name
    return _normalize_event(event)


def _normalize_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(event)
    raw_type = str(normalized.get("type") or normalized.get("event") or "").strip()
    canonical_type = EVENT_TYPE_ALIASES.get(raw_type, raw_type)
    if canonical_type:
        normalized["type"] = canonical_type
    if canonical_type != raw_type and raw_type:
        normalized.setdefault("raw_type", raw_type)
    if raw_type == "evidence_review" and not normalized.get("stage"):
        normalized["stage"] = "evidence_review"
    return normalized


def _stream_sse(
    url: str,
    payload: Mapping[str, Any],
    *,
    timeout: float,
) -> Iterator[Dict[str, Any]]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "Connection": "close",
        },
        method="POST",
    )
    try:
        response = _open_url(request, timeout)
    except HTTPError as exc:
        raise ApiError(f"POST {url} -> HTTP {exc.code}: {_decode_error_body(exc)}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ApiError(f"POST {url} 失败: {exc}") from exc

    event_name = ""
    data_lines: List[str] = []
    try:
        with response:
            for raw_line in response:
                try:
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError as exc:
                    raise ApiError("SSE 响应不是 UTF-8") from exc
                if not line:
                    event = _dispatch_sse_event(event_name, data_lines)
                    if event is not None:
                        yield event
                    event_name = ""
                    data_lines = []
                    continue
                if line.startswith(":"):
                    continue
                field, separator, value = line.partition(":")
                if separator and value.startswith(" "):
                    value = value[1:]
                if field == "event":
                    event_name = value.strip()
                elif field == "data":
                    data_lines.append(value)
                # id/retry and unknown SSE extension fields are intentionally ignored.
            event = _dispatch_sse_event(event_name, data_lines)
            if event is not None:
                yield event
    except (TimeoutError, OSError) as exc:
        raise ApiError(f"读取 SSE 超时或中断: {exc}") from exc


def _as_mapping(value: Any, context: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise EvaluationInputError(f"{context} 必须是 JSON 对象")
    return dict(value)


def _string_values(value: Any, context: str) -> List[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
    else:
        raise EvaluationInputError(f"{context} 必须是字符串或字符串数组")
    normalized: List[str] = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise EvaluationInputError(f"{context} 只能包含非空字符串")
        normalized.append(item.strip())
    return list(dict.fromkeys(normalized))


def _normalize_assertions(value: Any, context: str) -> Dict[str, Any]:
    raw = _as_mapping(value, context)
    unknown = sorted(set(raw) - SUPPORTED_ASSERTIONS)
    if unknown:
        raise EvaluationInputError(
            f"{context} 含未知断言 {unknown}；支持: {sorted(SUPPORTED_ASSERTIONS)}"
        )
    normalized: Dict[str, Any] = {}
    for key, item in raw.items():
        if key in ASSERTION_LIST_FIELDS:
            normalized[key] = _string_values(item, f"{context}.{key}")
        elif key in ASSERTION_INTEGER_FIELDS:
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise EvaluationInputError(f"{context}.{key} 必须是非负整数")
            normalized[key] = item
        elif key in ASSERTION_NUMBER_FIELDS:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or item <= 0:
                raise EvaluationInputError(f"{context}.{key} 必须是正数")
            normalized[key] = float(item)
    for pattern in normalized.get("answer_regex", []):
        try:
            re.compile(pattern)
        except re.error as exc:
            raise EvaluationInputError(
                f"{context}.answer_regex 含无效正则 {pattern!r}: {exc}"
            ) from exc
    unknown_facets = sorted(
        set(normalized.get("required_answer_facets", []))
        - SUPPORTED_ANSWER_FACETS
    )
    if unknown_facets:
        raise EvaluationInputError(
            f"{context}.required_answer_facets 含未知 facet {unknown_facets}；"
            f"支持: {sorted(SUPPORTED_ANSWER_FACETS)}"
        )
    return normalized


def _merge_assertion_maps(*values: Any, context: str) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for value in values:
        merged.update(_as_mapping(value, context))
    return _normalize_assertions(merged, context)


def _question_from_variant(value: Any, context: str) -> Tuple[str, Dict[str, Any]]:
    if isinstance(value, str):
        question = value.strip()
        details: Dict[str, Any] = {}
    elif isinstance(value, Mapping):
        details = dict(value)
        question = str(details.get("question") or details.get("query") or "").strip()
    else:
        raise EvaluationInputError(f"{context} 必须是字符串或 JSON 对象")
    if not question:
        raise EvaluationInputError(f"{context} 缺少非空 question")
    return question, details


def _expand_cases(payload: Any) -> Tuple[List[EvaluationCase], Dict[str, Any]]:
    if isinstance(payload, list):
        raw_cases = payload
        root_metadata: Dict[str, Any] = {}
        global_assertions: Mapping[str, Any] = {}
    elif isinstance(payload, Mapping):
        root = dict(payload)
        raw_cases = root.get("cases")
        if not isinstance(raw_cases, list):
            raise EvaluationInputError("顶层对象必须包含 cases 数组")
        defaults = _as_mapping(root.get("defaults"), "defaults")
        global_assertions = _as_mapping(defaults.get("assertions"), "defaults.assertions")
        root_metadata = {
            key: value
            for key, value in root.items()
            if key not in {"cases", "defaults"}
        }
    else:
        raise EvaluationInputError("用例文件顶层必须是数组或含 cases 的对象")
    if not raw_cases:
        raise EvaluationInputError("cases 不能为空")

    expanded: List[EvaluationCase] = []
    seen_case_ids: set[str] = set()
    for case_index, raw_case in enumerate(raw_cases, start=1):
        case = _as_mapping(raw_case, f"cases[{case_index - 1}]")
        group_id = str(case.get("id") or f"case-{case_index}").strip()
        if not group_id:
            raise EvaluationInputError(f"cases[{case_index - 1}].id 不能为空")
        case_assertions = _as_mapping(
            case.get("assertions"), f"case {group_id}.assertions"
        )
        case_metadata = _as_mapping(case.get("metadata"), f"case {group_id}.metadata")

        if "turns" in case:
            if "questions" in case or bool(
                str(case.get("question") or case.get("query") or "").strip()
            ):
                raise EvaluationInputError(
                    f"case {group_id} 的 turns 不能与 questions、question 或 query 并用"
                )
            raw_turns = case.get("turns")
            if not isinstance(raw_turns, list) or not raw_turns:
                raise EvaluationInputError(f"case {group_id}.turns 必须是非空数组")

            turns: List[EvaluationTurn] = []
            seen_turn_ids: set[str] = set()
            for turn_index, raw_turn in enumerate(raw_turns, start=1):
                context = f"case {group_id}.turns[{turn_index - 1}]"
                question, details = _question_from_variant(raw_turn, context)
                turn_id = str(details.get("id") or f"t{turn_index}").strip()
                if not turn_id:
                    raise EvaluationInputError(f"{context}.id 不能为空")
                if turn_id in seen_turn_ids:
                    raise EvaluationInputError(
                        f"case {group_id} 存在重复 turn id: {turn_id}"
                    )
                seen_turn_ids.add(turn_id)
                assertions = _merge_assertion_maps(
                    global_assertions,
                    case_assertions,
                    details.get("assertions"),
                    context=f"{context}.assertions",
                )
                turn_metadata = _as_mapping(
                    details.get("metadata"), f"{context}.metadata"
                )
                turns.append(
                    EvaluationTurn(
                        index=turn_index,
                        turn_id=turn_id,
                        question=question,
                        assertions=assertions,
                        metadata={**case_metadata, **turn_metadata},
                    )
                )

            if group_id in seen_case_ids:
                raise EvaluationInputError(f"重复评测 ID: {group_id}")
            seen_case_ids.add(group_id)
            expanded.append(
                EvaluationCase(
                    index=len(expanded),
                    case_id=group_id,
                    group_id=group_id,
                    variant_id="dialogue",
                    question=turns[0].question,
                    assertions=turns[-1].assertions,
                    metadata=case_metadata,
                    turns=tuple(turns),
                )
            )
            continue

        has_questions = "questions" in case
        has_question = bool(str(case.get("question") or case.get("query") or "").strip())
        if has_questions and has_question:
            raise EvaluationInputError(
                f"case {group_id} 不能同时设置 questions 与 question/query"
            )
        variants: List[Tuple[Any, str]] = []
        if has_questions:
            raw_questions = case.get("questions")
            if not isinstance(raw_questions, list) or not raw_questions:
                raise EvaluationInputError(f"case {group_id}.questions 必须是非空数组")
            variants = [(item, f"q{index}") for index, item in enumerate(raw_questions, start=1)]
        elif has_question:
            variants.append(
                (
                    {
                        "id": "base",
                        "question": case.get("question") or case.get("query"),
                    },
                    "base",
                )
            )
            paraphrases = case.get("paraphrases", [])
            if not isinstance(paraphrases, list):
                raise EvaluationInputError(f"case {group_id}.paraphrases 必须是数组")
            variants.extend(
                (item, f"p{index}")
                for index, item in enumerate(paraphrases, start=1)
            )
        else:
            raise EvaluationInputError(
                f"case {group_id} 必须提供 question/query 或 questions"
            )

        multiple = len(variants) > 1
        for variant_value, default_variant_id in variants:
            context = f"case {group_id}.{default_variant_id}"
            question, details = _question_from_variant(variant_value, context)
            variant_id = str(details.get("id") or default_variant_id).strip()
            if not variant_id:
                raise EvaluationInputError(f"{context}.id 不能为空")
            case_id = f"{group_id}/{variant_id}" if multiple else group_id
            if case_id in seen_case_ids:
                raise EvaluationInputError(f"重复评测 ID: {case_id}")
            seen_case_ids.add(case_id)
            assertions = _merge_assertion_maps(
                global_assertions,
                case_assertions,
                details.get("assertions"),
                context=f"{context}.assertions",
            )
            variant_metadata = _as_mapping(details.get("metadata"), f"{context}.metadata")
            expanded.append(
                EvaluationCase(
                    index=len(expanded),
                    case_id=case_id,
                    group_id=group_id,
                    variant_id=variant_id,
                    question=question,
                    assertions=assertions,
                    metadata={**case_metadata, **variant_metadata},
                    turns=(
                        EvaluationTurn(
                            index=1,
                            turn_id=variant_id,
                            question=question,
                            assertions=assertions,
                            metadata={**case_metadata, **variant_metadata},
                        ),
                    ),
                )
            )
    return expanded, root_metadata


def _load_cases(path_text: str) -> Tuple[List[EvaluationCase], Dict[str, Any]]:
    if path_text == "-":
        text = sys.stdin.read()
    else:
        try:
            text = Path(path_text).read_text(encoding="utf-8")
        except OSError as exc:
            raise EvaluationInputError(f"无法读取用例文件 {path_text}: {exc}") from exc
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise EvaluationInputError(f"用例文件不是合法 JSON: {exc}") from exc
    return _expand_cases(payload)


def _agent_id(agent: Mapping[str, Any]) -> str:
    return str(agent.get("agent_id") or agent.get("id") or "").strip()


def _resolve_agent(base_url: str, explicit_id: str, timeout: float) -> Dict[str, Any]:
    if explicit_id:
        payload = _request_json(
            "GET",
            _api_url(base_url, f"/api/agent/{quote(explicit_id, safe='')}"),
            timeout=timeout,
        )
        if not isinstance(payload, Mapping) or not _agent_id(payload):
            raise ApiError("指定 Agent 的详情响应无 agent_id")
        return dict(payload)

    payload = _request_json(
        "GET",
        _api_url(base_url, "/api/agent?active_only=true"),
        timeout=timeout,
    )
    if isinstance(payload, Mapping):
        payload = payload.get("items") or payload.get("agents")
    if not isinstance(payload, list):
        raise ApiError("Agent 列表响应不是数组")
    matches = [
        dict(item)
        for item in payload
        if isinstance(item, Mapping)
        and str(item.get("name") or "").strip() == DEFAULT_AGENT_NAME
        and _agent_id(item)
    ]
    if len(matches) != 1:
        ids = [_agent_id(item) for item in matches]
        raise ApiError(
            f"需要唯一的默认 Agent“{DEFAULT_AGENT_NAME}”，实际找到 {len(matches)} 个"
            f"（{ids}）；请显式传 --agent-id"
        )
    return matches[0]


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value if str(item)]
    return []


def _event_trace(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    trace: List[Dict[str, Any]] = []
    common_fields = (
        "tool",
        "status",
        "stage",
        "intent",
        "next_action",
        "need_more",
        "query_targets",
        "tools_plan",
        "tried_tools",
        "selected_tools",
        "metrics",
        "evidence_count",
        "evidence_ids",
        "reviewed_evidence_ids",
        "approved_evidence_ids",
        "blocked_evidence_ids",
        "terminal_reason",
        "message_id",
    )
    for index, event in enumerate(events, start=1):
        item: Dict[str, Any] = {
            "sequence": index,
            "type": str(event.get("type") or "unknown"),
        }
        for key in common_fields:
            if key in event:
                item[key] = event[key]
        if item["type"] in {"answer", "answer_delta"}:
            item["content_length"] = len(str(event.get("content") or ""))
        if item["type"] == "answer":
            item["citation_count"] = len(event.get("citations") or [])
            item["evidence_count"] = len(event.get("evidence_catalog") or [])
        trace.append(item)
    return trace


def _is_graph_event(event: Mapping[str, Any]) -> bool:
    event_type = str(event.get("type") or "").lower()
    tool = str(event.get("tool") or "").lower()
    return (
        "graph" in event_type
        or tool in {"graph_traversal", "kg_search"}
        or "graph" in tool
        or isinstance(event.get("graph"), Mapping)
    )


def _collect_evidence_ids(events: Sequence[Mapping[str, Any]], *field_names: str) -> List[str]:
    values: List[str] = []
    for event in events:
        for field_name in field_names:
            values.extend(_string_list(event.get(field_name)))
    return list(dict.fromkeys(values))


def _evidence_id(item: Mapping[str, Any]) -> str:
    return str(item.get("evidence_id") or item.get("id") or "").strip()


def _citation_dimensions(
    citations: Sequence[Mapping[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Return dimensions from compacted citations actually used by the answer.

    The evidence catalog contains candidates shown during execution, whereas
    the public ``citations`` array is compacted after the final answer and only
    contains references whose inline marker survived.  Keeping these dimensions
    separate prevents an unused candidate from satisfying a source assertion.
    """

    doc_types = sorted(
        {
            str(item.get("doc_type") or "").strip()
            for item in citations
            if str(item.get("doc_type") or "").strip()
        }
    )
    record_ids = list(
        dict.fromkeys(
            str(item.get("record_id") or "").strip()
            for item in citations
            if str(item.get("record_id") or "").strip()
        )
    )
    return doc_types, record_ids


def _detect_answer_facets(
    answer: str,
    *,
    used_doc_types: Sequence[str],
) -> List[str]:
    """Detect domain-wide answer facets without requiring fixed headings.

    History facets additionally require a cited fault-history record.  This
    avoids accepting a current recommendation or an arbitrary date as proof
    that a previous handling case was actually presented.
    """

    # Markdown emphasis may split a semantic label from its colon, for example
    # ``**根因**：``.  Ignore presentation markers during facet detection.
    text = re.sub(r"[*_`#]", "", str(answer or ""))
    # Treat the local model's occasional bilingual duplicates as presentation
    # aliases, not as text between a semantic label and its value.  Product
    # output removes them as well; this keeps historical reports evaluable.
    text = re.sub(r"(?i)原因\s*[\uff08(]\s*cause\s*[\uff09)]", "原因", text)
    text = re.sub(r"(?i)影响\s*[\uff08(]\s*impact\s*[\uff09)]", "影响", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    has_history_source = "fault_history" in set(used_doc_types)
    detected: List[str] = []
    for facet, patterns in ANSWER_FACET_PATTERNS.items():
        if facet.startswith("history_") and not has_history_source:
            continue
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            detected.append(facet)
    return detected


def _assertion_result(name: str, passed: bool, expected: Any, actual: Any) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "expected": expected,
        "actual": actual,
    }


def _evaluate_assertions(
    assertions: Mapping[str, Any],
    *,
    answer: str,
    prior_answer: str = "",
    tools: Sequence[str],
    doc_types: Sequence[str],
    used_doc_types: Sequence[str],
    used_record_ids: Sequence[str],
    evidence_count: int,
    used_evidence_count: int,
    citation_count: int,
    event_types: Sequence[str],
    scope_routes: Sequence[str] = (),
    duration_seconds: float,
    runtime_errors: Sequence[Mapping[str, Any]],
    done_received: bool,
    answer_received: bool,
) -> List[Dict[str, Any]]:
    # 追问轮“不得原样重述”检查：answer_not_repeat_prior 给出上轮答案中的
    # 关键短语；脚本在上轮中找到含该短语的完整句子，若同一句（去空白/换行
    # 后子串命中）在本轮再原样出现，视为重述失败。命中短语但上轮句子被本轮
    # 改写（补充新维度）时不受影响。仅在追问轮用例显式声明时启用。
    repeat_prior: List[str] = []
    if prior_answer and assertions.get("answer_not_repeat_prior"):
        prior_flat = re.sub(r"\s+", "", str(prior_answer))
        answer_flat = re.sub(r"\s+", "", str(answer))
        for phrase in assertions.get("answer_not_repeat_prior", []):
            phrase = str(phrase or "").strip()
            if not phrase or phrase not in prior_flat:
                continue
            for sentence in re.split(r"[。！？!?；;\n]", prior_flat):
                if (
                    phrase in sentence
                    and len(sentence) >= 8
                    and sentence in answer_flat
                ):
                    repeat_prior.append(sentence)
    tool_set = set(tools)
    doc_type_set = set(doc_types)
    used_doc_type_set = set(used_doc_types)
    used_record_id_set = set(used_record_ids)
    event_type_set = set(event_types)
    scope_route_set = set(scope_routes)
    answer_facets = set(
        _detect_answer_facets(answer, used_doc_types=used_doc_types)
    )
    results = [
        _assertion_result("runtime.no_error", not runtime_errors, [], list(runtime_errors)),
        _assertion_result("runtime.done_received", done_received, True, done_received),
        _assertion_result("runtime.answer_received", answer_received, True, answer_received),
    ]
    for _sentence in repeat_prior:
        results.append(
            _assertion_result(
                "answer_not_repeat_prior", False, _sentence, answer
            )
        )

    required_tools = set(assertions.get("required_tools", []))
    if "required_tools" in assertions:
        results.append(
            _assertion_result(
                "required_tools",
                required_tools.issubset(tool_set),
                sorted(required_tools),
                sorted(tool_set),
            )
        )
    any_tools = set(assertions.get("any_tools", []))
    if "any_tools" in assertions:
        results.append(
            _assertion_result(
                "any_tools",
                bool(any_tools & tool_set),
                sorted(any_tools),
                sorted(tool_set),
            )
        )
    forbidden_tools = set(assertions.get("forbidden_tools", []))
    if "forbidden_tools" in assertions:
        results.append(
            _assertion_result(
                "forbidden_tools",
                not bool(forbidden_tools & tool_set),
                sorted(forbidden_tools),
                sorted(tool_set),
            )
        )

    required_doc_types = set(assertions.get("required_doc_types", []))
    if "required_doc_types" in assertions:
        results.append(
            _assertion_result(
                "required_doc_types",
                required_doc_types.issubset(doc_type_set),
                sorted(required_doc_types),
                sorted(doc_type_set),
            )
        )
    any_doc_types = set(assertions.get("any_doc_types", []))
    if "any_doc_types" in assertions:
        results.append(
            _assertion_result(
                "any_doc_types",
                bool(any_doc_types & doc_type_set),
                sorted(any_doc_types),
                sorted(doc_type_set),
            )
        )

    used_set_assertions = (
        ("required_used_doc_types", used_doc_type_set, "required"),
        ("any_used_doc_types", used_doc_type_set, "any"),
        ("forbidden_used_doc_types", used_doc_type_set, "forbidden"),
        ("required_used_record_ids", used_record_id_set, "required"),
        ("any_used_record_ids", used_record_id_set, "any"),
        ("forbidden_used_record_ids", used_record_id_set, "forbidden"),
    )
    for name, actual_set, mode in used_set_assertions:
        if name not in assertions:
            continue
        expected_set = set(assertions[name])
        if mode == "required":
            passed = expected_set.issubset(actual_set)
        elif mode == "any":
            passed = bool(expected_set & actual_set)
        else:
            passed = not bool(expected_set & actual_set)
        results.append(
            _assertion_result(
                name,
                passed,
                sorted(expected_set),
                sorted(actual_set),
            )
        )

    if "required_answer_facets" in assertions:
        expected_facets = set(assertions["required_answer_facets"])
        results.append(
            _assertion_result(
                "required_answer_facets",
                expected_facets.issubset(answer_facets),
                sorted(expected_facets),
                sorted(answer_facets),
            )
        )

    count_assertions = {
        "min_evidence": evidence_count,
        "min_used_evidence": used_evidence_count,
        "min_citations": citation_count,
    }
    for key, actual in count_assertions.items():
        if key in assertions:
            expected = assertions[key]
            results.append(_assertion_result(key, actual >= expected, expected, actual))

    for expected in assertions.get("answer_contains", []):
        results.append(
            _assertion_result(
                f"answer_contains:{expected}", expected in answer, expected, answer
            )
        )
    for forbidden in assertions.get("answer_not_contains", []):
        results.append(
            _assertion_result(
                f"answer_not_contains:{forbidden}",
                forbidden not in answer,
                forbidden,
                answer,
            )
        )
    for pattern in assertions.get("answer_regex", []):
        results.append(
            _assertion_result(
                f"answer_regex:{pattern}",
                re.search(pattern, answer) is not None,
                pattern,
                answer,
            )
        )

    if "required_event_types" in assertions:
        expected = set(assertions["required_event_types"])
        results.append(
            _assertion_result(
                "required_event_types",
                expected.issubset(event_type_set),
                sorted(expected),
                sorted(event_type_set),
            )
        )
    required_scope_routes = set(assertions.get("required_scope_routes", []))
    if "required_scope_routes" in assertions:
        results.append(
            _assertion_result(
                "required_scope_routes",
                required_scope_routes.issubset(scope_route_set),
                sorted(required_scope_routes),
                sorted(scope_route_set),
            )
        )
    forbidden_scope_routes = set(assertions.get("forbidden_scope_routes", []))
    if "forbidden_scope_routes" in assertions:
        results.append(
            _assertion_result(
                "forbidden_scope_routes",
                not bool(forbidden_scope_routes & scope_route_set),
                sorted(forbidden_scope_routes),
                sorted(scope_route_set),
            )
        )
    if "max_duration_seconds" in assertions:
        expected = assertions["max_duration_seconds"]
        results.append(
            _assertion_result(
                "max_duration_seconds",
                duration_seconds <= expected,
                expected,
                round(duration_seconds, 3),
            )
        )
    return results


def _session_title(case: EvaluationCase) -> str:
    safe_id = re.sub(r"\s+", " ", case.case_id).strip()
    prefix = "多轮对话评测" if len(case.turns) > 1 else "同义改写评测"
    return f"{prefix}-{safe_id}"[:64]


def _stop_chat(config: RunnerConfig, session_id: str) -> Optional[str]:
    url = _api_url(config.base_url, "/api/chat/stop")
    url += "?" + urlencode({"session_id": session_id})
    try:
        _request_json("POST", url, timeout=min(config.timeout_seconds, 15.0))
    except Exception as exc:
        return str(exc)
    return None


def _delete_session(config: RunnerConfig, session_id: str) -> Optional[str]:
    try:
        _request_json(
            "DELETE",
            _api_url(config.base_url, f"/api/session/{quote(session_id, safe='')}"),
            timeout=min(config.timeout_seconds, 15.0),
        )
    except Exception as exc:
        return str(exc)
    return None


def _build_turn_result(
    case: EvaluationCase,
    turn: EvaluationTurn,
    *,
    include_raw_events: bool,
    prior_answer: str = "",
    events: Sequence[Mapping[str, Any]],
    runtime_errors: Sequence[Mapping[str, Any]],
    answer: str,
    citations: Sequence[Mapping[str, Any]],
    evidence_catalog: Sequence[Mapping[str, Any]],
    used_evidence_ids: Sequence[str],
    done_received: bool,
    answer_received: bool,
    started_at: str,
    duration_seconds: float,
) -> Dict[str, Any]:
    plans = [event for event in events if event.get("type") == "plan"]
    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    graph_events = [event for event in events if _is_graph_event(event)]
    stages = [event for event in events if event.get("type") == "stage"]
    reflections = [event for event in events if event.get("type") == "reflect"]
    tools = list(
        dict.fromkeys(
            str(event.get("tool") or "").strip()
            for event in tool_calls
            if str(event.get("tool") or "").strip()
        )
    )
    event_types = [
        str(event.get("type") or "unknown") for event in events
    ]
    scope_routes = list(
        dict.fromkeys(
            str(event.get("route") or "").strip()
            for event in events
            if event.get("type") == "stage"
            and str(event.get("stage") or "") == "scope_routing"
            and str(event.get("route") or "").strip()
        )
    )
    process_evidence_ids = _collect_evidence_ids(
        events,
        "evidence_ids",
        "reviewed_evidence_ids",
        "approved_evidence_ids",
        "blocked_evidence_ids",
        "excluded_evidence_ids",
    )
    approved_evidence_ids = _collect_evidence_ids(events, "approved_evidence_ids")
    blocked_evidence_ids = _collect_evidence_ids(
        events, "blocked_evidence_ids", "excluded_evidence_ids"
    )
    catalog_evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for evidence_id in (_evidence_id(item) for item in evidence_catalog)
            if evidence_id
        )
    )
    citation_evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for evidence_id in (_evidence_id(item) for item in citations)
            if evidence_id
        )
    )
    doc_types = sorted(
        {
            str(item.get("doc_type") or "").strip()
            for item in [*evidence_catalog, *citations]
            if str(item.get("doc_type") or "").strip()
        }
    )
    used_doc_types, used_record_ids = _citation_dimensions(citations)
    answer_facets = _detect_answer_facets(
        answer,
        used_doc_types=used_doc_types,
    )
    source_refs = list(
        dict.fromkeys(
            str(item.get("source_ref") or "").strip()
            for item in [*evidence_catalog, *citations]
            if str(item.get("source_ref") or "").strip()
        )
    )
    inline_reference_numbers = list(
        dict.fromkeys(
            int(value)
            for value in re.findall(r"\[(?:\^)?(\d+)\]", answer)
        )
    )
    assertion_results = _evaluate_assertions(
        turn.assertions,
        answer=answer,
        tools=tools,
        doc_types=doc_types,
        used_doc_types=used_doc_types,
        used_record_ids=used_record_ids,
        evidence_count=len(evidence_catalog),
        used_evidence_count=len(used_evidence_ids),
        citation_count=len(citations),
        event_types=event_types,
        scope_routes=scope_routes,
        duration_seconds=duration_seconds,
        runtime_errors=runtime_errors,
        done_received=done_received,
        answer_received=answer_received,
    )
    result: Dict[str, Any] = {
        "turn_index": turn.index,
        "turn_id": turn.turn_id,
        "question": turn.question,
        "metadata": dict(turn.metadata),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(duration_seconds, 3),
        "passed": all(item["passed"] for item in assertion_results),
        "answer": answer,
        "event_types": event_types,
        "scope_routes": scope_routes,
        "event_trace": _event_trace(events),
        "plan_events": plans,
        "tool_calls": tool_calls,
        "tools": tools,
        "graph_events": graph_events,
        "reflection_events": reflections,
        "stage_events": stages,
        "evidence": {
            "catalog": evidence_catalog,
            "catalog_evidence_ids": catalog_evidence_ids,
            "process_evidence_ids": process_evidence_ids,
            "approved_evidence_ids": approved_evidence_ids,
            "blocked_evidence_ids": blocked_evidence_ids,
            "used_evidence_ids": used_evidence_ids,
            "citation_evidence_ids": citation_evidence_ids,
            "doc_types": doc_types,
            "used_doc_types": used_doc_types,
            "used_record_ids": used_record_ids,
            "source_refs": source_refs,
        },
        "citations": citations,
        "inline_reference_numbers": inline_reference_numbers,
        "answer_facets": answer_facets,
        "assertions": dict(turn.assertions),
        "assertion_results": assertion_results,
        "runtime_errors": runtime_errors,
    }
    if include_raw_events:
        result["raw_sse_events"] = list(events)
    return result


def _empty_turn_result(turn: EvaluationTurn, error: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep later turns visible when a failed SSE stream ends a dialogue early."""

    return {
        "turn_index": turn.index,
        "turn_id": turn.turn_id,
        "question": turn.question,
        "metadata": dict(turn.metadata),
        "started_at": _utc_now(),
        "finished_at": _utc_now(),
        "duration_seconds": 0.0,
        "passed": False,
        "answer": "",
        "event_types": [],
        "scope_routes": [],
        "event_trace": [],
        "plan_events": [],
        "tool_calls": [],
        "tools": [],
        "graph_events": [],
        "reflection_events": [],
        "stage_events": [],
        "evidence": {
            "catalog": [],
            "catalog_evidence_ids": [],
            "process_evidence_ids": [],
            "approved_evidence_ids": [],
            "blocked_evidence_ids": [],
            "used_evidence_ids": [],
            "citation_evidence_ids": [],
            "doc_types": [],
            "used_doc_types": [],
            "used_record_ids": [],
            "source_refs": [],
        },
        "citations": [],
        "inline_reference_numbers": [],
        "answer_facets": [],
        "assertions": dict(turn.assertions),
        "assertion_results": [
            _assertion_result("runtime.no_error", False, [], [dict(error)])
        ],
        "runtime_errors": [dict(error)],
    }


def _run_case(case: EvaluationCase, config: RunnerConfig) -> Dict[str, Any]:
    started_at = _utc_now()
    started = time.monotonic()
    session_id = ""
    chat_attempted = False
    session_should_stop = False
    cleanup: Dict[str, Any] = {
        "kept": config.keep_sessions,
        "delete_attempted": False,
        "deleted": False,
        "error": None,
    }
    turn_results: List[Dict[str, Any]] = []
    turns = case.turns or (
        EvaluationTurn(
            index=1,
            turn_id=case.variant_id or "base",
            question=case.question,
            assertions=case.assertions,
            metadata=case.metadata,
        ),
    )

    try:
        session = _request_json(
            "POST",
            _api_url(config.base_url, "/api/session"),
            payload={
                "agent_id": config.agent_id,
                "user_id": config.user_id,
                "title": _session_title(case),
            },
            timeout=config.timeout_seconds,
        )
        if not isinstance(session, Mapping):
            raise ApiError("创建会话响应不是 JSON 对象")
        session_id = str(session.get("id") or session.get("session_id") or "").strip()
        if not session_id:
            raise ApiError("创建会话响应缺少 id")

        for turn_position, turn in enumerate(turns):
            turn_started_at = _utc_now()
            turn_started = time.monotonic()
            events: List[Dict[str, Any]] = []
            runtime_errors: List[Dict[str, Any]] = []
            answer = ""
            answer_delta_parts: List[str] = []
            citations: List[Dict[str, Any]] = []
            evidence_catalog: List[Dict[str, Any]] = []
            used_evidence_ids: List[str] = []
            done_received = False
            answer_received = False

            try:
                chat_attempted = True
                chat_payload = {
                    "agent_id": config.agent_id,
                    "session_id": session_id,
                    "query": turn.question,
                    "user_id": config.user_id,
                    "chat_model_id": config.chat_model_id,
                    "embedding_model_id": config.embedding_model_id,
                    "rerank_model_id": config.rerank_model_id,
                }
                for event in _stream_sse(
                    _api_url(config.base_url, "/api/chat"),
                    chat_payload,
                    timeout=config.timeout_seconds,
                ):
                    events.append(event)
                    event_type = str(event.get("type") or "")
                    if event_type == "answer_delta":
                        answer_delta_parts.append(str(event.get("content") or ""))
                    elif event_type == "answer":
                        answer_received = True
                        answer = str(event.get("content") or "")
                        if isinstance(event.get("citations"), list):
                            citations = [
                                dict(item)
                                for item in event["citations"]
                                if isinstance(item, Mapping)
                            ]
                        if isinstance(event.get("evidence_catalog"), list):
                            evidence_catalog = [
                                dict(item)
                                for item in event["evidence_catalog"]
                                if isinstance(item, Mapping)
                            ]
                        used_evidence_ids = _string_list(event.get("used_evidence_ids"))
                    elif event_type == "citations" and isinstance(event.get("items"), list):
                        citations = [
                            dict(item) for item in event["items"] if isinstance(item, Mapping)
                        ]
                    elif event_type == "done":
                        done_received = True
                        if isinstance(event.get("citations"), list):
                            citations = [
                                dict(item)
                                for item in event["citations"]
                                if isinstance(item, Mapping)
                            ]
                        if isinstance(event.get("evidence_catalog"), list):
                            evidence_catalog = [
                                dict(item)
                                for item in event["evidence_catalog"]
                                if isinstance(item, Mapping)
                            ]
                        if "used_evidence_ids" in event:
                            used_evidence_ids = _string_list(event.get("used_evidence_ids"))
                    elif event_type in {"error", "stopped"}:
                        runtime_errors.append(
                            {
                                "kind": f"sse_{event_type}",
                                "message": str(event.get("message") or event_type),
                                "terminal_reason": event.get("terminal_reason"),
                            }
                        )
                if not answer:
                    answer = "".join(answer_delta_parts)
                answer_received = answer_received or bool(answer.strip())
            except Exception as exc:
                runtime_errors.append({"kind": type(exc).__name__, "message": str(exc)})

            turn_results.append(
                _build_turn_result(
                    case,
                    turn,
                    include_raw_events=config.include_raw_events,
                    events=events,
                    runtime_errors=runtime_errors,
                    answer=answer,
                    citations=citations,
                    evidence_catalog=evidence_catalog,
                    used_evidence_ids=used_evidence_ids,
                    done_received=done_received,
                    answer_received=answer_received,
                    started_at=turn_started_at,
                    duration_seconds=time.monotonic() - turn_started,
                )
            )
            if runtime_errors or not done_received:
                session_should_stop = True
                interruption = {
                    "kind": "prior_turn_incomplete",
                    "message": "上一轮未正常结束，后续轮次未发送，以免误用不完整会话状态。",
                }
                for pending_turn in turns[turn_position + 1:]:
                    turn_results.append(_empty_turn_result(pending_turn, interruption))
                break
    except Exception as exc:
        failure = {"kind": type(exc).__name__, "message": str(exc)}
        for turn in turns[len(turn_results):]:
            turn_results.append(_empty_turn_result(turn, failure))
    finally:
        should_stop = bool(session_id and chat_attempted and session_should_stop)
        stop_error = _stop_chat(config, session_id) if should_stop else None
        if stop_error:
            cleanup["stop_error"] = stop_error
        if session_id and not config.keep_sessions:
            cleanup["delete_attempted"] = True
            cleanup_error = _delete_session(config, session_id)
            cleanup["deleted"] = cleanup_error is None
            cleanup["error"] = cleanup_error

    duration_seconds = time.monotonic() - started
    last_turn = turn_results[-1] if turn_results else _empty_turn_result(
        turns[-1], {"kind": "no_turn_result", "message": "评测没有产生轮次结果。"}
    )
    result: Dict[str, Any] = {
        "index": case.index,
        "case_id": case.case_id,
        "group_id": case.group_id,
        "variant_id": case.variant_id,
        "question": case.question,
        "metadata": dict(case.metadata),
        "session_id": session_id,
        "session_retained": bool(session_id and config.keep_sessions),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(duration_seconds, 3),
        "passed": bool(turn_results) and all(item["passed"] for item in turn_results),
        "turn_count": len(turns),
        "turns": turn_results,
        "cleanup": cleanup,
    }
    # Keep the v1 report fields at the case level for existing single-turn
    # dashboards, while the complete trace remains in ``turns``.
    for key in (
        "answer", "event_types", "scope_routes", "event_trace", "plan_events",
        "tool_calls", "tools", "graph_events", "reflection_events", "stage_events",
        "evidence", "citations", "inline_reference_numbers", "answer_facets",
        "assertions", "assertion_results", "runtime_errors", "raw_sse_events",
    ):
        if key in last_turn:
            result[key] = last_turn[key]
    return result


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _group_summaries(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result.get("group_id") or ""), []).append(result)
    summaries: List[Dict[str, Any]] = []
    for group_id, items in grouped.items():
        # A multi-turn case is one pass/fail dialogue, but its tool/evidence
        # summaries must include every turn instead of only inheriting the
        # final turn's compatibility fields.
        turn_items = [
            turn
            for item in items
            for turn in (
                item.get("turns")
                if isinstance(item.get("turns"), list) and item.get("turns")
                else [item]
            )
            if isinstance(turn, Mapping)
        ]
        tool_sets = [set(_string_list(item.get("tools"))) for item in turn_items]
        evidence_sets = []
        for item in turn_items:
            evidence = item.get("evidence") if isinstance(item.get("evidence"), Mapping) else {}
            used = set(_string_list(evidence.get("used_evidence_ids")))
            evidence_sets.append(
                used or set(_string_list(evidence.get("catalog_evidence_ids")))
            )
        tool_union = set().union(*tool_sets) if tool_sets else set()
        tool_intersection = set.intersection(*tool_sets) if tool_sets else set()
        evidence_union = set().union(*evidence_sets) if evidence_sets else set()
        evidence_intersection = (
            set.intersection(*evidence_sets) if evidence_sets else set()
        )
        similarities = [
            _jaccard(evidence_sets[left], evidence_sets[right])
            for left, right in combinations(range(len(evidence_sets)), 2)
        ]
        passed_count = sum(bool(item.get("passed")) for item in items)
        passed_turn_count = sum(bool(item.get("passed")) for item in turn_items)
        summaries.append(
            {
                "group_id": group_id,
                "variant_count": len(items),
                "passed_count": passed_count,
                "pass_rate": round(passed_count / len(items), 4),
                "all_passed": passed_count == len(items),
                "turn_count": len(turn_items),
                "passed_turn_count": passed_turn_count,
                "turn_pass_rate": round(
                    passed_turn_count / len(turn_items), 4
                ) if turn_items else 0.0,
                "tool_union": sorted(tool_union),
                "tool_intersection": sorted(tool_intersection),
                "evidence_id_union": sorted(evidence_union),
                "evidence_id_intersection": sorted(evidence_intersection),
                "pairwise_evidence_jaccard_min": (
                    round(min(similarities), 4) if similarities else None
                ),
                "pairwise_evidence_jaccard_average": (
                    round(sum(similarities) / len(similarities), 4)
                    if similarities
                    else None
                ),
            }
        )
    return summaries


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是正整数") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是正数") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正数")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    assertion_help = ", ".join(sorted(SUPPORTED_ASSERTIONS))
    parser = argparse.ArgumentParser(
        description="调用本地 ICP Agent，评测同义改写的路由、证据和回答稳定性。",
        epilog=(
            "支持的 assertions: " + assertion_help + "\n"
            "退出码: 0=全部通过，1=运行或断言失败，2=输入/API 配置错误。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cases", help="UTF-8 JSON 用例文件；传 - 从 stdin 读取")
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help=f"服务根 URL（默认 {DEFAULT_BASE_URL}）"
    )
    parser.add_argument(
        "--agent-id",
        default="",
        help=f"Agent ID；缺省时选择唯一的“{DEFAULT_AGENT_NAME}”",
    )
    parser.add_argument("--concurrency", type=_positive_int, default=1, help="并发题数（默认 1）")
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=300.0,
        help="每个 HTTP/SSE 操作的超时秒数（默认 300）",
    )
    parser.add_argument("--user-id", default="icp-paraphrase-eval", help="临时会话 user_id")
    parser.add_argument("--chat-model-id", default="", help="可选覆盖 chat model ID")
    parser.add_argument("--embedding-model-id", default="", help="可选覆盖 embedding model ID")
    parser.add_argument("--rerank-model-id", default="", help="可选覆盖 rerank model ID")
    parser.add_argument(
        "--keep-sessions",
        action="store_true",
        help="保留评测会话以便界面复查（默认完成后尽力删除）",
    )
    parser.add_argument(
        "--include-raw-events",
        action="store_true",
        help="在分类事件之外额外保存完整原始 SSE 事件",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="报告 JSON 路径；默认 - 表示 stdout",
    )
    parser.add_argument("--compact", action="store_true", help="输出紧凑 JSON")
    return parser


def _write_report(report: Mapping[str, Any], output: str, compact: bool) -> None:
    kwargs = {"ensure_ascii": False}
    if compact:
        text = json.dumps(report, separators=(",", ":"), **kwargs)
    else:
        text = json.dumps(report, indent=2, **kwargs)
    text += "\n"
    if output == "-":
        sys.stdout.write(text)
        return
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    cases, input_metadata = _load_cases(args.cases)
    base_url = str(args.base_url).strip().rstrip("/")
    agent = _resolve_agent(base_url, str(args.agent_id).strip(), args.timeout_seconds)
    selected_agent_id = _agent_id(agent)
    if not selected_agent_id:
        raise ApiError("无法确定 Agent ID")
    config = RunnerConfig(
        base_url=base_url,
        agent_id=selected_agent_id,
        user_id=str(args.user_id),
        timeout_seconds=float(args.timeout_seconds),
        keep_sessions=bool(args.keep_sessions),
        include_raw_events=bool(args.include_raw_events),
        chat_model_id=str(args.chat_model_id),
        embedding_model_id=str(args.embedding_model_id),
        rerank_model_id=str(args.rerank_model_id),
    )
    started_at = _utc_now()
    started = time.monotonic()
    results: List[Optional[Dict[str, Any]]] = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(cases))) as executor:
        futures = {executor.submit(_run_case, case, config): case for case in cases}
        completed = 0
        for future in as_completed(futures):
            case = futures[future]
            try:
                result = future.result()
            except BaseException as exc:
                # _run_case is defensive, but retaining an internal failure in
                # the report is preferable to losing all other concurrent cases.
                result = {
                    "index": case.index,
                    "case_id": case.case_id,
                    "group_id": case.group_id,
                    "variant_id": case.variant_id,
                    "question": case.question,
                    "passed": False,
                    "runtime_errors": [
                        {"kind": type(exc).__name__, "message": str(exc)}
                    ],
                    "tools": [],
                    "evidence": {
                        "catalog_evidence_ids": [],
                        "used_evidence_ids": [],
                    },
                    "assertion_results": [],
                }
            results[case.index] = result
            completed += 1
            state = "PASS" if result.get("passed") else "FAIL"
            print(
                f"[{completed}/{len(cases)}] {state} {case.case_id}",
                file=sys.stderr,
                flush=True,
            )
    ordered_results = [item for item in results if item is not None]
    duration = time.monotonic() - started
    passed_count = sum(bool(item.get("passed")) for item in ordered_results)
    turn_results = [
        turn
        for item in ordered_results
        for turn in (
            item.get("turns")
            if isinstance(item.get("turns"), list) and item.get("turns")
            else [item]
        )
        if isinstance(turn, Mapping)
    ]
    passed_turn_count = sum(bool(item.get("passed")) for item in turn_results)
    cleanup_failures = sum(
        bool((item.get("cleanup") or {}).get("error"))
        for item in ordered_results
        if isinstance(item.get("cleanup"), Mapping)
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(duration, 3),
        "input": {
            "path": args.cases,
            "metadata": input_metadata,
        },
        "configuration": {
            "base_url": base_url,
            "agent_id": selected_agent_id,
            "agent_name": str(agent.get("name") or ""),
            "concurrency": min(args.concurrency, len(cases)),
            "timeout_seconds": args.timeout_seconds,
            "user_id": args.user_id,
            "keep_sessions": bool(args.keep_sessions),
            "include_raw_events": bool(args.include_raw_events),
            "model_overrides": {
                "chat_model_id": args.chat_model_id,
                "embedding_model_id": args.embedding_model_id,
                "rerank_model_id": args.rerank_model_id,
            },
        },
        "summary": {
            "total": len(ordered_results),
            "passed": passed_count,
            "failed": len(ordered_results) - passed_count,
            "pass_rate": round(passed_count / len(ordered_results), 4),
            "total_turns": len(turn_results),
            "passed_turns": passed_turn_count,
            "failed_turns": len(turn_results) - passed_turn_count,
            "turn_pass_rate": round(
                passed_turn_count / len(turn_results), 4
            ) if turn_results else 0.0,
            "cleanup_failures": cleanup_failures,
        },
        "groups": _group_summaries(ordered_results),
        "cases": ordered_results,
    }
    return report, 0 if passed_count == len(ordered_results) else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report, exit_code = run(args)
        _write_report(report, args.output, args.compact)
        if args.output != "-":
            print(f"报告已写入 {args.output}", file=sys.stderr)
        return exit_code
    except (EvaluationInputError, ApiError) as exc:
        print(f"评测配置错误: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("评测被用户中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
