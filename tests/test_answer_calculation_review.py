import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent.contracts import make_evidence
from backend.agent.nodes import answer as answer_module


def _run(coroutine):
    return asyncio.run(coroutine)


def _state():
    values = (
        ("电源保障链可靠性模型", "0.99951007"),
        ("核心处理功能可靠性模型", "0.99929014"),
        ("外部通信功能可靠性模型", "0.99910032"),
        ("显示输出功能可靠性模型", "0.99888052"),
    )
    evidence = []
    for index, (name, reliability) in enumerate(values, start=1):
        evidence.append(
            make_evidence(
                source_kind="calculation",
                authority="deterministic_derivation",
                evidence_level="L1_DIRECT",
                claim_type="rbd_calculation",
                doc_type="rbd",
                doc_id="doc-rbd",
                record_id=f"RBD-{index}",
                title=f"RBD 确定性计算：{name}",
                content=f"RBD 确定性计算：{name}；系统可靠度：{reliability}。",
            )
        )
    ids = [item["evidence_id"] for item in evidence]
    return {
        "query": "十小时任务中四条链路哪条成功概率最低？",
        "intent": "reliability_query",
        "calculation_requests": ["rbd_reliability"],
        "evidence_registry": evidence,
        "context_evidence_ids": ids,
        "safety_decision": {
            "approved_evidence_ids": ids,
            "context_evidence_ids": ids,
        },
    }


def _fta_state():
    evidence = make_evidence(
        source_kind="calculation",
        authority="deterministic_derivation",
        evidence_level="L1_DIRECT",
        claim_type="fta_calculation",
        doc_type="fta",
        doc_id="doc-fta",
        record_id="FTA-1",
        title="FTA 确定性计算：系统无法启动",
        content=(
            "FTA 确定性计算：系统无法启动；顶事件：系统无法启动；"
            "最小割集 2 个（完整列表：1. 供电中断；2. 时钟丢失 + 复位失效）；"
            "顶事件概率：0.0012。"
        ),
    )
    return {
        "query": "系统无法启动的最小割集有哪些？",
        "intent": "reliability_query",
        "calculation_requests": ["fta_cut_sets"],
        "evidence_registry": [evidence],
        "context_evidence_ids": [evidence["evidence_id"]],
        "safety_decision": {
            "approved_evidence_ids": [evidence["evidence_id"]],
            "context_evidence_ids": [evidence["evidence_id"]],
        },
    }


def _fta_probability_state(*, blocked=False):
    result = {
        "model": "FTA",
        "top_event": "任务数据输出中断",
        "minimal_cut_sets": [["A通道丢失"], ["B通道丢失"]],
        "minimal_cut_set_count": 2,
        "top_event_probability": None if blocked else 0.00125,
        "mission_time": 10.0,
        "missing_probability_nodes": ["B通道丢失"] if blocked else [],
        "shared_dependency_nodes": [],
        "probability_block_reason": "缺少B通道基本事件概率" if blocked else "",
        "assumption": "基本事件相互独立；AND/OR/XOR 按标准布尔逻辑计算",
    }
    evidence = make_evidence(
        source_kind="calculation",
        authority="deterministic_derivation",
        evidence_level="L2_CORROBORATED",
        claim_type="fta_calculation",
        doc_type="fta",
        doc_id="doc-fta-probability",
        record_id="FTA-PROBABILITY",
        title="FTA 确定性计算：任务数据输出中断",
        content=json.dumps(result, ensure_ascii=False),
        limitations=[result["assumption"]],
    )
    return {
        "query": "十小时内任务数据输出中断的概率是多少？",
        "intent": "reliability_query",
        "calculation_requests": ["fta_probability"],
        "evidence_registry": [evidence],
        "context_evidence_ids": [evidence["evidence_id"]],
        "safety_decision": {
            "approved_evidence_ids": [evidence["evidence_id"]],
            "context_evidence_ids": [evidence["evidence_id"]],
        },
    }























if __name__ == "__main__":
    unittest.main()
