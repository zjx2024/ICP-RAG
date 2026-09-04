import asyncio
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch


try:
    _psycopg2_spec = importlib.util.find_spec("psycopg2")
except ValueError:
    _psycopg2_spec = object() if "psycopg2" in sys.modules else None

if _psycopg2_spec is None:
    psycopg2_stub = types.ModuleType("psycopg2")
    psycopg2_stub.__spec__ = importlib.util.spec_from_loader("psycopg2", loader=None)

    def _unavailable_connect(*_args, **_kwargs):
        raise RuntimeError("psycopg2 unavailable in focused test environment")

    psycopg2_stub.connect = _unavailable_connect
    sys.modules["psycopg2"] = psycopg2_stub
elif "psycopg2" in sys.modules and getattr(sys.modules["psycopg2"], "__spec__", None) is None:
    sys.modules["psycopg2"].__spec__ = importlib.util.spec_from_loader("psycopg2", loader=None)

from backend.agent.contracts import make_evidence, make_tool_result
from backend.agent.nodes import answer as answer_module
from backend.agent.tools import icp_calculation as calculation_module
from backend.agent.tools import icp_diagnosis as diagnosis_module


def _run(coroutine):
    return asyncio.run(coroutine)


def _evidence(index: int, *, content: str = "证据内容", source_ref: str = ""):
    return make_evidence(
        source_kind="postgres",
        authority="authoritative_record",
        evidence_level="L1_DIRECT",
        claim_type="fmeca_fact",
        doc_type="fmeca",
        doc_id="doc-fmeca",
        record_id=f"FM-{index}",
        source_ref=source_ref or f"FMECA.xlsx#row={index}",
        title=f"故障模式 {index}",
        content=content,
    )


class ToolCallEvidenceMappingTests(unittest.TestCase):
    def test_fta_structure_keeps_raw_graph_internal_but_public_payload_is_readable(self):
        graph = {
            "root_fault_id": "top",
            "metadata": {
                "title": "外部数据通信中断",
                "source_refs": ["04_FTA.docx#4.2;table=11"],
                "modeling_assumption": "A/B 总线作为冗余通道。",
            },
            "vertices": [
                {
                    "id": "top",
                    "label": "TopEvent",
                    "properties": {"name": "系统无法与外设交换数据"},
                },
                {
                    "id": "gate",
                    "label": "Gate",
                    "properties": {"name": "通道故障", "gate_type": "OR"},
                },
                {
                    "id": "a",
                    "label": "BasicEvent",
                    "properties": {
                        "name": "A 通道接收超时",
                        "description": "接收器件失效、线路异常或时序配置错误。",
                    },
                },
                {
                    "id": "b",
                    "label": "BasicEvent",
                    "properties": {
                        "name": "B 通道接收超时",
                        "description": "接收器件失效、线路异常或配置错误。",
                    },
                },
            ],
            "edges": [
                {"source": "top", "target": "gate", "properties": {"gate_type": "OR"}},
                {"source": "gate", "target": "a"},
                {"source": "gate", "target": "b"},
            ],
        }
        evidence = diagnosis_module._direct_evidence(
            "fta_query",
            [
                {
                    "doc_id": "doc-fta",
                    "doc_name": "04_FTA.docx",
                    "revision_id": "revision-fta-1",
                    "title": "外部数据通信中断",
                    "graph": graph,
                    "validation_status": "auto_validated",
                    "projection_status": "published",
                    "is_active": True,
                }
            ],
        )[0]

        self.assertTrue(evidence["content"].startswith("{"))
        self.assertIn("顶事件：系统无法与外设交换数据", evidence["display_summary"])
        # 摘要只报基本事件数量，明细由 answer 确定性补全，避免向回答模型暴露
        # name（description）对导致复述/JSON 字面量回归。
        self.assertIn("基本事件共 2 项", evidence["display_summary"])
        self.assertNotIn("基本事件（节选）", evidence["display_summary"])
        self.assertNotIn("A 通道接收超时", evidence["display_summary"])
        self.assertNotIn("关键基本事件", evidence["display_summary"])
        self.assertIn("建模说明：A/B 总线作为冗余通道。", evidence["display_summary"])
        self.assertIn("4 个节点、3 条关系", evidence["display_summary"])
        self.assertEqual("04_FTA.docx#4.2;table=11", evidence["source_ref"])

        payload = answer_module._evidence_payload(evidence)
        self.assertEqual(evidence["display_summary"], payload["display_summary"])
        self.assertEqual(evidence["display_summary"], payload["content"])
        self.assertFalse(payload["content"].startswith("{"))

        calls = [{"tool": "fta_query", "results_preview": []}]
        results = [
            make_tool_result(
                tool="fta_query",
                status="ok",
                evidence_ids=[evidence["evidence_id"]],
                backend="postgres",
            )
        ]
        diagnosis_module._attach_tool_call_evidence(calls, results, [evidence])
        self.assertEqual(evidence["display_summary"][:260], calls[0]["results_preview"][0]["snippet"])
        self.assertEqual(evidence["display_summary"][:260], calls[0]["results_preview"][0]["display_summary"])

    def test_diagram_search_hit_keeps_index_content_internal_and_exposes_locator_summary(self):
        evidence = diagnosis_module._search_evidence(
            [
                {
                    "doc_id": "doc-fta",
                    "doc_name": "04_FTA.docx",
                    "doc_type_kwd": "fta",
                    "chunk_id": "fta-structure-1",
                    "content_type": "fta_structured_facts",
                    "validation_status": "provisional",
                    "title": "外部数据通信中断",
                    "content": 'FTA 显式结构事实：{"vertices":[{"id":"top"}]}',
                }
            ]
        )[0]

        self.assertIn('{"vertices"', evidence["content"])
        self.assertEqual("diagram_structure", evidence["claim_type"])
        self.assertIn("FTA 故障树检索线索", evidence["display_summary"])
        payload = answer_module._evidence_payload(evidence)
        self.assertEqual(evidence["display_summary"], payload["content"])
        self.assertNotIn("vertices", payload["content"])

    def test_diagnosis_event_uses_bounded_deduplicated_tool_result_ids(self):
        evidence = [
            _evidence(
                index,
                content="很长的证据正文" * 500,
                source_ref="FMECA.xlsx#" + "x" * 500,
            )
            for index in range(12)
        ]
        result_ids = [evidence[0]["evidence_id"], *[item["evidence_id"] for item in evidence]]
        calls = [
            {
                "tool": "fmeca_query",
                "count": 12,
                "duration": 0.01,
                "params": {},
                "results_preview": [{"title": "旧预览", "snippet": "未绑定的原始行"}],
            }
        ]
        results = [
            make_tool_result(
                tool="fmeca_query",
                status="ok",
                evidence_ids=result_ids,
                backend="postgres",
            )
        ]

        diagnosis_module._attach_tool_call_evidence(calls, results, evidence)

        self.assertEqual(12, calls[0]["evidence_count"])
        self.assertEqual(
            [item["evidence_id"] for item in evidence[:8]],
            calls[0]["evidence_ids"],
        )
        self.assertEqual(8, len(calls[0]["results_preview"]))
        self.assertTrue(all(item.get("evidence_id") for item in calls[0]["results_preview"]))
        self.assertTrue(all(len(item["snippet"]) <= 260 for item in calls[0]["results_preview"]))
        self.assertTrue(all(len(item["source_ref"]) <= 320 for item in calls[0]["results_preview"]))

    def test_fault_reasoning_alias_keeps_candidate_preview_clickable(self):
        first = _evidence(1)
        second = _evidence(2)
        calls = [
            {
                "tool": "fault_reasoning",
                "count": 1,
                "duration": 0.0,
                "params": {},
                "results_preview": [
                    {
                        "title": "输入滤波性能退化",
                        "snippet": "排序启发式分数（非概率） 0.82",
                        "evidence_ids": [second["evidence_id"]],
                    }
                ],
            }
        ]
        results = [
            make_tool_result(
                tool="evidence_synthesis",
                status="ok",
                evidence_ids=[first["evidence_id"], second["evidence_id"]],
                backend="local_deterministic",
            )
        ]

        diagnosis_module._attach_tool_call_evidence(calls, results, [first, second])

        self.assertEqual(
            [second["evidence_id"], first["evidence_id"]],
            calls[0]["evidence_ids"],
        )
        preview = calls[0]["results_preview"][0]
        self.assertEqual(second["evidence_id"], preview["evidence_id"])
        self.assertEqual(second["source_ref"], preview["source_ref"])

    def test_diagnosis_node_attaches_ids_from_the_matching_tool_result(self):
        rows = [
            {
                "doc_id": "doc-fmeca",
                "doc_name": "FMECA.xlsx",
                "code": "FM-1",
                "fault_mode": "滤波性能退化",
                "fault_cause": "电容老化",
                "local_impact": "纹波增大",
                "final_impact": "间歇异常",
                "source_ref": "FMECA.xlsx#row=1",
            }
        ]
        state = {
            "intent": "impact_analysis",
            "query": "滤波电容老化有什么影响",
            "allowed_tools": ["fmeca_query", "evidence_synthesis"],
            "sql_limit": 8,
        }
        with (
            patch.object(diagnosis_module, "_query_fmeca", return_value=rows),
            patch.object(
                diagnosis_module,
                "_enrich_function_sources",
                side_effect=lambda values, errors=None: list(values),
            ),
        ):
            result = _run(diagnosis_module.icp_diagnosis_node(state))

        call = next(item for item in result["tool_calls"] if item["tool"] == "fmeca_query")
        tool_result = next(
            item for item in result["tool_results"] if item["tool"] == "fmeca_query"
        )
        self.assertEqual(tool_result["evidence_ids"], call["evidence_ids"])
        self.assertEqual(call["evidence_ids"][0], call["results_preview"][0]["evidence_id"])
        self.assertEqual("FMECA.xlsx#row=1", call["results_preview"][0]["source_ref"])

    def test_calculation_node_links_derived_evidence_without_full_graph_payload(self):
        row = {
            "diagram_type": "rbd",
            "revision_id": "revision-rbd-1",
            "doc_id": "doc-rbd",
            "doc_name": "RBD分析.docx",
            "image_title": "显示处理链路",
            "source_ref": "RBD分析.docx#figure=1",
            "review_status": "confirmed",
            "projection_status": "published",
            "is_active": True,
            "graph": {
                "vertices": [
                    {"id": "start", "label": "Start", "properties": {"name": "START"}},
                    {
                        "id": "block",
                        "label": "ReliabilityBlock",
                        "properties": {"name": "处理模块", "reliability": 0.98},
                    },
                    {"id": "end", "label": "End", "properties": {"name": "END"}},
                ],
                "edges": [
                    {"source": "start", "target": "block"},
                    {"source": "block", "target": "end"},
                ],
            },
        }
        parent = make_evidence(
            source_kind="postgres",
            authority="confirmed_relation",
            evidence_level="L1_DIRECT",
            claim_type="rbd_structure",
            doc_type="rbd",
            doc_id="doc-rbd",
            record_id="revision-rbd-1",
            review_status="confirmed",
            content="处理模块串联路径",
            metadata={"is_active": True, "projection_status": "published"},
        )
        result = _run(
            calculation_module.icp_calculation_node(
                {
                    "query": "计算显示处理链路可靠度",
                    "allowed_tools": ["rbd_calculator"],
                    "confirmed_graphs": [row],
                    "evidence_registry": [parent],
                }
            )
        )

        call = result["tool_calls"][0]
        tool_result = result["tool_results"][0]
        self.assertEqual(tool_result["evidence_ids"], call["evidence_ids"])
        self.assertEqual(1, call["evidence_count"])
        self.assertEqual(call["evidence_ids"][0], call["results_preview"][0]["evidence_id"])
        self.assertEqual("RBD分析.docx#figure=1", call["results_preview"][0]["source_ref"])
        self.assertLessEqual(len(call["results_preview"][0]["snippet"]), 260)
        derived = result["evidence_registry"][0]
        self.assertTrue(derived["content"].startswith("{"))
        self.assertIn("RBD 确定性计算", derived["display_summary"])
        self.assertIn("成功路径 1 条", derived["display_summary"])
        self.assertIn("系统可靠度：0.98", derived["display_summary"])
        self.assertEqual(
            derived["display_summary"][:260],
            call["results_preview"][0]["snippet"],
        )


if __name__ == "__main__":
    unittest.main()
