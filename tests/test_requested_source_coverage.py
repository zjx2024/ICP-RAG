import unittest

from backend.agent.answer_contract import focused_answer_contract
from backend.agent.nodes.answer import (
    _filter_answer_domain_evidence,
    _ensure_explicit_rbd_scope_boundary,
    _requested_source_coverage_notice,
    _source_coverage_preflight_instruction,
)
from backend.agent.tools.icp_diagnosis import _filter_diagrams_by_explicit_object_ids


class RequestedSourceCoverageTests(unittest.TestCase):
    def test_empty_explicit_rbd_query_is_visible_in_answer(self):
        answer = _requested_source_coverage_notice(
            "FMECA 给出了两个候选原因。[^1]",
            {
                "requested_source_tools": ["rbd_query"],
                "tool_results": [
                    {"tool": "rbd_query", "status": "empty", "evidence_ids": []}
                ],
                "evidence_registry": [],
                "safety_decision": {"approved_evidence_ids": []},
            },
        )

        self.assertIn("资料查询结果", answer)
        self.assertIn("RBD 可靠性框图分析报告", answer)
        self.assertIn("未找到", answer)
        self.assertIn("不能以其他资料替代", answer)

    def test_failed_explicit_source_is_not_reported_as_no_match(self):
        answer = _requested_source_coverage_notice(
            "已整理当前可用的 FMECA 资料。[^1]",
            {
                "query_frame": {"requested_source_tools": ["rbd_query"]},
                "tool_results": [{"tool": "rbd_query", "status": "failed"}],
                "evidence_registry": [],
                "safety_decision": {"approved_evidence_ids": []},
            },
        )

        self.assertIn("查询未成功", answer)
        self.assertNotIn("未找到与当前对象", answer)

    def test_approved_explicit_source_needs_no_status_footer(self):
        answer = _requested_source_coverage_notice(
            "RBD 结构已整理。[^1]",
            {
                "requested_source_tools": ["rbd_query"],
                "tool_results": [{"tool": "rbd_query", "status": "ok"}],
                "evidence_registry": [
                    {"evidence_id": "rbd-1", "doc_type": "rbd"}
                ],
                "safety_decision": {"approved_evidence_ids": ["rbd-1"]},
            },
        )

        self.assertNotIn("资料查询结果", answer)

    def test_explicit_rbd_source_sets_structure_not_calculation_boundary(self):
        instruction = _source_coverage_preflight_instruction(
            {"requested_source_tools": ["rbd_query"]}
        )

        self.assertIn("成功判据", instruction)
        self.assertIn("不能证明该模块为什么损坏", instruction)
        self.assertIn("未计算数值", instruction)
        self.assertIn("不得因此断言 RBD 文档缺少", instruction)

    def test_successful_explicit_rbd_adds_its_scope_boundary(self):
        answer = _ensure_explicit_rbd_scope_boundary(
            "RBD 成功判据已经整理。[^1]",
            {
                "requested_source_tools": ["rbd_query"],
                "_terminal_public_ledger": True,
                "context_evidence_ids": ["rbd-1"],
                "evidence_registry": [
                    {
                        "evidence_id": "rbd-1",
                        "doc_type": "rbd",
                        "claim_type": "rbd_structure",
                    }
                ],
                "safety_decision": {"approved_evidence_ids": ["rbd-1"]},
            },
        )

        self.assertIn("RBD 的结论边界", answer)
        self.assertIn("不能单独确认所询对象的根因或维修动作", answer)
        self.assertIn("[^1]", answer)

    def test_public_rbd_boundary_uses_final_citation_number(self):
        answer = _ensure_explicit_rbd_scope_boundary(
            "已整理结构。[^2]",
            {"requested_source_tools": ["rbd_query"]},
            citations=[
                {"id": 1, "doc_type": "fmeca", "type": "fmeca_fact"},
                {"id": 2, "doc_type": "rbd", "type": "rbd_structure"},
            ],
        )

        self.assertIn("RBD 的结论边界", answer)
        self.assertIn("[^2]", answer)

    def test_explicit_rbd_coverage_keeps_cause_answer_narrative(self):
        self.assertFalse(
            focused_answer_contract(
                {
                    "intent": "cause_analysis",
                    "requested_source_tools": ["rbd_query"],
                    "query_frame": {
                        "coverage_scope": "focused_detail",
                        "required_answer_fields": [
                            "fault_cause",
                            "rbd_path_structure",
                        ],
                    },
                }
            )
        )

    def test_explicit_rbd_keeps_only_graph_with_same_durable_subject_id(self):
        function_item = {
            "evidence_id": "function-msm",
            "doc_type": "function_locator",
            "metadata": {
                "answer_fields": {"component_id": "MOD-GPP-MSM"}
            },
        }
        matching_rbd = {
            "evidence_id": "rbd-msm",
            "doc_type": "rbd",
            "content": (
                '{"vertices":[{"business_id":"MOD-GPP-MSM",'
                '"properties":{"module_id":"MOD-GPP-MSM"}}]}'
            ),
        }
        unrelated_rbd = {
            "evidence_id": "rbd-pdu",
            "doc_type": "rbd",
            "content": (
                '{"vertices":[{"business_id":"MOD-PWR-PDU",'
                '"properties":{"module_id":"MOD-PWR-PDU"}}]}'
            ),
        }

        filtered = _filter_answer_domain_evidence(
            {
                "intent": "cause_analysis",
                "requested_source_tools": ["rbd_query"],
                "query_frame": {"requested_source_tools": ["rbd_query"]},
            },
            [function_item, matching_rbd, unrelated_rbd],
        )

        self.assertEqual(
            ["function-msm", "rbd-msm"],
            [item["evidence_id"] for item in filtered],
        )

    def test_explicit_rbd_tool_filter_rejects_a_broad_diagram_hit(self):
        core = {
            "diagram_key": "RBD-SYN-002",
            "graph": {
                "vertices": [
                    {"properties": {"module_id": "MOD-GPP-MSM"}},
                ]
            },
        }
        power = {
            "diagram_key": "RBD-SYN-001",
            "graph": {
                "vertices": [
                    {"properties": {"module_id": "MOD-PWR-PDU"}},
                ]
            },
        }

        filtered = _filter_diagrams_by_explicit_object_ids(
            [core, power], ["MOD-GPP-MSM"]
        )

        self.assertEqual(["RBD-SYN-002"], [row["diagram_key"] for row in filtered])


if __name__ == "__main__":
    unittest.main()
