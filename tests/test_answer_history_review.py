import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from backend.agent.contracts import make_evidence
from backend.agent.nodes import answer as answer_module


def _run(coroutine):
    return asyncio.run(coroutine)


def _history_evidence():
    evidence = make_evidence(
        source_kind="postgres",
        authority="authoritative_record",
        evidence_level="L1_DIRECT",
        claim_type="historical_case",
        doc_type="fault_history",
        doc_id="doc-history",
        record_id="HIS-SYN-020",
        title="HIS-SYN-020",
        content="某温度值固定为满量程；修复连接器后恢复。",
        metadata={
            "answer_fields": {
                "case_id": "HIS-SYN-020",
                "occurred_at": "2026-01-29 14:05 状态量巡检",
                "fault_symptom": "某温度值固定为满量程。",
                "root_cause": "连接器针脚后退导致线路开路。",
                "steps": ["检查原始计数", "重新锁止针脚"],
                "final_solution": "修复并锁止连接器针脚。",
                "verification_result": "温度变化时采集误差小于0.6 ℃。",
            }
        },
    )
    evidence["match_type"] = "semantic_id_hydration"
    evidence["retrieval_score"] = 0.94
    return evidence


def _state(evidence):
    return {
        "query": "某个温度值固定成满量程，现场怎样处理？",
        "evidence_registry": [evidence],
        "context_evidence_ids": [evidence["evidence_id"]],
        "safety_decision": {
            "approved_evidence_ids": [evidence["evidence_id"]],
            "context_evidence_ids": [evidence["evidence_id"]],
        },
        "query_frame": {
            "required_answer_fields": [
                "history_time",
                "history_root",
                "history_steps",
                "history_solution",
                "history_result",
            ],
        },
    }


class AnswerHistoryReviewTests(unittest.TestCase):





    def test_pure_history_query_keeps_authoritative_direct_history_row(self):
        evidence = _history_evidence()
        evidence["match_type"] = "structured_query"
        evidence["retrieval_score"] = None
        state = _state(evidence)
        state["intent"] = "history_query"
        state["query_targets"] = [{"intent": "history_query"}]

        context = answer_module._context_evidence(state)
        ledger = answer_module._citation_ledger(state)

        self.assertEqual(
            [evidence["evidence_id"]],
            [item["evidence_id"] for item in context],
        )
        self.assertEqual(
            [evidence["evidence_id"]],
            [item["evidence_id"] for item in ledger.values()],
        )

    def test_history_business_id_identity_query_keeps_authoritative_direct_row(self):
        evidence = _history_evidence()
        evidence["match_type"] = "structured_query"
        evidence["retrieval_score"] = None
        state = _state(evidence)
        state.update(
            {
                "query": "HIS-SYN-020 是什么历史记录？",
                "intent": "entity_profile",
                "query_targets": [
                    {
                        "intent": "entity_profile",
                        "required_fields": ["record_identity"],
                    }
                ],
                "query_frame": {
                    "coverage_scope": "focused_detail",
                    "required_answer_fields": ["record_identity"],
                },
            }
        )

        context = answer_module._context_evidence(state)
        ledger = answer_module._citation_ledger(state)

        self.assertEqual(
            [evidence["evidence_id"]],
            [item["evidence_id"] for item in context],
        )
        self.assertEqual(
            [evidence["evidence_id"]],
            [item["evidence_id"] for item in ledger.values()],
        )

    def test_history_is_strictly_filtered_when_supplementing_multi_intent_diagnosis(self):
        def history_variant(record_id, match_type, score=None):
            evidence = make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type="historical_case",
                doc_type="fault_history",
                doc_id="doc-history",
                record_id=record_id,
                title=record_id,
                content=f"{record_id} 的结构化历史案例。",
                metadata={
                    "answer_fields": {
                        "case_id": record_id,
                        "fault_symptom": "输入电压波动并触发复位。",
                        "root_cause": "输入链路器件性能退化。",
                        "final_solution": "更换异常器件并复测。",
                    }
                },
            )
            evidence["match_type"] = match_type
            evidence["retrieval_score"] = score
            return evidence

        broad = history_variant("HIS-BROAD", "structured_query")
        exact = history_variant("HIS-EXACT", "exact_history_reference")
        weak = history_variant(
            "HIS-WEAK",
            "semantic_id_hydration",
            0.15 - 0.01,
        )
        strong = history_variant(
            "HIS-STRONG",
            "semantic_id_hydration",
            0.15,
        )
        evidence = [broad, exact, weak, strong]
        evidence_ids = [item["evidence_id"] for item in evidence]
        state = {
            "query": "分析原因、影响和处理方法，如有相似历史也说明",
            "intent": "cause_analysis",
            "query_targets": [
                {"intent": "cause_analysis"},
                {"intent": "impact_analysis"},
                {"intent": "maintenance_query"},
                {"intent": "history_query"},
            ],
            "query_frame": {
                "required_answer_fields": [
                    "fault_cause",
                    "final_impact",
                    "maintenance_action",
                    "history_case",
                ],
                "diagnostic_closure": {
                    "required": True,
                    "required_dimensions": [
                        "fault_cause",
                        "final_impact",
                        "maintenance_action",
                        "history_case",
                    ],
                    "basis": "分析原因、影响和处理方法，如有相似历史也说明",
                },
            },
            "evidence_registry": evidence,
            "context_evidence_ids": evidence_ids,
            "safety_decision": {
                "approved_evidence_ids": evidence_ids,
                "context_evidence_ids": evidence_ids,
            },
        }

        context_ids = {
            item["record_id"] for item in answer_module._context_evidence(state)
        }
        ledger_ids = {
            item["record_id"]
            for item in answer_module._citation_ledger(state).values()
        }

        # The exact stored relationship is an identity boundary.  A semantic
        # neighbour remains auditable but cannot be co-presented as the same
        # parent merely because its score crosses a threshold.
        self.assertEqual({"HIS-EXACT", "HIS-STRONG", "HIS-BROAD", "HIS-WEAK"}, context_ids)
        self.assertEqual(context_ids, ledger_ids)

    def test_candidate_view_preserves_scores_and_distinct_provenance(self):
        def history_variant(record_id, match_type, score=None):
            evidence = make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type="historical_case",
                doc_type="fault_history",
                doc_id="doc-history",
                record_id=record_id,
                title=record_id,
                content=f"{record_id}：输入电压波动后触发复位。",
                metadata={
                    "answer_fields": {
                        "case_id": record_id,
                        "fault_symptom": "输入电压波动后触发复位。",
                        "root_cause": "输入链路器件性能退化。",
                        "final_solution": "更换异常器件并复测。",
                    }
                },
            )
            evidence["match_type"] = match_type
            evidence["retrieval_score"] = score
            return evidence

        best = history_variant("HIS-BEST", "semantic_id_hydration", 0.18)
        near = history_variant("HIS-NEAR", "semantic_id_hydration", 0.17)
        lagging = history_variant("HIS-LAGGING", "semantic_id_hydration", 0.08)
        below_floor = history_variant("HIS-BELOW-FLOOR", "semantic_id_hydration", 0.01)
        exact = history_variant("HIS-EXACT", "exact_history_reference")
        duplicate_exact = dict(exact)
        duplicate_exact["evidence_id"] = exact["evidence_id"] + "-duplicate"

        def state_for(evidence):
            evidence_ids = [item["evidence_id"] for item in evidence]
            return {
                "query": "分析原因、影响和处理方法，如有相似历史也说明",
                "intent": "cause_analysis",
                "query_targets": [
                    {"intent": "cause_analysis"},
                    {"intent": "impact_analysis"},
                    {"intent": "maintenance_query"},
                    {"intent": "history_query"},
                ],
                "query_frame": {
                    "required_answer_fields": ["fault_cause", "history_case"],
                    "diagnostic_closure": {
                        "required": True,
                        "required_dimensions": ["fault_cause", "history_case"],
                        "basis": "分析原因，如有相似历史也说明",
                    },
                },
                "evidence_registry": evidence,
                "context_evidence_ids": evidence_ids,
                "safety_decision": {
                    "approved_evidence_ids": evidence_ids,
                    "context_evidence_ids": evidence_ids,
                },
            }

        semantic_only = answer_module._context_evidence(
            state_for([below_floor, lagging, near, best])
        )
        with_exact = answer_module._context_evidence(
            state_for(
                [below_floor, lagging, near, best, exact, duplicate_exact]
            )
        )

        self.assertEqual(
            {"HIS-BEST", "HIS-NEAR", "HIS-LAGGING", "HIS-BELOW-FLOOR"},
            {item["record_id"] for item in semantic_only},
        )
        self.assertEqual(4, len({item["record_id"] for item in semantic_only}))
        self.assertIn(
            "HIS-LAGGING", {item["record_id"] for item in semantic_only}
        )
        self.assertIn(
            "HIS-BELOW-FLOOR", {item["record_id"] for item in semantic_only}
        )
        self.assertEqual(
            {"HIS-BEST", "HIS-NEAR", "HIS-LAGGING", "HIS-BELOW-FLOOR", "HIS-EXACT"},
            {item["record_id"] for item in with_exact},
        )
        self.assertEqual(6, len(with_exact))
        # Distinct evidence IDs remain distinct provenance, not a fuzzy merge.


    def test_low_raw_score_does_not_hide_an_approved_case(self):
        evidence = _history_evidence()
        evidence["retrieval_score"] = (
            0.15 - 0.01
        )
        state = _state(evidence)
        state["intent"] = "cause_analysis"
        state["query_targets"] = [{"intent": "cause_analysis"}]
        state["query_frame"] = {"required_answer_fields": []}

        self.assertEqual([evidence["record_id"]], [row["record_id"] for row in answer_module._context_evidence(state)])

    def test_strong_semantic_history_can_supplement_nonhistory_answer(self):
        evidence = _history_evidence()
        evidence["retrieval_score"] = 0.61
        state = _state(evidence)
        state["intent"] = "cause_analysis"
        state["query_targets"] = [{"intent": "cause_analysis"}]

        self.assertEqual(
            [evidence["evidence_id"]],
            [item["evidence_id"] for item in answer_module._context_evidence(state)],
        )





































if __name__ == "__main__":
    unittest.main()
