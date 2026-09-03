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
            answer_module.HISTORY_SUPPLEMENT_MIN_SCORE - 0.01,
        )
        strong = history_variant(
            "HIS-STRONG",
            "semantic_id_hydration",
            answer_module.HISTORY_SUPPLEMENT_MIN_SCORE,
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

        self.assertEqual({"HIS-EXACT", "HIS-STRONG"}, context_ids)
        self.assertEqual({"HIS-EXACT", "HIS-STRONG"}, ledger_ids)
        self.assertNotIn("HIS-BROAD", context_ids)
        self.assertNotIn("HIS-WEAK", context_ids)

    def test_supplemental_history_uses_absolute_relative_and_unique_case_limits(self):
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
            {"HIS-BEST", "HIS-NEAR"},
            {item["record_id"] for item in semantic_only},
        )
        self.assertEqual(2, len({item["record_id"] for item in semantic_only}))
        self.assertNotIn(
            "HIS-LAGGING", {item["record_id"] for item in semantic_only}
        )
        self.assertNotIn(
            "HIS-BELOW-FLOOR", {item["record_id"] for item in semantic_only}
        )
        self.assertEqual(
            {"HIS-EXACT", "HIS-BEST"},
            {item["record_id"] for item in with_exact},
        )
        self.assertEqual(2, len({item["record_id"] for item in with_exact}))
        self.assertEqual(2, len(with_exact))

    def test_history_similarity_wording_does_not_claim_current_root_cause_identity(self):
        state = {
            "intent": "history_query",
            "safety_decision": {
                "can_claim_unique": False,
                "blocked_claims": [],
                "questions": [],
            },
        }

        result = answer_module._enforce_answer_safety(
            "系统中曾发生过完全相同的案例，且根因均指向连接器。",
            state,
            architecture_query=False,
        )

        self.assertIn("高度相似的案例", result)
        self.assertIn("该历史案例的根因指向连接器", result)
        self.assertNotIn(answer_module._UNIQUE_DIAGNOSIS_BOUNDARY, result)

    def test_weak_semantic_history_stays_audit_only_for_nonhistory_answer(self):
        evidence = _history_evidence()
        evidence["retrieval_score"] = (
            answer_module.HISTORY_SUPPLEMENT_MIN_SCORE - 0.01
        )
        state = _state(evidence)
        state["intent"] = "cause_analysis"
        state["query_targets"] = [{"intent": "cause_analysis"}]

        self.assertEqual([], answer_module._context_evidence(state))

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

    def test_manifest_contains_nonempty_fields_for_referenced_approved_history(self):
        evidence = _history_evidence()

        manifest, unsupported = answer_module._history_review_manifest(
            "类似案例 HIS-SYN-020 已处理完成。[^1]",
            _state(evidence),
        )

        self.assertFalse(unsupported)
        self.assertEqual(1, len(manifest))
        self.assertEqual("2026-01-29 14:05 状态量巡检", manifest[0]["answer_fields"]["occurred_at"])
        self.assertEqual("温度变化时采集误差小于0.6 ℃。", manifest[0]["answer_fields"]["verification_result"])

    def test_llm_review_can_restore_omitted_history_time(self):
        evidence = _history_evidence()
        draft = "类似案例 HIS-SYN-020 中，重新锁止针脚后恢复并通过验证。[^1]"
        reviewed = (
            "类似案例 HIS-SYN-020 发生于 2026-01-29 14:05；根因是连接器针脚后退导致线路开路，"
            "当时检查原始计数并重新锁止针脚，最终修复连接器；验证采集误差小于 0.6 ℃。[^1]"
        )

        with patch.object(answer_module.llm_service, "_call_model", return_value=reviewed) as call:
            result = _run(answer_module._review_answer_history(draft, _state(evidence), None))

        call.assert_called_once()
        self.assertEqual(reviewed, result)
        prompt = call.call_args.args[0][0]["content"]
        self.assertIn("2026-01-29 14:05", prompt)
        self.assertIn("验证结果", prompt)
        self.assertNotIn('"answer_fields"', prompt)

    def test_strong_unreferenced_history_is_offered_for_complete_fault_answer(self):
        evidence = _history_evidence()
        manifest, unsupported = answer_module._history_review_manifest(
            "该现象通常与线路开路有关，建议先检查连接器。",
            _state(evidence),
        )

        self.assertFalse(unsupported)
        self.assertEqual("HIS-SYN-020", manifest[0]["record_id"])

    def test_explicit_history_query_restores_unreferenced_structured_record(self):
        evidence = _history_evidence()
        evidence["match_type"] = "structured_query"
        evidence["retrieval_score"] = None
        state = _state(evidence)
        state["intent"] = "history_query"
        state["query_targets"] = [{"intent": "history_query"}]

        manifest, unsupported = answer_module._history_review_manifest(
            "当前回答漏掉了检索到的历史记录。",
            state,
        )

        self.assertFalse(unsupported)
        self.assertEqual(["HIS-SYN-020"], [item["record_id"] for item in manifest])

    def test_explicit_history_prefers_strong_semantic_case_over_cited_broad_row(self):
        unrelated = _history_evidence()
        unrelated["match_type"] = "structured_query"
        unrelated["retrieval_score"] = None
        matching = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="historical_case",
            doc_type="fault_history",
            doc_id="doc-history",
            record_id="HIS-SYN-011",
            title="HIS-SYN-011",
            content="A通道接收计数不变，B通道正常；重新压接后恢复。",
            metadata={
                "answer_fields": {
                    "case_id": "HIS-SYN-011",
                    "occurred_at": "2025-08-02 14:15",
                    "fault_symptom": "A通道接收计数不变，B通道正常。",
                    "root_cause": "终端针脚未完全锁止。",
                    "steps": ["检查物理层幅值", "重新压接针脚"],
                    "final_solution": "重新压接并固定针脚。",
                    "verification_result": "连续接收两小时无丢帧。",
                }
            },
        )
        matching["match_type"] = "semantic_id_hydration"
        matching["retrieval_score"] = 0.81
        ids = [unrelated["evidence_id"], matching["evidence_id"]]
        state = {
            "query": "以前有没有A总线不收、B总线正常的案例？",
            "intent": "history_query",
            "query_targets": [{"intent": "history_query"}],
            "evidence_registry": [unrelated, matching],
            "context_evidence_ids": ids,
            "safety_decision": {
                "approved_evidence_ids": ids,
                "context_evidence_ids": ids,
            },
        }

        manifest, unsupported = answer_module._history_review_manifest(
            "温度线路案例曾通过锁止针脚恢复。[^1]",
            state,
        )

        self.assertFalse(unsupported)
        self.assertEqual(["HIS-SYN-011"], [item["record_id"] for item in manifest])
        self.assertEqual(2, manifest[0]["citation_id"])

    def test_structured_history_does_not_pollute_ordinary_fault_answer(self):
        evidence = _history_evidence()
        evidence["match_type"] = "structured_query"
        evidence["retrieval_score"] = None
        state = _state(evidence)
        state["intent"] = "cause_analysis"
        state["query_targets"] = [{"intent": "cause_analysis"}]

        manifest, unsupported = answer_module._history_review_manifest(
            "线路开路可能导致温度采集固定为满量程。",
            state,
        )

        self.assertFalse(unsupported)
        self.assertEqual([], manifest)

    def test_explicit_structured_history_enters_completeness_review(self):
        evidence = _history_evidence()
        evidence["match_type"] = "structured_query"
        evidence["retrieval_score"] = None
        state = _state(evidence)
        state["intent"] = "history_query"
        state["query_targets"] = [{"intent": "history_query"}]
        draft = "检索到了相关历史，但初稿没有整理具体记录。"
        reviewed = (
            "HIS-SYN-020 发生于 2026-01-29 14:05；根因是连接器针脚后退导致线路开路；"
            "当时检查原始计数并重新锁止针脚，最终修复并锁止连接器针脚；"
            "验证时温度变化的采集误差小于 0.6 ℃。[^1]"
        )

        with patch.object(answer_module.llm_service, "_call_model", return_value=reviewed) as call:
            result = _run(answer_module._review_answer_history(draft, state, None))

        call.assert_called_once()
        self.assertEqual(reviewed, result)
        prompt = call.call_args.args[0][0]["content"]
        self.assertIn("HIS-SYN-020", prompt)
        self.assertIn("2026-01-29 14:05", prompt)

    def test_second_review_attempt_targets_still_missing_time(self):
        evidence = _history_evidence()
        draft = "类似案例 HIS-SYN-020 中，重新锁止针脚后恢复。[^1]"
        first_review = "HIS-SYN-020 的根因是针脚后退，重新锁止后验证恢复。[^1]"
        complete_review = (
            "HIS-SYN-020 发生于 2026-01-29 14:05；根因是针脚后退，"
            "重新锁止后验证采集误差小于 0.6 ℃。[^1]"
        )

        with patch.object(
            answer_module.llm_service,
            "_call_model",
            side_effect=[first_review, complete_review],
        ) as call:
            result = _run(answer_module._review_answer_history(draft, _state(evidence), None))

        self.assertEqual(2, call.call_count)
        self.assertEqual(complete_review, result)
        self.assertIn("occurred_at=2026-01-29 14:05", call.call_args.args[0][0]["content"])

    def test_generic_current_advice_does_not_satisfy_history_details(self):
        evidence = _history_evidence()
        records, _unsupported = answer_module._history_review_manifest(
            "历史案例发生于 2026-01-29 14:05，根因是连接器针脚后退。[^1]",
            _state(evidence),
        )
        draft = (
            "建议当前先检查线路，必要时更换模块并完成验证。[^2]\n\n"
            "历史案例发生于 2026-01-29 14:05，根因是连接器针脚后退。[^1]"
        )

        missing = answer_module._history_review_missing_fields(draft, records)

        self.assertIn("[^1] steps/diagnosis_method", missing)
        self.assertTrue(any("final_solution=" in item for item in missing))
        self.assertTrue(any("verification_result=" in item for item in missing))

    def test_history_details_must_be_specific_and_bound_to_history_citation(self):
        evidence = _history_evidence()
        records, _unsupported = answer_module._history_review_manifest(
            "类似案例 HIS-SYN-020。[^1]",
            _state(evidence),
        )
        complete = (
            "类似案例发生于 2026-01-29 14:05，根因是连接器针脚后退导致线路开路。[^1]\n\n"
            "当时检查原始计数并重新锁止针脚，最终修复并锁止连接器针脚。[^1]\n\n"
            "验证时温度变化的采集误差小于 0.6 ℃。[^1]"
        )

        self.assertEqual(
            [],
            answer_module._history_review_missing_fields(complete, records),
        )

    def test_history_review_adds_specific_operation_solution_and_result(self):
        evidence = _history_evidence()
        draft = (
            "建议当前先检查线路，必要时更换模块并完成验证。[^2]\n\n"
            "类似案例发生于 2026-01-29 14:05，根因是连接器针脚后退。[^1]"
        )
        reviewed = (
            "建议当前先检查线路，必要时更换模块并完成验证。[^2]\n\n"
            "类似案例发生于 2026-01-29 14:05，根因是连接器针脚后退导致线路开路；"
            "当时检查原始计数并重新锁止针脚，最终修复并锁止连接器针脚；"
            "验证时温度变化的采集误差小于 0.6 ℃。[^1]"
        )

        with patch.object(answer_module.llm_service, "_call_model", return_value=reviewed) as call:
            result = _run(answer_module._review_answer_history(draft, _state(evidence), None))

        call.assert_called_once()
        self.assertEqual(reviewed, result)
        prompt = call.call_args.args[0][0]["content"]
        self.assertIn("steps/diagnosis_method", prompt)
        self.assertIn("final_solution=", prompt)
        self.assertIn("verification_result=", prompt)

    def test_history_tone_without_history_source_triggers_source_correction(self):
        fmeca = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-SYN-002",
            title="输入滤波退化",
            content="建议检查滤波器件。",
        )
        state = _state(fmeca)
        draft = "以前曾记录该问题，最终通过更换模块解决。[^1]"
        corrected = "FMECA 资料建议检查滤波器件。[^1]"

        with patch.object(answer_module.llm_service, "_call_model", return_value=corrected) as call:
            result = _run(answer_module._review_answer_history(draft, state, None))

        call.assert_called_once()
        self.assertEqual(corrected, result)

    def test_history_detail_block_without_history_citation_is_removed(self):
        fmeca = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-NONHISTORY",
            title="连接器接触异常",
            content="连接器接触异常会造成信号中断，建议检查连接器。",
        )
        state = _state(fmeca)
        answer = (
            "FMECA 记录的当前检查建议是检查连接器。[^1]\n\n"
            "### 相似历史\n"
            "- 当时现象：信号中断。[^1]\n"
            "- 根因：连接器接触异常。[^1]\n"
            "- 关键操作：重新压接。[^1]\n"
            "- 最终方案：更换连接器。[^1]\n"
            "- 验证结果：连续运行正常。[^1]"
        )

        result = answer_module._strip_unbound_history_assertions(answer, state)

        self.assertIn("当前检查建议", result)
        for text in ("相似历史", "当时现象", "根因：", "关键操作", "最终方案", "验证结果"):
            self.assertNotIn(text, result)

    def test_history_detail_block_with_history_citation_is_preserved(self):
        evidence = _history_evidence()
        state = _state(evidence)
        answer = (
            "### 相似历史\n"
            "- 当时现象：某温度值固定为满量程。[^1]\n"
            "- 根因：连接器针脚后退导致线路开路。[^1]\n"
            "- 关键操作：检查原始计数并重新锁止针脚。[^1]\n"
            "- 最终方案：修复并锁止连接器针脚。[^1]\n"
            "- 验证结果：温度变化时采集误差小于 0.6 ℃。[^1]"
        )

        self.assertEqual(
            answer,
            answer_module._strip_unbound_history_assertions(answer, state),
        )

    def test_nonhistorical_answer_skips_review(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-SYN-002",
            title="输入滤波性能退化",
            content="滤波性能退化会增大输入纹波。",
        )
        draft = "滤波性能退化会增大输入纹波。[^1]"

        with patch.object(
            answer_module.llm_service,
            "_call_model",
            side_effect=AssertionError("不应调用历史校对模型"),
        ):
            result = _run(answer_module._review_answer_history(draft, _state(evidence), None))

        self.assertEqual(draft, result)

    def test_review_failure_preserves_llm_draft(self):
        evidence = _history_evidence()
        draft = "类似案例 HIS-SYN-020 已处理完成。[^1]"

        with patch.object(answer_module.llm_service, "_call_model", side_effect=RuntimeError("offline")):
            result = _run(answer_module._review_answer_history(draft, _state(evidence), None))

        self.assertEqual(draft, result)

    def test_editor_manifest_json_leak_is_rejected(self):
        evidence = _history_evidence()
        draft = "类似案例 HIS-SYN-020 已处理完成。[^1]"
        leaked = (
            draft
            + '\n\n[\n  {\n    "record_id": "HIS-SYN-020",\n'
            + '    "answer_fields": {"occurred_at": "2026-01-29 14:05"}\n  }\n]'
        )

        with patch.object(
            answer_module.llm_service,
            "_call_model",
            side_effect=[leaked, leaked],
        ) as call:
            result = _run(answer_module._review_answer_history(draft, _state(evidence), None))

        self.assertEqual(2, call.call_count)
        self.assertEqual(draft, result)

    def test_internal_empty_manifest_text_is_never_user_visible(self):
        answer = (
            "建议检查网络队列。[^1]\n\n"
            "无。原回答没有引用获准使用的 fault_history 记录。[^1]"
        )

        self.assertEqual(
            "建议检查网络队列。[^1]",
            answer_module._strip_internal_editor_artifacts(answer),
        )

    def test_trailing_machine_manifest_is_removed_before_user_output(self):
        answer = (
            "历史案例在修复连接器后验证正常。[^1]\n\n"
            "[\n  {\n    \"record_id\": \"HIS-SYN-020\",\n"
            "    \"answer_fields\": {\"occurred_at\": \"2026-01-29\"}\n  }\n]"
        )

        self.assertEqual(
            "历史案例在修复连接器后验证正常。[^1]",
            answer_module._strip_internal_editor_artifacts(answer),
        )

    def test_explicit_history_final_closure_restores_all_authoritative_fields(self):
        evidence = _history_evidence()
        state = _state(evidence)
        state["intent"] = "history_query"
        state["query_targets"] = [{"intent": "history_query"}]
        draft = (
            "有高度相似案例。[^1]\n\n"
            "### 相似历史\n"
            "- **差异提示**：还需根据现场状态区分。[^1]"
        )

        result = answer_module._ensure_approved_history_completeness(draft, state)

        for expected in (
            "2026-01-29 14:05",
            "某温度值固定为满量程",
            "连接器针脚后退导致线路开路",
            "检查原始计数",
            "重新锁止针脚",
            "修复并锁止连接器针脚",
            "温度变化时采集误差小于0.6 ℃",
        ):
            self.assertIn(expected, result)
        detail_lines = [
            line
            for line in result.splitlines()
            if any(
                label in line
                for label in (
                    "发生时间",
                    "当时现象",
                    "确认根因",
                    "关键操作",
                    "最终方案",
                    "验证结果",
                )
            )
        ]
        self.assertTrue(detail_lines)
        self.assertTrue(all("[^1]" in line for line in detail_lines))

    def test_numbered_cited_history_steps_do_not_repeat_operations_or_solution(self):
        evidence = _history_evidence()
        state = _state(evidence)
        state["intent"] = "history_query"
        state["query_targets"] = [{"intent": "history_query"}]
        draft = (
            "### 相似历史\n"
            "- **发生时间**：2026-01-29 14:05 状态量巡检。[^1]\n"
            "- **当时现象**：某温度值固定为满量程。[^1]\n"
            "- **确认根因**：连接器针脚后退导致线路开路。[^1]\n"
            "    1. 检查原始计数。[^1]\n"
            "    2. 重新锁止针脚。[^1]\n"
            "    3. 修复并锁止连接器针脚。[^1]\n"
            "- **验证结果**：温度变化时采集误差小于0.6 ℃。[^1]"
        )

        result = answer_module._ensure_approved_history_completeness(draft, state)

        self.assertEqual(draft, result)
        self.assertNotIn("**关键操作**", result)
        self.assertNotIn("**最终方案**", result)

    def test_natural_complete_history_paragraph_is_idempotent_and_not_relisted(self):
        evidence = _history_evidence()
        state = _state(evidence)
        state["intent"] = "history_query"
        state["query_targets"] = [{"intent": "history_query"}]
        draft = (
            "### 相似历史\n"
            "该案例发生于 2026-01-29 14:05 状态量巡检；当时现象为某温度值固定为满量程；"
            "确认根因是连接器针脚后退导致线路开路；当时先检查原始计数，再重新锁止针脚；"
            "最终修复并锁止连接器针脚，复测时温度变化的采集误差小于0.6 ℃。[^1]"
        )

        once = answer_module._ensure_approved_history_completeness(draft, state)
        twice = answer_module._ensure_approved_history_completeness(once, state)

        self.assertEqual(draft, once)
        self.assertEqual(once, twice)
        self.assertEqual(1, once.count("相似历史"))
        self.assertNotIn("相关历史案例", once)

    def test_missing_history_fields_are_inserted_into_existing_history_section(self):
        evidence = _history_evidence()
        state = _state(evidence)
        state["intent"] = "history_query"
        state["query_targets"] = [{"intent": "history_query"}]
        draft = "### 相似历史\n已有一条相关记录。[^1]\n\n### 当前建议\n先检查连接器。[^1]"

        result = answer_module._ensure_approved_history_completeness(draft, state)

        self.assertEqual(1, result.count("### 相似历史"))
        self.assertNotIn("### 相关历史案例", result)
        self.assertLess(result.index("HIS-SYN-020"), result.index("### 当前建议"))

    def test_numbered_current_advice_does_not_hide_missing_history_details(self):
        evidence = _history_evidence()
        state = _state(evidence)
        state["intent"] = "history_query"
        state["query_targets"] = [{"intent": "history_query"}]
        draft = (
            "### 当前建议\n"
            "1. 建议当前检查原始计数。[^1]\n"
            "2. 建议修复并锁止连接器针脚。[^1]\n\n"
            "### 相似历史\n"
            "- **发生时间**：2026-01-29 14:05 状态量巡检。[^1]\n"
            "- **当时现象**：某温度值固定为满量程。[^1]\n"
            "- **确认根因**：连接器针脚后退导致线路开路。[^1]\n"
            "- **验证结果**：温度变化时采集误差小于0.6 ℃。[^1]"
        )

        result = answer_module._ensure_approved_history_completeness(draft, state)

        self.assertIn("**处理过程**", result)
        self.assertIn("**最终方案**", result)

    def test_missing_history_operations_use_readable_markdown_steps(self):
        evidence = _history_evidence()
        evidence["metadata"]["answer_fields"]["diagnosis_method"] = (
            "原始计数检查和连接器锁止确认。"
        )
        state = _state(evidence)
        state["intent"] = "history_query"
        state["query_targets"] = [{"intent": "history_query"}]

        result = answer_module._ensure_approved_history_completeness(
            "有一条高度相似历史。[^1]",
            state,
        )

        self.assertIn("- **排故方法**", result)
        self.assertIn("\n**处理过程**\n", result)
        self.assertIn("\n1. 检查原始计数[^1]\n", result)
        self.assertIn("\n2. 重新锁止针脚[^1]\n", result)
        self.assertNotIn("关键操作依次为", result)

    def test_strong_supplemental_history_is_completed_without_initial_citation(self):
        evidence = _history_evidence()
        evidence["retrieval_score"] = 0.61
        state = _state(evidence)
        state["intent"] = "cause_analysis"
        state["query_targets"] = [{"intent": "cause_analysis"}]
        draft = "优先检查连接器和采集通道。"

        result = answer_module._ensure_approved_history_completeness(draft, state)

        self.assertIn("相关历史案例", result)
        self.assertIn("HIS-SYN-020", result)
        self.assertIn("2026-01-29 14:05", result)
        self.assertIn("验证结果", result)
        self.assertIn("[^1]", result)

    def test_answer_node_terminally_restores_only_strong_requested_history(self):
        fmeca = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-SYN-002",
            title="输入滤波性能退化",
            content="输入滤波性能退化会导致输入纹波增大。",
            metadata={
                "answer_fields": {
                    "failure_mode": "输入滤波性能退化",
                    "fault_cause": "滤波电容老化",
                    "local_impact": "输入纹波增大",
                    "final_impact": "系统可能间歇复位",
                }
            },
        )
        strong = _history_evidence()
        strong["retrieval_score"] = 0.18

        weak = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="historical_case",
            doc_type="fault_history",
            doc_id="doc-history",
            record_id="HIS-WEAK",
            title="HIS-WEAK",
            content="低相关历史记录。",
            metadata={
                "answer_fields": {
                    "case_id": "HIS-WEAK",
                    "occurred_at": "2024-01-01 08:00",
                    "fault_symptom": "不相关的显示闪烁。",
                    "root_cause": "显示接口松动。",
                    "steps": ["检查显示接口"],
                    "final_solution": "重新固定显示接口。",
                    "verification_result": "显示恢复。",
                }
            },
        )
        weak["match_type"] = "semantic_id_hydration"
        weak["retrieval_score"] = answer_module.HISTORY_SUPPLEMENT_MIN_SCORE - 0.01

        broad = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="historical_case",
            doc_type="fault_history",
            doc_id="doc-history",
            record_id="HIS-BROAD",
            title="HIS-BROAD",
            content="普通宽关键词历史记录。",
            metadata={
                "answer_fields": {
                    "case_id": "HIS-BROAD",
                    "occurred_at": "2023-02-02 10:00",
                    "fault_symptom": "普通设备异常。",
                    "root_cause": "未明确定位。",
                    "steps": ["常规检查"],
                    "final_solution": "恢复配置。",
                    "verification_result": "设备恢复。",
                }
            },
        )
        broad["match_type"] = "structured_query"
        evidence = [fmeca, strong, weak, broad]
        evidence_ids = [item["evidence_id"] for item in evidence]
        state = {
            "query": "请综合说明原因、影响、处理建议，并补充相似历史",
            "intent": "cause_analysis",
            "query_targets": [
                {"intent": "cause_analysis"},
                {"intent": "impact_analysis"},
                {"intent": "maintenance_query"},
                {"intent": "history_query"},
            ],
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": [
                    "failure_mode",
                    "fault_cause",
                    "local_impact",
                    "final_impact",
                    "maintenance_action",
                    "history_time",
                    "history_root",
                    "history_steps",
                    "history_solution",
                    "history_result",
                ],
            },
            "messages": [],
            "candidate_faults": [],
            "diagnostic_support": 0.72,
            "evidence_registry": evidence,
            "context_evidence_ids": evidence_ids,
            "safety_decision": {
                "status": "answer",
                "can_claim_unique": False,
                "approved_evidence_ids": evidence_ids,
                "blocked_evidence_ids": [],
                "context_evidence_ids": evidence_ids,
                "questions": [],
            },
        }
        ledger = answer_module._citation_ledger(state)
        fmeca_citation = next(
            citation
            for citation, item in ledger.items()
            if item["record_id"] == "FM-SYN-002"
        )
        draft = (
            "输入滤波性能退化的原因是滤波电容老化，"
            "会造成输入纹波增大并可能引发系统间歇复位。"
            f"[^{fmeca_citation}]"
        )

        passthrough = AsyncMock(side_effect=lambda answer, *_args, **_kwargs: answer)
        with (
            patch.object(
                answer_module,
                "_with_answer_evidence_rerank",
                new=AsyncMock(return_value=state),
            ),
            patch.object(
                answer_module,
                "_generate_answer_response",
                new=AsyncMock(return_value=draft),
            ),
            patch.object(
                answer_module,
                "_review_answer_target_completeness",
                new=passthrough,
            ),
            patch.object(
                answer_module,
                "_review_answer_fault_completeness",
                new=passthrough,
            ),
            patch.object(
                answer_module,
                "_review_answer_history",
                new=passthrough,
            ),
            patch.object(
                answer_module,
                "_review_answer_calculations",
                new=passthrough,
            ),
            patch.object(
                answer_module,
                "_repair_answer_citations",
                new=passthrough,
            ),
            patch.object(
                answer_module.agent_model_role_service,
                "get_model_by_role",
                return_value=None,
            ),
            patch.object(
                answer_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(answer_module.answer_node(state))

        final_answer = result["final_answer"]
        for expected in (
            "2026-01-29 14:05",
            "连接器针脚后退导致线路开路",
            "检查原始计数",
            "重新锁止针脚",
            "修复并锁止连接器针脚",
            "温度变化时采集误差小于0.6 ℃",
        ):
            self.assertIn(expected, final_answer)
        for excluded in (
            "HIS-WEAK",
            "显示接口松动",
            "HIS-BROAD",
            "普通设备异常",
        ):
            self.assertNotIn(excluded, final_answer)
        self.assertIn(
            strong["evidence_id"],
            [item["evidence_id"] for item in result["citations"]],
        )
        self.assertNotIn(
            weak["evidence_id"],
            [item["evidence_id"] for item in result["citations"]],
        )
        self.assertNotIn(
            broad["evidence_id"],
            [item["evidence_id"] for item in result["citations"]],
        )

    def test_weak_or_unapproved_history_is_not_completed(self):
        weak = _history_evidence()
        weak["retrieval_score"] = 0.10
        weak_state = _state(weak)
        weak_state["intent"] = "cause_analysis"
        weak_state["query_targets"] = [{"intent": "cause_analysis"}]
        draft = "优先检查连接器。"
        self.assertEqual(
            draft,
            answer_module._ensure_approved_history_completeness(draft, weak_state),
        )

        approved = _history_evidence()
        unapproved_state = _state(approved)
        unapproved_state["intent"] = "history_query"
        unapproved_state["query_targets"] = [{"intent": "history_query"}]
        unapproved_state["safety_decision"]["approved_evidence_ids"] = []
        self.assertEqual(
            draft,
            answer_module._ensure_approved_history_completeness(draft, unapproved_state),
        )

    def test_completed_history_survives_deterministic_citation_cleanup(self):
        evidence = _history_evidence()
        state = _state(evidence)
        state["intent"] = "history_query"
        state["query_targets"] = [{"intent": "history_query"}]
        completed = answer_module._ensure_approved_history_completeness(
            "有一条相似历史。[^1]",
            state,
        )

        with patch.object(answer_module.llm_service, "_call_model", return_value="not json"):
            repaired = _run(
                answer_module._repair_answer_citations(
                    completed,
                    state,
                    None,
                    citation_count=1,
                )
            )
        result = answer_module._downgrade_unapproved_assertions(repaired, state)

        for expected in (
            "2026-01-29 14:05",
            "连接器针脚后退导致线路开路",
            "修复并锁止连接器针脚",
            "温度变化时采集误差小于0.6 ℃",
        ):
            self.assertIn(expected, result)
        self.assertNotIn("occurred_at=", result)
        self.assertNotIn("verification_result=", result)

    def test_internal_history_assignments_are_localized_not_leaked(self):
        answer = (
            "历史详情：occurred_at=2025-08-02 14:15；"
            "root_cause=终端针脚未锁止；"
            "verification_result=连续接收2 h无丢帧。[^1]"
        )

        result = answer_module._strip_internal_editor_artifacts(answer)

        self.assertIn("发生时间：2025-08-02 14:15", result)
        self.assertIn("确认根因：终端针脚未锁止", result)
        self.assertIn("验证结果：连续接收2 h无丢帧", result)
        self.assertNotRegex(
            result,
            r"\b(?:occurred_at|root_cause|verification_result)\s*=",
        )

    def test_explicit_history_query_admits_referenced_hybrid_recall_row(self):
        """显式历史问题下，被草稿引用的混合检索案例行必须进入补全清单。

        回归：此前 match_type 非 structured_query 且非强匹配的已引用记录被
        整条拒绝，导致引用修复后无历史字段可补（NO_RECORDS 但账本含已引用
        HIS 记录）。
        """

        evidence = _history_evidence()
        # 混合检索召回：既非结构化查询，也非强语义水合（无分数门槛可通过）。
        evidence["match_type"] = "hybrid_search"
        evidence["retrieval_score"] = None
        evidence["rerank_score"] = None
        state = _state(evidence)
        state["intent"] = "history_query"
        state["query"] = "以前的相近案例在什么条件下出现、最后怎么验证恢复？"
        state["query_targets"] = [{"intent": "history_query"}]

        answer = "相似历史案例曾出现同类现象。[^1]"
        records, unsupported = answer_module._history_review_manifest(
            answer, state
        )

        self.assertEqual(1, len(records))
        self.assertEqual("HIS-SYN-020", records[0]["record_id"])
        self.assertFalse(unsupported)

    def test_referenced_hybrid_row_without_history_intent_still_admitted(self):
        """草稿引用即绑定：无显式历史意图时已引用记录同样进入清单。

        无历史意图时，历史行进入账本的合法通道是强语义水合（见
        ``_supplemental_history_evidence_ids`` 的既有门控与测试）。此处验证：
        已在账本中的强语义行被草稿引用后，manifest 不得再用 match_type/
        强匹配门控二次拒绝。弱混合检索行在纯维修意图下本来就进不了账本，
        由上文 context/ledger 过滤测试覆盖，不在此重复。
        """

        evidence = _history_evidence()
        evidence["match_type"] = "semantic_id_hydration"
        evidence["rerank_score"] = answer_module.HISTORY_SUPPLEMENT_MIN_SCORE
        evidence["retrieval_score"] = None
        state = _state(evidence)
        state["intent"] = "maintenance_query"

        answer = "历史案例显示曾发生同类故障。[^1]"
        ledger = answer_module._citation_ledger(state)
        self.assertEqual(
            {"HIS-SYN-020"},
            {item["record_id"] for item in ledger.values()},
        )
        records, _unsupported = answer_module._history_review_manifest(
            answer, state
        )

        self.assertEqual(1, len(records))
        self.assertEqual("HIS-SYN-020", records[0]["record_id"])


if __name__ == "__main__":
    unittest.main()
