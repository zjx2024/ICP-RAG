# v3/v4 validator assertions below are offline compatibility; live v5 counterparts are in test_answer_integrated_review.
"""General regressions for candidate retention, provenance and real budgets."""
import asyncio
import copy
import json
import unittest
from unittest.mock import patch, Mock

from backend.agent import model_runtime_policy as policy
from scripts import legacy_answer_review as answer_review
from backend.agent.current_model_selection import model_content_evidence
from backend.agent.history_focus import select_history_parent_cases, history_parent_binding, resolve_history_parent_cases
from backend.agent.nodes import answer
from backend.core import llm_service as transport
from scripts.evaluate_icp_paraphrases import _materialize_judge_units, _validate_grounded_semantic_judgement, _judge_answer_units, _judge_quote_supported, ApiError
from tests.legacy_review_fixtures import state_for, response, check, validate, prepared
from tests import legacy_review_fixtures as review_fixtures


class CandidateRetentionTests(unittest.TestCase):
    def test_sql_count_cannot_freeze_case_before_later_semantic_read(self):
        state = {"query": "Which incident fits the reported conditions?"}
        first = {"case_id": "CASE-ALPHA", "match_type": "structured_query", "retrieval_match_score": 18}
        second = {"case_id": "CASE-BETA", "match_type": "semantic_id_hydration", "retrieval_match_score": .99}
        selection = select_history_parent_cases(state, [first])
        state["history_parent_binding"] = history_parent_binding(selection, [first])
        resolved = resolve_history_parent_cases(state, [first, second])
        self.assertEqual({"case-alpha", "case-beta"}, set(resolved.selected_case_ids))
        self.assertFalse(resolved.authorizes_exact_links)
        self.assertTrue(resolved.ambiguous)

    def test_all_approved_candidates_reach_writer_and_critic_not_only_gate_preview(self):
        state = state_for("fault_history")
        first = state["evidence_registry"][0]
        first.update(record_id="CASE-ALPHA", retrieval_score=18)
        second = copy.deepcopy(first)
        second.update(record_id="CASE-BETA", evidence_id="case-beta", retrieval_score=.01, answer_rerank_score=.999)
        state["evidence_registry"].append(second)
        state["safety_decision"]["approved_evidence_ids"].append("case-beta")
        before = copy.deepcopy(state)
        self.assertEqual({"CASE-ALPHA", "CASE-BETA"}, {r["record_id"] for r in answer._context_evidence(state)})
        material, ledger = prepared(state)
        self.assertEqual({"CASE-ALPHA", "CASE-BETA"}, {r["record_id"] for r in ledger.values()})
        self.assertIn("CASE-BETA", json.dumps(material, ensure_ascii=False))
        self.assertEqual(before, state)
        state["safety_decision"]["blocked_evidence_ids"] = ["case-beta"]
        self.assertNotIn("case-beta", {r["evidence_id"] for r in answer._context_evidence(state)})

    def test_delivery_is_per_goal_not_union_of_answer_sources(self):
        state = state_for()
        second = copy.deepcopy(state["evidence_registry"][0])
        second.update(evidence_id="other-row", record_id="ROW-832")
        state["evidence_registry"].append(second)
        state["safety_decision"]["approved_evidence_ids"].append("other-row")
        state["context_evidence_ids"].append("other-row")
        state["dialogue_context"] = {"requested_facets": ["first dimension", "second dimension"]}
        material, ledger = prepared(state, "第一段。\n第二段。")
        material["goals"] = {"q0": "all", "q1": "first", "q2": "second"}
        review = response(check(), check(unit_id="u2", sources=["e2:/answer_fields/local_impact"]))
        review["goals"] = [{"goal_id": gid, "status": "answered", "unit_ids": ids, "issue": ""}
                           for gid, ids in (("q0", ["u1", "u2"]), ("q1", ["u1"]), ("q2", ["u2"]))]
        result = answer_review.validate_review(json.dumps(review), prepared=material, ledger=ledger, safety={}, calculation_requests=[])
        self.assertEqual([second["evidence_id"]], result["manifest"][2]["evidence_ids"])
        self.assertNotIn(second["evidence_id"], result["manifest"][1]["evidence_ids"])
        review["goals"][1]["unit_ids"] = ["u999"]
        with self.assertRaises(answer_review.AnswerReviewError):
            answer_review.validate_review(json.dumps(review), prepared=material, ledger=ledger, safety={}, calculation_requests=[])

    def test_results_cannot_borrow_other_models_values_in_one_grouped_citation(self):
        state, value = review_fixtures.CalculationReviewTests().calculation()
        second = copy.deepcopy(state["evidence_registry"][0])
        second.update(record_id="ROW-832", evidence_id="other-calculation")
        second["metadata"]["calculation"]["reliability"] = .81234
        state["evidence_registry"].append(second)
        state["context_evidence_ids"].append(second["evidence_id"])
        state["safety_decision"]["approved_evidence_ids"].append(second["evidence_id"])
        value["units"][0]["sources"].append("e2:/calculation/reliability")
        with self.assertRaises(answer_review.AnswerReviewError) as caught:
            validate(value, state, draft="ROW-731 为 0.81234；ROW-832 为 0.91827。")
        self.assertIn("实际换行", caught.exception.feedback[0]["detail"])
        self.assertIn("不要删除用户所问", caught.exception.feedback[0]["detail"])
        value = response(
            check(sources=["e1:/calculation/reliability"], claims=["formal_numeric_result"]),
            check(unit_id="u2", sources=["e2:/calculation/reliability"], claims=["formal_numeric_result"]))
        self.assertEqual("full", validate(value, state, draft="ROW-731 为 0.91827。\nROW-832 为 0.81234。")["status"])

    def test_mislabeled_calculation_still_checks_literal_numeric_values(self):
        state, value = review_fixtures.CalculationReviewTests().calculation()
        for label in ("source_fact", "conditional_analysis", "formal_calculation"):
            value["units"][0]["claims"] = [label]
            with self.subTest(label=label), self.assertRaises(answer_review.AnswerReviewError):
                validate(value, state, draft="结果 0.123456。")
            self.assertEqual("full", validate(value, state, draft="结果 0.91827。")["status"])

    def test_model_source_expression_does_not_grant_part_applicability(self):
        row = {"doc_id": "doc-82", "revision_id": "rev-74", "title": "Parent model",
               "is_active": True, "projection_status": "published", "validation_status": "auto_validated",
               "graph": {"metadata": {"structure_expression": "AND(X,OR(Y,Z))", "success_criterion": "X and either branch"}}}
        source = model_content_evidence([row], "rbd")[0]
        self.assertEqual("AND(X,OR(Y,Z))", source["metadata"]["answer_fields"]["model_expression"])
        state = state_for()
        state.update(evidence_registry=[source], context_evidence_ids=[source["evidence_id"]],
                     safety_decision={"approved_evidence_ids": [source["evidence_id"]]})
        value = response(check(sources=["e1:/answer_fields/model_expression"]))
        self.assertEqual("full", validate(value, state, draft="模型原表达式为 AND(X,OR(Y,Z))。")["status"])
        value["units"][0]["claims"] = ["formal_topology"]
        with self.assertRaises(answer_review.AnswerReviewError):
            validate(value, state, draft="该细部部件直接适用 AND(X,OR(Y,Z))。")


class SharedBudgetTests(unittest.TestCase):
    def test_output_override_is_provider_neutral_and_never_mutates_model(self):
        original = {"provider": "ollama", "max_tokens": 3072, "num_predict": 99, "timeout": 3000}
        with policy.turn_budget(20):
            result = policy.bounded_model_transport(original)
        self.assertEqual(3072, result["num_predict"])
        self.assertEqual(3072, result["max_tokens"])
        self.assertLessEqual(result["timeout"], 20)
        self.assertEqual(99, original["num_predict"])

    def test_one_deadline_is_inherited_without_nested_reset(self):
        with patch.object(policy.time, "monotonic", return_value=100):
            with policy.turn_budget(40):
                with patch.object(policy.time, "monotonic", return_value=130), policy.turn_budget(200):
                    self.assertEqual(10, policy.remaining_seconds())
                    self.assertEqual(10, asyncio.run(asyncio.to_thread(policy.remaining_seconds)))
                with patch.object(policy.time, "monotonic", return_value=141):
                    with self.assertRaises(TimeoutError):
                        policy.remaining_seconds()

    def test_long_complete_prompt_uses_capacity_not_old_character_rules(self):
        messages = [{"role": "user", "content": "事实" * 9000 + "末尾纠正"}]
        original = copy.deepcopy(messages)
        model = {"provider": "ollama", "base_url": "http://test.invalid", "num_ctx": 262144, "max_tokens": 4096, "_icp_runtime": True}
        observed = Mock()
        observed.json.return_value = {"version": "0.33.3"}
        with patch("requests.get", return_value=observed):
            result = policy.prepare_model_request(model, messages)
            # Bytes are not tokens. The native provider rejects actual overflow
            # with truncate/shift disabled; a byte estimate cannot prove it.
            self.assertEqual(8192, policy.prepare_model_request({**model, "num_ctx": 8192}, messages)["num_ctx"])
            with self.assertRaises(policy.ModelBudgetError):
                policy.prepare_model_request({**model, "num_ctx": 4096}, messages)
        self.assertEqual(262144, result["num_ctx"])
        self.assertEqual("provider_tokenizer_no_truncation", result["_runtime_budget"]["context_enforcement"])
        self.assertIsNone(result["_runtime_budget"]["input_tokens"])
        self.assertEqual(original, messages)

    def test_ollama_observed_capacity_is_cached_only_for_this_turn(self):
        observed = Mock()
        observed.json.return_value = {"version": "0.33.3", "models": [{"name": "test-model", "context_length": 32768}]}
        model = {"provider": "ollama", "base_url": "http://test.invalid", "model_name": "test-model", "max_tokens": 2048, "_icp_runtime": True}
        with patch("requests.get", return_value=observed) as get:
            with policy.turn_budget():
                for _ in range(2):
                    self.assertEqual(32768, policy.prepare_model_request(model, [])["num_ctx"])
            self.assertEqual(2, get.call_count)
            with policy.turn_budget():
                policy.prepare_model_request(model, [])
            self.assertEqual(4, get.call_count)

    def test_planner_preserves_late_correction_and_all_natural_language_dimensions(self):
        from backend.agent.nodes import plan, scope_router
        messages = [{"role": "user", "content": "earliest correction"}] + [
            {"role": "assistant", "content": "later"} for _ in range(20)]
        self.assertIn("earliest correction", plan._format_history(messages))
        self.assertIn("earliest correction", scope_router._format_history(messages))
        long_history = [{"role": "user", "content": "x" * 800 + "末尾否定原对象"}]
        self.assertIn("末尾否定原对象", scope_router._format_history(long_history))
        dimensions = ["dimension " + str(n) + "x" * 700 for n in range(20)]
        self.assertEqual(dimensions, plan._bounded_text_list(dimensions))
        basis = "x" * 600 + " not applicable"
        self.assertEqual(basis, plan._literal_current_span(basis, basis))

    def test_actual_usage_is_traced_and_length_stop_cannot_publish_as_success(self):
        client = object.__new__(transport.LLMClient)
        def output(*args):
            transport._capture_usage({"prompt_eval_count": 123, "eval_count": 2048, "done_reason": "length"})
            return "模型留下的半段"
        with patch.object(client, "_call_model", side_effect=output), patch("backend.agent.tracing.record_llm") as trace:
            with self.assertRaises(transport.LLMStreamError):
                client.call_model([], {})
        self.assertFalse(trace.call_args.kwargs["ok"])
        self.assertEqual({"input_tokens": 123, "output_tokens": 2048}, trace.call_args.kwargs["token_usage"])
        self.assertEqual("length", trace.call_args.kwargs["finish_reason"])


class JudgeUnitProvenanceTests(unittest.TestCase):
    def test_units_preserve_r_n_and_multiline_list_boundaries(self):
        answer_text = "run normal restart。[^1]\n\n1. **条件**\n   - 35 min 后恢复。[^2]\n   - -20～85 ℃热循环。[^2]"
        units = _judge_answer_units(answer_text)
        self.assertEqual(4, len(units))
        self.assertEqual("run normal restart。[^1]", units["a1"])
        for quote in units.values():
            self.assertNotIn("\n", quote)
            self.assertTrue(_judge_quote_supported(quote, answer_text))
        tail = "后恢复。[^2]\n   - -20～85 ℃热循环。[^2]"
        self.assertTrue(_judge_quote_supported(tail, answer_text))
        self.assertFalse(_judge_quote_supported("已经排除所有其他根因。", answer_text))

    def test_program_quotes_exact_unit_without_reconstructing_the_answer(self):
        answer_text = "甲记录没有证明乙部件故障。[^2]"
        check = {"aspect": "boundary", "requirement_kind": "positive", "answer_unit_ids": ["a1"], "passed": True, "reason": "保持条件"}
        value = {"criteria": [{"id": "criterion", "passed": True, "reason": "边界已说明", "checks": [check]}]}
        materialized = _materialize_judge_units(value, {"a1": answer_text})
        self.assertEqual([answer_text], materialized["criteria"][0]["checks"][0]["answer_quotes"])
        result = _validate_grounded_semantic_judgement(materialized, [{"id": "criterion", "criterion": "说明边界"}], answer_text)
        self.assertTrue(result[0]["passed"])
        for bad in (["a999"], ["a1", "a1"], "a1"):
            check["answer_unit_ids"] = bad
            with self.assertRaises(ApiError):
                _materialize_judge_units(value, {"a1": answer_text})


if __name__ == "__main__":
    unittest.main()
