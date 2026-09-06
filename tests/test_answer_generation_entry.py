"""One generator consumes the current task; downstream review is a separate boundary."""
import asyncio
import copy
import unittest
from unittest.mock import AsyncMock, patch

from backend.agent.nodes import answer


MODES = ("fault_diagnosis", "architecture_knowledge", "exact_code_lookup",
         "knowledge_explanation", "maintenance_guidance")


class AnswerGenerationEntryTests(unittest.TestCase):
    def state(self):
        return {"query": "不是刚才那个对象，改看 Z-83：什么条件会推翻原解释？",
            "system_prompt": "项目角色设置",
            "dialogue_context": {"status": "resolved", "subjects": ["Z-83"],
                "contextualized_query": "Z-83 的解释在哪些条件下不成立？",
                "requested_facets": ["识别反证条件及其依据"]},
            "messages": [{"role": "user", "content": "先看 Z-82 的历史案例。"},
                         {"role": "assistant", "content": "Z-82 在复测后恢复。[^19]"}],
            "query_frame": {}}

    def test_all_modes_share_the_identical_task_prompt(self):
        state = self.state()
        expected = answer._build_answer_prompt(state, "同一组获准证据")
        for mode in MODES:
            changed = copy.deepcopy(state)
            changed.update(answer_mode=mode, intent=mode, rewritten_query="改写器选了另一故障",
                           task_profile={"goals": ["重列旧记录"]})
            changed["query_frame"].update(required_answer_fields=["fault_cause"],
                response_focus=["cause"], prior_delivered_fields=["local_impact"])
            self.assertEqual(expected, answer._build_answer_prompt(changed, "同一组获准证据"))

    def test_code_and_maintenance_questions_receive_history_with_namespaced_citations(self):
        for mode in MODES:
            state = {**self.state(), "answer_mode": mode}
            text = answer._build_answer_prompt(state, "新的引用空间")
            self.assertIn("Z-82 在复测后恢复。", text)
            self.assertIn("历史引用 h1:19", text)
            self.assertNotIn("[^19]", text)
            self.assertIn(state["query"], text)
            self.assertIn("Z-83", text)

    def test_history_after_old_six_hundred_character_boundary_is_not_lost(self):
        state = self.state()
        state["messages"][1]["content"] = "已有说明。" * 150 + "随后用户否定了该对象。"
        self.assertIn("随后用户否定了该对象。", answer._build_answer_prompt(state, "证据"))

    def test_truncated_history_remains_explicitly_incomplete(self):
        state = self.state()
        state["messages"][1]["content"] = "内容" * 4000 + "不可据此判断遗漏"
        text = answer._build_answer_prompt(state, "证据")
        self.assertIn('"truncated": true', text)
        self.assertNotIn("不可据此判断遗漏", text)

    def test_arbitrary_question_dimensions_do_not_need_a_generation_vocabulary(self):
        for query in ("What observation would reverse that inference?", "何时只能支持相关而不能支持因果？",
                      "暂时失效与长期失效对结论有什么不同？", "之前的反例还成立吗？"):
            state = self.state()
            state["query"] = query
            state["dialogue_context"]["requested_facets"] = [query]
            self.assertIn(query, answer._build_answer_prompt(state, "证据"))

    def test_missing_or_ambiguous_subject_is_not_selected_by_generator(self):
        for status in ("missing", "ambiguous"):
            state = self.state()
            state["query"] = "它是怎么回事？"
            state["dialogue_context"]["status"] = status
            text = answer._build_answer_prompt(state, "证据")
            self.assertNotIn("Z-83", text)
            self.assertIn("它是怎么回事？", text)

    def test_raw_candidates_and_confidence_are_not_a_second_fact_source(self):
        state = self.state()
        expected = answer._build_answer_prompt(state, "只读证据包")
        state.update(candidate_faults=[{"fault": "未批准候选", "score": .99}], diagnostic_confidence=.99)
        self.assertEqual(expected, answer._build_answer_prompt(state, "只读证据包"))

    def test_source_restrictions_and_request_local_evidence_are_not_discarded(self):
        state = self.state()
        state["query_frame"]["source_obligations"] = [{"doc_type": "rbd", "fields": ["rbd_structure"]}]
        before = copy.deepcopy(state)
        text = answer._build_answer_prompt(state, "精确记录 Z-83，仅有受限字段视图；provisional")
        self.assertIn('"doc_type":"rbd"', text)
        self.assertIn("仅有受限字段视图", text)
        self.assertIn("provisional", text)
        self.assertIn("确定性计算", text)
        self.assertEqual(before, state)

    def test_actual_node_calls_the_single_generator_for_every_mode(self):
        for mode in MODES:
            state = {**self.state(), "answer_mode": mode}
            with patch.object(answer, "_with_answer_evidence_rerank", new=AsyncMock(side_effect=lambda x: x)), \
                 patch.object(answer, "_format_context", return_value="证据包"), \
                 patch.object(answer, "_build_answer_prompt", wraps=answer._build_answer_prompt) as build, \
                 patch.object(answer, "_generate_answer_response", new=AsyncMock(side_effect=RuntimeError("probe"))) as generate, \
                 patch.object(answer.agent_model_role_service, "apply_agent_llm_params", return_value=None):
                with self.assertRaises(answer.AnswerGenerationError):
                    asyncio.run(answer.answer_node(state))
            build.assert_called_once()
            generate.assert_awaited_once()
            self.assertEqual(answer._build_answer_prompt(state, "证据包"), generate.call_args.args[0])

    def test_retired_templates_and_empty_renderer_have_no_compatibility_wrappers(self):
        for name in ("EXACT_HMC_ANSWER_PROMPT", "ARCHITECTURE_ANSWER_PROMPT", "KNOWLEDGE_ANSWER_PROMPT",
                     "MAINTENANCE_ANSWER_PROMPT", "_format_candidates", "_closed_collection_identity_only",
                     "_resolved_architecture_collection", "_render_closed_collection_identity_only"):
            self.assertFalse(hasattr(answer, name), name)


if __name__ == "__main__":
    unittest.main()
