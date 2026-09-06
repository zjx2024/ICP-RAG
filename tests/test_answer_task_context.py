"""Open current-turn goals have one owner; evidence fields are not a vocabulary."""
import asyncio
import copy
import json
import unittest
from unittest.mock import patch

from backend.agent.nodes import answer, plan
from backend.agent.nodes.answer_request import _answer_task_context
from backend.agent.tools.icp_diagnosis import _rbd_comparison_requested


class AnswerTaskContextTests(unittest.TestCase):
    def state(self, question="那它短时不可用会先影响什么？"):
        return {
            "query": question,
            "dialogue_context": {
                "contextualized_query": "模块 Ω 短时不可用会先影响什么？",
                "requested_facets": ["区分短时中断和长期不可用的任务后果"],
                "subjects": ["模块 Ω"], "context_used": True, "status": "resolved",
            },
            "query_frame": {},
        }

    def test_raw_question_survives_a_lossy_resolution(self):
        state = self.state("不是启动阶段；运行中暂时不可用，先影响哪个环节、有什么例外？")
        text = answer._answer_task_instruction(state)
        self.assertIn(state["query"], text)
        self.assertIn(state["dialogue_context"]["contextualized_query"], text)

    def test_arbitrary_goals_do_not_require_focus_or_field_dictionary_entries(self):
        for question in ("能否区分观测误差和真实退化？", "Which assumption would reverse that conclusion?",
                         "先前的论证在哪里依赖时间窗口？", "比较恢复前后的证据强度，不要重列现象。"):
            state = self.state(question)
            state["dialogue_context"]["requested_facets"] = [question]
            self.assertEqual([question], _answer_task_context(state)["information_goals"])
            self.assertIn(question, answer._answer_task_instruction(state))

    def test_intents_fields_and_previous_delivery_cannot_change_the_task(self):
        state = self.state()
        original = answer._answer_task_instruction(state)
        for intent in ("entity_profile", "maintenance_query", "reliability_query"):
            changed = copy.deepcopy(state)
            changed.update(intent=intent, rewritten_query="另一部件的原因和历史", task_profile={"goals": ["其他任务"]})
            changed["query_frame"].update(required_answer_fields=["fault_cause", "history_root"],
                response_focus=["cause"], question_semantics="describe", prior_delivered_fields=["final_impact"])
            self.assertEqual(original, answer._answer_task_instruction(changed))

    def test_unresolved_context_does_not_pick_a_subject_or_old_goal(self):
        for status in ("ambiguous", "missing"):
            state = self.state()
            state["dialogue_context"]["status"] = status
            task = _answer_task_context(state)
            self.assertEqual(state["query"], task["resolved_question"])
            self.assertEqual([], task["subjects"])
            self.assertEqual([], task["information_goals"])

    def test_standalone_topic_change_does_not_consume_retrieval_rewrite(self):
        state = {"query": "换个问题，传感链路有几个接口？", "rewritten_query": "上个部件的故障根因"}
        self.assertEqual(state["query"], _answer_task_context(state)["resolved_question"])
        self.assertNotIn(state["rewritten_query"], answer._answer_task_instruction(state))

    def test_generation_and_review_share_task_and_source_boundaries_without_mutation(self):
        state = self.state()
        state["query_frame"]["source_obligations"] = [{"tool": "mfl_query", "doc_type": "mfl", "fields": ["maintenance_api"]}]
        state["hmc_mfl_binding"] = {"required": True, "bound": False, "hmc_codes": ["0001**0007"]}
        before = copy.deepcopy(state)
        instruction = answer._answer_task_instruction(state)
        self.assertIn(instruction, answer._build_answer_prompt(state, "证据"))
        self.assertIn('"doc_type":"mfl"', instruction)
        self.assertIn("0001**0007", instruction)
        self.assertIn("不能据此认定MFL没有记录或没有API", instruction)
        self.assertEqual(before, state)

    def test_trace_summary_cannot_authorize_rbd_comparison(self):
        self.assertFalse(_rbd_comparison_requested({"task_profile": {"question_semantics": "compare"}}))
        self.assertTrue(_rbd_comparison_requested({"query_frame": {"question_semantics": "compare"}}))

    def test_retired_prompt_owners_are_not_available_in_production(self):
        for name in ("_knowledge_focus", "_followup_convergence_instruction", "_source_coverage_preflight_instruction",
                     "_target_preflight_instruction", "_fault_preflight_instruction", "_answer_contract_instruction"):
            self.assertFalse(hasattr(answer, name), name)

    def test_open_goals_activate_review_without_legacy_focus_or_delivery(self):
        state = self.state("什么观测会推翻刚才的解释？")
        draft = "该解释成立。"
        from backend.agent.answer_review import AnswerReviewError
        with patch.object(answer.llm_service, "call_model", return_value="{}") as call:
            with self.assertRaises(AnswerReviewError):
                asyncio.run(answer._review_answer(draft, state, None))
        call.assert_called_once()
        self.assertIn(state["query"], call.call_args.args[0][0]["content"])

    def test_plan_preserves_open_question_despite_retrieval_field_drift(self):
        question = "它的论证在哪些条件下不再成立？"
        resolved = "记录 Z-91 的论证在哪些条件下不再成立？"
        decision = plan._parse_target_response(json.dumps({
            "primary_intent": "entity_profile", "targets": ["entity_profile"],
            "contextualized_query": resolved, "subjects": ["Z-91"],
            "requested_facets": ["识别论证成立所依赖的假设及失效条件"],
            "context_used": True, "context_status": "resolved",
        }), raw_user_query=question)
        retrieval = json.dumps({"intent": "cause_analysis", "rewritten_query": "另一故障原因",
            "query_frame": {"required_answer_fields": ["fault_cause"], "requested_facets": ["重列全部原因"]},
            "tools_plan": [], "tool_queries": {}})
        initial = {"query": question, "messages": [], "memory_enabled": False, "iteration": 0}
        with patch.object(plan, "_classify_explicit_targets", return_value=decision), patch.object(
            plan, "_generate_plan_response", return_value=retrieval), patch.object(
            plan.agent_model_role_service, "apply_agent_llm_params", return_value=None):
            result = asyncio.run(plan.plan_node(initial))
        task = _answer_task_context({**initial, **result})
        self.assertEqual(question, task["original_question"])
        self.assertEqual(resolved, task["resolved_question"])
        self.assertEqual(decision["requested_facets"], task["information_goals"])
        self.assertEqual([], result["calculation_requests"])


if __name__ == "__main__":
    unittest.main()
