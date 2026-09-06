"""Task ownership is invariant under wording, unrelated metadata and row order."""
import asyncio
import copy
import json
import random
import unittest
from unittest.mock import patch

from backend.agent.nodes import adaptive_tools, plan
from backend.agent.query_understanding import build_hybrid_query
from backend.agent.query_understanding import normalize_search_term
from backend.agent.task_context import task_query


class TaskOwnershipTests(unittest.TestCase):
    def state(self):
        identity = "unit-" + str(random.Random(720).randrange(10000, 99999))
        return {"query": "那怎样检验这项判断？", "dialogue_context": {
            "status": "resolved", "context_used": True, "subjects": [identity],
            "contextualized_query": f"怎样检验 {identity} 在受热后失联的判断？",
            "requested_facets": ["区分关联与因果所需的观测"],
        }}

    def test_relevance_ignores_planner_drift_and_legacy_delivery(self):
        state = self.state()
        before = task_query(state)
        state.update(rewritten_query="另一部件的故障", task_profile={"goals": ["改题"]},
                     query_frame={"entities": ["错误对象"], "prior_delivered_fields": ["所有原因"],
                                  "prior_delivery_contracts": [{"goal": "无关历史"}]})
        self.assertEqual(before, task_query(state))
        self.assertIn(state["dialogue_context"]["subjects"][0], before)
        self.assertNotIn("prior_delivered_fields", before)

    def test_unknown_or_ambiguous_subject_does_not_enter_relevance_query(self):
        for status in ("missing", "ambiguous"):
            state = self.state()
            state["dialogue_context"]["status"] = status
            self.assertEqual(state["query"], task_query(state))

    def test_arbitrary_goals_survive_without_field_names(self):
        for goal in ("What would falsify that?", "需要怎样的对照才能排除偶然？", "只比较两个时间窗口"):
            state = self.state()
            state["dialogue_context"]["requested_facets"] = [goal]
            self.assertIn(goal, task_query(state))

    def test_retrieval_writer_cannot_replace_empty_semantic_slots(self):
        state = self.state()
        dialogue = {**state["dialogue_context"], "required_answer_fields": []}
        target = {"primary_intent": "entity_profile", "query_targets": [{"intent": "entity_profile"}]}
        raw = {"query_frame": {"entities": ["other"], "requested_facets": ["其他任务"],
                              "required_answer_fields": ["fault_cause"],
                              "coverage_scope": "complete_collection"},
               "entity_keywords": ["other"], "calculation_authorizations": [{"mode": "calculate"}]}
        before = copy.deepcopy((raw, target, dialogue))
        bound = plan._bind_retrieval_plan(raw, target, dialogue, state["query"])
        self.assertEqual([], bound["query_frame"]["required_answer_fields"])
        self.assertEqual(dialogue["subjects"], bound["query_frame"]["entities"])
        self.assertEqual(dialogue["requested_facets"], bound["query_frame"]["requested_facets"])
        self.assertEqual([], bound["calculation_authorizations"])
        self.assertEqual(before, (raw, target, dialogue))

    def test_missing_target_never_promotes_retrieval_authorizations(self):
        result = plan._bind_retrieval_plan(
            {"intent": "exact_code", "calculation_authorizations": [{"mode": "calculate"}]},
            None, {}, "需要什么证据？")
        self.assertEqual("hybrid", result["intent"])
        self.assertEqual([], result["calculation_authorizations"])
        self.assertEqual({}, result["query_frame"])

    def test_hybrid_query_keeps_current_object_even_with_a_bad_hint(self):
        state = self.state()
        state["tool_queries"] = {"hybrid_search": ["通用维修历史"]}
        query = build_hybrid_query(state)
        self.assertTrue(
            normalize_search_term(state["dialogue_context"]["contextualized_query"])
            in normalize_search_term(query)
        )
        self.assertIn("受热后失联", query)

    def test_planner_sees_prior_unresolved_goal_without_treating_it_as_evidence(self):
        messages = [{"role": "assistant", "content": "先前结论[^3]" + "甲" * 700,
                     "answer_delivery_manifest": [{"kind": "goal_delivery", "goal": "验证恢复",
                                                  "status": "unanswered", "evidence_ids": []}]}]
        rendered = plan._format_history(messages)
        self.assertIn("unanswered", rendered)
        self.assertIn("历史引用 h0:3", rendered)
        self.assertNotIn("[^3]", rendered)
        self.assertGreater(len(json.loads(rendered)[0]["content"]), 500)

    def test_gap_summary_does_not_hide_history_behind_many_hmc_claims(self):
        evidence = [{"evidence_id": f"h{i}", "doc_type": "hmc", "doc_id": "d",
                     "record_id": "001**009", "source_kind": "postgres",
                     "metadata": {"answer_fields": {"fault_description": "其他现象"}}}
                    for i in range(20)]
        evidence.append({"evidence_id": "case", "doc_type": "fault_history", "doc_id": "history-doc",
                         "record_id": "CASE-987", "display_summary": "低温复现，复测后恢复"})
        for rows in (evidence, list(reversed(evidence))):
            summary = adaptive_tools._result_summary({"evidence_registry": rows}, [])
            self.assertIn("CASE-987", summary)
            self.assertIn("低温复现", summary)

    def test_adaptive_query_keeps_task_and_cannot_grant_an_unavailable_tool(self):
        state = self.state()
        state.update(authorized_tools=["hybrid_search"], allowed_tools=["hybrid_search"],
                     tool_queries={}, tool_results=[], evidence_registry=[])
        before = copy.deepcopy(state)
        response = json.dumps({"decision": "retrieve", "next_tools": ["hybrid_search", "rbd_calculator"],
                               "tool_queries": {"hybrid_search": ["需要验证恢复的相关记录"],
                                                "rbd_calculator": ["请计算"]}, "reason": "缺少验证证据"})
        with patch.object(adaptive_tools, "_model_for_state", return_value=None), patch.object(
            adaptive_tools.llm_service, "call_model", return_value=response) as call:
            result = asyncio.run(adaptive_tools.adaptive_tool_route_node(state))
        prompt = call.call_args.args[0][0]["content"]
        self.assertIn(state["dialogue_context"]["contextualized_query"], prompt)
        self.assertEqual(["hybrid_search"], result["pending_tools"])
        self.assertEqual(task_query(state), result["tool_queries"]["hybrid_search"][0])
        self.assertNotIn("rbd_calculator", result["tool_queries"])
        self.assertNotIn("dialogue_context", result)
        self.assertNotIn("query_frame", result)
        self.assertEqual(before, state)


if __name__ == "__main__":
    unittest.main()
