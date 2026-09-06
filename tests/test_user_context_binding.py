import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch

from backend.agent import user_context_binding as binding
from backend.agent.nodes import plan
from backend.agent.nodes.evidence_gate import _hmc_request_match_type
from backend.agent.nodes.answer_request import _current_user_conditions
from backend.agent.tools import icp_diagnosis
from backend.agent.tool_policy import resolve_tool_policy


class UserContextBindingTests(unittest.TestCase):
    def state(self, query="继续查还缺什么原始信息？"):
        return {"query": query, "messages": [
            {"role": "user", "content": "LCN 98765432、HSC 0017，请核查。"},
            {"role": "assistant", "content": "未找到匹配记录；不能给出根因。"},
        ], "dialogue_context": {"context_used": True, "status": "resolved"},
            "query_frame": {"user_input_query": query, "user_context_bindings": [
                {"message_id": "h0", "quote": "LCN 98765432、HSC 0017"}]}}

    def test_no_citation_does_not_erase_selected_user_code(self):
        state = self.state()
        self.assertEqual({"lcn": "98765432", "hsc": "0017"}, icp_diagnosis._typed_hmc_components(state))
        self.assertNotIn("conversation_anchors", state)
        self.assertNotIn("calculation_authorizations", state)
        self.assertEqual("exact_components", _hmc_request_match_type(state))

    def test_selected_input_reaches_actual_sql_without_keyword_fallback(self):
        state = {**self.state(), "allowed_tools": ["hmc_query"]}
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.description = [("hmc_code",)]
        cursor.fetchall.return_value = []
        with patch.object(icp_diagnosis.postgres_client, "conn", connection):
            result = asyncio.run(icp_diagnosis.icp_diagnosis_node(state))
        hmc_calls = [call.args for call in cursor.execute.call_args_list if "fault_hmc" in call.args[0]]
        self.assertTrue(hmc_calls)
        for sql, params in hmc_calls:
            self.assertIn("h.lcn=%s AND h.hsc=%s", sql)
            self.assertNotIn("ILIKE", sql)
            self.assertEqual(("98765432", "0017"), params[:2])
        calls = [call for call in result["tool_calls"] if call["tool"] == "hmc_query"]
        self.assertEqual("exact_components", calls[0]["params"]["match_type"])
        self.assertEqual("继续查还缺什么原始信息？", state["query"])

    def test_plan_to_sql_preserves_selected_input_even_without_a_proposed_hmc_tool(self):
        for query in ("还差些什么才能继续？", "Which inputs should I collect next?", "这件事下一步需要哪些资料？"):
            with self.subTest(query=query):
                state = self.state(query)
                target = {"primary_intent": "maintenance_query", "targets": ["maintenance_query"],
                    "context_used": True, "context_status": "resolved", "contextualized_query": query,
                    "user_context_bindings": state["query_frame"]["user_context_bindings"]}
                planner = {"intent": "maintenance_query", "query_targets": [{"intent": "maintenance_query"}],
                    "rewritten_query": query, "tools_plan": [{"tool": "hybrid_search"}]}
                with patch.object(plan.llm_service, "call_model", side_effect=[json.dumps(target), json.dumps(planner)]), patch.object(
                    plan.agent_model_role_service, "apply_agent_llm_params", return_value=None):
                    planned = asyncio.run(plan.plan_node({**state, "iteration": 0, "memory_enabled": False}))
                self.assertIn("hmc_query", planned["allowed_tools"])
                self.assertNotIn("rbd_calculator", planned["authorized_tools"])
                connection = MagicMock()
                cursor = connection.cursor.return_value.__enter__.return_value
                cursor.description = [("hmc_code",)]
                cursor.fetchall.return_value = []
                with patch.object(icp_diagnosis.postgres_client, "conn", connection), patch.object(icp_diagnosis, "_hybrid_search", return_value=([], {})):
                    result = asyncio.run(icp_diagnosis.icp_diagnosis_node({**state, **planned}))
                queries = [c.args for c in cursor.execute.call_args_list if "fault_hmc" in c.args[0]]
                self.assertTrue(queries)
                for sql, params in queries:
                    self.assertIn("h.lcn=%s AND h.hsc=%s", sql)
                    self.assertNotIn("ILIKE", sql)
                    self.assertEqual(("98765432", "0017"), params[:2])
                self.assertTrue(any(c["tool"] == "hmc_query" for c in result["tool_calls"]))

    def test_policy_rereads_provenance_and_never_accepts_a_bare_identity_mapping(self):
        state = self.state()
        for invalid in ({"query": state["query"], "hmc_components": {"lcn": "98765432", "hsc": "0017"}},
                        {**state, "messages": [{"role": "assistant", "content": "LCN 98765432、HSC 0017"}]}):
            result = resolve_tool_policy(intent="hybrid", raw_user_query=state["query"], user_context_state=invalid)
            self.assertNotIn("hmc_query", result["allowed_tools"])

    def test_natural_followup_selection_is_model_owned_not_a_question_wordlist(self):
        for query in ("还差些什么才能继续？", "Which original inputs should I collect?", "这件事下一步需要哪些资料？"):
            state = self.state(query)
            response = json.dumps({"primary_intent": "exact_code", "targets": ["exact_code"],
                "context_used": True, "context_status": "resolved",
                "user_context_bindings": [{"message_id": "h0", "quote": "LCN 98765432、HSC 0017"}]})
            with self.subTest(query=query), patch.object(plan.llm_service, "call_model", return_value=response) as call:
                decision = asyncio.run(plan._classify_explicit_targets(query=query, history="", conversation_anchors=[],
                    model=None, user_context=binding.offered_user_inputs(state)))
            self.assertEqual("unverified_user_input", decision["user_context_bindings"][0]["authority"])
            self.assertEqual([], decision["calculation_authorizations"])
            self.assertIn(query, call.call_args.args[0][0]["content"])

    def test_assistant_claim_or_fabricated_quote_cannot_mint_user_input(self):
        offered = binding.offered_user_inputs(self.state())
        for proposal in ([{"message_id": "h1", "quote": "未找到匹配记录"}], [{"message_id": "h0", "quote": "LCN 11111111、HSC 0017"}]):
            self.assertEqual([], binding.validate_user_context_bindings(proposal, offered))

    def test_new_identity_partial_correction_or_unrelated_turn_cannot_reuse_old(self):
        for query in ("LCN 12345678、HSC 0022", "HSC 改为0022", "改查 FUNC-999"):
            self.assertEqual({}, binding.selected_user_hmc_components(self.state(query)))
        state = self.state()
        state["dialogue_context"]["context_used"] = False
        self.assertEqual({}, binding.selected_user_hmc_components(state))
        state = self.state()
        state["query"] = "一个新问题"
        self.assertEqual({}, binding.selected_user_hmc_components(state))

    def test_conflicting_selected_codes_do_not_choose_last_or_first(self):
        state = self.state()
        state["messages"].append({"role": "user", "content": "LCN 12345678、HSC 0033"})
        state["query_frame"]["user_context_bindings"].append({"message_id": "h2", "quote": "LCN 12345678、HSC 0033"})
        self.assertEqual({}, binding.selected_user_hmc_components(state))

    def test_consumer_rereads_actual_history_instead_of_trusting_frozen_binding(self):
        state = self.state()
        state["messages"][0]["content"] = "更正：那份日志不是本设备的。"
        self.assertEqual({}, binding.selected_user_hmc_components(state))

    def test_general_user_inputs_keep_literal_provenance_without_becoming_ids(self):
        state = self.state()
        state["messages"][0]["content"] = "只在低温启动时复现，持续两分钟。"
        state["query_frame"]["user_context_bindings"] = [{"message_id": "h0", "quote": "低温启动时复现，持续两分钟"}]
        self.assertEqual(["低温启动时复现，持续两分钟"], binding.selected_user_input_quotes(state))
        self.assertEqual({}, binding.selected_user_hmc_components(state))

    def test_answer_conditions_require_actual_user_provenance(self):
        state = self.state("高负载时复现")
        state["query_frame"].update(operating_stage="高负载", operating_stage_basis="高负载时")
        self.assertEqual("高负载", _current_user_conditions(state)["operating_stage"])
        state["query_frame"]["operating_stage_basis"] = "系统启动时"
        self.assertEqual("未知", _current_user_conditions(state)["operating_stage"])
        state["query_frame"]["operating_stage_basis"] = ""
        self.assertEqual("未知", _current_user_conditions(state)["operating_stage"])


if __name__ == "__main__":
    unittest.main()
