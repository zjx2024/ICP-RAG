# v3/v4 validator assertions below are offline compatibility; live v5 counterparts are in test_answer_integrated_review.
import copy
import importlib
import json
import unittest
from unittest.mock import Mock, patch

from backend.agent import model_runtime_policy as budget
from backend.agent.nodes.answer_domain_scope import _filter_answer_domain_evidence
from backend.agent.tool_policy import normalize_requested_sources
from backend.agent.user_context_binding import validate_user_context_bindings
service = importlib.import_module("backend.core.llm_service")


class NativeContextContractTests(unittest.TestCase):
    def test_no_hit_clarification_explains_query_result_not_source_wide_absence(self):
        from backend.agent.clarification_contract import make_clarification_reason
        from backend.agent.nodes.answer_request import render_clarification_only_answer
        reason = make_clarification_reason(kind="authoritative_record_not_found", source_kind="tool_result", source_name="hmc_query")
        answer = render_clarification_only_answer({"clarification_reasons": [reason]})
        self.assertIn("本轮权威查询未命中", answer)
        self.assertIn("不证明原始资料中不存在", answer)
        self.assertEqual("", render_clarification_only_answer({}))

    def test_unverified_provider_cannot_silently_ignore_no_truncation_flags(self):
        response = Mock()
        response.json.return_value = {"version": "0.1.0"}
        model = {"provider": "ollama", "base_url": "http://test.invalid", "num_ctx": 32768,
                 "max_tokens": 2048, "_icp_runtime": True}
        with patch("requests.get", return_value=response), self.assertRaises(budget.ModelBudgetError):
            budget.prepare_model_request(model, [{"role": "user", "content": "hello"}])

    def test_markdown_is_not_part_of_record_identity_and_wildcards_stay_exact(self):
        from scripts.legacy_answer_review import _check_permissions, AnswerReviewError
        from backend.agent.citation_binding import rendered_identity_text
        row = {"record_id": "CASE-782", "metadata": {"answer_fields": {
            "references": ["74210001**0081", "74210002**0082"]}}}
        ledger = {1: row}
        unit = {"text": "**CASE-782**：74210001\\*\\*0081 和 74210002**0082。",
                "claims": ["source_fact"], "bindings": [{"citation_id": 1}]}
        _check_permissions(unit, ledger, {}, [])
        self.assertIn("74210001**0081", rendered_identity_text(unit["text"]))
        self.assertIn("74210002**0082", rendered_identity_text(unit["text"]))
        for wrong in ("**CASE-783**", "74210001**0082", "742100010081",
                      "**74210001**0081", "74210001**0081 和 74210002**0082"):
            with self.subTest(wrong=wrong), self.assertRaises(AnswerReviewError):
                _check_permissions({**unit, "text": wrong}, ledger, {}, [])

    def test_published_citations_share_the_complete_review_ledger(self):
        from backend.agent.nodes import answer
        from tests.legacy_review_fixtures import state_for
        state = state_for()
        prototype = state["evidence_registry"][0]
        rows = [{**copy.deepcopy(prototype), "evidence_id": f"evidence-{n}", "record_id": f"ROW-{n}"}
                for n in range(40)]
        state["evidence_registry"] = rows
        state["context_evidence_ids"] = [r["evidence_id"] for r in rows]
        state["safety_decision"] = {"approved_evidence_ids": state["context_evidence_ids"]}
        ledger = answer._citation_ledger(state)
        citations = answer._extract_citations(state)
        self.assertEqual(set(ledger), {r["id"] for r in citations})
        text, used, _ = answer._compact_answer_citations("核验后的事实。[^40]", citations, state)
        self.assertEqual("核验后的事实。[^1]", text)
        self.assertEqual("evidence-39", used[0]["evidence_id"])

    def test_request_preserves_all_roles_and_delegates_token_capacity_to_provider(self):
        messages = [{"role": role, "content": role + "末尾纠正"} for role in ("system", "user", "assistant", "user")]
        original = copy.deepcopy(messages)
        payload = service._ollama_runtime_payload(messages, {
            "model_name": "model-under-test", "num_ctx": 32768, "max_tokens": 4096,
            "temperature": 0, "reasoning": False, "format": {"type": "object"},
        }, stream=True)
        self.assertEqual(original, payload["messages"])
        self.assertEqual(original, messages)
        self.assertIs(False, payload["truncate"])
        self.assertIs(False, payload["shift"])
        self.assertIs(False, payload["think"])
        self.assertEqual(4096, payload["options"]["num_predict"])
        self.assertEqual(32768, payload["options"]["num_ctx"])

    def test_provider_budget_error_is_distinct_and_partial_completion_is_not_success(self):
        with self.assertRaises(budget.ModelBudgetError):
            service._ollama_runtime_content({"error": "input length exceeds context length"})
        with self.assertRaises(service.LLMStreamError):
            service._ollama_runtime_content({"error": "server failed"})
        with self.assertRaises(service.LLMStreamError):
            service._ollama_runtime_content({"message": {"content": "half"}, "done": False}, final=True)

    def test_long_source_provenance_and_user_quotes_are_validated_without_prefix_cutting(self):
        quote = "conditions; " * 180 + "用户在末尾否定旧对象"
        requests = [{"doc_type": "mfl", "question_role": "evidence", "basis": quote,
                     "requested_fields": ["maintenance_action"], "scope": "exclusive", "scope_basis": quote}]
        normalized = normalize_requested_sources(requests, raw_user_query=quote)
        self.assertEqual(quote, normalized[0]["basis"])
        self.assertEqual(quote, normalized[0]["scope_basis"])
        offered = [{"message_id": f"h{i}", "role": "user", "content": quote} for i in range(10)]
        selected = [{"message_id": row["message_id"], "quote": quote} for row in offered]
        self.assertEqual(10, len(validate_user_context_bindings(selected, offered)))
        selected[0]["quote"] += "伪造原话"
        self.assertEqual([], validate_user_context_bindings(selected, offered))

    def test_legacy_compression_does_not_delete_source_messages_or_call_a_model(self):
        chat_router = importlib.import_module("backend.agent.api.chat_router")
        from fastapi import HTTPException
        with patch.object(chat_router.agent_db, "get_session", return_value={"id": "existing"}), \
                patch.object(chat_router.agent_db, "list_messages") as read, \
                patch.object(service.llm_service, "call_model") as call:
            with self.assertRaises(HTTPException) as caught:
                chat_router.compress_session("existing")
        self.assertEqual(409, caught.exception.status_code)
        read.assert_not_called()
        call.assert_not_called()


class NamedSourceViewTests(unittest.TestCase):
    def state(self):
        return {"query_frame": {"required_answer_fields": ["maintenance_action", "maintenance_api"],
            "source_obligations": [{"tool": "mfl_query", "doc_type": "mfl", "fields": ["maintenance_action", "maintenance_api"]}]}}

    def test_source_owned_goals_keep_named_sources_and_exact_reread_anchor_not_neighbors(self):
        state = self.state()
        state["selected_conversation_anchors"] = [{"doc_type": "hmc", "doc_id": "manual", "record_id": "CODE-832"}]
        rows = [{"evidence_id": "m", "doc_type": "mfl", "record_id": "ROW-41"},
                {"evidence_id": "a", "doc_type": "hmc", "doc_id": "manual", "record_id": "CODE-832"},
                {"evidence_id": "n", "doc_type": "hmc", "doc_id": "other", "record_id": "CODE-832"},
                {"evidence_id": "f", "doc_type": "fmeca", "record_id": "ROW-64"}]
        state["safety_decision"] = {"approved_evidence_ids": [r["evidence_id"] for r in rows]}
        self.assertEqual(["m", "a"], [r["evidence_id"] for r in _filter_answer_domain_evidence(state, rows)])

    def test_independent_goal_keeps_its_other_document_family(self):
        state = self.state()
        state["query_frame"]["required_answer_fields"].append("history_result")
        rows = [{"evidence_id": "m", "doc_type": "mfl"}, {"evidence_id": "h", "doc_type": "fault_history"}]
        self.assertEqual(rows, _filter_answer_domain_evidence(state, rows))


class NativeStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_transport_sends_flags_and_rejects_a_missing_terminal_chunk(self):
        import httpx
        client_type = httpx.AsyncClient
        model = {"_icp_runtime": True, "base_url": "http://model.invalid", "model_name": "sample",
                 "num_ctx": 32768, "max_tokens": 3072, "timeout": 10}
        client = service.LLMClient.__new__(service.LLMClient)
        for terminal in (True, False):
            captured = []
            def respond(request):
                captured.append(json.loads(request.content))
                chunks = [{"message": {"content": "draft"}, "done": False}]
                if terminal:
                    chunks.append({"message": {"content": ""}, "done": True, "done_reason": "stop"})
                return httpx.Response(200, content="\n".join(json.dumps(c) for c in chunks))
            with patch("httpx.AsyncClient", side_effect=lambda **kw: client_type(transport=httpx.MockTransport(respond), **kw)):
                stream = client._stream_ollama_with_config([{"role": "user", "content": "test"}], model)
                if terminal:
                    self.assertEqual(["draft"], [part async for part in stream])
                else:
                    with self.assertRaises(service.LLMStreamError):
                        _ = [part async for part in stream]
            self.assertIs(False, captured[0]["truncate"])
            self.assertIs(False, captured[0]["shift"])


if __name__ == "__main__":
    unittest.main()
