import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent import model_selection_service as service
from backend.core.llm_service import LLMClient


class ModelSelectionTransportTests(unittest.TestCase):
    def test_ollama_contract_disables_thinking_and_requests_json_without_mutation(self):
        original = {"provider": "ollama", "reasoning": True, "num_predict": 8192, "max_tokens": 8192}
        prepared = service._selection_model(original)
        self.assertFalse(prepared["reasoning"])
        self.assertEqual("json", prepared["format"])
        self.assertEqual(8192, prepared["num_predict"])
        self.assertEqual(8192, prepared["max_tokens"])
        self.assertTrue(original["reasoning"])
        self.assertEqual(8192, original["num_predict"])

    def test_transport_passes_a_traceable_complete_contract(self):
        state = {"query": "查这一条供能模型", "agent_id": "agent"}
        row = {"doc_id": "doc-x", "revision_id": "rev-x", "title": "供能模型"}
        proposal = {"status": "selected", "question_basis": "供能模型", "selections": [
            {"doc_id": "doc-x", "revision_id": "rev-x", "matched_title": "供能模型"},
        ]}
        with patch.object(service.agent_model_role_service, "get_model_by_role", return_value={"provider": "ollama"}), patch.object(
            service.agent_model_role_service, "apply_agent_llm_params", side_effect=lambda _state, model: model,
        ), patch.object(service.llm_service, "call_model", return_value=json.dumps(proposal)) as call:
            result = asyncio.run(service.select_current_models(state, [row], doc_type="rbd"))
        self.assertEqual("selected", result["status"])
        self.assertFalse(call.call_args.args[1]["reasoning"])
        self.assertEqual("current_model_selection", call.call_args.kwargs["template_name"])

    def test_empty_model_output_is_an_observable_contract_failure(self):
        with patch.object(service.agent_model_role_service, "get_model_by_role", return_value={"provider": "ollama"}), patch.object(
            service.agent_model_role_service, "apply_agent_llm_params", side_effect=lambda _state, model: model,
        ), patch.object(service.llm_service, "call_model", return_value=""), patch("backend.agent.tracing.record_degradation") as degradation:
            result = asyncio.run(service.select_current_models(
                {"query": "查当前模型", "agent_id": "a"},
                [{"doc_id": "d", "revision_id": "r", "title": "当前模型"}], doc_type="rbd",
            ))
        self.assertEqual({}, result)
        self.assertEqual("model_selection", degradation.call_args.args[0])

    def test_empty_and_raised_llm_calls_are_not_successful_traces(self):
        client = object.__new__(LLMClient)
        for output in ("", "   ", "[AI调用失败] unreachable"):
            with self.subTest(output=output), patch.object(client, "_call_model", return_value=output), patch("backend.agent.tracing.record_llm") as trace:
                self.assertEqual(output, client.call_model([], {}))
                self.assertFalse(trace.call_args.kwargs["ok"])
        with patch.object(client, "_call_model", side_effect=RuntimeError("offline")), patch("backend.agent.tracing.record_llm") as trace:
            with self.assertRaises(RuntimeError):
                client.call_model([], {})
            self.assertFalse(trace.call_args.kwargs["ok"])

    def test_invalid_contract_gets_one_correction_with_same_query_candidates_and_deadline(self):
        state = {"query": "继续查这个模型", "rewritten_query": "继续查供能模型", "agent_id": "agent"}
        row = {"doc_id": "doc-x", "revision_id": "rev-x", "title": "供能模型"}
        proposal = {"status": "selected", "question_basis": "供能模型", "selections": [
            {"doc_id": "doc-x", "revision_id": "rev-x", "matched_title": "供能模型"},
        ]}
        corrected = {**proposal, "question_basis": state["query"]}
        with patch.object(service.agent_model_role_service, "get_model_by_role", return_value={"provider": "ollama"}), patch.object(
            service.agent_model_role_service, "apply_agent_llm_params", side_effect=lambda _state, model: model,
        ), patch.object(service.llm_service, "call_model", side_effect=[json.dumps(proposal), json.dumps(corrected)]) as call, patch.object(
            service.asyncio, "wait_for", wraps=asyncio.wait_for,
        ) as bounded:
            result = asyncio.run(service.select_current_models(state, [row], doc_type="rbd"))
        self.assertEqual("selected", result["status"])
        self.assertEqual(2, call.call_count)
        first, second = [json.loads(item.args[0][-1]["content"]) for item in call.call_args_list]
        self.assertEqual(first["current_question"], second["current_question"])
        self.assertEqual(first["candidates"], second["candidates"])
        self.assertEqual("question_basis_not_literal_current_question", second["contract_correction"]["error"])
        self.assertLess(bounded.call_args_list[1].kwargs["timeout"], bounded.call_args_list[0].kwargs["timeout"])

    def test_repeated_unoffered_identity_is_never_repaired_by_code(self):
        proposal = {"status": "selected", "question_basis": "当前模型", "selections": [
            {"doc_id": "other-doc", "revision_id": "unknown-revision", "matched_title": "当前模型"},
        ]}
        with patch.object(service.agent_model_role_service, "get_model_by_role", return_value={"provider": "ollama"}), patch.object(
            service.agent_model_role_service, "apply_agent_llm_params", side_effect=lambda _state, model: model,
        ), patch.object(service.llm_service, "call_model", return_value=json.dumps(proposal)) as call:
            result = asyncio.run(service.select_current_models(
                {"query": "当前模型", "agent_id": "a"},
                [{"doc_id": "d", "revision_id": "r", "title": "当前模型"}], doc_type="rbd"))
        self.assertEqual({}, result)
        self.assertEqual(2, call.call_count)


if __name__ == "__main__":
    unittest.main()
