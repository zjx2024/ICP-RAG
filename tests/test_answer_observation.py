import asyncio
from contextlib import ExitStack
import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from backend.agent import answer_observation as observation


class AnswerObservationTests(unittest.TestCase):
    def test_disabled_is_noop_without_inspecting_inputs(self):
        class ExplodingMapping(dict):
            def __contains__(self, key):
                raise AssertionError("must not inspect production state")
        self.assertFalse(observation.answer_observation_enabled())
        observation.observe_answer_input(ExplodingMapping())
        observation.observe_answer_stage("ignored", None)

    def test_context_restores_after_nesting_and_exception(self):
        with observation.observe_answers() as outer:
            observation.observe_answer_stage("outer", "甲")
            with self.assertRaises(RuntimeError):
                with observation.observe_answers() as inner:
                    observation.observe_answer_stage("inner", "乙")
                    raise RuntimeError("cancel")
            observation.observe_answer_stage("outer-end", "丙")
        self.assertEqual(["outer", "outer-end"], [s["stage"] for s in outer.stages])
        self.assertEqual(["inner"], [s["stage"] for s in inner.stages])
        self.assertFalse(observation.answer_observation_enabled())

    def test_async_tasks_and_worker_contexts_are_isolated(self):
        async def run(label):
            with observation.observe_answers() as captured:
                await asyncio.sleep(0)
                await asyncio.to_thread(observation.observe_answer_stage, "thread", label)
                return captured.as_dict()
        async def both():
            return await asyncio.gather(run("甲"), run("乙"))
        left, right = asyncio.run(both())
        self.assertEqual("甲", left["stages"][0]["answer"])
        self.assertEqual("乙", right["stages"][0]["answer"])

    def test_snapshots_are_detached_and_connection_secrets_are_excluded(self):
        state = {"query": "Q", "api_key": "key-value", "messages": [],
                 "evidence_registry": [{"metadata": {"api_key": "nested-key", "value": "fact"}}]}
        with observation.observe_answers() as captured:
            observation.observe_answer_input(state, model={"model_name": "M", "max_tokens": 1200,
                                                          "api_key": "model-key", "base_url": "secret-endpoint"})
            observation.observe_answer_stage("draft", "甲", citations=[{"id": 8}])
            observation.observe_answer_stage("published", "乙")
        state["evidence_registry"][0]["metadata"]["value"] = "changed"
        result = captured.as_dict()
        self.assertEqual("fact", result["inputs"][0]["state"]["evidence_registry"][0]["metadata"]["value"])
        serialized = json.dumps(result)
        for secret in ("key-value", "nested-key", "model-key", "secret-endpoint"):
            self.assertNotIn(secret, serialized)
        self.assertTrue(result["stages"][1]["changed"])
        self.assertNotIn("passed", result["stages"][1])
        self.assertEqual(1200, result["inputs"][0]["model"]["max_tokens"])

    def test_observer_failure_is_explicit_and_non_mutating(self):
        with observation.observe_answers() as captured:
            observation.observe_answer_stage("bad", None)
        self.assertEqual(["stage:bad:TypeError"], captured.errors)

    def test_model_metrics_are_whitelisted_without_response_or_credentials(self):
        from backend.agent.tracing import record_llm
        with observation.observe_answers() as captured:
            record_llm(template_name="answer", duration_ms=12, ok=True, model_name="M", response_chars=6)
        self.assertEqual(1, len(captured.model_calls))
        self.assertEqual(6, captured.model_calls[0]["response_chars"])

    def test_model_observation_failure_does_not_escape_to_the_model_call(self):
        with observation.observe_answers() as captured, patch.object(
            observation, "detached", side_effect=ValueError("private detail")
        ):
            observation.observe_model_call({"ok": True})
        self.assertEqual(["model_call:ValueError"], captured.errors)
        self.assertEqual([], captured.model_calls)

    def test_actual_answer_node_is_unchanged_when_observed(self):
        from backend.agent.nodes import answer
        evidence = {"evidence_id": "fact-1", "record_id": "FM-RANDOM", "doc_id": "doc-random",
                    "doc_type": "fmeca", "claim_type": "fmeca_fact", "authority": "authoritative_record",
                    "title": "测试模式", "content": "输出更新时间滞后。", "metadata": {
                        "answer_fields": {"failure_mode": "测试模式", "local_impact": "输出更新时间滞后。"}}}
        state = {"query": "影响是什么？", "intent": "impact_analysis", "evidence_registry": [evidence],
                 "query_frame": {"required_answer_fields": ["local_impact"]},
                 "safety_decision": {"approved_evidence_ids": ["fact-1"]}}
        with ExitStack() as stack:
            stack.enter_context(patch.object(answer, "_with_answer_evidence_rerank", new=AsyncMock(side_effect=lambda state: state)))
            stack.enter_context(patch.object(answer, "_generate_answer_response", new=AsyncMock(return_value="输出更新时间滞后。[^1]")))
            from test_answer_integrated_review import response
            stack.enter_context(patch.object(answer.llm_service, "call_model", return_value=json.dumps(response())))
            stack.enter_context(patch.object(answer.agent_model_role_service, "apply_agent_llm_params", return_value=None))
            plain = asyncio.run(answer.answer_node(copy.deepcopy(state)))
            with observation.observe_answers() as captured:
                observed = asyncio.run(answer.answer_node(copy.deepcopy(state)))
        self.assertEqual(plain, observed)
        self.assertEqual("initial_draft", captured.stages[0]["stage"])
        self.assertEqual("published", captured.stages[-1]["stage"])
        self.assertEqual(observed["final_answer"], captured.stages[-1]["answer"])
        self.assertEqual([], captured.errors)
        self.assertEqual(["entry", "generation"], [item["stage"] for item in captured.inputs])


if __name__ == "__main__":
    unittest.main()
