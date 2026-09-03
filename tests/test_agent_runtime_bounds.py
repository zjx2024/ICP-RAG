import asyncio
import importlib
import json
import unittest
from typing import TypedDict
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from backend.agent.contracts import make_evidence
from backend.agent.nodes import answer as answer_module
from backend.agent.nodes import plan as plan_module
from backend.core.llm_service import LLMClient, LLMStreamError
from langgraph.graph import END, StateGraph

chat_router = importlib.import_module("backend.agent.api.chat_router")


def _runtime_diagnosis_state():
    evidence = make_evidence(
        source_kind="postgres",
        authority="authoritative_record",
        evidence_level="L1_DIRECT",
        claim_type="hmc_fault",
        doc_type="hmc",
        doc_id="doc-hmc",
        doc_name="HMC手册.docx",
        record_id="91860003**0002",
        source_ref="HMC手册.docx#table=1&row=2",
        title="91860003**0002",
        content="FAN 转速反馈丢失",
    )
    evidence_id = evidence["evidence_id"]
    return {
        "intent": "symptom_diagnosis",
        "query": "FAN转速反馈丢失，可能是什么故障？",
        "messages": [],
        "candidate_faults": [
            {
                "fault": "FAN 转速反馈丢失",
                "score": 0.68,
                "supporting_evidence_ids": [evidence_id],
                "exact_hmc_match": False,
            }
        ],
        "diagnostic_support": 0.68,
        "clarification_question": "请确认告警出现时风扇是否仍在转动。",
        "evidence_registry": [evidence],
        "context_evidence_ids": [evidence_id],
        "safety_decision": {
            "status": "clarify",
            "can_claim_unique": False,
            "questions": ["请确认告警出现时风扇是否仍在转动。"],
            "approved_evidence_ids": [evidence_id],
            "blocked_evidence_ids": [],
            "context_evidence_ids": [evidence_id],
        },
    }


class AgentModelBudgetTests(unittest.TestCase):
    def test_plan_uses_short_non_reasoning_json_budget_for_ollama(self):
        response = json.dumps(
            {
                "intent": "symptom_diagnosis",
                "rewritten_query": "FAN 转速反馈丢失",
                "entity_keywords": ["FAN", "转速反馈丢失"],
                "type_keywords": ["故障诊断"],
                "tools_plan": [{"tool": "hybrid_search", "reason": "召回候选"}],
                "reasoning": "按症状检索",
            },
            ensure_ascii=False,
        )
        model = {"provider": "ollama", "max_tokens": 4096}
        state = {"query": "FAN转速反馈丢失", "messages": [], "memory_enabled": False}
        with (
            patch.object(plan_module.llm_service, "_call_model", return_value=response) as call,
            patch.object(
                plan_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=model,
            ),
        ):
            asyncio.run(plan_module.plan_node(state))

        prepared = call.call_args.args[1]
        self.assertFalse(prepared["reasoning"])
        self.assertEqual(0.0, prepared["temperature"])
        self.assertEqual("json", prepared["format"])
        self.assertEqual(plan_module.PLAN_MAX_PREDICTION_TOKENS, prepared["num_predict"])
        self.assertGreaterEqual(
            prepared["num_predict"],
            1536,
            "nested planner JSON must have enough budget to close the object",
        )

    def test_plan_timeout_falls_back_to_deterministic_policy(self):
        observed = []

        async def slow_stream(_messages, model):
            observed.append(model)
            await asyncio.sleep(1)
            yield "{}"

        state = {"query": "FAN转速反馈丢失", "messages": [], "memory_enabled": False}
        config = {"configurable": {"stream_queue": asyncio.Queue()}}
        with (
            patch.object(plan_module.llm_service, "stream_model", new=slow_stream),
            patch.object(
                plan_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value={"provider": "ollama", "max_tokens": 4096},
            ),
            patch.object(plan_module, "PLAN_LLM_TIMEOUT_SECONDS", 0.01),
        ):
            result = asyncio.run(plan_module.plan_node(state, config))

        self.assertIn("超过时间预算", result["plan_reasoning"])
        self.assertTrue(result["allowed_tools"])
        self.assertFalse(observed[0]["reasoning"])

    def test_answer_timeout_raises_without_exposing_unprocessed_evidence(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            doc_id="doc-hmc",
            doc_name="HMC手册.docx",
            record_id="91860003**0002",
            source_ref="HMC手册.docx#table=1&row=2",
            title="91860003**0002",
            content="FAN 转速反馈丢失",
        )
        evidence_id = evidence["evidence_id"]
        state = {
            "intent": "symptom_diagnosis",
            "query": "FAN转速反馈丢失，可能是什么故障？",
            "messages": [],
            "candidate_faults": [
                {
                    "fault": "FAN 转速反馈丢失",
                    "score": 0.68,
                    "supporting_evidence_ids": [evidence_id],
                    "exact_hmc_match": False,
                }
            ],
            "diagnostic_support": 0.68,
            "clarification_question": "请确认告警出现时风扇是否仍在转动。",
            "evidence_registry": [evidence],
            "context_evidence_ids": [evidence_id],
            "safety_decision": {
                "status": "clarify",
                "can_claim_unique": False,
                "questions": ["请确认告警出现时风扇是否仍在转动。"],
                "approved_evidence_ids": [evidence_id],
                "blocked_evidence_ids": [],
                "context_evidence_ids": [evidence_id],
            },
        }
        observed = []

        async def slow_stream(_messages, model):
            observed.append(model)
            await asyncio.sleep(1)
            yield "迟到的模型回答"

        config = {"configurable": {"stream_queue": asyncio.Queue()}}
        with (
            patch.object(answer_module.llm_service, "stream_model", new=slow_stream),
            patch.object(
                answer_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value={"provider": "ollama", "max_tokens": 4096},
            ),
            patch.object(answer_module, "ANSWER_LLM_TIMEOUT_SECONDS", 0.01),
        ):
            with self.assertRaises(answer_module.AnswerGenerationError) as caught:
                asyncio.run(answer_module.answer_node(state, config))

        self.assertIn("时间预算内未完成", str(caught.exception))
        self.assertNotIn("FAN 转速反馈丢失", str(caught.exception))
        self.assertTrue(config["configurable"]["stream_queue"].empty())
        self.assertFalse(observed[0]["reasoning"])
        self.assertEqual(answer_module.ANSWER_MAX_PREDICTION_TOKENS, observed[0]["num_predict"])

    def test_partial_stream_failure_discards_partial_text_and_raises(self):
        async def broken_stream(_messages, _model):
            yield "不应保留的半截回答"
            raise LLMStreamError("connection lost")

        state = _runtime_diagnosis_state()
        config = {"configurable": {"stream_queue": asyncio.Queue()}}
        with (
            patch.object(answer_module.llm_service, "stream_model", new=broken_stream),
            patch.object(
                answer_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value={"provider": "ollama", "max_tokens": 4096},
            ),
        ):
            with self.assertRaises(answer_module.AnswerGenerationError) as caught:
                asyncio.run(answer_module.answer_node(state, config))

        self.assertNotIn("不应保留的半截回答", str(caught.exception))
        self.assertIn("回答模型调用失败", str(caught.exception))
        self.assertTrue(config["configurable"]["stream_queue"].empty())

    def test_blocked_exact_candidate_does_not_claim_postgres_exact_hit(self):
        state = _runtime_diagnosis_state()
        evidence = state["evidence_registry"][0]
        evidence["match_type"] = "exact_code"
        evidence_id = evidence["evidence_id"]
        state["candidate_faults"][0]["exact_hmc_match"] = True
        state["safety_decision"]["approved_evidence_ids"] = []
        state["safety_decision"]["blocked_evidence_ids"] = [evidence_id]

        result = answer_module._fallback_candidate_lines(state, state["candidate_faults"])

        self.assertNotIn("PostgreSQL HMC 精确代码命中", result)
        self.assertIn("不能单独支撑结论", result)


class LlmStreamFailureTests(unittest.TestCase):
    def test_unknown_stream_provider_raises_instead_of_yielding_error_text(self):
        client = object.__new__(LLMClient)
        client.provider = None
        client.chat_model = None

        async def consume():
            return [
                token
                async for token in client.stream_model(
                    [{"role": "user", "content": "test"}],
                    {"provider": "unknown"},
                )
            ]

        with self.assertRaises(LLMStreamError):
            asyncio.run(consume())


class ChatGraphRuntimeTests(unittest.TestCase):
    def test_real_state_graph_filters_private_thinking_and_bounds_process_evidence(self):
        class RuntimeState(TypedDict, total=False):
            intent: str
            plan_reasoning: str
            tool_calls: list
            tool_results: list
            safety_decision: dict
            final_answer: str
            citations: list
            evidence_catalog: list
            used_evidence_ids: list

        async def plan_node(_state, config=None):
            await config["configurable"]["stream_queue"].put({
                "type": "thinking_delta",
                "source": "plan",
                "content": "不应发送给浏览器的逐字思考",
            })
            return {
                "intent": "impact_analysis",
                "plan_reasoning": "不应出现在 plan 事件中的原始推理",
            }

        async def tool_node(_state):
            result_ids = [f"ev-{index}" for index in range(60)]
            return {
                "tool_calls": [{
                    "tool": "fmeca_query",
                    "count": 60,
                    "evidence_ids": ["ev-59"],
                }],
                "tool_results": [{
                    "tool": "fmeca_query",
                    "status": "ok",
                    "evidence_ids": result_ids,
                }],
            }

        async def gate_node(_state):
            return {
                "safety_decision": {
                    "status": "answer",
                    "context_evidence_ids": [f"ev-{index}" for index in range(50, 90)],
                    "approved_evidence_ids": [f"ev-{index}" for index in range(50, 70)],
                    "blocked_evidence_ids": [f"ev-{index}" for index in range(70, 90)],
                }
            }

        async def answer_node(_state, config=None):
            process_ids = list(config["configurable"]["process_evidence_ids"])
            return {
                "final_answer": "影响说明",
                "citations": [],
                "evidence_catalog": [
                    {"catalog_index": index, "evidence_id": evidence_id}
                    for index, evidence_id in enumerate(process_ids, start=1)
                ],
                "used_evidence_ids": [],
            }

        builder = StateGraph(RuntimeState)
        builder.add_node("plan", plan_node)
        builder.add_node("icp_diagnosis", tool_node)
        builder.add_node("evidence_gate", gate_node)
        builder.add_node("answer", answer_node)
        builder.set_entry_point("plan")
        builder.add_edge("plan", "icp_diagnosis")
        builder.add_edge("icp_diagnosis", "evidence_gate")
        builder.add_edge("evidence_gate", "answer")
        builder.add_edge("answer", END)
        graph = builder.compile()

        async def scenario():
            queue = asyncio.Queue()
            config = {"configurable": {}}
            await chat_router._stream_graph_events(
                graph, {}, "session-real-graph", "agent-1", queue, config
            )
            items = []
            while True:
                item = await queue.get()
                items.append(item)
                if item is None:
                    return items, config

        items, config = asyncio.run(scenario())
        public_items = items[:-1]
        self.assertNotIn("thinking_delta", [item["type"] for item in public_items])
        plan_event = next(item for item in public_items if item["type"] == "plan")
        self.assertNotIn("reasoning", plan_event)

        tool_event = next(item for item in public_items if item["type"] == "tool_call")
        self.assertEqual("ev-59", tool_event["evidence_ids"][0])
        self.assertEqual(60, tool_event["evidence_ids_total"])
        self.assertFalse(tool_event["evidence_ids_truncated"])

        stage = next(item for item in public_items if item["type"] == "stage")
        self.assertEqual(40, stage["reviewed_evidence_total"])
        self.assertTrue(stage["reviewed_evidence_ids_truncated"])
        self.assertEqual(20, stage["blocked_evidence_total"])
        self.assertTrue(stage["blocked_evidence_ids_truncated"])
        self.assertLessEqual(
            len(config["configurable"]["process_evidence_ids"]),
            chat_router.PROCESS_EVIDENCE_EVENT_LIMIT,
        )

        answer = next(item for item in public_items if item["type"] == "answer")
        catalog_ids = {item["evidence_id"] for item in answer["evidence_catalog"]}
        exposed_ids = set(tool_event["evidence_ids"])
        exposed_ids.update(stage["reviewed_evidence_ids"])
        exposed_ids.update(stage["approved_evidence_ids"])
        exposed_ids.update(stage["blocked_evidence_ids"])
        self.assertTrue(exposed_ids.issubset(catalog_ids))

    def test_total_timeout_emits_error_instead_of_unprocessed_answer(self):
        class SlowGraph:
            def astream(self, _state, config=None, stream_mode=None):
                async def events():
                    await asyncio.sleep(1)
                    yield {"answer": {"final_answer": "late", "citations": []}}
                return events()

        async def scenario():
            queue = asyncio.Queue()
            with patch.object(chat_router, "CHAT_TOTAL_TIMEOUT_SECONDS", 0.01):
                await chat_router._stream_graph_events(
                    SlowGraph(), {}, "session-timeout", "agent-1", queue, {}
                )
            return [await queue.get(), await queue.get()]

        items = asyncio.run(scenario())
        self.assertEqual("error", items[0]["type"])
        self.assertEqual("total_timeout", items[0]["terminal_reason"])
        self.assertIn("未生成经过大模型整理的答案", items[0]["message"])
        self.assertIsNone(items[1])

    def test_answer_and_citations_are_queued_atomically(self):
        citation = {"id": 1, "source_ref": "HMC.docx#row=2"}
        catalog = [{"catalog_index": 1, "evidence_id": "ev-1"}]

        class AnswerGraph:
            def astream(self, _state, config=None, stream_mode=None):
                async def events():
                    yield {"answer": {
                        "final_answer": "结果 [^1]",
                        "citations": [citation],
                        "evidence_catalog": catalog,
                        "used_evidence_ids": ["ev-1"],
                    }}
                return events()

        async def scenario():
            queue = asyncio.Queue()
            await chat_router._stream_graph_events(
                AnswerGraph(), {}, "session-answer", "agent-1", queue, {}
            )
            return [await queue.get(), await queue.get()]

        items = asyncio.run(scenario())
        self.assertEqual("answer", items[0]["type"])
        self.assertEqual([citation], items[0]["citations"])
        self.assertEqual(catalog, items[0]["evidence_catalog"])
        self.assertEqual(["ev-1"], items[0]["used_evidence_ids"])
        self.assertIsNone(items[1])

    def test_chat_persists_terminal_answer_before_emitting_it(self):
        timeline = []
        citation = {"id": 1, "evidence_id": "ev-1"}
        catalog = [{"catalog_index": 1, "evidence_id": "ev-1", "status": "answer_used"}]

        class TerminalGraph:
            def astream(self, _state, config=None, stream_mode=None):
                async def events():
                    await config["configurable"]["stream_queue"].put({
                        "type": "answer_delta",
                        "content": "已整理答案 [^1]",
                    })
                    yield {"answer": {
                        "final_answer": "已整理答案 [^1]",
                        "citations": [citation],
                        "evidence_catalog": catalog,
                        "used_evidence_ids": ["ev-1"],
                    }}
                return events()

        def add_message(_session_id, role, content="", **_kwargs):
            timeline.append(f"save:{role}")
            return {"id": f"message-{role}"}

        class ConnectedRequest:
            async def is_disconnected(self):
                return False

        async def scenario():
            body = chat_router.ChatRequest(
                agent_id="agent-1",
                session_id="session-persist-first",
                query="测试问题",
            )
            with (
                patch.object(chat_router.agent_db, "get_agent", return_value={
                    "kb_ids": [],
                    "enabled_tools": ["icp_diagnosis"],
                    "memory_enabled": False,
                }),
                patch.object(chat_router.agent_db, "get_session", return_value={"id": body.session_id}),
                patch.object(chat_router.agent_db, "add_message", side_effect=add_message),
                patch.object(chat_router.agent_db, "list_messages_for_state", return_value=[]),
                patch.object(
                    chat_router,
                    "get_compiled_graph",
                    AsyncMock(return_value=TerminalGraph()),
                ),
            ):
                response = await chat_router.chat(body, ConnectedRequest())
                async for raw_chunk in response.body_iterator:
                    chunk = raw_chunk.decode() if isinstance(raw_chunk, bytes) else raw_chunk
                    for line in chunk.splitlines():
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        timeline.append(f"emit:{event['type']}")

        asyncio.run(scenario())
        self.assertLess(timeline.index("save:assistant"), timeline.index("emit:answer_delta"))
        self.assertLess(timeline.index("save:assistant"), timeline.index("emit:answer"))
        self.assertLess(timeline.index("emit:answer"), timeline.index("emit:done"))

    def test_icp_execution_events_follow_graph_order_and_include_evidence_review(self):
        class IcpGraph:
            def astream(self, _state, config=None, stream_mode=None):
                async def events():
                    yield {
                        "plan": {
                            "intent": "impact_analysis",
                            "answer_mode": "knowledge_explanation",
                            "tools_plan": [{"tool": "fmeca_query", "reason": "查询影响"}],
                        }
                    }
                    yield {
                        "icp_diagnosis": {
                            "tool_calls": [{"tool": "fmeca_query", "count": 3, "duration": 0.01}],
                            "tool_results": [{
                                "tool": "fmeca_query",
                                "status": "ok",
                                "evidence_ids": ["ev-1", "ev-2"],
                            }],
                        }
                    }
                    yield {
                        "adaptive_tools": {
                            "need_more": True,
                            "next_action": "graph_traversal",
                            "reflection": "已检查 1 项查询和 2 条候选证据；需要检查受限的跨文档关系。",
                            "adaptive_decision": {
                                "next_tools": ["graph_traversal"],
                                "executed_tools": ["fmeca_query"],
                            },
                        }
                    }
                    yield {
                        "graph_traversal": {
                            "tool_calls": [{
                                "tool": "graph_traversal",
                                "count": 2,
                                "duration": 0.015,
                                "evidence_ids": ["ev-graph"],
                            }],
                            "tool_results": [{
                                "tool": "graph_traversal",
                                "status": "ok",
                                "evidence_ids": ["ev-graph"],
                            }],
                        }
                    }
                    yield {
                        "icp_calculation": {
                            "tool_calls": [{"tool": "fta_analyzer", "count": 1, "duration": 0.02}]
                        }
                    }
                    yield {
                        "evidence_gate": {
                            "safety_decision": {
                                "status": "clarify",
                                "approved_evidence_ids": ["ev-1"],
                                "blocked_evidence_ids": ["ev-2", "ev-3"],
                                "requires_clarification": True,
                            }
                        }
                    }
                    yield {"answer": {"final_answer": "影响说明 [^1]", "citations": []}}
                return events()

        async def scenario():
            queue = asyncio.Queue()
            await chat_router._stream_graph_events(
                IcpGraph(), {}, "session-steps", "agent-1", queue, {}
            )
            items = []
            while True:
                item = await queue.get()
                items.append(item)
                if item is None:
                    return items

        items = asyncio.run(scenario())
        self.assertEqual(
            ["plan", "tool_call", "reflect", "tool_call", "tool_call", "stage", "answer"],
            [item["type"] for item in items[:-1]],
        )
        self.assertEqual("graph_traversal", items[3]["tool"])
        self.assertEqual(["fmeca_query"], items[2]["tried_tools"])
        self.assertEqual(["graph_traversal"], items[2]["selected_tools"])
        self.assertEqual("evidence_review", items[5]["stage"])
        self.assertEqual("partial", items[5]["status"])
        self.assertEqual(
            {
                "reviewed_count": 3,
                "approved_count": 1,
                "excluded_count": 2,
                "requires_clarification": True,
            },
            items[5]["metrics"],
        )
        self.assertEqual(["ev-1", "ev-2"], items[1]["evidence_ids"])
        self.assertEqual(["ev-1"], items[5]["approved_evidence_ids"])
        self.assertEqual(["ev-2", "ev-3"], items[5]["blocked_evidence_ids"])
        self.assertEqual(["ev-1", "ev-2", "ev-3"], items[5]["reviewed_evidence_ids"])
        self.assertIsNone(items[-1])

    def test_same_session_concurrent_chat_is_rejected(self):
        session_id = "already-running"
        chat_router._active_sessions.add(session_id)
        body = chat_router.ChatRequest(
            agent_id="agent-1",
            session_id=session_id,
            query="test",
        )
        try:
            with (
                patch.object(chat_router.agent_db, "get_agent", return_value={"agent_id": "agent-1"}),
                patch.object(chat_router.agent_db, "get_session", return_value={"id": session_id}),
            ):
                with self.assertRaises(HTTPException) as caught:
                    asyncio.run(chat_router.chat(body, None))
        finally:
            chat_router._active_sessions.discard(session_id)

        self.assertEqual(409, caught.exception.status_code)


class StopChatTests(unittest.TestCase):
    def test_stop_chat_cancels_active_graph_task(self):
        async def scenario():
            session_id = "runtime-bound-session"
            task = asyncio.create_task(asyncio.sleep(60))
            chat_router._running_tasks[session_id] = task
            try:
                result = await chat_router.stop_chat(session_id)
                await asyncio.gather(task, return_exceptions=True)
                return result, task.cancelled()
            finally:
                chat_router._running_tasks.pop(session_id, None)
                chat_router._stop_requests.discard(session_id)

        result, cancelled = asyncio.run(scenario())
        self.assertTrue(result["stopped"])
        self.assertTrue(cancelled)

    def test_stop_chat_reports_when_nothing_is_running(self):
        result = asyncio.run(chat_router.stop_chat("missing-session"))
        self.assertFalse(result["stopped"])
        self.assertEqual("no_active_generation", result["reason"])


if __name__ == "__main__":
    unittest.main()
