"""Serving v5 contracts and node lifecycle; model quality is measured separately."""
import asyncio
import copy
import json
import random
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

from backend.agent import answer_observation, answer_review as protocol, tracing
from backend.agent.nodes import answer
from backend.agent.nodes.answer_request import _answer_task_context
from scripts.replay_icp_review import replay
from tests.legacy_review_fixtures import state_for

TEXT = "输出更新时间滞后。"
DRAFT = TEXT + "[^1]"


def response(*, issues=(), formal_claims=(), verdict="pass", completion="complete"):
    return {"version": 5, "verdict": verdict, "completion": completion,
            "issues": list(issues), "formal_claims": list(formal_claims)}


def issue(kind="factual", quote="", reason="把其他故障的方法归给当前记录", requirement=""):
    return {"kind": kind, "span_ids": ["p1"] if quote else [], "reason": reason,
            "requirement": requirement, "citation_ids": []}


def prepared(state, draft=DRAFT):
    ledger, conversation, observations = answer._answer_review_material(state)
    material = protocol.prepare_review(draft, ledger=ledger, conversation=conversation, observations=observations,
        task=_answer_task_context(state), document_semantics=answer.document_semantics_prompt())
    return material, ledger


def validate(value=None, state=None, draft=DRAFT):
    state = state or state_for()
    material, ledger = prepared(state, draft)
    return protocol.validate_review(json.dumps(response() if value is None else value, ensure_ascii=False),
        prepared=material, ledger=ledger, safety=state.get("safety_decision") or {},
        calculation_requests=state.get("calculation_requests") or [])


def node_run(state=None, review=None, *, draft=DRAFT, revision=None, second_review=None, config=None, observed=False):
    outputs = [draft, json.dumps(response() if review is None else review, ensure_ascii=False)]
    if revision is not None:
        outputs += [revision, json.dumps(response() if second_review is None else second_review, ensure_ascii=False)]
    sequence = iter(outputs)
    def next_response(*_args, **_kwargs):
        try:
            return next(sequence)
        except StopIteration:
            raise AssertionError("Unexpected extra model call") from None
    with ExitStack() as stack:
        stack.enter_context(patch.object(answer, "_with_answer_evidence_rerank", new=AsyncMock(side_effect=lambda x: x)))
        stack.enter_context(patch.object(answer.agent_model_role_service, "apply_agent_llm_params",
            return_value={"provider": "ollama", "model_name": "review-fixture", "num_predict": 1200, "max_tokens": 4096}))
        call = stack.enter_context(patch.object(answer.llm_service, "call_model", side_effect=next_response))
        async def stream(messages, model):
            yield answer.llm_service.call_model(messages, model)
        stack.enter_context(patch.object(answer.llm_service, "stream_model", side_effect=stream))
        captured = stack.enter_context(answer_observation.observe_answers()) if observed else None
        result = asyncio.run(answer.answer_node(copy.deepcopy(state or state_for()), config))
        return result, call.call_args_list, captured


class WholeAnswerReviewTests(unittest.TestCase):
    def test_quoted_review_reasons_survive_without_length_or_vocabulary_rules(self):
        for reason in ('资料写「更新时间滞后」，不支持完全丧失。',
                       'The source says "delayed", not "lost".',
                       '该条件尚未得到证据支持。' * 100):
            with self.subTest(reason=reason[:30]), self.assertRaises(protocol.AnswerReviewError) as caught:
                validate(response(verdict='revise', issues=[issue(quote=DRAFT, reason=reason)]))
            self.assertEqual('review_rejected', caught.exception.code)
            self.assertEqual(reason, caught.exception.feedback[0]['detail'])

    def test_unescaped_review_quote_is_not_silently_repaired_or_accepted(self):
        material, ledger = prepared(state_for())
        raw = json.dumps(response(verdict='revise', issues=[issue(quote=DRAFT, reason='source "delayed"')]))
        with self.assertRaises(protocol.AnswerReviewError) as caught:
            protocol.validate_review(raw.replace('\\"delayed\\"', '"delayed"'), prepared=material,
                                     ledger=ledger, safety={}, calculation_requests=[])
        self.assertEqual('invalid_review_json', caught.exception.code)

    def test_semantic_feedback_preserves_review_coordinates_without_selecting_sources(self):
        finding = {**issue(quote=DRAFT, reason='来源描述更新时间滞后，不支持完全丧失；应保留原失效程度。'),
                   'citation_ids': [1]}
        with self.assertRaises(protocol.AnswerReviewError) as caught:
            validate(response(verdict='revise', issues=[finding]))
        feedback = caught.exception.feedback[0]
        self.assertEqual(['p1'], feedback['span_ids'])
        self.assertEqual([1], feedback['citation_ids'])
        self.assertEqual(DRAFT, feedback['text'])
        self.assertEqual(finding['reason'], feedback['detail'])
        finding['citation_ids'].clear()
        self.assertEqual([1], feedback['citation_ids'])
        self.assertNotIn('replacement', feedback)

    def test_semantic_binding_and_formal_issues_are_returned_together(self):
        draft = 'OTHER-918 的输出必然停止。[^1]\n另一个结论。[^99]'
        value = response(verdict='revise', issues=[issue(quote=draft, reason='资料未证明必然停止')],
                         formal_claims=[{'span_id': 'p1', 'kind': 'current_unique_diagnosis'},
                                        {'span_id': 'p2', 'kind': 'formal_calculation'}])
        with self.assertRaises(protocol.AnswerReviewError) as caught:
            validate(value, draft=draft)
        details = '\n'.join(row['detail'] for row in caught.exception.feedback)
        for expected in ('资料未证明必然停止', 'OTHER-918', '[^99]', 'blocked_claim',
                         'unknown_formal_claim_citation'):
            self.assertIn(expected, details)
        self.assertEqual('review_rejected', caught.exception.code)

    def test_unverified_review_remains_unverified_even_with_binding_feedback(self):
        with self.assertRaises(protocol.AnswerReviewError) as caught:
            validate(response(verdict='unverified'), draft='OTHER-918。[^1]')
        self.assertEqual('review_unverified', caught.exception.code)
        self.assertIn('OTHER-918', caught.exception.feedback[0]['detail'])

    def test_ordinary_joint_citations_do_not_override_wrong_row_or_version_review(self):
        from backend.agent.tools.icp_diagnosis import _direct_evidence
        items = _direct_evidence('function_locator', [
            {'doc_id': 'manual', 'content_type': 'function_component', 'component_id': 'MOD-019',
             'name': '数据模块', 'description': '保存配置'},
            {'doc_id': 'manual', 'function_id': 'FUNC-8X', 'module_id': 'MOD-019',
             'module_name': '数据模块', 'function_name': '读取配置'},
        ])
        state = state_for()
        state.update(evidence_registry=items, context_evidence_ids=[i['evidence_id'] for i in items],
                     safety_decision={'approved_evidence_ids': [i['evidence_id'] for i in items]})
        draft = 'MOD-019 提供FUNC-8X。[^1][^2]'
        self.assertEqual(draft, validate(state=state, draft=draft)['text'])
        for reason in ('关联部件并不证明可借用邻居故障行的方法', '文档版本不兼容，不能联合断言',
                       '同属一个系统不能证明必然因果'):
            with self.assertRaises(protocol.AnswerReviewError):
                validate(response(verdict='revise', issues=[issue(quote=draft, reason=reason)]), state, draft)

    def test_binding_revision_feedback_identifies_text_code_and_owned_citations(self):
        with self.assertRaises(protocol.AnswerReviewError) as caught:
            validate(draft='OTHER-918 的输出滞后。[^1]')
        feedback = caught.exception.feedback[0]
        self.assertEqual('OTHER-918 的输出滞后。[^1]', feedback['text'])
        self.assertIn('OTHER-918', feedback['detail'])
        self.assertIn('[^1]', feedback['detail'])
        self.assertEqual('review_rejected', caught.exception.code)

    def test_writer_owns_bytes_and_refs_not_critic(self):
        for draft in (DRAFT, "数据新鲜度下降。[^1]", "**ROW-731** " + DRAFT, "The output is stale.[^1]"):
            self.assertEqual(draft, validate(draft=draft)["text"])
        for number in (0, 99):
            with self.assertRaises(protocol.AnswerReviewError):
                validate(draft=TEXT + f"[^{number}]")

    def test_review_has_no_per_unit_proof_and_no_source_reassignment(self):
        schema = protocol.review_schema()
        self.assertNotIn("units", schema["properties"])
        self.assertNotIn("goals", schema["properties"])
        self.assertEqual([5], schema["properties"]["version"]["enum"])
        value = response()
        value["text"] = "critic replacement"
        with self.assertRaises(protocol.AnswerReviewError):
            validate(value)

    def test_raw_question_remains_goal_planner_only_hints(self):
        state = state_for()
        state["dialogue_context"] = {"requested_facets": ["列举全部参数"]}
        material, ledger = prepared(state)
        self.assertEqual(state["query"], material["original_question"])
        self.assertEqual(["列举全部参数"], material["planning_hints"])
        self.assertIn("ROW-731", json.dumps(protocol.review_input(material, ledger)))
        fabricated_goal = response(verdict="revise", issues=[issue("task", requirement="列举全部参数")])
        with self.assertRaises(protocol.AnswerReviewError) as caught:
            validate(fabricated_goal, state)
        self.assertEqual("issue_not_grounded_in_user_request", caught.exception.code)

    def test_true_omission_revises_but_optional_advice_does_not(self):
        state = state_for()
        omission = response(verdict="revise", issues=[issue("task", requirement="最先受影响的是什么")])
        with self.assertRaises(protocol.AnswerReviewError) as caught:
            validate(omission, state)
        self.assertEqual("review_rejected", caught.exception.code)
        good = validate(response(issues=[issue("quality", reason="可以补充非必要背景")]))
        self.assertEqual("full", good["status"])
        self.assertEqual(1, len(good["quality_suggestions"]))

    def test_partial_is_honest_gap_not_permission_for_factual_error(self):
        self.assertEqual("partial", validate(response(completion="partial"))["status"])
        for verdict in ("pass", "revise"):
            with self.assertRaises(protocol.AnswerReviewError):
                validate(response(completion="partial", verdict=verdict, issues=[issue()]))

    def test_protocol_errors_or_empty_verdict_never_release(self):
        for value in ({}, [], {"version": 4}, response(verdict="unverified"),
                      response(verdict="revise"), response(issues=[{**issue(), "span_ids": ["p999"]}]),
                      response(issues=[issue("task")])):
            with self.subTest(value=value), self.assertRaises(protocol.AnswerReviewError):
                validate(value)
        material, ledger = prepared(state_for())
        for raw in ('{"version":5', '{"version":5,"version":5}', '{"x":NaN}', "[]", "已通过"):
            with self.assertRaises(protocol.AnswerReviewError):
                protocol.validate_review(raw, prepared=material, ledger=ledger, safety={}, calculation_requests=[])

    def test_hashes_bind_review_to_exact_draft_and_evidence(self):
        for change in ("draft", "source"):
            material, ledger = prepared(state_for())
            if change == "draft":
                material["draft"] += "新增内容"
            else:
                ledger[1]["metadata"]["answer_fields"]["local_impact"] = "changed"
            with self.assertRaises(protocol.AnswerReviewError):
                protocol.validate_review(json.dumps(response()), prepared=material, ledger=ledger, safety={}, calculation_requests=[])

    def test_random_wrong_rows_cannot_be_borrowed(self):
        rng = random.Random(501)
        for _ in range(12):
            code = f"FAULT-{rng.randrange(10000, 99999)}"
            state = state_for(record_id=code)
            self.assertEqual("full", validate(state=state, draft=code + DRAFT)["status"])
            with self.assertRaises(protocol.AnswerReviewError):
                validate(state=state, draft=code + "9 " + DRAFT)

    def test_exact_api_codes_wildcards_and_markdown(self):
        state = state_for(record_id="87654321**0001")
        self.assertEqual("full", validate(state=state, draft="87654321**0001 " + DRAFT)["status"])
        for text in ("87654321**001", "76543210**0001", "GetUnrecordedStatus"):
            with self.assertRaises(protocol.AnswerReviewError):
                validate(state=state, draft=text + " " + DRAFT)
        for draft in ("**说明\nROW-731** " + DRAFT, "* 本轮\n    * **ROW-731** " + DRAFT,
                      "**ROW-731 " + DRAFT + "**"):
            self.assertEqual(draft, validate(draft=draft)["text"])

    def test_provisional_and_model_only_allow_quoted_facts_not_formal_assertions(self):
        for meta in ({"validation_status": "provisional"}, {"usage_scope": "model_content_only"}):
            state = state_for("rbd")
            state["evidence_registry"][0]["metadata"].update(meta)
            ledger = {1: state["evidence_registry"][0]}
            material = protocol.prepare_review(DRAFT, ledger=ledger, conversation={}, observations=[],
                task={"original_question": "描述模型自身记载，不证明实际故障。"}, document_semantics="")
            def review(value):
                return protocol.validate_review(json.dumps(value), prepared=material, ledger=ledger,
                    safety=state["safety_decision"], calculation_requests=[])
            self.assertEqual("full", review(response())["status"])
            for kind in protocol.FORMAL_CLAIMS:
                with self.subTest(meta=meta, kind=kind), self.assertRaises(protocol.AnswerReviewError):
                    review(response(formal_claims=[{"span_id": "p1", "kind": kind}]))

    def test_unique_and_deterministic_permissions_are_code_owned(self):
        for kind in ("current_unique_diagnosis", "formal_calculation", "formal_numeric_result"):
            with self.assertRaises(protocol.AnswerReviewError):
                validate(response(formal_claims=[{"span_id": "p1", "kind": kind}]))
        state = state_for()
        state["safety_decision"]["can_claim_unique"] = True
        self.assertEqual("full", validate(response(formal_claims=[{"span_id": "p1", "kind": "current_unique_diagnosis"}]), state)["status"])
        state["safety_decision"]["blocked_claims"] = ["unique_diagnosis"]
        with self.assertRaises(protocol.AnswerReviewError):
            validate(response(formal_claims=[{"span_id": "p1", "kind": "current_unique_diagnosis"}]), state)

    def test_negative_semantic_findings_never_publish_as_quality_or_partial(self):
        for reason in (
            "将输入钳位错行解释为采样饱和", "混入邻居报码接口", "将FMECA事实冒称MFL",
            "用历史恢复结果授权当前继续运行", "把未知阶段扩大为任何阶段停机",
            "仅据结构数量证明主要可靠度贡献", "上轮案例条件与当前现象冲突",
            "有字段却宣称资料没有", "把未确认跨表关系变成没有API",
        ):
            with self.subTest(reason=reason), self.assertRaises(protocol.AnswerReviewError):
                validate(response(verdict="revise", issues=[issue(reason=reason)]))

    def test_current_approved_ledger_filters_stale_wrong_revision_anchors(self):
        state = state_for()
        state["selected_conversation_anchors"] = [{"record_id": "ROW-999", "revision_id": "old"}]
        self.assertNotIn("ROW-999", {v["record_id"] for v in prepared(state)[1].values()})
        with self.assertRaises(protocol.AnswerReviewError):
            validate(state=state, draft="ROW-999 " + DRAFT)
        state["safety_decision"]["approved_evidence_ids"] = []
        with self.assertRaises(protocol.AnswerReviewError):
            validate(state=state)


class AnswerEntryTests(unittest.TestCase):
    def test_unchanged_revision_stops_before_another_review_or_publication(self):
        queue = asyncio.Queue()
        call_count = []
        outputs = iter([DRAFT, json.dumps(response(verdict='revise', issues=[issue(quote=DRAFT)])), DRAFT])
        def output(*args, **kwargs):
            call_count.append(args)
            return next(outputs)
        with patch.object(answer, '_with_answer_evidence_rerank', new=AsyncMock(side_effect=lambda s: s)), \
             patch.object(answer.agent_model_role_service, 'apply_agent_llm_params',
                          return_value={'provider': 'ollama', 'max_tokens': 4096}), \
             patch.object(answer.llm_service, 'call_model', side_effect=output), \
             patch.object(answer, '_generate_answer_response', new=AsyncMock(side_effect=lambda *a: output())), \
             self.assertRaises(answer.AnswerGenerationError) as caught:
            asyncio.run(answer.answer_node(state_for(), {'configurable': {'stream_queue': queue}}))
        self.assertEqual('revision_unchanged', caught.exception.__cause__.code)
        self.assertEqual(3, len(call_count))
        self.assertTrue(queue.empty())

    def test_revision_receives_review_basis_and_preserves_writer_thinking_policy(self):
        bad = '输出完全丧失。[^1]'
        finding = {**issue(quote=bad, reason='记录只说明更新时间滞后，应纠正完全丧失的断言。'),
                   'citation_ids': [1]}
        result, calls, _ = node_run(draft=bad, review=response(verdict='revise', issues=[finding]), revision=DRAFT)
        revision_messages = calls[2].args[0]
        feedback = json.loads(revision_messages[-1]['content'].split('\n', 1)[1])['feedback'][0]
        self.assertEqual([1], feedback['citation_ids'])
        self.assertEqual(['p1'], feedback['span_ids'])
        self.assertEqual(finding['reason'], feedback['detail'])
        self.assertEqual(bad, revision_messages[-2]['content'])
        self.assertEqual([False, True, False, True], [call.args[1]['reasoning'] for call in calls])
        self.assertEqual(DRAFT, result['final_answer'])

    def test_only_review_enables_thinking_without_changing_output_budget(self):
        from backend.core.llm_service import _ollama_runtime_payload
        model = {"provider": "ollama", "model_name": "thinking-capable-fixture",
                 "max_tokens": 4096, "num_ctx": 32768, "temperature": 0.2}
        original = copy.deepcopy(model)
        reviewed = answer._prepare_answer_review_model(model)
        self.assertTrue(reviewed["reasoning"])
        self.assertFalse(answer._prepare_answer_model(model)["reasoning"])
        self.assertEqual(4096, reviewed["max_tokens"])
        self.assertEqual(model, original)
        payload = _ollama_runtime_payload([{"role": "user", "content": "fixed review"}], reviewed, stream=False)
        self.assertTrue(payload["think"])
        self.assertEqual(4096, payload["options"]["num_predict"])
        self.assertEqual(protocol.review_schema(), payload["format"])

    def test_one_revision_receives_both_checks_and_rechecks_new_citation_errors(self):
        bad = 'OTHER-918 必然停止。[^1]'
        review = response(verdict='revise', issues=[issue(quote=bad, reason='不支持必然停止')])
        result, calls, _ = node_run(draft=bad, review=review, revision=DRAFT)
        feedback = calls[2].args[0][-1]['content']
        self.assertIn('不支持必然停止', feedback)
        self.assertIn('这些引用未支持的精确身份：OTHER-918', feedback)
        self.assertEqual(4, len(calls))
        self.assertEqual(DRAFT, result['final_answer'])
        queue = asyncio.Queue()
        with self.assertRaises(answer.AnswerGenerationError):
            node_run(draft=bad, review=review, revision='输出更新时间滞后。[^99]',
                     config={'configurable': {'stream_queue': queue}})
        self.assertTrue(queue.empty())

    def test_normal_path_two_calls_unchanged_budget_and_text(self):
        result, calls, _ = node_run()
        self.assertEqual(2, len(calls))
        self.assertEqual(DRAFT, result["final_answer"])
        for call in calls:
            self.assertEqual(4096, call.args[1]["max_tokens"])
            self.assertEqual(4096, call.args[1]["num_predict"])
        self.assertEqual([5], calls[1].args[1]["format"]["properties"]["version"]["enum"])

    def test_one_revision_is_fully_rechecked_and_private_replayable(self):
        bad = "输出完全丧失。[^1]"
        result, calls, capture = node_run(review=response(verdict="revise", issues=[issue(quote=bad)]),
                                         draft=bad, revision=DRAFT, observed=True)
        self.assertEqual(4, len(calls))
        self.assertEqual(DRAFT, result["final_answer"])
        self.assertIn(DRAFT, calls[3].args[0][0]["content"])
        self.assertEqual(2, len(capture.reviews))
        for snapshot in capture.reviews:
            self.assertTrue(replay(snapshot)["same_verdict"])

    def test_second_failure_stops_no_third_draft_or_pruning(self):
        queue = asyncio.Queue()
        bad = response(verdict="revise", issues=[issue()])
        with self.assertRaises(answer.AnswerGenerationError):
            node_run(review=bad, revision=DRAFT, second_review=bad, config={"configurable": {"stream_queue": queue}})
        self.assertTrue(queue.empty())
        self.assertEqual(2, answer.MAX_ANSWER_REVIEW_ATTEMPTS)

    def test_failed_review_cannot_publish_a_separate_unreviewed_clarification(self):
        state = state_for()
        from backend.agent.clarification_contract import make_clarification_identity, make_clarification_reason
        state['clarification_reasons'] = [make_clarification_reason(kind='ambiguous_identity',
            source_kind='binding', source_name='rbd_query_binding', identity=make_clarification_identity(doc_type='rbd'))]
        bad = response(verdict='revise', issues=[issue()])
        queue = asyncio.Queue()
        with self.assertRaises(answer.AnswerGenerationError):
            node_run(state, bad, revision=DRAFT, second_review=bad, config={'configurable': {'stream_queue': queue}})
        self.assertTrue(queue.empty())

    def test_permission_feedback_locates_the_actual_text_and_declared_scope(self):
        with self.assertRaises(protocol.AnswerReviewError) as caught:
            validate(response(formal_claims=[{'span_id': 'p1', 'kind': 'current_unique_diagnosis'}]))
        self.assertEqual(DRAFT, caught.exception.feedback[0]['text'])
        self.assertIn('current_unique_diagnosis', caught.exception.feedback[0]['detail'])

    def test_invalid_review_is_system_error_not_writer_revision(self):
        with patch.object(answer, "_review_answer", new=AsyncMock(side_effect=protocol.AnswerReviewError("invalid review JSON"))) as critic, \
                patch.object(answer, "_generate_answer_response", new=AsyncMock(return_value=DRAFT)) as writer:
            with self.assertRaises(answer.AnswerGenerationError):
                node_run()
        critic.assert_awaited_once()
        writer.assert_awaited_once()

    def test_nonessential_advice_cannot_trigger_revision(self):
        result, calls, _ = node_run(review=response(issues=[issue("quality", reason="可补背景")]))
        self.assertEqual(DRAFT, result["final_answer"])
        self.assertEqual(2, len(calls))

    def test_revision_is_latest_message_with_original_evidence_unchanged(self):
        bad = '输出完全丧失。[^1]'
        result, calls, _ = node_run(review=response(verdict="revise", issues=[issue(reason="修订定位依据")]),
                                     draft=bad, revision=DRAFT)
        initial = calls[0].args[0]
        revised = calls[2].args[0]
        self.assertEqual(['user', 'assistant', 'user'], [m['role'] for m in revised])
        self.assertEqual(initial[0], revised[0])
        self.assertEqual(bad, revised[1]['content'])
        self.assertIn('修订定位依据', revised[-1]['content'])
        self.assertNotIn(initial[0]['content'], revised[-1]['content'])
        self.assertEqual(4, len(calls))
        self.assertEqual(DRAFT, result['final_answer'])

    def test_timeout_and_cancel_do_not_publish_or_retry(self):
        for error in (asyncio.TimeoutError, asyncio.CancelledError):
            queue = asyncio.Queue()
            with patch.object(answer, "_review_answer", new=AsyncMock(side_effect=error)) as critic:
                with self.assertRaises(asyncio.CancelledError if error is asyncio.CancelledError else answer.AnswerGenerationError):
                    node_run(config={"configurable": {"stream_queue": queue}})
            critic.assert_awaited_once()
            self.assertTrue(queue.empty())

    def test_total_deadline_is_not_reset_by_answer(self):
        from backend.agent.model_runtime_policy import turn_budget
        async def slow(*_args):
            await asyncio.sleep(.05)
            return DRAFT
        with turn_budget(.01), patch.object(answer, "_generate_answer_response", new=AsyncMock(side_effect=slow)):
            # Fixture setup can consume the deadline before node entry; both
            # the entry timeout and wrapped in-node timeout must abort the turn.
            with self.assertRaises((TimeoutError, answer.AnswerGenerationError)):
                node_run()

    def test_observation_pure_compaction_and_single_sse(self):
        queue = asyncio.Queue()
        result, _, capture = node_run(observed=True, config={"configurable": {"stream_queue": queue}})
        stages = {s["stage"]: s["answer"] for s in capture.stages}
        self.assertEqual(stages["integrated_review"], stages["before_public_compaction"])
        self.assertEqual(stages["before_public_compaction"], stages["after_public_compaction"])
        self.assertEqual({"type": "answer_delta", "content": result["final_answer"]}, queue.get_nowait())
        self.assertTrue(queue.empty())

    def test_open_followup_and_history_reach_both_models(self):
        state = state_for()
        state.update(query="什么观察能推翻刚才的解释？", messages=[{"role": "assistant", "content": "之前只解释影响"}])
        _, calls, _ = node_run(state)
        for call in calls:
            self.assertIn(state["query"], call.args[0][0]["content"])
            self.assertIn("之前只解释影响", call.args[0][0]["content"])

    def test_all_modes_and_legacy_labels_share_same_entry(self):
        for mode in ("fault_diagnosis", "architecture_knowledge", "exact_code_lookup", "maintenance_guidance", "knowledge_explanation"):
            state = state_for()
            state.update(answer_mode=mode, query_frame={"required_answer_fields": ["fault_cause", "history_steps"]})
            result, calls, _ = node_run(state)
            self.assertEqual(2, len(calls))
            self.assertEqual([], result["delivered_answer_fields"])

    def test_legacy_retrieval_keeps_verifiable_sources(self):
        result, _, _ = node_run({"query": "说明此记录", "vector_results": [{"doc_id": "legacy", "content": TEXT}]})
        self.assertEqual("legacy", result["citations"][0]["doc_id"])

    def test_public_trace_contains_no_private_feedback(self):
        tracing.begin_turn("review-test", "问题")
        try:
            node_run(review=response(verdict="revise", issues=[issue(reason="私有反馈不得公开")]),
                     draft='输出完全丧失。[^1]', revision=DRAFT)
            events = tracing.peek().events
            self.assertEqual(["rejected", "accepted"], [e["outcome"] for e in events if e["type"] == "answer_review"])
            self.assertNotIn("私有反馈", json.dumps(events, ensure_ascii=False))
        finally:
            tracing._current_trace.set(None)
