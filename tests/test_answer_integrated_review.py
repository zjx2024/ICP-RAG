"""Read-only critique/one revision. Fixtures test contracts, not model quality."""
import asyncio
import copy
import json
import random
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

from backend.agent import answer_observation, answer_review as protocol, tracing
from backend.agent.contracts import make_evidence
from backend.agent.nodes import answer
from backend.agent.nodes.answer_request import _answer_task_context

TEXT = "输出更新时间滞后。"
SOURCE = "e1:/answer_fields/local_impact"


def state_for(doc_type="fmeca", **changes):
    item = make_evidence(source_kind="postgres", authority="authoritative_record", evidence_level="L1_DIRECT",
        claim_type={"hmc": "hmc_fault", "rbd": "rbd_structure"}.get(doc_type, doc_type + "_fact"),
        doc_type=doc_type, doc_id="document-57", record_id="ROW-731", title="本轮对象",
        content=TEXT, metadata={"answer_fields": {"local_impact": TEXT}})
    item.update(changes)
    return {"query": "为什么它重要？最先受影响的是什么？", "messages": [],
        "evidence_registry": [item], "context_evidence_ids": [item["evidence_id"]],
        "safety_decision": {"approved_evidence_ids": [item["evidence_id"]], "can_claim_unique": False}}


def check(**changes):
    value = {"unit_id": "u1", "sources": [SOURCE], "claims": ["source_fact"],
             "attributed_doc_types": [], "issue": ""}
    value.update(changes)
    return value


def response(*units, status="answered"):
    return {"version": 2, "units": list(units) or [check()],
            "goals": [{"goal_id": "q0", "status": status, "issue": ""}]}


def prepared(state, draft=TEXT):
    ledger, conversation, observations = answer._answer_review_material(state)
    material = protocol.prepare_review(draft, ledger=ledger, conversation=conversation,
        observations=observations, task=_answer_task_context(state), document_semantics=answer.document_semantics_prompt())
    return material, ledger


def validate(value, state=None, draft=TEXT):
    state = state or state_for()
    material, ledger = prepared(state, draft)
    return protocol.validate_review(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
        prepared=material, ledger=ledger, safety=state.get("safety_decision") or {},
        calculation_requests=state.get("calculation_requests") or [])


def node_run(state=None, review=None, *, draft=TEXT, revision=None, second_review=None,
             third_revision=None, third_review=None, config=None, observed=False):
    state = state or state_for()
    outputs = [draft, json.dumps(review if review is not None else response(), ensure_ascii=False)]
    if revision is not None:
        outputs += [revision, json.dumps(second_review if second_review is not None else response(), ensure_ascii=False)]
    if third_revision is not None:
        outputs += [third_revision, json.dumps(third_review if third_review is not None else response(), ensure_ascii=False)]
    with ExitStack() as stack:
        stack.enter_context(patch.object(answer, "_with_answer_evidence_rerank", new=AsyncMock(side_effect=lambda x: x)))
        stack.enter_context(patch.object(answer.agent_model_role_service, "apply_agent_llm_params",
            return_value={"provider": "ollama", "model_name": "review-fixture", "num_predict": 1200, "max_tokens": 4096}))
        call = stack.enter_context(patch.object(answer.llm_service, "call_model", side_effect=outputs))
        async def stream(messages, model):
            yield answer.llm_service.call_model(messages, model)
        stack.enter_context(patch.object(answer.llm_service, "stream_model", side_effect=stream))
        captured = stack.enter_context(answer_observation.observe_answers()) if observed else None
        result = asyncio.run(answer.answer_node(copy.deepcopy(state), config))
        return result, call.call_args_list, captured


class ReviewProtocolTests(unittest.TestCase):
    def test_valid_and_rephrased_answers_preserve_writer_text(self):
        for draft in (TEXT, "数据新鲜度因此下降。", "The output is no longer fresh.", "使用方拿到的是较早时刻的数值。"):
            result = validate(response(), draft=draft)
            self.assertEqual(draft + "[^1]", result["text"])
            self.assertEqual("full", result["status"])
            self.assertEqual("answer", result["manifest"][0]["evidence_scope"])

    def test_lossy_planning_cannot_drop_whole_original_question(self):
        state = state_for()
        state["dialogue_context"] = {"status": "resolved", "requested_facets": ["未收录于字段目录的新目标"]}
        self.assertEqual(state["query"], prepared(state)[0]["goals"]["q0"])
        with self.assertRaises(protocol.AnswerReviewError):
            validate(response(), state)
        value = response()
        value["goals"].append({"goal_id": "q1", "status": "answered", "issue": ""})
        self.assertEqual("full", validate(value, state)["status"])

    def test_malformed_duplicate_nonfinite_old_protocol_fails(self):
        for raw in ('{"version":2', "已检查", "[AI调用失败]", "", '{"version":2,"version":2}',
                    '{"x":NaN}', '{"x":Infinity}', "[]", '{"version":1,"coverage_checked":true}'):
            with self.subTest(raw=raw), self.assertRaises(protocol.AnswerReviewError):
                validate(raw)

    def test_missing_extra_or_critic_text_fields_fail(self):
        for key in response():
            value = response()
            del value[key]
            with self.subTest(key=key), self.assertRaises(protocol.AnswerReviewError):
                validate(value)
        for key in check():
            value = response()
            del value["units"][0][key]
            with self.subTest(key=key), self.assertRaises(protocol.AnswerReviewError):
                validate(value)
        for key in ("text", "bindings", "kind", "history_bindings"):
            with self.subTest(key=key), self.assertRaises(protocol.AnswerReviewError):
                validate(response(check(**{key: "cannot publish critic text"})))

    def test_every_paragraph_checked_without_reverse_goal_lists(self):
        draft = TEXT + "\n\n须人工复核正式程序。"
        value = response(check(), check(unit_id="u2", sources=["policy:safety"], claims=["policy"]))
        self.assertIn("须人工复核", validate(value, draft=draft)["text"])
        with self.assertRaises(protocol.AnswerReviewError):
            validate(response(), draft=draft)

    def test_nonfactual_heading_needs_review_but_not_invented_source(self):
        value = response(check(unit_id="u1", sources=[], claims=["presentation"]), check(unit_id="u2"))
        self.assertIn("几点说明：", validate(value, draft="几点说明：\n" + TEXT)["text"])
        value["units"][0]["issue"] = "这一行夹带了未核验的工程断言。"
        with self.assertRaises(protocol.AnswerReviewError):
            validate(value, draft="几点说明：\n" + TEXT)

    def test_unknown_duplicate_units_goals_and_status_fail(self):
        for value in (response(check(unit_id="u99")), response(check(), check()), response(check(unit_id=True)),
                      response(status="looks_good")):
            with self.assertRaises(protocol.AnswerReviewError):
                validate(value)
        value = response()
        value["goals"][0]["goal_id"] = "q9"
        with self.assertRaises(protocol.AnswerReviewError):
            validate(value)

    def test_unknown_unapproved_or_hidden_sources_fail(self):
        for sid in ("e99:/content", "e1:/hidden", True, 1):
            with self.subTest(sid=sid), self.assertRaises(protocol.AnswerReviewError):
                validate(response(check(sources=[sid])))
        state = state_for()
        state["safety_decision"]["approved_evidence_ids"] = []
        with self.assertRaises(protocol.AnswerReviewError):
            validate(response(), state)
        self.assertEqual(TEXT, prepared(state_for())[0]["sources"][SOURCE]["quote"])

    def test_inline_citation_must_match_selected_record(self):
        # Draft citation numbers are provisional and may be stale after a
        # reranked/revision evidence window.  The critic's selected handles
        # own the final citation mapping, so both markers are replaced.
        for draft in (TEXT + "[^99]", TEXT + "[^0]"):
            self.assertEqual(TEXT + "[^1]", validate(response(), draft=draft)["text"])
        self.assertEqual(TEXT + "[^1]", validate(response(), draft=TEXT + "[^1]")["text"])

    def test_critic_owns_final_citations_not_two_independent_numbering_contracts(self):
        state = state_for()
        item = copy.deepcopy(state["evidence_registry"][0])
        item.update(evidence_id="neighbor", record_id="ROW-947")
        state["evidence_registry"].append(item)
        state["context_evidence_ids"].append("neighbor")
        state["safety_decision"]["approved_evidence_ids"].append("neighbor")
        self.assertEqual("ROW-731 " + TEXT + "[^1]", validate(response(), state, draft="ROW-731 " + TEXT + "[^2]")["text"])
        with self.assertRaises(protocol.AnswerReviewError):
            validate(response(), state, draft="ROW-947 " + TEXT + "[^2]")

    def test_explicit_source_owner_matches_bound_document_type(self):
        with self.assertRaises(protocol.AnswerReviewError):
            validate(response(check(attributed_doc_types=["mfl"])), draft="维修清单记录：" + TEXT)
        self.assertEqual("full", validate(response(check(attributed_doc_types=["fmeca"])), draft="FMECA 记录：" + TEXT)["status"])

    def test_neighbor_row_with_identical_words_is_not_same_record(self):
        state = state_for()
        item = copy.deepcopy(state["evidence_registry"][0])
        item.update(record_id="ROW-947", evidence_id="neighbor")
        state["evidence_registry"].append(item)
        state["context_evidence_ids"].append("neighbor")
        state["safety_decision"]["approved_evidence_ids"].append("neighbor")
        with self.assertRaises(protocol.AnswerReviewError):
            validate(response(), state, draft="ROW-947 " + TEXT)

    def test_user_grounded_identity_adds_its_offered_owner_handle(self):
        ledger = {
            1: {"record_id": "MOD-GPP-MSM"},
            2: {"record_id": "FUNC-230"},
        }
        offered = {
            "e1:/record_id": {"quote": "MOD-GPP-MSM"},
            "e2:/record_id": {"quote": "FUNC-230"},
        }
        selected = protocol._augment_exact_identity_sources(
            "MOD-GPP-MSM " + TEXT,
            ["e2:/record_id"],
            ledger,
            offered,
            "MOD-GPP-MSM 为什么重要？",
        )
        self.assertEqual(["e2:/record_id", "e1:/record_id"], selected)
        # An ungrounded neighboring ID is not mechanically promoted.
        self.assertEqual(
            ["e1:/record_id"],
            protocol._augment_exact_identity_sources(
                "FUNC-230 " + TEXT, ["e1:/record_id"], ledger, offered,
                "MOD-GPP-MSM 为什么重要？",
            ),
        )

    def test_random_exact_ids_not_fixed_test_values(self):
        rng = random.Random(9843)
        for _ in range(12):
            code = f"ROW-{rng.randrange(10000, 99999)}"
            state = state_for(record_id=code)
            self.assertIn(code, validate(response(), state, draft=code + TEXT)["text"])
            with self.assertRaises(protocol.AnswerReviewError):
                validate(response(), state, draft=code + "9" + TEXT)

    def test_unknown_api_code_leading_zero_and_wildcard_identity(self):
        state = state_for(record_id="87654321**0001")
        self.assertIn("87654321**0001", validate(response(), state, draft="87654321**0001 " + TEXT)["text"])
        for draft in ("87654321**001 " + TEXT, "ROW-947 " + TEXT, "调用 GetUnrecordedStatus。", "报码 76543210**0001。"):
            with self.subTest(draft=draft), self.assertRaises(protocol.AnswerReviewError):
                validate(response(), state, draft=draft)

    def test_hmc_fault_cannot_borrow_hidden_maintenance(self):
        state = state_for("hmc", metadata={"answer_fields": {"fault_symptom": "钳位", "maintenance_action": "更换板卡"}})
        with self.assertRaises(protocol.AnswerReviewError):
            validate(response(check(sources=["e1:/answer_fields/maintenance_action"])), state, draft="更换板卡。")

    def test_semantic_feedback_points_to_unit_and_omitted_goal(self):
        value = response(check(sources=[], claims=[], issue="把另一行检测方法当成当前故障处理。"))
        value["goals"][0]["issue"] = "未说明排除板卡的观测依据。"
        with self.assertRaises(protocol.AnswerReviewError) as caught:
            validate(value)
        self.assertEqual(["u1", "q0"], [f["location"] for f in caught.exception.feedback])
        self.assertIn("排除板卡", caught.exception.feedback[1]["detail"])

    def test_disclaimer_is_never_exempt_from_review(self):
        value = response(check(sources=["policy:safety"], claims=["policy"], issue="免责句夹带无依据的故障结论。"))
        with self.assertRaises(protocol.AnswerReviewError):
            validate(value, draft="仅供参考；该板已经确定损坏。")
        value["units"][0]["issue"] = ""
        value["goals"][0]["issue"] = "只有免责声明，未回应所问影响。"
        with self.assertRaises(protocol.AnswerReviewError):
            validate(value, draft="仅供参考。")

    def test_history_cannot_prove_current_fact(self):
        state = state_for()
        state["messages"] = [{"role": "assistant", "content": TEXT}]
        with self.assertRaises(protocol.AnswerReviewError):
            validate(response(check(sources=["history:h0"])), state)
        self.assertNotIn("[^", validate(response(check(sources=["history:h0"], claims=["conversation"])),
                         state, draft="此前提到过输出更新时间滞后。")["text"])

    def test_long_history_binding_does_not_retruncate_selected_identity(self):
        state = state_for()
        state["messages"] = [{"role": "assistant", "content": "很长的背景。" * 350 + " ROW-845。"}]
        value = response(check(sources=["history:h0"], claims=["conversation"]))
        self.assertIn("ROW-845", validate(value, state, draft="此前讨论过 ROW-845。")["text"])

    def test_nonengineering_sources_cannot_replace_each_other(self):
        for claim in ("query_status", "conversation", "source_fact", "formal_topology"):
            with self.subTest(claim=claim), self.assertRaises(protocol.AnswerReviewError):
                validate(response(check(sources=["policy:safety"], claims=[claim])))

    def test_scoped_negative_boundary_can_be_full_or_partial(self):
        state = state_for()
        state["requested_source_tools"] = ["fault_history_query"]
        material, _ = prepared(state)
        sid = next(s for s, v in material["sources"].items()
                   if v.get("category") == "query_coverage" and v["quote"] == "not_queried")
        value = response(check(sources=[sid], claims=["query_status"]))
        self.assertEqual("full", validate(value, state, draft="本轮尚未查询历史资料。")["status"])
        value["goals"][0]["status"] = "partial"
        self.assertEqual("partial", validate(value, state, draft="本轮尚未查询历史资料。")["status"])

    def test_unconfirmed_relation_does_not_establish_source_absence(self):
        state = state_for()
        state["hmc_mfl_binding"] = {"required": True, "bound": False, "hmc_codes": ["87654321**0001"]}
        relation = next(o for o in answer._answer_review_material(state)[2] if o["category"] == "source_relation")
        self.assertEqual("false", relation["fields"]["/bound"])
        self.assertIn("不证明原文", relation["fields"]["/meaning"])

    def test_selected_continuation_identity_can_bind_query_boundary_only(self):
        state = state_for()
        state["selected_conversation_anchors"] = [{
            "anchor_id": "anchor-1",
            "doc_type": "fmeca",
            "record_id": "ROW-731",
            "title": "本轮对象",
            "answer_fields": {"record_id": "ROW-731"},
        }]
        material, _ = prepared(state)
        target = next(
            sid for sid, source in material["sources"].items()
            if source.get("category") == "target_binding"
        )
        value = response(check(sources=[target], claims=["query_status"]))
        self.assertEqual(
            "full",
            validate(value, state, draft="当前目标 ROW-731 的本轮查询边界已明确。") ["status"],
        )

    def test_resource_bounds(self):
        for draft in ("字" * 4001, "\n".join([TEXT] * 41)):
            with self.assertRaises(protocol.AnswerReviewError):
                validate(response(), draft=draft)

    def test_footnotes_are_reviewed_not_rejected_or_deleted_by_format(self):
        draft = TEXT + "[^1]\n\n[^1]: 记录来源说明。"
        material, _ = prepared(state_for(), draft)
        self.assertEqual(2, len(material["units"]))
        value = response(check(), check(unit_id="u2", issue="来源说明不能由该字段证明。"))
        with self.assertRaises(protocol.AnswerReviewError) as caught:
            validate(value, draft=draft)
        self.assertEqual("u2", caught.exception.feedback[0]["location"])

    def test_exact_source_definition_keeps_its_label_and_cannot_invent_locator(self):
        state = state_for(source_ref="source.docx#table=2;row=3")
        draft = TEXT + "[^1]\n\n[^1]: source.docx#table=2;row=3"
        value = response(check(), check(unit_id="u2", sources=[], claims=["presentation"]))
        self.assertEqual(draft, validate(value, state, draft)["text"])
        with self.assertRaises(protocol.AnswerReviewError):
            validate(value, state, draft.replace("row=3", "row=4"))

    def test_prior_delivery_is_preserved_but_not_an_engineering_source(self):
        state = state_for()
        state["messages"] = [{"role": "assistant", "content": TEXT, "answer_delivery_manifest": [
            {"kind": "goal_delivery", "goal": "此前所问", "status": "answered", "evidence_ids": ["old-evidence"]}]}]
        source = prepared(state)[0]["sources"]["history:h0"]
        self.assertEqual("此前所问", source["prior_delivery"][0]["goal"])
        with self.assertRaises(protocol.AnswerReviewError):
            validate(response(check(sources=["history:h0"])), state)

    def test_unique_diagnosis_requires_permission_for_any_wording(self):
        for draft in ("仅此一种可能。", "问题已锁定。", "No alternatives remain."):
            with self.assertRaises(protocol.AnswerReviewError):
                validate(response(check(claims=["unique_diagnosis"])), draft=draft)

    def test_all_formal_claims_respect_blocked_and_unvalidated_evidence(self):
        for claim in ("formal_topology", "formal_calculation", "formal_numeric_result", "unique_diagnosis"):
            for status in ("provisional", "quarantined"):
                state = state_for(review_status=status)
                state["safety_decision"]["can_claim_unique"] = True
                with self.subTest(claim=claim, status=status), self.assertRaises(protocol.AnswerReviewError):
                    validate(response(check(claims=[claim])), state)
            state = state_for()
            state["safety_decision"]["blocked_claims"] = [claim]
            with self.assertRaises(protocol.AnswerReviewError):
                validate(response(check(claims=[claim])), state)


class CalculationReviewTests(unittest.TestCase):
    def calculation(self, doc_type="rbd"):
        state = state_for(doc_type, authority="deterministic_derivation", claim_type=doc_type + "_calculation",
            metadata={"answer_fields": {}, "calculation": {"reliability": 0.91827}, "numeric_complete": True})
        state["calculation_requests"] = ["rbd_reliability" if doc_type == "rbd" else "fta_probability"]
        return state, response(check(sources=["e1:/calculation/reliability"],
                                     claims=["formal_calculation", "formal_numeric_result"]))

    def test_authorized_deterministic_result_and_equivalent_scientific_notation(self):
        for doc_type in ("rbd", "fta"):
            state, value = self.calculation(doc_type)
            for number in ("0.91827", "9.1827e-1"):
                self.assertEqual("full", validate(value, state, draft=f"结果 {number}。")["status"])

    def test_missing_wrong_cutset_only_authorization_and_hidden_source_fact_fail(self):
        for doc_type, auth in (("rbd", []), ("rbd", ["fta_probability"]), ("fta", ["fta_cut_sets"])):
            state, value = self.calculation(doc_type)
            state["calculation_requests"] = auth
            with self.assertRaises(protocol.AnswerReviewError):
                validate(value, state, draft="结果 0.91827。")
        state, value = self.calculation()
        state["calculation_requests"] = []
        value["units"][0]["claims"] = ["source_fact"]
        with self.assertRaises(protocol.AnswerReviewError):
            validate(value, state, draft="结果 0.91827。")

    def test_cutsets_have_their_separate_permission(self):
        state, value = self.calculation("fta")
        state["calculation_requests"] = ["fta_cut_sets"]
        state["evidence_registry"][0]["metadata"]["calculation"] = {"minimal_cut_sets": [["EVENT-81"]]}
        value["units"][0].update(sources=["e1:/calculation/minimal_cut_sets/0/0"], claims=["formal_calculation"])
        self.assertEqual("full", validate(value, state, draft="最小割集为 EVENT-81。")["status"])

    def test_incomplete_numeric_and_missing_calculation_claim_fail(self):
        state, value = self.calculation()
        state["evidence_registry"][0]["metadata"]["numeric_complete"] = False
        with self.assertRaises(protocol.AnswerReviewError):
            validate(value, state, draft="结果 0.91827。")
        state, value = self.calculation()
        value["units"][0]["claims"] = ["formal_numeric_result"]
        with self.assertRaises(protocol.AnswerReviewError):
            validate(value, state, draft="结果 0.91827。")

    def test_invented_prefix_integer_or_exponent_cannot_borrow_true_number(self):
        state, value = self.calculation()
        for number in ("0.99999", "0.91", "1", "9.1827e-2"):
            with self.subTest(number=number), self.assertRaises(protocol.AnswerReviewError):
                validate(value, state, draft=f"结果 {number}。")
        with self.assertRaises(protocol.AnswerReviewError):
            validate(response(check(claims=["formal_calculation"])))


class AnswerEntryTests(unittest.TestCase):
    def test_review_schema_can_close_request_local_source_handles(self):
        schema = protocol.review_schema(["e1:/answer_fields/local_impact", "policy:safety"])
        source_items = schema["properties"]["units"]["items"]["properties"]["sources"]["items"]
        self.assertEqual(
            ["e1:/answer_fields/local_impact", "policy:safety"],
            source_items["enum"],
        )

    def test_normal_path_two_calls_with_smaller_separate_review_budget(self):
        result, calls, _ = node_run()
        self.assertEqual(2, len(calls))
        self.assertEqual(TEXT + "[^1]", result["final_answer"])
        self.assertEqual(1200, calls[0].args[1]["num_predict"])
        self.assertEqual(4096, calls[1].args[1]["num_predict"])
        self.assertEqual(4096, calls[1].args[1]["max_tokens"])
        review_format = calls[1].args[1]["format"]
        self.assertEqual(2, review_format["properties"]["version"]["enum"][0])
        source_items = review_format["properties"]["units"]["items"]["properties"]["sources"]["items"]
        self.assertIn(SOURCE, source_items["enum"])

    def test_feedback_drives_revision_and_new_text_is_reviewed(self):
        feedback = response(check(sources=[], claims=[], issue="不能把延迟说成完全丧失；须解释证据界限。"))
        result, calls, capture = node_run(review=feedback, draft="输出完全丧失。", revision=TEXT, observed=True)
        self.assertEqual(4, len(calls))
        self.assertIn("不能把延迟说成完全丧失", calls[2].args[0][0]["content"])
        self.assertIn(TEXT, calls[3].args[0][0]["content"])
        review_prompt = calls[3].args[0][0]["content"]
        self.assertIn("previous_validation_feedback", review_prompt)
        self.assertIn("不能把延迟说成完全丧失", review_prompt)
        self.assertEqual(TEXT + "[^1]", result["final_answer"])
        self.assertEqual(2, sum(s["stage"] == "review_proposal" for s in capture.stages))

    def test_revised_error_cannot_inherit_approval_or_trigger_unbounded_revision(self):
        queue = asyncio.Queue()
        feedback = response(check(issue="该事实无依据。"))
        with self.assertRaises(answer.AnswerGenerationError):
            node_run(review=feedback, revision="ROW-999 被确认。", second_review=response(),
                     third_revision="ROW-998 仍未被当前证据支持。", third_review=response(),
                     config={"configurable": {"stream_queue": queue}})
        self.assertTrue(queue.empty())

    def test_unsafe_unit_cannot_be_silently_removed_to_publish_rest(self):
        value = response(check(), check(unit_id="u2", issue="结论无依据。"))
        with self.assertRaises(answer.AnswerGenerationError):
            node_run(review=value, draft=TEXT + "\n其他结论。", revision=TEXT + "\n其他结论。", second_review=value,
                     third_revision=TEXT + "\n其他结论。", third_review=value)

    def test_transport_timeout_not_retried_without_actionable_feedback(self):
        with patch.object(answer, "_review_answer", new=AsyncMock(side_effect=asyncio.TimeoutError)) as review:
            with self.assertRaises(answer.AnswerGenerationError):
                node_run()
        review.assert_awaited_once()

    def test_total_deadline_also_bounds_transport(self):
        async def slow(*args, **kwargs):
            await asyncio.sleep(.05)
            return TEXT
        with patch.object(answer, "ANSWER_TOTAL_TIMEOUT_SECONDS", .01), patch.object(
                answer, "_generate_answer_response", new=AsyncMock(side_effect=slow)) as writer:
            with self.assertRaises(answer.AnswerGenerationError):
                node_run()
        self.assertLessEqual(writer.call_args.args[1]["timeout"], .01)

    def test_cancel_propagates_without_publication(self):
        queue = asyncio.Queue()
        with patch.object(answer, "_review_answer", new=AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):
                node_run(config={"configurable": {"stream_queue": queue}})
        self.assertTrue(queue.empty())

    def test_observation_and_pure_publication_preserve_body(self):
        plain, _, _ = node_run()
        observed, _, captured = node_run(observed=True)
        self.assertEqual(plain, observed)
        stages = {s["stage"]: s["answer"] for s in captured.stages}
        self.assertEqual(stages["integrated_review"], stages["before_public_compaction"])
        self.assertEqual(stages["before_public_compaction"], stages["after_public_compaction"])

    def test_verified_sse_published_once(self):
        queue = asyncio.Queue()
        with patch.object(answer, "_generate_answer_response", new=AsyncMock(return_value=TEXT)), patch.object(
            answer.llm_service, "call_model", return_value=json.dumps(response())), patch.object(
            answer.agent_model_role_service, "apply_agent_llm_params", return_value=None):
            result = asyncio.run(answer.answer_node(state_for(), {"configurable": {"stream_queue": queue}}))
        self.assertEqual({"type": "answer_delta", "content": result["final_answer"]}, queue.get_nowait())
        self.assertTrue(queue.empty())

    def test_all_modes_and_legacy_fields_share_one_entry(self):
        for mode in ("fault_diagnosis", "architecture_knowledge", "exact_code_lookup", "maintenance_guidance", "knowledge_explanation"):
            state = state_for()
            state.update(answer_mode=mode, query_frame={"required_answer_fields": ["fault_cause", "history_steps"]})
            result, calls, _ = node_run(state)
            self.assertEqual(2, len(calls))
            self.assertEqual([], result["delivered_answer_fields"])

    def test_original_followup_and_history_reach_both_models(self):
        state = state_for()
        state["query"] = "什么观察能推翻刚才的解释？"
        state["messages"] = [{"role": "assistant", "content": "前面只描述影响，没有判断原因。"}]
        _, calls, _ = node_run(state)
        for call in calls:
            self.assertIn(state["query"], call.args[0][0]["content"])
            self.assertIn(state["messages"][0]["content"], call.args[0][0]["content"])

    def test_old_editors_and_body_salvage_stay_retired(self):
        for name in ("_review_answer_target_completeness", "_review_answer_fault_completeness", "_review_answer_history",
                     "_review_answer_calculations", "_repair_answer_citations", "_incremental_answer_fallback"):
            self.assertFalse(hasattr(answer, name), name)

    def test_legacy_retrieval_keeps_verifiable_sources(self):
        state = {"query": "说明此记录", "vector_results": [{"doc_id": "legacy", "content": TEXT}]}
        result, _, _ = node_run(state, response(check(sources=["e1:/content"])))
        self.assertEqual("legacy", result["citations"][0]["doc_id"])

    def test_public_trace_never_contains_private_feedback(self):
        tracing.begin_turn("review-test", "问题")
        try:
            node_run(review=response(check(issue="私有工程草稿不得进入公开 trace")), revision=TEXT)
            events = tracing.peek().events
            self.assertEqual(["rejected", "accepted"], [e["outcome"] for e in events if e["type"] == "answer_review"])
            self.assertNotIn("私有工程草稿", json.dumps(events, ensure_ascii=False))
        finally:
            tracing._current_trace.set(None)


if __name__ == "__main__":
    unittest.main()
