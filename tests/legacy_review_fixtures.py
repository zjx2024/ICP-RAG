# Historical v3/v4 fixture tests; not evidence that the serving v5 review passes.
"""Read-only critique/one revision. Fixtures test contracts, not model quality."""
import asyncio
import copy
import json
import random
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

from backend.agent import answer_observation, tracing
from scripts import legacy_answer_review as protocol
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
    return {"version": 3, "units": list(units) or [check()],
            "goals": [{"goal_id": "q0", "status": status, "unit_ids": [u["unit_id"] for u in units] or ["u1"], "issue": ""}]}


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


class ReviewProtocolTests(unittest.TestCase):
    def test_valid_and_rephrased_answers_preserve_writer_text(self):
        for draft in (TEXT, "数据新鲜度因此下降。", "The output is no longer fresh.", "使用方拿到的是较早时刻的数值。"):
            result = validate(response(), draft=draft)
            self.assertEqual(draft + "[^1]", result["text"])
            self.assertEqual("full", result["status"])
            self.assertEqual("goal", result["manifest"][0]["evidence_scope"])

    def test_lossy_planning_cannot_drop_whole_original_question(self):
        state = state_for()
        state["dialogue_context"] = {"status": "resolved", "requested_facets": ["未收录于字段目录的新目标"]}
        material, ledger = prepared(state)
        self.assertEqual({"q0": state["query"]}, material["goals"])
        self.assertEqual(["未收录于字段目录的新目标"], protocol.review_input(material, ledger)["planning_hints"])
        self.assertEqual("full", validate(response(), state)["status"])
        value = response()
        value["goals"][0]["issue"] = "只列出了影响，未回答原问题中的重要性和首先受影响的对象。"
        with self.assertRaises(protocol.AnswerReviewError) as caught:
            validate(value, state)
        self.assertEqual("q0", caught.exception.feedback[0]["location"])

    def test_planner_cannot_turn_constraints_into_additional_delivery_requirements(self):
        for question, hint in (
            ("使用已核验数据比较两方案，哪种更可靠？", "逐项列出所有参数"),
            ("只按现行版本解释这个接口，不混其他构型。", "介绍所有构型及版本历史"),
            ("Use the validated sources to explain the cause.", "Enumerate every source field"),
        ):
            state = state_for()
            state.update(query=question, dialogue_context={"status": "resolved", "requested_facets": [hint]})
            material, _ = prepared(state)
            self.assertEqual({"q0": question}, material["goals"])
            self.assertEqual([hint], material["planning_hints"])
            value = response()
            value["goals"][0]["issue"] = "本段使用了其他构型或未经验证的参数，违反原始问题的条件。"
            with self.assertRaises(protocol.AnswerReviewError):
                validate(value, state)

    def test_explicit_detail_request_stays_in_original_goal_even_if_planner_omits_it(self):
        state = state_for()
        state.update(query="比较这两项，同时列出每个参数及其单位和验证依据。",
                     dialogue_context={"status": "resolved", "requested_facets": ["比较这两项"]})
        material, _ = prepared(state)
        self.assertEqual(state["query"], material["goals"]["q0"])
        value = response()
        value["goals"][0]["issue"] = "用户明确要求的参数、单位、验证依据尚未回答。"
        with self.assertRaises(protocol.AnswerReviewError):
            validate(value, state)

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
        material, _ = prepared(state_for(), "字" * 4001)
        self.assertEqual(4001, len(material["units"][0]["text"]))
        with self.assertRaises(protocol.AnswerReviewError):
            validate(response(), draft="\n".join([TEXT] * 41))

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

    def test_numeric_enum_implies_calculation_but_incomplete_values_still_fail(self):
        state, value = self.calculation()
        state["evidence_registry"][0]["metadata"]["numeric_complete"] = False
        with self.assertRaises(protocol.AnswerReviewError):
            validate(value, state, draft="结果 0.91827。")
        state, value = self.calculation()
        value["units"][0]["claims"] = ["formal_numeric_result"]
        self.assertEqual("full", validate(value, state, draft="结果 0.91827。")["status"])

    def test_invented_prefix_integer_or_exponent_cannot_borrow_true_number(self):
        state, value = self.calculation()
        for number in ("0.99999", "0.91", "1", "9.1827e-2"):
            with self.subTest(number=number), self.assertRaises(protocol.AnswerReviewError):
                validate(value, state, draft=f"结果 {number}。")
        with self.assertRaises(protocol.AnswerReviewError):
            validate(response(check(claims=["formal_calculation"])))

    def test_canonical_display_and_invocation_parameters_keep_exact_provenance(self):
        state, value = self.calculation()
        state["evidence_registry"][0]["metadata"]["calculation"] = {
            "results": {"system_reliability": 0.987654321123},
            "parameters": {"mission_time": 17.0, "mission_time_unit": "h"},
            "units": {"system_reliability": "1"}}
        value["units"][0]["sources"] = ["e1:/calculation/results/system_reliability"]
        for draft in ("17 小时可靠度为 0.98765432。", "2. 17 h: 0.98765432。", "结果 0.987654321123。"):
            self.assertEqual("full", validate(value, state, draft=draft)["status"])
        for draft in ("18 小时可靠度为 0.98765432。", "结果 0.9876543。", "结果 1。", "结果 0.98765433。"):
            with self.subTest(draft=draft), self.assertRaises(protocol.AnswerReviewError):
                validate(value, state, draft=draft)
        neighbour = copy.deepcopy(state["evidence_registry"][0])
        neighbour.update(evidence_id="other-model", record_id="ROW-999")
        neighbour["metadata"]["calculation"]["results"]["system_reliability"] = 0.87654321
        state["evidence_registry"].append(neighbour)
        state["context_evidence_ids"].append(neighbour["evidence_id"])
        state["safety_decision"]["approved_evidence_ids"].append(neighbour["evidence_id"])
        with self.assertRaises(protocol.AnswerReviewError):
            validate(value, state, draft="结果 0.87654321。")
