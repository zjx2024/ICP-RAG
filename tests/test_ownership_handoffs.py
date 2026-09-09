# v3/v4 validator assertions below are offline compatibility; live v5 counterparts are in test_answer_integrated_review.
"""Responsibility replacements: ordinary paths and adversarial neighbours.

These are protocol/unit tests, not claims about end-to-end model quality.
"""
import asyncio
import copy
import json
import random
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

from scripts import legacy_answer_review as answer_review
from backend.agent.current_model_selection import model_content_evidence, model_candidates, validate_selection
from backend.agent.contracts import make_evidence
from backend.agent.nodes.evidence_gate import evidence_gate_node
from backend.agent.nodes import answer
from backend.agent.nodes.answer_source_fields import source_owned_field_view
from backend.agent.nodes.answer_request import render_clarification_only_answer
from backend.agent.tool_policy import normalize_requested_sources, resolve_tool_policy
from backend.agent.tools.icp_diagnosis import _hydrate_record_search_hits
from backend.agent.tools.icp_calculation import icp_calculation_node
from backend.icp.reliability import calculate_fta, calculate_rbd
from tests.legacy_review_fixtures import check, prepared, response, state_for, validate


class SourceVisibilityTests(unittest.TestCase):
    def test_reference_is_not_an_exclusive_source_request(self):
        for wording in ("请参考那份分析表，再说明验证和处置", "Consult that analysis and explain verification and repair"):
            request = {"doc_type": "fmeca", "basis": wording, "question_role": "evidence",
                       "requested_fields": ["detection_method", "maintenance_action"]}
            decision = resolve_tool_policy(intent="maintenance_query", raw_user_query=wording,
                requested_sources=[request], required_answer_fields=request["requested_fields"])
            self.assertIn("mfl_query", decision["authorized_tools"])
            self.assertIn("fmeca_query", decision["authorized_tools"])
            item = {"doc_type": "hmc", "metadata": {"answer_fields": {"maintenance_action": "核对原始状态"}}}
            state = {"query_frame": {"source_obligations": decision["source_obligations"]}}
            self.assertIs(item, source_owned_field_view(item, state))

    def test_exclusive_scope_survives_policy_and_view_without_keyword_classifier(self):
        for wording in ("范围以此表为限", "Limit the evidence to document A", "仅凭这一来源作答"):
            request = {"doc_type": "fmeca", "basis": wording, "scope": "exclusive", "scope_basis": wording,
                       "question_role": "evidence", "requested_fields": ["maintenance_action"]}
            decision = resolve_tool_policy(intent="maintenance_query", raw_user_query=wording,
                requested_sources=[request], required_answer_fields=["maintenance_action"])
            self.assertNotIn("mfl_query", decision["authorized_tools"])
            self.assertEqual("exclusive", decision["source_obligations"][0]["scope"])

    def test_reference_duplicate_cannot_hide_a_later_exclusive_scope(self):
        query = "Use only this source."
        reference = {"doc_type": "mfl", "question_role": "evidence", "basis": query}
        exclusive = {**reference, "scope": "exclusive", "scope_basis": query}
        for values in ([reference, exclusive], [exclusive, reference]):
            requests = normalize_requested_sources(values, raw_user_query=query)
            self.assertEqual(2, len(requests))
            self.assertEqual(1, sum(item.get("scope") == "exclusive" for item in requests))

    def test_hydrate_every_unique_locator_in_bounded_window_despite_rank(self):
        for seed in range(4):
            hits = [{"doc_type_kwd": "fmeca", "doc_id": "document-81", "pg_id_kwd": str(n),
                     "content": "untrusted indexed payload", "score": 1 / (n + 1)} for n in range(8)]
            random.Random(seed).shuffle(hits)
            hits.insert(1, dict(hits[0]))
            def read(_type, locators, *, limit):
                return [{"doc_id": item["doc_id"], "id": item["pg_id_kwd"], "fault_mode": "PG fact"}
                        for item in locators[:limit]]
            with patch("backend.agent.tools.icp_diagnosis.icp_repository.get_records_by_storage_locators", side_effect=read) as query:
                rows = _hydrate_record_search_hits(hits, limit_per_type=8)["fmeca_query"]
            self.assertEqual({str(n) for n in range(8)}, {row["id"] for row in rows})
            self.assertEqual(8, len(query.call_args.args[1]))
            self.assertNotIn("untrusted", json.dumps(rows))


class ReviewerHandoffTests(unittest.TestCase):
    def test_product_rule_provenance_is_carried_without_a_second_handle_copy(self):
        value = response(check(sources=[], claims=["policy"]))
        self.assertEqual("full", validate(value, draft="辅助分析不能替代正式维修程序。")["status"])
        with self.assertRaises(answer_review.AnswerReviewError):
            validate(value, draft="可立即更换 MOD-OTHER-93。")

    def test_unbound_secondary_model_does_not_block_an_independent_calculation(self):
        source = make_evidence(source_kind="postgres", authority="validated_relation", evidence_level="L2_CORROBORATED",
            claim_type="fta_structure", doc_type="fta", doc_id="doc-83", record_id="FTA-X9", title="模型甲",
            content="AND(a,b)", review_status="auto_validated", metadata={"is_active": True,
                "projection_status": "published", "validation_status": "auto_validated", "revision_id": "revision-83"})
        result = make_evidence(source_kind="calculation", authority="deterministic_derivation", evidence_level="L2_CORROBORATED",
            claim_type="fta_calculation", doc_type="fta", doc_id="doc-83", record_id="revision-83", title="模型甲计算",
            content="cutsets", derived_from=[source["evidence_id"]])
        query = "Calculate minimal cut sets from the selected tree."
        state = {"query": query, "answer_mode": "knowledge_explanation", "evidence_registry": [source, result],
                 "allowed_tools": ["fta_query", "fta_analyzer", "rbd_query"],
                 "calculation_authorizations": [{"operation": "fta_cut_sets", "mode": "calculate", "target": "fta", "basis": query}],
                 "rbd_query_binding": {"required": True, "bound": False}}
        decision = asyncio.run(evidence_gate_node(state))["safety_decision"]
        self.assertIn(result["evidence_id"], decision["approved_evidence_ids"])
        self.assertNotIn("formal_calculation", decision["blocked_claims"])
        self.assertNotIn("formal_topology", decision["blocked_claims"])
        self.assertIn("rbd_model_identity", decision["blocked_claims"])

    def test_bound_field_carries_its_own_record_identity_for_new_questions(self):
        state = state_for()
        result = validate(response(check()), state, draft="ROW-731 的输出更新时间滞后。")
        self.assertEqual("full", result["status"])

    def test_server_carried_identity_does_not_consume_the_model_proposal_budget(self):
        state = state_for()
        state["evidence_registry"][0]["metadata"]["answer_fields"] = {f"field_{n}": f"事实 {n}" for n in range(24)}
        sources = [f"e1:/answer_fields/field_{n}" for n in range(24)]
        draft = "ROW-731 的相关字段已核对。"
        self.assertEqual("full", validate(response(check(sources=sources)), state, draft=draft)["status"])
        with self.assertRaises(answer_review.AnswerReviewError):
            validate(response(check(sources=sources + ["e1:/invented_field"])), state, draft=draft)

    def test_explicit_reference_in_selected_excerpt_is_not_a_fuzzy_join(self):
        state = state_for()
        state["evidence_registry"][0]["metadata"]["answer_fields"]["local_impact"] = "原记录明确引用 ROW-999。"
        neighbour = copy.deepcopy(state["evidence_registry"][0])
        neighbour.update(evidence_id="neighbour", record_id="ROW-999")
        state["evidence_registry"].append(neighbour)
        state["safety_decision"]["approved_evidence_ids"].append("neighbour")
        state["context_evidence_ids"].append("neighbour")
        result = validate(response(check()), state, draft="该字段明确引用 ROW-999。")
        self.assertNotIn("neighbour", result["manifest"][0]["evidence_ids"])
        state["evidence_registry"][0]["metadata"]["answer_fields"]["local_impact"] = "没有其他记录标识。"
        with self.assertRaises(answer_review.AnswerReviewError):
            validate(response(check()), state, draft="该字段明确引用 ROW-999。")

    def test_cutset_only_does_not_execute_probability_or_expose_it(self):
        graph = {"root_fault_id": "top", "vertices": [
            {"id": "top", "label": "TopEvent", "properties": {"gate_type": "OR"}},
            {"id": "leaf", "label": "BasicEvent", "properties": {"name": "EVENT-Z9", "probability": 0.17}},
        ], "edges": [{"source": "top", "target": "leaf"}]}
        with patch("backend.icp.reliability._node_probability", side_effect=AssertionError("Not requested")):
            result = calculate_fta(graph, include_probability=False)
        self.assertEqual([["EVENT-Z9"]], result["minimal_cut_sets"])
        self.assertNotIn("top_event_probability", result)

        row = {"doc_id": "doc-83", "revision_id": "revision-83", "diagram_type": "fta", "graph": graph,
               "is_active": True, "projection_status": "published", "review_status": "confirmed"}
        parent = make_evidence(source_kind="postgres", authority="confirmed_relation", evidence_level="L1_DIRECT",
            claim_type="fta_structure", doc_type="fta", doc_id="doc-83", record_id="revision-83",
            review_status="confirmed", scope={"document_number": "DOC-X7"},
            metadata={"is_active": True, "projection_status": "published"})
        query = "Compute minimal cut sets."
        state = {"query": query, "allowed_tools": ["fta_query", "fta_analyzer"], "confirmed_graphs": [row],
                 "evidence_registry": [parent], "calculation_authorizations": [
                     {"operation": "fta_cut_sets", "mode": "calculate", "target": "fta", "basis": query}]}
        output = asyncio.run(icp_calculation_node(state))
        derived = output["evidence_registry"][0]
        self.assertEqual("ok", output["tool_results"][0]["status"])
        self.assertEqual("DOC-X7", derived["scope"]["document_number"])
        self.assertIsNone(derived["metadata"]["numeric_complete"])
        self.assertEqual("not_requested", derived["metadata"]["calculation"]["numeric_status"])
        self.assertEqual({}, derived["metadata"]["calculation"]["parameters"])

    def test_result_retains_consumed_parameters_not_unused_neighbour_inputs(self):
        graph = {"vertices": [{"id": "s", "label": "Start"}, {"id": "e", "label": "End"},
                 {"id": "x", "properties": {"name": "Block X", "reliability": 0.81, "failure_rate": 0.07}},
                 {"id": "y", "properties": {"name": "Block Y", "failure_rate": 0.02}},
                 {"id": "unused", "properties": {"reliability": 0.91}}],
                 "edges": [{"source": "s", "target": "x"}, {"source": "x", "target": "y"},
                           {"source": "y", "target": "e"}]}
        result = calculate_rbd(graph, mission_time=17)
        self.assertEqual({"x", "y"}, set(result["block_parameters"]))
        self.assertEqual("reliability", result["block_parameters"]["x"]["field"])
        self.assertEqual(17, result["block_parameters"]["y"]["mission_time"])


class ModelContentAndClarificationTests(unittest.TestCase):
    def test_current_question_provenance_is_owned_by_server_not_transcribed_by_model(self):
        candidates = model_candidates([{"doc_id": "doc-1", "revision_id": "rev-7", "title": "模型甲"}], "rbd")
        proposal = {"status": "selected", "selections": [{"doc_id": "doc-1", "revision_id": "rev-7", "matched_title": "模型甲"}], "reason": "当前所问"}
        for query in ("按十小时计算它", "Recalculate for a ten-hour mission", "分析这份图"):
            selected = validate_selection(proposal, candidates, query=query)
            self.assertEqual(query, selected["question_basis"])
        # Explicit stale legacy provenance and unoffered identities still fail.
        self.assertEqual({}, validate_selection({**proposal, "question_basis": "改写而非原文"}, candidates, query="当前问题"))
        proposal["selections"][0]["revision_id"] = "invented"
        self.assertEqual({}, validate_selection(proposal, candidates, query="当前问题"))

    def test_operation_fields_do_not_force_a_different_document_family(self):
        for operation, target, source, unwanted in (("fta_cut_sets", "fta", "fta_query", "rbd_query"),
                                                     ("rbd_reliability", "rbd", "rbd_query", "fmeca_query")):
            query = "Execute the requested operation."
            decision = resolve_tool_policy(intent="hybrid", raw_user_query=query,
                required_answer_fields=["calculation_result", "calculation_parameters"],
                calculation_authorizations=[{"operation": operation, "mode": "calculate", "target": target, "basis": query}])
            self.assertIn(source, decision["authorized_tools"])
            self.assertNotIn(unwanted, decision["authorized_tools"])

    def test_formal_model_contents_do_not_authorize_component_topology(self):
        row = {"doc_id": "document-81", "revision_id": "rev-93", "diagram_key": "RBD-K9", "title": "上级模块模型",
               "is_active": True, "projection_status": "published", "validation_status": "auto_validated",
               "graph": {"vertices": [{"id": "node-1", "properties": {"name": "模块 X", "module_id": "MOD-X9"}}]}}
        item = model_content_evidence([row], "rbd")[0]
        self.assertEqual("model_content_only", item["metadata"]["usage_scope"])
        state = state_for()
        state.update(evidence_registry=[item], context_evidence_ids=[item["evidence_id"]],
                     safety_decision={"approved_evidence_ids": [item["evidence_id"]]})
        material, _ = prepared(state, "该模型列有模块 X。")
        source = next(sid for sid in material["sources"] if sid.endswith("/declared_components/0/name"))
        value = response(check(sources=[source]))
        self.assertEqual("full", validate(value, state, draft="该模型列有模块 X。")["status"])
        value["units"][0]["claims"] = ["formal_topology"]
        with self.assertRaises(answer_review.AnswerReviewError):
            validate(value, state, draft="该部件有一条直接路径。")
        for changes in ({"is_active": False}, {"validation_status": "provisional"}, {"projection_status": "pending"}):
            self.assertEqual([], model_content_evidence([{**row, **changes}], "rbd"))

    def test_reason_only_is_not_presented_as_missing_user_input(self):
        from backend.agent.clarification_contract import make_clarification_reason
        reason = make_clarification_reason(kind="authoritative_source_unavailable", source_kind="tool_result",
                                          source_name="hmc_query", blocked_claims=["exact_hmc_fact"])
        text = render_clarification_only_answer({"clarification_reasons": [reason]})
        self.assertIn("权威资料暂不可用", text)
        self.assertNotIn("需要补充以下信息", text)
        self.assertNotIn("exact_hmc_fact", text)


if __name__ == "__main__":
    unittest.main()
