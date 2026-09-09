# v3/v4 validator assertions below are offline compatibility; live v5 counterparts are in test_answer_integrated_review.
import asyncio
import copy
import json
import unittest
from unittest.mock import patch

from backend.agent.answer_observation import observe_answers
from scripts.legacy_answer_review import (
    AnswerReviewError,
    prepare_review,
    review_input,
    review_schema,
    validate_review,
)
from backend.agent.nodes import answer
from backend.core.llm_service import LLMOutputLimitError
from scripts.replay_icp_review import replay


class ReviewSourceAliasTests(unittest.TestCase):
    @staticmethod
    def compact(raw):
        return {"version": 4, "units": {u['unit_id']: [u[k] for k in
                ('sources', 'claims', 'attributed_doc_types', 'issue')] for u in raw['units']},
                "goals": {g['goal_id']: [g[k] for k in ('status', 'unit_ids', 'issue')] for g in raw['goals']}}

    def test_compact_transport_is_lossless_for_acceptance_and_semantic_rejection(self):
        args, raw = self.inputs()
        self.assertEqual(validate_review(json.dumps(raw), **args),
                         validate_review(json.dumps(self.compact(raw)), **args))
        raw['units'][0]['issue'] = 'API不支持该故障模式，不能跨行归属'
        errors = []
        for value in (raw, self.compact(raw)):
            with self.assertRaises(AnswerReviewError) as caught:
                validate_review(json.dumps(value), **args)
            errors.append(caught.exception.feedback)
        self.assertEqual(errors[0], errors[1])

    def test_compact_transport_cannot_omit_goals_units_fields_or_invent_sources(self):
        args, raw = self.inputs()
        good = self.compact(raw)
        malformed = [copy.deepcopy(good) for _ in range(5)]
        malformed[0]['goals'] = {}
        malformed[1]['units'] = {}
        malformed[2]['units']['u1'].pop()
        malformed[3]['units']['u1'][0] = ['s99999']
        malformed[4]['units']['u2'] = malformed[4]['units'].pop('u1')
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(AnswerReviewError):
                validate_review(json.dumps(value), **args)

    def test_compact_schema_closes_coordinates_and_reuses_source_enum(self):
        args, raw = self.inputs()
        schema = review_schema(args['prepared']['sources'], unit_ids=['u1'], goal_ids=['q0'])
        self.assertEqual(['u1'], schema['properties']['units']['required'])
        self.assertEqual('#/definitions/unit', schema['properties']['units']['properties']['u1']['$ref'])
        self.assertEqual(4, schema['definitions']['unit']['minItems'])
        self.assertEqual(4, schema['definitions']['unit']['maxItems'])
        self.assertEqual(4, len(schema['definitions']['unit']['items']))
        self.assertFalse(schema['definitions']['unit']['additionalItems'])
        bad = self.compact(raw)
        bad['units']['u1'].append('extra')
        with self.assertRaises(AnswerReviewError):
            validate_review(json.dumps(bad), **args)

    def test_visible_identity_hints_are_exact_source_options_not_auto_approval(self):
        ledger = {1: {"record_id": "ROW-OTHER-082", "doc_type": "fmeca", "metadata": {
            "answer_fields": {"product_flag": "PCU / FUNC-ALT-073", "configuration": "CFG-OTHER"}}}}
        prepared = prepare_review("**FUNC-ALT-073**", ledger=ledger, conversation={}, observations=[],
                                  task={"original_question": "解释"}, document_semantics="边界",
                                  compact_sources=True)
        offered = review_input(prepared, ledger)
        obligation = offered['units'][0]['identity_bindings'][0]
        self.assertEqual('FUNC-ALT-073', obligation['identity'])
        self.assertEqual(['e1:/answer_fields/product_flag'],
                         [prepared['source_aliases'][sid] for sid in obligation['source_ids']])
        _, raw = self.inputs()
        raw['units'][0].update(sources=['policy:presentation'], claims=['presentation'], attributed_doc_types=[])
        with self.assertRaises(AnswerReviewError) as caught:
            validate_review(json.dumps(raw), prepared=prepared, ledger=ledger, safety={}, calculation_requests=[])
        self.assertTrue(caught.exception.feedback[0]['review_repairable'])
        self.assertEqual('unbound_exact_identity', caught.exception.feedback[0]['code'])

    def test_unknown_identity_is_not_disguised_as_a_source_selection_error(self):
        ledger = {1: {"record_id": "ROW-OTHER-082", "doc_type": "fmeca"}}
        prepared = prepare_review("**FUNC-ALT-074**", ledger=ledger, conversation={}, observations=[],
                                  task={"original_question": "解释"}, document_semantics="边界")
        _, raw = self.inputs()
        raw['units'][0].update(sources=['policy:presentation'], claims=['presentation'], attributed_doc_types=[])
        with self.assertRaises(AnswerReviewError) as caught:
            validate_review(json.dumps(raw), prepared=prepared, ledger=ledger, safety={}, calculation_requests=[])
        self.assertFalse(caught.exception.feedback[0].get('review_repairable', False))
        self.assertEqual([], review_input(prepared, ledger)['units'][0]['identity_bindings'][0]['source_ids'])

    def inputs(self):
        ledger = {1: {"record_id": "FAULT-OTHER-082", "doc_type": "fmeca", "metadata": {
            "answer_fields": {"local_impact": "输出电压下降"}}}}
        prepared = prepare_review("输出电压下降", ledger=ledger, conversation={}, observations=[],
                                  task={"original_question": "影响是什么"}, document_semantics="辅助分析",
                                  compact_sources=True)
        raw = {"version": 3, "units": [{"unit_id": "u1", "sources": ["e1:/answer_fields/local_impact"],
               "claims": ["source_fact"], "attributed_doc_types": ["fmeca"], "issue": ""}],
               "goals": [{"goal_id": "q0", "status": "answered", "unit_ids": ["u1"], "issue": ""}]}
        return {"prepared": prepared, "ledger": ledger, "safety": {}, "calculation_requests": []}, raw

    def test_short_handles_expand_to_identical_verdict_fields_and_citations(self):
        args, raw = self.inputs()
        expected = validate_review(json.dumps(raw), **args)
        inverse = {sid: alias for alias, sid in args["prepared"]["source_aliases"].items()}
        raw["units"][0]["sources"] = [inverse[raw["units"][0]["sources"][0]]]
        self.assertEqual(expected, validate_review(json.dumps(raw), **args))
        offered = review_input(args["prepared"], args["ledger"])["sources"]
        for alias, sid in args["prepared"]["source_aliases"].items():
            self.assertEqual(sid, offered[alias]["canonical_source_id"])
            self.assertEqual(args["prepared"]["sources"][sid]["quote"], offered[alias]["quote"])

    def test_unoffered_handle_or_wrong_kind_cannot_acquire_engineering_authority(self):
        args, raw = self.inputs()
        inverse = {sid: alias for alias, sid in args["prepared"]["source_aliases"].items()}
        for handle in ("s99999", "s0", inverse["policy:safety"]):
            with self.subTest(handle=handle), self.assertRaises(AnswerReviewError):
                proposal = copy.deepcopy(raw)
                proposal["units"][0]["sources"] = [handle]
                validate_review(json.dumps(proposal), **args)

    def test_truncated_review_preserves_exact_private_input_without_validation(self):
        partial = '{"version":3,"units":['
        with observe_answers() as captured, patch.object(
            answer.llm_service, "call_model", side_effect=LLMOutputLimitError("预算耗尽", partial_content=partial)
        ), patch.object(answer, "validate_review") as validator, self.assertRaises(LLMOutputLimitError):
            asyncio.run(answer._review_answer("原始草稿", {"query": "本轮问题"}, {"max_tokens": 4096}))
        validator.assert_not_called()
        snapshot = captured.reviews[0]
        self.assertEqual(partial, snapshot["raw"])
        self.assertEqual("原始草稿", snapshot["prepared"]["draft"])
        self.assertFalse(snapshot["transport"]["validation_executed"])
        result = replay(snapshot)
        self.assertEqual("requires_complete_model_output", result["replay_status"])
        self.assertFalse(result["same_verdict"])

    def test_model_exception_keeps_only_explicit_output_out_of_public_error(self):
        import importlib
        service = importlib.import_module("backend.core.llm_service")
        def limited(*_args):
            service._capture_usage({"done_reason": "length", "eval_count": 4096})
            return "私有未完成审核"
        with patch.object(service.llm_service, "_call_model", side_effect=limited), \
                self.assertRaises(LLMOutputLimitError) as caught:
            service.llm_service.call_model([], {"max_tokens": 4096})
        self.assertEqual("私有未完成审核", caught.exception.partial_content)
        self.assertNotIn("私有未完成审核", str(caught.exception))
