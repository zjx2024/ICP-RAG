"""Offline experimental controls, not semantic quality scores."""
import unittest
import io
import tarfile
import tempfile
import json
import types
from pathlib import Path
from unittest.mock import patch

from backend.agent import answer_review
from backend.agent.citation_binding import validate_record_citations
from scripts.compare_icp_reviews import archived_review, load_inputs, review_once, stability, summary


class ReviewComparisonTests(unittest.TestCase):
    def test_default_current_replay_preserves_enabled_review_thinking(self):
        _, samples = load_inputs('evals/icp_identity_feedback_review_cases.json', include_saved=False)
        with patch('scripts.compare_icp_reviews.llm_service.call_model', return_value='{}') as call:
            review_once(samples[0], answer_review, {'provider': 'ollama', 'max_tokens': 4096})
        self.assertTrue(call.call_args.args[1]['reasoning'])

    def test_reasoning_comparison_changes_only_reasoning_not_prompt_or_verdict(self):
        _, samples = load_inputs('evals/icp_identity_feedback_review_cases.json', include_saved=False)
        configs, results = [], []
        for reasoning in (False, True):
            with patch('scripts.compare_icp_reviews.llm_service.call_model', return_value='{}') as call:
                results.append(review_once(samples[0], answer_review, {'max_tokens': 4096}, reasoning=reasoning))
                configs.append(call.call_args.args[1])
        self.assertEqual(results[0]['messages'], results[1]['messages'])
        self.assertFalse(configs[0].pop('reasoning'))
        self.assertTrue(configs[1].pop('reasoning'))
        for config in configs:
            config.pop('timeout')  # Elapsed transport deadline is not a model policy change.
        self.assertEqual(configs[0], configs[1])
        self.assertNotIn('expected_release', results[0]['messages'][0]['content'])

    def test_stability_keeps_disagreement_and_system_errors_without_majority_release(self):
        rows = [{'condition': 'new', 'sample_id': 'x', 'status': status}
                for status in ('released', 'released', 'system_error')]
        result = stability(rows)[0]
        self.assertTrue(result['changed'])
        self.assertEqual(['released', 'released', 'system_error'], result['statuses'])

    def test_config_holdout_pairs_are_new_and_mechanically_valid(self):
        _, samples = load_inputs('evals/icp_review_config_holdout.json', include_saved=False)
        known = []
        for path in ('evals/icp_identity_feedback_review_cases.json', 'evals/icp_semantic_completeness_cases.json'):
            known.extend(load_inputs(path, include_saved=False)[1])
        self.assertEqual(8, len(samples))
        self.assertFalse({s['question'] for s in samples} & {s['question'] for s in known})
        for index in range(0, len(samples), 2):
            positive, negative = samples[index:index + 2]
            self.assertTrue(positive['expected_release'])
            self.assertFalse(negative['expected_release'])
            for key in ('question', 'ledger', 'history', 'observations', 'safety', 'calculation_requests'):
                self.assertEqual(positive[key], negative[key])
            for sample in (positive, negative):
                validate_record_citations(sample['draft'], sample['ledger'])

    def test_comparison_uses_the_same_messages_as_serving(self):
        _, samples = load_inputs('evals/icp_identity_feedback_review_cases.json', include_saved=False)
        with patch('scripts.compare_icp_reviews.llm_service.call_model', return_value='{}') as call:
            result = review_once(samples[0], answer_review, {})
        self.assertEqual(result['messages'], call.call_args.args[0])
        self.assertEqual(['user'], [m['role'] for m in result['messages']])
        self.assertIn(samples[0]['draft'], result['messages'][0]['content'])

    def test_explicit_candidate_message_builder_is_preserved_in_replay(self):
        _, samples = load_inputs('evals/icp_identity_feedback_review_cases.json', include_saved=False)
        protocol = types.SimpleNamespace(**{key: getattr(answer_review, key) for key in (
            'prepare_review', 'review_input', 'review_schema', 'validate_review',
            'REVIEW_INSTRUCTION', 'AnswerReviewError')})
        protocol.review_messages = lambda payload: [
            {'role': 'system', 'content': protocol.REVIEW_INSTRUCTION},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
        with patch('scripts.compare_icp_reviews.llm_service.call_model', return_value='{}') as call:
            result = review_once(samples[0], protocol, {})
        self.assertEqual(result['messages'], call.call_args.args[0])
        self.assertEqual(['system', 'user'], [m['role'] for m in result['messages']])
        self.assertEqual(samples[0]['draft'], json.loads(result['messages'][1]['content'])['draft'])

    def test_semantic_completeness_pairs_do_not_weaken_explicit_requests(self):
        _, samples = load_inputs('evals/icp_semantic_completeness_cases.json', include_saved=False)
        self.assertEqual(8, len(samples))
        for index in range(0, len(samples), 2):
            positive, negative = samples[index:index + 2]
            self.assertTrue(positive['expected_release'])
            self.assertFalse(negative['expected_release'])
            for key in ('question', 'ledger', 'history', 'calculation_requests', 'safety'):
                self.assertEqual(positive[key], negative[key])
            # Negatives are semantically wrong, not rejected by broken IDs.
            for sample in (positive, negative):
                validate_record_citations(sample['draft'], sample['ledger'])

    def test_explicit_local_v5_reference_is_not_silently_legacy_v4(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'baseline.tar.gz'
            source = b'REVIEW_INSTRUCTION = "baseline-v5"\n'
            with tarfile.open(path, 'w:gz') as archive:
                info = tarfile.TarInfo('backend/agent/answer_review.py')
                info.size = len(source)
                archive.addfile(info, io.BytesIO(source))
            reference, digest = archived_review(path)
            self.assertEqual('baseline-v5', reference.REVIEW_INSTRUCTION)
            self.assertEqual(64, len(digest))

    def test_predeclared_samples_are_balanced_and_no_valid_numeric_pair_is_unbound(self):
        _, samples = load_inputs('evals/icp_gate_review_inputs.json', include_saved=False)
        self.assertEqual(22, len(samples))
        self.assertEqual(11, sum(s['expected_release'] for s in samples))
        for sample in samples:
            if sample['expected_release']:
                with self.subTest(sample=sample['id']):
                    validate_record_citations(sample['draft'], sample['ledger'],
                                              calculation_requests=sample['calculation_requests'])

    def test_system_error_is_not_correct_semantic_rejection(self):
        rows = [{'condition': 'new', 'expected_release': False, 'status': 'system_error',
                 'duration_seconds': 1, 'raw': '', 'model_calls': []}]
        result = summary(rows)['new']
        self.assertEqual(1, result['system_error'])
        self.assertEqual(0, result['correct_rejection'])
        self.assertEqual(0, result['false_release'])
