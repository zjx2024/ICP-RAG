import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.agent.answer_observation import observe_answers, observe_review_exchange
from backend.agent.answer_review import AnswerReviewError, prepare_review, validate_review
from backend.agent import tracing
from scripts.replay_icp_review import replay


class ReviewExchangeTests(unittest.TestCase):
    def inputs(self):
        return dict(raw='{}', prepared=prepare_review('有依据的原稿', ledger={}, conversation={},
                    observations=[], task={}, document_semantics=''), ledger={}, safety={},
                    calculation_requests=[])

    def test_exact_rejected_exchange_replays_without_models_and_detects_tampering(self):
        inputs = self.inputs()
        try:
            validate_review(**inputs)
        except AnswerReviewError as exc:
            with observe_answers() as captured:
                observe_review_exchange(**inputs, error=exc)
        snapshot = captured.as_dict()['reviews'][0]
        self.assertTrue(replay(snapshot)['same_verdict'])
        self.assertEqual(0, replay(snapshot)['model_calls'])
        snapshot['raw'] = 'changed'
        with self.assertRaises(ValueError):
            replay(snapshot)

    def test_private_failure_is_persisted_but_public_trace_contains_no_draft(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(tracing.config, 'TRACE_DIR', str(Path(tmp)/'traces')):
            tracing.begin_turn('session', 'question')
            try:
                observe_review_exchange(**self.inputs(), error=AnswerReviewError('invalid review version'))
                files = list((Path(tmp)/'private_answer_reviews').glob('*.json'))
                self.assertEqual(1, len(files))
                self.assertEqual(0o600, files[0].stat().st_mode & 0o777)
                saved = json.loads(files[0].read_text())
                self.assertEqual('有依据的原稿', saved['prepared']['draft'])
                self.assertNotIn('有依据的原稿', json.dumps(tracing.peek().events, ensure_ascii=False))
            finally:
                tracing.end_turn()

    def test_success_has_no_automatic_private_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(tracing.config, 'TRACE_DIR', str(Path(tmp)/'traces')):
            tracing.begin_turn('session', 'question')
            try:
                observe_review_exchange(**self.inputs(), result={'status': 'full'})
                self.assertFalse((Path(tmp)/'private_answer_reviews').exists())
            finally:
                tracing.end_turn()

    def test_recording_failure_does_not_replace_the_review_verdict(self):
        with observe_answers() as captured, patch('backend.agent.answer_observation.detached', side_effect=ValueError):
            observe_review_exchange(**self.inputs(), error=AnswerReviewError('invalid review version'))
        self.assertEqual(['review_exchange:ValueError'], captured.errors)
