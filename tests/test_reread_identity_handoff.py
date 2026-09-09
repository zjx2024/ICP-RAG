# v3/v4 validator assertions below are offline compatibility; live v5 counterparts are in test_answer_integrated_review.
import copy
import json
import unittest

from backend.agent.answer_context_binding import reread_identity_sources
from scripts.legacy_answer_review import AnswerReviewError, prepare_review, validate_review
from tests.legacy_review_fixtures import state_for, response, check


class RereadIdentityHandoffTests(unittest.TestCase):
    def fixture(self):
        state = state_for(doc_type='function_locator')
        subject = state['evidence_registry'][0]
        function = copy.deepcopy(subject)
        function.update(record_id='FUNC-482', evidence_id='function-482')
        ledger = {1: subject, 2: function}
        state['safety_decision']['approved_evidence_ids'].append(function['evidence_id'])
        state['dialogue_context'] = {'status': 'resolved'}
        state['selected_conversation_anchors'] = [{k: subject[k] for k in ('doc_type', 'doc_id', 'record_id')}]
        return state, ledger

    def review(self, state, ledger, *, draft='ROW-731 不可用首先影响输出更新。', sources=None, claims=None):
        prepared = prepare_review(draft, ledger=ledger, conversation={}, observations=[],
                                  task={'original_question': state['query']}, document_semantics='',
                                  current_identity_sources=reread_identity_sources(state, ledger))
        raw = response(check(sources=['e2:/answer_fields/local_impact'] if sources is None else sources,
                             claims=claims or ['source_fact']))
        return validate_review(json.dumps(raw), prepared=prepared, ledger=ledger,
                               safety=state['safety_decision'], calculation_requests=[])

    def test_reread_identity_handles_pronouns_without_rewriting_facts(self):
        for question in ('那它不可用呢？', 'What happens if it becomes unavailable?', '继续说明影响。'):
            state, ledger = self.fixture()
            state['query'] = question
            self.assertEqual(['e1:/record_id'], reread_identity_sources(state, ledger))
            result = self.review(state, ledger)
            self.assertEqual('full', result['status'])
            self.assertTrue(result['text'].startswith('ROW-731 不可用首先影响输出更新。'))

    def test_old_unapproved_wrong_document_revision_and_unresolved_context_cannot_bind(self):
        for mode in ('no_anchor', 'wrong_doc', 'wrong_revision', 'blocked', 'unapproved', 'search_copy', 'unresolved'):
            with self.subTest(mode=mode):
                state, ledger = self.fixture()
                anchor = state['selected_conversation_anchors'][0]
                if mode == 'no_anchor': state['selected_conversation_anchors'] = []
                if mode == 'wrong_doc': anchor['doc_id'] = 'other-document'
                if mode == 'wrong_revision': anchor['revision_id'] = 'old-revision'
                if mode == 'blocked': state['safety_decision']['blocked_evidence_ids'] = [ledger[1]['evidence_id']]
                if mode == 'unapproved': state['safety_decision']['approved_evidence_ids'] = []
                if mode == 'search_copy': ledger[1]['source_kind'] = 'opensearch'
                if mode == 'unresolved': state['dialogue_context']['status'] = 'ambiguous'
                self.assertEqual([], reread_identity_sources(state, ledger))
                with self.assertRaises(AnswerReviewError): self.review(state, ledger)

    def test_same_business_id_does_not_authorize_other_documents_or_ambiguous_versions(self):
        state, ledger = self.fixture()
        other = copy.deepcopy(ledger[1])
        other.update(doc_id='other-document', evidence_id='other-evidence')
        ledger[3] = other
        state['safety_decision']['approved_evidence_ids'].append('other-evidence')
        self.assertEqual(['e1:/record_id'], reread_identity_sources(state, ledger))
        self.assertNotIn('other-evidence', self.review(state, ledger)['manifest'][0]['evidence_ids'])
        other['doc_id'] = ledger[1]['doc_id']
        other['metadata']['revision_id'] = 'second-version'
        self.assertEqual([], reread_identity_sources(state, ledger))

    def test_current_identity_cannot_lend_neighbor_identity_or_factual_support(self):
        state, ledger = self.fixture()
        other = copy.deepcopy(ledger[1])
        other.update(record_id='ROW-732', evidence_id='other-evidence')
        ledger[3] = other
        with self.assertRaises(AnswerReviewError):
            self.review(state, ledger, draft='ROW-732 不可用首先影响输出更新。')
        for sources in ([], ['policy:safety']):
            with self.subTest(sources=sources), self.assertRaises(AnswerReviewError):
                self.review(state, ledger, sources=sources, claims=['conditional_analysis'])
