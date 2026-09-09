"""The offline experiment changes writer reasoning, not evidence or the reviewer."""
import asyncio
import types
import unittest
from unittest.mock import AsyncMock, Mock

from scripts.compare_icp_generation import publication_checks, run_node
from scripts.diagnose_icp_architecture import GenerationCaptured, stop_before_generation, was_generation_capture


class GenerationThinkingComparisonTests(unittest.TestCase):
    def test_semantic_pass_cannot_hide_missing_public_citations(self):
        result = {'core_passed': True, 'final_answer': '来源〔^1〕', 'citations': []}
        assertions = {'min_citations': 1, 'required_used_doc_types': ['hmc'],
                      'required_event_types': ['answer', 'done']}
        checks = publication_checks(result, assertions)
        self.assertEqual({'min_citations', 'required_used_doc_types'}, {c['name'] for c in checks})
        self.assertTrue(all(not c['passed'] for c in checks))
        result['citations'] = [{'doc_type': 'hmc', 'record_id': 'code'}]
        self.assertTrue(all(c['passed'] for c in publication_checks(result, assertions)))

    def module(self):
        module = types.SimpleNamespace(
            _with_answer_evidence_rerank=AsyncMock(),
            agent_model_role_service=types.SimpleNamespace(
                get_model_by_role=Mock(), apply_agent_llm_params=Mock()),
            _prepare_answer_model=lambda model: {**model, 'reasoning': False},
            _prepare_answer_review_model=lambda model: {**model, 'reasoning': True},
            _generate_answer_response=AsyncMock(return_value='draft'),
        )
        async def node(state):
            writer = module._prepare_answer_model(state['model'])
            text = await module._generate_answer_response(state['prompt'])
            reviewer = module._prepare_answer_review_model(state['model'])
            return {'final_answer': text, 'observed': [writer, reviewer]}
        module.answer_node = node
        return module

    def test_only_writer_reasoning_changes_and_patches_are_restored(self):
        module = self.module()
        original = module._prepare_answer_model
        model = {'provider': 'ollama', 'max_tokens': 4096}
        state = {'model': model, 'prompt': 'same evidence and question'}
        results = [asyncio.run(run_node(module, state, model, reasoning=enabled,
                                      expected_prompt=state['prompt'])) for enabled in (False, True)]
        self.assertFalse(results[0]['observed'][0]['reasoning'])
        self.assertTrue(results[1]['observed'][0]['reasoning'])
        for result in results:
            self.assertTrue(result['observed'][1]['reasoning'])
            self.assertEqual([4096, 4096], [value['max_tokens'] for value in result['observed']])
            self.assertEqual('ok', result['status'])
        self.assertIs(original, module._prepare_answer_model)
        self.assertNotIn('reasoning', model)

    def test_prompt_drift_stops_before_model_call(self):
        module = self.module()
        generate = module._generate_answer_response
        result = asyncio.run(run_node(module, {'model': {}, 'prompt': 'changed'}, {},
                                      reasoning=True, expected_prompt='frozen'))
        self.assertEqual('runtime_error', result['status'])
        generate.assert_not_called()

    def test_capture_stop_does_not_disguise_other_errors(self):
        with self.assertRaises(GenerationCaptured) as stopped:
            asyncio.run(stop_before_generation())
        outer = RuntimeError('node wrapper')
        outer.__cause__ = stopped.exception
        self.assertTrue(was_generation_capture(outer))
        self.assertFalse(was_generation_capture(RuntimeError('real failure')))
