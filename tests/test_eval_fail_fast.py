import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import evaluate_icp_paraphrases as evaluator


class FailFastTests(unittest.TestCase):
    def test_latest_outcomes_order_all_cases_without_changing_turns_or_original_indexes(self):
        cases = [evaluator.EvaluationCase(i, f'case-{i}', 'g', 'v', 'Q', {}, {}, turns=(
            evaluator.EvaluationTurn(1, 'first', 'question', {}, {}),
            evaluator.EvaluationTurn(2, 'followup', 'it?', {}, {}))) for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            old, latest = Path(tmp)/'old.json', Path(tmp)/'latest.json'
            old.write_text(json.dumps({'cases': [{'case_id': 'case-0', 'passed': False},
                                                  {'case_id': 'case-2', 'passed': True}]}))
            latest.write_text(json.dumps({'cases': [{'case_id': 'case-0', 'passed': True},
                                                     {'case_id': 'case-3', 'passed': False}]}))
            ordered, info = evaluator._prioritize_cases(cases, [str(old), str(latest)])
        self.assertEqual([3, 1, 0, 2], [case.index for case in ordered])
        self.assertEqual(4, len(ordered))
        for case in ordered:
            self.assertIs(cases[case.index], case)
            self.assertEqual(['first', 'followup'], [turn.turn_id for turn in case.turns])
        self.assertFalse(info['scores_reused'])
        self.assertEqual(2, len(info['reports']))

    def test_reordered_fail_fast_still_runs_previously_passed_case_and_does_not_copy_scores(self):
        cases = [evaluator.EvaluationCase(i, f'case-{i}', 'g', 'v', 'Q', {}, {}) for i in range(3)]
        args = evaluator.build_parser().parse_args(['cases.json', '--fail-fast', '--prioritize-report', 'old.json'])
        order = [cases[2], cases[1], cases[0]]
        def run_case(case, _config):
            return {'index': case.index, 'case_id': case.case_id, 'group_id': 'g',
                    'passed': case.index != 0, 'runtime_errors': []}
        with patch.object(evaluator, '_load_cases', return_value=(cases, {})), \
                patch.object(evaluator, '_prioritize_cases', return_value=(order, {'scores_reused': False})), \
                patch.object(evaluator, '_resolve_agent', return_value={'id': 'agent'}), \
                patch.object(evaluator, '_resolve_judge_config', return_value=None), \
                patch.object(evaluator, '_run_case', side_effect=run_case) as run:
            report, code = evaluator.run(args)
        self.assertEqual([2, 1, 0], [call.args[0].index for call in run.call_args_list])
        self.assertEqual(1, code)
        self.assertEqual(3, report['summary']['total'])
        self.assertEqual(1, report['summary']['failed'])

    def test_bad_ordering_report_rejects_before_api_and_never_silently_skips_cases(self):
        args = evaluator.build_parser().parse_args(['evals/icp_multiturn_dialogue_cases.json',
                                                   '--fail-fast', '--prioritize-report', '/does-not-exist.json'])
        with patch.object(evaluator, '_resolve_agent') as api, self.assertRaises(evaluator.EvaluationInputError):
            evaluator.run(args)
        api.assert_not_called()

    def test_first_failure_does_not_submit_other_cases_or_report_full_completion(self):
        cases = [evaluator.EvaluationCase(i, f'case-{i}', 'g', 'v', 'Q', {}, {}) for i in range(3)]
        args = evaluator.build_parser().parse_args(['cases.json', '--fail-fast', '--gate', '0'])
        with patch.object(evaluator, '_load_cases', return_value=(cases, {})), \
                patch.object(evaluator, '_resolve_agent', return_value={'id': 'agent'}), \
                patch.object(evaluator, '_resolve_judge_config', return_value=None), \
                patch.object(evaluator, '_run_case', return_value={'index': 0, 'case_id': 'case-0',
                             'group_id': 'g', 'passed': False, 'runtime_errors': []}) as run:
            report, code = evaluator.run(args)
        run.assert_called_once()
        self.assertEqual(1, code)
        self.assertFalse(report['evaluation_complete'])
        self.assertEqual('stopped_on_failure', report['run_status'])
        self.assertEqual(['case-1', 'case-2'], report['pending_case_ids'])

    def test_fail_fast_rejects_parallel_scheduling_before_api_calls(self):
        args = evaluator.build_parser().parse_args(['evals/icp_multiturn_dialogue_cases.json', '--fail-fast', '--concurrency', '2'])
        with patch.object(evaluator, '_resolve_agent') as api, self.assertRaises(evaluator.EvaluationInputError):
            evaluator.run(args)
        api.assert_not_called()

    def test_failed_first_turn_does_not_send_followup(self):
        turns = tuple(evaluator.EvaluationTurn(i+1, f't{i}', 'Q', {'answer_regex': ['expected']}, {}) for i in range(2))
        case = evaluator.EvaluationCase(0, 'case', 'g', 'v', 'Q', {}, {}, turns=turns)
        config = evaluator.RunnerConfig('http://local', 'agent', 'user', 10, True, False, '', '', '', fail_fast=True)
        with patch.object(evaluator, '_request_json', return_value={'id': 'session'}), \
                patch.object(evaluator, '_stream_sse', return_value=iter([
                    {'type': 'answer', 'content': 'wrong'}, {'type': 'done'}])) as stream:
            result = evaluator._run_case(case, config)
        stream.assert_called_once()
        self.assertEqual(1, len(result['turns']))
        self.assertEqual(['t1'], result['pending_turn_ids'])
