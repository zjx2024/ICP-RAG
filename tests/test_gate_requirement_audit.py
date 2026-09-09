import json
import unittest
from pathlib import Path

from scripts.calibrate_icp_judge import load_samples


class GateRequirementAuditTests(unittest.TestCase):
    def test_all_original_turns_have_explicit_audit_and_same_question(self):
        root = Path(__file__).resolve().parents[1] / 'evals'
        suite = json.loads((root / 'icp_multiturn_dialogue_cases.json').read_text())
        audit = json.loads((root / 'icp_gate_requirement_audit.json').read_text())
        actual = {(c['id'], t['id']): t['question'] for c in suite['cases'] for t in c['turns']}
        reviewed = {(t['case_id'], t['turn_id']): t['question'] for t in audit['turns']}
        self.assertEqual(19, len(suite['cases']))
        self.assertEqual(40, len(actual))
        self.assertEqual(actual, reviewed)
        self.assertTrue(all(t['rationale'] and t['core_requirement'] for t in audit['turns']))

    def test_calibration_contains_balanced_pairs_and_source_for_every_sample(self):
        samples = load_samples(Path('evals/icp_gate_calibration.json'))
        self.assertEqual(20, len(samples))
        self.assertEqual(10, sum(all(s['expected'].values()) for s in samples))
        self.assertTrue(all(s['reference_context'] for s in samples))

