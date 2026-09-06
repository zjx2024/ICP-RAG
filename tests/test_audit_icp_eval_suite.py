from pathlib import Path
import unittest

from scripts import audit_icp_eval_suite as audit


class EvalSuiteAuditTests(unittest.TestCase):
    def test_regression_files_load_with_unique_case_ids(self):
        root = Path(__file__).resolve().parents[1]
        result = audit.audit_suites([str(root / path) for path in audit.DEFAULT_SUITES])
        self.assertEqual(100, result["cases"])
        self.assertGreaterEqual(result["counts"]["turns"], 200)
        self.assertEqual(3, len(result["files"]))
        self.assertTrue(all(len(item["sha256"]) == 64 for item in result["files"]))
        self.assertTrue(result["ready_for_gold_review"])
        self.assertEqual(result["counts"]["turns"], result["counts"]["semantic_criteria"])
        self.assertEqual(0, result["counts"]["wide_or_regex"])
        self.assertIn("11", result["turn_lengths"])
        self.assertGreater(result["counts"]["numeric_expectations"], 0)

    def test_duplicate_suites_fail_without_network(self):
        path = str(Path(__file__).resolve().parents[1] / audit.DEFAULT_SUITES[0])
        with self.assertRaisesRegex(ValueError, "duplicate case"):
            audit.audit_suites([path, path])
