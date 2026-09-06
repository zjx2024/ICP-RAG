"""Partial evaluation artifacts are recoverable but never a completed score."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import evaluate_icp_paraphrases as evaluator


class EvaluationCheckpointTests(unittest.TestCase):
    def test_partial_checkpoint_keeps_original_order_and_does_not_replace_report(self):
        cases = [
            evaluator.EvaluationCase(i, f"case-{i}", "group", "base", "问题", {}, {})
            for i in (4, 12, 31)
        ]
        config = evaluator.RunnerConfig("http://localhost", "agent", "eval", 20, False, True, "chat", "embed", "rank")
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "report.json")
            evaluator._write_report({"previous_run": True}, output, False)
            args = evaluator.build_parser().parse_args(["cases.json", "-o", output])
            evaluator._write_checkpoint(
                args, started_at="start", cases=cases,
                results={31: {"case_id": "case-31"}, 4: {"case_id": "case-4"}},
                input_metadata={"version": "frozen"}, config=config,
            )
            checkpoint = json.loads(Path(f"{output}.checkpoint.json").read_text())
            self.assertEqual("in_progress", checkpoint["run_status"])
            self.assertFalse(checkpoint["evaluation_complete"])
            self.assertNotIn("summary", checkpoint)
            self.assertEqual(["case-4", "case-31"], [case["case_id"] for case in checkpoint["cases"]])
            self.assertEqual(["case-12"], checkpoint["pending_case_ids"])
            self.assertEqual({"previous_run": True}, json.loads(Path(output).read_text()))

    def test_stdout_run_does_not_emit_intermediate_json(self):
        args = evaluator.build_parser().parse_args(["cases.json"])
        with patch.object(evaluator, "_write_report") as write:
            evaluator._write_checkpoint(args, started_at="start", cases=[], results={}, input_metadata={}, config=None)
        write.assert_not_called()

    def test_final_checkpoint_marks_failed_but_finished_evaluation_honestly(self):
        report = {"evaluation_complete": True, "summary": {"passed": 0}}
        with patch.object(evaluator, "run", return_value=(report, 1)), patch.object(evaluator, "_write_report") as write:
            code = evaluator.main(["cases.json", "-o", "report.json"])
        self.assertEqual(1, code)
        self.assertEqual(2, write.call_count)
        self.assertEqual("finished", write.call_args_list[-1].args[0]["run_status"])
        self.assertEqual(0, write.call_args_list[-1].args[0]["summary"]["passed"])

    def test_archival_failure_has_nonzero_exit_instead_of_silent_success(self):
        with patch.object(evaluator, "run", side_effect=OSError("disk unavailable")):
            self.assertEqual(2, evaluator.main(["cases.json"]))


if __name__ == "__main__":
    unittest.main()
