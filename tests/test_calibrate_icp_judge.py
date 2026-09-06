from pathlib import Path
import unittest

from scripts import calibrate_icp_judge as calibration


class JudgeCalibrationTests(unittest.TestCase):
    def test_repository_samples_include_positive_and_negative_gold(self):
        samples = calibration.load_samples(Path(__file__).resolve().parents[1] / "evals/icp_judge_calibration.json")
        self.assertEqual(36, len(samples))
        self.assertEqual(18, sum(all(item["expected"].values()) for item in samples))
        self.assertEqual(18, sum(bool(item["reference_context"]) for item in samples))

    def test_judge_that_always_passes_cannot_pass_calibration(self):
        class AlwaysPass:
            def evaluate(self, **values):
                return [{"id": criterion["id"], "passed": True, "reason": "keyword"} for criterion in values["criteria"]]
        sample = {"id": "wrong", "question": "new dimension", "prior_turns": [], "answer": "old answer",
                  "criteria": [{"id": "relevance", "criterion": "respond to the new dimension"}], "expected": {"relevance": False}}
        self.assertFalse(calibration.evaluate_sample(sample, AlwaysPass())["passed"])
