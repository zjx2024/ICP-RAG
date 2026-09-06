import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_icp_eval_campaign as campaign


class EvalCampaignTests(unittest.TestCase):
    def test_cost_capture_filters_other_sessions_and_does_not_export_queries(self):
        report = {"cases": [{"session_id": "ours", "case_id": "c", "turns": [{"question": "q", "turn_id": "t"}]}]}
        traces = [{"session_id": "ours", "query": "q", "trace_id": "trace", "llm_calls": 5},
                  {"session_id": "other", "query": "not our test", "llm_calls": 99}]
        with patch.object(campaign.evaluator, "_request_json", return_value=traces):
            costs = campaign.collect_trace_costs(report, base_url="http://localhost")
        self.assertEqual(5, costs["total_llm_calls"])
        self.assertEqual(["t"], costs["traces"][0]["turn_ids"])
        self.assertNotIn("query", costs["traces"][0])
        self.assertNotIn("not our test", json.dumps(costs))

    def test_quality_gate_does_not_hide_runtime_errors_in_reduced_denominator(self):
        base = {"evaluation_complete": True, "total_turns": 10, "pass_rate": 0.9,
                "breakdowns": {"overall": {"semantic_judged_turns": 10, "semantic_passed_turns": 9}}}
        self.assertTrue(campaign.quality_gate(base)["passed"])
        for changes in [{"semantic_judged_turns": 9}, {"runtime_error_turns": 1},
                        {"judge_error_turns": 1}, {"semantic_passed_turns": 8}]:
            summary = {**base, "breakdowns": {"overall": {**base["breakdowns"]["overall"], **changes}}}
            self.assertFalse(campaign.quality_gate(summary)["passed"])

    def test_live_configuration_change_invalidates_frozen_campaign(self):
        def manifest(model="current", doc="v1"):
            return {"services": {"application": {"reachable": True, "models": {"ok": True, "configuration": {"id": model}},
                                                 "documents": {"doc": doc}}}}
        original = campaign.live_configuration(manifest(), {"role_models": {"answer": "m"}})
        for snapshot, agent in [(manifest("other"), {"role_models": {"answer": "m"}}),
                                (manifest(doc="v2"), {"role_models": {"answer": "m"}}),
                                (manifest(), {"role_models": {"answer": "different"}})]:
            self.assertNotEqual(original, campaign.live_configuration(snapshot, agent))
        with self.assertRaises(ValueError):
            campaign.live_configuration({}, {})

    def test_acceptance_keeps_19_100_and_holdout_separate_across_three_repeats(self):
        manifest = {"code_inputs": {"agent": {"aggregate_sha256": "a"}},
                    "fixtures": {"aggregate_sha256": "b"}, "evaluation_suites": {"aggregate_sha256": "c"}}

        def evaluate(command):
            report = {"evaluation_complete": True, "cases": [{"case_id": command[0], "passed": True,
                       "turns": [{"passed": True, "semantic_judgement": {"status": "completed", "criteria": [{"passed": True}]}}]}]}
            campaign.evaluator._write_report(report, command[command.index("-o") + 1], False)
            return 0

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(campaign.audit, "audit_suites", return_value={"ready_for_gold_review": True, "cases": 100}), \
             patch.object(campaign.baseline, "capture_baseline", return_value=manifest), \
             patch.object(campaign, "live_configuration", return_value="frozen"), \
             patch.object(campaign, "selected_runtime_configuration", return_value={}), \
             patch.object(campaign, "calibration_status", return_value={"qualified": True}), \
             patch.object(campaign, "collect_trace_costs", return_value={"status": "observed"}), \
             patch.object(campaign.evaluator, "main", side_effect=evaluate):
            output = Path(directory) / "acceptance"
            self.assertEqual(0, campaign.main(["--acceptance", "--repeats", "3", "--output-dir", str(output)]))
            report = json.loads((output / "campaign.json").read_text())
        self.assertEqual(15, len(report["runs"]))
        self.assertEqual(3, len(report["repeat_summaries"]))
        self.assertTrue(report["automated_quality_passed"])
        for repeat in report["repeat_summaries"]:
            self.assertNotIn("pass_rate", repeat)
            self.assertEqual({"baseline_19", "regression_100", "expression_holdout"}, set(repeat["groups"]))

    def test_partial_repeat_never_gets_a_completed_score(self):
        partial = campaign.repeat_summary([{}], expected_suites=3)
        self.assertEqual("in_progress", partial["run_status"])
        self.assertNotIn("pass_rate", partial)

    def test_duplicate_case_across_suites_cannot_inflate_total(self):
        report = {"cases": [{"case_id": "same", "turns": []}]}
        with self.assertRaises(ValueError):
            campaign.repeat_summary([report, report], expected_suites=2)

    def test_operational_failure_is_not_a_complete_quality_evaluation(self):
        report = {"evaluation_complete": False, "cases": [{
            "case_id": "a", "passed": False, "turns": [{"passed": False}],
        }]}
        with patch.object(campaign.evaluator, "_evaluation_breakdowns", return_value={}):
            result = campaign.repeat_summary([report], expected_suites=1)
        self.assertEqual("finished", result["run_status"])
        self.assertFalse(result["evaluation_complete"])

    def test_existing_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            campaign.audit, "audit_suites", return_value={"ready_for_gold_review": True},
        ), patch.object(campaign.baseline, "capture_baseline") as capture:
            self.assertEqual(2, campaign.main(["--output-dir", directory]))
            capture.assert_not_called()

    def test_failed_answers_do_not_skip_remaining_suites_or_pool_repeats(self):
        manifest = {"code_inputs": {"agent": {"aggregate_sha256": "a"}},
                    "fixtures": {"aggregate_sha256": "b"},
                    "evaluation_suites": {"aggregate_sha256": "c"}}
        observed = []

        def evaluate(command):
            target = Path(command[command.index("-o") + 1])
            observed.append(target)
            report = {"evaluation_complete": True, "cases": [{
                "case_id": command[0], "passed": False, "turns": [{"passed": False}],
            }]}
            campaign.evaluator._write_report(report, str(target), False)
            return 1

        with tempfile.TemporaryDirectory() as directory, patch.object(
            campaign.audit, "audit_suites", return_value={"ready_for_gold_review": True, "cases": 3},
        ), patch.object(campaign.baseline, "capture_baseline", return_value=manifest), patch.object(
            campaign.evaluator, "main", side_effect=evaluate,
        ), patch.object(campaign.evaluator, "_evaluation_breakdowns", return_value={}):
            output = Path(directory) / "fresh-run"
            self.assertEqual(1, campaign.main(["--output-dir", str(output), "--repeats", "2"]))
            archived = json.loads((output / "campaign.json").read_text())
        self.assertEqual(6, len(observed))
        self.assertEqual("finished", archived["run_status"])
        self.assertEqual([3, 3], [item["total"] for item in archived["repeat_summaries"]])
        self.assertNotIn("pass_rate", archived)

    def test_missing_fingerprint_cannot_authorize_frozen_run(self):
        with self.assertRaises(ValueError):
            campaign.input_fingerprint({"code_inputs": {}, "fixtures": {}, "evaluation_suites": {}})

    def test_failed_or_different_judge_calibration_cannot_authorize_acceptance(self):
        judge = {"model_id": "m", "model_name": "model", "protocol_version": "v9", "timeout_seconds": 120}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            payload = {"judge": judge, "total": 1, "cases": [{"passed": True, "judgement": {"status": "completed"}}]}
            path.write_text(json.dumps(payload))
            self.assertTrue(campaign.calibration_status(str(path), judge)["qualified"])
            self.assertFalse(campaign.calibration_status(str(path), {**judge, "model_id": "other"})["qualified"])
            payload["cases"][0]["passed"] = False
            path.write_text(json.dumps(payload))
            self.assertFalse(campaign.calibration_status(str(path), judge)["qualified"])
        self.assertFalse(campaign.calibration_status("", judge)["qualified"])


if __name__ == "__main__":
    unittest.main()
