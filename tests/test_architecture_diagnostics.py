import copy
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import diagnose_icp_architecture as diagnostics


class ArchitectureDiagnosticsTests(unittest.TestCase):
    def test_old_capture_recovers_only_exact_system_prompt_slot(self):
        from backend.agent.nodes import answer
        original = {"state": {"query": "当前问题"}, "context": "固定证据"}
        original["prompt"] = answer._build_answer_prompt(
            {**original["state"], "system_prompt": "原运行的系统设定，不是当前配置。"}, original["context"])
        restored = diagnostics.restore_generation(original)
        self.assertEqual("原运行的系统设定，不是当前配置。", restored["state"]["system_prompt"])
        self.assertNotIn("system_prompt", original["state"])
        original["prompt"] += "其他变化"
        with self.assertRaisesRegex(ValueError, "cannot reconstruct"):
            diagnostics.restore_generation(original)

    def test_answer_pair_requires_same_state_prompt_sources_and_model(self):
        original = {key: {"value": key} for key in ("state", "context", "citations", "model", "prompt")}
        diagnostics.assert_same_answer_input(original, copy.deepcopy(original))
        for key in original:
            changed = copy.deepcopy(original)
            changed[key] = "different"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                diagnostics.assert_same_answer_input(original, changed)

    def test_live_preflight_calls_mock_status_method_instead_of_testing_method_truthiness(self):
        client = Mock()
        client.conn_is_mock.return_value = False
        diagnostics.require_real_postgres(client)
        client.conn_is_mock.assert_called_once_with()
        client.conn_is_mock.return_value = True
        with self.assertRaisesRegex(ValueError, "mock fallback"):
            diagnostics.require_real_postgres(client)

    def generation(self):
        return {"context": "retrieved-only-fact", "citations": [{"id": 7, "evidence_id": "e"}],
                "state": {"query": "现在解释一下", "messages": [
                    {"role": "user", "content": "前问", "metadata": {"gold": "gold-marker"}},
                    {"role": "assistant", "content": "旧答", "resolved_gold_context": "gold-marker"},
                    {"role": "user", "content": "现在解释一下"},
                ], "query_frame": {"required_answer_fields": ["do-not-copy-fields"]},
                    "resolved_gold_context": "gold-marker", "safety_decision": {"allow_calculation": False}}}

    def test_minimal_condition_contains_neither_gold_nor_old_field_obligations(self):
        generation = self.generation()
        original = copy.deepcopy(generation)
        payload = diagnostics.minimal_input(generation)
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("gold-marker", serialized)
        self.assertNotIn("do-not-copy-fields", serialized)
        self.assertEqual("retrieved-only-fact", payload["evidence"])
        self.assertEqual(2, len(payload["previous_messages"]))
        self.assertEqual(original, generation)

    def test_reference_condition_replaces_evidence_and_does_not_change_authorization(self):
        generation = self.generation()
        sources = [{"path": "source.docx", "sha256": "hash", "excerpts": [{"text": "reference-only-fact"}]}]
        payload = diagnostics.minimal_input(generation, reference_sources=sources)
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("retrieved-only-fact", serialized)
        self.assertIn("reference-only-fact", serialized)
        self.assertFalse(payload["safety"]["allow_calculation"])
        self.assertNotIn("reference-only-fact", json.dumps(diagnostics.minimal_input(generation)))

    def test_fixed_history_ignores_metadata_and_does_not_use_future_turns(self):
        turns = [{"question": "第一问", "answer": "第一答", "metadata": {"gold": "gold-marker"}},
                 {"question": "当前问题", "answer": "future-answer", "resolved_gold_context": "gold-marker"},
                 {"question": "future-question", "answer": "future-answer"}]
        history = diagnostics.fixed_history(turns, 1)
        text = json.dumps(history, ensure_ascii=False)
        for forbidden in ("gold-marker", "future-answer", "future-question"):
            self.assertNotIn(forbidden, text)
        self.assertEqual("当前问题", history[-1]["content"])

    def test_diff_is_observation_not_quality_or_keyword_scoring(self):
        changes = diagnostics.stage_deltas({"stages": [
            {"stage": "draft", "answer": "事件历史记录职责。[^7]"},
            {"stage": "published", "answer": "没有可用信息。"},
        ]})
        self.assertTrue(changes[0]["changed"])
        self.assertIn("-事件历史记录职责。[^7]", changes[0]["diff"])
        self.assertNotIn("passed", changes[0])

    def test_private_artifacts_refuse_overwrite_and_have_private_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            diagnostics.write_private(path, {"answer": "甲"})
            self.assertEqual(0o600, stat.S_IMODE(path.stat().st_mode))
            with self.assertRaises(FileExistsError):
                diagnostics.write_private(path, {"answer": "乙"})
            self.assertEqual("甲", diagnostics.read_json(path)["answer"])

    def test_unknown_citation_not_replaced_by_nearest_record(self):
        ledger = [{"id": 7, "evidence_id": "a"}, {"id": 8, "evidence_id": "b"}]
        self.assertEqual([ledger[0]], diagnostics.used_citations("事实。[^7] 其他。[^90]", ledger))

    def test_code_change_stops_experiment(self):
        with patch.object(diagnostics, "capture_baseline", return_value={}), patch.object(
            diagnostics, "input_fingerprint", return_value={"code": "changed"}
        ):
            with self.assertRaisesRegex(ValueError, "changed"):
                diagnostics.verify_inputs({"code": "original"})

    def test_summary_never_counts_judge_errors_as_answer_failures_or_pairs(self):
        def row(turn, condition, status="ok", passed=True):
            result = {"case_id": "C", "turn_index": turn, "repeat": 1,
                      "condition": condition, "status": status}
            if status == "ok":
                result["semantic_judgement"] = [{"passed": passed}]
            return result
        summary = diagnostics.comparison_summary([
            row(1, "current_initial"), row(1, "current_published", passed=False),
            row(2, "current_initial", "judge_error"), row(2, "current_published"),
            row(3, "retrieved_minimal", "runtime_error"),
        ])
        self.assertEqual({"evaluable_pairs": 1, "improved": 0, "regressed": 1, "unchanged": 0},
                         summary["current_initial_to_published"])
        self.assertEqual(0, summary["conditions"]["current_initial"]["failed"])
        self.assertEqual(1, summary["conditions"]["current_initial"]["judge_error"])
        self.assertIsNone(summary["end_to_end_score"])

    def test_existing_directory_rejected_before_baseline_or_any_model(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(diagnostics, "capture_baseline") as probe:
            with self.assertRaises(FileExistsError):
                diagnostics.start_run(Path(directory))
            probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
