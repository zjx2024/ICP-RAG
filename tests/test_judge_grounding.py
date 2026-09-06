import copy
import json
import unittest
from unittest.mock import patch

from scripts import evaluate_icp_paraphrases as evaluator


class JudgeGroundingTests(unittest.TestCase):
    def setUp(self):
        self.criteria = [{"id": "quality", "criterion": "解释当前条件并区分证据出处"}]
        self.answer = "仅适用于启动阶段。处置来自故障代码手册。"
        self.checks = [
            {"aspect": "阶段", "requirement_kind": "positive", "answer_quotes": ["仅适用于启动阶段"], "passed": True, "reason": "限定可见"},
            {"aspect": "来源", "requirement_kind": "positive", "answer_quotes": ["处置来自故障代码手册"], "passed": True, "reason": "归属可见"},
        ]

    def evaluate(self, checks):
        return evaluator._validate_grounded_semantic_judgement({"criteria": [{"id": "quality", "passed": True, "reason": "维度审核", "checks": checks}]}, self.criteria, self.answer)

    def test_all_dimensions_must_pass_despite_global_pass(self):
        self.checks[1].update(passed=False, reason="出处与资料相冲突")
        result = self.evaluate(self.checks)
        self.assertFalse(result[0]["passed"])
        self.assertEqual(self.checks, result[0]["checks"])

    def test_judge_cannot_supply_a_qualification_absent_from_answer(self):
        self.checks[0]["answer_quotes"] = ["运行中不一定受影响"]
        with self.assertRaisesRegex(evaluator.ApiError, "不存在"):
            self.evaluate(self.checks)

    def test_pass_requires_actual_answer_quote_but_omission_can_be_empty(self):
        self.checks[0]["answer_quotes"] = []
        with self.assertRaisesRegex(evaluator.ApiError, "缺少答案原话"):
            self.evaluate(self.checks)
        self.checks[0]["passed"] = False
        self.assertFalse(self.evaluate(self.checks)[0]["passed"])

    def test_missing_checks_is_judge_error_not_product_fail(self):
        for checks in (None, [], "good"):
            with self.subTest(checks=checks), self.assertRaises(evaluator.ApiError):
                self.evaluate(checks)

    def test_bare_marker_error_identifies_the_exact_quote_and_required_correction(self):
        self.answer += "[^12]"
        self.checks[1]["answer_quotes"] = ["[^12]"]
        with self.assertRaises(evaluator.ApiError) as caught:
            self.evaluate(self.checks)
        self.assertIn("[^12]", str(caught.exception))
        self.assertIn("完整声明", str(caught.exception))
        self.checks[1]["answer_quotes"] = ["处置来自故障代码手册。[^12]"]
        self.assertTrue(self.evaluate(self.checks)[0]["passed"])

    def test_absent_prohibited_claim_does_not_require_nonexistent_quote(self):
        self.checks.append({"aspect": "没有捏造概率", "requirement_kind": "prohibition", "answer_quotes": [], "passed": True, "reason": "全文未声称概率数值"})
        self.assertTrue(self.evaluate(self.checks)[0]["passed"])

    def test_formatting_only_quote_normalization_does_not_allow_semantic_rewrite(self):
        self.answer = "仅适用于`启动阶段`[^2]。处置来自故障代码手册。"
        self.assertTrue(self.evaluate(self.checks)[0]["passed"])
        self.checks[0]["answer_quotes"] = ["仅适用于运行阶段"]
        with self.assertRaises(evaluator.ApiError):
            self.evaluate(self.checks)

    def judge(self):
        return evaluator.SemanticJudge(evaluator.JudgeConfig(
            provider="ollama", model_name="test", endpoint="http://localhost/api/chat",
            api_key="", timeout_seconds=120, source="test"))

    def payload(self, checks):
        return {"message": {"content": json.dumps({"criteria": [{"id": "quality", "passed": True, "reason": "审核", "checks": checks}]}, ensure_ascii=False)}}

    def test_protocol_retry_is_bounded_by_original_deadline(self):
        invalid = copy.deepcopy(self.checks)
        invalid[0]["answer_quotes"] = []
        with patch.object(evaluator, "_request_json", side_effect=[self.payload(invalid), self.payload(self.checks)]) as request, patch.object(evaluator.time, "monotonic", side_effect=[10, 11, 40]):
            result = self.judge().evaluate(question="问", prior_turns=[], answer=self.answer, criteria=self.criteria)
        self.assertTrue(result[0]["passed"])
        self.assertEqual([119, 90], [call.kwargs["timeout"] for call in request.call_args_list])

    def test_valid_failure_is_never_retried(self):
        self.checks[0].update(passed=False, answer_quotes=[], reason="未回答阶段")
        with patch.object(evaluator, "_request_json", return_value=self.payload(self.checks)) as request:
            result = self.judge().evaluate(question="问", prior_turns=[], answer=self.answer, criteria=self.criteria)
        self.assertFalse(result[0]["passed"])
        self.assertEqual(1, request.call_count)

    def test_repeated_protocol_failure_reports_raw_diagnostics_not_a_pass(self):
        self.checks[0]["answer_quotes"] = ["不存在的内容"]
        with patch.object(evaluator, "_request_json", return_value=self.payload(self.checks)) as request:
            _, report = evaluator._evaluate_semantic_criteria(self.criteria, judge=self.judge(), question="问", prior_turns=[], answer=self.answer)
        self.assertEqual(2, request.call_count)
        self.assertEqual("error", report["status"])
        self.assertIn("quality/阶段", report["error"])
        self.assertEqual(2, report["diagnostics"]["protocol_attempts"])
