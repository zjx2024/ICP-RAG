import copy
import unittest
from unittest.mock import patch

from scripts import evaluate_icp_paraphrases as evaluator
from scripts.rejudge_icp_report import rejudge_case


class ImmutableRegradeTests(unittest.TestCase):
    def fixture(self):
        cases, _ = evaluator._expand_cases({"cases": [{"id": "sample", "turns": [
            {"id": "first", "question": "原问题", "assertions": {"semantic_criteria": [{"id": "quality", "criterion": "回答原问题"}]}},
            {"id": "followup", "question": "追问", "assertions": {"semantic_criteria": [{"id": "quality", "criterion": "回答追问"}]}},
        ]}]})
        case = cases[0]
        record = {"case_id": case.case_id, "passed": False, "turns": [
            {"question": turn.question, "turn_id": turn.turn_id, "answer": f"原始正文{turn.index}",
             "event_types": ["answer", "done"], "runtime_errors": [], "evidence": {}, "citations": [],
             "assertions": {}, "assertion_results": [{"name": "old", "passed": False}], "passed": False}
            for turn in case.turns]}
        return record, case

    def test_every_turn_regraded_without_rewriting_answers_or_original_results(self):
        record, case = self.fixture()
        original = copy.deepcopy(record)
        decisions = ([{"name": "semantic_criterion:quality", "passed": True}], {"status": "completed", "criteria": []})
        with patch.object(evaluator, "_evaluate_semantic_criteria", return_value=decisions) as grade:
            result = rejudge_case(record, case, object())
        self.assertEqual(original, record)
        self.assertEqual(2, grade.call_count)
        self.assertEqual([t["answer"] for t in record["turns"]], [t["answer"] for t in result["turns"]])
        self.assertEqual([{"question": "原问题", "answer": "原始正文1"}], grade.call_args.kwargs["prior_turns"][:1])
        self.assertTrue(result["passed"])

    def test_runtime_failure_never_disappears_under_a_positive_judge(self):
        record, case = self.fixture()
        record["turns"][1]["runtime_errors"] = [{"kind": "timeout", "message": "recorded failure"}]
        with patch.object(evaluator, "_evaluate_semantic_criteria", return_value=([], {"status": "completed", "criteria": []})):
            result = rejudge_case(record, case, object())
        self.assertFalse(result["passed"])
        self.assertEqual(record["turns"][1]["runtime_errors"], result["turns"][1]["runtime_errors"])

    def test_changed_or_omitted_questions_cannot_be_regraded_as_the_same_run(self):
        for mode in ("question", "turn_count"):
            record, case = self.fixture()
            if mode == "question":
                record["turns"][1]["question"] = "另一道题"
            else:
                record["turns"].pop()
            with patch.object(evaluator, "_evaluate_semantic_criteria", return_value=([], {"status": "completed"})):
                with self.assertRaises(ValueError):
                    rejudge_case(record, case, object())


if __name__ == "__main__":
    unittest.main()
