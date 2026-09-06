import unittest

from backend.agent.nodes.answer_text import split_top_level_clauses



class AnswerSyntaxTests(unittest.TestCase):
    def test_nested_formal_expressions_are_atomic(self):
        for expression in ("OR(A；AND(B；C))", "变换（甲；函数[乙；丙]）", '`call("a；b")`', "“第一项；第二项”"):
            with self.subTest(expression=expression):
                self.assertEqual([expression + "。", "后续说明。"], split_top_level_clauses(expression + "。后续说明。"))

    def test_only_syntax_not_chinese_conjunctions_splits_a_unit(self):
        line = "信号源可能因条件甲或因条件乙失败，同时因共享输入影响两条链。"
        self.assertEqual([line], split_top_level_clauses(line))

    def test_citations_remain_attached_and_identical(self):
        self.assertEqual(["甲。 [^12][^3]", "乙！[^27]", "丙"], split_top_level_clauses("甲。 [^12][^3]乙！[^27]丙"))



if __name__ == "__main__":
    unittest.main()
