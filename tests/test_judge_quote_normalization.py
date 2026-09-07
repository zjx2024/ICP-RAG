"""Judge quote equivalence is presentation-only; emphasis markers survive.

The judge copies phrases out of markdown-formatted answers and naturally drops
``**bold**``/``_italic_`` markers.  Normalization strips emphasis on both
sides so a faithful quote passes.  Ellipsis-abbreviated quotes pass when every
segment appears in order; genuinely altered or invented phrases still fail.
"""
import unittest

from scripts.evaluate_icp_paraphrases import (
    _judge_quote_supported,
    _normalized_judge_quote,
)


class JudgeQuoteNormalizationTests(unittest.TestCase):
    def test_bold_markers_stripped_from_both_sides(self):
        answer = "- **电源与配电子系统（SUB-PWR / PWR）**：负责外部电源调理[^2]。"
        quote = "电源与配电子系统（SUB-PWR / PWR）：负责外部电源调理"
        self.assertIn(_normalized_judge_quote(quote), _normalized_judge_quote(answer))

    def test_inline_code_and_citation_markers_ignored(self):
        answer = "API 名称：`GetOIMFaultStatus` [^1]。"
        quote = "API 名称：GetOIMFaultStatus"
        self.assertIn(_normalized_judge_quote(quote), _normalized_judge_quote(answer))

    def test_wildcard_asterisks_inside_words_survive(self):
        # HMC wildcard semantics live inside the text; stripping emphasis must
        # not be confused by them on the quote side either.
        answer = "HSC 前导零与 ** 通配保留[^1]"
        quote = "HSC 前导零与 ** 通配保留"
        self.assertIn(_normalized_judge_quote(quote), _normalized_judge_quote(answer))

    def test_ellipsis_between_verbatim_segments_passes_in_order(self):
        answer = "不能单凭框图认定实际根因，也不能替代正式维修程序。"
        quote = "不能单凭框图认定实际根因...替代正式维修程序"
        self.assertTrue(_judge_quote_supported(quote, answer))

    def test_ellipsis_with_invented_segment_still_fails(self):
        answer = "优先排查方向包括电源链路、温度监测与连接器检查。"
        quote = "优先排查方向应聚焦于..."
        self.assertFalse(_judge_quote_supported(quote, answer))

    def test_out_of_order_segments_fail(self):
        answer = "先核对成功条件，再检查块之间的连接。"
        quote = "检查块之间的连接...先核对成功条件"
        self.assertFalse(_judge_quote_supported(quote, answer))

    def test_altered_word_still_fails(self):
        answer = "该模块负责输入调理与配电。"
        quote = "该模块负责输入调理与供能。"
        self.assertFalse(_judge_quote_supported(quote, answer))

    def test_bare_ellipsis_quote_fails(self):
        answer = "任何正文。"
        self.assertFalse(_judge_quote_supported("...", answer))
        self.assertFalse(_judge_quote_supported("", answer))


if __name__ == "__main__":
    unittest.main()
