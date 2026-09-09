# v3/v4 validator assertions below are offline compatibility; live v5 counterparts are in test_answer_integrated_review.
import json
import unittest

from scripts.legacy_answer_review import (
    AnswerReviewError,
    prepare_review,
    validate_review,
)
from backend.agent.citation_binding import markdown_review_units


class ReviewMarkdownContextTests(unittest.TestCase):
    def review(self, draft, *, selected=1):
        ledger = {n: {"record_id": identity, "doc_type": "fmeca", "metadata": {
            "answer_fields": {"description": "检查连接"}}}
            for n, identity in enumerate(("FM-ALT-073", "FM-ALT-074", "74210001**0081"), 1)}
        prepared = prepare_review(draft, ledger=ledger, conversation={}, observations=[],
                                  task={"original_question": "解释这条记录"}, document_semantics="")
        checks = [{"unit_id": u["unit_id"], "sources": [f"e{selected}:/answer_fields/description"],
                   "claims": ["source_fact"], "attributed_doc_types": [], "issue": ""}
                  for u in prepared["units"]]
        raw = json.dumps({"version": 3, "units": checks, "goals": [
            {"goal_id": "q0", "status": "answered", "unit_ids": [u["unit_id"] for u in checks], "issue": ""}]})
        return validate_review(raw, prepared=prepared, ledger=ledger, safety={}, calculation_requests=[])

    def test_nested_list_bold_is_rendered_in_parent_context_and_prose_is_unchanged(self):
        draft = "1. 先检查\n    * **FM-ALT-073**：检查连接。"
        units = markdown_review_units(draft)
        self.assertEqual("FM-ALT-073：检查连接。", units[1]["identity_text"])
        result = self.review(draft)
        self.assertEqual(draft, result["text"].replace("[^1]", ""))

    def test_same_indented_fragment_in_code_keeps_literal_stars(self):
        draft = "    * **FM-ALT-073**：检查连接。"
        self.assertIn("**FM-ALT-073**", markdown_review_units(draft)[0]["identity_text"])
        with self.assertRaises(AnswerReviewError):
            self.review(draft)

    def test_fenced_code_keeps_wildcards_and_does_not_parse_emphasis(self):
        draft = "```text\n74210001**0081\n```"
        self.assertEqual("74210001**0081", markdown_review_units(draft)[1]["identity_text"])
        self.review(draft, selected=3)
        with self.assertRaises(AnswerReviewError):
            self.review("```text\n**FM-ALT-073**\n```")

    def test_multiline_emphasis_is_not_reparsed_line_by_line(self):
        draft = "**引用\nFM-ALT-073**：检查连接。"
        self.assertEqual("FM-ALT-073：检查连接。", markdown_review_units(draft)[1]["identity_text"])
        self.review(draft)

    def test_multiline_code_normalization_reviews_complete_span_not_guessed_line(self):
        draft = "`FM-ALT-073\n检查连接`\n后续说明\n\n下一段"
        units = markdown_review_units(draft)
        self.assertEqual(2, len(units))
        self.assertEqual("FM-ALT-073 检查连接\n后续说明", units[0]["identity_text"])
        self.assertEqual(draft[:draft.index("\n\n")], units[0]["text"])
        self.review(draft)

    def test_tables_and_blockquotes_share_visible_identity_contract(self):
        for draft in ("| 标识 | 内容 |\n| --- | --- |\n| **FM-ALT-073** | 检查连接 |",
                      "> **FM-ALT-073**：检查连接。"):
            with self.subTest(draft=draft):
                self.review(draft)

    def test_neighbor_or_changed_wildcard_still_rejects_in_nested_lists(self):
        for identity, selected in (("FM-ALT-074", 1), ("742100010081", 3), ("74210001**0082", 3)):
            with self.subTest(identity=identity), self.assertRaises(AnswerReviewError):
                self.review(f"1. 检查\n    * `{identity}`：检查连接。", selected=selected)
        self.review("1. 检查\n    * 74210001\\*\\*0081：检查连接。", selected=3)

    def test_crlf_offsets_and_blank_lines_remain_exact(self):
        draft = "1. 检查\r\n\r\n    * **FM-ALT-073**：检查连接。\r\n"
        for unit in markdown_review_units(draft):
            self.assertEqual(unit["text"], draft[unit["start"]:unit["end"]])
        self.assertEqual(draft, self.review(draft)["text"].replace("[^1]", ""))
