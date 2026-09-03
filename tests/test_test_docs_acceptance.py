import unittest
from pathlib import Path

from backend.core.parsers.excel_row import extract_domain_drafts
from backend.core.parsers.fault_history import extract_history_drafts


TEST_DOCS = Path(__file__).resolve().parents[1] / "test-docs"


@unittest.skipUnless(TEST_DOCS.is_dir(), "test-docs 样例目录不存在")
class SyntheticTestDocsAcceptanceTests(unittest.TestCase):
    """Lock the generated samples as parser oracles, not real engineering truth."""

    def test_hmc_extracts_one_hundred_traceable_rows(self):
        drafts = extract_domain_drafts(
            str(TEST_DOCS / "02_HMC故障代码手册.docx"), "hmc"
        )

        self.assertEqual(len(drafts), 100)
        self.assertTrue(all(draft.valid for draft in drafts))
        self.assertEqual(drafts[0].values["hmc_code"], "91810001**0001")
        self.assertEqual(drafts[-1].values["hmc_code"], "91860003**0003")
        self.assertTrue(all(draft.source_ref for draft in drafts))
        self.assertTrue(all(draft.raw_data.get("columns") for draft in drafts))

    def test_fmeca_extracts_seventy_two_rows_without_coercing_probability_class(self):
        drafts = extract_domain_drafts(
            str(TEST_DOCS / "03_FMECA分析报告.docx"), "fmeca"
        )

        self.assertEqual(len(drafts), 72)
        self.assertTrue(all(draft.valid for draft in drafts))
        self.assertEqual(drafts[0].values["code"], "FM-SYN-001")
        self.assertEqual(drafts[-1].values["code"], "FM-SYN-072")
        self.assertIsNone(drafts[0].values["probability"])
        self.assertEqual(drafts[0].values["probability_class"], "C")
        self.assertIn(
            "probability_category_preserved",
            [issue.code for issue in drafts[0].issues],
        )

    def test_mfl_extracts_seventy_two_rows_and_preserves_status_and_action(self):
        drafts = extract_domain_drafts(
            str(TEST_DOCS / "06_MFL维修故障清单.docx"), "mfl"
        )

        self.assertEqual(len(drafts), 72)
        self.assertTrue(all(draft.valid for draft in drafts))
        self.assertEqual(drafts[0].values["mfl_code"], "MFL-SYN-001")
        self.assertEqual(drafts[-1].values["mfl_code"], "MFL-SYN-072")
        self.assertEqual(drafts[0].values["platform_test_status"], "部分完成")
        self.assertTrue(drafts[0].values["repair_action"])

    def test_history_extracts_twenty_four_cases_and_one_hundred_forty_four_steps(self):
        drafts = extract_history_drafts(
            str(TEST_DOCS / "07_故障历史查询文档.docx")
        )

        self.assertEqual(len(drafts), 24)
        self.assertTrue(all(draft.valid for draft in drafts))
        self.assertEqual(
            sum(len(draft.values.get("steps") or []) for draft in drafts), 144
        )
        first = drafts[0]
        self.assertEqual(first.values["case_id"], "HIS-SYN-001")
        self.assertEqual(first.values["hmc_code"], "91810001**0001")
        self.assertEqual(first.values["fmeca_code"], "FM-SYN-001")
        self.assertTrue(first.raw_data.get("steps_with_sources"))


if __name__ == "__main__":
    unittest.main()
