import unittest
from unittest.mock import patch

from backend.agent import document_semantics
from backend.icp.document_types import DOCUMENT_TYPES


class DocumentSemanticsTests(unittest.TestCase):
    def test_catalog_contains_exactly_the_seven_product_sources(self):
        prompt = document_semantics.document_semantics_prompt()
        lines = [line for line in prompt.splitlines() if line.startswith("- ")]
        self.assertEqual(7, len(lines))
        self.assertEqual([f"- {row[0]}｜{row[1]}：{row[2]}" for row in DOCUMENT_TYPES], lines)

    def test_catalog_is_derived_not_a_second_source_name_registry(self):
        changed = [(row[0], row[1] + "（当前版本）", *row[2:]) for row in DOCUMENT_TYPES]
        with patch.object(document_semantics, "DOCUMENT_TYPES", changed):
            prompt = document_semantics.document_semantics_prompt()
        for row in changed:
            self.assertIn(row[1], prompt)


if __name__ == "__main__":
    unittest.main()
