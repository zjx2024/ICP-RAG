import unittest
from unittest.mock import MagicMock, patch

from backend.agent.tools import icp_diagnosis as diagnosis_module


class FtaReferenceRelevanceTests(unittest.TestCase):
    def test_exact_business_link_does_not_invent_query_relevance_score(self):
        cursor = MagicMock()
        cursor.__enter__.return_value = cursor
        cursor.__exit__.return_value = False
        cursor.description = [("code",), ("fault_mode",), ("doc_id",)]
        cursor.fetchall.return_value = [("FM-SYN-002", "输入滤波性能退化", "doc-fmeca")]
        connection = MagicMock()
        connection.cursor.return_value = cursor

        with patch.object(diagnosis_module.postgres_client, "conn", connection):
            rows = diagnosis_module._query_fmeca_by_codes(["FM-SYN-002"], 3)

        self.assertEqual("exact_fta_reference", rows[0]["match_type"])
        self.assertIsNone(rows[0]["retrieval_match_score"])


if __name__ == "__main__":
    unittest.main()
