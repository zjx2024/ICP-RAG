import unittest
from unittest.mock import patch

from backend.core.parsers import ParserRegistry
from backend.icp import document_types as document_types_module
from backend.icp.document_types import DOCUMENT_TYPES


class _FailingConnection:
    def cursor(self):
        raise RuntimeError("registry unavailable")

    def rollback(self):
        pass


class DocumentTypeParserContractTests(unittest.TestCase):
    def test_registry_normalization_does_not_hide_database_failure(self):
        with patch.object(
            document_types_module.postgres_client,
            "conn",
            _FailingConnection(),
        ):
            with self.assertRaisesRegex(RuntimeError, "registry unavailable"):
                document_types_module.normalize_document_types()

    def test_exactly_seven_registered_upload_types_match_parser_validation(self):
        configured = {row[0]: row for row in DOCUMENT_TYPES}
        self.assertEqual(
            {
                "function_locator",
                "hmc",
                "fta",
                "rbd",
                "fmeca",
                "mfl",
                "fault_history",
            },
            set(configured),
        )

        for doc_type, row in configured.items():
            parser = ParserRegistry.get(doc_type)
            extensions = [value.strip() for value in row[4].split(",") if value.strip()]
            self.assertTrue(extensions, doc_type)
            for extension in extensions:
                self.assertEqual(
                    [],
                    parser.validate(f"contract{extension}"),
                    f"{doc_type} upload registry accepts {extension}, but its parser rejects it",
                )


if __name__ == "__main__":
    unittest.main()
