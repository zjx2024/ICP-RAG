import unittest
from unittest import mock

from backend.storage.registration import (
    ArtifactRegistrationUncertainError,
    confirm_registration_or_compensate,
)


class ArtifactRegistrationCompensationTests(unittest.TestCase):
    def test_visible_row_after_insert_error_keeps_artifact(self):
        rollback = mock.Mock()
        with mock.patch("backend.storage.registration.artifact_store") as store:
            committed = confirm_registration_or_compensate(
                doc_id="doc-1",
                storage_key="documents/kb/doc-1/source.docx",
                lookup=lambda _doc_id: {"doc_id": "doc-1"},
                rollback=rollback,
            )

        self.assertTrue(committed)
        rollback.assert_called_once_with()
        store.delete_file.assert_not_called()

    def test_confirmed_absent_row_is_compensated(self):
        with mock.patch("backend.storage.registration.artifact_store") as store:
            committed = confirm_registration_or_compensate(
                doc_id="doc-2",
                storage_key="icp/hmc/doc-2/source.xlsx",
                lookup=lambda _doc_id: None,
                rollback=mock.Mock(),
            )

        self.assertFalse(committed)
        store.delete_file.assert_called_once_with("icp/hmc/doc-2/source.xlsx")

    def test_unverifiable_commit_retains_artifact(self):
        def unavailable(_doc_id):
            raise ConnectionError("database unavailable")

        with mock.patch("backend.storage.registration.artifact_store") as store:
            with self.assertRaises(ArtifactRegistrationUncertainError):
                confirm_registration_or_compensate(
                    doc_id="doc-3",
                    storage_key="icp/rbd/doc-3/source.png",
                    lookup=unavailable,
                    rollback=mock.Mock(),
                )

        store.delete_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
