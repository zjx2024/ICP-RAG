import hashlib
import io
import errno
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from backend.storage.artifact_store import (
    ArtifactNotFoundError,
    ArtifactStoreError,
    InvalidStorageKeyError,
    LocalFSArtifactStore,
    safe_original_name,
    source_artifact_key,
)


class LocalFSArtifactStoreTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self._temp_dir.name)
        self.root = self.workspace / "artifacts"
        self.store = LocalFSArtifactStore(self.root)

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_upload_read_stat_and_hash_bytes(self):
        payload = "ICP 原始文档证据\n".encode("utf-8") + bytes(range(32))
        storage_key = "icp/hmc/doc-001/故障代码手册.bin"

        metadata = self.store.upload_file(
            storage_key,
            payload,
            content_type="application/octet-stream",
            original_name="故障代码手册.bin",
        )

        self.assertEqual(metadata.storage_key, storage_key)
        self.assertEqual(metadata.size_bytes, len(payload))
        self.assertEqual(metadata.sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(metadata.content_type, "application/octet-stream")
        self.assertEqual(metadata.original_name, "故障代码手册.bin")
        self.assertTrue(self.store.exists(storage_key))
        self.assertEqual(self.store.get_file(storage_key), payload)

        stat = self.store.stat(storage_key)
        self.assertEqual(stat.storage_key, storage_key)
        self.assertEqual(stat.size_bytes, len(payload))
        self.assertEqual(stat.sha256, hashlib.sha256(payload).hexdigest())

        with self.store.open_file(storage_key) as artifact_file:
            self.assertEqual(artifact_file.read(), payload)

    def test_upload_accepts_seekable_binary_stream(self):
        payload = b"streamed artifact data"
        stream = io.BytesIO(payload)

        metadata = self.store.upload_file("documents/doc-002/source.docx", stream)

        self.assertEqual(metadata.size_bytes, len(payload))
        self.assertEqual(metadata.sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(self.store.get_file(metadata.storage_key), payload)

    def test_zero_byte_artifact_has_stable_hash(self):
        metadata = self.store.upload_file("documents/empty.bin", b"")

        self.assertEqual(metadata.size_bytes, 0)
        self.assertEqual(metadata.sha256, hashlib.sha256(b"").hexdigest())
        self.assertEqual(self.store.get_file(metadata.storage_key), b"")

    def test_write_fsyncs_before_atomic_replace(self):
        payload = b"complete artifact"
        events = []
        replace_arguments = []
        real_fsync = os.fsync
        real_replace = os.replace

        def recording_fsync(file_descriptor):
            events.append("fsync")
            return real_fsync(file_descriptor)

        def recording_replace(source, destination):
            source_path = Path(source)
            self.assertTrue(source_path.is_file())
            events.append("replace")
            replace_arguments.append((source_path, Path(destination)))
            return real_replace(source, destination)

        with (
            mock.patch("backend.storage.artifact_store.os.fsync", side_effect=recording_fsync),
            mock.patch("backend.storage.artifact_store.os.replace", side_effect=recording_replace),
        ):
            self.store.upload_file("documents/atomic.bin", payload)

        self.assertIn("replace", events)
        self.assertIn("fsync", events)
        self.assertLess(events.index("fsync"), events.index("replace"))
        self.assertEqual(len(replace_arguments), 1)
        staging_path, final_path = replace_arguments[0]
        self.assertNotEqual(staging_path, final_path)
        self.assertFalse(staging_path.exists())
        self.assertEqual(final_path.read_bytes(), payload)

    def test_failed_atomic_replace_preserves_old_content_and_cleans_staging(self):
        storage_key = "documents/replaced.bin"
        original = b"old complete artifact"
        self.store.upload_file(storage_key, original)
        attempted_staging_paths = []

        def fail_replace(source, _destination):
            attempted_staging_paths.append(Path(source))
            raise OSError("simulated replace failure")

        with mock.patch(
            "backend.storage.artifact_store.os.replace", side_effect=fail_replace
        ):
            with self.assertRaises(ArtifactStoreError):
                self.store.upload_file(storage_key, b"new incomplete artifact")

        self.assertEqual(self.store.get_file(storage_key), original)
        self.assertEqual(len(attempted_staging_paths), 1)
        self.assertFalse(attempted_staging_paths[0].exists())

    def test_concurrent_writes_never_expose_mixed_content(self):
        storage_key = "documents/concurrent.bin"
        payloads = (b"A" * (1024 * 1024 + 17), b"B" * (1024 * 1024 + 31))

        with ThreadPoolExecutor(max_workers=2) as executor:
            metadata = list(
                executor.map(
                    lambda payload: self.store.upload_file(storage_key, payload),
                    payloads,
                )
            )

        self.assertEqual(
            {item.sha256 for item in metadata},
            {hashlib.sha256(payload).hexdigest() for payload in payloads},
        )
        self.assertIn(self.store.get_file(storage_key), payloads)

    def test_concurrent_first_writes_can_create_the_same_parent(self):
        shared_parent = self.root / "brand-new-parent"
        barrier = threading.Barrier(2, timeout=5)
        real_exists = Path.exists

        def synchronize_first_parent_check(path):
            exists = real_exists(path)
            if path == shared_parent and not exists:
                barrier.wait()
            return exists

        with (
            mock.patch.object(Path, "exists", autospec=True, side_effect=synchronize_first_parent_check),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            futures = [
                executor.submit(
                    self.store.upload_file,
                    f"brand-new-parent/{index}.bin",
                    str(index).encode("ascii"),
                )
                for index in range(2)
            ]
            metadata = [future.result() for future in futures]

        self.assertEqual({item.storage_key for item in metadata}, {
            "brand-new-parent/0.bin",
            "brand-new-parent/1.bin",
        })
        self.assertEqual(self.store.get_file("brand-new-parent/0.bin"), b"0")
        self.assertEqual(self.store.get_file("brand-new-parent/1.bin"), b"1")

    def test_read_range_uses_start_and_length_and_truncates_at_eof(self):
        storage_key = "documents/range.bin"
        payload = b"0123456789"
        self.store.upload_file(storage_key, payload)

        self.assertEqual(self.store.read_range(storage_key), payload)
        self.assertEqual(self.store.read_range(storage_key, start=2), b"23456789")
        self.assertEqual(self.store.read_range(storage_key, start=2, length=4), b"2345")
        self.assertEqual(self.store.read_range(storage_key, start=8, length=20), b"89")
        self.assertEqual(self.store.read_range(storage_key, start=10), b"")
        self.assertEqual(self.store.read_range(storage_key, start=20, length=3), b"")
        self.assertEqual(self.store.read_range(storage_key, start=3, length=0), b"")

        with self.assertRaises(ValueError):
            self.store.read_range(storage_key, start=-1)
        with self.assertRaises(ValueError):
            self.store.read_range(storage_key, length=-1)

    def test_delete_is_idempotent_and_missing_reads_are_explicit(self):
        storage_key = "documents/delete-me.bin"
        payload = b"evidence"
        self.store.upload_file(storage_key, payload)

        self.assertTrue(self.store.delete_file(storage_key))
        self.assertFalse(self.store.exists(storage_key))
        self.assertFalse(self.store.delete_file(storage_key))
        trash_files = [path for path in (self.root / ".trash").rglob("*") if path.is_file()]
        self.assertEqual(len(trash_files), 1)
        self.assertEqual(trash_files[0].read_bytes(), payload)

        for operation in (
            lambda: self.store.get_file(storage_key),
            lambda: self.store.stat(storage_key),
            lambda: self.store.read_range(storage_key),
            lambda: self.store.open_file(storage_key),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(ArtifactNotFoundError):
                    operation()

    def test_two_concurrent_deletes_are_idempotent(self):
        storage_key = "documents/concurrent-delete.bin"
        self.store.upload_file(storage_key, b"evidence")
        source_path = self.root / storage_key
        barrier = threading.Barrier(2, timeout=5)
        real_replace = os.replace

        def synchronized_replace(source, destination):
            if Path(source) == source_path:
                barrier.wait()
            return real_replace(source, destination)

        with (
            mock.patch("backend.storage.artifact_store.os.replace", side_effect=synchronized_replace),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = list(executor.map(lambda _index: self.store.delete_file(storage_key), range(2)))

        self.assertEqual(sorted(results), [False, True])

    def test_delete_file_never_moves_a_directory_tree(self):
        self.store.upload_file("documents/doc-1/source.txt", b"evidence")

        with self.assertRaises(ArtifactStoreError):
            self.store.delete_file("documents/doc-1")

        self.assertEqual(
            self.store.get_file("documents/doc-1/source.txt"),
            b"evidence",
        )

    def test_delete_prefix_only_removes_regular_files_below_that_prefix(self):
        removed = {
            "images/doc-001/a.png": b"image-a",
            "images/doc-001/nested/b.png": b"image-b",
        }
        retained = {
            "images/doc-010/keep.png": b"other-document",
            "documents/doc-001/source.docx": b"original-document",
        }
        for storage_key, payload in {**removed, **retained}.items():
            self.store.upload_file(storage_key, payload)

        self.assertEqual(self.store.delete_prefix("images/doc-001"), 2)
        self.assertEqual(self.store.delete_prefix("images/doc-001"), 0)

        for storage_key in removed:
            self.assertFalse(self.store.exists(storage_key))
        for storage_key, payload in retained.items():
            self.assertEqual(self.store.get_file(storage_key), payload)

        trash_payloads = {
            path.read_bytes()
            for path in (self.root / ".trash").rglob("*")
            if path.is_file()
        }
        self.assertEqual(trash_payloads, set(removed.values()))

    def test_delete_prefix_rejects_unsafe_prefix_and_does_not_follow_symlinks(self):
        outside = self.workspace / "outside-prefix"
        outside.mkdir()
        outside_file = outside / "keep.bin"
        outside_file.write_bytes(b"outside")

        self.store.upload_file("images/doc-001/owned.bin", b"owned")
        (self.root / "images" / "doc-001" / "external-link").symlink_to(
            outside, target_is_directory=True
        )

        self.assertEqual(self.store.delete_prefix("images/doc-001"), 1)
        self.assertEqual(outside_file.read_bytes(), b"outside")
        with self.assertRaises(InvalidStorageKeyError):
            self.store.delete_prefix("")
        with self.assertRaises(InvalidStorageKeyError):
            self.store.delete_prefix("../outside-prefix")

    def test_invalid_storage_keys_are_rejected_without_writing_outside_root(self):
        invalid_keys = (
            "",
            ".",
            "..",
            "../escape.bin",
            "safe/../../escape.bin",
            "/absolute/path.bin",
            "double//separator.bin",
            "./relative-dot.bin",
            "safe/./relative-dot.bin",
            ".staging/internal.bin",
            ".trash/internal.bin",
            "windows\\escape.bin",
            "C:\\absolute.bin",
            "nul\x00byte.bin",
            " documents/leading-space.bin",
            "documents/trailing-space.bin ",
        )

        for storage_key in invalid_keys:
            with self.subTest(storage_key=repr(storage_key)):
                with self.assertRaises(InvalidStorageKeyError):
                    self.store.upload_file(storage_key, b"must not be written")
                with self.assertRaises(InvalidStorageKeyError):
                    self.store.get_file(storage_key)

        self.assertFalse((self.workspace / "escape.bin").exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "platform does not support symlinks")
    def test_symlink_cannot_escape_storage_root(self):
        outside = self.workspace / "outside"
        outside.mkdir()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "escape-link").symlink_to(outside, target_is_directory=True)

        with self.assertRaises(InvalidStorageKeyError):
            self.store.upload_file("escape-link/new-dir/escaped.bin", b"must stay contained")

        self.assertFalse((outside / "new-dir").exists())

    def test_materialize_creates_scoped_temporary_file(self):
        storage_key = "documents/materialize.bin"
        payload = b"parser input"
        self.store.upload_file(storage_key, payload)

        with self.store.materialize(storage_key, suffix=".docx") as local_path:
            materialized_path = Path(local_path)
            self.assertTrue(materialized_path.is_file())
            self.assertEqual(materialized_path.suffix, ".docx")
            self.assertEqual(materialized_path.read_bytes(), payload)

        self.assertFalse(materialized_path.exists())

    def test_materialize_same_suffix_is_a_snapshot_during_delete(self):
        storage_key = "documents/materialize.docx"
        payload = b"stable parser snapshot"
        self.store.upload_file(storage_key, payload)

        with self.store.materialize(storage_key, suffix=".docx") as local_path:
            materialized_path = Path(local_path)
            self.assertNotEqual(materialized_path, self.root / storage_key)
            self.assertTrue(self.store.delete_file(storage_key))
            self.assertEqual(materialized_path.read_bytes(), payload)

        self.assertFalse(materialized_path.exists())

    def test_directory_fsync_io_error_is_not_reported_as_success(self):
        # Pre-create the final parent so the injected failure happens after
        # atomic publication rather than during parent creation.
        self.store.upload_file("documents/fsync/existing.bin", b"existing")
        failure = OSError(errno.EIO, "simulated directory durability failure")

        with mock.patch.object(self.store, "_fsync_dir", side_effect=failure):
            with self.assertRaises(ArtifactStoreError):
                self.store.upload_file("documents/fsync/new.bin", b"new")

    def test_generated_source_key_never_contains_the_user_filename(self):
        self.assertEqual(safe_original_name("../../维修手册.DOCX"), "维修手册.DOCX")
        self.assertEqual(safe_original_name(r"C:\upload\维修手册.DOCX"), "维修手册.DOCX")
        self.assertEqual(safe_original_name(".."), "document")

        key = source_artifact_key("icp/hmc", "doc-001", "../../维修手册.DOCX")
        self.assertEqual(key, "icp/hmc/doc-001/source.docx")
        self.assertNotIn("维修手册", key)
        self.assertEqual(
            source_artifact_key("icp/hmc", "doc-002", "payload.不可信扩展名"),
            "icp/hmc/doc-002/source",
        )


if __name__ == "__main__":
    unittest.main()
