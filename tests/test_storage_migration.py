import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import migrate_minio_to_localfs as migration


class _FakeMinioResponse(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.connection_released = False

    def release_conn(self):
        self.connection_released = True


class _FakeMinio:
    """Small in-memory stand-in for the three MinIO methods migration uses."""

    def __init__(self, objects, *, bucket="documents"):
        self.bucket = bucket
        self.objects = dict(objects)
        self.get_calls = []
        self.responses = []

    def bucket_exists(self, bucket):
        return bucket == self.bucket

    def list_objects(self, bucket, prefix=None, recursive=False):
        if bucket != self.bucket:
            return iter(())
        objects = []
        for storage_key, payload in sorted(self.objects.items()):
            if prefix and not storage_key.startswith(prefix):
                continue
            objects.append(
                SimpleNamespace(
                    object_name=storage_key,
                    size=len(payload),
                    etag=f'etag-{hashlib.sha256(payload).hexdigest()[:16]}',
                    is_dir=False,
                )
            )
        return iter(objects)

    def get_object(self, bucket, storage_key):
        if bucket != self.bucket:
            raise AssertionError(f"unexpected bucket: {bucket}")
        self.get_calls.append(storage_key)
        response = _FakeMinioResponse(self.objects[storage_key])
        self.responses.append(response)
        return response


class StorageMigrationTests(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self._temp_dir.name)
        self.artifact_root = self.workspace / "artifacts"
        self.manifest_path = self.workspace / "migration.jsonl"
        self.legacy_images = self.workspace / "legacy-images"

    def tearDown(self):
        self._temp_dir.cleanup()

    def _base_args(self):
        return [
            "--artifact-root",
            str(self.artifact_root),
            "--manifest",
            str(self.manifest_path),
        ]

    def _run_main(self, args, *, minio_client=None):
        stdout = io.StringIO()
        stderr = io.StringIO()
        client_patch = (
            mock.patch.object(migration, "create_minio_client", return_value=minio_client)
            if minio_client is not None
            else mock.patch.object(
                migration,
                "create_minio_client",
                side_effect=AssertionError("test unexpectedly requested MinIO"),
            )
        )
        with client_patch, redirect_stdout(stdout), redirect_stderr(stderr):
            return_code = migration.main(args)
        return return_code, stdout.getvalue(), stderr.getvalue()

    def _manifest_records(self):
        if not self.manifest_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_legacy_images_dry_run_lists_without_copying_or_reading_bodies(self):
        source_path = self.legacy_images / "doc-001" / "evidence.png"
        source_path.parent.mkdir(parents=True)
        payload = b"legacy image evidence"
        source_path.write_bytes(payload)

        with mock.patch.object(
            migration,
            "open_local_chunks",
            side_effect=AssertionError("dry-run must not read source bodies"),
        ):
            return_code, stdout, stderr = self._run_main(
                self._base_args()
                + [
                    "--skip-minio",
                    "--legacy-images-dir",
                    str(self.legacy_images),
                    "--dry-run",
                ]
            )

        self.assertEqual(return_code, 0, stderr)
        self.assertIn("[DRY-RUN] legacy_images:images/doc-001/evidence.png", stdout)
        self.assertFalse((self.artifact_root / "images/doc-001/evidence.png").exists())
        records = self._manifest_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "dry_run")
        self.assertEqual(records[0]["storage_key"], "images/doc-001/evidence.png")
        self.assertEqual(records[0]["size_bytes"], len(payload))

    def test_minio_migration_writes_content_and_sha256_manifest(self):
        storage_key = "documents/kb-001/doc-001/source.docx"
        payload = b"legacy minio document\x00with binary data"
        fake_minio = _FakeMinio({storage_key: payload})

        return_code, stdout, stderr = self._run_main(
            self._base_args() + ["--skip-legacy-images"],
            minio_client=fake_minio,
        )

        self.assertEqual(return_code, 0, stderr)
        self.assertIn("[MIGRATED]", stdout)
        self.assertEqual((self.artifact_root / storage_key).read_bytes(), payload)
        self.assertEqual(fake_minio.get_calls, [storage_key])
        self.assertTrue(fake_minio.responses[0].closed)
        self.assertTrue(fake_minio.responses[0].connection_released)

        records = self._manifest_records()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["status"], "migrated")
        self.assertEqual(record["source"], "minio")
        self.assertEqual(record["bucket"], "documents")
        self.assertEqual(record["storage_key"], storage_key)
        self.assertEqual(record["size_bytes"], len(payload))
        self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())

    def test_resume_rechecks_manifest_and_destination_without_reading_source(self):
        storage_key = "documents/kb-001/doc-resume/source.pdf"
        payload = b"resume payload"
        fake_minio = _FakeMinio({storage_key: payload})
        args = self._base_args() + ["--skip-legacy-images"]

        first_code, _stdout, first_stderr = self._run_main(args, minio_client=fake_minio)
        self.assertEqual(first_code, 0, first_stderr)
        self.assertEqual(fake_minio.get_calls, [storage_key])

        second_code, second_stdout, second_stderr = self._run_main(
            args + ["--resume"], minio_client=fake_minio
        )

        self.assertEqual(second_code, 0, second_stderr)
        self.assertEqual(fake_minio.get_calls, [storage_key])
        self.assertIn("[RESUME]", second_stdout)
        self.assertIn("[SUMMARY] resumed=1", second_stdout)
        self.assertNotIn("resumed=2", second_stdout)
        records = self._manifest_records()
        self.assertEqual([record["status"] for record in records], ["migrated", "resumed"])
        self.assertEqual(records[-1]["sha256"], hashlib.sha256(payload).hexdigest())

    def test_verify_only_hashes_both_sides_without_writing(self):
        storage_key = "documents/kb-001/doc-verify/source.xlsx"
        payload = b"verify payload"
        fake_minio = _FakeMinio({storage_key: payload})
        args = self._base_args() + ["--skip-legacy-images"]

        migrate_code, _stdout, migrate_stderr = self._run_main(
            args, minio_client=fake_minio
        )
        self.assertEqual(migrate_code, 0, migrate_stderr)
        destination = self.artifact_root / storage_key
        before = destination.stat()

        with mock.patch.object(
            migration,
            "write_atomically",
            side_effect=AssertionError("verify-only must not call write_atomically"),
        ):
            verify_code, verify_stdout, verify_stderr = self._run_main(
                args + ["--verify-only"], minio_client=fake_minio
            )

        self.assertEqual(verify_code, 0, verify_stderr)
        self.assertIn("[VERIFIED]", verify_stdout)
        self.assertEqual(destination.read_bytes(), payload)
        self.assertEqual(destination.stat().st_ino, before.st_ino)
        self.assertEqual(destination.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertEqual(fake_minio.get_calls, [storage_key, storage_key])
        self.assertEqual(self._manifest_records()[-1]["status"], "verified")

    def test_different_existing_content_is_not_overwritten_and_returns_failure(self):
        storage_key = "documents/kb-001/doc-conflict/source.docx"
        existing_payload = b"existing safe data"
        incoming_payload = b"different incoming"
        destination = self.artifact_root / storage_key
        destination.parent.mkdir(parents=True)
        destination.write_bytes(existing_payload)
        fake_minio = _FakeMinio({storage_key: incoming_payload})

        return_code, _stdout, stderr = self._run_main(
            self._base_args() + ["--skip-legacy-images"],
            minio_client=fake_minio,
        )

        self.assertEqual(return_code, 1)
        self.assertIn("destination conflict", stderr)
        self.assertEqual(destination.read_bytes(), existing_payload)
        self.assertEqual(list((self.artifact_root / ".staging").glob("*.part")), [])
        record = self._manifest_records()[-1]
        self.assertEqual(record["status"], "error")
        self.assertIn("destination conflict", record["error"])

    def test_storage_keys_reject_traversal_absolute_and_reserved_namespaces(self):
        invalid_keys = (
            "",
            " ../escape.bin",
            "../escape.bin",
            "safe/../../escape.bin",
            "/absolute.bin",
            "safe//empty.bin",
            "safe/./dot.bin",
            r"windows\escape.bin",
            ".staging/internal.part",
            ".trash/deleted.bin",
            "nul\x00byte.bin",
        )

        for storage_key in invalid_keys:
            with self.subTest(storage_key=repr(storage_key)):
                with self.assertRaises(migration.MigrationError):
                    migration.normalize_storage_key(storage_key)

    def test_unsafe_minio_key_fails_inventory_without_writing_outside_root(self):
        fake_minio = _FakeMinio({"../escaped.bin": b"unsafe"})

        return_code, _stdout, stderr = self._run_main(
            self._base_args() + ["--skip-legacy-images"],
            minio_client=fake_minio,
        )

        self.assertEqual(return_code, 1)
        self.assertIn("unsafe storage key", stderr)
        self.assertFalse((self.workspace / "escaped.bin").exists())
        self.assertEqual(self._manifest_records(), [])

    @unittest.skipUnless(hasattr(Path, "symlink_to"), "platform does not support symlinks")
    def test_destination_parent_symlink_is_rejected_without_touching_target(self):
        outside = self.workspace / "outside"
        outside.mkdir()
        self.artifact_root.mkdir()
        (self.artifact_root / "documents").symlink_to(outside, target_is_directory=True)
        storage_key = "documents/escaped.bin"
        fake_minio = _FakeMinio({storage_key: b"unsafe"})

        return_code, _stdout, stderr = self._run_main(
            self._base_args() + ["--skip-legacy-images"],
            minio_client=fake_minio,
        )

        self.assertEqual(return_code, 1)
        self.assertIn("unsafe destination directory", stderr)
        self.assertFalse((outside / "escaped.bin").exists())
        self.assertEqual(self._manifest_records()[-1]["status"], "error")

    @unittest.skipUnless(hasattr(Path, "symlink_to"), "platform does not support symlinks")
    def test_artifact_root_itself_cannot_be_a_symlink(self):
        real_root = self.workspace / "real-artifacts"
        real_root.mkdir()
        linked_root = self.workspace / "linked-artifacts"
        linked_root.symlink_to(real_root, target_is_directory=True)
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            mock.patch.object(
                migration,
                "create_minio_client",
                side_effect=AssertionError("unsafe root must fail before source access"),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            with self.assertRaisesRegex(
                migration.MigrationError, "artifact root must not be a symlink"
            ):
                migration.main(
                    [
                        "--artifact-root",
                        str(linked_root),
                        "--manifest",
                        str(self.manifest_path),
                        "--skip-legacy-images",
                    ]
                )

        self.assertEqual(list(real_root.iterdir()), [])
        self.assertFalse(self.manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
