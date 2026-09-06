import hashlib
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.core.parsers.base import ParseContext
from backend.core.parsers.diagram_extraction import DiagramExtraction
import backend.core.parsers.diagram as diagram_module


def _blocks(storage_key="images/doc-1/source.png"):
    return [
        (
            "section-1",
            {
                "title": "独立工程图章节",
                "text": "图像上下文",
                "tables": [],
                "images": [
                    {
                        "title": "独立工程图",
                        "storage_key": storage_key,
                        "image_url": "/images/doc-1/source.png",
                    }
                ],
            },
        )
    ]


def _deterministic_extraction():
    return DiagramExtraction(
        diagram_type="fta",
        business_key="FTA-DET-1",
        title="显式表格故障树",
        facts={"source_section": "显式结构章节"},
        candidate={
            "source_kind": "table_edges",
            "source_fingerprint": "d" * 64,
            "source_locator": "fta.docx#table=1",
            "graph": {
                "diagram_type": "fta",
                "root_fault_id": "top-det",
                "vertices": [
                    {
                        "id": "top-det",
                        "label": "TopEvent",
                        "properties": {"name": "显式顶事件"},
                    }
                ],
                "edges": [],
            },
        },
    )


def _vision_graph():
    return {
        "root_id": "top-image",
        "description": "视觉候选",
        "confidence": 0.8,
        "vertices": [
            {
                "id": "top-image",
                "label": "TopEvent",
                "properties": {"name": "图像顶事件"},
            }
        ],
        "edges": [],
    }


def _rbd_expression_graph():
    return {
        "diagram_type": "rbd",
        "metadata": {"topology_source": "structured_expression"},
        "vertices": [
            {"id": "START", "label": "Start", "properties": {"name": "START"}},
            {
                "id": "PSU-A",
                "label": "ReliabilityBlock",
                "properties": {"name": "PSU-A", "reliability": 0.99},
            },
            {"id": "END", "label": "End", "properties": {"name": "END"}},
        ],
        "edges": [
            {"id": "e1", "label": "series", "source": "START", "target": "PSU-A"},
            {"id": "e2", "label": "series", "source": "PSU-A", "target": "END"},
        ],
    }


def _rbd_expression_extraction():
    return DiagramExtraction(
        diagram_type="rbd",
        business_key="RBD:电源保障链可靠性模型",
        title="电源保障链可靠性模型",
        facts={"source_section": "2 电源保障链可靠性模型"},
        candidate={
            "source_kind": "structured_expression",
            "source_fingerprint": "e" * 64,
            "source_locator": "rbd.docx#2 电源保障链可靠性模型",
            "confidence": 1.0,
            "graph": _rbd_expression_graph(),
        },
    )


def _matching_rbd_blocks(storage_key="images/doc-1/rbd-power.png"):
    return [
        (
            "2-power",
            {
                "title": "2 电源保障链可靠性模型",
                "text": "结构表达式：SERIES(PSU-A)",
                "tables": [],
                "images": [
                    {
                        "title": "图2-1 电源保障链可靠性模型",
                        "storage_key": storage_key,
                        "image_url": "/images/doc-1/rbd-power.png",
                    }
                ],
            },
        )
    ]


def _rbd_vision_graph():
    return {
        "description": "图片中的电源保障链",
        "confidence": 0.95,
        "vertices": [
            {"id": "vision-start", "label": "Start", "properties": {"name": "输入"}},
            {
                "id": "vision-psu",
                "label": "ReliabilityBlock",
                "properties": {"name": "PSU-A"},
            },
            {"id": "vision-end", "label": "End", "properties": {"name": "成功"}},
        ],
        "edges": [
            {"id": "ve1", "label": "series", "source": "vision-start", "target": "vision-psu"},
            {"id": "ve2", "label": "series", "source": "vision-psu", "target": "vision-end"},
        ],
    }


class DiagramParserSnapshotTests(unittest.TestCase):
    def _context(self, doc_type="fta"):
        return ParseContext(
            doc_id="doc-1",
            doc_type=doc_type,
            file_name=f"{doc_type}.docx",
        )

    def _parser(self, doc_type="fta"):
        parser = diagram_module.DiagramParser()
        parser.doc_type = doc_type
        return parser

    def _patch_common(self, stack, parser, *, path, artifact, repository, postgres):
        stack.enter_context(patch.object(parser, "_load_blocks", return_value=_blocks()))
        stack.enter_context(patch.object(parser, "_index", return_value=0))
        stack.enter_context(patch.object(diagram_module, "load_document_ir", return_value=object()))
        stack.enter_context(patch.object(diagram_module, "artifact_store", artifact))
        stack.enter_context(patch.object(diagram_module, "diagram_repository", repository))
        stack.enter_context(patch.object(diagram_module, "postgres_client", postgres))
        artifact.materialize.return_value = nullcontext(str(path))

    def test_graph_search_locator_carries_stable_postgres_diagram_id(self):
        parser = self._parser("fta")
        with patch.object(parser, "_index", return_value=1) as index:
            count = parser._index_graph(
                ctx=self._context("fta"),
                image_id="image-1",
                image_title="运行中重复复位故障树",
                analysis={"description": "原因链", "confidence": 0.9},
                vertices=[{"id": "top", "properties": {"name": "重复复位"}}],
                edges=[],
                image_url="/images/doc-1/image-1.png",
                block_key="section-1",
                diagram_id="diagram-pg-1",
            )

        self.assertEqual(1, count)
        extra = index.call_args.args[5]
        self.assertEqual("diagram-pg-1", extra["diagram_id_kwd"])
        self.assertEqual("diagram_locator", extra["projection_role_kwd"])

    def test_vision_failure_keeps_stable_original_evidence_and_converges_snapshot(self):
        image_bytes = b"same-engineering-image"
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        events = []
        artifact = MagicMock()
        artifact.stat.return_value = SimpleNamespace(sha256=image_sha)
        postgres = MagicMock()
        postgres.insert_image.side_effect = lambda *_args, **_kwargs: events.append("image")
        repository = MagicMock()
        repository.upsert_diagram.side_effect = lambda **_kwargs: (
            events.append("diagram") or {"diagram_id": "diagram-image"}
        )
        repository.upsert_candidate.side_effect = lambda **_kwargs: (
            events.append("failed_candidate")
            or {"candidate_id": "candidate-failed-vision"}
        )
        repository.create_or_reuse_revision.side_effect = lambda **_kwargs: (
            events.append("revision")
            or {
                "revision_id": "revision-failed-vision",
                "canonical_graph": {"vertices": [], "edges": []},
                "review_status": "pending",
                "projection_status": "not_projected",
            }
        )
        repository.archive_missing_diagrams.side_effect = lambda **_kwargs: (
            events.append("archive")
            or {"archived_diagram_ids": [], "stale_revision_ids": []}
        )

        def fail_vision(**_kwargs):
            events.append("vision")
            raise RuntimeError("vision unavailable")

        parser = self._parser()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(image_bytes)
            with ExitStack() as stack:
                self._patch_common(
                    stack,
                    parser,
                    path=path,
                    artifact=artifact,
                    repository=repository,
                    postgres=postgres,
                )
                stack.enter_context(
                    patch.object(diagram_module, "extract_fta_from_ir", return_value=[])
                )
                stack.enter_context(
                    patch.object(diagram_module, "analyze_icp_diagram", side_effect=fail_vision)
                )
                result = parser.parse("fta.docx", self._context())

        expected_image_id = f"doc-1_img_{image_sha}"
        self.assertEqual(expected_image_id, postgres.insert_image.call_args.args[0])
        self.assertIsNone(postgres.insert_image.call_args.args[2])
        self.assertEqual(
            [
                "image",
                "diagram",
                "vision",
                "failed_candidate",
                "revision",
                "archive",
            ],
            events,
        )
        failed_candidate = repository.upsert_candidate.call_args.kwargs
        self.assertEqual("failed", failed_candidate["extraction_status"])
        self.assertEqual([], failed_candidate["raw_graph"]["vertices"])
        self.assertEqual([], failed_candidate["raw_graph"]["edges"])
        self.assertIn(
            "vision unavailable",
            failed_candidate["source_locator"]["extraction_error"],
        )
        self.assertEqual(
            ["candidate-failed-vision"],
            repository.create_or_reuse_revision.call_args.kwargs["candidate_ids"],
        )
        self.assertEqual(
            ["diagram-image"],
            repository.archive_missing_diagrams.call_args.kwargs["present_diagram_ids"],
        )
        self.assertFalse(repository.archive_missing_diagrams.call_args.kwargs["allow_empty"])
        self.assertEqual(1, result.stats["present_diagrams"])
        self.assertIn("视觉解析失败", result.stats["failed"][0]["reason"])

    def test_repeat_parse_reuses_image_identity_and_both_diagram_master_keys(self):
        image_bytes = b"repeatable-engineering-image"
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        artifact = MagicMock()
        artifact.stat.return_value = SimpleNamespace(sha256=image_sha)
        postgres = MagicMock()
        repository = MagicMock()
        diagram_keys = []

        def upsert_diagram(**kwargs):
            diagram_keys.append(kwargs["diagram_key"])
            return {
                "diagram_id": (
                    "diagram-deterministic"
                    if kwargs["diagram_key"] == "FTA-DET-1"
                    else "diagram-image"
                )
            }

        def upsert_candidate(**kwargs):
            return {
                "candidate_id": (
                    "candidate-deterministic"
                    if kwargs["source_kind"] == "table_edges"
                    else "candidate-image"
                )
            }

        repository.upsert_diagram.side_effect = upsert_diagram
        repository.upsert_candidate.side_effect = upsert_candidate
        repository.create_or_reuse_revision.return_value = {
            "revision_id": "revision-stable",
            "canonical_graph": _vision_graph(),
            "review_status": "pending",
            "projection_status": "not_projected",
        }
        repository.archive_missing_diagrams.return_value = {
            "archived_diagram_ids": [],
            "stale_revision_ids": [],
        }
        parser = self._parser()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(image_bytes)
            with ExitStack() as stack:
                self._patch_common(
                    stack,
                    parser,
                    path=path,
                    artifact=artifact,
                    repository=repository,
                    postgres=postgres,
                )
                stack.enter_context(
                    patch.object(
                        diagram_module,
                        "extract_fta_from_ir",
                        return_value=[_deterministic_extraction()],
                    )
                )
                stack.enter_context(
                    patch.object(
                        diagram_module,
                        "analyze_icp_diagram",
                        return_value=_vision_graph(),
                    )
                )
                first = parser.parse("fta.docx", self._context())
                second = parser.parse("fta.docx", self._context())

        image_ids = [call.args[0] for call in postgres.insert_image.call_args_list]
        self.assertEqual([f"doc-1_img_{image_sha}"] * 2, image_ids)
        self.assertEqual(
            ["FTA-DET-1", f"IMAGE:{image_sha}"] * 2,
            diagram_keys,
        )
        self.assertEqual(4, repository.upsert_candidate.call_count)
        self.assertEqual(4, repository.create_or_reuse_revision.call_count)
        self.assertEqual(2, repository.archive_missing_diagrams.call_count)
        for call in repository.archive_missing_diagrams.call_args_list:
            self.assertEqual(
                ["diagram-deterministic", "diagram-image"],
                call.kwargs["present_diagram_ids"],
            )
            self.assertFalse(call.kwargs["allow_empty"])
        self.assertEqual(2, first.stats["present_diagrams"])
        self.assertEqual(2, second.stats["present_diagrams"])

    def test_candidate_or_revision_failure_never_archives_previous_snapshot(self):
        image_bytes = b"authority-write-failure"
        image_sha = hashlib.sha256(image_bytes).hexdigest()

        for failure_stage in ("candidate", "revision"):
            with self.subTest(failure_stage=failure_stage):
                artifact = MagicMock()
                artifact.stat.return_value = SimpleNamespace(sha256=image_sha)
                postgres = MagicMock()
                repository = MagicMock()
                repository.upsert_diagram.return_value = {"diagram_id": "diagram-image"}
                repository.upsert_candidate.return_value = {"candidate_id": "candidate-image"}
                repository.create_or_reuse_revision.return_value = {
                    "revision_id": "revision-image",
                    "canonical_graph": _vision_graph(),
                    "review_status": "pending",
                    "projection_status": "not_projected",
                }
                if failure_stage == "candidate":
                    repository.upsert_candidate.side_effect = RuntimeError("candidate PG failed")
                else:
                    repository.create_or_reuse_revision.side_effect = RuntimeError(
                        "revision PG failed"
                    )
                parser = self._parser()

                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "image.png"
                    path.write_bytes(image_bytes)
                    with ExitStack() as stack:
                        self._patch_common(
                            stack,
                            parser,
                            path=path,
                            artifact=artifact,
                            repository=repository,
                            postgres=postgres,
                        )
                        stack.enter_context(
                            patch.object(diagram_module, "extract_fta_from_ir", return_value=[])
                        )
                        stack.enter_context(
                            patch.object(
                                diagram_module,
                                "analyze_icp_diagram",
                                return_value=_vision_graph(),
                            )
                        )
                        with self.assertRaisesRegex(RuntimeError, "PG failed"):
                            parser.parse("fta.docx", self._context())

                postgres.insert_image.assert_called_once()
                repository.archive_missing_diagrams.assert_not_called()

    def test_graph_free_parse_cannot_explicitly_empty_previous_snapshot(self):
        repository = MagicMock()
        parser = self._parser()
        blocks = [
            (
                "section-1",
                {
                    "title": "仅说明文字",
                    "text": "未包含工程图或显式结构。",
                    "tables": [],
                    "images": [],
                },
            )
        ]

        with patch.object(parser, "_load_blocks", return_value=blocks), patch.object(
            parser, "_index", return_value=0
        ), patch.object(diagram_module, "load_document_ir", return_value=object()), patch.object(
            diagram_module, "extract_fta_from_ir", return_value=[]
        ), patch.object(diagram_module, "diagram_repository", repository):
            with self.assertRaisesRegex(ValueError, "保留上一版图快照"):
                parser.parse("fta.docx", self._context())

        repository.archive_missing_diagrams.assert_not_called()

    def test_rbd_image_uses_the_same_evidence_and_snapshot_boundary(self):
        image_bytes = b"rbd-engineering-image"
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        artifact = MagicMock()
        artifact.stat.return_value = SimpleNamespace(sha256=image_sha)
        postgres = MagicMock()
        repository = MagicMock()
        repository.upsert_diagram.return_value = {"diagram_id": "diagram-rbd"}
        repository.upsert_candidate.return_value = {"candidate_id": "candidate-rbd"}
        repository.create_or_reuse_revision.return_value = {
            "revision_id": "revision-rbd",
            "canonical_graph": _vision_graph(),
            "review_status": "pending",
            "projection_status": "not_projected",
        }
        repository.archive_missing_diagrams.return_value = {
            "archived_diagram_ids": [],
            "stale_revision_ids": [],
        }
        parser = self._parser("rbd")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(image_bytes)
            with ExitStack() as stack:
                self._patch_common(
                    stack,
                    parser,
                    path=path,
                    artifact=artifact,
                    repository=repository,
                    postgres=postgres,
                )
                stack.enter_context(
                    patch.object(diagram_module, "extract_rbd_from_ir", return_value=[])
                )
                stack.enter_context(
                    patch.object(
                        diagram_module,
                        "analyze_icp_diagram",
                        return_value=_vision_graph(),
                    )
                )
                result = parser.parse("rbd.docx", self._context("rbd"))

        self.assertEqual("rbd", repository.upsert_diagram.call_args.kwargs["diagram_type"])
        self.assertEqual("rbd_diagram", postgres.insert_image.call_args.args[6])
        self.assertEqual(
            ["diagram-rbd"],
            repository.archive_missing_diagrams.call_args.kwargs["present_diagram_ids"],
        )
        self.assertEqual(1, result.stats["graph_count"])

    def test_complete_rbd_expression_excludes_matched_visual_topology_from_canonical(self):
        image_bytes = b"rbd-power-topology"
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        artifact = MagicMock()
        artifact.stat.return_value = SimpleNamespace(sha256=image_sha)
        postgres = MagicMock()
        repository = MagicMock()
        repository.upsert_diagram.return_value = {"diagram_id": "diagram-rbd-power"}

        def upsert_candidate(**kwargs):
            return {
                "candidate_id": (
                    "candidate-expression"
                    if kwargs["source_kind"] == "structured_expression"
                    else "candidate-vision"
                )
            }

        repository.upsert_candidate.side_effect = upsert_candidate
        repository.create_or_reuse_revision.return_value = {
            "revision_id": "revision-rbd-power",
            "canonical_graph": _rbd_expression_graph(),
            "review_status": "pending",
            "projection_status": "not_projected",
        }
        repository.archive_missing_diagrams.return_value = {
            "archived_diagram_ids": [],
            "stale_revision_ids": [],
        }
        parser = self._parser("rbd")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rbd-power.png"
            path.write_bytes(image_bytes)
            with ExitStack() as stack:
                self._patch_common(
                    stack,
                    parser,
                    path=path,
                    artifact=artifact,
                    repository=repository,
                    postgres=postgres,
                )
                stack.enter_context(
                    patch.object(
                        parser,
                        "_load_blocks",
                        return_value=_matching_rbd_blocks(),
                    )
                )
                stack.enter_context(
                    patch.object(
                        diagram_module,
                        "extract_rbd_from_ir",
                        return_value=[_rbd_expression_extraction()],
                    )
                )
                stack.enter_context(
                    patch.object(
                        diagram_module,
                        "analyze_icp_diagram",
                        return_value=_rbd_vision_graph(),
                    )
                )
                result = parser.parse("rbd.docx", self._context("rbd"))

        self.assertEqual(1, repository.upsert_diagram.call_count)
        self.assertEqual(2, repository.upsert_candidate.call_count)
        revision_call = repository.create_or_reuse_revision.call_args
        self.assertEqual(
            ["candidate-expression"],
            revision_call.kwargs["candidate_ids"],
        )
        self.assertEqual(
            ["candidate-vision"],
            revision_call.kwargs["isolated_candidate_ids"],
        )
        self.assertEqual(1, result.stats["graph_count"])
        self.assertEqual(3, result.stats["nodes"])
        self.assertEqual(2, result.stats["edges"])

    def test_visual_is_not_isolated_when_deterministic_candidate_failed_constraints(self):
        canonical, isolated = diagram_module._revision_candidate_ids(
            ["candidate-expression", "candidate-vision"],
            ["candidate-expression"],
            [],
            ["candidate-vision"],
        )

        self.assertEqual(
            ["candidate-expression", "candidate-vision"],
            canonical,
        )
        self.assertEqual([], isolated)


if __name__ == "__main__":
    unittest.main()
