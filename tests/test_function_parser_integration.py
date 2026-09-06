import tempfile
import unittest
import importlib
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.core.parsers.base import ParseContext
from backend.core.parsers.document_ir import DocumentIR, ParagraphIR
from backend.core.parsers.function_extraction import FunctionExtractionResult
from backend.core.parsers.records import RecordDraft, SourceRef
from backend.core.parsers.registry import ParserRegistry  # noqa: F401

architecture = importlib.import_module("backend.core.parsers.function_architecture")


def _draft(record_type, values, business_id, row):
    source = SourceRef(
        file_name="architecture.docx",
        element_type="row",
        table_index=1,
        row_index=row,
    )
    return RecordDraft(
        record_type=record_type,
        values=values,
        raw_data={"raw_values": dict(values)},
        source_refs=[source],
        record_key=business_id,
    )


class FunctionArchitectureParserIntegrationTests(unittest.TestCase):
    def test_vision_failure_persists_empty_candidate_for_terminal_quarantine(self):
        parsed = {
            "blocks": {
                "architecture": {
                    "title": "总体架构",
                    "text": "",
                    "tables": [],
                    "images": [
                        {
                            "index": 1,
                            "title": "架构图",
                            "storage_key": "images/doc-vision-failed/architecture.png",
                            "image_url": "/images/doc-vision-failed/architecture.png",
                            "content_type": "image/png",
                            "source_ref": "architecture.docx#image=1",
                        }
                    ],
                }
            }
        }
        postgres = MagicMock()
        artifact = MagicMock()
        artifact.stat.return_value = SimpleNamespace(sha256="c" * 64)
        diagrams = MagicMock()
        diagrams.upsert_diagram.return_value = {"diagram_id": "diagram-failed"}
        diagrams.upsert_candidate.return_value = {"candidate_id": "candidate-failed"}
        diagrams.create_or_reuse_revision.return_value = {
            "review_status": "pending"
        }
        diagrams.archive_missing_diagrams.return_value = {}
        docx_parser = MagicMock()
        docx_parser.process.return_value = parsed

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "architecture.docx"
            source_path.touch()
            image_path = Path(directory) / "architecture.png"
            image_path.write_bytes(b"mock-image")
            artifact.materialize.return_value = nullcontext(str(image_path))
            parser = architecture.FunctionArchitectureParser()
            with (
                patch.object(architecture, "postgres_client", postgres),
                patch.object(architecture, "artifact_store", artifact),
                patch.object(architecture, "diagram_repository", diagrams),
                patch.object(architecture, "DocumentParser", return_value=docx_parser),
                patch.object(
                    architecture,
                    "analyze_icp_diagram",
                    side_effect=RuntimeError("vision unavailable"),
                ),
                patch.object(parser, "_safe_index", return_value=0),
            ):
                result = parser._process_images(
                    str(source_path),
                    ParseContext(
                        doc_id="doc-vision-failed",
                        doc_type="function_locator",
                        file_name="architecture.docx",
                    ),
                    [],
                )

        failed_candidate = diagrams.upsert_candidate.call_args.kwargs
        self.assertEqual("failed", failed_candidate["extraction_status"])
        self.assertEqual([], failed_candidate["raw_graph"]["vertices"])
        self.assertEqual([], failed_candidate["raw_graph"]["edges"])
        self.assertIn(
            "vision unavailable",
            failed_candidate["source_locator"]["extraction_error"],
        )
        self.assertEqual(
            ["candidate-failed"],
            diagrams.create_or_reuse_revision.call_args.kwargs["candidate_ids"],
        )
        self.assertEqual(1, result["vision_candidates"])
        self.assertEqual({"pending": 1}, result["review_status_counts"])
        self.assertIn("vision unavailable", result["vision_failed"][0]["reason"])

    def test_authority_snapshot_stable_projections_and_unconfirmed_vision_boundary(self):
        extraction = FunctionExtractionResult(
            components=[
                _draft(
                    "function_component",
                    {
                        "component_id": "MOD-DET-1",
                        "component_type": "module",
                        "parent_component_id": "SUB-DET",
                        "name": "确定性模块",
                        "abbreviation": "DM1",
                        "description": "来自明确模块表",
                    },
                    "MOD-DET-1",
                    2,
                )
            ],
            functions=[
                _draft(
                    "function_record",
                    {
                        "function_id": "FUNC-DET-1",
                        "parent_function_id": "",
                        "function_name": "确定性功能",
                        "description": "来自明确功能表",
                        "system_name": "演示系统",
                        "subsystem_name": "演示子系统",
                        "module_id": "MOD-DET-1",
                        "module_name": "确定性模块",
                        "inputs": ["输入A"],
                        "outputs": ["输出B"],
                        "failure_symptoms": "输出不更新",
                    },
                    "FUNC-DET-1",
                    3,
                )
            ],
            interfaces=[
                _draft(
                    "function_interface",
                    {
                        "interface_id": "IF-DET-1",
                        "source_component_id": "MOD-DET-1",
                        "target_component_id": "MOD-DET-2",
                        "relation_type": "data_flow",
                        "signal_name": "状态字",
                        "direction": "source_to_target",
                        "protocol": "LOCAL-BUS",
                        "description": "来自明确接口表",
                    },
                    "IF-DET-1",
                    4,
                )
            ],
            image_evidence=[
                {
                    "sha256": "a" * 64,
                    "source_ref": "architecture.docx#image=1",
                    "authority": "unconfirmed_extraction",
                    "eligible_for_authoritative_records": False,
                }
            ],
        )
        paragraph_source = SourceRef(
            file_name="architecture.docx",
            element_type="paragraph",
            sequence=1,
            section_path=("1 系统说明",),
        )
        document_ir = DocumentIR(
            file_name="architecture.docx",
            file_type="docx",
            elements=[
                ParagraphIR(
                    text="系统说明正文",
                    raw_text="系统说明正文",
                    style_name="正文",
                    heading_level=0,
                    section_path=("1 系统说明",),
                    source=paragraph_source,
                )
            ],
        )

        repository = MagicMock()
        # Different surrogate PG IDs on the second run prove projection IDs do
        # not depend on insertion order or database sequences.
        repository.upsert_component.side_effect = [101, 1101]
        repository.upsert_function.side_effect = [201, 1201]
        repository.upsert_interface.side_effect = [301, 1301]
        search = MagicMock()
        embedding = MagicMock()
        embedding.embed.return_value = [0.0] * 1024
        postgres = MagicMock()
        artifact = MagicMock()
        artifact.stat.return_value = SimpleNamespace(sha256="a" * 64)
        diagrams = MagicMock()
        diagrams.upsert_diagram.return_value = {"diagram_id": "diagram-1"}
        diagrams.upsert_candidate.return_value = {"candidate_id": "candidate-vision-1"}
        diagrams.create_or_reuse_revision.return_value = {
            "revision_id": "revision-1",
            "review_status": "pending",
        }
        vision_result = {
            "description": "视觉识别出的候选功能，不是权威事实",
            "confidence": 0.91,
            "vertices": [
                {
                    "id": "FUNC-VISION-ONLY",
                    "label": "Function",
                    "properties": {"name": "视觉候选功能"},
                }
            ],
            "edges": [],
        }
        docx_parser = MagicMock()
        docx_parser.process.return_value = {
            "blocks": {
                "block-1": {
                    "title": "总体架构",
                    "text": "架构图上下文",
                    "tables": [],
                    "images": [
                        {
                            "index": 1,
                            "title": "总体架构图",
                            "storage_key": "documents/doc-1/images/architecture.png",
                            "image_url": "/images/architecture.png",
                            "content_type": "image/png",
                            "source_ref": "architecture.docx#body=3;container=table;rel=rId9",
                        }
                    ],
                }
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "architecture.docx"
            source_path.touch()
            image_path = Path(directory) / "architecture.png"
            image_path.write_bytes(b"mock-image")
            artifact.materialize.side_effect = lambda *_args, **_kwargs: nullcontext(
                str(image_path)
            )
            context = ParseContext(
                doc_id="doc-1",
                doc_type="function_locator",
                file_name="architecture.docx",
                common_tags={
                    "document_metadata": {
                        "applicable_system": "演示系统",
                        "applicable_configuration": "CONFIG-A",
                    }
                },
            )

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(architecture, "extract_function_drafts", return_value=extraction)
                )
                stack.enter_context(
                    patch.object(architecture, "load_document_ir", return_value=document_ir)
                )
                stack.enter_context(patch.object(architecture, "icp_repository", repository))
                stack.enter_context(patch.object(architecture, "search_store", search))
                stack.enter_context(patch.object(architecture, "embedding_service", embedding))
                stack.enter_context(patch.object(architecture, "postgres_client", postgres))
                stack.enter_context(patch.object(architecture, "artifact_store", artifact))
                stack.enter_context(patch.object(architecture, "diagram_repository", diagrams))
                stack.enter_context(
                    patch.object(architecture, "analyze_icp_diagram", return_value=vision_result)
                )
                stack.enter_context(
                    patch.object(architecture, "DocumentParser", return_value=docx_parser)
                )

                parser = architecture.FunctionArchitectureParser()
                first = parser.parse(str(source_path), context)
                first_projection_count = search.upsert.call_count
                first_projection_ids = [
                    call.args[0]
                    for call in search.upsert.call_args_list[:first_projection_count]
                ]
                second = parser.parse(str(source_path), context)
                second_projection_ids = [
                    call.args[0]
                    for call in search.upsert.call_args_list[first_projection_count:]
                ]

        self.assertEqual(repository.clear_function_snapshot.call_count, 2)
        self.assertEqual(
            [call[0] for call in repository.method_calls],
            [
                "clear_function_snapshot",
                "upsert_component",
                "upsert_function",
                "upsert_interface",
                "clear_function_snapshot",
                "upsert_component",
                "upsert_function",
                "upsert_interface",
            ],
        )
        self.assertEqual(first_projection_count, 5)
        self.assertEqual(first_projection_ids, second_projection_ids)
        self.assertEqual(len(first_projection_ids), len(set(first_projection_ids)))
        self.assertTrue(
            all(projection_id.startswith("doc-1_function_locator_") for projection_id in first_projection_ids)
        )

        # The vision result contains a Function node, but only the one explicit
        # deterministic function reaches the authority repository on each run.
        self.assertEqual(repository.upsert_function.call_count, 2)
        self.assertTrue(
            all(
                call.args[1]["function_id"] == "FUNC-DET-1"
                for call in repository.upsert_function.call_args_list
            )
        )
        self.assertTrue(
            all(
                call.args[1]["component_id"] == "MOD-DET-1"
                for call in repository.upsert_component.call_args_list
            )
        )
        self.assertTrue(
            all(
                call.args[1]["interface_id"] == "IF-DET-1"
                for call in repository.upsert_interface.call_args_list
            )
        )
        postgres.insert_document_fault_tree.assert_not_called()
        self.assertEqual(diagrams.upsert_candidate.call_count, 2)
        candidate_graph = diagrams.upsert_candidate.call_args_list[0].kwargs["raw_graph"]
        self.assertEqual(candidate_graph["vertices"][0]["id"], "FUNC-VISION-ONLY")
        self.assertEqual(candidate_graph["metadata"]["mime_type"], "image/png")
        self.assertIn("container=table", candidate_graph["metadata"]["source_ref"])
        self.assertEqual(postgres.insert_image.call_args_list[0].args[8], "image/png")
        self.assertIn("container=table", postgres.insert_image.call_args_list[0].args[9])
        self.assertEqual(
            diagrams.create_or_reuse_revision.call_args_list[0].kwargs["candidate_ids"],
            ["candidate-vision-1"],
        )
        self.assertEqual(first.stats["review_status_counts"], {"pending": 1})
        self.assertEqual(first.stats["vision_candidates"], 1)
        self.assertEqual(second.stats["functions_written"], 1)

    def test_all_invalid_authority_does_not_destroy_previous_snapshot(self):
        invalid = _draft(
            "function_record",
            {"function_id": "", "function_name": ""},
            "",
            2,
        )
        invalid.add_issue(
            "missing_function_id",
            "功能缺少可用标识",
            field="function_id",
            severity="error",
        )
        extraction = FunctionExtractionResult(functions=[invalid])
        repository = MagicMock()
        with patch.object(
            architecture, "extract_function_drafts", return_value=extraction
        ), patch.object(architecture, "icp_repository", repository):
            with self.assertRaisesRegex(ValueError, "保留上一版快照"):
                architecture.FunctionArchitectureParser().parse(
                    "invalid.docx",
                    ParseContext(
                        doc_id="doc-invalid",
                        doc_type="function_locator",
                        file_name="invalid.docx",
                    ),
                )

        repository.clear_function_snapshot.assert_not_called()
        repository.upsert_function.assert_not_called()

    def test_function_image_authority_failures_are_hard_failures(self):
        extraction = FunctionExtractionResult(
            image_evidence=[
                {
                    "sha256": "b" * 64,
                    "source_ref": "architecture.docx#image=1",
                    "authority": "unconfirmed_extraction",
                    "eligible_for_authoritative_records": False,
                }
            ]
        )
        document_ir = DocumentIR(
            file_name="architecture.docx",
            file_type="docx",
            elements=[],
        )
        parsed = {
            "blocks": {
                "architecture": {
                    "title": "总体架构",
                    "text": "",
                    "tables": [],
                    "images": [
                        {
                            "index": 1,
                            "title": "架构图",
                            "storage_key": "images/doc-failure/architecture.png",
                            "image_url": "/images/doc-failure/architecture.png",
                            "content_type": "image/png",
                            "source_ref": "architecture.docx#body=2;container=table;rel=rId1",
                        }
                    ],
                }
            }
        }
        analysis = {
            "description": "候选架构",
            "confidence": 0.8,
            "vertices": [{"id": "M1", "label": "Module", "properties": {"name": "模块"}}],
            "edges": [],
        }

        for failure_stage in ("image", "candidate", "lifecycle"):
            with self.subTest(failure_stage=failure_stage), tempfile.TemporaryDirectory() as directory:
                source_path = Path(directory) / "architecture.docx"
                source_path.touch()
                image_path = Path(directory) / "architecture.png"
                image_path.write_bytes(b"\x89PNG\r\n\x1a\nmock")
                repository = MagicMock()
                postgres = MagicMock()
                artifact = MagicMock()
                artifact.stat.return_value = SimpleNamespace(
                    sha256="b" * 64,
                    content_type="image/png",
                )
                artifact.materialize.return_value = nullcontext(str(image_path))
                diagrams = MagicMock()
                diagrams.upsert_diagram.return_value = {"diagram_id": "diagram-1"}
                diagrams.upsert_candidate.return_value = {"candidate_id": "candidate-1"}
                diagrams.create_or_reuse_revision.return_value = {"review_status": "pending"}
                diagrams.archive_missing_diagrams.return_value = {}
                if failure_stage == "image":
                    postgres.insert_image.side_effect = RuntimeError("image PG failed")
                elif failure_stage == "candidate":
                    diagrams.upsert_candidate.side_effect = RuntimeError("candidate PG failed")
                else:
                    diagrams.archive_missing_diagrams.side_effect = RuntimeError(
                        "lifecycle PG failed"
                    )

                with ExitStack() as stack:
                    stack.enter_context(
                        patch.object(architecture, "extract_function_drafts", return_value=extraction)
                    )
                    stack.enter_context(
                        patch.object(architecture, "load_document_ir", return_value=document_ir)
                    )
                    stack.enter_context(patch.object(architecture, "icp_repository", repository))
                    stack.enter_context(patch.object(architecture, "postgres_client", postgres))
                    stack.enter_context(patch.object(architecture, "artifact_store", artifact))
                    stack.enter_context(patch.object(architecture, "diagram_repository", diagrams))
                    stack.enter_context(
                        patch.object(architecture, "analyze_icp_diagram", return_value=analysis)
                    )
                    stack.enter_context(
                        patch.object(architecture, "DocumentParser")
                    ).return_value.process.return_value = parsed
                    stack.enter_context(patch.object(architecture, "search_store", MagicMock()))
                    stack.enter_context(patch.object(architecture, "embedding_service", MagicMock()))

                    with self.assertRaisesRegex(RuntimeError, "PostgreSQL"):
                        architecture.FunctionArchitectureParser().parse(
                            str(source_path),
                            ParseContext(
                                doc_id="doc-failure",
                                doc_type="function_locator",
                                file_name="architecture.docx",
                            ),
                        )


if __name__ == "__main__":
    unittest.main()
