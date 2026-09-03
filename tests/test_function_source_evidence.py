import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from backend.agent.tools import icp_diagnosis as diagnosis_module


class FunctionSourceEvidenceTests(unittest.TestCase):
    def _hit(self, content: str) -> dict:
        return {
            "doc_type_kwd": "function_locator",
            "doc_id": "doc-arch",
            "doc_name": "01_系统架构与功能说明.docx",
            "content_type": "function_text",
            "chunk_id": "chunk-flow",
            "title": "6 主要数据流与支撑关系 / 6.1 静态数据流与支撑关系",
            "source_ref": "materialized-temp.docx#6 主要数据流与支撑关系 / 6.1 静态数据流与支撑关系",
            "content": content,
            "rerank_score": 0.91,
        }

    def _document(self):
        section = ("6 主要数据流与支撑关系", "6.1 静态数据流与支撑关系")
        return SimpleNamespace(
            paragraphs=[
                SimpleNamespace(text="6.1 静态数据流与支撑关系", section_path=section),
                SimpleNamespace(
                    text="业务数据从外部总线进入IFC，经标准化后送入GPP处理。",
                    section_path=section,
                ),
            ]
        )

    def test_exact_artifact_section_promotes_function_text(self):
        content = (
            "6.1 静态数据流与支撑关系\n"
            "业务数据从外部总线进入IFC，经标准化后送入GPP处理。"
        )
        doc = {
            "doc_id": "doc-arch",
            "file_name": "01_系统架构与功能说明.docx",
            "file_path": "sources/doc-arch.docx",
            "storage_key": "sources/doc-arch.docx",
            "status": "kg_done",
        }
        with patch.object(
            diagnosis_module.postgres_client,
            "get_icp_document",
            return_value=doc,
        ), patch.object(
            diagnosis_module.artifact_store,
            "materialize",
            return_value=nullcontext("/tmp/source.docx"),
        ), patch.object(
            diagnosis_module,
            "load_document_ir",
            return_value=self._document(),
        ):
            evidence = diagnosis_module._verified_function_text_evidence(
                [self._hit(content)]
            )

        self.assertEqual(1, len(evidence))
        self.assertEqual("source_artifact", evidence[0]["authority"])
        self.assertEqual("L1_DIRECT", evidence[0]["evidence_level"])
        self.assertEqual("source_text_fact", evidence[0]["claim_type"])
        self.assertIn("送入GPP处理", evidence[0]["content"])
        self.assertTrue(evidence[0]["source_ref"].startswith("01_系统架构与功能说明.docx#"))

    def test_mismatched_projection_is_not_promoted(self):
        doc = {
            "doc_id": "doc-arch",
            "file_name": "01_系统架构与功能说明.docx",
            "file_path": "sources/doc-arch.docx",
            "storage_key": "sources/doc-arch.docx",
            "status": "kg_done",
        }
        with patch.object(
            diagnosis_module.postgres_client,
            "get_icp_document",
            return_value=doc,
        ), patch.object(
            diagnosis_module.artifact_store,
            "materialize",
            return_value=nullcontext("/tmp/source.docx"),
        ), patch.object(
            diagnosis_module,
            "load_document_ir",
            return_value=self._document(),
        ):
            evidence = diagnosis_module._verified_function_text_evidence(
                [self._hit("外部数据直接送入显示器")]
            )

        self.assertEqual([], evidence)


if __name__ == "__main__":
    unittest.main()
