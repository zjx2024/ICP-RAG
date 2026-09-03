import unittest
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys


# Load the pure module without executing backend.core.__init__, whose legacy
# re-exports initialize database-backed services at import time.
_MODULE_PATH = Path(__file__).resolve().parents[1] / "backend/core/retrieval_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("_retrieval_pipeline_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
RetrievalPipeline = _MODULE.RetrievalPipeline


@dataclass
class FakeHit:
    id: str
    score: float
    document: dict


class FakeSearchBackend:
    def __init__(self, count=8):
        self.hits = [
            FakeHit(str(index), 1.0 / (index + 1), {"chunk_id": str(index), "content": f"candidate-{index}"})
            for index in range(count)
        ]
        self.calls = []

    def keyword_search(self, query, filters=None, top_k=10):
        self.calls.append(("keyword", top_k, filters))
        return self.hits[:top_k]

    def vector_search(self, vector, filters=None, top_k=10):
        self.calls.append(("vector", top_k, filters))
        return list(reversed(self.hits[:top_k]))


def make_pipeline(backend, rerank, resolver):
    return RetrievalPipeline(
        search_backend=backend,
        embed=lambda _query, model_id=None: [0.1, 0.2],
        rerank=rerank,
        resolve_rerank_model=resolver,
        recall_multiplier=4,
    )


class RetrievalPipelineTests(unittest.TestCase):
    def test_rerank_receives_all_candidates_before_top_k_slice(self):
        backend = FakeSearchBackend(count=8)
        observed = {}

        def rerank(_query, documents, _model, top_n=None):
            observed["documents"] = list(documents)
            observed["top_n"] = top_n
            target = documents.index("candidate-7")
            return [(target, 0.99)] + [
                (index, 0.8 - index / 100)
                for index in range(len(documents))
                if index != target
            ]

        pipeline = make_pipeline(
            backend,
            rerank,
            lambda _agent_id, _model_id: ({"model_name": "local-rerank"}, "agent_binding"),
        )
        outcome = pipeline.retrieve(query="fault", top_k=2, filters={"kb_id": "hmc"})

        self.assertEqual(8, len(observed["documents"]))
        self.assertEqual(8, observed["top_n"])
        self.assertEqual("7", outcome.results[0]["chunk_id"])
        self.assertEqual(2, len(outcome.results))
        self.assertEqual([8, 8], [call[1] for call in backend.calls])

    def test_unbound_agent_uses_postgres_default_rerank(self):
        backend = FakeSearchBackend(count=3)
        resolver_calls = []

        def resolve(agent_id, rerank_model_id):
            resolver_calls.append((agent_id, rerank_model_id))
            return {"model_name": "pg-default-rerank"}, "postgres_default"

        pipeline = make_pipeline(
            backend,
            lambda _query, documents, _model, top_n=None: [
                (index, 1.0 - index / 10) for index in range(len(documents))
            ],
            resolve,
        )
        outcome = pipeline.retrieve(query="fault", top_k=2, agent_id="agent-without-binding")

        self.assertEqual([("agent-without-binding", "")], resolver_calls)
        self.assertTrue(outcome.reranked)
        self.assertEqual("postgres_default", outcome.rerank_source)
        self.assertEqual("pg-default-rerank", outcome.rerank_model)

    def test_rerank_failure_keeps_fused_results_and_reports_fallback(self):
        backend = FakeSearchBackend(count=5)

        def fail_rerank(*_args, **_kwargs):
            raise RuntimeError("service unavailable")

        pipeline = make_pipeline(
            backend,
            fail_rerank,
            lambda _agent_id, _model_id: ({"model_name": "broken-rerank"}, "postgres_default"),
        )
        outcome = pipeline.retrieve(query="fault", top_k=3)

        self.assertFalse(outcome.reranked)
        self.assertEqual(3, len(outcome.results))
        self.assertIn("service unavailable", outcome.rerank_error)
        self.assertEqual("rerank 失败，已保留融合召回排序", outcome.fallback)
        self.assertTrue(all("rerank_score" not in item for item in outcome.results))

    def test_content_deduplication_preserves_duplicate_evidence(self):
        backend = FakeSearchBackend(count=2)
        backend.hits[0].document.update({"content": "same", "doc_id": "doc-a"})
        backend.hits[1].document.update({"content": "same", "doc_id": "doc-b"})
        pipeline = make_pipeline(
            backend,
            lambda *_args, **_kwargs: None,
            lambda _agent_id, _model_id: (None, ""),
        )

        outcome = pipeline.retrieve(query="fault", top_k=2)

        self.assertEqual(1, outcome.deduplicated_count)
        evidence = outcome.results[0]["duplicate_evidence"]
        self.assertTrue(any(item.get("doc_id") == "doc-b" for item in evidence))

    def test_identical_graph_summaries_keep_distinct_postgres_locators(self):
        backend = FakeSearchBackend(count=2)
        for index, diagram_id in enumerate(("diagram-a", "diagram-b")):
            backend.hits[index].document.update(
                {
                    "content": "相同的图摘要文本",
                    "doc_id": "doc-fta",
                    "doc_type_kwd": "fta",
                    "projection_role_kwd": "diagram_locator",
                    "diagram_id_kwd": diagram_id,
                }
            )
        pipeline = make_pipeline(
            backend,
            lambda *_args, **_kwargs: None,
            lambda _agent_id, _model_id: (None, ""),
        )

        outcome = pipeline.retrieve(query="fault", top_k=2)

        self.assertEqual(2, outcome.deduplicated_count)
        self.assertEqual(
            {"diagram-a", "diagram-b"},
            {item["diagram_id_kwd"] for item in outcome.results},
        )

    def test_invalid_fusion_weights_are_rejected(self):
        pipeline = make_pipeline(
            FakeSearchBackend(count=2),
            lambda *_args, **_kwargs: None,
            lambda _agent_id, _model_id: (None, ""),
        )

        with self.assertRaisesRegex(ValueError, "融合权重"):
            pipeline.retrieve(
                query="fault",
                top_k=2,
                keyword_weight=0,
                vector_weight=0,
            )

if __name__ == "__main__":
    unittest.main()
