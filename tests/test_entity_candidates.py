import unittest

from backend.icp.entity_candidates import (
    CandidateDiscoveryPolicy,
    EntityCandidateDiscoveryService,
    compatible_scope_score,
    lexical_similarity,
)
from backend.icp.entity_resolution import EntityResolutionRepository


class _FakeRepository:
    calculate_candidate_score = EntityResolutionRepository.calculate_candidate_score

    def __init__(self):
        self.events = []
        self.sources = [
            {
                "mention_id": "m-1",
                "doc_id": "doc-a",
                "entity_type": "Symptom",
                "field_type": "fault_history_fault_symptom",
                "raw_text": "系统反复复位",
                "normalized_text": "系统反复复位",
                "scope": {"system_code": "SYS", "configuration": "A"},
                "resolved_entity_id": "source-entity",
            }
        ]
        self.targets = [
            {
                "entity_id": "candidate-good",
                "entity_type": "Symptom",
                "canonical_name": "系统重复复位",
                "identity_scope": {"system_code": "SYS", "configuration": "A"},
                "origin_doc_id": "doc-b",
            },
            {
                "entity_id": "candidate-unrelated",
                "entity_type": "Symptom",
                "canonical_name": "系统过热",
                "identity_scope": {"system_code": "SYS", "configuration": "A"},
                "origin_doc_id": "doc-c",
            },
            {
                "entity_id": "candidate-conflict",
                "entity_type": "Symptom",
                "canonical_name": "系统反复复位",
                "identity_scope": {"system_code": "SYS", "configuration": "B"},
                "origin_doc_id": "doc-d",
            },
        ]

    def list_candidate_source_mentions(self, **kwargs):
        self.events.append(("sources", kwargs))
        return list(self.sources)

    def list_candidate_target_entities(self, **kwargs):
        self.events.append(("targets", kwargs))
        return list(self.targets)

    def clear_similarity_candidates(self, mention_ids, **kwargs):
        self.events.append(("clear", list(mention_ids), kwargs))
        return 2

    def add_similarity_candidate(self, **kwargs):
        self.events.append(("add", kwargs))
        return {"candidate_id": "saved"}


class _FakeEmbedder:
    def embed_many(self, texts, model_id=None):
        values = {
            "系统反复复位": [1.0, 0.0],
            "系统重复复位": [0.99, 0.1],
            "系统过热": [0.0, 1.0],
        }
        return [values[text] for text in texts]


class _FailingEmbedder:
    def embed_many(self, texts, model_id=None):
        raise RuntimeError("model unavailable")


class EntityCandidatePureFunctionTests(unittest.TestCase):
    def test_lexical_similarity_is_only_a_rank_score(self):
        self.assertEqual(lexical_similarity("复位", "复位"), 1.0)
        self.assertGreater(lexical_similarity("系统反复复位", "系统重复复位"), 0.5)
        self.assertEqual(lexical_similarity("", "复位"), 0.0)

    def test_scope_conflict_is_rejected(self):
        self.assertIsNone(
            compatible_scope_score(
                {"system_code": "SYS", "configuration": "A"},
                {"system_code": "SYS", "configuration": "B"},
            )
        )
        self.assertEqual(
            compatible_scope_score(
                {"system_code": "SYS", "configuration": "A"},
                {"system_code": "SYS", "configuration": "A"},
            ),
            1.0,
        )


class EntityCandidateDiscoveryTests(unittest.TestCase):
    def test_embedding_creates_candidates_but_never_formal_resolution(self):
        repository = _FakeRepository()
        service = EntityCandidateDiscoveryService(repository, _FakeEmbedder())

        result = service.discover_for_document(
            "doc-a",
            model_version="bge-m3@test",
            policy=CandidateDiscoveryPolicy(
                min_embedding_score=0.6,
                min_lexical_score=0.3,
                min_combined_score=0.6,
            ),
        )

        self.assertEqual(result["candidate_count"], 1)
        self.assertFalse(result["formal_resolution_changed"])
        clear_index = next(i for i, event in enumerate(repository.events) if event[0] == "clear")
        add_events = [
            (i, event) for i, event in enumerate(repository.events) if event[0] == "add"
        ]
        self.assertEqual(len(add_events), 1)
        self.assertLess(clear_index, add_events[0][0])
        saved = add_events[0][1][1]
        self.assertEqual(saved["candidate_entity_id"], "candidate-good")
        self.assertEqual(saved["resolution_method"], "hybrid_similarity")
        self.assertTrue(saved["metadata"]["candidate_only"])
        self.assertFalse(saved["metadata"]["formal_identity"])

    def test_model_failure_preserves_previous_candidate_snapshot(self):
        repository = _FakeRepository()
        service = EntityCandidateDiscoveryService(repository, _FailingEmbedder())

        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            service.discover_for_document("doc-a", model_version="bge-m3@test")

        self.assertFalse(any(event[0] == "clear" for event in repository.events))
        self.assertFalse(any(event[0] == "add" for event in repository.events))


if __name__ == "__main__":
    unittest.main()
