"""Semantic candidate discovery for ICP entity mentions.

This module deliberately stops before identity resolution.  BGE-M3 similarity
helps reviewers and retrieval find possible aliases, but every result remains
``candidate_semantic`` and cannot create a relation assertion or ``same_as``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from backend.icp.entity_resolution import (
    EntityResolutionRepository,
    clean_text,
    cosine_similarity,
    entity_resolution_repository,
    normalize_scope,
)


SEMANTIC_ENTITY_TYPES = frozenset(
    {
        "Module",
        "Component",
        "Function",
        "Symptom",
        "FaultMode",
        "FaultCause",
        "FaultEffect",
        "DetectionMethod",
        "MaintenanceAction",
        "SourceFragment",
    }
)


@dataclass(frozen=True)
class CandidateDiscoveryPolicy:
    top_k: int = 10
    source_limit: int = 2000
    target_limit_per_type: int = 5000
    batch_size: int = 64
    min_embedding_score: float = 0.62
    min_lexical_score: float = 0.35
    min_combined_score: float = 0.66

    def validate(self) -> "CandidateDiscoveryPolicy":
        if not 1 <= self.top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        if not 1 <= self.source_limit <= 10000:
            raise ValueError("source_limit must be between 1 and 10000")
        if not 1 <= self.target_limit_per_type <= 20000:
            raise ValueError("target_limit_per_type must be between 1 and 20000")
        if not 1 <= self.batch_size <= 256:
            raise ValueError("batch_size must be between 1 and 256")
        for name in (
            "min_embedding_score",
            "min_lexical_score",
            "min_combined_score",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        return self


def lexical_similarity(left: str, right: str) -> float:
    """Return a deterministic text score used only for candidate ranking."""

    normalized_left = clean_text(left).casefold()
    normalized_right = clean_text(right).casefold()
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0
    return round(SequenceMatcher(None, normalized_left, normalized_right).ratio(), 6)


def _mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def compatible_scope_score(left: Any, right: Any) -> Optional[float]:
    """Score compatible applicability; return ``None`` for a hard conflict."""

    left_scope = normalize_scope(_mapping(left))
    right_scope = normalize_scope(_mapping(right))
    shared = sorted(set(left_scope) & set(right_scope))
    for key in shared:
        if left_scope[key] != right_scope[key]:
            return None
    if left_scope and right_scope and left_scope == right_scope:
        return 1.0
    if shared:
        return 0.85
    return 0.5


def _source_text(row: Mapping[str, Any]) -> str:
    return clean_text(
        row.get("normalized_text")
        or row.get("raw_text")
        or row.get("resolved_entity_name")
    )


def _target_text(row: Mapping[str, Any]) -> str:
    return clean_text(row.get("canonical_name") or row.get("business_id"))


class EntityCandidateDiscoveryService:
    def __init__(
        self,
        repository: EntityResolutionRepository = entity_resolution_repository,
        embedder: Any = None,
    ) -> None:
        self.repository = repository
        self._embedder = embedder

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from backend.core.embedding import embedding_service

            self._embedder = embedding_service
        return self._embedder

    @staticmethod
    def _resolve_model_version(embedding_model_id: Optional[str]) -> str:
        try:
            if embedding_model_id:
                from backend.storage.postgres_client import postgres_client

                model = postgres_client.get_model(embedding_model_id) or {}
            else:
                from backend.core.model_config import model_config_service

                model = model_config_service.get_embedding_model() or {}
            model_id = clean_text(model.get("model_id") or embedding_model_id)
            model_name = clean_text(model.get("model_name") or model.get("name"))
            return ":".join(value for value in (model_id, model_name) if value)
        except Exception:
            return clean_text(embedding_model_id)

    def _embed_many(
        self,
        texts: Sequence[str],
        *,
        model_id: Optional[str],
        batch_size: int,
    ) -> Dict[str, Sequence[float]]:
        unique = list(dict.fromkeys(clean_text(text) for text in texts if clean_text(text)))
        vectors: Dict[str, Sequence[float]] = {}
        for start in range(0, len(unique), batch_size):
            batch = unique[start : start + batch_size]
            try:
                embedded = self.embedder.embed_many(batch, model_id=model_id)
            except TypeError:
                embedded = self.embedder.embed_many(batch)
            if len(embedded) != len(batch):
                raise ValueError("embedding result count does not match candidate texts")
            vectors.update(zip(batch, embedded))
        return vectors

    def discover_for_document(
        self,
        doc_id: str,
        *,
        embedding_model_id: Optional[str] = None,
        model_version: str = "",
        policy: Optional[CandidateDiscoveryPolicy] = None,
        entity_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Rebuild semantic candidates for one document without resolving any.

        All embeddings are computed successfully before the previous candidate
        snapshot is removed, so a model outage leaves existing suggestions
        intact.  Formal mention resolution and relation assertions are never
        touched by this workflow.
        """

        document_id = clean_text(doc_id)
        if not document_id:
            raise ValueError("doc_id is required")
        active_policy = (policy or CandidateDiscoveryPolicy()).validate()
        selected_types = sorted(
            set(entity_types or SEMANTIC_ENTITY_TYPES) & SEMANTIC_ENTITY_TYPES
        )
        if not selected_types:
            return {
                "doc_id": document_id,
                "mention_count": 0,
                "candidate_count": 0,
                "model_version": clean_text(model_version),
                "status": "no_supported_entity_types",
            }

        sources = self.repository.list_candidate_source_mentions(
            doc_id=document_id,
            entity_types=selected_types,
            limit=active_policy.source_limit,
        )
        sources = [row for row in sources if len(_source_text(row)) >= 2]
        if not sources:
            return {
                "doc_id": document_id,
                "mention_count": 0,
                "candidate_count": 0,
                "model_version": clean_text(model_version),
                "status": "no_document_scoped_mentions",
            }

        target_by_type: Dict[str, List[Mapping[str, Any]]] = {}
        for entity_type in sorted({clean_text(row.get("entity_type")) for row in sources}):
            if not entity_type:
                continue
            target_by_type[entity_type] = self.repository.list_candidate_target_entities(
                entity_type=entity_type,
                exclude_doc_id=document_id,
                limit=active_policy.target_limit_per_type,
            )

        texts = [_source_text(row) for row in sources]
        texts.extend(
            _target_text(row)
            for rows in target_by_type.values()
            for row in rows
            if len(_target_text(row)) >= 2
        )
        vectors = self._embed_many(
            texts,
            model_id=embedding_model_id,
            batch_size=active_policy.batch_size,
        )
        resolved_model_version = clean_text(model_version) or self._resolve_model_version(
            embedding_model_id
        )

        ranked: Dict[str, List[Tuple[float, str, Mapping[str, Any], float, float, float]]] = {}
        for source in sources:
            mention_id = clean_text(source.get("mention_id"))
            source_text = _source_text(source)
            source_vector = vectors.get(source_text)
            if not mention_id or source_vector is None:
                continue
            candidates: List[
                Tuple[float, str, Mapping[str, Any], float, float, float]
            ] = []
            for target in target_by_type.get(clean_text(source.get("entity_type")), []):
                target_id = clean_text(target.get("entity_id"))
                target_text = _target_text(target)
                target_vector = vectors.get(target_text)
                if (
                    not target_id
                    or target_id == clean_text(source.get("resolved_entity_id"))
                    or target_vector is None
                ):
                    continue
                context_score = compatible_scope_score(
                    source.get("scope") or source.get("entity_identity_scope"),
                    target.get("identity_scope"),
                )
                if context_score is None:
                    continue
                lexical_score = lexical_similarity(source_text, target_text)
                embedding_score = max(
                    0.0, cosine_similarity(source_vector, target_vector)
                )
                if (
                    embedding_score < active_policy.min_embedding_score
                    and lexical_score < active_policy.min_lexical_score
                ):
                    continue
                combined = self.repository.calculate_candidate_score(
                    lexical_score=lexical_score,
                    embedding_score=embedding_score,
                    context_score=context_score,
                )
                if combined < active_policy.min_combined_score:
                    continue
                candidates.append(
                    (
                        combined,
                        target_id,
                        target,
                        lexical_score,
                        embedding_score,
                        context_score,
                    )
                )
            candidates.sort(key=lambda item: (-item[0], item[1]))
            ranked[mention_id] = candidates[: active_policy.top_k]

        mention_ids = [
            clean_text(row.get("mention_id"))
            for row in sources
            if clean_text(row.get("mention_id"))
        ]
        deleted_count = self.repository.clear_similarity_candidates(
            mention_ids,
            resolution_method="hybrid_similarity",
            model_version=resolved_model_version,
        )
        created_count = 0
        by_type: Dict[str, int] = {}
        source_by_id = {
            clean_text(row.get("mention_id")): row for row in sources
        }
        for mention_id in sorted(ranked):
            source = source_by_id[mention_id]
            for (
                combined,
                target_id,
                target,
                lexical_score,
                embedding_score,
                context_score,
            ) in ranked[mention_id]:
                self.repository.add_similarity_candidate(
                    mention_id=mention_id,
                    candidate_entity_id=target_id,
                    resolution_method="hybrid_similarity",
                    lexical_score=lexical_score,
                    embedding_score=embedding_score,
                    context_score=context_score,
                    combined_score=combined,
                    model_version=resolved_model_version,
                    metadata={
                        "candidate_only": True,
                        "formal_identity": False,
                        "source_doc_id": document_id,
                        "source_field_type": clean_text(source.get("field_type")),
                        "candidate_origin_doc_id": clean_text(
                            target.get("origin_doc_id")
                        ),
                    },
                )
                created_count += 1
                entity_type = clean_text(source.get("entity_type"))
                by_type[entity_type] = by_type.get(entity_type, 0) + 1

        return {
            "doc_id": document_id,
            "mention_count": len(sources),
            "candidate_count": created_count,
            "deleted_previous_count": deleted_count,
            "candidate_count_by_type": dict(sorted(by_type.items())),
            "model_version": resolved_model_version,
            "status": "completed",
            "formal_resolution_changed": False,
        }


entity_candidate_discovery_service = EntityCandidateDiscoveryService()


__all__ = [
    "CandidateDiscoveryPolicy",
    "EntityCandidateDiscoveryService",
    "SEMANTIC_ENTITY_TYPES",
    "compatible_scope_score",
    "entity_candidate_discovery_service",
    "lexical_similarity",
]
