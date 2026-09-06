"""Tunable candidate relevance weights, separate from authority and identity.

Defaults preserve the existing scores. No setting changes exact-match flags,
evidence approval, graph status, calculation permissions or diagnosis gates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os


@dataclass(frozen=True)
class CandidateRankingWeights:
    hmc_exact: float = 0.55
    hmc_components: float = 0.45
    hmc_keyword: float = 0.3
    fmeca: float = 0.25
    search_base: float = 0.1
    search_score: float = 0.15

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid candidate ranking weight: {name}")

    def public_dict(self) -> dict:
        return {"version": "candidate-relevance/v1", "weights": asdict(self)}


def candidate_ranking_weights() -> CandidateRankingWeights:
    raw = os.getenv("ICP_CANDIDATE_RANKING_WEIGHTS", "").strip()
    if not raw:
        return CandidateRankingWeights()
    try:
        values = json.loads(raw)
    except ValueError as exc:
        raise ValueError("ICP_CANDIDATE_RANKING_WEIGHTS must be a JSON object") from exc
    if not isinstance(values, dict) or set(values) - set(CandidateRankingWeights.__dataclass_fields__):
        raise ValueError("Unknown candidate ranking settings")
    return CandidateRankingWeights(**values)
