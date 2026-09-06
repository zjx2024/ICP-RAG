"""Candidate compatibility rules over planner slots and source fields.

Dependencies flow toward source views and pure rules, never answer.py.
"""
from __future__ import annotations

from backend.agent.nodes.answer_payload import _evidence_answer_fields
from typing import Any
from typing import List
from typing import Mapping
import re


def _fault_candidate_fit(
    state: Mapping[str, Any],
    item: Mapping[str, Any],
) -> str:
    """Compare planner-normalized constraints with one structured FMECA row.

    The planner remains responsible for understanding free-form user language.
    This deterministic layer only compares its closed ``query_frame`` slots
    with the source's own structured phase/impact fields.  Missing source data
    stays ``unknown``; only explicit conflicts are rejected.
    """

    if str(item.get("doc_type") or "") != "fmeca":
        return "unknown"
    frame = (
        state.get("query_frame")
        if isinstance(state.get("query_frame"), Mapping)
        else {}
    )
    fields = _evidence_answer_fields(item)
    stage_values = {"上电", "初始化", "运行中", "高负载", "持续运行后"}
    temporal_values = {"持续发生", "间歇发生", "重复发生", "首次发生"}

    query_stage = str(frame.get("operating_stage") or "").strip()
    query_stage = query_stage if query_stage in stage_values else ""
    query_temporal = str(frame.get("temporal_pattern") or "").strip()
    query_temporal = query_temporal if query_temporal in temporal_values else ""
    source_stage = str(
        fields.get("operating_stage") or fields.get("mission_phase_enum") or ""
    ).strip()
    source_stage = source_stage if source_stage in stage_values else ""
    source_temporal = str(
        fields.get("temporal_pattern") or fields.get("occurrence_pattern") or ""
    ).strip()
    source_temporal = source_temporal if source_temporal in temporal_values else ""

    stage_fit = "unknown"
    if query_stage and source_stage:
        conditional_pairs = {
            ("运行中", "高负载"),
            ("运行中", "持续运行后"),
            ("高负载", "运行中"),
            ("持续运行后", "运行中"),
        }
        startup = {"上电", "初始化"}
        runtime = {"运行中", "高负载", "持续运行后"}
        if query_stage == source_stage:
            stage_fit = "direct"
        elif (query_stage, source_stage) in conditional_pairs:
            stage_fit = "conditional"
        elif (
            (query_stage in startup and source_stage in runtime)
            or (query_stage in runtime and source_stage in startup)
        ):
            stage_fit = "contradictory"

    temporal_fit = "unknown"
    if query_temporal and source_temporal:
        if query_temporal == source_temporal:
            temporal_fit = "direct"
        elif {query_temporal, source_temporal} in (
            {"持续发生", "间歇发生"},
            {"首次发生", "重复发生"},
        ):
            temporal_fit = "contradictory"

    fits = {stage_fit, temporal_fit}
    if "contradictory" in fits:
        return "contradictory"
    if "conditional" in fits:
        return "conditional"
    if "direct" in fits:
        return "direct"
    return "unknown"


def _query_requests_parallel_failure_modes(state: Mapping[str, Any]) -> bool:
    """Whether the user explicitly asks to distinguish multiple branches.

    This is a completeness-contract trigger, not a fault classifier.  The
    actual branch identities still come exclusively from approved structured
    FMECA/MFL fields below.
    """

    frame = (
        state.get("query_frame")
        if isinstance(state.get("query_frame"), Mapping)
        else {}
    )
    # Comparison is an open discourse intent classified once by the target
    # model.  The deterministic layer requires two distinct, current-turn
    # grounded failure-mode objects; a generic ``compare`` label alone cannot
    # authorize a multi-branch completion template.
    query = str(frame.get("original_query") or state.get("query") or "")
    subjects: List[str] = []
    for raw in frame.get("comparison_subjects") or []:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("kind") or "").strip() != "failure_mode":
            continue
        value = str(raw.get("value") or "").strip()
        basis = str(raw.get("basis") or "").strip()
        if not value or not basis or basis not in query or value not in basis:
            continue
        if value not in subjects:
            subjects.append(value)
    return len(subjects) >= 2


def _fault_public_scalar(value: Any) -> str:
    """Return one source scalar without exposing an internal field payload."""

    if isinstance(value, (Mapping, list, tuple, set)):
        return ""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _payload_direction(value: Any) -> str:
    """Return only a validated structured payload-direction enum.

    Direction is an open interpretation of natural language.  The answer
    layer must not recreate it from a list of send/receive phrases and then
    delete otherwise approved evidence.  It consumes the planner's validated
    ``payload_contract`` or an explicit source-side enum instead.
    """

    raw = value.get("direction") if isinstance(value, Mapping) else value
    direction = str(raw or "").strip()
    if direction in {"external_to_system", "system_to_external", "internal"}:
        return direction
    return ""


_FAULT_FIT_RANK = {
    "direct": 0,
    "unknown": 1,
    "conditional": 2,
    "contradictory": 3,
}
