import backend.agent.nodes.answer_structure_views as _source_answer_structure_views
import json
import unittest

from backend.agent.contracts import make_evidence
from backend.agent.nodes import answer as answer_module
from backend.agent.rbd_contract import (
    RBD_SINGLE_FAILURE_RELATION_VERSION,
    bind_rbd_models,
    selected_rbd_anchor_selectors,
)


def _graph(*, criterion="输出端可达", relations=()):
    return {
        "diagram_type": "rbd",
        "metadata": {
            "success_criterion": criterion,
            "validation_status": "auto_validated",
            "single_failure_relations": list(relations),
        },
        "vertices": [
            {
                "id": "local-a",
                "label": "ReliabilityBlock",
                "properties": {
                    "branch_id": "root.power:branch:1",
                    "connection_type": "parallel",
                    "module_id": "MOD-PWR-A",
                    "name": "A路供电",
                },
            },
            {
                "id": "local-b",
                "label": "ReliabilityBlock",
                "properties": {
                    "branch_id": "root.power:branch:2",
                    "connection_type": "parallel",
                    "module_id": "MOD-PWR-B",
                    "name": "B路供电",
                },
            },
        ],
        "edges": [
            {
                "source": "local-a",
                "target": "local-b",
                "properties": {"connection_type": "parallel"},
            }
        ],
    }


def _relation(
    *,
    revision_id="revision-rbd-power",
    statement="A 支路失效时，B 支路仍满足模型成功判据。",
    contract_version=RBD_SINGLE_FAILURE_RELATION_VERSION,
    group_id="root.power",
    failed=("MOD-PWR-A",),
    surviving=("MOD-PWR-B",),
    validation_status="auto_validated",
    derivation="deterministic_topology_validation",
):
    return {
        "contract_version": contract_version,
        "relation_type": "single_failure_survivability",
        "revision_id": revision_id,
        "validation_status": validation_status,
        "derivation": derivation,
        "outcome": "success_criterion_satisfied",
        "group_id": group_id,
        "failed_business_ids": list(failed),
        "surviving_business_ids": list(surviving),
        "statement": statement,
    }


def _evidence(
    *,
    criterion="输出端可达",
    relations=(),
    record_id="RBD-POWER-A-B",
    revision_id="revision-rbd-power",
    validation_status="auto_validated",
    legacy_boundary="",
):
    graph = _graph(criterion=criterion, relations=relations)
    graph["metadata"]["validation_status"] = validation_status
    if legacy_boundary:
        graph["metadata"]["single_failure_boundary"] = legacy_boundary
    authority = (
        "validated_relation"
        if validation_status == "auto_validated"
        else "unconfirmed_extraction"
    )
    return make_evidence(
        source_kind="postgres",
        authority=authority,
        evidence_level=(
            "L2_CORROBORATED"
            if validation_status == "auto_validated"
            else "L4_UNRESOLVED"
        ),
        claim_type="rbd_structure",
        doc_type="rbd",
        doc_id="doc-rbd",
        record_id=record_id,
        title="双路供电 RBD",
        content=json.dumps(graph, ensure_ascii=False),
        display_summary=criterion,
        review_status=validation_status,
        metadata={
            "revision_id": revision_id,
            "diagram_key": record_id,
            "validation_status": validation_status,
            "is_active": True,
            "projection_status": "published",
        },
    )


def _bound_state(
    evidence,
    *,
    query="如果该支路暂时不可用会怎样？",
    target="MOD-PWR-A",
):
    anchor = {
        "doc_type": "rbd",
        "doc_id": evidence["doc_id"],
        "record_id": evidence["record_id"],
        "revision_id": evidence["metadata"]["revision_id"],
        "diagram_key": evidence["metadata"]["diagram_key"],
    }
    _, binding = bind_rbd_models(
        [evidence],
        required=True,
        raw_user_query=query,
        anchor_selectors=selected_rbd_anchor_selectors([anchor]),
    )
    evidence_id = evidence["evidence_id"]
    return {
        "query": query,
        "query_frame": {
            "coverage_scope": "focused_detail",
            "required_answer_fields": [
                "rbd_success_criterion",
                "rbd_single_failure_boundary",
            ],
            "topology_dependency": {
                "required": True,
                "canonical_business_ids": [target],
            },
        },
        "selected_conversation_anchors": [anchor],
        "rbd_query_binding": binding,
        "evidence_registry": [evidence],
        "context_evidence_ids": [evidence_id],
        "safety_decision": {
            "approved_evidence_ids": [evidence_id],
            "context_evidence_ids": [evidence_id],
        },
    }


class RbdSuccessCriterionProjectionTests(unittest.TestCase):
    def test_success_criterion_wording_never_creates_failure_relation(self):
        criteria = (
            "A/B两路电源中至少一路正常",
            "任一通道正常即可维持功能",
            "任一路失效时，另一路仍可工作",
            "minimum one branch remains available",
            "只要留下一条健康支路便满足任务条件",
        )
        for criterion in criteria:
            with self.subTest(criterion=criterion):
                evidence = _evidence(criterion=criterion)
                fields = _source_answer_structure_views._rbd_structure_answer_fields(evidence)
                self.assertEqual(criterion, fields["success_criterion"])
                self.assertNotIn("single_failure_boundary", fields)
                self.assertEqual(
                    [],
                    _source_answer_structure_views._rbd_target_boundary_clauses(
                        evidence, _bound_state(evidence)
                    ),
                )

    def test_legacy_string_boundary_and_parallel_topology_fail_closed(self):
        evidence = _evidence(
            criterion="输出端可达",
            legacy_boundary="任一路失效时另一条仍可工作",
        )
        fields = _source_answer_structure_views._rbd_structure_answer_fields(evidence)
        self.assertEqual(["并联"], fields["connection_types"])
        self.assertNotIn("single_failure_boundary", fields)
        self.assertEqual(
            {},
            _source_answer_structure_views._rbd_structure_answer_fields_for_state(
                evidence, _bound_state(evidence)
            ),
        )

    def test_bound_formal_structured_relation_is_exposed(self):
        relation = _relation()
        evidence = _evidence(relations=[relation])
        state = _bound_state(evidence)

        fields = _source_answer_structure_views._rbd_structure_answer_fields_for_state(evidence, state)

        self.assertEqual("输出端可达", fields["success_criterion"])
        self.assertEqual(relation["statement"], fields["single_failure_boundary"])
        self.assertEqual(
            [relation["statement"]],
            _source_answer_structure_views._rbd_target_boundary_clauses(evidence, state),
        )

    def test_natural_wording_does_not_change_structured_relation_authority(self):
        relation = _relation(statement="结构化验证确认目标支路失效后模型仍成功。")
        criteria = (
            "至少一路供电正常",
            "keep one feed alive",
            "任务成功条件见正式模型",
        )
        boundaries = []
        for criterion in criteria:
            evidence = _evidence(criterion=criterion, relations=[relation])
            boundaries.append(
                _source_answer_structure_views._rbd_target_boundary_clauses(
                    evidence,
                    _bound_state(
                        evidence,
                        query=f"换一种问法 {criterion}，目标仍是同一引用吗？",
                    ),
                )
            )
        self.assertEqual([[relation["statement"]]] * len(criteria), boundaries)

    def test_relation_requires_current_revision_and_closed_contract(self):
        invalid_relations = (
            _relation(revision_id="stale-revision"),
            _relation(contract_version="invented-v9"),
            _relation(validation_status="provisional"),
            _relation(derivation="llm_interpretation"),
        )
        for relation in invalid_relations:
            with self.subTest(relation=relation):
                evidence = _evidence(relations=[relation])
                self.assertEqual(
                    [],
                    _source_answer_structure_views._rbd_target_boundary_clauses(
                        evidence, _bound_state(evidence)
                    ),
                )

    def test_relation_requires_exact_group_and_business_members(self):
        invalid_relations = (
            _relation(group_id="root.unknown"),
            _relation(failed=("local-a",)),
            _relation(surviving=("A路供电",)),
            _relation(failed=("MOD-PWR-A", "MOD-PWR-B")),
            _relation(failed=("MOD-PWR-A",), surviving=("MOD-PWR-A",)),
        )
        for relation in invalid_relations:
            with self.subTest(relation=relation):
                evidence = _evidence(relations=[relation])
                self.assertEqual(
                    [],
                    _source_answer_structure_views._rbd_target_boundary_clauses(
                        evidence, _bound_state(evidence)
                    ),
                )

    def test_relation_must_match_exact_target_business_id(self):
        evidence = _evidence(relations=[_relation()])
        state = _bound_state(evidence, target="MOD-PWR-B")
        self.assertEqual([], _source_answer_structure_views._rbd_target_boundary_clauses(evidence, state))
        self.assertNotIn(
            "single_failure_boundary",
            _source_answer_structure_views._rbd_structure_answer_fields_for_state(evidence, state),
        )

    def test_missing_or_forged_rbd_binding_cannot_expose_relation(self):
        evidence = _evidence(relations=[_relation()])
        state = _bound_state(evidence)
        for binding in (
            None,
            {"required": True, "bound": True, "basis": "natural_subject"},
        ):
            with self.subTest(binding=binding):
                state["rbd_query_binding"] = binding
                self.assertEqual(
                    {},
                    _source_answer_structure_views._rbd_structure_answer_fields_for_state(
                        evidence, state
                    ),
                )

    def test_provisional_model_cannot_expose_relation(self):
        evidence = _evidence(
            relations=[_relation(validation_status="provisional")],
            validation_status="provisional",
        )
        self.assertEqual({}, _source_answer_structure_views._rbd_structure_answer_fields(evidence))
        self.assertEqual(
            {},
            _source_answer_structure_views._rbd_structure_answer_fields_for_state(
                evidence, _bound_state(evidence)
            ),
        )

    def test_prompt_does_not_receive_untyped_boundary(self):
        boundary = "任一路失效时另一条仍可工作"
        evidence = _evidence(legacy_boundary=boundary)
        context = answer_module._format_context(_bound_state(evidence))
        self.assertNotIn(boundary, context)


if __name__ == "__main__":
    unittest.main()
