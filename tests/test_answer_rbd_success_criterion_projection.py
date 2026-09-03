import json
import unittest

from backend.agent.contracts import make_evidence
from backend.agent.nodes import answer as answer_module


def _rbd_evidence(
    criterion,
    *,
    record_id="RBD-POWER-A-B",
    title="A/B 电源 RBD",
    review_status="auto_validated",
    authority="validated_relation",
    graph_extra=None,
):
    graph = {
        "diagram_type": "rbd",
        "metadata": {
            "success_criterion": criterion,
            "validation_status": review_status,
        },
        "vertices": [],
        "edges": [],
    }
    graph.update(graph_extra or {})
    return make_evidence(
        source_kind="postgres",
        authority=authority,
        evidence_level="L2_CORROBORATED",
        claim_type="rbd_structure",
        doc_type="rbd",
        doc_id="doc-rbd",
        record_id=record_id,
        title=title,
        content=json.dumps(graph, ensure_ascii=False),
        display_summary=criterion,
        review_status=review_status,
        metadata={
            "validation_status": review_status,
            "is_active": True,
            "projection_status": "published",
        },
    )


def _parallel_group_vertices():
    def vertex(node_id, branch_id, module_id, name, module_name):
        return {
            "id": node_id,
            "label": "ReliabilityBlock",
            "properties": {
                "connection_type": "parallel",
                "branch_id": branch_id,
                "module_id": module_id,
                "name": name,
                "module_name": module_name,
            },
        }

    return [
        vertex(
            "psu-a",
            "root.1:branch:1",
            "MOD-PWR-PSUA",
            "PSU-A",
            "A路稳压供电模块",
        ),
        vertex(
            "psu-b",
            "root.1:branch:2",
            "MOD-PWR-PSUB",
            "PSU-B",
            "B路稳压供电模块",
        ),
        vertex(
            "gpp-a",
            "root.2:branch:1",
            "MOD-GPP-A",
            "GPP-A",
            "A通道通用处理模块",
        ),
        vertex(
            "gpp-b",
            "root.2:branch:2",
            "MOD-GPP-B",
            "GPP-B",
            "B通道通用处理模块",
        ),
        vertex(
            "bca",
            "root.3:branch:1",
            "MOD-IFC-BCA",
            "BCA",
            "A通道总线控制模块",
        ),
        vertex(
            "bcb",
            "root.3:branch:2",
            "MOD-IFC-BCB",
            "BCB",
            "B通道总线控制模块",
        ),
    ]


def _grouped_rbd_evidence(criterion, record_id):
    return _rbd_evidence(
        criterion,
        record_id=record_id,
        title=f"{record_id} RBD",
        graph_extra={"vertices": _parallel_group_vertices()},
    )


def _identity_group_vertices():
    def vertex(node_id, branch_id, module_id, function_id, name, module_name):
        return {
            "id": node_id,
            "label": "ReliabilityBlock",
            "properties": {
                "connection_type": "parallel",
                "branch_id": branch_id,
                "module_id": module_id,
                "function_id": function_id,
                "name": name,
                "module_name": module_name,
            },
        }

    return [
        vertex(
            "psu-a",
            "root.power:branch:1",
            "MOD-PWR-PSUA",
            "FUNC-120",
            "PSU-A",
            "A路稳压供电模块",
        ),
        vertex(
            "psu-b",
            "root.power:branch:2",
            "MOD-PWR-PSUB",
            "FUNC-121",
            "PSU-B",
            "B路稳压供电模块",
        ),
        vertex(
            "aux-a",
            "root.aux:branch:1",
            "MOD-PWR-AUXA",
            "FUNC-920",
            "AUX-A",
            "A路辅助供电模块",
        ),
        vertex(
            "aux-b",
            "root.aux:branch:2",
            "MOD-PWR-AUXB",
            "FUNC-921",
            "AUX-B",
            "B路辅助供电模块",
        ),
    ]


def _identity_group_rbd_evidence(criterion, record_id):
    return _rbd_evidence(
        criterion,
        record_id=record_id,
        title=f"{record_id} RBD",
        graph_extra={"vertices": _identity_group_vertices()},
    )


def _edge_only_group_graph(*, shared_business_id=""):
    def vertex(node_id, module_id="", name="", business_id=""):
        return {
            "id": node_id,
            "label": "ReliabilityBlock",
            "properties": {
                "module_id": module_id,
                "name": name,
                "business_id": business_id,
            },
        }

    def edge(source, target, branch_id):
        return {
            "source": source,
            "target": target,
            "properties": {
                "connection_type": "parallel",
                "branch_id": branch_id,
            },
        }

    shared = vertex(
        "shared-entry",
        name="共享入口",
        business_id=shared_business_id,
    )
    return {
        "vertices": [
            shared,
            vertex("psu-a", "MOD-PWR-PSUA", "PSU-A"),
            vertex("psu-b", "MOD-PWR-PSUB", "PSU-B"),
            vertex("aux-a", "MOD-PWR-AUXA", "AUX-A"),
            vertex("aux-b", "MOD-PWR-AUXB", "AUX-B"),
        ],
        "edges": [
            edge("shared-entry", "psu-a", "root.power:branch:1"),
            edge("shared-entry", "psu-b", "root.power:branch:2"),
            edge("shared-entry", "aux-a", "root.aux:branch:1"),
            edge("shared-entry", "aux-b", "root.aux:branch:2"),
        ],
    }


def _approved_state(*evidence, topology_dependency=None):
    evidence_ids = [item["evidence_id"] for item in evidence]
    state = {
        "query": "一路电源损坏后还能维持功能吗",
        "intent": "reliability_query",
        "query_targets": [{"intent": "reliability_query"}],
        "calculation_requests": [],
        "query_frame": {
            "coverage_scope": "focused_detail",
            "required_answer_fields": [
                "rbd_success_criterion",
                "rbd_single_failure_boundary",
            ],
        },
        "evidence_registry": list(evidence),
        "context_evidence_ids": evidence_ids,
        "safety_decision": {
            "approved_evidence_ids": evidence_ids,
            "context_evidence_ids": evidence_ids,
        },
    }
    if topology_dependency:
        state["query_frame"]["topology_dependency"] = dict(topology_dependency)
        state["query"] = (
            f"若{topology_dependency['subject']}损坏，会有什么影响，还能维持功能吗"
        )
    return state


class RbdSuccessCriterionProjectionTests(unittest.TestCase):
    def test_explicit_success_criteria_project_single_failure_boundary(self):
        criteria = (
            "A/B两路电源中至少一路正常",
            "任一通道正常即可维持功能",
        )
        for criterion in criteria:
            with self.subTest(criterion=criterion):
                evidence = _rbd_evidence(criterion)
                fields = answer_module._rbd_structure_answer_fields(evidence)

                self.assertEqual(criterion, fields["success_criterion"])
                self.assertEqual(criterion, fields["single_failure_boundary"])
                self.assertFalse(
                    any("reliability" in key.casefold() for key in fields)
                )

                missing, manifest = answer_module._target_completion_manifest(
                    "", _approved_state(evidence)
                )
                supports = {
                    facet for entry in manifest for facet in entry["supports"]
                }
                self.assertIn("rbd_success_criterion", supports)
                self.assertNotIn("rbd_single_failure_boundary", supports)
                self.assertNotIn("rbd_single_failure_boundary", missing)

    def test_multi_normal_criterion_stops_at_the_target_clause(self):
        criterion = (
            "电源链正常,A/B总线至少一路正常,"
            "网关、协议转换与时钟模块正常。"
        )
        evidence = _grouped_rbd_evidence(criterion, "RBD-EXTERNAL-COMMS")
        state = _approved_state(
            evidence,
            topology_dependency={
                "required": True,
                "subject": "A通道对外发送链路",
                "subject_kind": "channel",
                "failure_state": "中断",
                "question_aspects": ["impact", "continued_operation"],
                "canonical_business_ids": ["MOD-IFC-BCA", "FUNC-310"],
            },
        )

        fields = answer_module._rbd_structure_answer_fields(evidence)
        self.assertEqual(
            "A/B总线至少一路正常",
            fields["single_failure_boundary"],
        )
        self.assertEqual(
            ["A/B总线至少一路正常"],
            answer_module._rbd_target_boundary_clauses(evidence, state),
        )
        self.assertEqual(
            ["RBD-EXTERNAL-COMMS"],
            [item["record_id"] for item in answer_module._context_evidence(state)],
        )

    def test_internal_comma_between_failure_condition_and_result_is_preserved(self):
        criteria = (
            "任一路失效时，另一路仍可工作",
            "任一通道正常，即可维持功能",
        )
        for criterion in criteria:
            with self.subTest(criterion=criterion):
                fields = answer_module._rbd_structure_answer_fields(
                    _rbd_evidence(criterion)
                )
                self.assertEqual(criterion, fields["single_failure_boundary"])

    def test_answer_fields_success_criterion_is_projected_without_graph_inference(self):
        criterion = "任一通道正常即可维持功能"
        evidence = make_evidence(
            source_kind="postgres",
            authority="validated_relation",
            evidence_level="L2_CORROBORATED",
            claim_type="rbd_structure",
            doc_type="rbd",
            doc_id="doc-rbd",
            record_id="RBD-EXPLICIT-FIELD",
            title="明确成功判据",
            content="来源表明确写明任一通道正常即可维持功能。",
            review_status="auto_validated",
            metadata={
                "validation_status": "auto_validated",
                "is_active": True,
                "projection_status": "published",
                "answer_fields": {"success_criterion": criterion},
            },
        )

        fields = answer_module._rbd_structure_answer_fields(evidence)

        self.assertEqual(criterion, fields["success_criterion"])
        self.assertEqual(criterion, fields["single_failure_boundary"])

    def test_parallel_topology_alone_does_not_create_failure_boundary(self):
        evidence = _rbd_evidence(
            "输出端可达",
            graph_extra={
                "edges": [
                    {
                        "source": "A",
                        "target": "OUT",
                        "properties": {"connection_type": "parallel"},
                    },
                    {
                        "source": "B",
                        "target": "OUT",
                        "properties": {"connection_type": "parallel"},
                    },
                ]
            },
        )

        fields = answer_module._rbd_structure_answer_fields(evidence)

        self.assertEqual(["并联"], fields["connection_types"])
        self.assertNotIn("single_failure_boundary", fields)
        _, manifest = answer_module._target_completion_manifest(
            "", _approved_state(evidence)
        )
        supports = {facet for entry in manifest for facet in entry["supports"]}
        self.assertNotIn("rbd_single_failure_boundary", supports)

    def test_negative_or_failed_path_wording_is_not_a_success_boundary(self):
        criteria = (
            "两路并联，其中一路故障会导致功能无法维持",
            "至少一路工作异常时系统失败",
        )
        for criterion in criteria:
            with self.subTest(criterion=criterion):
                fields = answer_module._rbd_structure_answer_fields(
                    _rbd_evidence(criterion)
                )
                self.assertNotIn("single_failure_boundary", fields)

    def test_provisional_relation_cannot_be_promoted_by_explicit_wording(self):
        criterion = "A/B两路电源中至少一路正常"
        # Deliberately construct an inconsistent payload: even if a stale item
        # claims validated_relation, provisional status must fail closed.
        evidence = _rbd_evidence(
            criterion,
            review_status="provisional",
            authority="validated_relation",
        )

        self.assertEqual({}, answer_module._rbd_structure_answer_fields(evidence))
        self.assertEqual(
            ([], []),
            answer_module._target_completion_manifest("", _approved_state(evidence)),
        )

    def test_supply_path_keeps_only_criterion_for_the_same_parallel_group(self):
        power = _grouped_rbd_evidence(
            "A/B两路电源中至少一路正常",
            "RBD-POWER",
        )
        gpp = _grouped_rbd_evidence(
            "GPP-A/B至少一路正常",
            "RBD-GPP",
        )
        bus = _grouped_rbd_evidence(
            "A/B总线至少一路正常",
            "RBD-BUS",
        )
        state = _approved_state(
            power,
            gpp,
            bus,
            topology_dependency={
                "required": True,
                "subject": "A路电源",
                "subject_kind": "supply_path",
                "failure_state": "损坏",
                "question_aspects": ["impact", "continued_operation"],
                "canonical_business_ids": ["MOD-PWR-PSUA", "PSU-A"],
            },
        )

        context = answer_module._context_evidence(state)
        self.assertEqual(["RBD-POWER"], [item["record_id"] for item in context])

        missing, manifest = answer_module._target_completion_manifest("", state)
        self.assertEqual(
            {"rbd_success_criterion", "rbd_single_failure_boundary"},
            set(missing),
        )
        self.assertEqual(["RBD-POWER"], [item["record_id"] for item in manifest])
        self.assertEqual(
            "A/B两路电源中至少一路正常",
            manifest[0]["answer_fields"]["success_criterion"],
        )

    def test_gpp_subject_keeps_gpp_parallel_group_and_rejects_power_group(self):
        power = _grouped_rbd_evidence(
            "A/B两路电源中至少一路正常",
            "RBD-POWER",
        )
        gpp = _grouped_rbd_evidence(
            "GPP-A/B至少一路正常",
            "RBD-GPP",
        )
        state = _approved_state(
            power,
            gpp,
            topology_dependency={
                "required": True,
                "subject": "GPP-A",
                "subject_kind": "channel",
                "failure_state": "失效",
                "question_aspects": ["continued_operation"],
                "canonical_business_ids": ["MOD-GPP-A", "GPP-A"],
            },
        )

        context = answer_module._context_evidence(state)
        self.assertEqual(["RBD-GPP"], [item["record_id"] for item in context])
        _, manifest = answer_module._target_completion_manifest("", state)
        self.assertEqual(["RBD-GPP"], [item["record_id"] for item in manifest])

    def test_durable_business_ids_authorize_group_when_subject_text_cannot(self):
        power = _identity_group_rbd_evidence(
            "PSU-A/B至少一路正常",
            "RBD-PRIMARY-POWER",
        )
        auxiliary = _identity_group_rbd_evidence(
            "AUX-A/B至少一路正常",
            "RBD-AUX-POWER",
        )
        state = _approved_state(
            power,
            auxiliary,
            topology_dependency={
                "required": True,
                # The resolved follow-up wording has no safe lexical overlap
                # with either persisted group; authorization must use the
                # durable Function Locator/FMECA identity handoff.
                "subject": "该供电支路",
                "subject_kind": "supply_path",
                "failure_state": "失效",
                "question_aspects": ["continued_operation"],
                "canonical_business_ids": [
                    "MOD-PWR-PSUA",
                    "PSU-A",
                    "FUNC-120",
                ],
            },
        )

        context = answer_module._context_evidence(state)
        self.assertEqual(
            ["RBD-PRIMARY-POWER"],
            [item["record_id"] for item in context],
        )
        _, manifest = answer_module._target_completion_manifest("", state)
        self.assertEqual(
            ["RBD-PRIMARY-POWER"],
            [item["record_id"] for item in manifest],
        )

    def test_mismatched_durable_business_ids_override_short_text_overlap(self):
        power = _identity_group_rbd_evidence(
            "PSU-A/B至少一路正常",
            "RBD-PRIMARY-POWER",
        )
        state = _approved_state(
            power,
            topology_dependency={
                "required": True,
                # This wording would lexically match the PSU group, but the
                # authoritative binding explicitly identifies another group.
                "subject": "A路电源",
                "subject_kind": "supply_path",
                "failure_state": "失效",
                "question_aspects": ["continued_operation"],
                "canonical_business_ids": [
                    "MOD-PWR-AUXA",
                    "AUX-A",
                    "FUNC-920",
                ],
            },
        )

        self.assertEqual([], answer_module._context_evidence(state))
        self.assertEqual(
            ([], []),
            answer_module._target_completion_manifest("", state),
        )

    def test_edge_only_branch_properties_authorize_canonical_endpoint_id(self):
        evidence = _rbd_evidence(
            "PSU-A/B至少一路正常",
            record_id="RBD-EDGE-ONLY-POWER",
            graph_extra=_edge_only_group_graph(),
        )
        state = _approved_state(
            evidence,
            topology_dependency={
                "required": True,
                "subject": "该供电支路",
                "subject_kind": "supply_path",
                "failure_state": "失效",
                "question_aspects": ["continued_operation"],
                "canonical_business_ids": ["MOD-PWR-PSUA"],
            },
        )

        groups = answer_module._rbd_parallel_groups(json.loads(evidence["content"]))
        power_group = next(group for group in groups if group["key"] == "root.power")

        self.assertIn("MOD-PWR-PSUA", power_group["business_ids"])
        self.assertIn("psu-a", power_group["labels"])
        self.assertEqual(
            ["RBD-EDGE-ONLY-POWER"],
            [item["record_id"] for item in answer_module._context_evidence(state)],
        )
        _, manifest = answer_module._target_completion_manifest("", state)
        self.assertEqual(
            ["RBD-EDGE-ONLY-POWER"],
            [item["record_id"] for item in manifest],
        )

    def test_shared_edge_endpoint_identity_across_groups_fails_closed(self):
        evidence = _rbd_evidence(
            "PSU-A/B至少一路正常",
            record_id="RBD-SHARED-ENDPOINT",
            graph_extra=_edge_only_group_graph(shared_business_id="FUNC-SHARED"),
        )
        state = _approved_state(
            evidence,
            topology_dependency={
                "required": True,
                "subject": "共享入口",
                "subject_kind": "path",
                "failure_state": "失效",
                "question_aspects": ["continued_operation"],
                "canonical_business_ids": ["FUNC-SHARED"],
            },
        )

        groups = answer_module._rbd_parallel_groups(json.loads(evidence["content"]))

        self.assertEqual(
            "",
            answer_module._unique_rbd_group_match_by_business_ids(
                groups, ["FUNC-SHARED"]
            ),
        )
        self.assertEqual([], answer_module._context_evidence(state))
        self.assertEqual(
            ([], []),
            answer_module._target_completion_manifest("", state),
        )

    def test_business_id_separator_collision_does_not_authorize_group(self):
        groups = [
            {
                "key": "root.wrong",
                "connection_type": "parallel",
                "business_ids": ["A-BC"],
                "labels": ["错误分支"],
            }
        ]

        self.assertEqual(
            "",
            answer_module._unique_rbd_group_match_by_business_ids(
                groups, ["AB-C"]
            ),
        )

    def test_code_shaped_vertex_name_cannot_authorize_another_business_id(self):
        graph = {
            "vertices": [
                {
                    "id": "local-a",
                    "properties": {
                        "connection_type": "parallel",
                        "branch_id": "root.other:branch:1",
                        "module_id": "MOD-OTHER-A",
                        "name": "MOD-PWR-PSUA",
                    },
                },
                {
                    "id": "local-b",
                    "properties": {
                        "connection_type": "parallel",
                        "branch_id": "root.other:branch:2",
                        "module_id": "MOD-OTHER-B",
                    },
                },
            ]
        }
        groups = answer_module._rbd_parallel_groups(graph)

        self.assertIn("MOD-PWR-PSUA", groups[0]["labels"])
        self.assertNotIn("MOD-PWR-PSUA", groups[0]["business_ids"])
        self.assertEqual(
            "",
            answer_module._unique_rbd_group_match_by_business_ids(
                groups, ["MOD-PWR-PSUA"]
            ),
        )

    def test_topology_filter_does_not_drop_requested_fmeca_impacts(self):
        power = _grouped_rbd_evidence(
            "A/B两路电源中至少一路正常",
            "RBD-POWER",
        )
        fmeca = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-PSU-A",
            title="A路输出欠压",
            content="A路稳压供电能力丧失，系统失去一条供电冗余。",
            metadata={
                "answer_fields": {
                    "product_flag": "PSU-A / FUNC-120",
                    "failure_mode": "A路输出欠压",
                    "local_impact": "A路稳压供电能力丧失。",
                    "final_impact": "系统失去一条供电冗余。",
                }
            },
        )
        state = _approved_state(
            fmeca,
            power,
            topology_dependency={
                "required": True,
                "subject": "A路电源",
                "subject_kind": "supply_path",
                "failure_state": "损坏",
                "question_aspects": ["impact", "continued_operation"],
                "canonical_business_ids": ["MOD-PWR-PSUA", "PSU-A"],
            },
        )
        state["query_frame"]["required_answer_fields"] = [
            "local_impact",
            "final_impact",
            "rbd_success_criterion",
            "rbd_single_failure_boundary",
        ]
        state["query_targets"].insert(0, {"intent": "impact_analysis"})

        missing, manifest = answer_module._target_completion_manifest("", state)

        self.assertEqual(
            {
                "local_impact",
                "final_impact",
                "rbd_success_criterion",
                "rbd_single_failure_boundary",
            },
            set(missing),
        )
        self.assertEqual(
            {"FM-PSU-A", "RBD-POWER"},
            {item["record_id"] for item in manifest},
        )

    def test_bound_topology_excludes_fmeca_for_another_business_object(self):
        def fmeca(record_id, product_flag, mode):
            return make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type="fmeca_fact",
                doc_type="fmeca",
                doc_id="doc-fmeca",
                record_id=record_id,
                title=mode,
                content=mode,
                metadata={
                    "answer_fields": {
                        "product_flag": product_flag,
                        "failure_mode": mode,
                        "local_impact": f"{mode}的局部影响",
                        "final_impact": f"{mode}的最终影响",
                    }
                },
            )

        relevant = fmeca(
            "FM-SYN-032",
            "BCA / FUNC-310",
            "A通道发送占用",
        )
        unrelated = fmeca(
            "FM-SYN-021",
            "GPP-A / FUNC-220",
            "A通道内部存储校验错",
        )
        state = _approved_state(
            relevant,
            unrelated,
            topology_dependency={
                "required": True,
                "subject": "A通道对外发送链路",
                "subject_kind": "data_path",
                "failure_state": "中断",
                "question_aspects": ["impact", "continued_operation"],
                "canonical_business_ids": ["MOD-IFC-BCA", "FUNC-310"],
            },
        )
        state["query"] = (
            "内部数据已经正常生成，但A通道对外发送链路断了，外部设备还能收到业务数据吗？"
        )

        filtered = answer_module._filter_answer_domain_evidence(
            state,
            [relevant, unrelated],
        )

        self.assertEqual(
            ["FM-SYN-032"],
            [item["record_id"] for item in filtered],
        )

    def test_no_topology_dependency_keeps_criteria_but_not_failure_boundary(self):
        evidence = (
            _grouped_rbd_evidence(
                "A/B两路电源中至少一路正常",
                "RBD-POWER",
            ),
            _grouped_rbd_evidence(
                "GPP-A/B至少一路正常",
                "RBD-GPP",
            ),
            _grouped_rbd_evidence(
                "A/B总线至少一路正常",
                "RBD-BUS",
            ),
        )
        state = _approved_state(*evidence)
        state["allowed_tools"] = ["rbd_query"]
        state["tool_calls"] = [{"tool": "rbd_query", "status": "ok"}]

        context = answer_module._context_evidence(state)
        self.assertEqual(
            {"RBD-POWER", "RBD-GPP", "RBD-BUS"},
            {item["record_id"] for item in context},
        )
        _, manifest = answer_module._target_completion_manifest("", state)
        self.assertEqual(
            {"RBD-POWER", "RBD-GPP", "RBD-BUS"},
            {item["record_id"] for item in manifest},
        )
        supports = {
            facet for item in manifest for facet in (item.get("supports") or [])
        }
        self.assertIn("rbd_success_criterion", supports)
        self.assertNotIn("rbd_single_failure_boundary", supports)
        self.assertTrue(
            all(
                "single_failure_boundary" not in (item.get("answer_fields") or {})
                for item in manifest
            )
        )

    def test_answer_prompt_hides_unbound_failure_boundary_from_raw_summary(self):
        evidence = _rbd_evidence(
            "输出端可达",
            graph_extra={
                "metadata": {
                    "success_criterion": "输出端可达",
                    "validation_status": "auto_validated",
                    "single_failure_boundary": "任一路失效时另一条仍可工作",
                }
            },
        )
        evidence["display_summary"] = (
            "成功判据：输出端可达；单路边界：任一路失效时另一条仍可工作"
        )
        state = _approved_state(evidence)

        context = answer_module._format_context(state)

        self.assertIn("输出端可达", context)
        self.assertNotIn("任一路失效时另一条仍可工作", context)


if __name__ == "__main__":
    unittest.main()
