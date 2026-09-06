import asyncio
import unittest

from backend.agent.contracts import make_evidence, make_tool_result
from backend.agent.nodes.evidence_gate import evidence_gate_node
from backend.agent.rbd_contract import bind_rbd_models


def _run(state):
    return asyncio.run(evidence_gate_node(state))


def _calculation_authorization(operation, target, basis):
    return [{
        "operation": operation,
        "mode": "calculate",
        "target": target,
        "basis": basis,
    }]


def _rbd_binding(*, query, revision_id, diagram_key, doc_id, status):
    row = {
        "diagram_type": "rbd",
        "revision_id": revision_id,
        "diagram_id": f"diagram-{revision_id}",
        "diagram_key": diagram_key,
        "doc_id": doc_id,
        "validation_status": status,
        "projection_status": "published",
        "is_active": True,
        "graph": {"vertices": [], "edges": []},
    }
    _rows, binding = bind_rbd_models(
        [row],
        required=True,
        raw_user_query=query,
        explicit_rbd_ids=[diagram_key],
    )
    return binding


class EvidenceGateCompatibilityTests(unittest.TestCase):
    def test_empty_registry_preserves_legacy_state_by_omission(self):
        result = _run(
            {
                "evidence_registry": [],
                "candidate_faults": [{"fault": "legacy"}],
                "diagnostic_confidence": 0.8,
                "diagnostic_context": "legacy context",
            }
        )
        self.assertEqual("unassessed", result["safety_decision"]["status"])
        self.assertNotIn("candidate_faults", result)
        self.assertNotIn("diagnostic_confidence", result)
        self.assertNotIn("diagnostic_context", result)


class EvidenceGateRulesTests(unittest.TestCase):
    def test_planned_answer_mode_is_invariant_under_intent_perturbation(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            record_id="FM-MODE-INVARIANT",
            content="结构化事实",
        )
        base = {
            "answer_mode": "knowledge_explanation",
            "evidence_registry": [evidence],
            "allowed_tools": ["fmeca_query"],
            "candidate_faults": [
                {
                    "fault": "候选记录",
                    "score": 0.5,
                    "exact_hmc_match": False,
                }
            ],
            "diagnostic_support": 0.5,
        }

        diagnostic_label = _run({**base, "intent": "symptom_diagnosis"})
        architecture_label = _run({**base, "intent": "architecture_query"})

        self.assertEqual(
            diagnostic_label["safety_decision"],
            architecture_label["safety_decision"],
        )
        self.assertFalse(
            diagnostic_label["safety_decision"]["requires_clarification"]
        )

    def test_candidate_seed_graph_path_is_context_only_until_query_binding_is_proved(self):
        path = make_evidence(
            source_kind="hugegraph",
            authority="validated_relation",
            evidence_level="L3_INFERRED",
            claim_type="relation_path",
            doc_type="fta",
            doc_id="doc-fta",
            record_id="path-1",
            content="候选起点到故障模式的两跳路径",
            match_type="candidate_seed_multihop_path",
            review_status="auto_validated",
            metadata={
                "validation_status": "auto_validated",
                "projection_status": "published",
                "is_active": True,
                "seed_binding_status": "candidate",
            },
        )
        result = _run(
            {
                "intent": "cause_analysis",
                "evidence_registry": [path],
                "allowed_tools": ["graph_traversal"],
            }
        )

        self.assertIn(path["evidence_id"], result["context_evidence_ids"])
        self.assertNotIn(path["evidence_id"], result["approved_evidence_ids"])
        self.assertIn(path["evidence_id"], result["safety_decision"]["blocked_evidence_ids"])

    def test_impact_context_prefers_postgres_record_and_drops_low_rerank_noise(self):
        direct = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            record_id="FM-SYN-002",
            title="输入滤波性能退化",
            content="局部影响为输入纹波和传导干扰增大。",
            retrieval_score=56,
        )
        generic_direct = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            record_id="FM-SYN-065",
            title="时钟频率漂移",
            content="晶振老化导致系统时基精度下降。",
            retrieval_score=4,
        )
        duplicate_projection = make_evidence(
            source_kind="opensearch",
            authority="search_projection",
            evidence_level="L3_INFERRED",
            claim_type="retrieval_hit",
            doc_type="fmeca",
            chunk_id="chunk-002",
            title="FM-SYN-002",
            content="滤波电容老化。",
            rerank_score=0.86,
        )
        generic_aging = make_evidence(
            source_kind="opensearch",
            authority="search_projection",
            evidence_level="L3_INFERRED",
            claim_type="retrieval_hit",
            doc_type="fmeca",
            chunk_id="chunk-065",
            title="FM-SYN-065",
            content="晶振老化。",
            rerank_score=0.004,
        )

        result = _run(
            {
                "intent": "impact_analysis",
                "evidence_registry": [
                    direct,
                    generic_direct,
                    duplicate_projection,
                    generic_aging,
                ],
                "allowed_tools": ["fmeca_query", "hybrid_search"],
            }
        )

        self.assertEqual([direct["evidence_id"]], result["context_evidence_ids"])
        self.assertEqual([direct["evidence_id"]], result["approved_evidence_ids"])

    def test_complete_collection_keeps_all_closed_members_despite_score_threshold(self):
        evidence = []
        for index in range(1, 16):
            record_id = f"FM-SYN-{index:03d}"
            evidence.append(make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type="fmeca_fact",
                doc_type="fmeca",
                doc_id="doc-fmeca",
                record_id=record_id,
                title=f"故障模式{index}",
                content=f"故障模式{index}的原因与影响。",
                # Ordinary focused explanation would retain only the first
                # few high-scoring rows.  Complete collection must retain all.
                retrieval_score=100 if index <= 3 else 1,
                metadata={"answer_fields": {
                    "record_code": record_id,
                    "failure_mode": f"故障模式{index}",
                    "collection_closed": True,
                    "collection_role": "member",
                    "collection_parent_id": "SUB-PWR",
                    "collection_parent_name": "电源与配电子系统",
                    "collection_member_kind": "failure_mode",
                    "collection_member_count": 15,
                    "collection_member_id": record_id,
                    "collection_member_name": f"故障模式{index}",
                }},
            ))

        result = _run({
            "intent": "cause_analysis",
            "query_frame": {
                "coverage_scope": "complete_collection",
                "collection_subject": "电源与配电模块的故障模式",
                "collection_member_kind": "failure_mode",
            },
            "collection_resolution": {"parent_id": "SUB-PWR"},
            "evidence_registry": evidence,
            "allowed_tools": ["function_locator", "fmeca_query"],
        })

        self.assertEqual(15, len(result["context_evidence_ids"]))
        self.assertEqual(15, len(result["approved_evidence_ids"]))
        self.assertEqual(
            {item["evidence_id"] for item in evidence},
            set(result["context_evidence_ids"]),
        )

    def test_direct_score_thresholds_do_not_compare_structured_and_semantic_scales(self):
        broad_structured = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            record_id="FM-BROAD",
            title="宽泛热故障",
            content="宽泛结构化候选",
            match_type="structured_query",
            retrieval_score=34,
        )
        exact_semantic = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            record_id="FM-SYN-006",
            title="A路间歇停机",
            content="由搜索定位后按 PostgreSQL 存储身份精确回填。",
            match_type="semantic_id_hydration",
            retrieval_score=0.94,
        )

        result = _run(
            {
                "intent": "cause_analysis",
                "evidence_registry": [broad_structured, exact_semantic],
                "allowed_tools": ["fmeca_query", "hybrid_search"],
            }
        )

        self.assertIn(exact_semantic["evidence_id"], result["context_evidence_ids"])
        self.assertIn(exact_semantic["evidence_id"], result["approved_evidence_ids"])

    def test_impact_question_does_not_inherit_diagnosis_clarification_contract(self):
        impact = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            record_id="FM-SYN-002",
            content="局部影响为输入纹波和传导干扰增大；最终影响为多个子系统间歇异常。",
        )
        result = _run(
            {
                "intent": "impact_analysis",
                "evidence_registry": [impact],
                "allowed_tools": ["fmeca_query"],
                "candidate_faults": [{"fault": "输入滤波性能退化", "score": 0.5}],
                "diagnostic_support": 0.2,
            }
        )
        decision = result["safety_decision"]
        self.assertEqual("answer", decision["status"])
        self.assertFalse(decision["requires_clarification"])
        self.assertEqual([], decision["questions"])
        self.assertNotIn("unique_diagnosis", decision["blocked_claims"])

    def test_direct_pg_is_approved_search_projection_is_context_only(self):
        pg = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_diagnosis",
            doc_type="hmc",
            doc_id="doc-hmc",
            record_id="HMC-001",
            content="显示链路故障",
        )
        search = make_evidence(
            source_kind="opensearch",
            authority="search_projection",
            evidence_level="L3_INFERRED",
            claim_type="retrieval_hit",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            chunk_id="chunk-1",
            content="可能的显示故障模式",
        )
        result = _run({"evidence_registry": [pg, search], "allowed_tools": ["hmc_query"]})
        decision = result["safety_decision"]
        self.assertEqual([pg["evidence_id"]], result["approved_evidence_ids"])
        self.assertEqual(
            [pg["evidence_id"], search["evidence_id"]],
            result["context_evidence_ids"],
        )
        self.assertIn(search["evidence_id"], decision["blocked_evidence_ids"])
        self.assertFalse(decision["can_claim_unique"])
        self.assertEqual("answer", decision["status"])

    def test_unconfirmed_graph_blocks_topology_and_formal_calculation(self):
        graph = make_evidence(
            source_kind="postgres",
            authority="unconfirmed_extraction",
            evidence_level="L3_INFERRED",
            claim_type="fta_structure",
            doc_type="fta",
            doc_id="doc-fta",
            record_id="diagram-1",
            review_status="pending",
            content="待确认故障树",
        )
        result = _run(
            {
                "answer_mode": "fault_diagnosis",
                "evidence_registry": [graph],
                "allowed_tools": ["fta_query", "fta_analyzer"],
            }
        )
        decision = result["safety_decision"]
        self.assertIn("formal_topology", decision["blocked_claims"])
        self.assertIn("formal_calculation", decision["blocked_claims"])
        self.assertTrue(decision["requires_clarification"])
        self.assertNotIn(graph["evidence_id"], result["approved_evidence_ids"])

    def test_requested_relation_with_empty_graph_result_blocks_formal_topology(self):
        query = "说明 MOD-A 到 FUNC-B 的影响传播"
        result = _run(
            {
                "query": query,
                "query_frame": {
                    "required_answer_fields": ["impact_propagation"],
                },
                "relation_expansion": {
                    "needed": True,
                    "relation_kind": "impact_propagation",
                    "reference_mode": "current_query",
                    "basis": "MOD-A 到 FUNC-B 的影响传播",
                    "selected_anchor_ids": [],
                    "reason": "需要核验结构关系",
                },
                "evidence_registry": [],
                "tool_results": [
                    make_tool_result(
                        tool="graph_traversal",
                        status="empty",
                        data={"path_count": 0},
                    )
                ],
                "allowed_tools": ["graph_traversal"],
            }
        )

        self.assertIn(
            "formal_topology", result["safety_decision"]["blocked_claims"]
        )

    def test_candidate_seed_path_cannot_satisfy_formal_topology(self):
        query = "说明 MOD-A 到 FUNC-B 的关系"
        graph = make_evidence(
            source_kind="hugegraph",
            authority="validated_relation",
            evidence_level="L3_INFERRED",
            claim_type="relation_path",
            doc_type="function_locator",
            doc_id="doc-function",
            record_id="path-candidate",
            review_status="auto_validated",
            content="MOD-A -> FUNC-B",
            metadata={
                "validation_status": "auto_validated",
                "is_active": True,
                "projection_status": "published",
                "seed_binding_status": "candidate",
            },
        )
        result = _run(
            {
                "query": query,
                "query_frame": {"required_answer_fields": ["relation_path"]},
                "relation_expansion": {
                    "needed": True,
                    "relation_kind": "relation_path",
                    "reference_mode": "current_query",
                    "basis": "MOD-A 到 FUNC-B 的关系",
                    "selected_anchor_ids": [],
                    "reason": "需要核验结构关系",
                },
                "evidence_registry": [graph],
                "allowed_tools": ["graph_traversal"],
            }
        )

        self.assertNotIn(graph["evidence_id"], result["approved_evidence_ids"])
        self.assertIn(
            "formal_topology", result["safety_decision"]["blocked_claims"]
        )

    def test_exact_seed_path_can_satisfy_formal_topology(self):
        query = "说明 MOD-A 到 FUNC-B 的关系"
        graph = make_evidence(
            source_kind="hugegraph",
            authority="validated_relation",
            evidence_level="L2_CORROBORATED",
            claim_type="relation_path",
            doc_type="function_locator",
            doc_id="doc-function",
            record_id="path-exact",
            review_status="auto_validated",
            content="MOD-A -> FUNC-B",
            metadata={
                "validation_status": "auto_validated",
                "is_active": True,
                "projection_status": "published",
                "seed_binding_status": "exact",
            },
        )
        result = _run(
            {
                "query": query,
                "query_frame": {"required_answer_fields": ["relation_path"]},
                "relation_expansion": {
                    "needed": True,
                    "relation_kind": "relation_path",
                    "reference_mode": "current_query",
                    "basis": "MOD-A 到 FUNC-B 的关系",
                    "selected_anchor_ids": [],
                    "reason": "需要核验结构关系",
                },
                "evidence_registry": [graph],
                "allowed_tools": ["graph_traversal"],
            }
        )

        self.assertNotIn(
            "formal_topology", result["safety_decision"]["blocked_claims"]
        )

    def test_confirmed_relation_and_derived_calculation_are_approved(self):
        query = "给出 RBD-GATE-1 的系统可靠度"
        graph = make_evidence(
            source_kind="hugegraph",
            authority="confirmed_relation",
            evidence_level="L1_DIRECT",
            claim_type="rbd_structure",
            doc_type="rbd",
            doc_id="doc-rbd",
            record_id="RBD-GATE-1",
            review_status="confirmed",
            content="A 与 B 串联",
            metadata={
                "revision_id": "revision-1",
                "diagram_id": "diagram-revision-1",
                "diagram_key": "RBD-GATE-1",
                "validation_status": "manual_confirmed",
                "is_active": True,
                "projection_status": "published",
                "rbd_block_business_ids": [],
            },
        )
        calculation = make_evidence(
            source_kind="calculation",
            authority="deterministic_derivation",
            evidence_level="L2_CORROBORATED",
            claim_type="rbd_reliability",
            doc_type="rbd",
            doc_id="doc-rbd",
            record_id="calc-1",
            content="R=0.98",
            derived_from=[graph["evidence_id"]],
        )
        result = _run(
            {
                "query": query,
                "evidence_registry": [graph, calculation],
                "allowed_tools": ["rbd_query", "rbd_calculator"],
                "rbd_query_binding": _rbd_binding(
                    query=query,
                    revision_id="revision-1",
                    diagram_key="RBD-GATE-1",
                    doc_id="doc-rbd",
                    status="manual_confirmed",
                ),
                "calculation_authorizations": _calculation_authorization(
                    "rbd_reliability", "rbd", "给出 RBD-GATE-1 的系统可靠度"
                ),
            }
        )
        self.assertEqual(
            [graph["evidence_id"], calculation["evidence_id"]],
            result["approved_evidence_ids"],
        )
        self.assertNotIn("formal_calculation", result["safety_decision"]["blocked_claims"])

    def test_auto_validated_relation_is_approved_for_structure_and_derivation(self):
        graph = make_evidence(
            source_kind="postgres",
            authority="validated_relation",
            evidence_level="L2_CORROBORATED",
            claim_type="fta_structure",
            doc_type="fta",
            doc_id="doc-fta",
            record_id="revision-auto",
            review_status="auto_validated",
            content="自动验证通过的故障树",
            metadata={
                "validation_status": "auto_validated",
                "is_active": True,
                "projection_status": "published",
            },
        )
        calculation = make_evidence(
            source_kind="calculation",
            authority="deterministic_derivation",
            evidence_level="L2_CORROBORATED",
            claim_type="fta_calculation",
            doc_type="fta",
            record_id="revision-auto",
            content="最小割集=[A]",
            derived_from=[graph["evidence_id"]],
            metadata={"numeric_complete": True},
        )

        result = _run(
            {
                "query": "列出不能再约简的底层组合",
                "evidence_registry": [graph, calculation],
                "allowed_tools": ["fta_query", "fta_analyzer"],
                "calculation_authorizations": _calculation_authorization(
                    "fta_cut_sets", "fta", "不能再约简的底层组合"
                ),
            }
        )

        self.assertEqual(
            [graph["evidence_id"], calculation["evidence_id"]],
            result["approved_evidence_ids"],
        )
        self.assertNotIn("formal_topology", result["safety_decision"]["blocked_claims"])
        self.assertNotIn("formal_calculation", result["safety_decision"]["blocked_claims"])

    def test_provisional_graph_is_context_only_and_blocks_formal_use(self):
        graph = make_evidence(
            source_kind="postgres",
            authority="unconfirmed_extraction",
            evidence_level="L4_UNRESOLVED",
            claim_type="rbd_structure",
            doc_type="rbd",
            record_id="revision-clue",
            review_status="provisional",
            content="视觉结构线索",
            metadata={
                "validation_status": "provisional",
                "is_active": True,
                "projection_status": "published",
            },
        )

        result = _run(
            {
                "query": "给出系统可靠度",
                "evidence_registry": [graph],
                "allowed_tools": ["rbd_query", "rbd_calculator"],
                "calculation_authorizations": _calculation_authorization(
                    "rbd_reliability", "rbd", "给出系统可靠度"
                ),
            }
        )

        self.assertIn(graph["evidence_id"], result["context_evidence_ids"])
        self.assertNotIn(graph["evidence_id"], result["approved_evidence_ids"])
        self.assertIn("formal_topology", result["safety_decision"]["blocked_claims"])
        self.assertIn("formal_calculation", result["safety_decision"]["blocked_claims"])

    def test_quarantined_graph_is_removed_from_reasoning_context(self):
        graph = make_evidence(
            source_kind="postgres",
            authority="unconfirmed_extraction",
            evidence_level="L4_UNRESOLVED",
            claim_type="fta_structure",
            doc_type="fta",
            record_id="revision-quarantined",
            review_status="quarantined",
            content="存在冲突的图",
            metadata={
                "validation_status": "quarantined",
                "is_active": True,
                "projection_status": "published",
            },
        )

        result = _run(
            {
                "evidence_registry": [graph],
                "allowed_tools": ["fta_query", "fta_analyzer"],
            }
        )

        self.assertNotIn(graph["evidence_id"], result["context_evidence_ids"])
        self.assertNotIn(graph["evidence_id"], result["approved_evidence_ids"])
        self.assertIn(graph["evidence_id"], result["safety_decision"]["blocked_evidence_ids"])
        self.assertIn("formal_topology", result["safety_decision"]["blocked_claims"])
        self.assertIn("formal_calculation", result["safety_decision"]["blocked_claims"])

    def test_derivation_without_parent_is_not_approved(self):
        calculation = make_evidence(
            source_kind="calculation",
            authority="deterministic_derivation",
            evidence_level="L2_CORROBORATED",
            claim_type="fta_probability",
            doc_type="fta",
            record_id="calc-orphan",
            content="P=0.1",
            derived_from=["ev-missing"],
        )
        result = _run({"evidence_registry": [calculation], "allowed_tools": ["fta_analyzer"]})
        self.assertEqual([], result["approved_evidence_ids"])
        self.assertIn(calculation["evidence_id"], result["safety_decision"]["blocked_evidence_ids"])
        self.assertIn("formal_calculation", result["safety_decision"]["blocked_claims"])

    def test_calculation_request_without_calculation_evidence_is_blocked(self):
        query = "检查 RBD-GATE-2"
        graph = make_evidence(
            source_kind="hugegraph",
            authority="confirmed_relation",
            evidence_level="L1_DIRECT",
            claim_type="rbd_structure",
            doc_type="rbd",
            doc_id="doc-rbd",
            record_id="RBD-GATE-2",
            review_status="confirmed",
            content="A 与 B 串联",
            metadata={
                "revision_id": "revision-1",
                "diagram_id": "diagram-revision-1",
                "diagram_key": "RBD-GATE-2",
                "validation_status": "manual_confirmed",
                "is_active": True,
                "projection_status": "published",
                "rbd_block_business_ids": [],
            },
        )
        result = _run(
            {
                "query": query,
                "evidence_registry": [graph],
                "allowed_tools": ["rbd_query", "rbd_calculator"],
                "rbd_query_binding": _rbd_binding(
                    query=query,
                    revision_id="revision-1",
                    diagram_key="RBD-GATE-2",
                    doc_id="doc-rbd",
                    status="manual_confirmed",
                ),
            }
        )
        self.assertIn("formal_calculation", result["safety_decision"]["blocked_claims"])
        self.assertFalse(result["safety_decision"]["requires_clarification"])
        self.assertIn(
            "incomplete_calculation",
            {
                item["kind"]
                for item in result["safety_decision"]["clarification_reasons"]
            },
        )
        self.assertEqual([], result["safety_decision"]["questions"])

    def test_confirmed_but_inactive_graph_cannot_support_topology(self):
        graph = make_evidence(
            source_kind="postgres",
            authority="confirmed_relation",
            evidence_level="L1_DIRECT",
            claim_type="fta_structure",
            doc_type="fta",
            record_id="revision-old",
            review_status="confirmed",
            content="旧版故障树",
            metadata={"is_active": False, "projection_status": "published"},
        )
        result = _run({"evidence_registry": [graph], "allowed_tools": ["fta_query"]})
        self.assertNotIn(graph["evidence_id"], result["approved_evidence_ids"])
        self.assertIn("formal_topology", result["safety_decision"]["blocked_claims"])
        self.assertTrue(
            any("不是当前有效版本" in value for value in result["safety_decision"]["warnings"])
        )

    def test_configuration_conflict_removes_both_from_approved_set(self):
        evidence = []
        for index, configuration in enumerate(("CONFIG-A", "CONFIG-B"), start=1):
            evidence.append(
                make_evidence(
                    source_kind="postgres",
                    authority="authoritative_record",
                    evidence_level="L1_DIRECT",
                    claim_type="fmeca_record",
                    doc_type="fmeca",
                    doc_id=f"doc-{index}",
                    record_id=f"F-{index}",
                    content=f"mode {index}",
                    scope={"configuration": configuration},
                )
            )
        result = _run({"evidence_registry": evidence, "allowed_tools": ["fmeca_query"]})
        self.assertEqual("conflict", result["safety_decision"]["scope_status"])
        self.assertEqual([], result["approved_evidence_ids"])
        self.assertTrue(result["safety_decision"]["requires_clarification"])

    def test_postgres_failure_without_direct_evidence_blocks_authoritative_answer(self):
        search = make_evidence(
            source_kind="opensearch",
            authority="search_projection",
            evidence_level="L3_INFERRED",
            claim_type="retrieval_hit",
            doc_type="hmc",
            doc_id="doc-1",
            chunk_id="chunk-1",
            content="检索线索",
        )
        failed = make_tool_result(
            tool="hmc_query",
            status="failed",
            backend="postgres",
            errors=["database unavailable"],
        )
        result = _run(
            {
                "evidence_registry": [search],
                "tool_results": [failed],
                "allowed_tools": ["hmc_query"],
            }
        )
        self.assertEqual("blocked", result["safety_decision"]["status"])
        self.assertIn("authoritative_diagnosis", result["safety_decision"]["blocked_claims"])
        self.assertIn(
            "authoritative_source_unavailable",
            result["safety_decision"]["blocked_claims"],
        )

    def test_exact_hmc_failure_is_not_hidden_by_other_postgres_evidence(self):
        fmeca = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            record_id="F-1",
            content="显示失效模式",
        )
        failed_hmc = make_tool_result(
            tool="hmc_query",
            status="failed",
            backend="postgres",
            errors=["HMC table unavailable"],
        )
        result = _run(
            {
                "intent": "exact_code",
                "evidence_registry": [fmeca],
                "tool_results": [failed_hmc],
                "allowed_tools": ["hmc_query", "fmeca_query"],
            }
        )
        decision = result["safety_decision"]
        self.assertEqual("blocked", decision["status"])
        self.assertNotIn("exact_hmc_fact", decision["blocked_claims"])
        self.assertIn("hmc_query_authoritative_facts", decision["blocked_claims"])
        self.assertEqual([], decision["unresolved_identity_slots"])
        self.assertIn(
            "authoritative_source_unavailable",
            {item["kind"] for item in decision["clarification_reasons"]},
        )
        self.assertTrue(any("不能替代" in value for value in decision["warnings"]))

    def test_exact_hmc_empty_result_is_not_replaced_by_search_projection(self):
        search = make_evidence(
            source_kind="opensearch",
            authority="search_projection",
            evidence_level="L3_INFERRED",
            claim_type="retrieval_hit",
            doc_type="hmc",
            chunk_id="hmc-search-1",
            content="相似代码线索",
        )
        empty_hmc = make_tool_result(
            tool="hmc_query",
            status="empty",
            backend="postgres",
        )
        result = _run(
            {
                "intent": "exact_code",
                "query": "LCN=31810001 HAMC=** HSC=0001",
                "evidence_registry": [search],
                "tool_results": [empty_hmc],
                "allowed_tools": ["hmc_query"],
            }
        )
        decision = result["safety_decision"]
        self.assertEqual("clarify", decision["status"])
        self.assertIn("exact_hmc_fact", decision["blocked_claims"])
        self.assertNotIn("authoritative_source_unavailable", decision["blocked_claims"])
        self.assertEqual([], decision["questions"])
        self.assertEqual(
            ["hmc_record_identity"],
            [item["slot"] for item in decision["unresolved_identity_slots"]],
        )
        self.assertIn(
            "authoritative_record_not_found",
            {item["kind"] for item in decision["clarification_reasons"]},
        )
        self.assertNotIn(search["evidence_id"], result["approved_evidence_ids"])

    def test_exact_hmc_requires_exact_code_match_not_keyword_pg_row(self):
        keyword = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            record_id="31810001**0001",
            content="仅关键词命中的 HMC 行",
            match_type="keyword",
        )
        result = _run(
            {
                "intent": "exact_code",
                "query": "LCN=31810001 HAMC=** HSC=0001",
                "evidence_registry": [keyword],
                "tool_results": [
                    make_tool_result(
                        tool="hmc_query",
                        status="ok",
                        backend="postgres",
                        evidence_ids=[keyword["evidence_id"]],
                    )
                ],
                "allowed_tools": ["hmc_query"],
            }
        )

        decision = result["safety_decision"]
        self.assertEqual("clarify", decision["status"])
        self.assertIn("exact_hmc_fact", decision["blocked_claims"])
        self.assertNotIn("authoritative_source_unavailable", decision["blocked_claims"])
        self.assertEqual([], decision["questions"])

    def test_single_lcn_hsc_component_match_is_accepted_without_guessing_hamc(self):
        component = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            record_id="91810001**0001",
            content="ICU 输入过压钳位失效",
            match_type="exact_components",
        )
        result = _run(
            {
                "intent": "exact_code",
                "query": "LCN 91810001 HSC 0001 故障如何修复",
                "evidence_registry": [component],
                "tool_results": [
                    make_tool_result(
                        tool="hmc_query",
                        status="ok",
                        backend="postgres",
                        evidence_ids=[component["evidence_id"]],
                    )
                ],
                "allowed_tools": ["hmc_query"],
                "candidate_faults": [
                    {
                        "fault": "ICU 输入过压钳位失效",
                        "structured_hmc_match": True,
                        "exact_hmc_match": False,
                    }
                ],
                "diagnostic_support": 0.84,
            }
        )

        decision = result["safety_decision"]
        self.assertEqual("answer", decision["status"])
        self.assertIn(component["evidence_id"], result["approved_evidence_ids"])
        self.assertFalse(decision["requires_clarification"])
        self.assertTrue(any("LCN+HSC" in value for value in decision["warnings"]))

    def test_multiple_lcn_hsc_component_matches_require_hamc_clarification(self):
        components = [
            make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type="hmc_fault",
                doc_type="hmc",
                record_id=record_id,
                content=content,
                match_type="exact_components",
            )
            for record_id, content in (
                ("91810001**0001", "通配记录"),
                ("91810001010001", "HAMC 01 记录"),
            )
        ]
        result = _run(
            {
                "intent": "exact_code",
                "query": "LCN 91810001 HSC 0001",
                "evidence_registry": components,
                "tool_results": [
                    make_tool_result(
                        tool="hmc_query",
                        status="ok",
                        backend="postgres",
                        evidence_ids=[item["evidence_id"] for item in components],
                    )
                ],
                "allowed_tools": ["hmc_query"],
                "diagnostic_support": 0.72,
            }
        )

        decision = result["safety_decision"]
        self.assertEqual("clarify", decision["status"])
        self.assertTrue(decision["requires_clarification"])
        self.assertEqual([], decision["questions"])
        self.assertEqual(
            ["hamc"],
            [item["slot"] for item in decision["unresolved_identity_slots"]],
        )
        source = decision["unresolved_identity_slots"][0]["source"]
        self.assertEqual(("tool_result", "hmc_query"), (source["kind"], source["name"]))

    def test_lcn_hsc_empty_match_clarifies_without_claiming_pg_unavailable(self):
        empty_hmc = make_tool_result(
            tool="hmc_query",
            status="empty",
            backend="postgres",
        )
        result = _run(
            {
                "intent": "exact_code",
                "query": "LCN 91810001 HSC 0001 故障如何修复",
                "evidence_registry": [],
                "tool_results": [empty_hmc],
                "allowed_tools": ["hmc_query"],
            }
        )

        decision = result["safety_decision"]
        self.assertEqual("clarify", decision["status"])
        self.assertIn("exact_hmc_fact", decision["blocked_claims"])
        self.assertNotIn("authoritative_source_unavailable", decision["blocked_claims"])
        self.assertEqual([], decision["questions"])
        self.assertEqual(
            ["hamc"],
            [item["slot"] for item in decision["unresolved_identity_slots"]],
        )
        self.assertIn(
            "authoritative_record_not_found",
            {item["kind"] for item in decision["clarification_reasons"]},
        )

    def test_single_weak_keyword_candidate_forces_clarification(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            record_id="F-1",
            content="显示链路异常",
        )
        result = _run(
            {
                "intent": "symptom_diagnosis",
                "answer_mode": "fault_diagnosis",
                "evidence_registry": [evidence],
                "allowed_tools": ["fmeca_query"],
                "candidate_faults": [
                    {
                        "fault": "显示链路异常",
                        "score": 0.55,
                        "exact_hmc_match": False,
                    }
                ],
                "diagnostic_support": 0.68,
            }
        )

        self.assertEqual("clarify", result["safety_decision"]["status"])
        self.assertTrue(result["safety_decision"]["requires_clarification"])
        self.assertEqual([], result["safety_decision"]["questions"])
        self.assertIn(
            "insufficient_evidence",
            {
                item["kind"]
                for item in result["safety_decision"]["clarification_reasons"]
            },
        )
        self.assertEqual(
            {"kind": "generic_clarification", "required": True},
            result["safety_decision"]["clarification_fallback"],
        )

    def test_new_contract_empty_registry_is_assessed_not_legacy_compatible(self):
        empty_hmc = make_tool_result(tool="hmc_query", status="empty", backend="postgres")
        result = _run(
            {
                "intent": "exact_code",
                "answer_mode": "exact_code_lookup",
                "evidence_registry": [],
                "tool_results": [empty_hmc],
                "allowed_tools": ["hmc_query"],
            }
        )
        self.assertEqual("clarify", result["safety_decision"]["status"])
        self.assertNotIn("exact_hmc_fact", result["safety_decision"]["blocked_claims"])
        self.assertEqual([], result["safety_decision"]["unresolved_identity_slots"])
        self.assertEqual([], result["safety_decision"]["questions"])

    def test_composite_function_tool_does_not_mislabel_opensearch_error_as_pg_failure(self):
        function_result = make_tool_result(
            tool="function_locator",
            status="failed",
            backend="postgres+opensearch",
            data={"postgres_errors": [], "opensearch_errors": ["index unavailable"]},
            errors=["index unavailable"],
        )
        result = _run(
            {
                "intent": "function_location",
                "evidence_registry": [],
                "tool_results": [function_result],
                "allowed_tools": ["function_locator"],
            }
        )
        decision = result["safety_decision"]
        self.assertNotIn("function_locator_authoritative_facts", decision["blocked_claims"])
        self.assertFalse(any("PostgreSQL 权威事实查询失败" in value for value in decision["warnings"]))


if __name__ == "__main__":
    unittest.main()
