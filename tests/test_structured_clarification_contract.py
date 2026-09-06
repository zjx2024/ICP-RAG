import asyncio
import importlib.util
import json
import sys
import types
import unittest
from unittest.mock import patch

try:
    _psycopg2_spec = importlib.util.find_spec("psycopg2")
except ValueError:
    _psycopg2_spec = object() if "psycopg2" in sys.modules else None
if _psycopg2_spec is None:
    psycopg2_stub = types.ModuleType("psycopg2")
    psycopg2_stub.__spec__ = importlib.util.spec_from_loader("psycopg2", loader=None)

    def _unavailable_connect(*_args, **_kwargs):
        raise RuntimeError("psycopg2 unavailable in focused test environment")

    psycopg2_stub.connect = _unavailable_connect
    sys.modules["psycopg2"] = psycopg2_stub

from backend.agent.clarification_contract import (
    make_clarification_identity,
    make_clarification_reason,
    make_missing_input_requirement,
)
from backend.agent.contracts import make_evidence, make_tool_result
from backend.agent.nodes.evidence_gate import evidence_gate_node
from backend.agent.rbd_contract import bind_rbd_models
from backend.agent.tools import icp_diagnosis as diagnosis_module


def _run(coroutine):
    return asyncio.run(coroutine)


class ClarificationSchemaTests(unittest.TestCase):
    def test_unknown_closed_kinds_and_identity_doc_types_fail_closed(self):
        with self.assertRaises(ValueError):
            make_missing_input_requirement(
                kind="open_ended_question",
                slot="anything",
                source_kind="state",
                source_name="test",
            )
        with self.assertRaises(ValueError):
            make_clarification_reason(
                kind="ask_for_a_symptom_example",
                source_kind="evidence_gate",
                source_name="test",
            )
        with self.assertRaises(ValueError):
            make_clarification_identity(doc_type="eighth_document_type")
        with self.assertRaises(ValueError):
            make_missing_input_requirement(
                kind="identity_slot",
                slot="forged",
                source_kind="state",
                source_name="test",
                identity={
                    "doc_type": "hmc",
                    "record_ids": [],
                    "fields": {"arbitrary_open_field": "value"},
                },
            )


class DiagnosisStructuredClarificationTests(unittest.TestCase):
    def test_diagnosis_mode_is_invariant_under_intent_perturbation(self):
        async def run(intent):
            return await diagnosis_module.icp_diagnosis_node(
                {
                    "intent": intent,
                    "answer_mode": "knowledge_explanation",
                    "query": "说明这条记录的局部影响",
                    "rewritten_query": "说明这条记录的局部影响",
                    "query_frame": {
                        "required_answer_fields": ["local_impact"],
                        "response_mode": {
                            "mode": "explanation",
                            "basis": "说明这条记录的局部影响",
                        },
                    },
                    "allowed_tools": ["evidence_synthesis"],
                }
            )

        first = _run(run("symptom_diagnosis"))
        second = _run(run("architecture_query"))
        for result in (first, second):
            self.assertEqual([], result["missing_input_requirements"])
            self.assertEqual(
                {"kind": "generic_clarification", "required": False},
                result["clarification_fallback"],
            )
            self.assertNotIn(
                "fault_reasoning", [item["tool"] for item in result["tool_calls"]]
            )
        self.assertEqual(first["clarification_reasons"], second["clarification_reasons"])

    def test_rbd_source_contract_cannot_drift_to_function_path_by_intent(self):
        query = "请说明这份 RBD 资料能支持哪些结论"
        base = {
            "answer_mode": "knowledge_explanation",
            "query": query,
            "query_frame": {
                "required_answer_fields": [
                    "rbd_success_criterion",
                    "rbd_path_structure",
                ],
                "requested_sources": [
                    {
                        "doc_type": "rbd",
                        "basis": query,
                        "requested_fields": [
                            "rbd_success_criterion",
                            "rbd_path_structure",
                        ],
                        "question_role": "capability",
                    }
                ],
                "response_focus": ["source_capability"],
            },
        }
        routes = []
        for intent in ("entity_profile", "architecture_query", "reliability_query"):
            routes.append(
                diagnosis_module._structured_query_routes(
                    {**base, "intent": intent},
                    allowed_tools={
                        "function_locator",
                        "rbd_query",
                        "evidence_synthesis",
                    },
                    raw_user_query=query,
                    explicit_business_codes=[],
                )
            )

        self.assertTrue(all(route == routes[0] for route in routes[1:]))
        self.assertFalse(routes[0]["architecture_query"])
        self.assertFalse(routes[0]["function_knowledge_query"])
        self.assertFalse(routes[0]["function_knowledge_only"])

    def test_structured_source_boundary_is_wording_invariant_and_basis_closed(self):
        def state(query, basis=None):
            return {
                "query": query,
                "query_frame": {
                    "response_focus": ["source_limitation"],
                    "requested_sources": [
                        {
                            "doc_type": "rbd",
                            "basis": query if basis is None else basis,
                            "requested_fields": ["rbd_success_criterion"],
                            "question_role": "limitation",
                        }
                    ],
                },
            }

        for query in (
            "请限定这份资料所能支撑的范围",
            "基于当前材料，哪些判断还不能成立",
        ):
            self.assertTrue(
                diagnosis_module._source_boundary_explanation_from_contract(
                    state(query), query
                )
            )
        forged = state("基于当前材料说明范围", basis="当前 材料")
        self.assertFalse(
            diagnosis_module._source_boundary_explanation_from_contract(
                forged, forged["query"]
            )
        )
        self.assertFalse(
            diagnosis_module._source_boundary_explanation_from_contract(
                {
                    "query": "任意文本",
                    "query_frame": {
                        "requested_facets": ["capability_boundary", "limitation"]
                    },
                },
                "任意文本",
            )
        )

    def test_machine_gap_preserves_basis_from_structured_source_locator(self):
        requirement = make_missing_input_requirement(
            kind="calculation_parameter",
            slot="mission_time",
            source_kind="query_frame",
            source_name="missing_parameters",
            basis="任务时间",
        )

        missing, identities = diagnosis_module._machine_missing_input_contract(
            {
                "missing_parameters": ["mission_time"],
                "query_frame": {"missing_input_requirements": [requirement]},
            }
        )

        self.assertEqual([], identities)
        self.assertEqual("任务时间", missing[0]["basis"])

    def test_ambiguous_component_hmc_emits_traced_hamc_slot_not_a_question(self):
        rows = [
            {
                "doc_id": "doc-hmc",
                "hmc_code": record_id,
                "lcn": "91810001",
                "hamc": hamc,
                "hsc": "0001",
                "fault_desc": description,
                "repair_method": "",
                "match_type": "ambiguous_components",
            }
            for record_id, hamc, description in (
                ("91810001**0001", "**", "候选甲"),
                ("91810001010001", "01", "候选乙"),
            )
        ]
        state = {
            "intent": "exact_code",
            "query": "LCN 91810001 HSC 0001",
            "rewritten_query": "LCN 91810001 HSC 0001",
            "allowed_tools": ["hmc_query", "evidence_synthesis"],
            "sql_limit": 8,
        }
        with (
            patch.object(diagnosis_module, "_query_hmc", return_value=rows),
            patch.object(
                diagnosis_module,
                "_enrich_function_sources",
                side_effect=lambda values, errors=None: list(values),
            ),
        ):
            result = _run(diagnosis_module.icp_diagnosis_node(state))

        self.assertEqual("", result["clarification_question"])
        self.assertNotIn("【仍需确认】", result["diagnostic_context"])
        self.assertEqual(
            ["hamc"],
            [item["slot"] for item in result["unresolved_identity_slots"]],
        )
        slot = result["unresolved_identity_slots"][0]
        self.assertEqual(
            ("tool_result", "hmc_query"),
            (slot["source"]["kind"], slot["source"]["name"]),
        )
        self.assertEqual(
            {"lcn": "91810001", "hsc": "0001"},
            slot["identity"]["fields"],
        )
        rendered = json.dumps(result, ensure_ascii=False)
        for fixed_example in ("上电/初始化/运行中", "过温", "显示异常", "请补充"):
            self.assertNotIn(fixed_example, rendered)

    def test_candidate_score_logic_is_paraphrase_invariant_and_adds_no_slot(self):
        async def run(query):
            return await diagnosis_module.icp_diagnosis_node(
                {
                    "intent": "symptom_diagnosis",
                    "answer_mode": "fault_diagnosis",
                    "query": query,
                    "rewritten_query": query,
                    "query_frame": {
                        "required_answer_fields": ["failure_mode"],
                        "diagnostic_closure": {
                            "required": True,
                            "required_dimensions": ["failure_mode"],
                            "basis": query,
                        },
                    },
                    "allowed_tools": ["evidence_synthesis"],
                }
            )

        first = _run(run("这个现象怎么回事？"))
        second = _run(run("请分析一下当前异常。"))
        for key in (
            "missing_input_requirements",
            "unresolved_identity_slots",
            "clarification_reasons",
            "clarification_fallback",
        ):
            self.assertEqual(first[key], second[key])
        self.assertEqual([], first["missing_input_requirements"])
        self.assertEqual(
            {"kind": "generic_clarification", "required": True},
            first["clarification_fallback"],
        )
        self.assertEqual("", first["clarification_question"])


class EvidenceGateStructuredClarificationTests(unittest.TestCase):
    def test_exact_code_intent_alone_cannot_create_hmc_identity_slots(self):
        async def gate(query):
            return await evidence_gate_node(
                {
                    "intent": "exact_code",
                    "query": query,
                    "allowed_tools": ["hmc_query"],
                    "evidence_registry": [],
                    "tool_results": [
                        make_tool_result(
                            tool="hmc_query", status="empty", backend="postgres"
                        )
                    ],
                }
            )

        first = _run(gate("帮我核对这个报码"))["safety_decision"]
        second = _run(gate("它在资料里是什么意思"))["safety_decision"]
        for decision in (first, second):
            self.assertNotIn("exact_hmc_fact", decision["blocked_claims"])
            self.assertEqual([], decision["missing_input_requirements"])
            self.assertEqual([], decision["unresolved_identity_slots"])
            self.assertEqual([], decision["questions"])
        self.assertEqual(first["clarification_reasons"], second["clarification_reasons"])
        self.assertEqual(first["clarification_fallback"], second["clarification_fallback"])

    def test_parsed_hmc_gap_is_the_only_component_slot(self):
        components = [
            make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type="hmc_fault",
                doc_type="hmc",
                record_id=record_id,
                content=description,
                match_type="exact_components",
            )
            for record_id, description in (
                ("91810001**0001", "候选甲"),
                ("91810001010001", "候选乙"),
            )
        ]
        result = _run(
            evidence_gate_node(
                {
                    "intent": "exact_code",
                    "query": "LCN 91810001 HSC 0001",
                    "allowed_tools": ["hmc_query"],
                    "evidence_registry": components,
                    "tool_results": [
                        make_tool_result(
                            tool="hmc_query",
                            status="ok",
                            backend="postgres",
                            evidence_ids=[item["evidence_id"] for item in components],
                        )
                    ],
                }
            )
        )
        decision = result["safety_decision"]
        self.assertEqual(["hamc"], [item["slot"] for item in decision["unresolved_identity_slots"]])
        self.assertEqual([], decision["questions"])
        identity = decision["unresolved_identity_slots"][0]["identity"]
        self.assertEqual("hmc", identity["doc_type"])
        self.assertEqual({"lcn": "91810001", "hsc": "0001"}, identity["fields"])

    def test_complete_hmc_no_match_requests_identity_not_three_fixed_slots(self):
        decision = _run(
            evidence_gate_node(
                {
                    "intent": "exact_code",
                    "query": "LCN=31810001 HAMC=** HSC=0001",
                    "allowed_tools": ["hmc_query"],
                    "evidence_registry": [],
                    "tool_results": [
                        make_tool_result(
                            tool="hmc_query", status="empty", backend="postgres"
                        )
                    ],
                }
            )
        )["safety_decision"]
        self.assertEqual(
            ["hmc_record_identity"],
            [item["slot"] for item in decision["unresolved_identity_slots"]],
        )
        self.assertTrue(
            {"lcn", "hamc", "hsc"}
            <= set(decision["unresolved_identity_slots"][0]["identity"]["fields"])
        )
        self.assertEqual([], decision["questions"])

    def test_numeric_algorithm_gap_becomes_a_parameter_slot(self):
        query = "计算 RBD-GAP-1 的可靠度"
        model = {
            "diagram_type": "rbd",
            "revision_id": "rev-1",
            "diagram_id": "diagram-1",
            "diagram_key": "RBD-GAP-1",
            "doc_id": "doc-rbd-1",
            "validation_status": "manual_confirmed",
            "projection_status": "published",
            "is_active": True,
            "graph": {"vertices": [], "edges": []},
        }
        _models, binding = bind_rbd_models(
            [model],
            required=True,
            raw_user_query=query,
            explicit_rbd_ids=["RBD-GAP-1"],
        )
        parent = make_evidence(
            source_kind="postgres",
            authority="confirmed_relation",
            evidence_level="L1_DIRECT",
            claim_type="rbd_structure",
            doc_type="rbd",
            record_id="rev-1",
            review_status="confirmed",
            content="formal graph",
            metadata={
                "revision_id": "rev-1",
                "diagram_id": "diagram-1",
                "diagram_key": "RBD-GAP-1",
                "doc_id": "doc-rbd-1",
                "validation_status": "manual_confirmed",
                "is_active": True,
                "projection_status": "published",
            },
        )
        calculation = make_evidence(
            source_kind="calculation",
            authority="deterministic_derivation",
            evidence_level="L2_CORROBORATED",
            claim_type="rbd_calculation",
            doc_type="rbd",
            record_id="rev-1",
            content=json.dumps(
                {
                    "system_reliability": None,
                    "missing_reliability_blocks": ["BLOCK-7"],
                }
            ),
            derived_from=[parent["evidence_id"]],
            metadata={"numeric_complete": False},
        )
        decision = _run(
            evidence_gate_node(
                {
                    "intent": "reliability_query",
                    "query": query,
                    "calculation_authorizations": [
                        {
                            "operation": "rbd_reliability",
                            "mode": "calculate",
                            "target": "rbd",
                            "basis": query,
                        }
                    ],
                    "allowed_tools": ["rbd_query", "rbd_calculator"],
                    "rbd_query_binding": binding,
                    "evidence_registry": [parent, calculation],
                }
            )
        )["safety_decision"]
        self.assertIn(
            "block_reliability",
            [item["slot"] for item in decision["missing_input_requirements"]],
        )
        item = next(
            item
            for item in decision["missing_input_requirements"]
            if item["slot"] == "block_reliability"
        )
        self.assertEqual("BLOCK-7", item["identity"]["fields"]["block_ids"])
        self.assertEqual([], decision["questions"])

    def test_unknown_incoming_contract_kinds_do_not_cross_the_gate(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            record_id="FM-1",
            content="direct fact",
        )
        decision = _run(
            evidence_gate_node(
                {
                    "intent": "impact_analysis",
                    "allowed_tools": ["fmeca_query"],
                    "evidence_registry": [evidence],
                    "missing_input_requirements": [
                        {
                            "kind": "invented_kind",
                            "slot": "invented_slot",
                            "source": {
                                "kind": "state",
                                "name": "forged",
                                "evidence_ids": [],
                            },
                        },
                        {
                            "kind": "identity_slot",
                            "slot": "forged_identity",
                            "source": {
                                "kind": "state",
                                "name": "forged",
                                "evidence_ids": [],
                            },
                            "identity": {
                                "doc_type": "eighth_document_type",
                                "record_ids": ["X-1"],
                                "fields": {"arbitrary_open_field": "value"},
                            },
                        },
                    ],
                    "clarification_reasons": [
                        {
                            "kind": "invented_reason",
                            "source": {
                                "kind": "evidence_gate",
                                "name": "forged",
                                "evidence_ids": [],
                            },
                        },
                        {
                            "kind": "unresolved_identity",
                            "source": {
                                "kind": "evidence_gate",
                                "name": "forged",
                                "evidence_ids": [],
                            },
                            "identity": {
                                "doc_type": "hmc",
                                "record_ids": "not-a-list",
                                "fields": {"arbitrary_open_field": "value"},
                            },
                        },
                    ],
                }
            )
        )["safety_decision"]
        self.assertEqual([], decision["missing_input_requirements"])
        self.assertEqual([], decision["clarification_reasons"])
        self.assertEqual([], decision["questions"])

    def test_unbound_hmc_mfl_relation_uses_closed_relation_reason(self):
        hmc = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            record_id="31810001**0001",
            content="direct HMC fact",
            match_type="exact_code",
        )
        decision = _run(
            evidence_gate_node(
                {
                    "intent": "maintenance_query",
                    "query": "LCN=31810001 HAMC=** HSC=0001",
                    "allowed_tools": ["hmc_query", "mfl_query"],
                    "evidence_registry": [hmc],
                    "tool_results": [
                        make_tool_result(
                            tool="hmc_query",
                            status="ok",
                            backend="postgres",
                            evidence_ids=[hmc["evidence_id"]],
                        )
                    ],
                    "hmc_mfl_binding": {
                        "required": True,
                        "bound": False,
                        "hmc_codes": ["31810001**0001"],
                        "excluded_mfl_codes": ["MFL-NEIGHBOUR"],
                    },
                }
            )
        )["safety_decision"]
        reason = next(
            item
            for item in decision["clarification_reasons"]
            if item["kind"] == "unresolved_relation"
        )
        self.assertEqual("hmc_mfl_binding", reason["source"]["name"])
        self.assertIn("hmc_mfl_relation", reason["blocked_claims"])
        self.assertEqual([], decision["questions"])


if __name__ == "__main__":
    unittest.main()
