import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from backend.agent.contracts import make_evidence, make_tool_result
from backend.agent.nodes import adaptive_tools as adaptive_module
from backend.agent.tools import icp_calculation as calculation_module
from backend.agent.nodes.adaptive_tools import (
    adaptive_retrieval_node,
    adaptive_tool_route_node,
    route_after_adaptive_retrieval,
    route_after_adaptive_tools,
)


def _run(coro):
    return asyncio.run(coro)


def _evidence(record_id="FM-001"):
    return make_evidence(
        source_kind="postgres",
        authority="authoritative_record",
        evidence_level="L1_DIRECT",
        claim_type="fmeca_fact",
        doc_type="fmeca",
        doc_id="doc-fmeca",
        record_id=record_id,
        title="输入滤波性能退化",
        content="滤波电容老化会使输入纹波增大。",
        metadata={"answer_fields": {"final_impact": "输入纹波增大"}},
    )


class AdaptiveToolRouteTests(unittest.TestCase):
    def _route(self, state, response):
        with (
            patch.object(adaptive_module.llm_service, "_call_model", return_value=response),
            patch.object(
                adaptive_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
            patch.object(
                adaptive_module.agent_model_role_service,
                "get_model_by_role",
                return_value=None,
            ),
        ):
            return _run(adaptive_tool_route_node(state))

    @staticmethod
    def _exact_hmc_state(
        *,
        query="查询 31810001-**-0001",
        evidence_code="31810001**0001",
        required_fields=(),
    ):
        components = {
            "hmc_code": evidence_code,
            "lcn": evidence_code[:8],
            "hamc": evidence_code[8:10],
            "hsc": evidence_code[10:],
            "fault_symptom": "ICU 输入过压钳位失效",
        }
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            doc_id="doc-hmc",
            record_id=evidence_code,
            source_ref="HMC手册#row=1",
            title=evidence_code,
            content="ICU 输入过压钳位失效",
            match_type="exact_code",
            metadata={
                **components,
                "answer_fields": components,
            },
        )
        return {
            "query": query,
            # These coarse labels are intentionally unrelated to shortcut
            # authority; only the exact identity and typed obligations matter.
            "intent": "hybrid",
            "query_targets": [{"intent": "history_query"}],
            "query_frame": {
                "required_answer_fields": list(required_fields),
            },
            "authorized_tools": [
                "hmc_query", "hybrid_search", "evidence_synthesis",
            ],
            "allowed_tools": ["hmc_query", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(
                    tool="hmc_query",
                    status="ok",
                    data={"count": 1},
                    evidence_ids=[evidence["evidence_id"]],
                )
            ],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [{
                "logical_id": f"sg:b:hmc_code:{evidence_code}",
                "entity_kind": "hmc_code",
                "source_tool": "hmc_query",
                "binding_status": "exact",
            }],
        }

    def test_record_identity_coverage_requires_id_and_primary_content_pair(self):
        identity_shapes = (
            (
                "function_locator",
                "component_fact",
                {"component_id": "MOD-PWR-ICU"},
                {"component_name": "输入调理模块"},
            ),
            (
                "fmeca",
                "fmeca_fact",
                {"failure_mode_id": "FM-SYN-002"},
                {"failure_mode": "输入滤波性能退化"},
            ),
            (
                "mfl",
                "maintenance_locator",
                {"record_code": "MFL-SYN-041"},
                {"module_failure": "协议转换模块故障"},
            ),
            (
                "mfl",
                "maintenance_locator",
                {"record_code": "MFL-SYN-042"},
                {"sur_failure": "SUR 接口板收发异常"},
            ),
            (
                "mfl",
                "maintenance_locator",
                {"record_code": "MFL-SYN-043"},
                {"subcircuit_failure": "转换队列配置异常"},
            ),
            (
                "fault_history",
                "historical_case",
                {"case_id": "CASE-SYN-017"},
                {"fault_symptom": "上电后输入电压反复波动"},
            ),
            (
                "hmc",
                "hmc_fault",
                {"hmc_code": "91810001**0001"},
                {"fault_symptom": "ICU 输入过压钳位失效"},
            ),
        )
        for doc_type, claim_type, identity, description in identity_shapes:
            with self.subTest(doc_type=doc_type, description=tuple(description)):
                record_id = next(iter(identity.values()))
                id_only = make_evidence(
                    source_kind="postgres",
                    authority="authoritative_record",
                    evidence_level="L1_DIRECT",
                    claim_type=claim_type,
                    doc_type=doc_type,
                    doc_id=f"doc-{doc_type}",
                    record_id=record_id,
                    title=record_id,
                    content="结构化业务记录",
                    metadata={"answer_fields": dict(identity)},
                )
                paired = make_evidence(
                    source_kind="postgres",
                    authority="authoritative_record",
                    evidence_level="L1_DIRECT",
                    claim_type=claim_type,
                    doc_type=doc_type,
                    doc_id=f"doc-{doc_type}",
                    record_id=record_id,
                    title=record_id,
                    content="结构化业务记录",
                    metadata={
                        "answer_fields": {**identity, **description},
                    },
                )

                self.assertNotIn(
                    "record_identity",
                    adaptive_module._coverage_slots(
                        {"evidence_registry": [id_only]}
                    ),
                )
                self.assertIn(
                    "record_identity",
                    adaptive_module._coverage_slots(
                        {"evidence_registry": [paired]}
                    ),
                )

    def test_direct_evidence_can_stop_without_graph_or_broad_retrieval(self):
        evidence = _evidence()
        state = {
            "query": "滤波电容老化会有什么影响？",
            "intent": "impact_analysis",
            "query_targets": [{"intent": "impact_analysis"}],
            "authorized_tools": [
                "function_locator", "fmeca_query",
                "hybrid_search", "graph_traversal", "evidence_synthesis",
            ],
            "allowed_tools": ["fmeca_query", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(
                    tool="fmeca_query",
                    status="ok",
                    data={"count": 1},
                    evidence_ids=[evidence["evidence_id"]],
                )
            ],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [
                {"logical_id": "sg:b:fmeca_record:1", "binding_status": "candidate"}
            ],
        }
        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "直接证据足够"}),
        )
        self.assertEqual([], result["pending_tools"])
        self.assertFalse(result["adaptive_graph_requested"])
        self.assertEqual({}, result["graph_traversal_request"])
        self.assertNotIn("graph_traversal", result["allowed_tools"])
        self.assertEqual("icp_calculation", route_after_adaptive_tools(result))

    def test_structural_cause_contract_adds_fta_after_fmeca_reason(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-CHAIN-001",
            title="输入波动",
            content="结构化记录",
            metadata={"answer_fields": {"fault_cause": "采样链路失效"}},
        )
        followups = adaptive_module._completion_followups(
            {
                "query": "运行中复位",
                "intent": "cause_analysis",
                "query_targets": [{"intent": "cause_analysis"}],
                "query_frame": {
                    "coverage_scope": "focused_detail",
                    "required_answer_fields": [
                        "fault_cause", "structural_cause_chain",
                    ],
                },
                "evidence_registry": [evidence],
            },
            executed=["fmeca_query"],
            available=["fta_query", "hybrid_search"],
        )

        # A scalar FMECA cause cannot stand in for the validated FTA chain.
        self.assertIn("fta_query", followups)

    def test_current_incident_closure_adds_handling_and_history_when_available(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-CLOSE-001",
            title="运行中复位",
            content="结构化记录",
            metadata={"answer_fields": {"fault_cause": "电源瞬断"}},
        )
        followups = adaptive_module._completion_followups(
            {
                "query": "运行中复位",
                "intent": "cause_analysis",
                "query_targets": [{"intent": "cause_analysis"}],
                "query_frame": {
                    "coverage_scope": "focused_detail",
                    "required_answer_fields": ["fault_cause"],
                    "diagnostic_closure": {
                        "required": True,
                        "required_dimensions": [
                            "maintenance_action", "history_result",
                        ],
                        "basis": "运行中复位",
                        "reason": "当前异常需要有限处置闭环",
                    },
                },
                "evidence_registry": [evidence],
            },
            executed=["fmeca_query"],
            available=["mfl_query", "hmc_query", "fault_history_query"],
        )

        self.assertEqual(
            ["hmc_query", "fault_history_query", "mfl_query"], followups
        )

    def test_validated_fta_structure_covers_structural_chain_only(self):
        fta = make_evidence(
            source_kind="postgres",
            authority="validated_relation",
            evidence_level="L2_CORROBORATED",
            claim_type="fta_structure",
            doc_type="fta",
            doc_id="doc-fta",
            record_id="FTA-001",
            title="启动失败",
            content="图结构证据",
            metadata={
                "answer_fields": {
                    "structural_cause_chain": "顶事件经 OR 门连接两个基本事件",
                    "top_event": "启动失败",
                }
            },
        )
        coverage = adaptive_module._coverage_slots({"evidence_registry": [fta]})
        self.assertIn("structural_cause_chain", coverage)
        self.assertNotIn("impact", coverage)

    def test_complete_exact_hmc_skips_adaptive_model_and_hugegraph(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            doc_id="doc-hmc",
            record_id="31810001**0001",
            title="31810001**0001",
            content="ICU 输入过压铁位失效",
            match_type="exact_code",
        )
        state = {
            "query": "查询 31810001-**-0001",
            "intent": "exact_code",
            "query_targets": [{"intent": "exact_code"}],
            "authorized_tools": ["hmc_query", "hybrid_search", "evidence_synthesis"],
            "allowed_tools": ["hmc_query", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(
                    tool="hmc_query",
                    status="ok",
                    data={"count": 1},
                    evidence_ids=[evidence["evidence_id"]],
                )
            ],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [
                {"logical_id": "sg:b:hmc_code:1", "binding_status": "exact"}
            ],
        }
        with patch.object(
            adaptive_module.llm_service,
            "_call_model",
            side_effect=AssertionError("精确 HMC 不应启动补查路由模型"),
        ):
            result = _run(adaptive_tool_route_node(state))
        self.assertEqual("exact_hmc_shortcut", result["adaptive_decision"]["source"])
        self.assertEqual({}, result["graph_traversal_request"])
        self.assertNotIn("graph_traversal", result["allowed_tools"])

    def test_exact_hmc_shortcut_uses_identity_and_field_coverage_not_intent(self):
        state = self._exact_hmc_state(
            required_fields=("hmc_identity", "hmc_symptom"),
        )
        state["query_frame"].update({
            "source_obligations": [{
                "tool": "hmc_query",
                "doc_type": "hmc",
                "fields": ["hmc_identity", "hmc_symptom"],
            }],
            "requested_source_tools": ["hmc_query"],
        })

        self.assertTrue(
            adaptive_module._complete_exact_hmc_answer_available(state)
        )

    def test_exact_hmc_shortcut_rejects_uncovered_compound_and_closure_fields(self):
        query = "查询 31810001-**-0001，并形成维修与历史恢复闭环"
        state = self._exact_hmc_state(
            query=query,
            required_fields=("hmc_identity", "maintenance_api"),
        )
        state["query_targets"] = [
            {"intent": "exact_code", "required_fields": ["hmc_identity"]},
            {
                "intent": "maintenance_query",
                "required_fields": ["maintenance_api"],
            },
        ]
        state["query_frame"]["diagnostic_closure"] = {
            "required": True,
            "required_dimensions": ["maintenance_action", "history_result"],
            "basis": "形成维修与历史恢复闭环",
        }
        state["authorized_tools"].extend(["mfl_query", "fault_history_query"])

        self.assertFalse(
            adaptive_module._complete_exact_hmc_answer_available(state)
        )
        result = self._route(
            state,
            json.dumps({
                "decision": "retrieve",
                "next_tools": ["mfl_query", "fault_history_query"],
                "reason": "补齐结构化义务",
            }, ensure_ascii=False),
        )
        self.assertTrue(
            {"mfl_query", "fault_history_query"}.issubset(
                result["pending_tools"]
            )
        )
        self.assertNotEqual(
            "exact_hmc_shortcut", result["adaptive_decision"]["source"]
        )

    def test_exact_hmc_shortcut_rejects_wrong_code_evidence(self):
        state = self._exact_hmc_state(evidence_code="31810001**0002")

        self.assertFalse(
            adaptive_module._complete_exact_hmc_answer_available(state)
        )

    def test_exact_hmc_shortcut_requires_full_raw_code_and_one_pg_identity(self):
        incomplete = self._exact_hmc_state(
            query="查询 LCN 31810001 HSC 0001"
        )
        multiple_codes = self._exact_hmc_state(
            query="比较 31810001-**-0001 与 31810001-**-0002"
        )
        duplicate = self._exact_hmc_state()
        second = dict(duplicate["evidence_registry"][0])
        second.update({
            "evidence_id": "ev-hmc-second-document",
            "doc_id": "doc-hmc-second",
            "source_ref": "另一份HMC手册#row=9",
        })
        duplicate["evidence_registry"].append(second)
        duplicate["tool_results"][0]["evidence_ids"].append(
            second["evidence_id"]
        )

        for state in (incomplete, multiple_codes, duplicate):
            with self.subTest(query=state["query"], count=len(state["evidence_registry"])):
                self.assertFalse(
                    adaptive_module._complete_exact_hmc_answer_available(state)
                )

    def test_exact_hmc_shortcut_rejects_uncovered_explicit_source(self):
        state = self._exact_hmc_state(
            query="查询 31810001-**-0001，并回查维修清单",
            required_fields=("hmc_identity",),
        )
        state["query_frame"]["source_obligations"] = [{
            "tool": "mfl_query",
            "doc_type": "mfl",
            "fields": [],
            "question_role": "evidence",
        }]
        state["query_frame"]["requested_source_tools"] = ["mfl_query"]
        state["requested_source_tools"] = ["mfl_query"]
        state["authorized_tools"].append("mfl_query")

        self.assertFalse(
            adaptive_module._complete_exact_hmc_answer_available(state)
        )
        result = self._route(
            state,
            json.dumps({
                "decision": "retrieve",
                "next_tools": ["mfl_query"],
                "reason": "显式来源仍未执行",
            }, ensure_ascii=False),
        )
        self.assertEqual(["mfl_query"], result["pending_tools"])

    def test_exact_hmc_shortcut_rejects_relation_or_calculation_contract(self):
        relation_query = "查询 31810001-**-0001，并核对它如何连接到下游"
        relation_state = self._exact_hmc_state(query=relation_query)
        relation_state["relation_expansion"] = {
            "needed": True,
            "relation_kind": "relation_path",
            "reference_mode": "current_query",
            "basis": "核对它如何连接到下游",
        }
        relation_state["authorized_tools"].append("graph_traversal")

        calculation_query = "查询 31810001-**-0001，并计算该 RBD 的可靠度"
        calculation_state = self._exact_hmc_state(query=calculation_query)
        calculation_state["calculation_authorizations"] = [{
            "operation": "rbd_reliability",
            "mode": "calculate",
            "target": "rbd",
            "basis": "计算该 RBD 的可靠度",
        }]
        calculation_state["calculation_requests"] = ["rbd_reliability"]

        for state in (relation_state, calculation_state):
            with self.subTest(query=state["query"]):
                self.assertFalse(
                    adaptive_module._complete_exact_hmc_answer_available(state)
                )

    def test_empty_direct_result_can_trigger_one_scoped_hybrid_followup(self):
        state = {
            "query": "滤波电容老化会有什么影响？",
            "intent": "impact_analysis",
            "query_targets": [{"intent": "impact_analysis"}],
            "authorized_tools": ["fmeca_query", "hybrid_search", "evidence_synthesis"],
            "allowed_tools": ["fmeca_query", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(tool="fmeca_query", status="empty", data={"count": 0})
            ],
            "evidence_registry": [],
            "graph_seed_bindings": [],
        }
        result = self._route(
            state,
            json.dumps(
                {
                    "decision": "retrieve",
                    "next_tools": ["hybrid_search"],
                    "reason": "直接记录为空，补充原文召回",
                },
                ensure_ascii=False,
            ),
        )
        self.assertEqual(["hybrid_search"], result["pending_tools"])
        self.assertEqual("adaptive_retrieval", result["next_action"])
        self.assertEqual("adaptive_retrieval", route_after_adaptive_tools(result))

    def test_relationship_question_can_select_graph_only_after_seed_exists(self):
        history_evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="historical_case",
            doc_type="fault_history",
            doc_id="doc-history",
            record_id="HIS-001",
            title="历史案例",
            content="结构化历史案例",
            metadata={
                "answer_fields": {
                    "occurred_at": "2026-08-01 09:00:00",
                    "root_cause": "接口松动",
                    "steps": ["检查接口"],
                    "final_solution": "重新锁紧",
                    "verification_result": "复测正常",
                }
            },
        )
        state = {
            "query": "该历史案例经哪些功能关联到哪个模块？",
            "intent": "history_query",
            "query_targets": [{"intent": "history_query"}],
            "query_frame": {
                "source_obligations": [{"tool": "fault_history_query"}],
            },
            "authorized_tools": [
                "fault_history_query", "hybrid_search", "graph_traversal", "evidence_synthesis"
            ],
            "allowed_tools": ["fault_history_query", "graph_traversal", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(
                    tool="fault_history_query",
                    status="ok",
                    data={"count": 1},
                    evidence_ids=[history_evidence["evidence_id"]],
                )
            ],
            "evidence_registry": [history_evidence],
            "graph_seed_bindings": [
                {
                    "logical_id": "sg:b:fault_history_case:HIS-001",
                    "entity_kind": "fault_history_case",
                    "source_tool": "fault_history_query",
                    "binding_status": "candidate",
                }
            ],
        }
        result = self._route(
            state,
            json.dumps(
                {
                    "decision": "graph",
                    "next_tools": ["graph_traversal"],
                    "reason": "用户明确询问跨文档路径",
                },
                ensure_ascii=False,
            ),
        )
        self.assertTrue(result["adaptive_graph_requested"])
        self.assertEqual(
            ["sg:b:fault_history_case:HIS-001"],
            result["graph_traversal_request"]["seed_ids"],
        )
        self.assertEqual("graph_traversal", route_after_adaptive_tools(result))

    def test_cause_intent_does_not_override_model_answer_with_graph_or_fta(self):
        evidence = _evidence()
        state = {
            "query": "系统运行中总是无故反复重启，通常是什么原因？",
            "intent": "cause_analysis",
            "query_targets": [{"intent": "cause_analysis"}],
            "authorized_tools": [
                "fmeca_query", "fta_query", "hybrid_search",
                "graph_traversal", "evidence_synthesis",
            ],
            "allowed_tools": ["fmeca_query", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(
                    tool="fmeca_query",
                    status="ok",
                    data={"count": 1},
                    evidence_ids=[evidence["evidence_id"]],
                )
            ],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [
                {
                    "logical_id": "sg:b:fmeca_record:FM-001",
                    "entity_kind": "fmeca_record",
                    "source_tool": "fmeca_query",
                    "binding_status": "candidate",
                }
            ],
        }
        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "已有原因记录"}),
        )
        self.assertFalse(result["adaptive_graph_requested"])
        self.assertEqual([], result["pending_tools"])
        self.assertEqual({}, result["graph_traversal_request"])
        self.assertEqual("", result["adaptive_decision"]["graph_trigger"])
        self.assertEqual("icp_calculation", route_after_adaptive_tools(result))

    def test_model_answer_is_not_overridden_by_an_intent_fault_bundle(self):
        evidence = _evidence()
        state = {
            "query": "滤波电容老化会有什么影响？",
            "intent": "impact_analysis",
            "query_targets": [{"intent": "impact_analysis"}],
            "authorized_tools": [
                "function_locator", "hmc_query", "fmeca_query", "mfl_query",
                "fault_history_query", "hybrid_search", "evidence_synthesis",
            ],
            "allowed_tools": ["fmeca_query", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(
                    tool="fmeca_query",
                    status="ok",
                    data={"count": 1},
                    evidence_ids=[evidence["evidence_id"]],
                )
            ],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [],
        }
        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "影响已命中"}),
        )
        self.assertEqual([], result["pending_tools"])
        self.assertEqual("done", result["next_action"])

    def test_planner_can_request_implicit_cross_document_impact_expansion(self):
        evidence = _evidence()
        state = {
            "query": "滤波电容老化会有什么影响？",
            "intent": "impact_analysis",
            "query_targets": [{"intent": "impact_analysis"}],
            "query_frame": {
                "required_answer_fields": ["final_impact"],
            },
            "relation_expansion": {
                "needed": True,
                "reason": "需要补全故障模式到模块和功能的跨文档影响链",
            },
            "authorized_tools": [
                "fmeca_query", "hybrid_search", "graph_traversal", "evidence_synthesis"
            ],
            "allowed_tools": ["fmeca_query", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(
                    tool="fmeca_query",
                    status="ok",
                    data={"count": 1},
                    evidence_ids=[evidence["evidence_id"]],
                )
            ],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [
                {
                    "logical_id": "sg:b:fmeca_record:FM-001",
                    "entity_kind": "fmeca_record",
                    "source_tool": "fmeca_query",
                    "binding_status": "candidate",
                }
            ],
        }
        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "直接影响已命中"}),
        )
        self.assertTrue(result["adaptive_graph_requested"])
        self.assertEqual("graph_traversal", route_after_adaptive_tools(result))
        self.assertIn("跨文档影响链", result["adaptive_decision"]["graph_trigger"])

    def test_intent_alone_does_not_defer_an_unrequested_graph_lookup(self):
        state = {
            "query": "系统运行中反复重启，可能是什么原因？",
            "intent": "cause_analysis",
            "query_targets": [{"intent": "cause_analysis"}],
            "authorized_tools": [
                "fmeca_query", "hybrid_search", "graph_traversal", "evidence_synthesis"
            ],
            "allowed_tools": ["fmeca_query", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(tool="fmeca_query", status="empty", data={"count": 0})
            ],
            "evidence_registry": [],
            "graph_seed_bindings": [],
        }
        result = self._route(
            state,
            json.dumps(
                {
                    "decision": "retrieve",
                    "next_tools": ["hybrid_search"],
                    "reason": "先补充可绑定候选",
                },
                ensure_ascii=False,
            ),
        )
        self.assertEqual(["hybrid_search"], result["pending_tools"])
        self.assertFalse(result["adaptive_graph_requested"])
        self.assertEqual({}, result["graph_traversal_request"])
        self.assertEqual("adaptive_retrieval", route_after_adaptive_tools(result))

    def test_keyword_hmc_candidate_cannot_anchor_unrelated_cause_graph(self):
        hmc_evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            doc_id="doc-hmc",
            record_id="91820001**0001",
            title="91820001**0001",
            content="BCM 启动时序超时",
            match_type="keyword",
        )
        state = {
            "query": "没有正确传出数据到外设",
            "intent": "cause_analysis",
            "query_targets": [{"intent": "cause_analysis"}],
            "query_frame": {
                "required_answer_fields": [
                    "fault_cause", "structural_cause_chain",
                ],
            },
            "relation_expansion": {
                "needed": True,
                "reason": "需要补全外部数据通信故障的跨文档原因链",
            },
            "authorized_tools": [
                "function_locator", "hmc_query", "fmeca_query", "fta_query",
                "hybrid_search", "graph_traversal", "evidence_synthesis",
            ],
            "allowed_tools": [
                "function_locator", "hmc_query", "hybrid_search", "evidence_synthesis"
            ],
            "tool_results": [
                make_tool_result(
                    tool="hmc_query",
                    status="ok",
                    data={"count": 1},
                    evidence_ids=[hmc_evidence["evidence_id"]],
                ),
                make_tool_result(tool="hybrid_search", status="ok", data={"count": 2}),
            ],
            "evidence_registry": [hmc_evidence],
            "graph_seed_bindings": [
                {
                    "logical_id": "sg:b:hmc_code:91820001**0001",
                    "entity_kind": "hmc_code",
                    "source_tool": "hmc_query",
                    "binding_status": "candidate",
                    "match_type": "keyword",
                }
            ],
        }
        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "已有候选"}),
        )
        self.assertEqual(["fmeca_query", "fta_query"], result["pending_tools"])
        self.assertTrue(result["adaptive_graph_requested"])
        self.assertEqual({}, result["graph_traversal_request"])
        self.assertEqual("adaptive_retrieval", route_after_adaptive_tools(result))

    def test_graph_is_not_offered_without_bound_business_entity(self):
        state = {
            "query": "这些原因如何关联？",
            "intent": "cause_analysis",
            "query_targets": [{"intent": "cause_analysis"}],
            "authorized_tools": ["fmeca_query", "graph_traversal", "evidence_synthesis"],
            "allowed_tools": ["fmeca_query", "graph_traversal", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(tool="fmeca_query", status="empty", data={"count": 0})
            ],
            "evidence_registry": [],
            "graph_seed_bindings": [],
        }
        # 即使模型输出图工具，策略也会删除当前不可执行的选项。
        result = self._route(
            state,
            json.dumps({"decision": "graph", "next_tools": ["graph_traversal"]}),
        )
        self.assertFalse(result["adaptive_graph_requested"])
        self.assertEqual({}, result["graph_traversal_request"])

    def test_invalid_adaptive_response_uses_single_safe_hybrid_fallback(self):
        state = {
            "query": "未知故障现象",
            "intent": "symptom_diagnosis",
            "query_targets": [{"intent": "symptom_diagnosis"}],
            "authorized_tools": ["fmeca_query", "hybrid_search", "evidence_synthesis"],
            "allowed_tools": ["fmeca_query", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(tool="fmeca_query", status="empty", data={"count": 0})
            ],
            "evidence_registry": [],
            "graph_seed_bindings": [],
        }
        result = self._route(state, "not-json")
        self.assertEqual(["hybrid_search"], result["pending_tools"])
        self.assertEqual("safe_fallback", result["adaptive_decision"]["source"])

    def test_followup_retrieval_merges_seeds_and_builds_deferred_graph_request(self):
        new_evidence = _evidence("FM-NEW")
        state = {
            "intent": "cause_analysis",
            "query_targets": [{"intent": "cause_analysis"}],
            "query_frame": {"required_answer_fields": ["fault_cause"]},
            "pending_tools": ["fmeca_query"],
            "allowed_tools": [
                "hmc_query", "fmeca_query", "graph_traversal", "evidence_synthesis"
            ],
            "authorized_doc_types": ["hmc", "fmeca"],
            "adaptive_graph_requested": True,
            "graph_seed_bindings": [
                {"logical_id": "sg:b:hmc_code:1", "binding_status": "exact"}
            ],
            "candidate_faults": [],
            "domain_results": [],
            "confirmed_graphs": [],
            "diagnostic_support": 0.4,
            "structured_entity_binding": {
                "bound": True,
                "ambiguous": False,
                "business_ids": ["MOD-NAV-A"],
                "terms": ["MOD-NAV-A", "NAVA"],
                "row_indexes": [0],
                "basis": "exact_business_id",
            },
        }
        batch_update = {
            "tool_calls": [{"tool": "fmeca_query", "count": 1}],
            "tool_results": [
                make_tool_result(
                    tool="fmeca_query",
                    status="ok",
                    data={"count": 1},
                    evidence_ids=[new_evidence["evidence_id"]],
                )
            ],
            "evidence_registry": [new_evidence],
            "graph_seed_bindings": [
                {
                    "logical_id": "sg:b:fmeca_record:2",
                    "entity_kind": "fmeca_record",
                    "source_tool": "fmeca_query",
                    "binding_status": "candidate",
                }
            ],
            "candidate_faults": [],
            "domain_results": [],
            "confirmed_graphs": [],
            "diagnostic_support": 0.7,
            "structured_entity_binding": {
                "bound": False,
                "ambiguous": False,
                "business_ids": [],
                "terms": [],
                "row_indexes": [],
                "basis": "",
            },
        }
        with patch.object(
            adaptive_module,
            "icp_diagnosis_node",
            new=AsyncMock(return_value=batch_update),
        ) as diagnosis_mock:
            result = _run(adaptive_retrieval_node(state))
        batch_state = diagnosis_mock.await_args.args[0]
        self.assertEqual(["fmeca_query"], batch_state["allowed_tools"][:1])
        self.assertEqual(
            state["allowed_tools"],
            batch_state["structured_hydration_tools"],
        )
        self.assertEqual(2, len(result["graph_seed_bindings"]))
        self.assertEqual(2, len(result["graph_traversal_request"]["seed_ids"]))
        self.assertEqual([], result["pending_tools"])
        self.assertEqual(
            ["MOD-NAV-A"],
            result["structured_entity_binding"]["business_ids"],
        )
        self.assertEqual("graph_traversal", route_after_adaptive_retrieval(result))

    def test_followup_retrieval_preserves_cross_family_identity_boundaries(self):
        """A batch which does not run HMC/MFL/RBD cannot reset their guards."""

        state = {
            "intent": "impact_analysis",
            "query_targets": [{"intent": "impact_analysis"}],
            "pending_tools": ["fmeca_query"],
            "allowed_tools": ["fmeca_query", "evidence_synthesis"],
            "authorized_doc_types": ["fmeca"],
            "adaptive_graph_requested": False,
            "graph_seed_bindings": [],
            "candidate_faults": [],
            "domain_results": [],
            "confirmed_graphs": [],
            "diagnostic_support": 0.2,
            "hmc_mfl_binding": {
                "required": True,
                "bound": False,
                "ambiguous": False,
                "basis": "no_explicit_authoritative_relation",
                "hmc_codes": ["91810001**0001"],
                "mfl_codes": [],
                "excluded_mfl_codes": ["MFL-NEIGHBOUR"],
            },
            "rbd_query_binding": {
                "required": True,
                "bound": False,
                "ambiguous": False,
                "basis": "single_unbound_candidate",
                "record_ids": ["RBD-ONE"],
                "candidate_record_ids": ["RBD-ONE"],
            },
        }
        batch_update = {
            "tool_calls": [],
            "tool_results": [],
            "evidence_registry": [],
            "graph_seed_bindings": [],
            "candidate_faults": [],
            "domain_results": [],
            "confirmed_graphs": [],
            "diagnostic_support": 0.3,
            "hmc_mfl_binding": {
                "required": False,
                "bound": False,
                "basis": "",
            },
            "rbd_query_binding": {
                "required": False,
                "bound": False,
                "basis": "",
            },
        }

        with patch.object(
            adaptive_module,
            "icp_diagnosis_node",
            new=AsyncMock(return_value=batch_update),
        ):
            result = _run(adaptive_retrieval_node(state))

        self.assertTrue(result["hmc_mfl_binding"]["required"])
        self.assertFalse(result["hmc_mfl_binding"]["bound"])
        self.assertEqual(
            ["MFL-NEIGHBOUR"],
            result["hmc_mfl_binding"]["excluded_mfl_codes"],
        )
        self.assertTrue(result["rbd_query_binding"]["required"])
        self.assertFalse(result["rbd_query_binding"]["bound"])
        self.assertEqual(
            ["RBD-ONE"],
            result["rbd_query_binding"]["candidate_record_ids"],
        )

    def test_hmc_mfl_merge_rejects_an_unproven_bound_flag(self):
        result = adaptive_module._merge_hmc_mfl_binding(
            {
                "required": True,
                "bound": True,
                "basis": "no_explicit_authoritative_relation",
                "hmc_codes": ["91810001**0001"],
                "mfl_codes": ["MFL-SIMILAR"],
            },
            {},
        )

        self.assertFalse(result["bound"])
        self.assertEqual([], result["mfl_codes"])
        self.assertEqual(["MFL-SIMILAR"], result["excluded_mfl_codes"])

    def test_hmc_mfl_merge_fails_closed_when_target_code_changes(self):
        def bound(hmc_code, mfl_code):
            return {
                "required": True,
                "bound": True,
                "ambiguous": False,
                "basis": "explicit_authoritative_reference",
                "hmc_codes": [hmc_code],
                "mfl_codes": [mfl_code],
            }

        result = adaptive_module._merge_hmc_mfl_binding(
            bound("91810001**0001", "MFL-A"),
            bound("91810002**0001", "MFL-B"),
        )

        self.assertFalse(result["bound"])
        self.assertTrue(result["ambiguous"])
        self.assertEqual([], result["mfl_codes"])
        self.assertEqual({"MFL-A", "MFL-B"}, set(result["excluded_mfl_codes"]))

    def test_rbd_merge_rejects_untyped_multi_model_bound_flag(self):
        result = adaptive_module._merge_rbd_query_binding(
            {
                "required": True,
                "bound": True,
                "ambiguous": False,
                "basis": "unresolved",
                "record_ids": ["RBD-A", "RBD-B"],
                "candidate_record_ids": ["RBD-A", "RBD-B"],
            },
            {},
        )

        self.assertFalse(result["bound"])
        self.assertTrue(result["ambiguous"])
        self.assertEqual([], result["record_ids"])
        self.assertEqual(
            {"RBD-A", "RBD-B"}, set(result["candidate_record_ids"])
        )

    def test_rbd_merge_rejects_legacy_bound_basis_and_conflicting_candidate(self):
        result = adaptive_module._merge_rbd_query_binding(
            {
                "required": True,
                "bound": True,
                "ambiguous": False,
                "basis": "stable_rbd_identity",
                "record_ids": ["RBD-A"],
                "candidate_record_ids": ["RBD-A"],
            },
            {
                "required": True,
                "bound": False,
                "ambiguous": False,
                "basis": "single_unbound_candidate",
                "record_ids": ["RBD-B"],
                "candidate_record_ids": ["RBD-B"],
            },
        )

        self.assertFalse(result["bound"])
        self.assertTrue(result["ambiguous"])
        self.assertEqual([], result["record_ids"])

    def test_binding_merge_keeps_stronger_identity_and_audits_conflict(self):
        previous = {
            "bound": True,
            "ambiguous": False,
            "business_ids": ["MODULE-PRIMARY"],
            "terms": ["primary"],
            "row_indexes": [0],
            "basis": "exact_business_id",
        }
        current = {
            "bound": True,
            "ambiguous": False,
            "business_ids": ["MODULE-NEIGHBOUR"],
            "terms": ["neighbour"],
            "row_indexes": [1],
            "basis": "identity_label_containment",
        }

        result = adaptive_module._merge_structured_entity_binding(previous, current)

        self.assertTrue(result["bound"])
        self.assertEqual(["MODULE-PRIMARY"], result["business_ids"])
        self.assertEqual(
            "kept_stronger_previous",
            result["binding_conflicts"][-1]["resolution"],
        )
        self.assertEqual(
            ["MODULE-NEIGHBOUR"],
            result["binding_conflicts"][-1]["current"]["business_ids"],
        )

    def test_binding_merge_allows_stronger_current_identity_with_audit(self):
        previous = {
            "bound": True,
            "ambiguous": False,
            "business_ids": ["MODULE-CANDIDATE"],
            "terms": ["candidate"],
            "row_indexes": [0],
            "basis": "specific_identity_overlap",
        }
        current = {
            "bound": True,
            "ambiguous": False,
            "business_ids": ["MODULE-EXACT"],
            "terms": ["exact"],
            "row_indexes": [2],
            "basis": "exact_business_id",
        }

        result = adaptive_module._merge_structured_entity_binding(previous, current)

        self.assertTrue(result["bound"])
        self.assertEqual(["MODULE-EXACT"], result["business_ids"])
        self.assertEqual(
            "kept_stronger_current",
            result["binding_conflicts"][-1]["resolution"],
        )

    def test_binding_merge_fails_closed_on_equal_strength_conflict(self):
        def binding(business_id):
            return {
                "bound": True,
                "ambiguous": False,
                "business_ids": [business_id],
                "terms": [business_id],
                "row_indexes": [0],
                "basis": "exact_identity_label",
            }

        result = adaptive_module._merge_structured_entity_binding(
            binding("MODULE-ONE"), binding("MODULE-TWO")
        )

        self.assertFalse(result["bound"])
        self.assertTrue(result["ambiguous"])
        self.assertEqual([], result["business_ids"])
        self.assertEqual(
            "ambiguous_equal_or_unknown_authority",
            result["binding_conflicts"][-1]["resolution"],
        )

    def test_binding_merge_combines_compatible_bindings(self):
        previous = {
            "bound": True,
            "ambiguous": False,
            "business_ids": ["MODULE-SHARED", "FUNCTION-ONE"],
            "terms": ["module"],
            "row_indexes": [0],
            "basis": "exact_identity_label",
        }
        current = {
            "bound": True,
            "ambiguous": False,
            "business_ids": ["module-shared", "INTERFACE-ONE"],
            "terms": ["interface"],
            "row_indexes": [1],
            "basis": "same_row_module_function_chain",
        }

        result = adaptive_module._merge_structured_entity_binding(previous, current)

        self.assertTrue(result["bound"])
        self.assertFalse(result["ambiguous"])
        self.assertEqual(
            ["MODULE-SHARED", "FUNCTION-ONE", "INTERFACE-ONE"],
            result["business_ids"],
        )
        self.assertEqual("same_row_module_function_chain", result["basis"])

    def test_api_only_mfl_evidence_does_not_satisfy_repair_slot(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="maintenance_locator",
            doc_type="mfl",
            doc_id="doc-mfl",
            record_id="MFL-API-1",
            title="接口定位",
            content="只包含 API 定位信息",
            metadata={
                "answer_fields": {
                    "api_name": "read_status",
                    "api_method": "读取状态字",
                }
            },
        )
        state = {
            "query": "这个故障怎么处理？",
            "intent": "maintenance_query",
            "query_targets": [{
                "intent": "maintenance_query",
                "required_fields": ["maintenance_action"],
            }],
            "query_frame": {
                "required_answer_fields": ["maintenance_action"],
            },
            "authorized_tools": ["hmc_query", "mfl_query", "evidence_synthesis"],
            "allowed_tools": ["mfl_query", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(
                    tool="mfl_query",
                    status="ok",
                    data={"count": 1},
                    evidence_ids=[evidence["evidence_id"]],
                )
            ],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [],
        }
        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "已有 MFL"}),
        )
        self.assertEqual(["hmc_query"], result["pending_tools"])
        self.assertIn("mfl_api", adaptive_module._coverage_slots(state))
        self.assertNotIn("mfl_repair", adaptive_module._coverage_slots(state))

    def test_structured_mfl_repair_field_satisfies_repair_slot(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="maintenance_locator",
            doc_type="mfl",
            doc_id="doc-mfl",
            record_id="MFL-REPAIR-1",
            title="维修定位",
            content="结构化维修记录",
            metadata={
                "answer_fields": {
                    "maintenance_action": "检查连接器并更换失效触点",
                }
            },
        )
        state = {
            "query": "这个故障怎么处理？",
            "intent": "maintenance_query",
            "query_targets": [{
                "intent": "maintenance_query",
                "required_fields": ["maintenance_action"],
            }],
            "query_frame": {
                "required_answer_fields": ["maintenance_action"],
            },
            "authorized_tools": ["hmc_query", "mfl_query", "evidence_synthesis"],
            "allowed_tools": ["mfl_query", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(
                    tool="mfl_query",
                    status="ok",
                    data={"count": 1},
                    evidence_ids=[evidence["evidence_id"]],
                )
            ],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [],
        }
        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "维修字段完整"}),
        )
        self.assertEqual([], result["pending_tools"])
        self.assertIn("mfl_repair", adaptive_module._coverage_slots(state))

    def test_maintenance_repair_does_not_expand_to_unrequested_cause_and_impact(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="maintenance_locator",
            doc_type="mfl",
            doc_id="doc-mfl",
            record_id="MFL-REPAIR-PARTIAL",
            title="维修定位",
            content="结构化维修记录",
            metadata={
                "answer_fields": {
                    "maintenance_action": "检查连接器并更换失效触点",
                }
            },
        )
        state = {
            "query": "这个故障怎么处理？",
            "intent": "maintenance_query",
            "query_targets": [{
                "intent": "maintenance_query",
                "required_fields": ["maintenance_action"],
            }],
            "query_frame": {
                "required_answer_fields": ["maintenance_action"],
            },
            "authorized_tools": ["fmeca_query", "mfl_query", "evidence_synthesis"],
            "allowed_tools": ["mfl_query", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(
                    tool="mfl_query",
                    status="ok",
                    data={"count": 1},
                    evidence_ids=[evidence["evidence_id"]],
                )
            ],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [],
        }

        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "已有维修动作"}),
        )

        self.assertEqual([], result["pending_tools"])
        self.assertIn("mfl_repair", adaptive_module._coverage_slots(state))
        self.assertNotIn("cause", adaptive_module._coverage_slots(state))
        self.assertNotIn("impact", adaptive_module._coverage_slots(state))

    def test_impact_contract_does_not_add_an_unrequested_cause_lookup(self):
        evidence = _evidence("FM-IMPACT-PARTIAL")
        evidence.update(
            {
                "match_type": "semantic_id_hydration",
                "retrieval_score": 0.91,
            }
        )
        state = {
            "query": "这个退化之后会带来什么后果？",
            "intent": "impact_analysis",
            "query_targets": [{
                "intent": "impact_analysis",
                "required_fields": ["final_impact"],
            }],
            "query_frame": {"required_answer_fields": ["final_impact"]},
            "authorized_tools": ["fmeca_query", "hybrid_search", "evidence_synthesis"],
            "allowed_tools": ["hybrid_search", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(
                    tool="hybrid_search",
                    status="ok",
                    data={"count": 1},
                    evidence_ids=[evidence["evidence_id"]],
                )
            ],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [],
        }

        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "已有影响"}),
        )

        self.assertEqual([], result["pending_tools"])
        self.assertIn("impact", adaptive_module._coverage_slots(state))
        self.assertNotIn("cause", adaptive_module._coverage_slots(state))

    def test_fta_cause_answer_does_not_add_an_unrequested_impact_lookup(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fault_cause",
            doc_type="fta",
            doc_id="doc-fta",
            record_id="FTA-CAUSE-ONLY",
            title="FTA 原因链",
            content="结构化 FTA 原因链",
            metadata={"answer_fields": {"fault_cause": "电源瞬断"}},
        )
        state = {
            "query": "设备运行中总是自行复位，是什么原因？",
            "intent": "cause_analysis",
            "query_targets": [{
                "intent": "cause_analysis",
                "required_fields": ["fault_cause"],
            }],
            "query_frame": {"required_answer_fields": ["fault_cause"]},
            "authorized_tools": [
                "fmeca_query", "fta_query", "mfl_query", "hmc_query",
                "fault_history_query", "evidence_synthesis",
            ],
            "allowed_tools": ["fta_query", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(
                    tool="fta_query",
                    status="ok",
                    data={"count": 1},
                    evidence_ids=[evidence["evidence_id"]],
                )
            ],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [],
        }

        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "已有原因链"}),
        )

        self.assertEqual([], result["pending_tools"])
        self.assertIn("cause", adaptive_module._coverage_slots(state))
        self.assertNotIn("impact", adaptive_module._coverage_slots(state))

    def test_strong_semantic_fmeca_still_gets_one_structured_fault_lookup(self):
        evidence = _evidence("FM-SEMANTIC-COMPLETE")
        evidence.update(
            {
                "match_type": "semantic_id_hydration",
                "retrieval_score": 0.95,
                "metadata": {
                    "answer_fields": {
                        "fault_cause": "输入开路",
                        "final_impact": "状态量不可用",
                    }
                },
            }
        )
        state = {
            "query": "这个输入异常怎么区分和处理？",
            "intent": "maintenance_query",
            "query_targets": [{
                "intent": "maintenance_query",
                "required_fields": ["detection_method"],
            }],
            "query_frame": {"required_answer_fields": ["detection_method"]},
            "authorized_tools": ["fmeca_query", "hybrid_search", "evidence_synthesis"],
            "allowed_tools": ["hybrid_search", "evidence_synthesis"],
            "tool_results": [
                make_tool_result(
                    tool="hybrid_search",
                    status="ok",
                    data={"count": 1},
                    evidence_ids=[evidence["evidence_id"]],
                )
            ],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [],
        }

        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "语义证据已有"}),
        )

        self.assertEqual(["fmeca_query"], result["pending_tools"])
        self.assertTrue({"cause", "impact"}.issubset(adaptive_module._coverage_slots(state)))

    def test_hmc_repair_claim_and_fmeca_detect_field_are_separate_slots(self):
        hmc_repair = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="maintenance_action",
            doc_type="hmc",
            doc_id="doc-hmc",
            record_id="91810001**0001",
            title="HMC 维修方法",
            content="更换输入调理模块",
        )
        fmeca_detect = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-DETECT",
            title="检测记录",
            content="结构化检测记录",
            metadata={"answer_fields": ["detect_method"]},
        )
        coverage = adaptive_module._coverage_slots(
            {"evidence_registry": [hmc_repair, fmeca_detect]}
        )
        self.assertIn("hmc_repair", coverage)
        self.assertIn("detect", coverage)
        self.assertNotIn("mfl_repair", coverage)

    def test_fmeca_row_with_only_cause_does_not_satisfy_impact(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-CAUSE-ONLY",
            title="原因记录",
            content="结构化 FMECA 记录",
            metadata={"answer_fields": {"fault_cause": "连接器松动"}},
        )
        state = {
            "query": "之后会有什么后果？",
            "intent": "impact_analysis",
            "query_targets": [{
                "intent": "impact_analysis",
                "required_fields": ["local_impact", "final_impact"],
            }],
            "query_frame": {
                "required_answer_fields": ["local_impact", "final_impact"],
            },
            "authorized_tools": ["fmeca_query", "evidence_synthesis"],
            "allowed_tools": ["evidence_synthesis"],
            "tool_results": [],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [],
        }
        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "已有 FMECA"}),
        )
        self.assertEqual(["fmeca_query"], result["pending_tools"])
        self.assertIn("cause", adaptive_module._coverage_slots(state))
        self.assertNotIn("impact", adaptive_module._coverage_slots(state))

    def test_weak_semantic_hydration_does_not_prematurely_complete_slots(self):
        evidence = _evidence("FM-WEAK")
        evidence.update(
            {
                "match_type": "semantic_id_hydration",
                "retrieval_score": 0.08,
            }
        )
        state = {
            "query": "内部报文没有送到外接设备，哪里可能出错？",
            "intent": "cause_analysis",
            "query_targets": [{
                "intent": "cause_analysis",
                "required_fields": ["fault_cause", "structural_cause_chain"],
            }],
            "query_frame": {
                "required_answer_fields": [
                    "fault_cause", "structural_cause_chain",
                ],
            },
            "evidence_registry": [evidence],
        }

        self.assertNotIn("impact", adaptive_module._coverage_slots(state))
        self.assertEqual(
            ["fmeca_query", "fta_query"],
            adaptive_module._completion_followups(
                state,
                executed=["hybrid_search", "mfl_query", "hmc_query", "fault_history_query"],
                available=["fmeca_query", "fta_query"],
            ),
        )

    def test_strong_semantic_hydration_can_complete_declared_slot(self):
        evidence = _evidence("FM-STRONG")
        evidence.update(
            {
                "match_type": "semantic_id_hydration",
                "retrieval_score": 0.91,
            }
        )

        self.assertIn(
            "impact",
            adaptive_module._coverage_slots({"evidence_registry": [evidence]}),
        )

    def test_partial_history_fields_do_not_mark_history_complete(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="historical_case",
            doc_type="fault_history",
            doc_id="doc-history",
            record_id="HIS-PARTIAL",
            title="部分历史记录",
            content="结构化历史记录",
            metadata={
                "original": {
                    "occurred_at": "2026-08-01 09:00:00",
                    "root_cause": "接触电阻增大",
                }
            },
        )
        state = {
            "query": "以前怎么处理的？",
            "intent": "history_query",
            "query_targets": [{
                "intent": "history_query",
                "required_fields": list(adaptive_module._HISTORY_SLOTS),
            }],
            "query_frame": {
                "required_answer_fields": list(adaptive_module._HISTORY_SLOTS),
            },
            "authorized_tools": ["fault_history_query", "evidence_synthesis"],
            "allowed_tools": ["evidence_synthesis"],
            "tool_results": [],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [],
        }
        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "已有历史"}),
        )
        self.assertEqual(["fault_history_query"], result["pending_tools"])
        self.assertEqual(
            {"cause", "history_time", "history_root"},
            adaptive_module._coverage_slots(state),
        )

    def test_complete_history_fields_satisfy_history_target(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="historical_case",
            doc_type="fault_history",
            doc_id="doc-history",
            record_id="HIS-COMPLETE",
            title="完整历史记录",
            content="结构化历史记录",
            metadata={
                "raw_data": {
                    "occurred_at": "2026-08-01 09:00:00",
                    "root_cause": "接触电阻增大",
                    "steps": [{"step_no": 1, "content": "测量触点阻抗"}],
                    "final_solution": "更换触点",
                    "verification_result": "连续运行四小时未复现",
                }
            },
        )
        state = {
            "query": "以前怎么处理的？",
            "intent": "history_query",
            "query_targets": [{
                "intent": "history_query",
                "required_fields": list(adaptive_module._HISTORY_SLOTS),
            }],
            "query_frame": {
                "required_answer_fields": list(adaptive_module._HISTORY_SLOTS),
            },
            "authorized_tools": ["fault_history_query", "evidence_synthesis"],
            "allowed_tools": ["evidence_synthesis"],
            "tool_results": [],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [],
        }
        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "历史字段完整"}),
        )
        self.assertEqual([], result["pending_tools"])
        self.assertTrue(
            set(adaptive_module._HISTORY_SLOTS).issubset(
                adaptive_module._coverage_slots(state)
            )
        )

    def test_architecture_fact_without_path_requires_interface_lookup(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="function_fact",
            doc_type="function_locator",
            doc_id="doc-function",
            record_id="FUNC-1",
            title="接口处理功能",
            content="功能事实",
        )
        state = {
            "query": "外部数据经过哪些模块？",
            "intent": "architecture_query",
            "query_targets": [{
                "intent": "architecture_query",
                "required_fields": ["architecture_interface"],
            }],
            "query_frame": {
                "required_answer_fields": ["architecture_interface"],
            },
            "authorized_tools": ["function_locator", "evidence_synthesis"],
            "allowed_tools": ["evidence_synthesis"],
            "tool_results": [],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [],
        }
        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "已有功能"}),
        )
        self.assertEqual(["function_locator"], result["pending_tools"])

        interface = dict(evidence)
        interface["evidence_id"] = "ev-interface"
        interface["claim_type"] = "interface_fact"
        interface_state = {**state, "evidence_registry": [interface]}
        interface_result = self._route(
            interface_state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "已有接口路径"}),
        )
        self.assertEqual([], interface_result["pending_tools"])

    def test_rbd_claim_without_typed_field_does_not_satisfy_structure_slot(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="validated_relation",
            evidence_level="L2_CORROBORATED",
            claim_type="rbd_structure",
            doc_type="rbd",
            doc_id="doc-rbd",
            record_id="RBD-1",
            title="任务路径",
            content="{}",
        )
        state = {
            "query": "一条支路失效后还能完成任务吗？",
            "intent": "reliability_query",
            "query_targets": [{
                "intent": "reliability_query",
                "required_fields": ["rbd_path_structure"],
            }],
            "query_frame": {"required_answer_fields": ["rbd_path_structure"]},
            "authorized_tools": ["rbd_query", "evidence_synthesis"],
            "allowed_tools": ["evidence_synthesis"],
            "tool_results": [],
            "evidence_registry": [evidence],
            "graph_seed_bindings": [],
        }
        result = self._route(
            state,
            json.dumps({"decision": "answer", "next_tools": [], "reason": "RBD 已覆盖"}),
        )
        self.assertEqual(["rbd_query"], result["pending_tools"])

    def test_explicit_target_slots_precede_secondary_sources_within_budget(self):
        state = {
            "intent": "cause_analysis",
            "query_targets": [
                {"intent": "cause_analysis", "required_fields": ["fault_cause"]},
                {"intent": "maintenance_query", "required_fields": ["maintenance_action"]},
                {"intent": "history_query", "required_fields": ["history_result"]},
                {"intent": "architecture_query", "required_fields": ["architecture_interface"]},
            ],
            "evidence_registry": [],
        }
        tools = adaptive_module._completion_followups(
            state,
            executed=[],
            available=[
                "function_locator", "hmc_query", "fmeca_query", "fta_query",
                "mfl_query", "fault_history_query", "hybrid_search",
            ],
        )
        self.assertEqual(
            [
                "fmeca_query", "hmc_query", "fault_history_query",
                "function_locator", "fta_query",
            ],
            tools,
        )
        self.assertEqual(adaptive_module.MAX_FOLLOWUP_QUERY_TOOLS, len(tools))

    def test_focused_severity_with_structured_value_does_not_expand_to_fault_bundle(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-SEVERITY",
            title="输入滤波性能退化",
            content="结构化 FMECA 记录",
            metadata={"answer_fields": {"severity": "III"}},
        )
        state = {
            "intent": "impact_analysis",
            "query_targets": [
                {"intent": "impact_analysis", "required_fields": ["severity"]}
            ],
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["severity"],
            },
            "evidence_registry": [evidence],
        }

        tools = adaptive_module._completion_followups(
            state,
            executed=["fmeca_query"],
            available=[
                "hmc_query",
                "fmeca_query",
                "mfl_query",
                "fault_history_query",
                "hybrid_search",
            ],
        )

        self.assertEqual([], tools)
        self.assertIn("severity", adaptive_module._coverage_slots(state))

    def test_focused_severity_missing_uses_only_fmeca_and_hybrid(self):
        state = {
            "intent": "impact_analysis",
            "query_targets": [
                {"intent": "impact_analysis", "required_fields": ["severity"]}
            ],
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["severity"],
            },
            "evidence_registry": [],
        }

        tools = adaptive_module._completion_followups(
            state,
            executed=[],
            available=[
                "hmc_query",
                "fmeca_query",
                "fta_query",
                "mfl_query",
                "fault_history_query",
                "hybrid_search",
            ],
        )

        self.assertEqual(["fmeca_query", "hybrid_search"], tools)
        self.assertNotIn("hmc_query", tools)
        self.assertNotIn("mfl_query", tools)
        self.assertNotIn("fault_history_query", tools)

    def test_closed_failure_mode_collection_does_not_expand_to_fault_bundle(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-CLOSED-001",
            title="输入滤波性能退化",
            content="结构化 FMECA 集合成员",
            metadata={
                "answer_fields": {
                    "failure_mode": "输入滤波性能退化",
                    "collection_closed": True,
                    "collection_parent_id": "MOD-PWR-ICU",
                    "collection_member_kind": "failure_mode",
                    "collection_member_count": 1,
                    "collection_member_id": "FM-CLOSED-001",
                    "collection_member_name": "输入滤波性能退化",
                    "collection_role": "member",
                }
            },
        )
        state = {
            "intent": "cause_analysis",
            "query_targets": [{"intent": "cause_analysis"}],
            "query_frame": {
                "coverage_scope": "complete_collection",
                "collection_subject": "MOD-PWR-ICU",
                "collection_member_kind": "failure_mode",
                "required_answer_fields": ["closed_collection_members"],
            },
            "evidence_registry": [evidence],
        }

        tools = adaptive_module._completion_followups(
            state,
            executed=["function_locator", "fmeca_query"],
            available=[
                "hmc_query",
                "fmeca_query",
                "mfl_query",
                "fault_history_query",
                "hybrid_search",
            ],
        )

        self.assertEqual([], tools)
        self.assertIn(
            "closed_collection_members", adaptive_module._coverage_slots(state)
        )

    def test_detection_method_does_not_satisfy_maintenance_action(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-DETECT-ONLY",
            title="检测记录",
            content="仅包含检测方法",
            metadata={"answer_fields": {"detection_method": "读取状态位"}},
        )
        state = {
            "intent": "maintenance_query",
            "query_targets": [
                {
                    "intent": "maintenance_query",
                    "required_fields": ["maintenance_action"],
                }
            ],
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["maintenance_action"],
            },
            "evidence_registry": [evidence],
        }

        coverage = adaptive_module._coverage_slots(state)
        tools = adaptive_module._completion_followups(
            state,
            executed=["fmeca_query"],
            available=["hmc_query", "mfl_query", "hybrid_search"],
        )

        self.assertNotIn("maintenance_action", coverage)
        self.assertEqual(["hmc_query", "mfl_query", "hybrid_search"], tools)

    def test_calculator_name_alone_cannot_bypass_raw_query_authorization(self):
        state = {
            "query": "说明 FTA 的最小割集概念",
            "allowed_tools": ["fta_analyzer"],
            "calculation_requests": [],
            "confirmed_graphs": [],
        }
        with patch.object(
            calculation_module,
            "calculate_fta",
            side_effect=AssertionError("未授权的计算器不应执行"),
        ):
            result = _run(calculation_module.icp_calculation_node(state))
        self.assertEqual([], result["tool_calls"])
        self.assertEqual([], result["calculation_results"])


if __name__ == "__main__":
    unittest.main()
