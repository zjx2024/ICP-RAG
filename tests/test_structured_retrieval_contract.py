"""Metamorphic tests for model-owned retrieval and completion contracts."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent.answer_contract import ensure_intent_answer_fields
from backend.agent.contracts import make_evidence, make_tool_result
from backend.agent.nodes import adaptive_tools as adaptive_module
from backend.agent.nodes import plan as plan_module
from backend.agent.tool_policy import QUERY_TOOL_NAMES, resolve_tool_policy


def _run(coroutine):
    return asyncio.run(coroutine)


def _target(query: str, **updates):
    value = {
        "primary_intent": "cause_analysis",
        "targets": ["cause_analysis"],
        "contextualized_query": query,
        "subjects": [],
        "requested_facets": [],
        "required_answer_fields": ["fault_cause"],
        "question_semantics": "explain",
        "response_focus": ["cause"],
        "response_scope": "standalone",
        "response_scope_basis": query,
        "coverage_scope": "focused_detail",
        "collection_subject": "",
        "collection_member_kind": "unknown",
        "context_used": False,
        "context_status": "standalone",
        "selected_anchor_ids": [],
        "topology_dependency": {"required": False},
    }
    value.update(updates)
    return value


class StructuredRetrievalContractTests(unittest.TestCase):
    def test_plan_transmits_grounded_closure_and_response_mode_to_query_frame(self):
        query = "设备反复复位，请闭合现场处置与恢复结果"
        target = _target(
            query,
            diagnostic_closure={
                "required": True,
                "required_dimensions": ["maintenance_action", "history_result"],
                "basis": "反复复位，请闭合现场处置与恢复结果",
            },
            response_mode={"mode": "diagnostic", "basis": "请闭合现场处置与恢复结果"},
        )
        planner = {
            "intent": "cause_analysis",
            "query_targets": [{
                "intent": "cause_analysis", "required_fields": ["fault_cause"],
            }],
            "rewritten_query": query,
            "entity_keywords": ["设备反复复位"],
            "type_keywords": [],
            "query_frame": {
                "entities": ["设备反复复位"],
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["fault_cause"],
            },
            "tool_queries": {"fmeca_query": [query]},
            "tools_plan": [{"tool": "fmeca_query", "reason": "读取直接原因"}],
            "reasoning": "按显式字段读取权威记录",
        }
        with (
            patch.object(
                plan_module.llm_service,
                "_call_model",
                side_effect=[
                    json.dumps(target, ensure_ascii=False),
                    json.dumps(planner, ensure_ascii=False),
                ],
            ),
            patch.object(
                plan_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(plan_module.plan_node({
                "query": query,
                "messages": [{"role": "user", "content": query}],
                "memory_enabled": False,
                "iteration": 0,
            }))

        self.assertEqual(
            target["diagnostic_closure"],
            {key: value for key, value in result["query_frame"]["diagnostic_closure"].items()
             if key != "reason"},
        )
        self.assertEqual(target["response_mode"], result["query_frame"]["response_mode"])
        self.assertEqual("fault_diagnosis", result["answer_mode"])

    def test_planner_prompt_does_not_delegate_answer_shape_to_intent(self):
        self.assertNotIn("最终由系统根据 intent 确定性映射", plan_module.PLAN_PROMPT)
        self.assertIn(
            "只把已解析任务转换为查询，不重新解释用户目标",
            plan_module.PLAN_PROMPT,
        )

    def test_legacy_answer_mode_is_derived_from_structured_contract(self):
        focused_cause = {
            "required_answer_fields": ["fault_cause"],
            "response_mode": {"mode": "focused", "basis": "只说原因"},
        }
        diagnostic_cause = {
            **focused_cause,
            "response_mode": {"mode": "diagnostic", "basis": "形成诊断闭环"},
        }
        maintenance = {
            "required_answer_fields": ["maintenance_action"],
            "response_mode": {"mode": "focused", "basis": "只说处置"},
        }

        self.assertEqual(
            "knowledge_explanation",
            plan_module._answer_mode_from_contract(focused_cause),
        )
        self.assertEqual(
            "fault_diagnosis",
            plan_module._answer_mode_from_contract(diagnostic_cause),
        )
        self.assertEqual(
            "maintenance_guidance",
            plan_module._answer_mode_from_contract(maintenance),
        )

    def test_same_model_contract_has_same_policy_across_paraphrases(self):
        cases = (
            ("运行阶段反复复位，请形成现场闭环", "反复复位，请形成现场闭环"),
            ("设备多次自行重启，请把现场证据闭合", "多次自行重启，请把现场证据闭合"),
        )
        decisions = []
        for query, basis in cases:
            decisions.append(resolve_tool_policy(
                intent="cause_analysis",
                raw_user_query=query,
                query_targets=[{
                    "intent": "cause_analysis",
                    "required_fields": ["fault_cause"],
                }],
                required_answer_fields=["fault_cause"],
                diagnostic_closure={
                    "required": True,
                    "required_dimensions": [
                        "maintenance_action", "history_result",
                    ],
                    "basis": basis,
                },
                proposed_plan=[{"tool": "fmeca_query"}],
            ))

        self.assertEqual(
            decisions[0]["authorized_tools"], decisions[1]["authorized_tools"]
        )
        self.assertEqual(decisions[0]["allowed_tools"], decisions[1]["allowed_tools"])
        self.assertEqual(
            decisions[0]["query_targets"][0]["required_fields"],
            decisions[1]["query_targets"][0]["required_fields"],
        )

    def test_same_question_with_different_field_contracts_stays_strictly_different(self):
        query = "请补充这条记录当前缺少的信息"
        cause = resolve_tool_policy(
            intent="cause_analysis",
            raw_user_query=query,
            query_targets=[{
                "intent": "cause_analysis", "required_fields": ["fault_cause"],
            }],
            required_answer_fields=["fault_cause"],
            coverage_scope="focused_detail",
        )
        history = resolve_tool_policy(
            intent="cause_analysis",
            raw_user_query=query,
            query_targets=[{
                "intent": "cause_analysis", "required_fields": ["history_result"],
            }],
            required_answer_fields=["history_result"],
            coverage_scope="focused_detail",
        )

        self.assertIn("fmeca_query", cause["allowed_tools"])
        self.assertNotIn("fault_history_query", cause["authorized_tools"])
        self.assertIn("fault_history_query", history["allowed_tools"])
        self.assertNotIn("fmeca_query", history["authorized_tools"])

    def test_reliability_intent_without_fields_does_not_authorize_rbd(self):
        self.assertEqual(
            [], ensure_intent_answer_fields([], intents=["reliability_query"])
        )
        decision = resolve_tool_policy(
            intent="reliability_query",
            query_targets=[{"intent": "reliability_query"}],
            proposed_plan=[],
        )
        self.assertEqual(
            ["hybrid_search", "evidence_synthesis"], decision["allowed_tools"]
        )
        self.assertNotIn("rbd_query", decision["authorized_tools"])

    def test_response_mode_requires_closed_enum_and_current_turn_basis(self):
        query = "只说明这棵树的节点和逻辑关系"
        valid = plan_module._parse_target_response(
            json.dumps(_target(
                query,
                response_mode={"mode": "structure_only", "basis": "节点和逻辑关系"},
            ), ensure_ascii=False),
            raw_user_query=query,
        )
        invalid_mode = plan_module._parse_target_response(
            json.dumps(_target(
                query,
                response_mode={"mode": "verbose_report", "basis": "节点和逻辑关系"},
            ), ensure_ascii=False),
            raw_user_query=query,
        )
        invented_basis = plan_module._parse_target_response(
            json.dumps(_target(
                query,
                response_mode={"mode": "structure_only", "basis": "完整诊断报告"},
            ), ensure_ascii=False),
            raw_user_query=query,
        )

        self.assertEqual(
            {"mode": "structure_only", "basis": "节点和逻辑关系"},
            valid["response_mode"],
        )
        self.assertNotIn("response_mode", invalid_mode)
        self.assertNotIn("response_mode", invented_basis)

    def test_response_scope_requires_explicit_enum_and_current_turn_basis(self):
        cases = (
            ("只补充这次新问的验证结果", "只补充这次新问的", "incremental"),
            ("请从头完整重述这个结论", "从头完整重述", "full_restatement"),
        )
        for query, basis, scope in cases:
            with self.subTest(query=query):
                parsed = plan_module._parse_target_response(
                    json.dumps(_target(
                        query,
                        context_used=True,
                        response_scope=scope,
                        response_scope_basis=basis,
                    ), ensure_ascii=False),
                    raw_user_query=query,
                )
                self.assertEqual(scope, parsed["response_scope"])
                self.assertEqual(basis, parsed["response_scope_basis"])

        query = "那它的验证结果呢"
        for updates in (
            {"context_used": True, "response_scope": "incremental",
             "response_scope_basis": ""},
            {"context_used": True, "response_scope": "incremental",
             "response_scope_basis": "只补充本轮新维度"},
            {"context_used": True, "response_scope": "invented",
             "response_scope_basis": query},
        ):
            with self.subTest(updates=updates):
                parsed = plan_module._parse_target_response(
                    json.dumps(_target(query, **updates), ensure_ascii=False),
                    raw_user_query=query,
                )
                self.assertEqual("unknown", parsed["response_scope"])
                self.assertNotIn("response_scope_basis", parsed)

    def test_relation_expansion_requires_kind_basis_and_valid_selected_anchor(self):
        query = "它如何连接到下游模块"
        valid = plan_module._parse_target_response(
            json.dumps(_target(
                query,
                context_used=True,
                context_status="resolved",
                relation_expansion={
                    "needed": True,
                    "relation_kind": "relation_path",
                    "reference_mode": "selected_anchor",
                    "basis": "它如何连接到下游模块",
                    "selected_anchor_ids": ["ev-real", "ev-forged"],
                },
            ), ensure_ascii=False),
            offered_anchor_ids=["ev-real"],
            raw_user_query=query,
        )
        invalid_values = (
            {"needed": True},
            {
                "needed": True,
                "relation_kind": "invented",
                "reference_mode": "current_query",
                "basis": query,
            },
            {
                "needed": True,
                "relation_kind": "relation_path",
                "reference_mode": "current_query",
                "basis": "模型改写出的文字",
            },
            {
                "needed": True,
                "relation_kind": "relation_path",
                "reference_mode": "selected_anchor",
                "basis": query,
                "selected_anchor_ids": ["ev-forged"],
            },
        )

        self.assertTrue(valid["relation_expansion"]["needed"])
        self.assertEqual(
            ["ev-real"], valid["relation_expansion"]["selected_anchor_ids"]
        )
        self.assertIn("ev-real", valid["selected_anchor_ids"])
        for relation in invalid_values:
            with self.subTest(relation=relation):
                parsed = plan_module._parse_target_response(
                    json.dumps(_target(
                        query, relation_expansion=relation,
                    ), ensure_ascii=False),
                    offered_anchor_ids=["ev-real"],
                    raw_user_query=query,
                )
                self.assertFalse(parsed["relation_expansion"]["needed"])

    def test_target_relation_denial_cannot_be_widened_by_retrieval_planner(self):
        query = "只核对该记录直接记载的原因"
        target = _target(
            query,
            relation_expansion={"needed": False},
        )
        planner = {
            "intent": "cause_analysis",
            "query_targets": [{
                "intent": "cause_analysis",
                "required_fields": ["fault_cause", "structural_cause_chain"],
            }],
            "rewritten_query": query,
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": [
                    "fault_cause", "structural_cause_chain",
                ],
            },
            "relation_expansion": {
                "needed": True,
                "relation_kind": "structural_cause_chain",
                "reference_mode": "current_query",
                "basis": "核对该记录直接记载的原因",
            },
            "tools_plan": [
                {"tool": "fmeca_query"},
                {"tool": "graph_traversal"},
            ],
        }
        with (
            patch.object(
                plan_module.llm_service,
                "_call_model",
                side_effect=[
                    json.dumps(target, ensure_ascii=False),
                    json.dumps(planner, ensure_ascii=False),
                ],
            ),
            patch.object(
                plan_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(plan_module.plan_node({
                "query": query,
                "messages": [{"role": "user", "content": query}],
                "memory_enabled": False,
                "iteration": 0,
            }))

        self.assertFalse(result["relation_expansion"]["needed"])
        self.assertNotIn("graph_traversal", result["allowed_tools"])
        self.assertEqual(["fault_cause"], result["query_frame"]["required_answer_fields"])

    def test_query_writer_cannot_supply_missing_target_relation_authority(self):
        query = "核对这个对象如何连接到下游模块"
        target = _target(
            query,
            primary_intent="architecture_query",
            targets=["architecture_query"],
            required_answer_fields=["relation_path"],
            response_focus=[],
        )
        planner = {
            "intent": "architecture_query",
            "query_targets": [{
                "intent": "architecture_query",
                "required_fields": ["relation_path"],
            }],
            "rewritten_query": query,
            "query_frame": {
                "coverage_scope": "focused_detail",
                "required_answer_fields": ["relation_path"],
            },
            "relation_expansion": {
                "needed": True,
                "relation_kind": "relation_path",
                "reference_mode": "current_query",
                "basis": "如何连接到下游模块",
            },
            "tools_plan": [{"tool": "graph_traversal"}],
        }
        with (
            patch.object(
                plan_module.llm_service,
                "_call_model",
                side_effect=[
                    json.dumps(target, ensure_ascii=False),
                    json.dumps(planner, ensure_ascii=False),
                ],
            ),
            patch.object(
                plan_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(plan_module.plan_node({
                "query": query,
                "messages": [{"role": "user", "content": query}],
                "memory_enabled": False,
                "iteration": 0,
            }))

        self.assertFalse(result["relation_expansion"]["needed"])
        self.assertNotIn("graph_traversal", result["allowed_tools"])

    def test_only_grounded_coverage_request_authorizes_closed_collection(self):
        query = "供电子系统的所有故障模式都有哪些"
        bare = plan_module._parse_target_response(
            json.dumps(_target(
                query,
                coverage_scope="complete_collection",
                collection_subject="供电子系统",
                collection_member_kind="failure_mode",
                required_answer_fields=["closed_collection_members"],
            ), ensure_ascii=False),
            raw_user_query=query,
        )
        grounded = plan_module._parse_target_response(
            json.dumps(_target(
                query,
                coverage_request={
                    "scope": "complete_collection",
                    "subject": "供电子系统",
                    "member_kind": "failure_mode",
                    "reference_mode": "current_query",
                    "basis": "所有故障模式都有哪些",
                },
                required_answer_fields=[],
            ), ensure_ascii=False),
            raw_user_query=query,
        )
        forged = plan_module._parse_target_response(
            json.dumps(_target(
                query,
                coverage_request={
                    "scope": "complete_collection",
                    "subject": "显示子系统",
                    "member_kind": "failure_mode",
                    "reference_mode": "current_query",
                    "basis": "所有故障模式都有哪些",
                },
                required_answer_fields=["closed_collection_members"],
            ), ensure_ascii=False),
            raw_user_query=query,
        )

        for denied in (bare, forged):
            self.assertEqual("unknown", denied["coverage_scope"])
            self.assertEqual("", denied["collection_subject"])
            self.assertNotIn(
                "closed_collection_members", denied["required_answer_fields"]
            )
        self.assertEqual("complete_collection", grounded["coverage_scope"])
        self.assertIn(
            "closed_collection_members", grounded["required_answer_fields"]
        )

    def test_invalid_closure_basis_or_dimension_fails_closed(self):
        query = "运行中出现复位，请分析原因"
        for closure in (
            {
                "required": True,
                "required_dimensions": ["maintenance_action"],
                "basis": "现场完整闭环",
            },
            {
                "required": True,
                "required_dimensions": ["invented_dimension"],
                "basis": "出现复位",
            },
            {"required": True, "basis": "出现复位"},
        ):
            with self.subTest(closure=closure):
                parsed = plan_module._parse_target_response(
                    json.dumps(_target(query, diagnostic_closure=closure), ensure_ascii=False),
                    raw_user_query=query,
                )
                self.assertNotIn("diagnostic_closure", parsed)

    def test_forged_topology_subject_cannot_authorize_rbd(self):
        query = "A路供电中断后还能继续运行吗"
        forged = _target(
            query,
            primary_intent="impact_analysis",
            targets=["impact_analysis"],
            required_answer_fields=["local_impact"],
            topology_dependency={
                "required": True,
                "subject": "B路供电",
                "subject_kind": "supply_path",
                "failure_state": "中断",
                "question_aspects": ["continued_operation"],
                "basis": query,
            },
        )
        parsed = plan_module._parse_target_response(
            json.dumps(forged, ensure_ascii=False), raw_user_query=query
        )

        self.assertFalse(parsed["topology_dependency"]["required"])
        self.assertEqual(
            ["impact_analysis"],
            [item["intent"] for item in parsed["query_targets"]],
        )
        self.assertNotIn("rbd_success_criterion", parsed["required_answer_fields"])

    def test_separately_grounded_topology_spans_are_accepted(self):
        query = "A路供电当前已经中断；请判断设备能否继续运行"
        parsed = plan_module._parse_target_response(
            json.dumps(_target(
                query,
                primary_intent="impact_analysis",
                targets=["impact_analysis"],
                required_answer_fields=["local_impact"],
                topology_dependency={
                    "required": True,
                    "subject": "A路供电",
                    "subject_kind": "supply_path",
                    "failure_state": "中断",
                    "question_aspects": ["continued_operation"],
                    "subject_basis": "A路供电当前已经中断",
                    "failure_state_basis": "当前已经中断",
                },
            ), ensure_ascii=False),
            raw_user_query=query,
        )
        self.assertTrue(parsed["topology_dependency"]["required"])
        self.assertIn("reliability_query", {
            item["intent"] for item in parsed["query_targets"]
        })

    def test_model_answer_is_not_overridden_without_explicit_dimensions(self):
        state = {
            "query": "请回答当前问题",
            "intent": "cause_analysis",
            "query_targets": [{"intent": "cause_analysis"}],
            "authorized_tools": [
                "fmeca_query", "hmc_query", "mfl_query",
                "fault_history_query", "hybrid_search", "evidence_synthesis",
            ],
            "allowed_tools": ["evidence_synthesis"],
            "tool_results": [],
            "evidence_registry": [],
            "graph_seed_bindings": [],
        }
        with (
            patch.object(
                adaptive_module.llm_service,
                "_call_model",
                return_value=json.dumps({
                    "decision": "answer", "next_tools": [], "reason": "合同已满足",
                }, ensure_ascii=False),
            ),
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
            result = _run(adaptive_module.adaptive_tool_route_node(state))
        self.assertEqual([], result["pending_tools"])

    def test_graph_seed_selection_is_invariant_to_intent_labels(self):
        state = {
            "query": "请核对已选记录与目标模块的连接",
            "query_frame": {
                "required_answer_fields": ["relation_path"],
                "source_obligations": [{"tool": "fault_history_query"}],
            },
            "authorized_tools": [
                "function_locator", "fmeca_query", "fault_history_query",
                "graph_traversal", "evidence_synthesis",
            ],
            "tool_results": [
                make_tool_result(tool="function_locator", status="ok"),
                make_tool_result(tool="fmeca_query", status="ok"),
                make_tool_result(tool="fault_history_query", status="ok"),
            ],
            "graph_seed_bindings": [
                {
                    "logical_id": "sg:b:component:1",
                    "source_tool": "function_locator",
                    "binding_status": "candidate",
                },
                {
                    "logical_id": "sg:b:fmeca_record:1",
                    "source_tool": "fmeca_query",
                    "binding_status": "candidate",
                },
                {
                    "logical_id": "sg:b:fault_history_case:1",
                    "source_tool": "fault_history_query",
                    "binding_status": "candidate",
                },
                {
                    "logical_id": "sg:b:rbd_component:exact",
                    "source_tool": "rbd_query",
                    "binding_status": "exact",
                },
            ],
        }
        results = []
        for intent in ("cause_analysis", "history_query", "hybrid"):
            results.append(adaptive_module._eligible_graph_seed_bindings({
                **state,
                "intent": intent,
                "query_targets": [{"intent": intent}],
            }))

        identities = [
            [item["logical_id"] for item in result]
            for result in results
        ]
        self.assertEqual(identities[0], identities[1])
        self.assertEqual(identities[1], identities[2])
        self.assertIn("sg:b:component:1", identities[0])
        self.assertIn("sg:b:fault_history_case:1", identities[0])
        self.assertIn("sg:b:rbd_component:exact", identities[0])
        self.assertNotIn("sg:b:fmeca_record:1", identities[0])

    def test_relation_contract_does_not_gain_fta_from_an_intent_label(self):
        decisions = []
        for intent in (
            "cause_analysis", "impact_analysis", "architecture_query", "hybrid",
        ):
            query = "请核对这个对象的连接"
            decisions.append(resolve_tool_policy(
                intent=intent,
                raw_user_query=query,
                query_targets=[{
                    "intent": intent,
                    "required_fields": ["relation_path"],
                }],
                required_answer_fields=["relation_path"],
                coverage_scope="focused_detail",
                relation_expansion={
                    "needed": True,
                    "relation_kind": "relation_path",
                    "reference_mode": "current_query",
                    "basis": "核对这个对象的连接",
                },
                proposed_plan=[],
            ))

        self.assertEqual(
            1, len({tuple(item["allowed_tools"]) for item in decisions})
        )
        self.assertTrue(all(
            "fta_query" not in item["authorized_tools"] for item in decisions
        ))
        self.assertTrue(all(
            {"function_locator", "graph_traversal"}.issubset(item["allowed_tools"])
            for item in decisions
        ))

    def test_exact_business_identity_resolution_is_invariant_to_intent_labels(self):
        decisions = []
        for intent in (
            "entity_profile", "cause_analysis", "architecture_query", "reliability_query",
        ):
            decisions.append(resolve_tool_policy(
                intent=intent,
                raw_user_query="请查询 FM-SYN-042",
                query_targets=[{
                    "intent": intent,
                    "required_fields": ["record_identity"],
                }],
                required_answer_fields=["record_identity"],
                coverage_scope="focused_detail",
                proposed_plan=[],
            ))

        self.assertEqual(
            1, len({tuple(item["allowed_tools"]) for item in decisions})
        )
        self.assertEqual(
            1, len({tuple(item["authorized_tools"]) for item in decisions})
        )

    def test_grounded_closure_dimensions_can_override_answer_to_fill_only_the_contract(self):
        query = "设备反复复位，请闭合现场处置与恢复结果"
        cause = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-1",
            title="复位",
            content="结构化记录",
            metadata={"answer_fields": {"fault_cause": "电源瞬断"}},
        )
        state = {
            "query": query,
            "intent": "cause_analysis",
            "query_targets": [{
                "intent": "cause_analysis", "required_fields": ["fault_cause"],
            }],
            "query_frame": {
                "required_answer_fields": ["fault_cause"],
                "diagnostic_closure": {
                    "required": True,
                    "required_dimensions": [
                        "maintenance_action", "history_result",
                    ],
                    "basis": "反复复位，请闭合现场处置与恢复结果",
                },
            },
            "authorized_tools": [
                "fmeca_query", "hmc_query", "mfl_query",
                "fault_history_query", "hybrid_search", "evidence_synthesis",
            ],
            "allowed_tools": ["fmeca_query", "evidence_synthesis"],
            "tool_results": [make_tool_result(
                tool="fmeca_query", status="ok", data={"count": 1},
                evidence_ids=[cause["evidence_id"]],
            )],
            "evidence_registry": [cause],
            "graph_seed_bindings": [],
        }
        with (
            patch.object(
                adaptive_module.llm_service,
                "_call_model",
                return_value=json.dumps({
                    "decision": "answer", "next_tools": [], "reason": "直接原因已命中",
                }, ensure_ascii=False),
            ),
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
            result = _run(adaptive_module.adaptive_tool_route_node(state))

        self.assertTrue({
            "hmc_query", "mfl_query", "fault_history_query",
        }.issubset(result["pending_tools"]))
        self.assertNotIn("rbd_query", result["pending_tools"])
        self.assertNotIn("function_locator", result["pending_tools"])

    def test_planner_failure_uses_at_most_one_non_domain_fallback(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            planner_failed=True,
            proposed_plan=[{"tool": "fta_query"}, {"tool": "graph_traversal"}],
        )
        query_tools = [
            tool for tool in decision["allowed_tools"] if tool in QUERY_TOOL_NAMES
        ]
        self.assertEqual(["hybrid_search"], query_tools)


if __name__ == "__main__":
    unittest.main()
