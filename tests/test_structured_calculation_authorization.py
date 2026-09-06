"""Contract tests for model-authored, deterministically validated calculations.

These tests intentionally describe the replacement for raw-query calculation
keyword detection.  The target classifier and retrieval planner may understand
arbitrary natural language, but neither one is trusted on its own.  Only their
matching ``calculation_authorizations`` objects may authorize a deterministic
FTA/RBD calculator, after the server validates the operation, mode, target and
verbatim basis against the current user turn.

The production implementation is deliberately not part of this test-only
change, so the new contract tests are expected to be red until that migration
is implemented.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent.contracts import make_evidence
from backend.agent.nodes import plan as plan_module
from backend.agent.nodes.evidence_gate import evidence_gate_node
from backend.agent.tools import icp_calculation as calculation_module


def _run(coroutine):
    return asyncio.run(coroutine)


def _authorization(
    operation: str,
    target: str,
    basis: str,
    *,
    mode: str = "calculate",
):
    return {
        "operation": operation,
        "mode": mode,
        "target": target,
        "basis": basis,
    }


def _rbd_graph():
    return {
        "vertices": [
            {"id": "start", "label": "Start", "properties": {"name": "START"}},
            {
                "id": "block",
                "label": "ReliabilityBlock",
                "properties": {"name": "处理模块", "reliability": 0.98},
            },
            {"id": "end", "label": "End", "properties": {"name": "END"}},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "block"},
            {"id": "e2", "source": "block", "target": "end"},
        ],
    }


def _formal_rbd_row():
    return {
        "diagram_type": "rbd",
        "revision_id": "revision-rbd-auth",
        "diagram_key": "RBD-AUTH-01",
        "doc_id": "doc-rbd-auth",
        "doc_name": "可靠性框图.docx",
        "image_title": "任务链路",
        "validation_status": "auto_validated",
        "projection_status": "published",
        "is_active": True,
        "graph": _rbd_graph(),
    }


def _formal_rbd_evidence():
    return make_evidence(
        source_kind="postgres",
        authority="validated_relation",
        evidence_level="L2_CORROBORATED",
        claim_type="rbd_structure",
        doc_type="rbd",
        doc_id="doc-rbd-auth",
        record_id="revision-rbd-auth",
        review_status="auto_validated",
        content="START 到处理模块再到 END",
        metadata={
            "revision_id": "revision-rbd-auth",
            "diagram_key": "RBD-AUTH-01",
            "validation_status": "auto_validated",
            "projection_status": "published",
            "is_active": True,
        },
    )


class StructuredCalculationAuthorizationTests(unittest.TestCase):
    """Pin the semantic boundary without pinning any particular wording."""

    @staticmethod
    def _target(query, intent, authorizations=None):
        authorizations = list(authorizations or [])
        return {
            "primary_intent": intent,
            "targets": [intent],
            "contextualized_query": query,
            "subjects": ["当前工程模型"],
            "requested_facets": [],
            "required_answer_fields": [],
            "question_semantics": (
                "act"
                if any(item.get("mode") == "calculate" for item in authorizations)
                else "explain"
            ),
            "response_focus": [],
            "response_scope": "standalone",
            "diagnostic_closure": {"required": False, "reason": ""},
            "topology_dependency": {"required": False},
            "coverage_scope": "focused_detail",
            "collection_subject": "",
            "collection_member_kind": "unknown",
            "context_used": False,
            "context_status": "standalone",
            "selected_anchor_ids": [],
            "calculation_authorizations": authorizations,
        }

    @staticmethod
    def _planner(query, intent, tool, authorizations=None):
        query_tool = "fta_query" if tool == "fta_analyzer" else "rbd_query"
        return {
            "intent": intent,
            "query_targets": [{"intent": intent}],
            "rewritten_query": query,
            "query_frame": {},
            "tool_queries": {query_tool: [query]},
            "tools_plan": [
                {"tool": query_tool, "reason": "读取正式工程图"},
                {"tool": tool, "reason": "执行确定性算法"},
            ],
            "relation_expansion": {"needed": False, "reason": ""},
            "calculation_authorizations": list(authorizations or []),
        }

    def _plan(
        self,
        query,
        *,
        intent,
        tool,
        target_authorizations=None,
        planner_authorizations=None,
        messages=None,
    ):
        target = self._target(query, intent, target_authorizations)
        planner = self._planner(query, intent, tool, planner_authorizations)
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
            return _run(
                plan_module.plan_node(
                    {
                        "query": query,
                        "messages": messages
                        or [{"role": "user", "content": query}],
                        "memory_enabled": False,
                        "iteration": 0,
                    }
                )
            )

    def assert_authorized(self, result, operation, tool):
        self.assertIn("calculation_authorizations", result)
        authorizations = result["calculation_authorizations"]
        self.assertEqual(1, len(authorizations))
        self.assertEqual(operation, authorizations[0]["operation"])
        self.assertEqual("calculate", authorizations[0]["mode"])
        self.assertIn(tool, result["allowed_tools"])
        self.assertIn(operation, result["calculation_requests"])

    def assert_not_authorized(self, result, tool):
        self.assertIn("calculation_authorizations", result)
        executable = [
            item
            for item in result["calculation_authorizations"]
            if item.get("mode") == "calculate"
        ]
        self.assertEqual([], executable)
        self.assertNotIn(tool, result["allowed_tools"])
        self.assertEqual([], result["calculation_requests"])

    def test_natural_paraphrases_are_authorized_by_matching_model_structures(self):
        cases = [
            (
                "把这棵故障树化成不能再约简的底层组合给我",
                "cause_analysis",
                "fta_cut_sets",
                "fta",
                "fta_analyzer",
                "不能再约简的底层组合",
            ),
            (
                "根据这张树把顶层失效的量化风险推出来",
                "cause_analysis",
                "fta_probability",
                "fta",
                "fta_analyzer",
                "顶层失效的量化风险推出来",
            ),
            (
                "依据框图已有参数，给出这次十小时任务完成的量化把握",
                "reliability_query",
                "rbd_reliability",
                "rbd",
                "rbd_calculator",
                "给出这次十小时任务完成的量化把握",
            ),
        ]
        for query, intent, operation, target, tool, basis in cases:
            with self.subTest(query=query):
                authorization = _authorization(operation, target, basis)
                result = self._plan(
                    query,
                    intent=intent,
                    tool=tool,
                    target_authorizations=[authorization],
                    planner_authorizations=[authorization],
                )
                self.assert_authorized(result, operation, tool)

    def test_calculation_words_do_not_override_explain_or_read_document_mode(self):
        cases = [
            (
                "解释计算顶事件概率需要哪些输入",
                "cause_analysis",
                "fta_probability",
                "fta",
                "fta_analyzer",
                "explain",
                "计算顶事件概率",
            ),
            (
                "资料中记录的系统可靠度是多少，不要重新计算",
                "reliability_query",
                "rbd_reliability",
                "rbd",
                "rbd_calculator",
                "read_document",
                "资料中记录的系统可靠度是多少",
            ),
        ]
        for query, intent, operation, target, tool, mode, basis in cases:
            with self.subTest(mode=mode):
                authorization = _authorization(
                    operation, target, basis, mode=mode
                )
                result = self._plan(
                    query,
                    intent=intent,
                    tool=tool,
                    target_authorizations=[authorization],
                    planner_authorizations=[authorization],
                )
                self.assertIn("calculation_authorizations", result)
                self.assertEqual(mode, result["calculation_authorizations"][0]["mode"])
                self.assertNotIn(tool, result["allowed_tools"])
                self.assertEqual([], result["calculation_requests"])

    def test_basis_must_be_one_contiguous_span_from_the_current_user_turn(self):
        query = "依据已有参数，给出任务完成的量化把握"
        invalid_bases = [
            # Same words, but punctuation and omitted words make this a bag of
            # terms rather than one exact span.
            "依据已有参数量化把握",
            # This phrase exists only in the prior assistant answer.
            "请计算十小时可靠度",
            # This is a plausible rewrite, not the current user's wording.
            "算出十小时任务成功概率",
        ]
        messages = [
            {"role": "assistant", "content": "如果需要，请计算十小时可靠度。"},
            {"role": "user", "content": query},
        ]
        for basis in invalid_bases:
            with self.subTest(basis=basis):
                authorization = _authorization(
                    "rbd_reliability", "rbd", basis
                )
                result = self._plan(
                    query,
                    intent="reliability_query",
                    tool="rbd_calculator",
                    target_authorizations=[authorization],
                    planner_authorizations=[authorization],
                    messages=messages,
                )
                self.assert_not_authorized(result, "rbd_calculator")

    def test_operation_and_operation_target_pair_are_closed(self):
        query = "把当前模型做成量化结果"
        invalid = [
            _authorization(
                "monte_carlo_reliability", "rbd", "做成量化结果"
            ),
            _authorization(
                "rbd_reliability",
                "rbd",
                "做成量化结果",
                mode="estimate",
            ),
            _authorization(
                "rbd_reliability", "fta", "做成量化结果"
            ),
            _authorization(
                "fta_probability", "rbd", "做成量化结果"
            ),
        ]
        for authorization in invalid:
            with self.subTest(authorization=authorization):
                result = self._plan(
                    query,
                    intent="reliability_query",
                    tool="rbd_calculator",
                    target_authorizations=[authorization],
                    planner_authorizations=[authorization],
                )
                self.assert_not_authorized(result, "rbd_calculator")

    def test_only_target_owns_authorization_and_conflicting_target_modes_fail_closed(self):
        query = "把这次任务完成的量化把握给出来"
        rbd = _authorization(
            "rbd_reliability", "rbd", "任务完成的量化把握"
        )
        fta = _authorization(
            "fta_probability", "fta", "任务完成的量化把握"
        )
        cases = [
            ([rbd], []),
            ([], [rbd]),
            ([rbd], [fta]),
            (
                [
                    rbd,
                    _authorization(
                        "rbd_reliability",
                        "rbd",
                        "任务完成的量化把握",
                        mode="explain",
                    ),
                ],
                [rbd],
            ),
        ]
        for target_authorizations, planner_authorizations in cases:
            with self.subTest(
                target=target_authorizations,
                planner=planner_authorizations,
            ):
                result = self._plan(
                    query,
                    intent="reliability_query",
                    tool="rbd_calculator",
                    target_authorizations=target_authorizations,
                    planner_authorizations=planner_authorizations,
                )
                if target_authorizations == [rbd]:
                    self.assert_authorized(result, "rbd_reliability", "rbd_calculator")
                else:
                    self.assert_not_authorized(result, "rbd_calculator")

    def test_explicit_calculation_words_without_target_contract_fail_closed(self):
        query = "请计算这个 RBD 的系统可靠度"
        result = self._plan(
            query,
            intent="reliability_query",
            tool="rbd_calculator",
            target_authorizations=[],
            planner_authorizations=[],
        )

        self.assert_not_authorized(result, "rbd_calculator")

    def test_calculation_node_never_reauthorizes_from_raw_query(self):
        state = {
            "query": "请计算这个 RBD 的系统可靠度",
            "intent": "reliability_query",
            "allowed_tools": ["rbd_query", "rbd_calculator"],
            # Explicitly empty is authoritative.  A downstream node must not
            # recreate permission from words in ``query``.
            "calculation_authorizations": [],
            "confirmed_graphs": [_formal_rbd_row()],
        }
        with (
            patch.object(calculation_module, "_query_graphs", return_value=[]),
            patch.object(
                calculation_module,
                "calculate_rbd",
                wraps=calculation_module.calculate_rbd,
            ) as calculator,
        ):
            result = _run(calculation_module.icp_calculation_node(state))

        calculator.assert_not_called()
        self.assertEqual([], result["tool_calls"])
        self.assertEqual([], result["calculation_results"])

    def test_evidence_gate_never_reauthorizes_injected_derivation_from_raw_query(self):
        parent = _formal_rbd_evidence()
        calculation = make_evidence(
            source_kind="calculation",
            authority="deterministic_derivation",
            evidence_level="L2_CORROBORATED",
            claim_type="rbd_calculation",
            doc_type="rbd",
            doc_id="doc-rbd-auth",
            record_id="revision-rbd-auth",
            content='{"system_reliability": 0.98}',
            derived_from=[parent["evidence_id"]],
            metadata={"numeric_complete": True},
        )
        result = _run(
            evidence_gate_node(
                {
                    "query": "请计算这个 RBD 的系统可靠度",
                    "intent": "reliability_query",
                    "allowed_tools": ["rbd_query", "rbd_calculator"],
                    "calculation_authorizations": [],
                    "evidence_registry": [parent, calculation],
                }
            )
        )

        self.assertIn(
            "formal_calculation", result["safety_decision"]["blocked_claims"]
        )
        self.assertNotIn(calculation["evidence_id"], result["approved_evidence_ids"])

    def test_duration_is_parsed_deterministically_from_current_query_only(self):
        query = "对 RBD-AUTH-01 以 12.5 小时为任务时长，给出系统可靠度"
        authorization = _authorization(
            "rbd_reliability", "rbd", "给出系统可靠度"
        )
        state = {
            "query": query,
            # Neither a retrieval rewrite nor another model-controlled field
            # may replace the numeric parameter in the current user turn.
            "rewritten_query": "以 999 小时为任务时长计算系统可靠度",
            "intent": "reliability_query",
            "allowed_tools": ["rbd_query", "rbd_calculator"],
            "calculation_authorizations": [authorization],
            # Compatibility projection derived from the validated structure;
            # it is not an independent source of authority.
            "calculation_requests": ["rbd_reliability"],
            "confirmed_graphs": [_formal_rbd_row()],
            "evidence_registry": [_formal_rbd_evidence()],
        }
        with patch.object(calculation_module, "_query_graphs", return_value=[]):
            result = _run(calculation_module.icp_calculation_node(state))

        self.assertEqual(12.5, calculation_module._mission_time(query))
        self.assertEqual(
            25.0,
            calculation_module._mission_time("按二十五小时任务时长给出结果"),
        )
        self.assertEqual(12.5, result["tool_calls"][0]["params"]["mission_time"])
        calculation_evidence = next(item for item in result["evidence_registry"] if item["authority"] == "deterministic_derivation")
        public_parameters = calculation_evidence["metadata"]["calculation"]["parameters"]
        self.assertEqual({"mission_time": 12.5, "mission_time_unit": "h"}, public_parameters)
        self.assertEqual("1", calculation_evidence["metadata"]["calculation"]["units"]["system_reliability"])

    def test_formal_graph_alone_never_grants_calculation_authority(self):
        state = {
            "query": "说明这个正式 RBD 的结构和适用边界",
            "intent": "reliability_query",
            # Even a stale or malicious upstream tool list is insufficient.
            "allowed_tools": ["rbd_query", "rbd_calculator"],
            "calculation_authorizations": [],
            "confirmed_graphs": [_formal_rbd_row()],
        }
        with (
            patch.object(calculation_module, "_query_graphs", return_value=[]),
            patch.object(
                calculation_module,
                "calculate_rbd",
                wraps=calculation_module.calculate_rbd,
            ) as calculator,
        ):
            result = _run(calculation_module.icp_calculation_node(state))

        calculator.assert_not_called()
        self.assertEqual([], result["calculation_results"])


if __name__ == "__main__":
    unittest.main()
