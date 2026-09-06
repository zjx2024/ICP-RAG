"""Red tests for the structured, non-lexical plan contract.

These tests describe the intended boundary between open-language understanding
and deterministic validation:

* the target model decides discourse meaning and returns typed objects;
* Python validates closed enums, verbatim current-turn provenance, offered
  anchor IDs, and machine-produced missing slots;
* downstream code consumes those objects and must not reconstruct intent from
  a small list of Chinese words.

The backend migration is intentionally outside this test-only change.  This
module is therefore expected to fail until the structured semantic contract is
implemented.
"""

from __future__ import annotations

import backend.agent.nodes.answer_fault_rules as _source_answer_fault_rules

import asyncio
import json
import unittest
from typing import Any, Mapping, Sequence
from unittest.mock import patch

from backend.agent.nodes import answer as answer_module
from backend.agent.nodes import plan as plan_module


SEVEN_DOC_TYPES = {
    "function_locator",
    "hmc",
    "fta",
    "rbd",
    "fmeca",
    "mfl",
    "fault_history",
}


def _run(coroutine):
    return asyncio.run(coroutine)


def _source(
    doc_type: str,
    basis: str,
    *,
    requested_fields: Sequence[str] = (),
    question_role: str = "evidence",
) -> dict[str, Any]:
    return {
        "doc_type": doc_type,
        "basis": basis,
        "requested_fields": list(requested_fields),
        "question_role": question_role,
    }


def _role(role: str, subject: str, basis: str) -> dict[str, str]:
    return {"role": role, "subject": subject, "basis": basis}


def _comparison_subject(
    value: str, basis: str, *, kind: str = "failure_mode"
) -> dict[str, str]:
    return {"kind": kind, "value": value, "basis": basis}


class StructuredSemanticPlanContractTests(unittest.TestCase):
    @staticmethod
    def _target(query: str, intent: str = "architecture_query", **extra):
        target = {
            "primary_intent": intent,
            "targets": [intent],
            "contextualized_query": query,
            "subjects": [],
            "requested_facets": [],
            "required_answer_fields": [],
            "question_semantics": "describe",
            "response_focus": [],
            "response_scope": "standalone",
            "diagnostic_closure": {"required": False, "reason": ""},
            "topology_dependency": {"required": False},
            # Legacy fields deliberately remain a focused contract.  The new
            # typed objects below must be the only semantic authority.
            "coverage_scope": "focused_detail",
            "collection_subject": "",
            "collection_member_kind": "unknown",
            "context_used": False,
            "context_status": "standalone",
            "selected_anchor_ids": [],
        }
        target.update(extra)
        return target

    @staticmethod
    def _planner(query: str, intent: str = "architecture_query", **extra):
        default_tool = {
            "architecture_query": "function_locator",
            "cause_analysis": "fmeca_query",
            "impact_analysis": "fmeca_query",
            "reliability_query": "rbd_query",
            "history_query": "fault_history_query",
            "hybrid": "function_locator",
        }.get(intent, "function_locator")
        planner = {
            "intent": intent,
            "query_targets": [{"intent": intent}],
            "rewritten_query": query,
            "query_frame": {},
            "tool_queries": {default_tool: [query]},
            "tools_plan": [
                {"tool": default_tool, "reason": "读取权威结构化记录"}
            ],
            "relation_expansion": {"needed": False, "reason": ""},
        }
        planner.update(extra)
        return planner

    def _plan(
        self,
        query: str,
        *,
        target: Mapping[str, Any],
        planner: Mapping[str, Any] | None = None,
        messages: Sequence[Mapping[str, Any]] | None = None,
        state_extra: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int]:
        planned = planner or self._planner(
            query, str(target.get("primary_intent") or "architecture_query")
        )
        target_json = json.dumps(dict(target), ensure_ascii=False)
        planner_json = json.dumps(dict(planned), ensure_ascii=False)
        call_index = 0

        def model_response(*_args, **_kwargs):
            # Legacy collection code may ask for an extra repair pass.  Keep a
            # red test fast and deterministic instead of exhausting a finite
            # side-effect list inside ``asyncio.to_thread``.
            nonlocal call_index
            call_index += 1
            return target_json if call_index == 1 else planner_json

        with (
            patch.object(
                plan_module.llm_service,
                "_call_model",
                side_effect=model_response,
            ) as model_call,
            patch.object(
                plan_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            state = {
                "query": query,
                "messages": list(messages or [{"role": "user", "content": query}]),
                "memory_enabled": False,
                "iteration": 0,
                **dict(state_extra or {}),
            }
            result = _run(plan_module.plan_node(state))
        return result, model_call.call_count

    def test_missing_input_question_reaches_target_model_and_uses_machine_slots(self):
        query = "还需要我提供什么信息？"
        target = self._target(
            query,
            intent="hybrid",
            question_semantics="verify",
            response_focus=["missing_input_requirements"],
            response_scope="incremental",
            context_used=True,
            context_status="resolved",
            missing_input_requirements=[
                {
                    "slot": "mission_time",
                    "basis": "还需要我提供什么信息",
                }
            ],
        )
        messages = [
            {"role": "user", "content": "按这张可靠性图继续分析。"},
            {
                "role": "assistant",
                "content": "现有记录还不能形成数值结论。",
            },
            {"role": "user", "content": query},
        ]
        result, call_count = self._plan(
            query,
            target=target,
            messages=messages,
            state_extra={
                "missing_parameters": ["mission_time"],
                "unresolved_identity_slots": ["rbd_model_identity"],
            },
        )

        # A wording shortcut must not bypass semantic classification.
        self.assertGreaterEqual(call_count, 1)
        self.assertIn("missing_input_requirements", result["query_frame"])
        requirements = result["query_frame"]["missing_input_requirements"]
        self.assertEqual(
            {"mission_time", "rbd_model_identity"},
            {str(item.get("slot")) for item in requirements},
        )
        self.assertEqual(
            {"missing_parameters", "unresolved_identity_slots"},
            {
                str((item.get("source") or {}).get("name"))
                for item in requirements
            },
        )
        self.assertEqual("", result["clarification_question"])
        self.assertEqual("knowledge_base", result["scope_route"])

    def test_assistant_miss_wording_cannot_change_missing_input_requirements(self):
        query = "还需要我提供什么信息？"
        target = self._target(
            query,
            intent="hybrid",
            question_semantics="verify",
            response_focus=["missing_input_requirements"],
            response_scope="incremental",
            context_used=True,
            context_status="resolved",
            missing_input_requirements=[
                {
                    "slot": "mission_time",
                    "basis": "还需要我提供什么信息",
                }
            ],
        )

        def run_with_assistant(text: str) -> dict[str, Any]:
            result, _ = self._plan(
                query,
                target=target,
                messages=[
                    {"role": "user", "content": "继续做可靠性量化。"},
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": query},
                ],
                state_extra={"missing_parameters": ["mission_time"]},
            )
            return result

        with_miss_words = run_with_assistant("未找到匹配记录，当前无匹配。")
        without_miss_words = run_with_assistant("量化输入尚不齐备。")

        self.assertIn("missing_input_requirements", with_miss_words["query_frame"])
        self.assertIn("missing_input_requirements", without_miss_words["query_frame"])
        self.assertEqual(
            with_miss_words["query_frame"]["missing_input_requirements"],
            without_miss_words["query_frame"]["missing_input_requirements"],
        )
        self.assertEqual(
            with_miss_words.get("clarification_question"),
            without_miss_words.get("clarification_question"),
        )
        rendered = str(with_miss_words.get("clarification_question") or "")
        # None of these is a machine-reported missing slot in this test.
        for fixed_default in ("HMC", "LCN", "HAMC", "HSC", "构型", "故障现象"):
            self.assertNotIn(fixed_default, rendered)

    def test_requested_sources_accept_the_seven_closed_document_types(self):
        query = (
            "回查 Function Locator、HMC、FMECA、FTA、RBD、MFL 和故障历史，"
            "分别说明本轮可引用的资料。"
        )
        bases = {
            "function_locator": "Function Locator",
            "hmc": "HMC",
            "fmeca": "FMECA",
            "fta": "FTA",
            "rbd": "RBD",
            "mfl": "MFL",
            "fault_history": "故障历史",
        }
        target = self._target(
            query,
            intent="hybrid",
            requested_sources=[
                _source(doc_type, bases[doc_type]) for doc_type in bases
            ],
        )
        result, _ = self._plan(query, target=target)

        self.assertIn("requested_sources", result["query_frame"])
        self.assertEqual(
            SEVEN_DOC_TYPES,
            {
                str(item.get("doc_type"))
                for item in result["query_frame"]["requested_sources"]
            },
        )

    def test_requested_source_basis_and_enums_fail_closed(self):
        query = "请回查 RBD 中这条任务链的资料边界。"
        target = self._target(
            query,
            intent="hybrid",
            requested_sources=[
                _source("rbd", "RBD"),
                # Unknown document family.
                _source("eighth_document", "资料边界"),
                # Basis is absent from the current turn.
                _source("hmc", "上一轮提到的 HMC"),
                # A known source with an unknown question role is still invalid.
                _source("fta", "任务链", question_role="invent_facts"),
            ],
        )
        result, _ = self._plan(query, target=target)

        self.assertIn("requested_sources", result["query_frame"])
        requested = result["query_frame"]["requested_sources"]
        self.assertEqual(1, len(requested))
        self.assertEqual("rbd", requested[0]["doc_type"])
        self.assertEqual("RBD", requested[0]["basis"])
        self.assertEqual([], requested[0]["requested_fields"])
        self.assertEqual("evidence", requested[0]["question_role"])

    def test_naming_rbd_does_not_create_default_structure_obligations(self):
        query = "本轮只回查 RBD 这一资料来源。"
        target = self._target(
            query,
            intent="hybrid",
            requested_sources=[_source("rbd", "RBD")],
        )
        result, _ = self._plan(
            query,
            target=target,
            planner=self._planner(
                query,
                "hybrid",
                tool_queries={"rbd_query": [query]},
                tools_plan=[
                    {"tool": "rbd_query", "reason": "读取用户指定资料"}
                ],
            ),
        )

        fields = set(result["query_frame"].get("required_answer_fields") or [])
        self.assertFalse(
            fields
            & {
                "rbd_success_criterion",
                "rbd_path_structure",
                "rbd_redundancy_structure",
                "rbd_single_failure_boundary",
            }
        )
        task = answer_module._answer_task_instruction(result)
        self.assertIn('"doc_type":"rbd","fields":[]', task)
        self.assertNotIn("rbd_success_criterion", task)

    def test_source_obligation_contains_only_model_requested_fields(self):
        query = "RBD 对该支路能支持到什么能力边界？"
        requested = _source(
            "rbd",
            "RBD",
            requested_fields=["rbd_single_failure_boundary"],
            question_role="capability",
        )
        target = self._target(
            query,
            intent="reliability_query",
            required_answer_fields=["rbd_single_failure_boundary"],
            response_focus=["source_capability"],
            requested_sources=[requested],
        )
        result, _ = self._plan(query, target=target)

        obligations = result["query_frame"]["source_obligations"]
        self.assertEqual(
            ["rbd_single_failure_boundary"], obligations[0]["fields"]
        )
        self.assertNotIn(
            "rbd_success_criterion",
            result["query_frame"]["required_answer_fields"],
        )
        self.assertNotIn(
            "rbd_path_structure",
            result["query_frame"]["required_answer_fields"],
        )

    def test_coverage_request_can_authorize_closure_without_magic_words(self):
        query = "请对 ICP系统 的直属功能域按权威父级做一次闭合核验。"
        request = {
            "scope": "complete_collection",
            "subject": "ICP系统",
            "member_kind": "subsystem",
            "reference_mode": "current_query",
            "basis": "按权威父级做一次闭合核验",
        }
        target = self._target(query, coverage_request=request)
        result, _ = self._plan(query, target=target)

        self.assertIn("coverage_request", result["query_frame"])
        validated_request = result["query_frame"]["coverage_request"]
        for key, value in request.items():
            self.assertEqual(value, validated_request[key])
        self.assertEqual(
            "complete_collection", result["query_frame"]["coverage_scope"]
        )
        self.assertEqual("ICP系统", result["query_frame"]["collection_subject"])
        self.assertEqual(
            "subsystem", result["query_frame"]["collection_member_kind"]
        )

    def test_collection_words_cannot_promote_a_focused_model_contract(self):
        query = "ICP系统总数有几个，哪些子系统？"
        target = self._target(query)
        result, _ = self._plan(query, target=target)

        self.assertEqual("unknown", result["query_frame"]["coverage_scope"])
        self.assertNotIn(
            "closed_collection_members",
            result["query_frame"].get("required_answer_fields") or [],
        )

    def test_coverage_anchor_and_authority_claims_are_validated_structurally(self):
        query = "请对前述集合做闭合核验。"
        request = {
            "scope": "complete_collection",
            "subject": "电源子系统",
            "member_kind": "failure_mode",
            "reference_mode": "selected_anchor",
            "basis": "前述集合做闭合核验",
            "selected_anchor_ids": ["ev-real", "ev-invented"],
            # A model may request closure, but cannot declare PostgreSQL facts.
            "parent_id": "FAKE-PARENT",
            "member_ids": ["FAKE-MEMBER"],
            "authoritative_count": 99,
        }
        target = self._target(
            query,
            intent="cause_analysis",
            context_used=True,
            context_status="resolved",
            coverage_request=request,
        )
        messages = [
            {"role": "user", "content": "列一下这个子系统的故障模式。"},
            {
                "role": "assistant",
                "content": "已列出权威记录。",
                "conversation_anchors": [
                    {
                        "anchor_id": "ev-real",
                        "doc_type": "fmeca",
                        "record_id": "FM-REAL-001",
                        "answer_fields": {
                            "collection_parent_id": "SUB-REAL",
                            "collection_member_id": "FM-REAL-001",
                            "collection_member_kind": "failure_mode",
                            "collection_member_count": 3,
                            "collection_closed": True,
                        },
                    }
                ],
            },
            {"role": "user", "content": query},
        ]
        result, _ = self._plan(query, target=target, messages=messages)

        self.assertIn("coverage_request", result["query_frame"])
        validated = result["query_frame"]["coverage_request"]
        self.assertEqual(["ev-real"], validated["selected_anchor_ids"])
        self.assertNotIn("parent_id", validated)
        self.assertNotIn("member_ids", validated)
        self.assertNotIn("authoritative_count", validated)

    def test_architecture_roles_use_grounded_typed_objects_only(self):
        query = "甲接收单元处理后衔接乙处理单元。"
        valid_receiver = _role("receiver", "甲接收单元", "甲接收单元")
        valid_downstream = _role("downstream", "乙处理单元", "乙处理单元")
        target = self._target(
            query,
            architecture_roles=[
                valid_receiver,
                valid_downstream,
                _role("teleporter", "甲接收单元", "甲接收单元"),
                _role("responsible_component", "伪造模块", "甲接收单元"),
            ],
        )
        result, _ = self._plan(query, target=target)

        self.assertIn("architecture_roles", result["query_frame"])
        roles = result["query_frame"]["architecture_roles"]
        self.assertEqual(2, len(roles))
        self.assertEqual(
            [("receiver", "甲接收单元"), ("downstream", "乙处理单元")],
            [(item.get("role"), item.get("subject")) for item in roles],
        )
        self.assertEqual(
            ["甲接收单元", "乙处理单元"],
            [item.get("basis") for item in roles],
        )

    def test_architecture_question_words_do_not_create_roles_without_model_contract(self):
        query = "哪个模块负责接收，之后交给谁？"
        target = self._target(query, architecture_roles=[])
        result, _ = self._plan(query, target=target)

        self.assertIn("architecture_roles", result["query_frame"])
        self.assertEqual([], result["query_frame"]["architecture_roles"])

    def test_comparison_subjects_drive_parallel_mode_semantics(self):
        query = "把甲模式与乙模式放在一起权衡差异。"
        first = _comparison_subject("甲模式", "甲模式")
        second = _comparison_subject("乙模式", "乙模式")
        target = self._target(
            query,
            intent="cause_analysis",
            question_semantics="compare",
            comparison_subjects=[
                first,
                second,
                _comparison_subject("伪造模式", "甲模式"),
            ],
        )
        result, _ = self._plan(query, target=target)

        self.assertIn("comparison_subjects", result["query_frame"])
        subjects = result["query_frame"]["comparison_subjects"]
        self.assertEqual(2, len(subjects))
        self.assertEqual(
            [("failure_mode", "甲模式"), ("failure_mode", "乙模式")],
            [(item.get("kind"), item.get("value")) for item in subjects],
        )
        self.assertTrue(_source_answer_fault_rules._query_requests_parallel_failure_modes(result))

    def test_comparison_words_cannot_override_structured_negative(self):
        query = "不要比较甲模式还是乙模式，当前只解释甲模式。"
        target = self._target(
            query,
            intent="cause_analysis",
            question_semantics="explain",
            comparison_subjects=[],
        )
        result, _ = self._plan(query, target=target)

        self.assertIn("comparison_subjects", result["query_frame"])
        self.assertEqual([], result["query_frame"]["comparison_subjects"])
        self.assertFalse(_source_answer_fault_rules._query_requests_parallel_failure_modes(result))

    def test_non_contiguous_basis_never_becomes_a_semantic_binding(self):
        query = "甲模块经由中继连接乙模块，请查 RBD。"
        target = self._target(
            query,
            requested_sources=[
                _source("rbd", "RBD"),
                # The words exist separately but this exact span does not.
                _source("fta", "甲模块乙模块"),
            ],
            architecture_roles=[
                _role("receiver", "甲模块", "甲模块"),
                _role("downstream", "乙模块", "甲模块乙模块"),
            ],
        )
        result, _ = self._plan(query, target=target)

        self.assertIn("requested_sources", result["query_frame"])
        self.assertIn("architecture_roles", result["query_frame"])
        requested = result["query_frame"]["requested_sources"]
        roles = result["query_frame"]["architecture_roles"]
        self.assertEqual(["rbd"], [item.get("doc_type") for item in requested])
        self.assertEqual(
            [("receiver", "甲模块")],
            [(item.get("role"), item.get("subject")) for item in roles],
        )

    def test_payload_contract_is_grounded_and_reaches_query_frame(self):
        query = "外部仪器发来的测量帧进入系统接收端。"
        contract = {
            "kind": "data_flow",
            "direction": "external_to_system",
            "source_role": "外部仪器",
            "destination_role": "系统接收端",
            "basis": "外部仪器发来的测量帧进入系统接收端",
        }
        target = self._target(query, payload_contract=contract)

        result, _ = self._plan(query, target=target)

        self.assertEqual(contract, result["query_frame"]["payload_contract"])
        self.assertNotIn("subject_bindings", result["query_frame"])

    def test_payload_contract_invalid_enum_or_forged_basis_fails_closed(self):
        query = "采集单元把状态帧交给记录单元。"
        invalid_contracts = (
            {
                "kind": "teleport_flow",
                "direction": "internal",
                "source_role": "采集单元",
                "destination_role": "记录单元",
                "basis": "采集单元把状态帧交给记录单元",
            },
            {
                "kind": "signal_flow",
                "direction": "internal",
                "source_role": "伪造源",
                "destination_role": "记录单元",
                "basis": "采集单元把状态帧交给记录单元",
            },
            {
                "kind": "signal_flow",
                "direction": "internal",
                "source_role": "采集单元",
                "destination_role": "记录单元",
                "basis": "采集单元记录单元",
            },
        )
        for contract in invalid_contracts:
            with self.subTest(contract=contract):
                target = self._target(query, payload_contract=contract)
                result, _ = self._plan(query, target=target)
                self.assertNotIn("payload_contract", result["query_frame"])


if __name__ == "__main__":
    unittest.main()
