"""Red tests for model-owned routing and retrieval semantics.

The model owns open-language interpretation.  Deterministic Python may reject
malformed enums, invented spans, and unsupported identities, but it must not
promote a request by matching a short list of Chinese phrasings.  This module
captures that boundary before the corresponding backend migration.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import unittest
from typing import Any, Mapping, Sequence
from unittest.mock import patch

from backend.agent import query_understanding
from backend.agent.nodes import scope_router as scope_module
from backend.agent.tools import icp_diagnosis as diagnosis_module


def _run(coroutine):
    return asyncio.run(coroutine)


def _scope_response(route: str) -> str:
    return json.dumps(
        {
            "route": route,
            "confidence": 0.91,
            "reason": "结构化范围判断",
            "clarification_question": "",
        },
        ensure_ascii=False,
    )


def _knowledge_history(current_query: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "请定位上一轮观测到的异常。"},
        {
            "role": "assistant",
            "content": "已返回可追溯候选。",
            "scope_route": "knowledge_base",
            "conversation_anchors": [
                {
                    "anchor_id": "ev-prior-1",
                    "doc_type": "fmeca",
                    "record_id": "FM-PRIOR-1",
                }
            ],
        },
        {"role": "user", "content": current_query},
    ]


class ScopeModelAuthorityContractTests(unittest.TestCase):
    def _route(
        self,
        query: str,
        *,
        model_route: str,
        messages: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[dict[str, Any], int]:
        with (
            patch.object(
                scope_module.llm_service,
                "call_model",
                return_value=_scope_response(model_route),
            ) as model_call,
            patch.object(
                scope_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(
                scope_module.scope_router_node(
                    {
                        "query": query,
                        "messages": list(messages),
                    }
                )
            )
        return result, model_call.call_count

    def test_every_nonempty_rewrite_uses_the_same_model_route(self):
        """Capability, supplement, boundary, and continuation are semantics.

        Returning an intentionally identical route makes any deterministic
        wording override immediately observable without blessing one phrase.
        """

        cases = (
            ("你能查哪些资料？", []),
            ("请补查 RBD 依据。", _knowledge_history("请补查 RBD 依据。")),
            ("RBD 能证明什么？", []),
            (
                "为了继续排查还需要我提供什么信息？",
                _knowledge_history("为了继续排查还需要我提供什么信息？"),
            ),
        )
        for query, messages in cases:
            with self.subTest(query=query):
                result, call_count = self._route(
                    query,
                    model_route="out_of_scope",
                    messages=messages,
                )
                self.assertEqual(1, call_count)
                self.assertEqual("model", result["scope_route_source"])
                self.assertEqual("out_of_scope", result["scope_route"])

    def test_model_route_is_not_rewritten_from_request_vocabulary(self):
        cases = (
            ("你能查 RBD 吗？", "knowledge_base"),
            ("为何没有追加故障树资料？", "agent_capability"),
            ("可靠性框图能说明哪些边界？", "clarification"),
            ("接下来还缺哪些输入？", "out_of_scope"),
        )
        for query, route in cases:
            history = _knowledge_history(query)
            with self.subTest(query=query, route=route):
                result, call_count = self._route(
                    query,
                    model_route=route,
                    messages=history,
                )
                self.assertEqual(1, call_count)
                self.assertEqual(route, result["scope_route"])
                self.assertEqual("model", result["scope_route_source"])

    def test_empty_input_is_the_only_deterministic_scope_exception(self):
        with patch.object(scope_module.llm_service, "call_model") as model_call:
            result = _run(
                scope_module.scope_router_node(
                    {"query": " \n\t ", "messages": []}
                )
            )

        self.assertEqual("clarification", result["scope_route"])
        self.assertEqual(
            "deterministic_empty_input", result["scope_route_source"]
        )
        model_call.assert_not_called()

    def test_prior_knowledge_requires_a_machine_proof(self):
        query = "延续当前对象。"
        by_scope = [
            {"role": "user", "content": "此前的任意请求"},
            {
                "role": "assistant",
                "content": "已处理。",
                "scope_route": "knowledge_base",
            },
            {"role": "user", "content": query},
        ]
        by_anchor = [
            {"role": "user", "content": "此前的任意请求"},
            {
                "role": "assistant",
                "content": "已处理。",
                "conversation_anchors": [
                    {"doc_type": "hmc", "record_id": "HMC-REAL-1"}
                ],
            },
            {"role": "user", "content": query},
        ]

        self.assertTrue(scope_module._has_prior_knowledge_turn(by_scope, query))
        self.assertTrue(scope_module._has_prior_knowledge_turn(by_anchor, query))

    def test_old_question_words_are_not_prior_knowledge_proof(self):
        query = "延续当前对象。"
        messages = [
            {
                "role": "user",
                "content": "RBD、故障、维修、模块和可靠性都是什么？",
            },
            {"role": "assistant", "content": "这是一段没有运行元数据的回答。"},
            {"role": "user", "content": query},
        ]

        self.assertFalse(
            scope_module._has_prior_knowledge_turn(messages, query)
        )


class DiagramCollectionSemanticAuthorizationTests(unittest.TestCase):
    @staticmethod
    def _revision(revision_id: str, title: str, block_name: str = ""):
        vertices = []
        if block_name:
            vertices.append(
                {
                    "id": f"block-{revision_id}",
                    "label": "block",
                    "properties": {"name": block_name},
                }
            )
        return {
            "revision_id": revision_id,
            "diagram_key": revision_id,
            "title": title,
            "canonical_graph": {"vertices": vertices, "edges": []},
            "validation_status": "auto_validated",
            "projection_status": "published",
            "is_active": True,
        }

    def setUp(self):
        self.revisions = [
            self._revision("R1", "第一工程模型", "甲域供电单元"),
            self._revision("R2", "第二工程模型", "乙域处理单元"),
            self._revision("R3", "第三工程模型", "丙域呈现单元"),
        ]

    def _query(
        self,
        terms: Sequence[str],
        *,
        query_frame: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        # ``query_frame`` is the intended boundary for the migration.  Keep
        # today's red test an assertion failure (rather than a TypeError)
        # until the optional keyword is added to the production helper.
        kwargs: dict[str, Any] = {}
        if "query_frame" in inspect.signature(
            diagnosis_module._query_diagrams
        ).parameters:
            kwargs["query_frame"] = dict(query_frame or {})
        with patch.object(
            diagnosis_module.diagram_repository,
            "list_active_revisions",
            return_value=self.revisions,
        ):
            return diagnosis_module._query_diagrams(
                "rbd", list(terms), 2, **kwargs
            )

    def test_list_punctuation_cannot_expand_an_entire_diagram_family(self):
        phrasings = (
            "甲域、乙域和丙域分别依赖哪些环节",
            "甲域、乙域和丙域各自包括什么",
            "甲域、乙域、丙域的覆盖情况",
        )
        for query in phrasings:
            with self.subTest(query=query):
                rows = self._query(
                    [query, "甲域", "乙域别称", "丙域别称"]
                )
                self.assertEqual(
                    ["R1"], [row["revision_id"] for row in rows]
                )
                self.assertFalse(
                    any(row.get("category_context") for row in rows)
                )

    def test_grounded_multi_subject_contract_can_authorize_family_completion(self):
        query = "把甲域连同乙域和丙域纳入同一层级核验。"
        frame = {
            "original_query": query,
            "subject_bindings": [
                {
                    "doc_type": "rbd",
                    "kind": "component",
                    "value": name,
                    "basis": name,
                    "grounded": True,
                    "source": "current_query",
                }
                for name in ("甲域", "乙域", "丙域")
            ],
        }
        rows = self._query(
            [query, "甲域", "乙域别称", "丙域别称"],
            query_frame=frame,
        )

        self.assertEqual(
            ["R1", "R2", "R3"],
            [row["revision_id"] for row in rows],
        )
        self.assertTrue(all(row.get("category_context") for row in rows[1:]))

    def test_coverage_request_can_authorize_family_completion(self):
        query = "对第一工程模型所属资料族做闭合集合核验。"
        frame = {
            "original_query": query,
            "coverage_request": {
                "scope": "complete_collection",
                "subject": "第一工程模型所属资料族",
                "member_kind": "record",
                "reference_mode": "current_query",
                "basis": "第一工程模型所属资料族做闭合集合核验",
            },
        }
        rows = self._query(
            [query, "甲域", "乙域别称", "丙域别称"],
            query_frame=frame,
        )

        self.assertEqual(
            ["R1", "R2", "R3"],
            [row["revision_id"] for row in rows],
        )

    def test_ungrounded_multi_subject_objects_do_not_authorize_completion(self):
        query = "核验甲域。"
        frame = {
            "original_query": query,
            "subject_bindings": [
                {
                    "doc_type": "rbd",
                    "kind": "component",
                    "value": name,
                    "basis": name,
                    "grounded": True,
                    "source": "current_query",
                }
                for name in ("甲域", "伪造乙域", "伪造丙域")
            ],
        }
        rows = self._query(
            [query, "甲域", "乙域别称", "丙域别称"],
            query_frame=frame,
        )

        self.assertEqual(["R1"], [row["revision_id"] for row in rows])


class QuerySlotModelAuthorityContractTests(unittest.TestCase):
    def test_unlisted_natural_phrases_can_support_known_model_enums(self):
        query = "设备处在算力饱和阶段，隔三次任务便再现一次瞬断。"
        frame = query_understanding.build_query_frame(
            query=query,
            planner_frame={
                "operating_stage": "高负载",
                "operating_stage_basis": "算力饱和阶段",
                "temporal_pattern": "重复发生",
                "temporal_pattern_basis": "隔三次任务便再现一次",
            },
        )

        self.assertEqual("高负载", frame["operating_stage"])
        self.assertEqual("算力饱和阶段", frame["operating_stage_basis"])
        self.assertEqual("重复发生", frame["temporal_pattern"])
        self.assertEqual(
            "隔三次任务便再现一次", frame["temporal_pattern_basis"]
        )

    def test_validation_depends_on_enum_and_verbatim_basis_not_synonym_table(self):
        cases = (
            (
                "任务峰段资源占满时发生异常。",
                "高负载",
                "任务峰段资源占满时",
                "operating_stage",
            ),
            (
                "故障呈现有一阵没一阵的节奏。",
                "间歇发生",
                "有一阵没一阵",
                "temporal_pattern",
            ),
        )
        for query, value, basis, slot in cases:
            with self.subTest(query=query, slot=slot):
                planner_frame = {
                    "operating_stage": "未知",
                    "operating_stage_basis": "",
                    "temporal_pattern": "未知",
                    "temporal_pattern_basis": "",
                    slot: value,
                    f"{slot}_basis": basis,
                }
                frame = query_understanding.build_query_frame(
                    query=query,
                    planner_frame=planner_frame,
                )
                self.assertEqual(value, frame[slot])
                self.assertEqual(basis, frame[f"{slot}_basis"])

    def test_unknown_slot_enums_fail_closed_even_with_verbatim_basis(self):
        query = "巡航段偶尔出现瞬断。"
        frame = query_understanding.build_query_frame(
            query=query,
            planner_frame={
                "operating_stage": "巡航段",
                "operating_stage_basis": "巡航段",
                "temporal_pattern": "随机发生",
                "temporal_pattern_basis": "偶尔出现",
            },
        )

        self.assertEqual("未知", frame["operating_stage"])
        self.assertEqual("", frame["operating_stage_basis"])
        self.assertEqual("未知", frame["temporal_pattern"])
        self.assertEqual("", frame["temporal_pattern_basis"])

    def test_noncontiguous_basis_fails_closed_before_semantic_use(self):
        query = "设备进入高、负载工作，并反复、出现短时掉电。"
        frame = query_understanding.build_query_frame(
            query=query,
            planner_frame={
                "operating_stage": "高负载",
                "operating_stage_basis": "高负载",
                "temporal_pattern": "重复发生",
                "temporal_pattern_basis": "反复出现",
            },
        )

        self.assertEqual("未知", frame["operating_stage"])
        self.assertEqual("", frame["operating_stage_basis"])
        self.assertEqual("未知", frame["temporal_pattern"])
        self.assertEqual("", frame["temporal_pattern_basis"])


if __name__ == "__main__":
    unittest.main()
