import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent.nodes import plan as plan_module
from backend.agent.tools import icp_diagnosis as diagnosis_module


def _run(coro):
    return asyncio.run(coro)


class StructuredIdentityPlanContractTests(unittest.TestCase):
    def _plan(self, query, target, planner):
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
                        "messages": [{"role": "user", "content": query}],
                        "memory_enabled": False,
                        "iteration": 0,
                    }
                )
            )

    @staticmethod
    def _target(intent, **values):
        return {
            "primary_intent": intent,
            "targets": [intent],
            "contextualized_query": values.pop("query"),
            "subjects": [],
            "requested_facets": [],
            "required_answer_fields": values.pop("required_answer_fields", []),
            "question_semantics": "describe",
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
            **values,
        }

    @staticmethod
    def _planner(intent, query, **frame):
        tool = "hmc_query" if intent == "exact_code" else (
            "rbd_query" if intent == "reliability_query" else "fmeca_query"
        )
        return {
            "intent": intent,
            "query_targets": [{"intent": intent}],
            "rewritten_query": query,
            "query_frame": frame,
            "tool_queries": {tool: [query]},
            "tools_plan": [{"tool": tool, "reason": "读取权威记录"}],
            "relation_expansion": {"needed": False, "reason": ""},
        }

    def test_hmc_slot_is_derived_from_raw_turn_not_model_mapping(self):
        query = "核对 LCN 91810001 HSC 1"
        result = self._plan(
            query,
            self._target(
                "exact_code",
                query=query,
                required_answer_fields=["hmc_identity"],
            ),
            self._planner(
                "exact_code",
                query,
                hmc_components={"lcn": "99999999", "hsc": "9"},
            ),
        )

        self.assertEqual(
            {"lcn": "91810001", "hsc": "0001"},
            result["query_frame"]["hmc_components"],
        )
        self.assertEqual(
            "current_query", result["query_frame"]["hmc_components_source"]
        )
        self.assertEqual(
            {"lcn": "91810001", "hsc": "0001"},
            diagnosis_module._typed_hmc_components({"query": query, **result}),
        )

    def test_bare_number_cannot_become_hmc_through_planner_slot(self):
        query = "任务批次号 91810001 持续 10 小时"
        result = self._plan(
            query,
            self._target("exact_code", query=query),
            self._planner(
                "exact_code",
                query,
                hmc_components={"lcn": "91810001", "hsc": "1"},
            ),
        )

        self.assertNotIn("hmc_components", result["query_frame"])

    def test_typed_record_fields_require_literal_current_turn_value(self):
        query = "入口滤波性能退化会有什么影响？"
        result = self._plan(
            query,
            self._target(
                "impact_analysis",
                query=query,
                required_answer_fields=["local_impact", "final_impact"],
                typed_record_fields={
                    "fmeca": {
                        "failure_mode": "入口滤波性能退化",
                        "fault_cause": "模型凭空补出的原因",
                    }
                },
            ),
            self._planner("impact_analysis", query),
        )

        self.assertEqual(
            {"fmeca": {"failure_mode": "入口滤波性能退化"}},
            result["query_frame"]["typed_record_fields"],
        )

    def test_subject_binding_requires_verbatim_basis_and_marks_provenance(self):
        query = "输入端滤波器是否位于 RBD 模型中？"
        result = self._plan(
            query,
            self._target(
                "reliability_query",
                query=query,
                subject_bindings=[
                    {
                        "doc_type": "rbd",
                        "kind": "component",
                        "value": "输入端滤波器",
                        "basis": "输入端滤波器",
                    },
                    {
                        "doc_type": "rbd",
                        "kind": "component",
                        "value": "显示处理模块",
                        "basis": "显示处理模块",
                    },
                ],
            ),
            self._planner("reliability_query", query),
        )

        self.assertEqual(1, len(result["query_frame"]["subject_bindings"]))
        binding = result["query_frame"]["subject_bindings"][0]
        self.assertEqual("输入端滤波器", binding["value"])
        self.assertTrue(binding["grounded"])
        self.assertEqual("current_query", binding["source"])
        _, authority = diagnosis_module._bind_requested_rbd_rows(
            [
                {
                    "diagram_type": "rbd",
                    "revision_id": "rev-input-filter",
                    "diagram_key": "RBD-INPUT-FILTER",
                    "validation_status": "auto_validated",
                    "projection_status": "published",
                    "is_active": True,
                    "graph": {
                        "vertices": [
                            {"properties": {"name": "输入端滤波器"}}
                        ]
                    },
                }
            ],
            requested=True,
            literal_subject=binding["value"],
        )
        self.assertFalse(authority["bound"])


if __name__ == "__main__":
    unittest.main()
