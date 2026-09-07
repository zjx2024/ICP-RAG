import json
import re
import unittest

from backend.agent.nodes.answer import _answer_task_instruction, _answer_review_material


def _contract_json(instruction: str) -> dict:
    """Extract the JSON contract object from the task instruction text."""

    match = re.search(r"【本轮任务与来源边界（内部材料，不得原样输出）】\n(\{.*?\})\n", instruction, re.S)
    assert match, "contract JSON not found"
    return json.loads(match.group(1))


def _base_state():
    return {
        "query": "这个报码在维修清单里有处理办法吗？",
        "query_frame": {
            "required_answer_fields": ["maintenance_action"],
            "source_obligations": [],
            "coverage_request": {},
        },
        "evidence_registry": [],
        "safety_decision": {},
        "calculation_requests": [],
        "tool_results": [],
    }


class DerivedSourceRelationTests(unittest.TestCase):
    def test_structured_identity_binding_injects_derived_qualifier(self):
        """结构化身份链绑定时，任务合同必须声明推导性质边界。"""
        state = _base_state()
        state["hmc_mfl_binding"] = {
            "required": True,
            "bound": True,
            "basis": "structured_component_identity",
            "hmc_codes": ["91810001**0001"],
            "mfl_codes": ["MFL-SYN-001"],
        }
        contract = _contract_json(_answer_task_instruction(state))
        self.assertIn("derived_source_relation", contract)
        self.assertIn("推导", contract["derived_source_relation"]["meaning"])
        # 指引句在合同存在时才解释推导边界。
        self.assertIn("不得写成报码级显式登记", _answer_task_instruction(state))

    def test_explicit_reference_binding_has_no_derived_qualifier(self):
        """显式引用绑定的关联是权威登记，合同不需要推导限定符。"""
        state = _base_state()
        state["hmc_mfl_binding"] = {
            "required": True,
            "bound": True,
            "basis": "explicit_authoritative_reference",
            "hmc_codes": ["91810001**0001"],
        }
        contract = _contract_json(_answer_task_instruction(state))
        self.assertNotIn("derived_source_relation", contract)
        self.assertNotIn("unconfirmed_source_relation", contract)

    def test_unbound_relation_keeps_unconfirmed_contract(self):
        """未绑定路径行为不变：仍是 unconfirmed_source_relation。"""
        state = _base_state()
        state["hmc_mfl_binding"] = {
            "required": True,
            "bound": False,
            "basis": "",
            "hmc_codes": ["91810001**0001"],
        }
        contract = _contract_json(_answer_task_instruction(state))
        self.assertIn("unconfirmed_source_relation", contract)
        self.assertNotIn("derived_source_relation", contract)

    def test_review_observation_carries_binding_basis(self):
        """审核观察携带 basis，供裁判核对推导关联措辞。"""
        state = _base_state()
        state["hmc_mfl_binding"] = {
            "required": True,
            "bound": True,
            "basis": "structured_component_identity",
            "hmc_codes": ["91810001**0001"],
        }
        _, _, observations = _answer_review_material(state)
        relation_obs = [
            o for o in observations if o.get("category") == "source_relation"
        ]
        self.assertTrue(relation_obs)
        fields = relation_obs[0]["fields"]
        self.assertIn("/basis", fields)
        self.assertEqual("structured_component_identity", fields["/basis"])
        self.assertIn("推导", fields.get("/meaning", ""))


class RbdNotRepresentedWordingTests(unittest.TestCase):
    def _not_represented_state(self):
        state = _base_state()
        state["rbd_query_binding"] = {
            "contract_version": "rbd-authority-v1",
            "required": True,
            "bound": False,
            "basis": "current_model_selection_not_represented",
            "candidate_record_ids": ["revision_id:a", "revision_id:b"],
            "current_selection": {
                "status": "not_represented",
                "reason": "候选 RBD 框图组件列表中均未出现该部件",
            },
        }
        return state

    def test_statement_distinguishes_module_level_from_direct_path(self):
        """否定语句须区分模块级建模与部件直接路径，且不再索要图号。"""
        instruction = _answer_task_instruction(self._not_represented_state())
        self.assertIn("rbd_not_represented", instruction)
        self.assertIn("未被独立建为功能块", instruction)
        self.assertIn("模块级建模", instruction)
        self.assertIn("不再索要图号或标识", instruction)

    def test_review_observation_marks_terminal_state(self):
        _, _, observations = _answer_review_material(self._not_represented_state())
        obs = [o for o in observations if o.get("category") == "rbd_representation"]
        self.assertTrue(obs)
        fields = obs[0]["fields"]
        self.assertEqual("not_represented", fields.get("/status"))
        self.assertIn("模块级模型不等于该部件直接路径", fields.get("/meaning", ""))

    def test_clarification_slots_skip_terminal_negation(self):
        """not_represented 终态不再生成 rbd_model_identity 未解析槽。

        该槽与确定性否定结论矛盾（否定是终态，不是待消解缺口）；
        真正的 unresolved 绑定仍生成槽。
        """
        from backend.agent.tools.icp_diagnosis import (
            _rbd_binding_needs_identity_clarification,
        )
        # not_represented：不需要身份澄清槽。
        self.assertFalse(
            _rbd_binding_needs_identity_clarification(
                self._not_represented_state()["rbd_query_binding"]
            )
        )
        # unresolved：仍需要该槽。
        self.assertTrue(
            _rbd_binding_needs_identity_clarification({
                "required": True,
                "bound": False,
                "basis": "current_model_selection_unresolved",
                "candidate_record_ids": ["revision_id:a"],
            })
        )
        # 已绑定：不需要。
        self.assertTrue(
            not _rbd_binding_needs_identity_clarification({
                "required": True,
                "bound": True,
                "basis": "current_model_selection",
            })
        )


if __name__ == "__main__":
    unittest.main()
