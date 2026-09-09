"""P6 失效遏制机制测试：审核 JSON 容错、澄清轮确定性发布、RBD 终态锚点。

三类修复都作用于"模型路径失败后"的最后防线：解析层救回有效审核、
澄清缺口确定性渲染、终态否定跨轮复核。它们不得改变主路径行为——
正常审核、正常答案、正常绑定的输入输出与修复前逐字节一致。
"""

import json
import unittest

from backend.agent.answer_review import _extract_json_payload
from backend.agent.clarification_contract import (
    make_clarification_identity,
    make_clarification_reason,
    make_missing_input_requirement,
)
from backend.agent.nodes.answer_request import render_clarification_only_answer


class ReviewJsonExtractionTests(unittest.TestCase):
    """Fix 1：审核输出的确定性围栏/前置文字剥离。"""

    def test_fenced_json_is_extracted(self):
        raw = '```json\n{"version": 2, "units": [], "goals": []}\n```'
        self.assertEqual(
            _extract_json_payload(raw),
            '{"version": 2, "units": [], "goals": []}',
        )

    def test_bare_fence_is_extracted(self):
        raw = '```\n{"version": 2}\n```'
        self.assertEqual(_extract_json_payload(raw), '{"version": 2}')

    def test_leading_remark_is_stripped(self):
        raw = '好的，以下是审核结果：{"version": 2, "units": []}'
        self.assertEqual(
            _extract_json_payload(raw), '{"version": 2, "units": []}'
        )

    def test_trailing_text_is_dropped(self):
        raw = '{"version": 2} 以上就是全部审核。'
        self.assertEqual(_extract_json_payload(raw), '{"version": 2}')

    def test_braces_inside_strings_do_not_break_balance(self):
        payload = '{"issue": "该段声称 {模块} 与 {报码} 存在显式登记"}'
        raw = f'审核如下：{payload}请修订。'
        self.assertEqual(_extract_json_payload(raw), payload)

    def test_plain_json_unchanged(self):
        raw = '{"version": 2, "units": [], "goals": []}'
        self.assertEqual(_extract_json_payload(raw), raw)

    def test_unbalanced_braces_report_truncation_tail(self):
        # 截断的 JSON 返回尾部使 json.loads 报出真实截断错误，
        # 而不是被前置文字误导成别的错误。
        raw = '说明：{"version": 2, "units": ['
        self.assertEqual(_extract_json_payload(raw), '{"version": 2, "units": [')

    def test_no_json_at_all_returns_text(self):
        raw = '模型完全没输出 JSON'
        self.assertEqual(_extract_json_payload(raw), '模型完全没输出 JSON')

    def test_extracted_payload_parses(self):
        raw = (
            '审核结果如下\n```json\n'
            '{"version":2,"units":[{"unit_id":"u1","issue":"ok"}],"goals":[]}\n'
            '```\n完毕'
        )
        parsed = json.loads(_extract_json_payload(raw))
        self.assertEqual(parsed["version"], 2)
        self.assertEqual(parsed["units"][0]["unit_id"], "u1")


def _slot_requirement(slot="hmc_identity", kind="identity_slot"):
    return make_missing_input_requirement(
        kind=kind,
        slot=slot,
        source_kind="state",
        source_name="hmc_query_binding",
        identity=make_clarification_identity(
            doc_type="hmc",
            record_ids=["91810001**0001"],
            fields={"lcn": "91810001", "hsc": "0001"},
        ),
        basis="原始问题给出的部件代码在 HMC 中无精确命中",
    )


class ClarificationOnlyRenderTests(unittest.TestCase):
    """Fix 2：澄清缺口确定性渲染。"""

    def test_missing_requirement_renders_list(self):
        state = {"missing_input_requirements": [_slot_requirement()]}
        text = render_clarification_only_answer(state)
        self.assertTrue(text)
        self.assertIn("还需要补充以下信息", text)
        self.assertIn("对象身份", text)
        self.assertIn("完整原始报码", text)
        self.assertNotIn("hmc_identity", text)
        self.assertIn("91810001", text)
        # 清单不是工程结论，结尾引导补充信息后重新检索。
        self.assertIn("重新检索", text)

    def test_unreviewed_free_text_basis_is_not_published_as_a_fact(self):
        requirement = _slot_requirement()
        requirement["basis"] = "已确认唯一故障，必须立即更换整机。"
        text = render_clarification_only_answer({"missing_input_requirements": [requirement]})
        self.assertNotIn(requirement["basis"], text)

    def test_unresolved_slot_and_reason_render(self):
        state = {
            "unresolved_identity_slots": [_slot_requirement(slot="rbd_model_identity")],
            "clarification_reasons": [
                make_clarification_reason(
                    kind="unresolved_identity",
                    source_kind="binding",
                    source_name="rbd_query_binding",
                    identity=make_clarification_identity(doc_type="rbd"),
                    blocked_claims=["rbd_model_identity", "formal_calculation"],
                )
            ],
        }
        text = render_clarification_only_answer(state)
        self.assertIn("可靠性框图名称", text)
        self.assertIn("对象身份未解析", text)
        self.assertNotIn("formal_calculation", text)

    def test_empty_contracts_render_nothing(self):
        """没有已校验缺口时必须返回空串：不得发明澄清。"""
        self.assertEqual(render_clarification_only_answer({}), "")
        self.assertEqual(
            render_clarification_only_answer({"missing_input_requirements": []}), ""
        )

    def test_malformed_contracts_render_nothing(self):
        self.assertEqual(
            render_clarification_only_answer(
                {"missing_input_requirements": [{"slot": "x"}]}
            ),
            "",
        )

    def test_safety_decision_preferred_over_state(self):
        state = {
            "missing_input_requirements": [_slot_requirement(slot="mission_time")],
            "safety_decision": {
                "missing_input_requirements": [_slot_requirement(slot="hamc")]
            },
        }
        text = render_clarification_only_answer(state)
        self.assertIn("HAMC 字段", text)
        self.assertNotIn("任务持续时间", text)
