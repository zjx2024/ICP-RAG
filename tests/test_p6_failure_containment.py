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
from backend.agent.rbd_contract import (
    RBD_NOT_REPRESENTED_BASE,
    RBD_TERMINAL_NEGATION_ANCHOR_VERSION,
    build_rbd_terminal_negation_anchor,
    reinstate_rbd_terminal_negation_anchor,
)


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
        self.assertIn("hmc_identity", text)
        self.assertIn("91810001", text)
        # 清单不是工程结论，结尾引导补充信息后重新检索。
        self.assertIn("重新检索", text)

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
        self.assertIn("rbd_model_identity", text)
        self.assertIn("对象身份未解析", text)
        self.assertIn("formal_calculation", text)

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
            "missing_input_requirements": [_slot_requirement(slot="from_state")],
            "safety_decision": {
                "missing_input_requirements": [_slot_requirement(slot="from_safety")]
            },
        }
        text = render_clarification_only_answer(state)
        self.assertIn("from_safety", text)
        self.assertNotIn("from_state", text)


def _terminal_binding(fingerprint=("revision_id:rbd-a", "revision_id:rbd-b")):
    return {
        "contract_version": "rbd-authority-v1",
        "required": True,
        "bound": False,
        "ambiguous": False,
        "basis": RBD_NOT_REPRESENTED_BASE,
        "record_ids": [],
        "candidate_record_ids": list(fingerprint),
        "model_bindings": [],
        "current_selection": {
            "contract_version": "current-model-selection/v1",
            "status": "not_represented",
            "question_basis": "输入端滤波器变差时，现有框图有没有直接写出它所在的成功路径",
            "selections": [],
            "reason": "候选 RBD 框图组件列表中均未出现该部件",
        },
        "not_represented_subjects": ["输入端滤波器"],
        "published_universe_fingerprint": list(fingerprint),
    }


class RbdTerminalNegationAnchorTests(unittest.TestCase):
    """Fix 3：终态否定锚点的构建、持久化与确定性复核恢复。"""

    def test_build_anchor_from_terminal_binding(self):
        anchor = build_rbd_terminal_negation_anchor(
            _terminal_binding(),
            published_universe=["revision_id:rbd-a", "revision_id:rbd-b"],
        )
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor["kind"], "terminal_negation_anchor")
        self.assertEqual(anchor["doc_type"], "rbd")
        self.assertEqual(
            anchor["contract_version"], RBD_TERMINAL_NEGATION_ANCHOR_VERSION
        )
        self.assertIn("输入端滤波器", anchor["subject"])
        self.assertEqual(
            sorted(anchor["verified_universe"]),
            ["revision_id:rbd-a", "revision_id:rbd-b"],
        )
        self.assertEqual(
            anchor["current_selection"]["status"], "not_represented"
        )

    def test_build_anchor_rejects_non_terminal_binding(self):
        binding = _terminal_binding()
        binding["basis"] = "current_model_selection_unresolved"
        self.assertIsNone(
            build_rbd_terminal_negation_anchor(
                binding, published_universe=["revision_id:rbd-a"]
            )
        )

    def test_build_anchor_requires_nonempty_universe(self):
        self.assertIsNone(
            build_rbd_terminal_negation_anchor(
                _terminal_binding(), published_universe=[]
            )
        )

    def test_reinstate_with_matching_universe(self):
        anchor = build_rbd_terminal_negation_anchor(
            _terminal_binding(),
            published_universe=["revision_id:rbd-a", "revision_id:rbd-b"],
        )
        binding = reinstate_rbd_terminal_negation_anchor(
            anchor,
            published_universe=["revision_id:rbd-b", "revision_id:rbd-a"],
        )
        self.assertIsNotNone(binding)
        self.assertEqual(binding["basis"], RBD_NOT_REPRESENTED_BASE)
        self.assertFalse(binding["ambiguous"])
        self.assertEqual(
            binding["current_selection"]["status"], "not_represented"
        )
        # 恢复绑定带锚点来源，答案合同据此注明"复核一致"。
        self.assertEqual(
            binding["anchor_provenance"]["kind"],
            "prior_turn_terminal_negation",
        )
        # 恢复绑定同样携带指纹，支持跨多轮连续再持久化。
        self.assertEqual(
            sorted(binding["published_universe_fingerprint"]),
            ["revision_id:rbd-a", "revision_id:rbd-b"],
        )

    def test_reinstate_rejects_changed_universe(self):
        """新发布的修订使指纹失配：世界已变，旧判定不能沿用。"""
        anchor = build_rbd_terminal_negation_anchor(
            _terminal_binding(),
            published_universe=["revision_id:rbd-a", "revision_id:rbd-b"],
        )
        self.assertIsNone(
            reinstate_rbd_terminal_negation_anchor(
                anchor,
                published_universe=[
                    "revision_id:rbd-a",
                    "revision_id:rbd-b",
                    "revision_id:rbd-new",
                ],
            )
        )
        # 撤回同理。
        self.assertIsNone(
            reinstate_rbd_terminal_negation_anchor(
                anchor, published_universe=["revision_id:rbd-a"]
            )
        )

    def test_reinstate_rejects_malformed_anchor(self):
        self.assertIsNone(
            reinstate_rbd_terminal_negation_anchor(
                None, published_universe=["revision_id:rbd-a"]
            )
        )
        self.assertIsNone(
            reinstate_rbd_terminal_negation_anchor(
                {"kind": "goal_delivery"},
                published_universe=["revision_id:rbd-a"],
            )
        )
        # 缺失 current_selection 的终态形状。
        anchor = build_rbd_terminal_negation_anchor(
            _terminal_binding(), published_universe=["revision_id:rbd-a"]
        )
        del anchor["current_selection"]
        self.assertIsNone(
            reinstate_rbd_terminal_negation_anchor(
                anchor, published_universe=["revision_id:rbd-a"]
            )
        )


class ManifestAnchorTests(unittest.TestCase):
    """锚点随 manifest 持久化与追问轮读取。"""

    def test_manifest_appends_anchor_for_rbd_answer(self):
        from backend.agent.nodes.answer import (
            _manifest_with_terminal_negation_anchor,
        )

        manifest = [{"kind": "goal_delivery", "goal": {}, "status": "answered"}]
        state = {"rbd_query_binding": _terminal_binding()}
        result = _manifest_with_terminal_negation_anchor(
            manifest, state, "现有 RBD 框图中没有直接写出该部件的成功路径。"
        )
        anchors = [
            e for e in result if e.get("kind") == "terminal_negation_anchor"
        ]
        self.assertEqual(len(anchors), 1)
        self.assertEqual(len(result), 2)

    def test_manifest_skips_off_topic_answer(self):
        from backend.agent.nodes.answer import (
            _manifest_with_terminal_negation_anchor,
        )

        state = {"rbd_query_binding": _terminal_binding()}
        result = _manifest_with_terminal_negation_anchor(
            [{"kind": "goal_delivery"}], state, "今天天气不错。"
        )
        self.assertEqual(len(result), 1)

    def test_manifest_skips_non_terminal_binding(self):
        from backend.agent.nodes.answer import (
            _manifest_with_terminal_negation_anchor,
        )

        binding = _terminal_binding()
        binding["basis"] = "current_model_selection_unresolved"
        state = {"rbd_query_binding": binding}
        result = _manifest_with_terminal_negation_anchor(
            [{"kind": "goal_delivery"}], state, "RBD 查询结果为空。"
        )
        self.assertEqual(len(result), 1)

    def test_prior_anchor_read_from_messages(self):
        from backend.agent.tools.icp_diagnosis import (
            _prior_rbd_terminal_negation_anchor,
        )

        anchor = build_rbd_terminal_negation_anchor(
            _terminal_binding(), published_universe=["revision_id:rbd-a"]
        )
        state = {
            "messages": [
                {"role": "user", "content": "上一问"},
                {
                    "role": "assistant",
                    "content": "RBD 未直接写出该部件路径。",
                    "answer_delivery_manifest": [anchor],
                },
            ]
        }
        read = _prior_rbd_terminal_negation_anchor(state)
        self.assertEqual(read["kind"], "terminal_negation_anchor")

    def test_prior_anchor_none_without_manifest(self):
        from backend.agent.tools.icp_diagnosis import (
            _prior_rbd_terminal_negation_anchor,
        )

        self.assertIsNone(_prior_rbd_terminal_negation_anchor({}))
        self.assertIsNone(
            _prior_rbd_terminal_negation_anchor(
                {"messages": [{"role": "assistant", "content": "普通回答"}]}
            )
        )


class GuidanceIsolationTests(unittest.TestCase):
    """Fix 4：指引句只在推导关联存在时进入合同（共享面隔离）。"""

    def test_instruction_without_relation_has_no_guidance(self):
        from backend.agent.nodes.answer import _answer_task_instruction

        state = {
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
        instruction = _answer_task_instruction(state)
        self.assertNotIn("derived_source_relation_guidance", instruction)
        self.assertNotIn("不得写成报码级显式登记", instruction)

    def test_reinstated_binding_statement_mentions_recheck(self):
        from backend.agent.nodes.answer import _answer_task_instruction

        anchor = build_rbd_terminal_negation_anchor(
            _terminal_binding(), published_universe=["revision_id:rbd-a"]
        )
        binding = reinstate_rbd_terminal_negation_anchor(
            anchor, published_universe=["revision_id:rbd-a"]
        )
        state = {
            "query": "哪些结论能说，哪些不能说？",
            "query_frame": {
                "required_answer_fields": [],
                "source_obligations": [],
                "coverage_request": {},
            },
            "evidence_registry": [],
            "safety_decision": {},
            "calculation_requests": [],
            "tool_results": [],
            "rbd_query_binding": binding,
        }
        instruction = _answer_task_instruction(state)
        self.assertIn("rbd_not_represented", instruction)
        self.assertIn("确定性复核", instruction)
        self.assertIn("结论仍然成立", instruction)


if __name__ == "__main__":
    unittest.main()
