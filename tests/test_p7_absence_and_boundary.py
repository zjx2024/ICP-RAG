"""P7 缺席声明门禁与能力边界出口的合同测试。

假缺席（证据已有而答案声称缺失/拒绝回答）与资料能力边界陈述
（无工程来源但合法）是两类生成-审核协议级约束：本测试锁定提示
协议中对应义务的存在与关键措辞，防止后续漂移静默弱化。
"""

import unittest

from backend.agent.answer_review import REVIEW_INSTRUCTION
from backend.agent.nodes.answer import (
    ANSWER_PROMPT,
    REVISION_PROMPT_HEADER,
    _answer_task_instruction,
)


class AbsenceClaimGateTests(unittest.TestCase):
    """Fix A：缺席声明必须先核对来源。"""

    def test_answer_prompt_requires_verification_before_absence(self):
        self.assertIn("7.1", ANSWER_PROMPT)
        self.assertIn("逐项核对", ANSWER_PROMPT)
        self.assertIn("不得用缺席声明", ANSWER_PROMPT)
        self.assertIn("假缺席", ANSWER_PROMPT)

    def test_review_instruction_defines_false_absence_obligation(self):
        self.assertIn("缺席声明必须核对", REVIEW_INSTRUCTION)
        self.assertIn("假缺席", REVIEW_INSTRUCTION)
        self.assertIn("逐字查找", REVIEW_INSTRUCTION)

    def test_revision_header_constrains_false_absence_revision(self):
        self.assertIn("声称缺失", REVISION_PROMPT_HEADER)
        self.assertIn("改为引用该证据作答", REVISION_PROMPT_HEADER)
        self.assertIn("资料未提供", REVISION_PROMPT_HEADER)

    def test_revision_header_keeps_original_constraint_codes(self):
        for code in (
            "unsupported_unit",
            "unbound_exact_identity",
            "wrong_nonengineering_provenance",
            "unknown_draft_citation",
            "unknown_source_handle",
        ):
            self.assertIn(code, REVISION_PROMPT_HEADER)


class CapabilityBoundaryExitTests(unittest.TestCase):
    """Fix C：资料能力边界陈述有合法证明路径。"""

    def test_answer_prompt_answers_capability_boundary_by_semantics(self):
        self.assertIn("10.", ANSWER_PROMPT)
        self.assertIn("能证明什么", ANSWER_PROMPT)
        self.assertIn("不要求工程引用", ANSWER_PROMPT)

    def test_review_instruction_allows_policy_for_capability_boundary(self):
        self.assertIn("policy:document_semantics", REVIEW_INSTRUCTION)
        self.assertIn("不得以缺少工程来源为由拒绝", REVIEW_INSTRUCTION)


class RbdTerminalStatementGuardTests(unittest.TestCase):
    """Fix B：终态否定不被“查询为空”误读推翻。"""

    def _state(self):
        return {
            "query": "哪些结论能说？",
            "query_frame": {
                "required_answer_fields": [],
                "source_obligations": [],
                "coverage_request": {},
            },
            "evidence_registry": [],
            "safety_decision": {},
            "calculation_requests": [],
            "tool_results": [],
            "rbd_query_binding": {
                "contract_version": "rbd-authority-v1",
                "required": True,
                "bound": False,
                "basis": "current_model_selection_not_represented",
                "candidate_record_ids": ["revision_id:a"],
                "current_selection": {
                    "status": "not_represented",
                    "reason": "候选中未出现该部件",
                },
            },
        }

    def test_statement_forbids_query_empty_override(self):
        instruction = _answer_task_instruction(self._state())
        self.assertIn("0 行输出", instruction)
        self.assertIn("不是检索失败或资料缺失", instruction)
        self.assertIn("推翻、质疑或淡化", instruction)


class RbdObservationPermissionTests(unittest.TestCase):
    """Fix D：rbd_representation observation 可支持终态否定句。

    终态否定（部件未被独立建模）没有工程记录可绑——它本身是本轮
    确定性核对结论。审核必须允许该 observation 支持否定句的条件分析，
    否则合法答案会被 engineering_claim_without_evidence 反复拒绝。
    """

    def test_boundary_observation_set_includes_rbd_representation(self):
        import inspect
        from backend.agent import answer_review

        source = inspect.getsource(answer_review)
        self.assertIn('"rbd_representation"', source)
        # 类别集合与豁免逻辑同处一个表达式（boundary_observation）。
        self.assertIn("boundary_observation", source)

    def test_review_instruction_grants_observation_permission(self):
        self.assertIn("rbd_representation", REVIEW_INSTRUCTION)
        self.assertIn("不得因缺少工程记录绑定而拒绝", REVIEW_INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
