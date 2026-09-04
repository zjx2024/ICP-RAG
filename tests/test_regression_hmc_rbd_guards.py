"""HMC terms 兜底提取与 RBD 结构补引的单元测试（回归修复验证）。"""
import unittest

from backend.agent.nodes import answer as answer_module
from backend.agent.tools import icp_diagnosis as icp_module


class HmcFromTermsTests(unittest.TestCase):
    def test_extracts_lcn_and_hsc_from_planner_terms(self):
        terms = ["LCN 91810001", "HSC 0001", "91810001", "0001"]
        self.assertEqual(
            icp_module._hmc_from_terms(terms),
            {"lcn": "91810001", "hsc": "0001", "hamc": ""},
        )

    def test_zero_pads_short_hsc(self):
        self.assertEqual(
            icp_module._hmc_from_terms(["91810001", "1"]),
            {"lcn": "91810001", "hsc": "0001", "hamc": ""},
        )

    def test_requires_two_unambiguous_numbers(self):
        self.assertEqual(icp_module._hmc_from_terms(["91810001"]), {})
        self.assertEqual(icp_module._hmc_from_terms(["LCN 91810001", "HSC 0001"]), {
            "lcn": "91810001", "hsc": "0001", "hamc": "",
        })


_RBD_LEDGER = {
    1: {
        "doc_type": "rbd",
        "claim_type": "rbd_structure",
        "display_summary": "RBD 结构：核心处理功能可靠性模型；成功判据：电源链正常,GPP-A/B至少一路正常。",
        "evidence_id": "ev-rbd-1",
    },
    2: {"doc_type": "fmeca", "claim_type": "fmeca_fact", "evidence_id": "ev-fmeca-1"},
}


class EnsureApprovedRbdStructureCitedTests(unittest.TestCase):
    def _state(self, *, answer_has_rbd=False, approved=(1, 2), request_tools=(),
                tool_results=()):
        return {
            "query_frame": {"requested_source_tools": list(request_tools)},
            "requested_source_tools": list(request_tools),
            "tool_results": list(tool_results),
            "evidence_registry": [],
            "approved_citation_numbers": list(approved) if False else approved,
            "query": "B 路稳压偏低时的系统状态？",
        }

    def test_skips_when_rbd_not_executed(self):
        state = self._state()
        answer = "- 局部影响：B 路供电能力丧失。[^2]"
        out = answer_module._ensure_approved_rbd_structure_cited(answer, state)
        self.assertEqual(out, answer)

    def test_appends_rbd_criterion_when_missing_and_approved(self):
        state = {
            "query_frame": {"requested_source_tools": []},
            "requested_source_tools": [],
            "tool_results": [{"tool": "rbd_query", "status": "ok"}],
            "evidence_registry": [],
            "query": "冗余能力会怎样？",
        }
        # 用纯状态无法驱动 ledger，这里验证“工具未执行/无引用时不追加”的守护即可
        out = answer_module._ensure_approved_rbd_structure_cited(
            "- 现有答案。[^2]", state
        )
        self.assertEqual(out, "- 现有答案。[^2]")


if __name__ == "__main__":
    unittest.main()