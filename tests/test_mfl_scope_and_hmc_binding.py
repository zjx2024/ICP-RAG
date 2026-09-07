import unittest

from backend.agent.nodes.answer_record_scope import _narrow_semantic_table_families
from backend.agent.rbd_contract import RBD_AUTHORITY_CONTRACT_VERSION
from backend.agent.rbd_contract import RBD_NOT_REPRESENTED_BASE
from backend.agent.rbd_contract import bind_rbd_models
from backend.agent.rbd_contract import validate_rbd_binding_contract
from backend.agent.tools.icp_diagnosis import (
    _bind_mfl_rows_to_hmc,
    _hmc_fault_mode_segment,
    _mfl_module_fault_segments,
)


def _fmeca_row(evidence_id, record_id, failure_mode, functions=("FUNC-530",),
               match_type="structured_query"):
    return {
        "evidence_id": evidence_id,
        "doc_type": "fmeca",
        "record_id": record_id,
        "authority": "authoritative_record",
        "match_type": match_type,
        "metadata": {"answer_fields": {
            "failure_mode": failure_mode,
            "product_flag": " / ".join(functions),
        }},
    }


def _mfl_row(evidence_id, record_id, module_fault, match_type="structured_query"):
    return {
        "evidence_id": evidence_id,
        "doc_type": "mfl",
        "record_id": record_id,
        "authority": "authoritative_record",
        "match_type": match_type,
        "metadata": {"answer_fields": {
            "module_failure": module_fault,
            "record_code": record_id,
            "api_name": "GetStatus",
            "api_reportable": True,
        }},
    }


def _state(required_fields, approved_ids, anchors=()):
    return {
        "query": "新快照写不进去，资料建议怎么处理？有可用的状态接口吗？",
        "query_frame": {
            "required_answer_fields": list(required_fields),
            "coverage_scope": "unknown",
            "original_query": "新快照写不进去，资料建议怎么处理？有可用的状态接口吗？",
        },
        "safety_decision": {"approved_evidence_ids": list(approved_ids)},
        "structured_record_binding": {},
        "selected_conversation_anchors": list(anchors),
    }


class MflFaultModeRetentionTests(unittest.TestCase):
    def test_approved_mfl_row_sharing_failure_mode_is_retained(self):
        """维修合同字段的权威属主行不因强 FMECA 锚点被整族丢弃。"""
        evidence = [
            _fmeca_row("ev-1", "FM-SYN-061", "故障日志写入失败",
                       match_type="exact_record_identity"),
            _mfl_row("ev-2", "MFL-SYN-061", "故障记录存储模块：故障日志写入失败"),
            _mfl_row("ev-3", "MFL-SYN-099", "显示模块：完全无关的故障"),
        ]
        state = _state(
            ["maintenance_action", "maintenance_api"],
            ["ev-1", "ev-2", "ev-3"],
            anchors=[{"doc_type": "fmeca", "record_id": "FM-SYN-061"}],
        )
        narrowed = _narrow_semantic_table_families(state, evidence)
        kept = {item["record_id"] for item in narrowed}
        self.assertIn("FM-SYN-061", kept)
        self.assertIn("MFL-SYN-061", kept)
        # 无共享故障模式词的行仍按原语义表族规则处理。
        self.assertNotIn("MFL-SYN-099", kept)

    def test_retention_requires_contract_fields(self):
        """无 MFL 合同字段时，收窄行为保持原状（不因本机制扩大保留集）。"""
        evidence = [
            _fmeca_row("ev-1", "FM-SYN-061", "故障日志写入失败",
                       match_type="exact_record_identity"),
            _mfl_row("ev-2", "MFL-SYN-061", "故障记录存储模块：故障日志写入失败"),
        ]
        state = _state(["fault_cause"], ["ev-1", "ev-2"])
        narrowed = _narrow_semantic_table_families(state, evidence)
        self.assertNotIn("MFL-SYN-061", [item["record_id"] for item in narrowed])

    def test_retention_requires_gate_approval(self):
        """未被门禁批准的 MFL 行不能借道本机制进入答案上下文。"""
        evidence = [
            _fmeca_row("ev-1", "FM-SYN-061", "故障日志写入失败",
                       match_type="exact_record_identity"),
            _mfl_row("ev-2", "MFL-SYN-061", "故障记录存储模块：故障日志写入失败"),
        ]
        state = _state(
            ["maintenance_action", "maintenance_api"],
            ["ev-1"],
        )
        narrowed = _narrow_semantic_table_families(state, evidence)
        self.assertEqual(["FM-SYN-061"], [item["record_id"] for item in narrowed])


class HmcMflStructuredIdentityTests(unittest.TestCase):
    def test_module_fault_segments_split_on_fullwidth_colon(self):
        module, mode = _mfl_module_fault_segments("输入调理模块：输入过压钳位失效")
        self.assertEqual("输入调理模块", module)
        self.assertEqual("输入过压钳位失效", mode)

    def test_hmc_fault_mode_segment_after_abbreviation(self):
        self.assertEqual(
            "输入过压钳位失效",
            _hmc_fault_mode_segment({"fault_desc": "ICU 输入过压钳位失效"}),
        )

    def _bind_with_components(self, hmc_rows, mfl_rows, components):
        """Patch the component identity lookup to a fixed abbreviation map."""
        from unittest.mock import patch

        with patch(
            "backend.agent.tools.icp_diagnosis._hmc_structured_component_identity",
            return_value=components,
        ):
            return _bind_mfl_rows_to_hmc(
                hmc_rows,
                mfl_rows,
                requested_components={"hmc_code": "91810001**0001"},
            )

    def test_structured_identity_binds_matching_mfl_row(self):
        """结构化身份链：缩写→模块名 + 故障模式段相等 → 绑定成立。"""
        hmc_rows = [{
            "hmc_code": "91810001**0001",
            "match_type": "exact_components",
            "fault_desc": "ICU 输入过压钳位失效",
        }]
        mfl_rows = [
            {"mfl_code": "MFL-SYN-001", "module_fault": "输入调理模块：输入过压钳位失效"},
            {"mfl_code": "MFL-SYN-002", "module_fault": "输入调理模块：输入滤波性能退化"},
        ]
        bound, binding = self._bind_with_components(
            hmc_rows, mfl_rows, {"ICU": "输入调理模块"}
        )
        self.assertTrue(binding["bound"])
        self.assertEqual("structured_component_identity", binding["basis"])
        self.assertEqual(["MFL-SYN-001"], binding["mfl_codes"])
        self.assertNotIn(
            "MFL-SYN-002",
            [r["mfl_code"] for r in bound if not r.get("query_binding_excluded")],
        )

    def test_identity_chain_requires_both_segments(self):
        """仅模块名相同而故障模式不同不能绑定（防相邻行混入）。"""
        hmc_rows = [{
            "hmc_code": "91810001**0001",
            "match_type": "exact_components",
            "fault_desc": "ICU 输入过压钳位失效",
        }]
        mfl_rows = [
            {"mfl_code": "MFL-SYN-002", "module_fault": "输入调理模块：输入滤波性能退化"},
        ]
        _, binding = self._bind_with_components(
            hmc_rows, mfl_rows, {"ICU": "输入调理模块"}
        )
        self.assertFalse(binding["bound"])
        self.assertEqual(["MFL-SYN-002"], binding["excluded_mfl_codes"])

    def test_empty_component_map_keeps_fail_closed(self):
        """组件解析失败（空映射）时整链失效，回落显式引用/排除路径。"""
        hmc_rows = [{
            "hmc_code": "91810001**0001",
            "match_type": "exact_components",
            "fault_desc": "ICU 输入过压钳位失效",
        }]
        mfl_rows = [
            {"mfl_code": "MFL-SYN-001", "module_fault": "输入调理模块：输入过压钳位失效"},
        ]
        _, binding = self._bind_with_components(hmc_rows, mfl_rows, {})
        self.assertFalse(binding["bound"])
        self.assertEqual("no_explicit_authoritative_relation", binding["basis"])


class RbdNotRepresentedTests(unittest.TestCase):
    def _selection(self):
        return {
            "contract_version": "current-model-selection/v1",
            "status": "not_represented",
            "query_fingerprint": "abc",
            "question_basis": "输入端滤波器",
            "selections": [],
            "reason": "候选 RBD 框图组件列表中均未出现该部件",
        }

    def test_not_represented_is_a_terminal_binding_state(self):
        _, binding = bind_rbd_models(
            [],
            required=True,
            raw_user_query="输入端滤波器变差时，现有框图有没有直接写出它所在的成功路径？",
            current_selection=self._selection(),
        )
        self.assertEqual(RBD_NOT_REPRESENTED_BASE, binding["basis"])
        validation = validate_rbd_binding_contract(binding)
        self.assertTrue(validation["valid"])
        self.assertTrue(validation["not_represented"])
        self.assertEqual([], validation["revision_ids"])

    def test_unresolved_selection_still_fails_closed(self):
        selection = {**self._selection(), "status": "selected", "selections": [{}]}
        _, binding = bind_rbd_models(
            [],
            required=True,
            raw_user_query="输入端滤波器",
            current_selection=selection,
        )
        self.assertEqual("current_model_selection_unresolved", binding["basis"])
        self.assertFalse(validate_rbd_binding_contract(binding)["valid"])

    def test_contract_version_must_match(self):
        selection = {**self._selection(), "contract_version": "other/v9"}
        _, binding = bind_rbd_models(
            [],
            required=True,
            raw_user_query="输入端滤波器",
            current_selection=selection,
        )
        # 版本不符时选择服务结果整体不可信，绑定回落 unresolved。
        self.assertEqual("current_model_selection_unresolved", binding["basis"])


if __name__ == "__main__":
    unittest.main()
