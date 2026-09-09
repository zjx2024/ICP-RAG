import unittest

from backend.agent.nodes.answer_record_scope import _narrow_semantic_table_families


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
    def test_shared_failure_mode_does_not_create_cross_document_identity(self):
        """An approved row is not necessarily the selected object's relation."""
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
        self.assertIn("MFL-SYN-099", kept)
        # Visibility does not create a relation or promote the matching prose.
        self.assertEqual(evidence, narrowed)
        self.assertTrue(all(row["match_type"] == "structured_query" for row in narrowed[1:]))

    def test_visibility_is_not_a_field_driven_cross_family_deletion(self):
        """Independent approved sources remain visible, never merged."""
        evidence = [
            _fmeca_row("ev-1", "FM-SYN-061", "故障日志写入失败",
                       match_type="exact_record_identity"),
            _mfl_row("ev-2", "MFL-SYN-061", "故障记录存储模块：故障日志写入失败"),
        ]
        state = _state(["fault_cause"], ["ev-1", "ev-2"])
        narrowed = _narrow_semantic_table_families(state, evidence)
        self.assertIn("MFL-SYN-061", [item["record_id"] for item in narrowed])

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
