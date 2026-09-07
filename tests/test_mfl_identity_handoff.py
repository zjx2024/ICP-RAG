# -*- coding: utf-8 -*-
"""MFL 跨文档身份锚点（fault-mode 词表 handoff）单元测试。

机制：FMECA ``fault_mode``、FTA 基本事件名与 MFL ``module_fault``（存储为
“模块名：故障模式”）共用同一故障模式词表。同轮直查行或对话选中的 FTA
引用图提供有界身份词，仅提升 MFL 召回排序，不授予事实。
"""

import unittest
from unittest import mock

from backend.agent.tools import icp_diagnosis as diagnosis_module
from backend.agent.tools.icp_diagnosis import (
    _anchor_fta_graph_rows,
    _fta_basic_event_name_terms,
    _mfl_fault_mode_terms,
    _mfl_identity_terms,
    _query_mfl,
)


def _fta_graph(event_names, *, top_business_id="FTA-SYN-004"):
    vertices = [
        {
            "id": "node_top",
            "label": "TopEvent",
            "properties": {"name": "外部显示设备无有效或可用画面",
                           "business_id": top_business_id},
        }
    ]
    for index, name in enumerate(event_names):
        vertices.append(
            {
                "id": f"node_be_{index}",
                "label": "BasicEvent",
                "properties": {"name": name, "module": "OIM",
                               "business_id": f"EVT-BE-{index}"},
            }
        )
    return {"vertices": vertices, "edges": [], "diagram_type": "fta"}


class _CapturingConnection:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []
        self._cursor = _CapturingCursor(rows, self.calls)

    def cursor(self, *_args, **_kwargs):
        return self._cursor

    def rollback(self):
        return None


class _CapturingCursor:
    def __init__(self, rows, calls):
        self._rows = rows
        self._calls = calls
        self.description = [("module_fault",)]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=None):
        self._calls.append((sql, params))

    def fetchall(self):
        return list(self._rows)


class MflFaultModeTermsTests(unittest.TestCase):
    def test_fmeca_rows_contribute_fault_mode_names(self):
        rows = [
            {"fault_mode": "故障日志写入失败"},
            {"fault_mode": "存储满未回收"},
        ]
        self.assertEqual(
            ["故障日志写入失败", "存储满未回收"],
            _mfl_fault_mode_terms(rows),
        )

    def test_short_and_duplicate_modes_are_dropped(self):
        rows = [
            {"fault_mode": "丢包"},          # 归一后不足 4 字符
            {"fault_mode": "突发丢包"},
            {"fault_mode": "突发丢包"},      # 重复
            {"fault_mode": ""},
        ]
        self.assertEqual(["突发丢包"], _mfl_fault_mode_terms(rows))

    def test_fmeca_source_is_bounded_to_eight_rows(self):
        rows = [{"fault_mode": f"模式编号{index:02d}失效"} for index in range(20)]
        self.assertEqual(8, len(_mfl_fault_mode_terms(rows)))


class FtaBasicEventNameTermsTests(unittest.TestCase):
    def test_basic_event_names_are_extracted_from_graph(self):
        rows = [
            {
                "canonical_graph": _fta_graph(
                    ["输出通道无信号", "接口时钟丢失", "输出编码格式错误"]
                )
            }
        ]
        self.assertEqual(
            ["输出通道无信号", "接口时钟丢失", "输出编码格式错误"],
            _fta_basic_event_name_terms(rows),
        )

    def test_legacy_nodes_shape_and_graph_alias_are_accepted(self):
        rows = [
            {
                "graph": {
                    "nodes": [
                        {"label": "BasicEvent", "properties": {"name": "帧缓存溢出"}},
                        {"label": "TopEvent", "properties": {"name": "顶事件名"}},
                    ]
                }
            }
        ]
        self.assertEqual(["帧缓存溢出"], _fta_basic_event_name_terms(rows))

    def test_top_events_are_not_identity_terms(self):
        rows = [
            {"canonical_graph": _fta_graph([])},
        ]
        self.assertEqual([], _fta_basic_event_name_terms(rows))


class AnchorFtaGraphRowsTests(unittest.TestCase):
    def _revision(self, business_id, validation_status="auto_validated"):
        return {
            "revision_id": f"rev-{business_id}",
            "validation_status": validation_status,
            "projection_status": "published",
            "is_active": True,
            "canonical_graph": _fta_graph(["输出通道无信号"], top_business_id=business_id),
        }

    def test_anchor_record_id_matches_published_revision(self):
        state = {
            "selected_conversation_anchors": [
                {"doc_type": "fta", "record_id": "FTA-SYN-004"},
            ]
        }
        with mock.patch.object(
            diagnosis_module.diagram_repository,
            "list_active_revisions",
            return_value=[self._revision("FTA-SYN-009"), self._revision("FTA-SYN-004")],
        ):
            rows = _anchor_fta_graph_rows(state)
        self.assertEqual(1, len(rows))
        self.assertEqual("rev-FTA-SYN-004", rows[0]["revision_id"])

    def test_quarantined_revisions_are_never_reread(self):
        state = {
            "selected_conversation_anchors": [
                {"doc_type": "fta", "record_id": "FTA-SYN-004"},
            ]
        }
        with mock.patch.object(
            diagnosis_module.diagram_repository,
            "list_active_revisions",
            return_value=[self._revision("FTA-SYN-004", "quarantined")],
        ):
            self.assertEqual([], _anchor_fta_graph_rows(state))

    def test_non_fta_anchors_and_repository_errors_are_ignored(self):
        self.assertEqual([], _anchor_fta_graph_rows({
            "selected_conversation_anchors": [
                {"doc_type": "rbd", "record_id": "RBD-SYN-002"},
            ]
        }))
        state = {
            "selected_conversation_anchors": [
                {"doc_type": "fta", "record_id": "FTA-SYN-004"},
            ]
        }
        with mock.patch.object(
            diagnosis_module.diagram_repository,
            "list_active_revisions",
            side_effect=RuntimeError("db down"),
        ):
            self.assertEqual([], _anchor_fta_graph_rows(state))


class MflIdentityTermsTests(unittest.TestCase):
    def test_fmeca_and_fta_sources_are_merged_and_deduplicated(self):
        fmeca_rows = [{"fault_mode": "输出通道无信号"}]
        fta_rows = [
            {
                "canonical_graph": _fta_graph(
                    ["输出通道无信号", "接口时钟丢失"]
                )
            }
        ]
        terms, sources = _mfl_identity_terms(fmeca_rows, fta_rows, {})
        self.assertEqual(["输出通道无信号", "接口时钟丢失"], terms)
        self.assertIn("fmeca_fault_mode", sources)
        self.assertIn("fta_basic_events", sources)

    def test_anchor_reread_only_runs_without_same_turn_fta_rows(self):
        fmeca_rows = []
        fta_rows = []
        state = {
            "selected_conversation_anchors": [
                {"doc_type": "fta", "record_id": "FTA-SYN-004"},
            ]
        }
        revision = {
            "revision_id": "rev-1",
            "validation_status": "auto_validated",
            "projection_status": "published",
            "is_active": True,
            "canonical_graph": _fta_graph(["输出通道无信号"]),
        }
        with mock.patch.object(
            diagnosis_module.diagram_repository,
            "list_active_revisions",
            return_value=[revision],
        ) as read:
            terms, sources = _mfl_identity_terms(fmeca_rows, fta_rows, state)
        self.assertEqual(["输出通道无信号"], terms)
        self.assertEqual(["fta_anchor_reread"], sources)
        read.assert_called_once_with(diagram_type="fta")

        # 同轮已有 FTA 图行时不做锚点回读。
        same_turn_rows = [
            {"canonical_graph": _fta_graph(["帧缓存溢出"])}
        ]
        with mock.patch.object(
            diagnosis_module.diagram_repository,
            "list_active_revisions",
        ) as read:
            terms, sources = _mfl_identity_terms([], same_turn_rows, state)
        self.assertEqual(["帧缓存溢出"], terms)
        read.assert_not_called()

    def test_term_set_is_bounded(self):
        fmeca_rows = [{"fault_mode": f"故障模式编号{index:02d}退化"} for index in range(10)]
        fta_rows = [
            {"canonical_graph": _fta_graph([f"事件名称编号{index:02d}失效" for index in range(10)])}
        ]
        terms, _sources = _mfl_identity_terms(fmeca_rows, fta_rows, {})
        self.assertLessEqual(len(terms), 12)


class QueryMflEntityTermsTests(unittest.TestCase):
    def test_entity_terms_are_forwarded_as_priority_terms(self):
        connection = _CapturingConnection(rows=[])
        with mock.patch.object(
            diagnosis_module.postgres_client, "conn", connection
        ), mock.patch.object(
            diagnosis_module,
            "_rank_table_rows",
            wraps=diagnosis_module._rank_table_rows,
        ) as rank:
            _query_mfl(
                ["输出侧分支", "维修清单"],
                5,
                entity_terms=["输出通道无信号"],
            )
        self.assertEqual(1, len(connection.calls))
        # 与 _query_fmeca 相同的契约：entity_terms 只进 Python 端身份排序；
        # SQL 召回的词表扩充由调用点 _prepend_query_terms 完成。
        self.assertEqual(
            ("mfl_code", "module_fault"),
            rank.call_args.kwargs.get("priority_columns"),
        )
        self.assertEqual(
            ("输出通道无信号",),
            tuple(rank.call_args.kwargs.get("priority_terms") or ()),
        )


if __name__ == "__main__":
    unittest.main()
