import backend.agent.nodes.answer_collections as _source_answer_collections
import backend.agent.nodes.answer_target_presence as _source_answer_target_presence
import backend.agent.nodes.answer_text as _source_answer_text
import importlib.util
import json
import sys
import types
import unittest
from unittest.mock import patch

try:
    _psycopg2_spec = importlib.util.find_spec("psycopg2")
except ValueError:
    _psycopg2_spec = object() if "psycopg2" in sys.modules else None

if _psycopg2_spec is None:
    psycopg2_stub = types.ModuleType("psycopg2")
    psycopg2_stub.__spec__ = importlib.util.spec_from_loader("psycopg2", loader=None)

    def _unavailable_connect(*_args, **_kwargs):
        raise RuntimeError("psycopg2 unavailable in focused test environment")

    psycopg2_stub.connect = _unavailable_connect
    sys.modules["psycopg2"] = psycopg2_stub
elif "psycopg2" in sys.modules and getattr(sys.modules["psycopg2"], "__spec__", None) is None:
    sys.modules["psycopg2"].__spec__ = importlib.util.spec_from_loader("psycopg2", loader=None)

from backend.agent import evidence_display
from backend.agent.nodes import answer as answer_module
from backend.agent.nodes import plan as plan_module
from backend.agent.tools import icp_diagnosis as diagnosis_module






class SqlCandidatePoolSpecificityTests(unittest.TestCase):
    """The bounded SQL pool must not truncate authoritative phrase hits.

    Question-noise terms (e.g. “查询”) match nearly every row, so fetching
    the candidate pool ordered only by updated_at used to drop the row that
    contains the complete phrase before the deterministic ranker ever saw it.
    """

    def _run(self, terms, limit=5):
        connection = _FakeConnection(rows=[])
        with patch.object(
            diagnosis_module.postgres_client,
            "conn",
            connection,
            create=True,
        ):
            rows = diagnosis_module._query_table(
                "mfl",
                ("record_code", "module_fault_mode", "repair_method"),
                terms,
                limit,
            )
        self.assertEqual(len(connection.calls), 1)
        query, params = connection.calls[0]
        return query, params, rows

    def test_pool_orders_by_term_specificity_before_updated_at(self):
        query, params, rows = self._run(["输出通道无信号", "查询"])
        self.assertIn("ORDER BY (CASE WHEN", query)
        # len² specificity weights: 7-char phrase outranks the 2-char noise word.
        self.assertIn("THEN 49 ELSE 0 END", query)
        self.assertIn("THEN 4 ELSE 0 END", query)
        self.assertLess(
            query.index("THEN 49 ELSE 0 END"),
            query.index("THEN 4 ELSE 0 END"),
        )
        # updated_at survives only as the final tie-break after the score sum.
        self.assertIn(") DESC, t.updated_at DESC LIMIT %s", query)
        self.assertEqual(rows, [])
        del params

    def test_user_terms_stay_bound_parameters(self):
        query, params, _rows = self._run(["输出通道无信号", "MFL-SYN-049"])
        for term in ("输出通道无信号", "MFL-SYN-049"):
            self.assertNotIn(term, query)
        self.assertIn("%输出通道无信号%", params)
        self.assertIn("%MFL-SYN-049%", params)
        self.assertEqual(query.count("%s"), len(params))


class AnchorSubjectEventExpansionTests(unittest.TestCase):
    """Selected FTA citations must carry their event wording into the next turn."""

    def test_structured_events_expand_to_names_only(self):
        anchor = {
            "record_id": "FTA-SYN-004",
            "title": "处理机故障树",
            "answer_fields": {
                "top_event": "处理机无法完成接口帧数据交换",
                # Persisted rows may hold JSON-stringified structured entries.
                "basic_events": json.dumps(
                    [
                        {
                            "name": "输出通道无信号",
                            "description": "该事件表现为通道侧描述性长文本，不应进入检索词。",
                        }
                    ],
                    ensure_ascii=False,
                ),
                "intermediate_events": [
                    {"name": "输出驱动模块失效", "description": "中间事件描述文本。"}
                ],
            },
        }
        subjects = plan_module._anchor_subjects([anchor])
        self.assertIn("输出通道无信号", subjects)
        self.assertIn("输出驱动模块失效", subjects)
        self.assertIn("处理机无法完成接口帧数据交换", subjects)
        self.assertTrue(
            all("描述" not in subject for subject in subjects),
            f"descriptive prose leaked into retrieval subjects: {subjects}",
        )


class CandidateTermLinkageTests(unittest.TestCase):
    """Compound anchors must also reach tables that store the CJK core only."""

    def test_latin_prefixed_anchor_yields_cjk_fragment_term(self):
        terms = diagnosis_module._sql_candidate_terms(["ICU 输入过压钳位失效"])
        self.assertIn("输入过压钳位失效", terms)

    def test_pure_cjk_short_anchor_is_not_duplicated_as_fragment(self):
        terms = diagnosis_module._sql_candidate_terms(["时钟丢失"])
        self.assertEqual(
            sum(1 for term in terms if term == "时钟丢失"),
            1,
        )


class RankTableSpecificityFloorTests(unittest.TestCase):
    """Rows matched only by 2-3 char generic words must lose to phrase hits."""

    def test_generic_only_rows_are_dropped_for_code_anchored_queries(self):
        columns = ("record_code", "module_fault", "sur_fault")
        rows = [
            {
                "record_code": "MFL-A",
                "module_fault": "输入调理模块：输入过压钳位失效",
                "sur_fault": "SUR-PWR-01状态字第3位异常",
            },
            {
                "record_code": "MFL-B",
                "module_fault": "风扇组件：风量衰减",
                "sur_fault": "SUR-TCS-03状态字第1位异常",
            },
        ]
        kept = diagnosis_module._rank_table_rows(
            rows,
            columns,
            ["FM-SYN-002", "输入过压钳位失效", "状态", "检查"],
            limit=10,
        )
        self.assertEqual([row["record_code"] for row in kept], ["MFL-A"])

    def test_symptom_only_queries_keep_the_multi_candidate_pool(self):
        columns = ("record_code", "module_fault", "sur_fault")
        rows = [
            {
                "record_code": "MFL-A",
                "module_fault": "输入调理模块：输入过压钳位失效",
                "sur_fault": "SUR-PWR-01状态字第3位异常",
            },
            {
                "record_code": "MFL-B",
                "module_fault": "风扇组件：风量衰减",
                "sur_fault": "SUR-TCS-03状态字第1位异常",
            },
        ]
        kept = diagnosis_module._rank_table_rows(
            rows, columns, ["输入过压钳位失效", "状态", "衰减"], limit=10
        )
        # 没有业务编码时不得把弱命中候选整批裁掉：症状问题保留排查方向。
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["record_code"], "MFL-A")

    def test_weak_only_pool_is_preserved_for_short_identifier_queries(self):
        columns = ("record_code", "module_fault", "api_name")
        rows = [
            {
                "record_code": "MFL-C",
                "module_fault": "输出接口模块：输出通道无信号",
                "api_name": "GetOIMFaultStatus",
            }
        ]
        kept = diagnosis_module._rank_table_rows(
            rows, columns, ["OIM"], limit=10
        )
        self.assertEqual([row["record_code"] for row in kept], ["MFL-C"])


class DiagramCategoryFallbackTests(unittest.TestCase):
    """Multi-object diagram questions must not answer from one lucky hit."""

    @staticmethod
    def _revision(revision_id: str, title: str, block_name: str = ""):
        vertices = (
            [{"id": "b1", "label": "block", "properties": {"name": block_name}}]
            if block_name
            else []
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

    def _query(self, revisions, terms, *, query_frame=None):
        with patch.object(
            diagnosis_module.diagram_repository,
            "list_active_revisions",
            return_value=revisions,
        ):
            return diagnosis_module._query_diagrams(
                "rbd", terms, 2, query_frame=query_frame
            )

    def test_grounded_multi_subject_contract_expands_the_requested_family(self):
        revisions = [
            self._revision("R1", "电源保障链可靠性模型", block_name="A路稳压供电"),
            self._revision("R2", "核心处理功能可靠性模型", block_name="GPP-A"),
            self._revision("R3", "显示输出功能可靠性模型", block_name="DPM"),
        ]
        query = "供电、核心运算、外部信息交换和显示通路分别依赖哪些环节"
        rows = self._query(
            revisions,
            [
                query,
                "供电",
                "核心运算",
                "外部信息交换",
                "显示通路",
            ],
            query_frame={
                "original_query": query,
                "subject_bindings": [
                    {
                        "doc_type": "rbd",
                        "kind": "component",
                        "value": value,
                        "basis": value,
                        "grounded": True,
                        "source": "current_query",
                    }
                    for value in (
                        "供电", "核心运算", "外部信息交换", "显示通路"
                    )
                ],
            },
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["diagram_key"], "R1")
        self.assertFalse(rows[0].get("category_context"))
        self.assertTrue(all(row.get("category_context") for row in rows[1:]))

    def test_single_object_question_does_not_widen_the_family(self):
        revisions = [
            self._revision("R1", "电源保障链可靠性模型", block_name="A路稳压供电"),
            self._revision("R2", "核心处理功能可靠性模型", block_name="GPP-A"),
        ]
        rows = self._query(revisions, ["供电"])
        self.assertEqual([row["diagram_key"] for row in rows], ["R1"])

    def test_total_miss_does_not_fabricate_category_context(self):
        # 资料真未命中时补齐全族只会制造跨域噪声：必须保持空结果。
        revisions = [
            self._revision("R1", "电源保障链可靠性模型", block_name="A路稳压供电"),
            self._revision("R2", "核心处理功能可靠性模型", block_name="GPP-A"),
        ]
        rows = self._query(
            revisions,
            ["输入端滤波器、稳压器分别依赖哪些路径", "滤波器", "稳压器", "输入端"],
        )
        self.assertEqual(rows, [])


class BoundedEnumerationTests(unittest.TestCase):
    """Small authoritative collections are answers, not “excerpts”."""

    @staticmethod
    def _fta_graph(event_count: int):
        vertices = [
            {
                "id": "top",
                "label": "TopEvent",
                "properties": {"name": "顶事件"},
            },
            *[
                {
                    "id": f"be-{index}",
                    "label": "BasicEvent",
                    "properties": {"name": f"事件{index}"},
                }
                for index in range(event_count)
            ],
        ]
        return {"vertices": vertices, "edges": []}

    def test_events_render_as_bounded_count_without_details(self):
        summary = evidence_display.summarize_graph_structure(
            self._fta_graph(8), diagram_type="fta", title="示例树"
        )
        self.assertIn("基本事件共 8 项", summary)
        self.assertNotIn("事件1", summary)

    def test_large_event_sets_still_report_only_count(self):
        summary = evidence_display.summarize_graph_structure(
            self._fta_graph(15), diagram_type="fta", title="示例树"
        )
        self.assertIn("基本事件共 15 项", summary)
        self.assertNotIn("事件1", summary)




class AnswerHygieneTests(unittest.TestCase):

    def test_empty_full_width_parentheses_are_stripped(self):
        normalized = _source_answer_text._normalize_generated_markdown(
            "由供电缺失引起（），导致无画面。"
        )
        self.assertEqual(normalized, "由供电缺失引起，导致无画面。")
        kept = _source_answer_text._normalize_generated_markdown(
            "调用 GetOIMFaultStatus() 确认"
        )
        self.assertIn("GetOIMFaultStatus()", kept)


class CollectionFacetViewBridgeTests(unittest.TestCase):
    """Sparse inventory rows may borrow fields only from exact-ID siblings."""

    @staticmethod
    def _sparse():
        return {
            "citation_id": 1,
            "record_id": "CMP-900",
            "answer_fields": {
                "collection_member_kind": "failure_mode",
                "collection_member_id": "FM-D1",
                "collection_member_name": "日志写入失败",
                "fault_cause": "",
            },
        }

    def test_exact_business_id_bridges_fields_and_citation(self):
        rich = {
            "citation_id": 5,
            "record_id": "FM-D1",
            "answer_fields": {
                "record_code": "FM-D1",
                "fault_cause": "存储通道被长期占用",
            },
        }
        sparse = self._sparse()
        fields, citation = _source_answer_collections._collection_facet_view(
            sparse, [sparse, rich]
        )
        self.assertEqual(fields.get("fault_cause"), "存储通道被长期占用")
        self.assertEqual(fields.get("collection_member_name"), "日志写入失败")
        self.assertEqual(citation, 5)

    def test_different_identity_never_bridges(self):
        unrelated = {
            "citation_id": 5,
            "record_id": "FM-X9",
            "answer_fields": {
                "record_code": "FM-X9",
                "fault_cause": "完全另一条记录的原因",
            },
        }
        sparse = self._sparse()
        fields, citation = _source_answer_collections._collection_facet_view(
            sparse, [sparse, unrelated]
        )
        self.assertEqual(fields.get("fault_cause"), "")
        self.assertEqual(citation, 1)


class CollectionMemberFacetPresenceTests(unittest.TestCase):
    """A failure-mode inventory is only complete with each record's facets."""

    @staticmethod
    def _items():
        common = {
            "collection_closed": True,
            "collection_role": "member",
            "collection_parent_id": "FUNC-777",
            "collection_parent_name": "某个记录功能",
            "collection_member_kind": "failure_mode",
            "collection_member_count": 2,
        }
        return [
            {
                "citation_id": 1,
                "doc_id": "doc-1",
                "record_id": "FM-D1",
                "answer_fields": {
                    **common,
                    "collection_member_id": "FM-D1",
                    "collection_member_name": "日志写入失败",
                    "fault_cause": "存储通道被长期占用",
                },
            },
            {
                "citation_id": 2,
                "doc_id": "doc-1",
                "record_id": "FM-D2",
                "answer_fields": {
                    **common,
                    "collection_member_id": "FM-D2",
                    "collection_member_name": "索引校验错误",
                    "fault_cause": "写入竞争损坏了索引区",
                },
            },
        ]

    def test_title_only_list_is_not_present(self):
        answer = (
            "1. 日志写入失败（FM-D1）。[^1]\n"
            "2. 索引校验错误（FM-D2）。[^2]"
        )
        self.assertFalse(
            _source_answer_target_presence._closed_collection_members_present(
                answer, self._items()
            )
        )

    def test_list_with_per_member_cause_is_present(self):
        answer = (
            "1. 日志写入失败（FM-D1）。[^1]\n"
            "   - 原文原因：存储通道被长期占用。[^1]\n"
            "2. 索引校验错误（FM-D2）。[^2]\n"
            "   - 原文原因：写入竞争损坏了索引区。[^2]"
        )
        self.assertTrue(
            _source_answer_target_presence._closed_collection_members_present(
                answer, self._items()
            )
        )


class _FakeCursor:
    def __init__(self, connection, rows):
        self.connection = connection
        self.rows = rows
        self.description = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=()):
        self.connection.calls.append((" ".join(query.split()), tuple(params or ())))
        # The production statement selects t.*; emulate the column contract
        # from the rows the test pre-supplies.
        columns = list(self.rows[0].keys()) if self.rows else ["doc_id"]
        self.description = [(name, None, None, None, None, None, None) for name in columns]

    def fetchall(self):
        if not self.rows:
            return []
        columns = list(self.rows[0].keys())
        return [tuple(row.get(column) for column in columns) for row in self.rows]

class _FakeConnection:
    def __init__(self, rows):
        self.calls = []
        self.rows = rows

    def cursor(self):
        return _FakeCursor(self, self.rows)

    def rollback(self):
        pass


if __name__ == "__main__":
    unittest.main()
