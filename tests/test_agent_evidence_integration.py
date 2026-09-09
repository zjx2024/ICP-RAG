import backend.agent.nodes.answer_claim_units as _source_answer_claim_units
import backend.agent.nodes.answer_evidence_window as _source_answer_evidence_window
import backend.agent.nodes.answer_payload as _source_answer_payload
import backend.agent.nodes.answer_text as _source_answer_text
import asyncio
import importlib.util
import json
import sys
import types
import unittest
from unittest.mock import patch

try:
    _psycopg2_spec = importlib.util.find_spec("psycopg2")
except ValueError:
    # Another focused test may already have installed an in-memory stub whose
    # ``__spec__`` is intentionally absent.
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

from backend.agent.contracts import make_evidence
from backend.agent.nodes import answer as answer_module
from backend.agent.nodes.evidence_gate import evidence_gate_node
from backend.agent.rbd_contract import bind_rbd_models
from backend.agent.tools import icp_calculation as calculation_module
from backend.agent.tools import icp_diagnosis as diagnosis_module


def _run(coroutine):
    return asyncio.run(coroutine)






def _rbd_graph():
    return {
        "vertices": [
            {"id": "start", "label": "Start", "properties": {"name": "START"}},
            {
                "id": "block",
                "label": "ReliabilityBlock",
                "properties": {"name": "处理模块", "reliability": 0.98},
            },
            {"id": "end", "label": "End", "properties": {"name": "END"}},
        ],
        "edges": [
            {"id": "e1", "source": "start", "target": "block"},
            {"id": "e2", "source": "block", "target": "end"},
        ],
    }


def _calculation_authorization(operation, target, basis):
    return [{
        "operation": operation,
        "mode": "calculate",
        "target": target,
        "basis": basis,
    }]


def _rbd_authority_binding(row, query):
    selected, binding = bind_rbd_models(
        [row],
        required=True,
        raw_user_query=query,
        explicit_rbd_ids=[row["revision_id"]],
    )
    assert selected == [row]
    assert binding["bound"] is True
    return binding


class DiagnosisIntegrationTests(unittest.TestCase):








    def test_fmeca_sql_recall_uses_planner_normalized_phrase(self):
        connection = _CapturingConnection()
        with patch.object(diagnosis_module.postgres_client, "conn", connection):
            diagnosis_module._query_fmeca(
                ["滤波 电容老化！"],
                5,
            )

        self.assertEqual(1, len(connection.calls))
        _sql, params = connection.calls[0]
        self.assertIn("%滤波电容老化%", params)

    def test_fmeca_sql_recall_prioritizes_late_business_id_and_short_anchor(self):
        connection = _CapturingConnection()
        terms = [
            "请结合当前运行阶段解释输入状态反复变化属于什么异常级别",
            "当前现象可能涉及多个原因以及局部影响和最终影响",
            "还需要说明不同候选故障之间有哪些可区分的表现",
            "请根据已有结构化资料而不是一般经验进行判断",
            "需要检查该现象是否会进一步影响其他相关子系统",
            "同时说明现有资料是否覆盖当前构型以及版本范围",
            "如果存在多个候选请分别列出各自的直接记录依据",
            "最后给出仍然缺少哪些能够核验结论的信息",
            "FUNC-110",
            "输入调理模块",
        ]
        with patch.object(diagnosis_module.postgres_client, "conn", connection):
            diagnosis_module._query_fmeca(terms, 5)

        self.assertEqual(1, len(connection.calls))
        _sql, params = connection.calls[0]
        patterns = list(dict.fromkeys(
            value
            for value in params
            if isinstance(value, str) and value.startswith("%")
        ))
        self.assertLessEqual(len(patterns), diagnosis_module._SQL_TERM_BUDGET)
        self.assertEqual("%FUNC-110%", patterns[0])
        self.assertIn("%输入调理模块%", patterns)

    def test_nonmaintenance_hmc_lookup_does_not_match_repair_text_only(self):
        connection = _CapturingConnection()
        with patch.object(diagnosis_module.postgres_client, "conn", connection):
            diagnosis_module._query_hmc(
                "没有正确传出数据到外设",
                ["外设"],
                5,
                include_repair_method=False,
            )

        sql, _params = connection.calls[0]
        self.assertIn("h.fault_desc", sql)
        self.assertNotIn("h.repair_method", sql)

    def test_maintenance_hmc_lookup_can_still_search_repair_text(self):
        connection = _CapturingConnection()
        with patch.object(diagnosis_module.postgres_client, "conn", connection):
            diagnosis_module._query_hmc(
                "外设应该怎样检查",
                ["外设"],
                5,
                include_repair_method=True,
            )

        sql, _params = connection.calls[0]
        self.assertIn("h.repair_method", sql)

    def test_hybrid_fta_locator_handoff_rehydrates_by_stable_postgres_id(self):
        hit = {
            "doc_type_kwd": "fta",
            "doc_id": "doc-fta",
            "diagram_id_kwd": "diagram-fta-3",
            "business_id_kwd": "FTA-SYN-003",
            "projection_role_kwd": "diagram_locator",
            "content_type": "fta_structured_facts",
            "title": "外部通信分析",
            "score": 0.91,
        }
        revision = {
            "doc_id": "doc-fta",
            "diagram_id": "diagram-fta-3",
            "revision_id": "rev-3",
            "diagram_type": "fta",
            "diagram_key": "FTA-SYN-003",
            "validation_status": "auto_validated",
            "projection_status": "published",
            "is_active": True,
            "canonical_graph": {"vertices": [], "edges": []},
        }
        handoff = diagnosis_module._search_locator_handoffs([hit])
        with patch.object(
            diagnosis_module.diagram_repository,
            "get_active_revision",
            return_value=revision,
        ):
            hydrated = diagnosis_module._hydrate_diagram_search_hits(
                handoff,
                allowed_tools=["fta_query"],
                limit_per_type=4,
            )

        self.assertEqual("diagram-fta-3", handoff[0]["diagram_id_kwd"])
        self.assertEqual("rev-3", hydrated["fta_query"][0]["revision_id"])
        self.assertEqual(
            "semantic_locator_hydration",
            hydrated["fta_query"][0]["match_type"],
        )

    def test_locator_handoff_keeps_four_named_rbd_models_for_comparison(self):
        hits = [
            {
                "doc_type_kwd": "rbd",
                "doc_id": "doc-rbd",
                "diagram_id_kwd": f"diagram-rbd-{index}",
                "business_id_kwd": f"RBD-SYN-{index:03d}",
                "projection_role_kwd": "diagram_locator",
                "content_type": "rbd_structured_facts",
                "score": 1.0 - index / 10,
            }
            for index in range(1, 5)
        ]

        handoffs = diagnosis_module._search_locator_handoffs(hits)

        self.assertEqual(4, len(handoffs))
        self.assertEqual(
            [f"diagram-rbd-{index}" for index in range(1, 5)],
            [item["diagram_id_kwd"] for item in handoffs],
        )

    def test_hybrid_table_hit_rehydrates_exact_postgres_storage_identity(self):
        hit = {
            "doc_type_kwd": "fmeca",
            "doc_id": "doc-fmeca",
            "pg_id_kwd": "6",
            "business_id_kwd": "FM-SYN-006",
            "content_type": "fmeca_row",
            "chunk_id": "fmeca-hit-6",
            "title": "FM-SYN-006",
            "score": 0.72,
        }
        authoritative = {
            "id": 6,
            "doc_id": "doc-fmeca",
            "code": "FM-SYN-006",
            "fault_mode": "A路间歇停机",
            "fault_cause": "功率器件热失稳、焊点裂纹或内部连接间歇。",
        }

        handoff = diagnosis_module._search_locator_handoffs([hit])
        with patch.object(
            diagnosis_module.icp_repository,
            "get_records_by_storage_locators",
            return_value=[authoritative],
        ) as lookup:
            hydrated = diagnosis_module._hydrate_record_search_hits(handoff)

        lookup.assert_called_once_with("fmeca", handoff, limit=1)
        self.assertEqual("6", handoff[0]["pg_id_kwd"])
        self.assertEqual("FM-SYN-006", hydrated["fmeca_query"][0]["code"])
        self.assertEqual(
            "semantic_id_hydration",
            hydrated["fmeca_query"][0]["match_type"],
        )

    def test_two_temperature_modes_hydrate_bounded_hmc_fmeca_and_mfl_truth(self):
        specs = {
            "hmc": (
                "hmc_query",
                [
                    (581, "91850002**0001", "传感器通道开路"),
                    (582, "91850002**0002", "A/D采样饱和"),
                    (583, "91850002**0099", "不相关温度记录"),
                ],
            ),
            "fmeca": (
                "fmeca_query",
                [
                    (591, "FM-SYN-058", "传感器通道开路"),
                    (592, "FM-SYN-059", "A/D采样饱和"),
                    (593, "FM-SYN-099", "不相关温度记录"),
                ],
            ),
            "mfl": (
                "mfl_query",
                [
                    (601, "MFL-SYN-058", "检查传感器线路与接口"),
                    (602, "MFL-SYN-059", "校验量程与增益"),
                    (603, "MFL-SYN-099", "不相关温度记录"),
                ],
            ),
        }
        hits = []
        authoritative = {}
        for doc_type, (_tool, records) in specs.items():
            for storage_id, business_id, text in records:
                hits.append(
                    {
                        "doc_type_kwd": doc_type,
                        "doc_id": f"doc-{doc_type}",
                        "pg_id_kwd": str(storage_id),
                        "business_id_kwd": business_id,
                        "content_type": f"{doc_type}_row",
                        "chunk_id": f"search-{storage_id}",
                        "title": business_id,
                        # This indexed payload is deliberately false and must
                        # never survive the locator handoff as answer evidence.
                        "content": "OpenSearch 投影正文不得作为权威事实",
                        "score": 1.0 - len(hits) / 100,
                    }
                )
                row = {"id": storage_id, "doc_id": f"doc-{doc_type}"}
                if doc_type == "hmc":
                    row.update(hmc_code=business_id, fault_desc=text)
                elif doc_type == "fmeca":
                    row.update(code=business_id, fault_mode=text)
                else:
                    row.update(mfl_code=business_id, maintenance_action=text)
                authoritative[(doc_type, str(storage_id))] = row

        handoffs = diagnosis_module._search_locator_handoffs(
            hits, limit=12, limit_per_type=4
        )

        def exact_lookup(doc_type, locators, *, limit):
            return [
                authoritative[(doc_type, str(locator["pg_id_kwd"]))]
                for locator in locators[:limit]
            ]

        with patch.object(
            diagnosis_module.icp_repository,
            "get_records_by_storage_locators",
            side_effect=exact_lookup,
        ) as lookup:
            hydrated = diagnosis_module._hydrate_record_search_hits(
                handoffs,
                limit_per_type=4,
            )

        self.assertEqual(3, lookup.call_count)
        for call in lookup.call_args_list:
            self.assertEqual(4, call.kwargs["limit"])
            self.assertEqual(3, len(call.args[1]))
        self.assertEqual(
            ["91850002**0001", "91850002**0002", "91850002**0099"],
            [row["hmc_code"] for row in hydrated["hmc_query"]],
        )
        self.assertEqual(
            ["FM-SYN-058", "FM-SYN-059", "FM-SYN-099"],
            [row["code"] for row in hydrated["fmeca_query"]],
        )
        self.assertEqual(
            ["MFL-SYN-058", "MFL-SYN-059", "MFL-SYN-099"],
            [row["mfl_code"] for row in hydrated["mfl_query"]],
        )
        for rows in hydrated.values():
            for row in rows:
                self.assertEqual("semantic_id_hydration", row["match_type"])
                self.assertNotIn("OpenSearch 投影正文", json.dumps(row, ensure_ascii=False))

    def test_function_locator_hydration_requires_exact_storage_kind(self):
        hits = [
            {
                "doc_type_kwd": "function_locator",
                "doc_id": "doc-arch",
                "pg_id_kwd": "31",
                "content_type": "function_record",
                "business_id_kwd": "FUNC-330",
            },
            {
                "doc_type_kwd": "function_locator",
                "doc_id": "doc-arch",
                "pg_id_kwd": "12",
                "content_type": "function_component",
                "business_id_kwd": "SUB-IFC",
            },
            {
                "doc_type_kwd": "function_locator",
                "doc_id": "doc-arch",
                "pg_id_kwd": "9",
                "content_type": "function_interface",
                "business_id_kwd": "IF-IFC-GPP",
            },
            {
                "doc_type_kwd": "function_locator",
                "doc_id": "doc-arch",
                "pg_id_kwd": "999",
                "content_type": "function_text",
                "content": "外部数据经IFC标准化送入GPP",
            },
        ]
        authoritative = [
            {"id": 31, "doc_id": "doc-arch", "function_id": "FUNC-330", "content_type": "function_record"},
            {"id": 12, "doc_id": "doc-arch", "component_id": "SUB-IFC", "content_type": "function_component"},
            {"id": 9, "doc_id": "doc-arch", "interface_id": "IF-IFC-GPP", "content_type": "function_interface"},
        ]

        handoffs = diagnosis_module._search_locator_handoffs(
            hits, limit=8, limit_per_type=8
        )
        with patch.object(
            diagnosis_module.icp_repository,
            "get_records_by_storage_locators",
            return_value=authoritative,
        ) as lookup:
            hydrated = diagnosis_module._hydrate_record_search_hits(
                handoffs, limit_per_type=8
            )

        expected_locators = [
            item for item in handoffs if item["content_type"] != "function_text"
        ]
        lookup.assert_called_once_with(
            "function_locator", expected_locators, limit=8
        )
        self.assertEqual(
            ["FUNC-330", "SUB-IFC", "IF-IFC-GPP"],
            [
                row.get("function_id") or row.get("component_id") or row.get("interface_id")
                for row in hydrated["function_locator"]
            ],
        )
        self.assertTrue(
            all(row["match_type"] == "semantic_id_hydration" for row in hydrated["function_locator"])
        )

    def test_table_handoff_requires_doc_and_postgres_id_not_title(self):
        title_only = {
            "doc_type_kwd": "fmeca",
            "doc_id": "doc-fmeca",
            "business_id_kwd": "FM-SYN-006",
            "title": "FM-SYN-006",
        }
        self.assertEqual([], diagnosis_module._search_locator_handoffs([title_only]))

    def test_formal_fta_exposes_explicit_fmeca_ids_without_text_similarity(self):
        rows = [
            {
                "validation_status": "auto_validated",
                "projection_status": "published",
                "is_active": True,
                "graph": {
                    "vertices": [
                        {
                            "id": "be-1",
                            "properties": {"failure_mode_id": "FM-SYN-041"},
                        },
                        {
                            "id": "be-2",
                            "properties": {"failure_mode_id": "FM-SYN-042"},
                        },
                    ]
                },
            }
        ]
        self.assertEqual(
            ["FM-SYN-041", "FM-SYN-042"],
            diagnosis_module._explicit_failure_mode_ids(rows),
        )

    def test_fmeca_phrase_coverage_ranks_filter_capacitor_row_before_generic_aging(self):
        rows = [
            {
                "code": "FM-SYN-065",
                "fault_mode": "系统时钟频率漂移",
                "fault_cause": "晶振老化或温度补偿异常",
                "local_impact": "时间基准偏移",
            },
            {
                "code": "FM-SYN-015",
                "fault_mode": "保持时间不足",
                "fault_cause": "储能器件老化",
                "local_impact": "保持能力下降",
            },
            {
                "code": "FM-SYN-002",
                "fault_mode": "输入滤波性能退化",
                "fault_cause": "滤波电容老化、连接阻抗增大或屏蔽接地不良",
                "local_impact": "输入纹波和传导干扰增大",
            },
        ]
        ranked = diagnosis_module._rank_table_rows(
            rows,
            ("code", "fault_mode", "fault_cause", "local_impact"),
            ("滤波电容有点老化了，会有什么影响吗？", "滤波电容", "老化"),
            3,
        )

        self.assertEqual("FM-SYN-002", ranked[0]["code"])
        self.assertGreater(
            ranked[0]["retrieval_match_score"],
            ranked[1]["retrieval_match_score"],
        )

    def test_impact_explanation_does_not_generate_diagnostic_narrowing_question(self):
        fmeca_rows = [
            {
                "doc_id": "doc-fmeca",
                "code": "FM-SYN-002",
                "fault_mode": "输入滤波性能退化",
                "fault_cause": "滤波电容老化、连接阻抗增大或屏蔽接地不良",
                "local_impact": "输入纹波和传导干扰增大",
                "final_impact": "多个子系统间歇异常",
                "severity": "III",
                "detect_method": "检测输入纹波",
            }
        ]
        state = {
            "intent": "impact_analysis",
            "query": "滤波电容有点老化了，会有什么影响吗？",
            "rewritten_query": "滤波电容老化的影响",
            "entity_keywords": ["滤波电容", "老化"],
            "allowed_tools": ["fmeca_query", "evidence_synthesis"],
            "sql_limit": 8,
        }
        with (
            patch.object(diagnosis_module, "_query_fmeca", return_value=fmeca_rows),
            patch.object(diagnosis_module, "_enrich_function_sources", side_effect=lambda rows, _errors=None: list(rows)),
        ):
            result = _run(diagnosis_module.icp_diagnosis_node(state))

        self.assertEqual("", result["clarification_question"])
        self.assertTrue(result["evidence_registry"])

    def test_source_boundary_followup_does_not_generate_fault_clarification(self):
        for facets in (
            ["evidence_boundaries", "conclusion_validity"],
            ["evidence_boundary", "conclusion_scope"],
            ["capability_boundary", "limitation"],
        ):
            with self.subTest(facets=facets):
                state = {
                    "intent": "hybrid",
                    "query": "哪些结论能说，哪些不能说？不要用不相干的框图补足。",
                    "rewritten_query": "说明已查 RBD 的证据边界",
                    "query_frame": {
                        "entities": ["输入端滤波器"],
                        "requested_facets": facets,
                        "requested_source_tools": ["rbd_query"],
                    },
                    "allowed_tools": ["rbd_query", "evidence_synthesis"],
                    "requested_source_tools": ["rbd_query"],
                    "sql_limit": 8,
                }
                with (
                    patch.object(diagnosis_module, "_query_diagrams", return_value=[]),
                    patch.object(
                        diagnosis_module,
                        "_enrich_function_sources",
                        side_effect=lambda rows, _errors=None: list(rows),
                    ),
                ):
                    result = _run(diagnosis_module.icp_diagnosis_node(state))

                self.assertEqual("", result["clarification_question"])
                self.assertNotIn(
                    "fault_reasoning",
                    [item["tool"] for item in result["tool_calls"]],
                )

    def test_direct_fact_queries_exclude_failed_or_in_progress_documents(self):
        connection = _CapturingConnection()
        with patch.object(diagnosis_module.postgres_client, "conn", connection):
            diagnosis_module._query_hmc("查询 31810001-**-0001", [], 5)
            diagnosis_module._query_fmeca(["显示异常"], 5)
            diagnosis_module._query_mfl(["显示异常"], 5)

        self.assertEqual(3, len(connection.calls))
        for sql, _params in connection.calls:
            self.assertIn("JOIN icp_documents", sql)
            self.assertIn("d.status IN ('indexed','kg_pending','kg_done')", sql)
        mfl_sql, _ = connection.calls[-1]
        for column in (
            "t.maintenance_action",
            "t.platform_test_status",
            "t.platform_test_raw",
            "t.api_reportable",
        ):
            self.assertIn(column, mfl_sql)

    def test_hmc_query_without_hamc_uses_only_supplied_string_components(self):
        connection = _CapturingConnection()
        with patch.object(diagnosis_module.postgres_client, "conn", connection):
            diagnosis_module._query_hmc(
                "LCN 91810001 HSC 0001 故障如何修复",
                [],
                5,
            )

        self.assertEqual(1, len(connection.calls))
        sql, params = connection.calls[0]
        self.assertIn("h.lcn=%s AND h.hsc=%s", sql)
        self.assertNotIn("h.hamc=%s", sql)
        self.assertEqual(("91810001", "0001", 5), params)

    def test_complete_hmc_query_keeps_three_component_exact_match(self):
        connection = _CapturingConnection()
        with patch.object(diagnosis_module.postgres_client, "conn", connection):
            diagnosis_module._query_hmc("查询 31810001-**-0001", [], 5)

        sql, params = connection.calls[0]
        self.assertIn("h.lcn=%s AND h.hamc=%s AND h.hsc=%s", sql)
        self.assertEqual(("31810001", "**", "0001", 5), params)

    def test_hmc_evidence_preserves_source_components_and_document_status(self):
        evidence = diagnosis_module._direct_evidence(
            "hmc_query",
            [
                {
                    "doc_id": "doc-hmc",
                    "doc_name": "HMC手册.docx",
                    "hmc_code": "91810001**0001",
                    "lcn": "91810001",
                    "hamc": "**",
                    "hsc": "0001",
                    "fault_desc": "ICU 输入过压钳位失效",
                    "repair_method": "检查钳位器件。",
                    "match_type": "exact_components",
                    "source_ref": "HMC手册.docx#table=2;row=3",
                    "document_metadata": {
                        "document_number": "SYN-ICP-DOC-002",
                        "data_source": "程序化合成",
                        "data_status": "合成测试数据 / 非运行资料",
                    },
                }
            ],
        )

        self.assertEqual(["hmc_fault", "maintenance_action"], [item["claim_type"] for item in evidence])
        for item in evidence:
            self.assertEqual("91810001**0001", item["metadata"]["hmc_code"])
            self.assertEqual("91810001", item["metadata"]["lcn"])
            self.assertEqual("**", item["metadata"]["hamc"])
            self.assertEqual("0001", item["metadata"]["hsc"])
            self.assertEqual("合成测试数据 / 非运行资料", item["metadata"]["data_status"])
            self.assertEqual("HMC手册.docx#table=2;row=3", item["source_ref"])

    def test_function_locator_does_not_run_opensearch_without_hybrid_permission(self):
        rows = [
            {
                "doc_id": "doc-function",
                "function_id": "FUNC-1",
                "function_name": "数据处理",
                "module_name": "处理模块",
                "description": "处理总线数据",
            }
        ]
        state = {
            "intent": "function_location",
            "query": "定位数据处理功能",
            "allowed_tools": ["function_locator", "evidence_synthesis"],
        }
        with (
            patch.object(diagnosis_module.icp_repository, "search_functions", return_value=rows),
            patch.object(diagnosis_module, "_hybrid_search", side_effect=AssertionError("OpenSearch 越过白名单")),
            patch.object(
                diagnosis_module,
                "_enrich_function_sources",
                side_effect=lambda values, errors=None: list(values),
            ),
        ):
            result = _run(diagnosis_module.icp_diagnosis_node(state))
        function_result = next(
            item for item in result["tool_results"] if item["tool"] == "function_locator"
        )
        self.assertEqual("postgres", function_result["diagnostics"]["backend"])
        self.assertEqual(1, function_result["data"]["postgres_count"])
        self.assertNotIn("opensearch_count", function_result["data"])

    def test_non_architecture_flow_executes_hybrid_search_once(self):
        state = {
            "intent": "symptom_diagnosis",
            "query": "显示闪烁",
            "allowed_tools": ["function_locator", "hybrid_search", "evidence_synthesis"],
            "hybrid_doc_types": ["function_locator"],
        }
        with (
            patch.object(diagnosis_module.icp_repository, "search_functions", return_value=[]),
            patch.object(
                diagnosis_module,
                "_hybrid_search",
                return_value=(
                    [{"chunk_id": "f-1", "title": "显示功能", "content": "显示链路功能"}],
                    {"reranked": False},
                ),
            ) as hybrid_search,
        ):
            _run(diagnosis_module.icp_diagnosis_node(state))
        hybrid_search.assert_called_once()

    def test_hybrid_search_with_no_valid_seven_type_scope_is_blocked(self):
        state = {
            "intent": "hybrid",
            "query": "检索资料",
            "allowed_tools": ["hybrid_search", "evidence_synthesis"],
            "hybrid_doc_types": ["other_knowledge_base"],
        }
        with patch.object(
            diagnosis_module,
            "_hybrid_search",
            side_effect=AssertionError("无合法范围时不应执行 OpenSearch"),
        ):
            result = _run(diagnosis_module.icp_diagnosis_node(state))
        hybrid = next(item for item in result["tool_results"] if item["tool"] == "hybrid_search")
        self.assertEqual("blocked", hybrid["status"])
        self.assertTrue(hybrid["diagnostics"]["warnings"])
        self.assertTrue(hybrid["diagnostics"]["fallbacks"])

    def test_explicit_whitelist_executes_only_hmc_and_tracks_candidate_evidence(self):
        hmc_rows = [
            {
                "doc_id": "doc-hmc",
                "doc_name": "HMC手册.xlsx",
                "hmc_code": "31810001**0001",
                "lcn": "31810001",
                "hamc": "**",
                "hsc": "0001",
                "fault_desc": "显示处理链路异常",
                "repair_method": "按正式手册检查连接",
                "match_type": "exact_code",
                "source_ref": "HMC手册.xlsx#row=12",
            }
        ]
        state = {
            "intent": "exact_code",
            "query": "查询 31810001-**-0001",
            "rewritten_query": "查询 31810001-**-0001",
            "allowed_tools": ["hmc_query", "evidence_synthesis"],
            "sql_limit": 8,
        }
        with (
            patch.object(diagnosis_module, "_query_hmc", return_value=hmc_rows) as hmc_query,
            patch.object(diagnosis_module, "_query_fmeca", side_effect=AssertionError("FMECA 越过白名单")),
            patch.object(diagnosis_module, "_query_mfl", side_effect=AssertionError("MFL 越过白名单")),
            patch.object(diagnosis_module, "_query_diagrams", side_effect=AssertionError("图查询越过白名单")),
            patch.object(diagnosis_module, "_hybrid_search", side_effect=AssertionError("混合检索越过白名单")),
            patch.object(
                diagnosis_module,
                "_enrich_function_sources",
                side_effect=lambda rows, errors=None: list(rows),
            ),
        ):
            result = _run(diagnosis_module.icp_diagnosis_node(state))

        hmc_query.assert_called_once()
        self.assertEqual(
            ["hmc_query", "evidence_synthesis"],
            [item["tool"] for item in result["tool_results"]],
        )
        self.assertEqual("ok", result["tool_results"][0]["status"])
        self.assertEqual(1, len(result["candidate_faults"]))
        supporting = result["candidate_faults"][0]["supporting_evidence_ids"]
        self.assertEqual(1, len(supporting))
        self.assertEqual("hmc_fault", result["evidence_registry"][0]["claim_type"])
        self.assertNotIn(
            result["evidence_registry"][1]["evidence_id"],
            supporting,
            "维修动作不得反向支撑故障候选",
        )

    def test_opensearch_failure_and_genuine_empty_are_distinct(self):
        base_state = {
            "intent": "hybrid",
            "query": "显示闪烁",
            "rewritten_query": "显示闪烁",
            "allowed_tools": ["hybrid_search", "evidence_synthesis"],
            "hybrid_doc_types": ["hmc"],
        }
        with patch.object(
            diagnosis_module,
            "_hybrid_search",
            return_value=([], {"error": "index unavailable", "fallback": "统一检索失败"}),
        ):
            failed = _run(diagnosis_module.icp_diagnosis_node(base_state))
        with patch.object(
            diagnosis_module,
            "_hybrid_search",
            return_value=([], {"reranked": False}),
        ):
            empty = _run(diagnosis_module.icp_diagnosis_node(base_state))

        failed_result = next(item for item in failed["tool_results"] if item["tool"] == "hybrid_search")
        empty_result = next(item for item in empty["tool_results"] if item["tool"] == "hybrid_search")
        self.assertEqual("failed", failed_result["status"])
        self.assertEqual(["index unavailable"], failed_result["diagnostics"]["errors"])
        self.assertEqual("empty", empty_result["status"])
        self.assertEqual([], empty_result["diagnostics"]["errors"])

    def test_diagram_query_only_accepts_active_confirmed_published_revision(self):
        revisions = [
            {
                "revision_id": "pending",
                "title": "A",
                "review_status": "pending",
                "projection_status": "published",
                "is_active": True,
            },
            {
                "revision_id": "inactive",
                "title": "A",
                "review_status": "confirmed",
                "projection_status": "published",
                "is_active": False,
            },
            {
                "revision_id": "unpublished",
                "title": "A",
                "review_status": "confirmed",
                "projection_status": "not_published",
                "is_active": True,
            },
            {
                "revision_id": "usable",
                "title": "A",
                "review_status": "confirmed",
                "projection_status": "published",
                "is_active": True,
                "canonical_graph": _rbd_graph(),
            },
        ]
        with patch.object(
            diagnosis_module.diagram_repository,
            "list_active_revisions",
            return_value=revisions,
        ):
            result = diagnosis_module._query_diagrams("rbd", ["A"], 10)
        self.assertEqual(["usable"], [item["revision_id"] for item in result])

    def test_diagram_query_keeps_auto_validated_and_provisional_but_excludes_quarantine(self):
        base = {
            "title": "显示链路",
            "review_status": "pending",
            "projection_status": "published",
            "is_active": True,
            "canonical_graph": _rbd_graph(),
        }
        revisions = [
            {**base, "revision_id": "auto", "validation_status": "auto_validated"},
            {**base, "revision_id": "manual", "validation_status": "manual_confirmed"},
            {**base, "revision_id": "clue", "validation_status": "provisional"},
            {**base, "revision_id": "blocked", "validation_status": "quarantined"},
            {**base, "revision_id": "failed", "validation_status": "failed"},
        ]
        with patch.object(
            diagnosis_module.diagram_repository,
            "list_active_revisions",
            return_value=revisions,
        ):
            result = diagnosis_module._query_diagrams("rbd", ["显示链路"], 10)

        self.assertEqual(["auto", "manual", "clue"], [item["revision_id"] for item in result])

    def test_semantic_fta_locator_hydrates_current_postgres_revision_without_alias_rules(self):
        semantic_hit = {
            "doc_id": "doc-fta",
            "doc_name": "FTA分析.docx",
            "chunk_id": "semantic-hit-1",
            "title": "运行阶段非指令性重复复位",
            "content": "顶层事件由启动控制逻辑异常或供电瞬态触发。",
            "doc_type_kwd": "fta",
            "content_type": "fta_structured_facts",
            "projection_role_kwd": "diagram_locator",
            "diagram_id_kwd": "diagram-fta-1",
            "score": 0.91,
        }
        active_revision = {
            "diagram_id": "diagram-fta-1",
            "revision_id": "revision-fta-1",
            "diagram_type": "fta",
            "doc_id": "doc-fta",
            "title": "非指令性重复复位故障树",
            "diagram_key": "FTA-SYN-005",
            "validation_status": "auto_validated",
            "projection_status": "published",
            "is_active": True,
            "canonical_graph": {
                "root_fault_id": "top",
                "vertices": [
                    {"id": "top", "label": "TopEvent", "properties": {"name": "重复复位"}},
                ],
                "edges": [],
            },
        }
        state = {
            "intent": "cause_analysis",
            "query": "系统跑着跑着会自己一遍遍重新启动，可能哪里有问题？",
            "rewritten_query": "系统运行过程中自行多次重新启动的原因",
            "tool_queries": {
                "fta_query": ["系统运行过程中自行多次重新启动的原因链"],
                "hybrid_search": ["系统运行过程中自行多次重新启动的原因"],
            },
            "allowed_tools": ["fta_query", "hybrid_search", "evidence_synthesis"],
            "hybrid_doc_types": ["fta"],
            "sql_limit": 8,
            "retrieval_top_k": 8,
        }
        from backend.agent.current_model_selection import model_candidates, validate_selection
        selection = validate_selection({"status": "selected", "question_basis": state["query"], "selections": [
            {"doc_id": "doc-fta", "revision_id": "revision-fta-1", "matched_title": active_revision["title"]},
        ]}, model_candidates([active_revision], "fta"), query=state["query"])
        with (
            patch("backend.agent.diagram_query_scope.select_current_models", return_value=selection),
            patch.object(diagnosis_module, "_query_diagrams", return_value=[]),
            patch.object(
                diagnosis_module,
                "_hybrid_search",
                return_value=([semantic_hit], {"reranked": True}),
            ),
            patch.object(
                diagnosis_module.diagram_repository,
                "get_active_revision",
                return_value=active_revision,
            ),
            patch.object(
                diagnosis_module,
                "_enrich_function_sources",
                side_effect=lambda rows, errors=None: list(rows),
            ),
        ):
            result = _run(diagnosis_module.icp_diagnosis_node(state))

        self.assertEqual(["revision-fta-1"], [row["revision_id"] for row in result["confirmed_graphs"]])
        fta_call = next(item for item in result["tool_calls"] if item["tool"] == "fta_query")
        self.assertEqual(1, fta_call["semantic_hydrated_count"])
        self.assertTrue(
            any(
                item.get("record_id") == "FTA-SYN-005"
                and item.get("authority") == "validated_relation"
                and item.get("metadata", {}).get("revision_id") == "revision-fta-1"
                and item.get("metadata", {}).get("answer_fields", {}).get(
                    "structural_cause_chain"
                )
                for item in result["evidence_registry"]
            )
        )

    def test_graph_evidence_preserves_automatic_authority_boundary(self):
        base = {
            "diagram_id": "diagram-1",
            "doc_id": "doc-rbd",
            "title": "显示链路",
            "review_status": "pending",
            "projection_status": "published",
            "is_active": True,
            "graph": _rbd_graph(),
        }
        auto = diagnosis_module._direct_evidence(
            "rbd_query",
            [{**base, "revision_id": "auto", "validation_status": "auto_validated"}],
        )
        manual = diagnosis_module._direct_evidence(
            "rbd_query",
            [{**base, "revision_id": "manual", "validation_status": "manual_confirmed"}],
        )
        provisional = diagnosis_module._direct_evidence(
            "rbd_query",
            [{**base, "revision_id": "clue", "validation_status": "provisional"}],
        )
        quarantined = diagnosis_module._direct_evidence(
            "rbd_query",
            [{**base, "revision_id": "blocked", "validation_status": "quarantined"}],
        )

        self.assertEqual("validated_relation", auto[0]["authority"])
        self.assertEqual("auto_validated", auto[0]["review_status"])
        self.assertEqual("L2_CORROBORATED", auto[0]["evidence_level"])
        self.assertEqual("confirmed_relation", manual[0]["authority"])
        self.assertEqual("manual_confirmed", manual[0]["review_status"])
        self.assertEqual("L1_DIRECT", manual[0]["evidence_level"])
        self.assertEqual("unconfirmed_extraction", provisional[0]["authority"])
        self.assertEqual("provisional", provisional[0]["review_status"])
        self.assertEqual([], quarantined)

    def test_quarantined_graph_search_projection_is_not_reasoning_evidence(self):
        evidence = diagnosis_module._search_evidence(
            [
                {
                    "content_type": "fta_diagram",
                    "validation_status_kwd": "quarantined",
                    "doc_id": "doc-fta",
                    "chunk_id": "graph-1",
                    "content": "不得使用的图",
                }
            ]
        )

        self.assertEqual([], evidence)

    def test_only_hmc_and_fmeca_search_hits_can_become_fault_candidates(self):
        rows = [
            {"kb_id": "function_locator", "chunk_id": "f1", "title": "处理模块", "score": 1},
            {"kb_id": "fault_history", "chunk_id": "c1", "title": "一次历史处置", "score": 1},
            {"kb_id": "hmc", "chunk_id": "h1", "title": "显示链路异常", "score": 0.8},
        ]
        evidence = diagnosis_module._search_evidence(rows)
        candidates = diagnosis_module._candidate_faults([], [], rows, evidence)
        self.assertEqual(["显示链路异常"], [item["fault"] for item in candidates])

    def test_exact_postgres_hmc_candidate_outranks_duplicate_search_clues(self):
        hmc_rows = [
            {
                "doc_id": "doc-hmc",
                "hmc_code": "31810001**0001",
                "fault_desc": "精确代码故障",
                "repair_method": "",
                "match_type": "exact_code",
            }
        ]
        search_rows = [
            {
                "kb_id": "hmc",
                "chunk_id": f"chunk-{index}",
                "title": "相似检索故障",
                "score": 1.0,
            }
            for index in range(4)
        ]
        evidence = diagnosis_module._direct_evidence("hmc_query", hmc_rows)
        evidence.extend(diagnosis_module._search_evidence(search_rows))

        candidates = diagnosis_module._candidate_faults(
            hmc_rows,
            [],
            search_rows,
            evidence,
        )

        self.assertEqual("精确代码故障", candidates[0]["fault"])
        self.assertTrue(candidates[0]["exact_hmc_match"])
        search_candidate = next(item for item in candidates if item["fault"] == "相似检索故障")
        self.assertEqual(0.25, search_candidate["score"])

    def test_fmeca_code_search_hit_reinforces_hydrated_failure_mode(self):
        fmeca_rows = [
            {
                "doc_id": "doc-fmeca",
                "code": "FM-SYN-006",
                "fault_mode": "A路间歇停机",
            },
            {
                "doc_id": "doc-fmeca",
                "code": "FM-SYN-053",
                "fault_mode": "切换状态振荡",
            },
        ]
        search_rows = [
            {
                "kb_id": "fmeca",
                "chunk_id": "chunk-hot-power",
                "title": "FM-SYN-006",
                "score": 0.9,
            }
        ]
        evidence = diagnosis_module._direct_evidence("fmeca_query", fmeca_rows)
        evidence.extend(diagnosis_module._search_evidence(search_rows))

        candidates = diagnosis_module._candidate_faults(
            [], fmeca_rows, search_rows, evidence
        )

        self.assertEqual("A路间歇停机", candidates[0]["fault"])
        self.assertGreater(candidates[0]["score"], candidates[1]["score"])
        self.assertNotIn("FM-SYN-006", [item["fault"] for item in candidates])

    def test_component_matched_hmc_is_structured_authoritative_candidate(self):
        hmc_rows = [
            {
                "doc_id": "doc-hmc",
                "hmc_code": "91810001**0001",
                "fault_desc": "ICU 输入过压钳位失效",
                "repair_method": "检查输入采样与钳位器件",
                "match_type": "exact_components",
            }
        ]
        evidence = diagnosis_module._direct_evidence("hmc_query", hmc_rows)

        candidates = diagnosis_module._candidate_faults(hmc_rows, [], [], evidence)

        self.assertEqual(1, len(candidates))
        self.assertFalse(candidates[0]["exact_hmc_match"])
        self.assertTrue(candidates[0]["structured_hmc_match"])
        self.assertTrue(candidates[0]["authoritative_support"])


class CalculationAndGateIntegrationTests(unittest.TestCase):
    def test_calculation_recognizes_colon_diagram_keys_from_selected_anchors(self):
        keys = calculation_module._requested_graph_keys(
            {
                "query": "按刚才四条通路重新计算",
                "selected_conversation_anchors": [
                    {
                        "doc_type": "rbd",
                        "record_id": "RBD:显示输出功能可靠性模型",
                        "title": "显示输出功能可靠性模型",
                        "answer_fields": {
                            "diagram_key": "RBD:显示输出功能可靠性模型"
                        },
                    }
                ],
            },
            "rbd",
        )

        self.assertIn("rbd:显示输出功能可靠性模型", keys)
        self.assertNotIn("显示输出功能可靠性模型", keys)

    def test_calculation_parent_binds_business_diagram_key_to_revision(self):
        parent = make_evidence(
            source_kind="postgres",
            authority="validated_relation",
            evidence_level="L2_CORROBORATED",
            claim_type="fta_structure",
            doc_type="fta",
            doc_id="doc-fta",
            record_id="FTA-SYN-005",
            content="系统运行过程中出现非指令性重复复位的已验证结构",
            metadata={
                "diagram_key": "FTA-SYN-005",
                "validation_status": "auto_validated",
                "is_active": True,
                "projection_status": "published",
            },
        )
        row = {
            "diagram_type": "fta",
            "revision_id": "revision-uuid",
            "diagram_key": "FTA-SYN-005",
            "projection_status": "published",
            "is_active": True,
        }

        self.assertEqual(
            [parent["evidence_id"]],
            calculation_module._calculation_parent_evidence_ids(
                row, "fta", [parent]
            ),
        )

    def test_inactive_state_graph_is_never_calculated(self):
        state = {
            "query": "计算 RBD 可靠度",
            "calculation_authorizations": _calculation_authorization(
                "rbd_reliability", "rbd", "计算 RBD 可靠度"
            ),
            "entity_keywords": ["处理模块"],
            "allowed_tools": ["rbd_calculator"],
            "confirmed_graphs": [
                {
                    "diagram_type": "rbd",
                    "revision_id": "old-revision",
                    "review_status": "confirmed",
                    "projection_status": "published",
                    "is_active": False,
                    "graph": _rbd_graph(),
                }
            ],
        }
        with (
            patch.object(calculation_module, "_query_graphs", return_value=[]),
            patch.object(calculation_module, "calculate_rbd", side_effect=AssertionError("旧修订被计算")),
        ):
            result = _run(calculation_module.icp_calculation_node(state))
        self.assertEqual("partial", result["tool_results"][0]["status"])
        self.assertEqual([], result["calculation_results"])

    def test_provisional_and_quarantined_graphs_are_never_calculated(self):
        for validation_status in ("provisional", "quarantined"):
            with self.subTest(validation_status=validation_status):
                state = {
                    "query": "计算 RBD 可靠度",
                    "calculation_authorizations": _calculation_authorization(
                        "rbd_reliability", "rbd", "计算 RBD 可靠度"
                    ),
                    "entity_keywords": ["处理模块"],
                    "allowed_tools": ["rbd_calculator"],
                    "confirmed_graphs": [
                        {
                            "diagram_type": "rbd",
                            "revision_id": validation_status,
                            "review_status": "pending",
                            "validation_status": validation_status,
                            "projection_status": "published",
                            "is_active": True,
                            "graph": _rbd_graph(),
                        }
                    ],
                }
                with (
                    patch.object(calculation_module, "_query_graphs", return_value=[]),
                    patch.object(
                        calculation_module,
                        "calculate_rbd",
                        side_effect=AssertionError("非正式图被计算"),
                    ),
                ):
                    result = _run(calculation_module.icp_calculation_node(state))

                self.assertEqual("partial", result["tool_results"][0]["status"])
                self.assertEqual([], result["calculation_results"])

    def test_auto_validated_graph_calculation_has_approved_parent_chain(self):
        query = "计算 revision_id=revision-auto 的 RBD 可靠度"
        row = {
            "diagram_type": "rbd",
            "revision_id": "revision-auto",
            "doc_id": "doc-rbd",
            "doc_name": "RBD分析.docx",
            "image_title": "显示链路 RBD",
            "review_status": "pending",
            "validation_status": "auto_validated",
            "projection_status": "published",
            "is_active": True,
            "graph": _rbd_graph(),
        }
        graph_evidence = make_evidence(
            source_kind="postgres",
            authority="validated_relation",
            evidence_level="L2_CORROBORATED",
            claim_type="rbd_structure",
            doc_type="rbd",
            doc_id="doc-rbd",
            doc_name="RBD分析.docx",
            record_id="revision-auto",
            review_status="auto_validated",
            content="处理模块串联系统路径",
            metadata={
                "revision_id": "revision-auto",
                "validation_status": "auto_validated",
                "is_active": True,
                "projection_status": "published",
            },
        )
        state = {
            "intent": "reliability_query",
            "query": query,
            "calculation_authorizations": _calculation_authorization(
                "rbd_reliability", "rbd", query
            ),
            "allowed_tools": ["rbd_query", "rbd_calculator"],
            "confirmed_graphs": [row],
            "evidence_registry": [graph_evidence],
            "rbd_query_binding": _rbd_authority_binding(row, query),
        }

        calculated = _run(calculation_module.icp_calculation_node(state))
        derived = calculated["evidence_registry"][0]
        self.assertEqual([graph_evidence["evidence_id"]], derived["derived_from"])
        self.assertEqual("ok", calculated["tool_results"][0]["status"])

        gated = _run(
            evidence_gate_node(
                {
                    **state,
                    "evidence_registry": [graph_evidence, derived],
                    "tool_results": calculated["tool_results"],
                }
            )
        )
        self.assertEqual(
            [graph_evidence["evidence_id"], derived["evidence_id"]],
            gated["approved_evidence_ids"],
        )
        self.assertNotIn("formal_calculation", gated["safety_decision"]["blocked_claims"])

    def test_active_graph_calculation_has_complete_parent_chain_and_passes_gate(self):
        query = "计算 revision_id=revision-1 的 RBD 可靠度"
        row = {
            "diagram_type": "rbd",
            "revision_id": "revision-1",
            "doc_id": "doc-rbd",
            "doc_name": "RBD分析.docx",
            "image_title": "显示链路 RBD",
            "review_status": "confirmed",
            "projection_status": "published",
            "is_active": True,
            "graph": _rbd_graph(),
        }
        graph_evidence = make_evidence(
            source_kind="postgres",
            authority="confirmed_relation",
            evidence_level="L1_DIRECT",
            claim_type="rbd_structure",
            doc_type="rbd",
            doc_id="doc-rbd",
            doc_name="RBD分析.docx",
            record_id="revision-1",
            review_status="confirmed",
            content="处理模块串联系统路径",
            metadata={
                "revision_id": "revision-1",
                "validation_status": "manual_confirmed",
                "is_active": True,
                "projection_status": "published",
            },
        )
        state = {
            "intent": "reliability_query",
            "query": query,
            "calculation_authorizations": _calculation_authorization(
                "rbd_reliability", "rbd", query
            ),
            "allowed_tools": ["rbd_query", "rbd_calculator"],
            "confirmed_graphs": [row],
            "evidence_registry": [graph_evidence],
            "rbd_query_binding": _rbd_authority_binding(row, query),
        }
        calculated = _run(calculation_module.icp_calculation_node(state))
        derived = calculated["evidence_registry"][0]
        self.assertEqual([graph_evidence["evidence_id"]], derived["derived_from"])
        self.assertEqual("ok", calculated["tool_results"][0]["status"])

        gated = _run(
            evidence_gate_node(
                {
                    **state,
                    "evidence_registry": [graph_evidence, derived],
                    "tool_results": calculated["tool_results"],
                }
            )
        )
        self.assertEqual(
            [graph_evidence["evidence_id"], derived["evidence_id"]],
            gated["approved_evidence_ids"],
        )
        self.assertNotIn("formal_calculation", gated["safety_decision"]["blocked_claims"])

    def test_reconvergent_fta_cannot_pass_probability_gate(self):
        graph = {
            "root_fault_id": "top",
            "vertices": [
                {"id": "top", "label": "TopEvent", "properties": {"name": "TOP", "gate_type": "OR"}},
                {"id": "gate", "label": "Gate", "properties": {"name": "G", "gate_type": "AND"}},
                {"id": "a", "label": "BasicEvent", "properties": {"name": "A", "probability": 0.1}},
                {"id": "b", "label": "BasicEvent", "properties": {"name": "B", "probability": 0.2}},
            ],
            "edges": [
                {"source": "top", "target": "a"},
                {"source": "top", "target": "gate"},
                {"source": "gate", "target": "a"},
                {"source": "gate", "target": "b"},
            ],
        }
        row = {
            "diagram_type": "fta",
            "revision_id": "revision-shared-event",
            "doc_id": "doc-fta",
            "review_status": "confirmed",
            "projection_status": "published",
            "is_active": True,
            "graph": graph,
        }
        parent = make_evidence(
            source_kind="postgres",
            authority="confirmed_relation",
            evidence_level="L1_DIRECT",
            claim_type="fta_structure",
            doc_type="fta",
            doc_id="doc-fta",
            record_id="revision-shared-event",
            review_status="confirmed",
            content="A 在两个分支中共享",
            metadata={"is_active": True, "projection_status": "published"},
        )
        state = {
            "intent": "cause_analysis",
            "query": "计算 FTA 顶事件概率",
            "calculation_authorizations": _calculation_authorization(
                "fta_probability", "fta", "计算 FTA 顶事件概率"
            ),
            "allowed_tools": ["fta_query", "fta_analyzer"],
            "confirmed_graphs": [row],
            "evidence_registry": [parent],
        }

        calculated = _run(calculation_module.icp_calculation_node(state))
        derived = calculated["evidence_registry"][0]
        self.assertEqual("partial", calculated["tool_results"][0]["status"])
        self.assertIsNone(calculated["calculation_results"][0]["top_event_probability"])
        self.assertFalse(derived["metadata"]["numeric_complete"])

        gated = _run(
            evidence_gate_node(
                {
                    **state,
                    "evidence_registry": [parent, derived],
                    "tool_results": calculated["tool_results"],
                }
            )
        )
        self.assertIn("formal_numeric_result", gated["safety_decision"]["blocked_claims"])

    def test_graph_repository_failure_is_auditable(self):
        state = {
            "query": "计算显示链路 RBD 可靠度",
            "calculation_authorizations": _calculation_authorization(
                "rbd_reliability", "rbd", "计算显示链路 RBD 可靠度"
            ),
            "entity_keywords": ["显示链路"],
            "allowed_tools": ["rbd_calculator"],
            "confirmed_graphs": [],
        }
        with patch.object(
            calculation_module.diagram_repository,
            "list_active_revisions",
            side_effect=RuntimeError("postgres unavailable"),
        ):
            result = _run(calculation_module.icp_calculation_node(state))
        tool_result = result["tool_results"][0]
        self.assertEqual("failed", tool_result["status"])
        self.assertTrue(any("postgres unavailable" in value for value in tool_result["diagnostics"]["errors"]))

    def test_missing_rbd_parameters_cannot_become_formal_numeric_result(self):
        graph = _rbd_graph()
        graph["vertices"][1]["properties"].pop("reliability")
        row = {
            "diagram_type": "rbd",
            "revision_id": "revision-missing-param",
            "doc_id": "doc-rbd",
            "review_status": "confirmed",
            "projection_status": "published",
            "is_active": True,
            "graph": graph,
        }
        query = "计算 revision_id=revision-missing-param 的 RBD 系统可靠度"
        parent = make_evidence(
            source_kind="postgres",
            authority="confirmed_relation",
            evidence_level="L1_DIRECT",
            claim_type="rbd_structure",
            doc_type="rbd",
            doc_id="doc-rbd",
            record_id="revision-missing-param",
            review_status="confirmed",
            content="处理模块串联路径",
            metadata={
                "revision_id": "revision-missing-param",
                "validation_status": "manual_confirmed",
                "is_active": True,
                "projection_status": "published",
            },
        )
        state = {
            "intent": "reliability_query",
            "query": query,
            "calculation_authorizations": _calculation_authorization(
                "rbd_reliability", "rbd", query
            ),
            "allowed_tools": ["rbd_query", "rbd_calculator"],
            "confirmed_graphs": [row],
            "evidence_registry": [parent],
            "rbd_query_binding": _rbd_authority_binding(row, query),
        }
        calculated = _run(calculation_module.icp_calculation_node(state))
        derived = calculated["evidence_registry"][0]
        self.assertEqual("partial", calculated["tool_results"][0]["status"])
        self.assertIsNone(calculated["calculation_results"][0]["system_reliability"])
        self.assertFalse(derived["metadata"]["numeric_complete"])

        gated = _run(
            evidence_gate_node(
                {
                    **state,
                    "evidence_registry": [parent, derived],
                    "tool_results": calculated["tool_results"],
                }
            )
        )
        self.assertIn("formal_numeric_result", gated["safety_decision"]["blocked_claims"])
        self.assertTrue(gated["safety_decision"]["requires_clarification"])

    def test_truncated_rbd_paths_cannot_pass_numeric_gate(self):
        vertices = [
            {"id": "start", "label": "Start", "properties": {"name": "START"}},
            {"id": "end", "label": "End", "properties": {"name": "END"}},
        ]
        edges = []
        for index in range(13):
            block_id = f"b{index}"
            vertices.append(
                {
                    "id": block_id,
                    "label": "ReliabilityBlock",
                    "properties": {"name": f"B{index}", "reliability": 0.5},
                }
            )
            edges.extend(
                [
                    {"source": "start", "target": block_id},
                    {"source": block_id, "target": "end"},
                ]
            )
        row = {
            "diagram_type": "rbd",
            "revision_id": "revision-truncated",
            "doc_id": "doc-rbd",
            "review_status": "confirmed",
            "projection_status": "published",
            "is_active": True,
            "graph": {"vertices": vertices, "edges": edges},
        }
        query = "计算 revision_id=revision-truncated 的 RBD 系统可靠度"
        parent = make_evidence(
            source_kind="postgres",
            authority="confirmed_relation",
            evidence_level="L1_DIRECT",
            claim_type="rbd_structure",
            doc_type="rbd",
            doc_id="doc-rbd",
            record_id="revision-truncated",
            review_status="confirmed",
            content="13 条并联路径",
            metadata={
                "revision_id": "revision-truncated",
                "validation_status": "manual_confirmed",
                "is_active": True,
                "projection_status": "published",
            },
        )
        state = {
            "intent": "reliability_query",
            "query": query,
            "calculation_authorizations": _calculation_authorization(
                "rbd_reliability", "rbd", query
            ),
            "allowed_tools": ["rbd_query", "rbd_calculator"],
            "confirmed_graphs": [row],
            "evidence_registry": [parent],
            "rbd_query_binding": _rbd_authority_binding(row, query),
        }

        calculated = _run(calculation_module.icp_calculation_node(state))
        derived = calculated["evidence_registry"][0]
        self.assertEqual("partial", calculated["tool_results"][0]["status"])
        self.assertTrue(calculated["calculation_results"][0]["path_limit_reached"])
        self.assertIsNone(calculated["calculation_results"][0]["system_reliability"])
        self.assertFalse(derived["metadata"]["numeric_complete"])

        gated = _run(
            evidence_gate_node(
                {
                    **state,
                    "evidence_registry": [parent, derived],
                    "tool_results": calculated["tool_results"],
                }
            )
        )
        self.assertIn("formal_numeric_result", gated["safety_decision"]["blocked_claims"])


class AnswerEvidenceIntegrationTests(unittest.TestCase):
    def test_supplemental_history_requires_semantic_identity_hydration(self):
        broad_history = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="historical_case",
            doc_type="fault_history",
            record_id="HIS-UNRELATED",
            title="仅由宽泛关键词命中的历史",
            content="与当前故障无直接相似性的旧案例",
            match_type="structured_query",
        )
        relevant_history = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="historical_case",
            doc_type="fault_history",
            record_id="HIS-RELEVANT",
            title="语义定位后精确回填的历史",
            content="与当前现象、阶段和方向一致的旧案例",
            match_type="semantic_id_hydration",
            retrieval_score=0.86,
        )
        evidence = [broad_history, relevant_history]
        state = {
            "intent": "impact_analysis",
            "query_targets": [{"intent": "impact_analysis"}],
            "evidence_registry": evidence,
            "context_evidence_ids": [item["evidence_id"] for item in evidence],
            "safety_decision": {
                "context_evidence_ids": [item["evidence_id"] for item in evidence],
                "approved_evidence_ids": [item["evidence_id"] for item in evidence],
            },
        }

        selected = answer_module._context_evidence(state)

        self.assertEqual(
            [broad_history["evidence_id"], relevant_history["evidence_id"]],
            [item["evidence_id"] for item in selected],
        )

    def test_explicit_history_question_keeps_structured_history_rows(self):
        history = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="historical_case",
            doc_type="fault_history",
            record_id="HIS-001",
            content="按用户明确条件查到的历史记录",
            match_type="structured_query",
        )
        state = {
            "intent": "history_query",
            "query_targets": [{"intent": "history_query"}],
            "query_frame": {
                "required_answer_fields": ["history_result"],
                "response_focus": ["recovery_verification"],
            },
            "evidence_registry": [history],
            "context_evidence_ids": [history["evidence_id"]],
            "safety_decision": {
                "context_evidence_ids": [history["evidence_id"]],
                "approved_evidence_ids": [history["evidence_id"]],
            },
        }

        selected = answer_module._context_evidence(state)

        self.assertEqual([history["evidence_id"]], [item["evidence_id"] for item in selected])

    def test_history_window_obeys_persisted_branch_parent_binding(self):
        selected_mfl = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="maintenance_locator",
            doc_type="mfl",
            record_id="MFL-SYN-006",
            title="A路电源模块：A路间歇停机",
            content="A路受热后退出，冷却后恢复。",
            metadata={
                "answer_fields": {
                    "record_code": "MFL-SYN-006",
                    "module_failure": "A路电源模块：A路间歇停机",
                }
            },
        )
        thermal_fmeca = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            record_id="FM-SYN-067",
            title="温控闭环失效",
            content="高温高负载时发生过温降级。",
            metadata={
                "answer_fields": {
                    "record_code": "FM-SYN-067",
                    "product_flag": "TCM / FUNC-620",
                    "failure_mode": "温控闭环失效",
                }
            },
        )
        matching_history = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="historical_case",
            doc_type="fault_history",
            record_id="HIS-SYN-002",
            title="HIS-SYN-002",
            content="A路退出后B路接管，冷却后恢复。",
            match_type="semantic_id_hydration",
            metadata={
                "answer_fields": {
                    "case_id": "HIS-SYN-002",
                    "fault_symptom": "A路退出，B路自动接管，冷却后恢复。",
                    "related_module": "MOD-PWR-PSUA A路电源模块",
                    "related_function": "FUNC-120 A路稳压供电",
                    "explicit_references": {"mfl": ["MFL-SYN-006"]},
                }
            },
        )
        unrelated_history = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="historical_case",
            doc_type="fault_history",
            record_id="HIS-SYN-023",
            title="HIS-SYN-023",
            content="温控输出卡滞导致过温降级。",
            match_type="structured_query",
            metadata={
                "answer_fields": {
                    "case_id": "HIS-SYN-023",
                    "fault_symptom": "温度升高且风扇命令不变，随后过温降级。",
                    "related_module": "MOD-TCS-TCM 温度控制模块",
                    "related_function": "FUNC-620 温度控制与保护",
                    "explicit_references": {"fmeca": ["FM-SYN-067"]},
                }
            },
        )
        for item, score in (
            (selected_mfl, 0.60),
            (thermal_fmeca, 0.90),
            (matching_history, 0.10),
            (unrelated_history, 0.80),
        ):
            item["answer_rerank_score"] = score
        evidence = [
            selected_mfl,
            thermal_fmeca,
            unrelated_history,
            matching_history,
        ]
        evidence_ids = [item["evidence_id"] for item in evidence]
        state = {
            "query": (
                "主供电A路受热后退出、另一路接管，冷却后恢复；"
                "以前的相近案例怎么验证？"
            ),
            "intent": "maintenance_query",
            "query_targets": [
                {"intent": "maintenance_query"},
                {"intent": "history_query"},
            ],
            "query_frame": {
                "required_answer_fields": [
                    "maintenance_action",
                    "history_result",
                ],
                "diagnostic_closure": {
                    "required": True,
                    "required_dimensions": [
                        "maintenance_action",
                        "history_result",
                    ],
                    "basis": "以前的相近案例怎么验证",
                },
            },
            "selected_conversation_anchors": [
                {"doc_type": "mfl", "record_id": "MFL-SYN-006"},
                {"doc_type": "fmeca", "record_id": "FM-SYN-067"},
            ],
            # Upstream retrieval has already bound the current branch to this
            # exact history parent.  The answer window must reuse that durable
            # decision instead of reselecting the higher-scoring neighbour.
            "history_parent_binding": {
                "version": 1,
                "case_ids": ["HIS-SYN-002"],
                "row_ids": [],
                "source": "selected_branch_anchor",
                "authority": "identity",
                "comparison": False,
                "ambiguous": False,
                "conflict": False,
                "authorizes_exact_links": True,
            },
            "safety_decision": {"approved_evidence_ids": evidence_ids},
        }

        selected = _source_answer_evidence_window._select_evidence_window(state, evidence, limit=12)
        selected_records = [item["record_id"] for item in selected]

        self.assertIn("HIS-SYN-002", selected_records)
        self.assertIn("HIS-SYN-023", selected_records)
        self.assertEqual(["HIS-SYN-002"], state["history_parent_binding"]["case_ids"])


    def test_context_and_citations_share_deduplicated_gate_selection(self):
        original = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            doc_id="doc-1",
            doc_name="HMC手册.xlsx",
            record_id="HMC-001",
            title="HMC-001",
            content="旧内容",
        )
        updated = {**original, "content": "显示链路故障"}
        omitted = make_evidence(
            source_kind="opensearch",
            authority="search_projection",
            evidence_level="L3_INFERRED",
            claim_type="retrieval_hit",
            doc_type="fmeca",
            chunk_id="chunk-2",
            title="仅线索",
            content="不应进入本次上下文",
        )
        state = {
            "diagnostic_context": "旧上下文不应进入提示 [^99]",
            "evidence_registry": [original, omitted, updated],
            "context_evidence_ids": [original["evidence_id"]],
            "safety_decision": {
                "approved_evidence_ids": [original["evidence_id"]],
                "blocked_evidence_ids": [omitted["evidence_id"]],
                "context_evidence_ids": [original["evidence_id"]],
            },
        }
        context = answer_module._format_context(state)
        citations = answer_module._extract_citations(state)
        packets = [json.loads(line) for line in context.splitlines() if line.startswith('{"record_identity":')]
        self.assertEqual(1, len(packets))
        self.assertEqual("HMC-001", packets[0]["record_identity"]["record_id"])
        self.assertEqual([1], [claim["citation_id"] for claim in packets[0]["claims"]])
        self.assertIn("显示链路故障", context)
        self.assertNotIn("旧上下文不应进入", context)
        self.assertNotIn("不应进入本次上下文", context)
        self.assertEqual(1, len(citations))
        self.assertEqual("HMC手册.xlsx", citations[0]["doc_name"])
        self.assertEqual("HMC-001", citations[0]["source_ref"])



    def test_citation_numbers_restart_per_answer_but_evidence_ids_do_not(self):
        first = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            record_id="HMC-FIRST",
            title="第一轮证据",
            content="第一轮事实",
        )
        second = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            record_id="FM-SECOND",
            title="第二轮证据",
            content="第二轮事实",
        )
        first_citations = answer_module._extract_citations(
            {"evidence_registry": [first]}
        )
        second_citations = answer_module._extract_citations(
            {"evidence_registry": [second]}
        )

        self.assertEqual([1], [item["id"] for item in first_citations])
        self.assertEqual([1], [item["id"] for item in second_citations])
        self.assertNotEqual(
            first_citations[0]["evidence_id"],
            second_citations[0]["evidence_id"],
        )


    def test_factual_headings_and_maintenance_actions_require_citations(self):
        self.assertTrue(
            _source_answer_claim_units._line_is_factual_assertion("### 电源故障会导致系统重启")
        )
        self.assertTrue(
            _source_answer_claim_units._line_is_factual_assertion("请检查并更换电源模块。")
        )
        self.assertFalse(_source_answer_claim_units._line_is_factual_assertion("### 原因分析"))
        self.assertTrue(_source_answer_claim_units._line_is_factual_assertion("请补充故障代码。"))
        self.assertFalse(
            _source_answer_claim_units._line_is_factual_assertion(
                "请补充故障代码。",
                {"请补充故障代码。"},
            )
        )
        self.assertTrue(_source_answer_claim_units._line_is_factual_assertion("其等效串联电阻会增大。"))











    def test_bounded_catalog_reserves_every_exposed_tool_call_evidence_id(self):
        evidence = [
            make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type="fmeca_fact",
                doc_type="fmeca",
                record_id=f"FM-{index}",
                title=f"证据 {index}",
                content=f"证据内容 {index}",
            )
            for index in range(answer_module.EVIDENCE_CATALOG_LIMIT + 10)
        ]
        exposed_ids = [item["evidence_id"] for item in evidence[-8:]]
        catalog = answer_module._extract_evidence_catalog(
            {
                "evidence_registry": evidence,
                "tool_calls": [{"tool": "fmeca_query", "evidence_ids": exposed_ids}],
                "tool_results": [{
                    "tool": "fmeca_query",
                    "evidence_ids": [item["evidence_id"] for item in evidence],
                }],
            },
            priority_evidence_ids=[evidence[-9]["evidence_id"]],
        )

        catalog_ids = {item["evidence_id"] for item in catalog}
        self.assertEqual(answer_module.EVIDENCE_CATALOG_LIMIT, len(catalog))
        self.assertIn(evidence[-9]["evidence_id"], catalog_ids)
        self.assertTrue(set(exposed_ids).issubset(catalog_ids))

    def test_evidence_catalog_payload_is_bounded_and_json_serializable(self):
        class CustomValue:
            def __str__(self):
                return "自定义值" * 1000

        payload = answer_module._evidence_payload(
            {
                "evidence_id": "ev-bounded",
                "title": "标题" * 1000,
                "source_ref": "文档定位" * 1000,
                "content": "正文" * 3000,
                "metadata": {
                    **{f"key-{index}": "值" * 1000 for index in range(80)},
                    "nested": {"a": {"b": {"c": {"d": CustomValue()}}}},
                    "values": set(range(80)),
                    "custom": CustomValue(),
                    "binary": b"abc\xff",
                },
                "derived_from": [f"ev-{index}-" + "x" * 500 for index in range(60)],
                "limitations": ["限制" * 1000 for _ in range(60)],
            },
            content_limit=1800,
        )

        json.dumps(payload, ensure_ascii=False)
        self.assertLessEqual(len(payload["title"]), 500)
        self.assertLessEqual(len(payload["source_ref"]), 800)
        self.assertLessEqual(len(payload["content"]), 1800)
        self.assertLessEqual(
            len(payload["document_metadata"]),
            _source_answer_payload.EVIDENCE_METADATA_MAX_ITEMS,
        )
        self.assertLessEqual(
            len(payload["derived_from"]),
            _source_answer_payload.EVIDENCE_METADATA_MAX_LIST_ITEMS,
        )
        self.assertTrue(all(len(value) <= 240 for value in payload["derived_from"]))
        self.assertLessEqual(
            len(payload["limitations"]),
            _source_answer_payload.EVIDENCE_METADATA_MAX_LIST_ITEMS,
        )
        self.assertTrue(all(len(value) <= 600 for value in payload["limitations"]))



    def test_citation_removal_drops_orphan_section_and_hides_number_gaps(self):
        answer = _source_answer_text._cleanup_numbered_bold_sections(
            "开场结论。[^1]\n\n"
            "**2. 复位控制逻辑故障**[^1]\n\n"
            "**3. 散热与环境因素**\n"
            "* 风扇故障会导致过热。[^2]"
        )

        self.assertNotIn("复位控制逻辑故障", answer)
        self.assertNotIn("**3.", answer)
        self.assertIn("**散热与环境因素**", answer)
        self.assertIn("风扇故障会导致过热。[^2]", answer)


    def test_cleanup_removes_orphan_heading_and_incomplete_table(self):
        answer = _source_answer_text._cleanup_generated_markdown_structure(
            "### 原因分析\n\n"
            "### 仍有内容\n"
            "有证据的正文。[^1]\n\n"
            "| 字段 | 内容 |\n"
            "| --- | --- |"
        )

        self.assertNotIn("原因分析", answer)
        self.assertIn("### 仍有内容", answer)
        self.assertNotIn("| --- | --- |", answer)
        self.assertNotIn("| 字段 | 内容 |", answer)

    def test_cleanup_renumbers_remaining_top_level_sections(self):
        answer = _source_answer_text._cleanup_generated_markdown_structure(
            "### 1. 状态查询\n内容一。[^1]\n\n"
            "### 3. 原因与影响\n内容二。[^2]\n\n"
            "#### 情形 A\n内容三。[^2]\n\n"
            "### 5. 历史参考\n内容四。[^3]"
        )

        self.assertIn("### 1. 状态查询", answer)
        self.assertIn("### 2. 原因与影响", answer)
        self.assertIn("### 3. 历史参考", answer)
        self.assertIn("#### 情形 A", answer)
        self.assertNotIn("### 5.", answer)








class _CapturingCursor:
    def __init__(self, connection):
        self.connection = connection
        self.description = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=()):
        self.connection.calls.append((" ".join(query.split()), tuple(params or ())))

    def fetchall(self):
        return []

class _CapturingConnection:
    def __init__(self):
        self.calls = []

    def cursor(self):
        return _CapturingCursor(self)

    def rollback(self):
        pass


if __name__ == "__main__":
    unittest.main()
