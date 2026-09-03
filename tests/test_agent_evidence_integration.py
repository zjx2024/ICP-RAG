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
from backend.agent.tools import icp_calculation as calculation_module
from backend.agent.tools import icp_diagnosis as diagnosis_module


def _run(coroutine):
    return asyncio.run(coroutine)


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


class DiagnosisIntegrationTests(unittest.TestCase):
    def test_hmc_symptom_item_cannot_support_a_repair_action(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            doc_id="doc-hmc",
            record_id="91810001**0002",
            title="91810001**0002",
            content="故障现象：ICU 输入滤波性能退化",
            metadata={
                "answer_fields": {
                    "maintenance_action": "检查滤波器件，必要时更换输入调理模块。"
                }
            },
        )
        bindings = [{"citation_id": 1, "quote": "检查滤波器件，必要时更换输入调理模块。"}]

        self.assertEqual(
            [],
            answer_module._verified_bindings_for_unit(
                "检查滤波器件，必要时更换输入调理模块。",
                bindings,
                {1: evidence},
            ),
        )

    def test_specific_module_examples_require_all_named_identities_in_sources(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-SYN-002",
            title="输入滤波性能退化",
            content="输入纹波和传导干扰增大，多个子系统出现间歇性异常。",
        )
        bindings = [{"citation_id": 1, "quote": "多个子系统出现间歇性异常"}]

        self.assertEqual(
            [],
            answer_module._verified_bindings_for_unit(
                "PDU 误跳闸、GPP-A 存储校验错和 TCM 门限漂移均由该故障引起。",
                bindings,
                {1: evidence},
            ),
        )

    def test_failure_mode_identity_prevents_cross_mode_impact_binding(self):
        sensor_open = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-SYN-058",
            title="传感器通道开路",
            content="传感器通道开路会导致相关保护或健康判断失去依据。",
        )
        saturation = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-SYN-059",
            title="A/D 采样饱和",
            content="A/D 采样饱和会导致采样结果失真，健康监测可能误判。",
        )
        unit = "A/D 采样饱和会导致采样结果失真，健康监测可能误判。"

        self.assertEqual(
            [],
            answer_module._verified_bindings_for_unit(
                unit,
                [{"citation_id": 1, "quote": "相关保护或健康判断失去依据"}],
                {1: sensor_open},
            ),
        )
        self.assertEqual(
            1,
            len(
                answer_module._verified_bindings_for_unit(
                    unit,
                    [{"citation_id": 2, "quote": "采样结果失真，健康监测可能误判"}],
                    {2: saturation},
                )
            ),
        )

    def test_open_circuit_fmeca_cannot_support_ad_saturation_causes(self):
        sensor_open = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-SYN-058",
            title="传感器通道开路",
            content=(
                "故障模式：传感器通道开路。"
                "故障原因：传感器线路开路、连接器松动或输入保护损坏。"
                "检测方法：检查原始 A/D 计数、开路诊断和连接状态。"
            ),
        )
        saturation_claim = (
            "A/D 采样饱和：若原始计数达到上限且开路诊断正常，"
            "则候选原因包括输入超量程、增益配置错误或转换器异常。"
        )

        self.assertEqual(
            [],
            answer_module._verified_bindings_for_unit(
                saturation_claim,
                [
                    {
                        "citation_id": 1,
                        "quote": sensor_open["content"],
                    }
                ],
                {1: sensor_open},
            ),
        )

    def test_verified_bindings_apply_each_quotes_long_claim_guard(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-SYN-002",
            title="输入滤波性能退化",
            content=(
                "故障原因：滤波电容老化、连接阻抗增大或屏蔽接地不良。"
                "局部影响：输入纹波和传导干扰增大。"
                "最终影响：多个子系统出现间歇性异常。"
            ),
        )
        long_claim = (
            "输入滤波性能退化的原因包括滤波电容老化、连接阻抗增大或屏蔽接地不良；"
            "局部影响是输入纹波和传导干扰增大，最终会使多个子系统出现间歇性异常，"
            "并进一步造成负载配电保护误跳闸、存储校验错误和复位信号卡滞。"
        )
        bindings = [
            {
                "citation_id": 1,
                "quote": "滤波电容老化、连接阻抗增大或屏蔽接地不良",
            },
            {"citation_id": 1, "quote": "输入纹波和传导干扰增大"},
            {"citation_id": 1, "quote": "多个子系统出现间歇性异常"},
        ]

        self.assertEqual(
            [],
            answer_module._verified_bindings_for_unit(
                long_claim,
                bindings,
                {1: evidence},
            ),
        )

    def test_named_architecture_source_requires_function_locator_evidence(self):
        fmeca = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-SYN-038",
            title="NGW 突发丢包",
            content="NGW 在高负载下发生突发丢包。",
        )
        unit = "根据系统架构说明，NGW 在高负载下会发生突发丢包。"

        self.assertEqual(
            [],
            answer_module._verified_bindings_for_unit(
                unit,
                [{"citation_id": 1, "quote": "NGW 在高负载下发生突发丢包"}],
                {1: fmeca},
            ),
        )

    def test_history_wording_cannot_bind_to_fmeca_or_mfl_only(self):
        fmeca = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-SYN-002",
            title="输入滤波性能退化",
            content="建议检查滤波器件，必要时更换输入调理模块。",
        )
        unit = "以前曾记录过该故障，最终通过更换输入调理模块解决。"

        self.assertEqual(
            [],
            answer_module._deterministic_unit_citations(unit, {1: fmeca}, preferred=[1]),
        )

    def test_opposite_payload_direction_cannot_bind_as_same_fact(self):
        ingress = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="historical_case",
            doc_type="fault_history",
            doc_id="doc-history",
            record_id="HIS-SYN-011",
            title="A 通道接收历史",
            content="外部设备持续发送数据，但系统 A 通道接收计数长期不变。",
        )
        unit = "历史案例表明，内部已经产生报文但外部设备收不到数据。"

        self.assertEqual(
            [],
            answer_module._deterministic_unit_citations(unit, {1: ingress}, preferred=[1]),
        )

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
                limit_per_type=1,
                limits_by_type=diagnosis_module._DIAGNOSTIC_RECORD_HYDRATION_LIMITS,
            )

        self.assertEqual(3, lookup.call_count)
        for call in lookup.call_args_list:
            self.assertEqual(2, call.kwargs["limit"])
            self.assertEqual(2, len(call.args[1]))
        self.assertEqual(
            ["91850002**0001", "91850002**0002"],
            [row["hmc_code"] for row in hydrated["hmc_query"]],
        )
        self.assertEqual(
            ["FM-SYN-058", "FM-SYN-059"],
            [row["code"] for row in hydrated["fmeca_query"]],
        )
        self.assertEqual(
            ["MFL-SYN-058", "MFL-SYN-059"],
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
        with (
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
        self.assertEqual("empty", result["tool_results"][0]["status"])
        self.assertEqual([], result["calculation_results"])

    def test_provisional_and_quarantined_graphs_are_never_calculated(self):
        for validation_status in ("provisional", "quarantined"):
            with self.subTest(validation_status=validation_status):
                state = {
                    "query": "计算 RBD 可靠度",
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

                self.assertEqual("empty", result["tool_results"][0]["status"])
                self.assertEqual([], result["calculation_results"])

    def test_auto_validated_graph_calculation_has_approved_parent_chain(self):
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
                "validation_status": "auto_validated",
                "is_active": True,
                "projection_status": "published",
            },
        )
        state = {
            "intent": "reliability_query",
            "query": "计算 RBD 可靠度",
            "allowed_tools": ["rbd_query", "rbd_calculator"],
            "confirmed_graphs": [row],
            "evidence_registry": [graph_evidence],
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
            metadata={"is_active": True, "projection_status": "published"},
        )
        state = {
            "intent": "reliability_query",
            "query": "计算 RBD 可靠度",
            "allowed_tools": ["rbd_query", "rbd_calculator"],
            "confirmed_graphs": [row],
            "evidence_registry": [graph_evidence],
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
            metadata={"is_active": True, "projection_status": "published"},
        )
        state = {
            "intent": "reliability_query",
            "query": "计算 RBD 系统可靠度",
            "allowed_tools": ["rbd_query", "rbd_calculator"],
            "confirmed_graphs": [row],
            "evidence_registry": [parent],
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
            metadata={"is_active": True, "projection_status": "published"},
        )
        state = {
            "intent": "reliability_query",
            "query": "计算 RBD 系统可靠度",
            "allowed_tools": ["rbd_query", "rbd_calculator"],
            "confirmed_graphs": [row],
            "evidence_registry": [parent],
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
            [relevant_history["evidence_id"]],
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
            "evidence_registry": [history],
            "context_evidence_ids": [history["evidence_id"]],
            "safety_decision": {
                "context_evidence_ids": [history["evidence_id"]],
                "approved_evidence_ids": [history["evidence_id"]],
            },
        }

        selected = answer_module._context_evidence(state)

        self.assertEqual([history["evidence_id"]], [item["evidence_id"] for item in selected])

    def test_impact_answer_is_direct_llm_summary_without_diagnosis_template(self):
        impact = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            doc_name="FMECA.xlsx",
            record_id="FM-SYN-002",
            title="输入滤波性能退化",
            content=(
                "故障原因=滤波电容老化、连接阻抗增大或屏蔽接地不良；"
                "局部影响=输入纹波和传导干扰增大；最终影响=多个子系统间歇异常；严酷度=III"
            ),
        )
        state = {
            "intent": "impact_analysis",
            "query": "滤波电容有点老化了，会有什么影响吗？",
            "messages": [],
            "evidence_registry": [impact],
            "context_evidence_ids": [impact["evidence_id"]],
            "candidate_faults": [{"fault": "不应向用户展示的内部候选", "score": 0.6}],
            "diagnostic_support": 0.5,
            "clarification_question": "",
            "safety_decision": {
                "status": "answer",
                "can_claim_unique": False,
                "questions": [],
                "approved_evidence_ids": [impact["evidence_id"]],
                "blocked_evidence_ids": [],
                "blocked_claims": [],
                "context_evidence_ids": [impact["evidence_id"]],
            },
        }
        verbose_model_answer = (
            "### 初步判断\n"
            "滤波电容老化会使输入滤波能力下降，输入纹波和传导干扰增大，并可能引发多个子系统间歇异常；资料严酷度为 III。[^1]\n\n"
            "### 资料明确记载\n"
            "FMECA 将其记录为输入滤波性能退化。[^1]\n\n"
            "### 建议检查\n"
            "请补充告警码并执行完整排故。\n\n"
            "### 下一步需要确认\n"
            "请补充故障阶段。\n\n"
            "### 适用性说明\n"
            "以上内容需要人工复核。"
        )
        repair = json.dumps(
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "bindings": [
                            {
                                "citation_id": 1,
                                "quote": (
                                    "故障原因=滤波电容老化、连接阻抗增大或屏蔽接地不良；"
                                    "局部影响=输入纹波和传导干扰增大；"
                                    "最终影响=多个子系统间歇异常；严酷度=III"
                                ),
                            }
                        ],
                    },
                    {
                        "unit_id": "u2",
                        "bindings": [{"citation_id": 1, "quote": "输入滤波性能退化"}],
                    },
                ]
            },
            ensure_ascii=False,
        )
        with (
            patch.object(
                answer_module.llm_service,
                "_call_model",
                side_effect=[verbose_model_answer, repair],
            ) as call,
            patch.object(
                answer_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(answer_module.answer_node(state))

        prompt = call.call_args_list[0].args[0][0]["content"]
        answer = result["final_answer"]
        self.assertIn("第一段直接、准确回答", prompt)
        self.assertIn("不得展示成固定章节", prompt)
        self.assertNotIn("候选故障：\n", prompt)
        self.assertIn("输入纹波和传导干扰增大", answer)
        self.assertNotIn("初步判断", answer)
        self.assertNotIn("资料明确记载", answer)
        self.assertNotIn("建议检查", answer)
        self.assertNotIn("下一步需要确认", answer)
        self.assertNotIn("适用性说明", answer)

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
        self.assertIn("[^1] HMC-001", context)
        self.assertIn("显示链路故障", context)
        self.assertNotIn("旧上下文不应进入", context)
        self.assertNotIn("不应进入本次上下文", context)
        self.assertEqual(1, len(citations))
        self.assertEqual("HMC手册.xlsx", citations[0]["doc_name"])
        self.assertEqual("HMC-001", citations[0]["source_ref"])

    def test_cause_answer_reserves_formal_fta_in_crowded_evidence_window(self):
        hmc_evidence = [
            make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type="hmc_fault",
                doc_type="hmc",
                doc_id="doc-hmc",
                doc_name="HMC手册.docx",
                record_id=f"HMC-{index:03d}",
                title=f"HMC-{index:03d}",
                content=f"第 {index} 条 HMC 故障现象",
            )
            for index in range(16)
        ]
        fmeca_evidence = [
            make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type="fmeca_fact",
                doc_type="fmeca",
                doc_id="doc-fmeca",
                doc_name="FMECA分析报告.docx",
                record_id=f"FM-{index:03d}",
                title=f"FM-{index:03d}",
                content=f"第 {index} 条 FMECA 原因记录",
            )
            for index in range(12)
        ]
        fta = make_evidence(
            source_kind="postgres",
            authority="validated_relation",
            evidence_level="L2_CORROBORATED",
            claim_type="fta_structure",
            doc_type="fta",
            doc_id="doc-fta",
            doc_name="FTA故障树分析.docx",
            record_id="FTA-RESTART-001",
            title="系统非指令性重复复位故障树",
            content="顶事件=系统运行中非指令性重复复位；原因链包含电源异常与复位控制异常。",
            review_status="auto_validated",
            metadata={
                "validation_status": "auto_validated",
                "is_active": True,
                "projection_status": "published",
            },
        )
        registry = [*hmc_evidence, *fmeca_evidence, fta]
        base_state = {
            "intent": "cause_analysis",
            "query_targets": [{"intent": "cause_analysis"}],
            "query": "系统运行中为什么会反复自动重启？",
            "messages": [],
            "allowed_tools": ["hmc_query", "fmeca_query", "fta_query"],
            "evidence_registry": registry,
            "candidate_faults": [],
            "diagnostic_support": 0.7,
            "clarification_question": "",
        }
        gated = _run(evidence_gate_node(base_state))
        self.assertIn(fta["evidence_id"], gated["approved_evidence_ids"])
        state = {**base_state, **gated}

        repair = json.dumps(
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "bindings": [
                            {
                                "citation_id": 1,
                                "quote": "原因链包含电源异常与复位控制异常",
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        )
        with (
            patch.object(
                answer_module.llm_service,
                "_call_model",
                side_effect=["资料中的故障树记录了重复复位原因链。[^1]", repair],
            ) as call,
            patch.object(
                answer_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(answer_module.answer_node(state))

        citations = result["citations"]
        self.assertLessEqual(len(citations), 25)
        fta_citation = next(
            item for item in citations if item["evidence_id"] == fta["evidence_id"]
        )
        self.assertEqual("fta", fta_citation["doc_type"])
        self.assertEqual("validated_relation", fta_citation["authority"])
        self.assertIn("fmeca", {item["doc_type"] for item in result["evidence_catalog"]})
        self.assertEqual(1, len(citations))
        self.assertEqual("answer_used", result["evidence_catalog"][0]["status"])
        prompt = call.call_args_list[0].args[0][0]["content"]
        self.assertIn("系统非指令性重复复位故障树", prompt)
        self.assertIn("原因链包含电源异常与复位控制异常", prompt)
        self.assertIn(
            f"[^{fta_citation['id']}] 系统非指令性重复复位故障树｜fta",
            prompt,
        )

    def test_uncited_summary_is_bound_by_verified_quote_without_global_warning(self):
        fta = make_evidence(
            source_kind="postgres",
            authority="validated_relation",
            evidence_level="L2_CORROBORATED",
            claim_type="fta_structure",
            doc_type="fta",
            doc_id="doc-fta",
            doc_name="FTA故障树分析报告.docx",
            record_id="FTA-RESTART-001",
            title="系统非指令性重复复位",
            content=(
                "系统运行过程中出现非指令性重复复位。"
                "复位控制异常会导致系统反复复位。"
            ),
            review_status="auto_validated",
            metadata={
                "validation_status": "auto_validated",
                "is_active": True,
                "projection_status": "published",
            },
        )
        state = {
            "intent": "cause_analysis",
            "query": "系统为什么反复重启？",
            "messages": [],
            "evidence_registry": [fta],
            "context_evidence_ids": [fta["evidence_id"]],
            "candidate_faults": [],
            "diagnostic_support": 0.8,
            "clarification_question": "",
            "safety_decision": {
                "status": "answer",
                "can_claim_unique": False,
                "approved_evidence_ids": [fta["evidence_id"]],
                "blocked_evidence_ids": [],
                "context_evidence_ids": [fta["evidence_id"]],
                "questions": [],
            },
        }
        draft = (
            "根据 FTA 记录，系统反复复位与复位控制异常有关。\n\n"
            "综上，复位控制异常会导致系统反复复位。"
        )
        repair = json.dumps(
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "bindings": [
                            {
                                "citation_id": 1,
                                "quote": "系统运行过程中出现非指令性重复复位",
                            }
                        ],
                    },
                    {
                        "unit_id": "u2",
                        "bindings": [
                            {
                                "citation_id": 1,
                                "quote": "复位控制异常会导致系统反复复位",
                            }
                        ],
                    },
                ]
            },
            ensure_ascii=False,
        )
        with (
            patch.object(
                answer_module.llm_service,
                "_call_model",
                side_effect=[draft, repair],
            ) as call,
            patch.object(
                answer_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(answer_module.answer_node(state))

        self.assertEqual(2, call.call_count)
        self.assertEqual(2, result["final_answer"].count("[^1]"))
        self.assertNotIn("部分说明未附可直接核验的来源", result["final_answer"])
        self.assertIn("根据 FTA 记录", result["final_answer"])
        self.assertIn("综上", result["final_answer"])

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

    def test_model_supplied_current_marker_is_revalidated_before_output(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            record_id="HMC-001",
            title="显示链路异常",
            content="显示链路出现间歇异常。",
        )
        state = {
            "intent": "cause_analysis",
            "query": "为什么反复重启？",
            "messages": [],
            "evidence_registry": [evidence],
            "context_evidence_ids": [evidence["evidence_id"]],
            "candidate_faults": [],
            "diagnostic_support": 0.4,
            "clarification_question": "",
            "safety_decision": {
                "status": "answer",
                "can_claim_unique": False,
                "approved_evidence_ids": [evidence["evidence_id"]],
                "blocked_evidence_ids": [],
                "context_evidence_ids": [evidence["evidence_id"]],
                "questions": [],
            },
        }
        unsupported_draft = "电源模块已经损坏，会导致系统反复重启。[^1]"
        rejected_binding = json.dumps(
            {"units": [{"unit_id": "u1", "bindings": []}]},
            ensure_ascii=False,
        )
        with (
            patch.object(
                answer_module.llm_service,
                "_call_model",
                side_effect=[unsupported_draft, rejected_binding],
            ) as call,
            patch.object(
                answer_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(answer_module.answer_node(state))

        self.assertEqual(2, call.call_count)
        self.assertNotIn("电源模块已经损坏", result["final_answer"])
        self.assertNotIn("[^1]", result["final_answer"])
        self.assertEqual(answer_module._NO_SUPPORTED_FACTS_MESSAGE, result["final_answer"])

    def test_factual_headings_and_maintenance_actions_require_citations(self):
        self.assertTrue(
            answer_module._line_is_factual_assertion("### 电源故障会导致系统重启")
        )
        self.assertTrue(
            answer_module._line_is_factual_assertion("请检查并更换电源模块。")
        )
        self.assertFalse(answer_module._line_is_factual_assertion("### 原因分析"))
        self.assertTrue(answer_module._line_is_factual_assertion("请补充故障代码。"))
        self.assertFalse(
            answer_module._line_is_factual_assertion(
                "请补充故障代码。",
                {"请补充故障代码。"},
            )
        )
        self.assertTrue(answer_module._line_is_factual_assertion("其等效串联电阻会增大。"))

    def test_safety_reminder_with_model_marker_is_kept_without_fake_binding(self):
        state = {
            "evidence_registry": [],
            "safety_decision": {"approved_evidence_ids": []},
        }
        answer = _run(
            answer_module._repair_answer_citations(
                "实际处置以适用构型和正式维修程序为准。[^1]",
                state,
                None,
                citation_count=1,
            )
        )

        self.assertEqual("实际处置以适用构型和正式维修程序为准。", answer)

    def test_applicability_notice_is_appended_once_by_backend(self):
        answer = answer_module._ensure_applicability_notice(
            f"已核验结论。[^1]\n\n{answer_module._APPLICABILITY_NOTICE}",
            "fault_diagnosis",
        )

        self.assertEqual(1, answer.count(answer_module._APPLICABILITY_NOTICE))
        self.assertTrue(answer.endswith(answer_module._APPLICABILITY_NOTICE))

    def test_semantic_applicability_notice_is_replaced_by_one_canonical_footer(self):
        answer = answer_module._ensure_applicability_notice(
            "已核验结论。[^1]\n\n现场维修应结合适用机型、有效文档版本，并遵循正式维修程序复核。",
            "fault_diagnosis",
        )

        self.assertNotIn("现场维修应结合", answer)
        self.assertEqual(1, answer.count(answer_module._APPLICABILITY_NOTICE))
        self.assertTrue(answer.endswith(answer_module._APPLICABILITY_NOTICE))

    def test_source_location_text_cannot_verify_a_factual_binding(self):
        ledger = {
            1: {
                "title": "显示链路异常",
                "content": "显示链路出现间歇异常。",
                "scope": {},
                "metadata": {},
                "source_ref": "电源模块损坏导致反复重启.docx#1",
            }
        }

        self.assertFalse(
            answer_module._binding_is_verifiable(
                {
                    "citation_id": 1,
                    "quote": "电源模块损坏导致反复重启",
                },
                ledger,
            )
        )

    def test_short_quote_cannot_support_a_long_multi_claim_paragraph(self):
        ledger = {
            1: {
                "title": "91810003**0003",
                "content": "PSU-B B路间歇停机",
                "scope": {},
                "metadata": {},
            }
        }
        long_claim = (
            "外部电源输入滤波性能退化会增大纹波和干扰；功率器件热失稳、"
            "焊点裂纹或反馈网络异常会引起欠压保护和间歇停机，进而触发系统复位。"
        )

        self.assertFalse(
            answer_module._binding_is_verifiable(
                {
                    "citation_id": 1,
                    "quote": "PSU-B B路间歇停机",
                },
                ledger,
                long_claim,
            )
        )
        self.assertTrue(
            answer_module._binding_is_verifiable(
                {
                    "citation_id": 1,
                    "quote": "PSU-B B路间歇停机",
                },
                ledger,
                "HMC 记录为 PSU-B B路间歇停机。",
            )
        )

    def test_citation_repair_drops_unsupported_segments_from_mixed_fmeca_paragraph(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="fmeca_fact",
            doc_type="fmeca",
            doc_id="doc-fmeca",
            record_id="FM-SYN-002",
            title="输入滤波性能退化",
            content=(
                "局部影响：输入纹波和传导干扰增大。"
                "最终影响：多个子系统出现间歇性异常，故障具有相关性。"
            ),
        )
        state = {
            "intent": "impact_analysis",
            "query": "滤波元件退化后为什么会多处零星异常？",
            "evidence_registry": [evidence],
            "context_evidence_ids": [evidence["evidence_id"]],
            "safety_decision": {
                "approved_evidence_ids": [evidence["evidence_id"]],
                "blocked_evidence_ids": [],
                "context_evidence_ids": [evidence["evidence_id"]],
            },
        }
        draft = (
            "输入滤波性能退化会导致输入纹波和传导干扰增大；"
            "多个子系统可能出现具有相关性的间歇性异常；"
            "还会导致负载配电保护误跳闸、存储校验错误和复位信号卡滞。[^1]"
        )
        unsupported = ("保护误跳闸", "存储校验错误", "复位信号卡滞")

        def citation_editor(messages, _model):
            prompt = messages[0]["content"]
            raw_units = prompt.split("待校对事实单元：\n", 1)[1].split(
                "\n\n本轮已批准证据：", 1
            )[0]
            units = json.loads(raw_units)
            response_units = []
            for unit in units:
                text = str(unit.get("text") or "")
                if any(value in text for value in unsupported):
                    bindings = []
                elif "纹波和传导干扰" in text:
                    bindings = [
                        {"citation_id": 1, "quote": "输入纹波和传导干扰增大"}
                    ]
                elif "多个子系统" in text:
                    bindings = [
                        {
                            "citation_id": 1,
                            "quote": "多个子系统出现间歇性异常，故障具有相关性",
                        }
                    ]
                else:
                    bindings = []
                response_units.append(
                    {"unit_id": unit["unit_id"], "bindings": bindings}
                )
            return json.dumps({"units": response_units}, ensure_ascii=False)

        with patch.object(
            answer_module.llm_service,
            "_call_model",
            side_effect=citation_editor,
        ):
            repaired = _run(
                answer_module._repair_answer_citations(
                    draft,
                    state,
                    None,
                    citation_count=1,
                )
            )

        self.assertIn("输入纹波和传导干扰增大", repaired)
        self.assertIn("多个子系统可能出现具有相关性的间歇性异常", repaired)
        for value in unsupported:
            self.assertNotIn(value, repaired)
        self.assertIn("[^1]", repaired)

    def test_nonstandard_superscript_citation_marker_is_removed(self):
        answer = "多个子系统出现间歇性异常^[1]。[^1]"

        sanitized = answer_module._sanitize_citation_markers(answer, 1)

        self.assertEqual("多个子系统出现间歇性异常。[^1]", sanitized)

    def test_uncertainty_or_question_prefix_cannot_hide_a_second_assertion(self):
        state = {
            "evidence_registry": [],
            "safety_decision": {
                "approved_evidence_ids": [],
                "questions": ["请补充适用构型。"],
            },
        }
        answer = answer_module._downgrade_unapproved_assertions(
            "资料未记载，但通常是电源模块损坏。\n"
            "请补充适用构型；故障原因是电源损坏。",
            state,
        )

        self.assertEqual(answer_module._NO_SUPPORTED_FACTS_MESSAGE, answer)
        self.assertNotIn("电源损坏", answer)

    def test_answer_removes_invalid_markers_and_unique_diagnosis_language(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            record_id="HMC-001",
            title="HMC-001",
            content="显示链路异常",
        )
        question = "请补充适用构型。"
        state = {
            "intent": "symptom_diagnosis",
            "query": "显示闪烁",
            "messages": [],
            "evidence_registry": [evidence],
            "context_evidence_ids": [evidence["evidence_id"]],
            "candidate_faults": [{"fault": "显示链路异常", "score": 0.5}],
            "diagnostic_support": 0.5,
            "safety_decision": {
                "status": "clarify",
                "can_claim_unique": False,
                "questions": [question],
                "approved_evidence_ids": [evidence["evidence_id"]],
                "blocked_evidence_ids": [],
                "context_evidence_ids": [evidence["evidence_id"]],
            },
        }
        with (
            patch.object(
                answer_module.llm_service,
                "_call_model",
                return_value="已经确定为唯一故障。[^99]",
            ) as call,
            patch.object(
                answer_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(answer_module.answer_node(state))

        self.assertNotIn("已经确定", result["final_answer"])
        self.assertNotIn("电源损坏", result["final_answer"])
        self.assertNotIn("[^99]", result["final_answer"])
        self.assertIn("证据不足以支持唯一故障结论", result["final_answer"])
        self.assertIn(question, result["final_answer"])
        prompt = call.call_args_list[0].args[0][0]["content"]
        self.assertIn(question, prompt)
        self.assertEqual([], result["citations"])
        self.assertEqual(1, len(result["evidence_catalog"]))
        self.assertEqual("approved", result["evidence_catalog"][0]["status"])

    def test_final_citations_are_compact_subset_and_catalog_keeps_process_evidence(self):
        evidence = [
            make_evidence(
                source_kind="postgres",
                authority="authoritative_record",
                evidence_level="L1_DIRECT",
                claim_type="fmeca_fact",
                doc_type="fmeca",
                record_id=f"FM-{index}",
                title=f"证据 {index}",
                content=content,
            )
            for index, content in enumerate(
                ("输入纹波增大。", "温度升高。", "滤波能力下降。"),
                start=1,
            )
        ]
        state = {
            "intent": "impact_analysis",
            "query": "老化有什么影响？",
            "messages": [],
            "evidence_registry": evidence,
            "context_evidence_ids": [item["evidence_id"] for item in evidence],
            "tool_results": [
                {
                    "tool": "fmeca_query",
                    "evidence_ids": [item["evidence_id"] for item in evidence],
                }
            ],
            "candidate_faults": [],
            "diagnostic_support": 0.8,
            "safety_decision": {
                "status": "answer",
                "approved_evidence_ids": [item["evidence_id"] for item in evidence],
                "blocked_evidence_ids": [],
                "context_evidence_ids": [item["evidence_id"] for item in evidence],
                "questions": [],
            },
        }
        draft = "滤波能力会下降。\n\n输入纹波会增大。"
        repair = json.dumps(
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "bindings": [{"citation_id": 3, "quote": "滤波能力下降"}],
                    },
                    {
                        "unit_id": "u2",
                        "bindings": [{"citation_id": 1, "quote": "输入纹波增大"}],
                    },
                ]
            },
            ensure_ascii=False,
        )
        with (
            patch.object(answer_module.llm_service, "_call_model", side_effect=[draft, repair]),
            patch.object(
                answer_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(answer_module.answer_node(state))

        self.assertIn("滤波能力会下降。[^1]", result["final_answer"])
        self.assertIn("输入纹波会增大。[^2]", result["final_answer"])
        self.assertEqual([1, 2], [item["id"] for item in result["citations"]])
        self.assertEqual(
            [evidence[2]["evidence_id"], evidence[0]["evidence_id"]],
            [item["evidence_id"] for item in result["citations"]],
        )
        self.assertEqual(3, len(result["evidence_catalog"]))
        self.assertEqual(
            ["answer_used", "answer_used", "approved"],
            [item["status"] for item in result["evidence_catalog"]],
        )
        self.assertEqual([1, 2, None], [
            item["answer_citation_id"] for item in result["evidence_catalog"]
        ])

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
            answer_module.EVIDENCE_METADATA_MAX_ITEMS,
        )
        self.assertLessEqual(
            len(payload["derived_from"]),
            answer_module.EVIDENCE_METADATA_MAX_LIST_ITEMS,
        )
        self.assertTrue(all(len(value) <= 240 for value in payload["derived_from"]))
        self.assertLessEqual(
            len(payload["limitations"]),
            answer_module.EVIDENCE_METADATA_MAX_LIST_ITEMS,
        )
        self.assertTrue(all(len(value) <= 600 for value in payload["limitations"]))

    def test_answer_strips_blocked_numeric_and_uses_single_evidence_boundary(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            record_id="HMC-001",
            title="HMC-001",
            content="显示链路异常",
        )
        state = {
            "intent": "symptom_diagnosis",
            "query": "为什么显示异常，可靠度是多少？",
            "messages": [],
            "evidence_registry": [evidence],
            "context_evidence_ids": [evidence["evidence_id"]],
            "candidate_faults": [{"fault": "显示链路异常", "score": 0.5}],
            "diagnostic_support": 0.5,
            "safety_decision": {
                "status": "clarify",
                "can_claim_unique": False,
                "blocked_claims": ["formal_numeric_result"],
                "questions": ["请补充可靠度参数。"],
                "approved_evidence_ids": [evidence["evidence_id"]],
                "blocked_evidence_ids": [],
                "context_evidence_ids": [evidence["evidence_id"]],
            },
        }
        unsafe = (
            "HMC 手册表明显示链路已经失效。\n"
            "可以确认故障原因为电源损坏。\n"
            "系统可靠度为 0.97。\n"
            "顶事件概率为百分之九十七。"
        )
        with (
            patch.object(answer_module.llm_service, "_call_model", return_value=unsafe),
            patch.object(
                answer_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(answer_module.answer_node(state))

        answer = result["final_answer"]
        self.assertNotIn("0.97", answer)
        self.assertNotIn("百分之九十七", answer)
        self.assertIn("数值边界", answer)
        self.assertNotIn("电源损坏", answer)
        self.assertNotIn("待复核（未绑定已批准直接证据）", answer)
        self.assertNotIn("部分说明未附可直接核验的来源", answer)
        self.assertEqual(1, answer.count("证据边界："))

    def test_uncited_assertions_are_removed_without_global_boundary(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            record_id="HMC-001",
            title="HMC-001",
            content="显示链路异常",
        )
        state = {
            "evidence_registry": [evidence],
            "context_evidence_ids": [evidence["evidence_id"]],
            "safety_decision": {
                "approved_evidence_ids": [],
                "context_evidence_ids": [evidence["evidence_id"]],
            },
        }

        answer = answer_module._downgrade_unapproved_assertions(
            "HMC 手册表明显示链路失效。\nFMECA 记录指出该故障会影响显示输出。",
            state,
        )

        self.assertEqual(answer_module._NO_SUPPORTED_FACTS_MESSAGE, answer)
        self.assertNotIn("证据边界：", answer)
        self.assertNotIn("待复核（未绑定已批准直接证据）", answer)
        self.assertNotIn("HMC 手册表明显示链路失效。", answer)
        self.assertNotIn("FMECA 记录指出该故障会影响显示输出。", answer)

    def test_citation_removal_drops_orphan_section_and_hides_number_gaps(self):
        answer = answer_module._cleanup_numbered_bold_sections(
            "开场结论。[^1]\n\n"
            "**2. 复位控制逻辑故障**[^1]\n\n"
            "**3. 散热与环境因素**\n"
            "* 风扇故障会导致过热。[^2]"
        )

        self.assertNotIn("复位控制逻辑故障", answer)
        self.assertNotIn("**3.", answer)
        self.assertIn("**散热与环境因素**", answer)
        self.assertIn("风扇故障会导致过热。[^2]", answer)

    def test_existing_evidence_boundary_is_not_duplicated(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            record_id="HMC-001",
            title="HMC-001",
            content="显示链路异常",
        )
        state = {
            "evidence_registry": [evidence],
            "safety_decision": {"approved_evidence_ids": []},
        }

        answer = answer_module._downgrade_unapproved_assertions(
            f"{answer_module._UNIQUE_DIAGNOSIS_BOUNDARY}\n"
            "证据边界：这是模型重复生成的边界。\n\n"
            "待复核（未绑定已批准直接证据）：HMC 手册表明显示链路失效。",
            state,
        )

        self.assertEqual(1, answer.count("证据边界："))
        self.assertNotIn("这是模型重复生成的边界", answer)
        self.assertNotIn("待复核（未绑定已批准直接证据）", answer)
        self.assertNotIn("HMC 手册表明显示链路失效。", answer)

    def test_cleanup_removes_orphan_heading_and_incomplete_table(self):
        answer = answer_module._cleanup_generated_markdown_structure(
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
        answer = answer_module._cleanup_generated_markdown_structure(
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

    def test_public_output_localizes_internal_fields_and_english_labels(self):
        answer = answer_module._strip_internal_editor_artifacts(
            "### Cause (FTA 路径)\n"
            "- **Cause**：滤波异常（fault_cause）。[^1]\n"
            "- **Impact (局部)**：输出抖动（local_impact / final_impact）。[^1]\n"
            "- **原因 (cause)**：元件老化。[^1]\n"
            "- **影响（impact）**：业务数据延迟。[^1]\n"
            "- compensation_action=切换冗余通道；maintenance_action：检查连接器。[^2]\n"
            "- detection_method=读取状态字；verification_result=连续运行正常。[^2]\n"
            "- 普通代码与 API 保留：CauseMonitor API、fault_cause_v2。"
        )

        self.assertIn("### 原因（FTA 路径）", answer)
        self.assertIn("**原因**：滤波异常。", answer)
        self.assertIn("**影响（局部）**：输出抖动。", answer)
        self.assertIn("**原因**：元件老化。", answer)
        self.assertIn("**影响**：业务数据延迟。", answer)
        self.assertIn("补偿措施：切换冗余通道", answer)
        self.assertIn("维修方法：检查连接器", answer)
        self.assertIn("检测方法：读取状态字", answer)
        self.assertIn("验证结果：连续运行正常", answer)
        self.assertIn("CauseMonitor API", answer)
        self.assertIn("fault_cause_v2", answer)
        self.assertNotRegex(
            answer,
            r"(?i)(?:fault_cause|local_impact|final_impact|compensation_action|"
            r"maintenance_action|detection_method|verification_result)\s*[:：=)]",
        )
        self.assertNotIn("**Cause", answer)
        self.assertNotIn("**Impact", answer)

    def test_exact_hmc_direct_hit_does_not_get_false_insufficient_or_unavailable_banner(self):
        evidence = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            record_id="31810001**0001",
            title="31810001**0001",
            content="显示链路异常",
            match_type="exact_code",
        )
        state = {
            "intent": "exact_code",
            "query": "LCN=31810001 HAMC=** HSC=0001",
            "messages": [],
            "evidence_registry": [evidence],
            "context_evidence_ids": [evidence["evidence_id"]],
            "safety_decision": {
                "status": "answer",
                "can_claim_unique": False,
                "questions": [],
                "approved_evidence_ids": [evidence["evidence_id"]],
                "blocked_evidence_ids": [],
                "blocked_claims": ["unique_diagnosis"],
                "context_evidence_ids": [evidence["evidence_id"]],
            },
        }
        model_answer = "### HMC 查询结果\nPostgreSQL HMC 精确代码命中 31810001**0001。[^1]"
        repair = json.dumps(
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "bindings": [{"citation_id": 1, "quote": "31810001**0001"}],
                    }
                ]
            },
            ensure_ascii=False,
        )
        with (
            patch.object(
                answer_module.llm_service,
                "_call_model",
                side_effect=[model_answer, repair],
            ) as call,
            patch.object(
                answer_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(answer_module.answer_node(state))

        answer = result["final_answer"]
        self.assertNotIn("权威事实源当前不可用", answer)
        self.assertNotIn("证据不足以支持唯一故障结论", answer)
        self.assertIn("精确代码命中", answer)
        prompt = call.call_args_list[0].args[0][0]["content"]
        self.assertIn("正文通常控制在 700 个中文字以内", prompt)

    def test_component_hmc_repair_invokes_llm_and_preserves_cited_facts(self):
        metadata = {
            "hmc_code": "91810001**0001",
            "lcn": "91810001",
            "hamc": "**",
            "hsc": "0001",
            "data_source": "程序化合成并经过跨文档一致性校验",
            "data_status": "合成测试数据 / 非运行资料",
        }
        scope = {
            "system": "SYN-ICP-9000 综合信息处理与显示系统",
            "configuration": "CONFIG-A",
            "document_number": "SYN-ICP-DOC-002",
            "document_version": "V1.0",
        }
        fault = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="hmc_fault",
            doc_type="hmc",
            doc_id="doc-hmc",
            doc_name="02_HMC故障代码手册.docx",
            record_id="91810001**0001",
            source_ref="02_HMC故障代码手册.docx#table=2;row=3",
            title="91810001**0001",
            content="ICU 输入过压钳位失效",
            match_type="exact_components",
            scope=scope,
            metadata=metadata,
        )
        repair = make_evidence(
            source_kind="postgres",
            authority="authoritative_record",
            evidence_level="L1_DIRECT",
            claim_type="maintenance_action",
            doc_type="hmc",
            doc_id="doc-hmc",
            doc_name="02_HMC故障代码手册.docx",
            record_id="91810001**0001",
            source_ref="02_HMC故障代码手册.docx#table=2;row=3",
            title="91810001**0001 维修方法",
            content="检查输入采样与钳位器件，确认异常后更换输入调理模块。",
            match_type="exact_components",
            scope=scope,
            metadata=metadata,
        )
        state = {
            "intent": "exact_code",
            "query": "LCN 91810001 HSC 0001 故障如何修复",
            "messages": [],
            "evidence_registry": [fault, repair],
            "context_evidence_ids": [fault["evidence_id"], repair["evidence_id"]],
            "safety_decision": {
                "status": "answer",
                "can_claim_unique": False,
                "questions": [],
                "approved_evidence_ids": [fault["evidence_id"], repair["evidence_id"]],
                "blocked_evidence_ids": [],
                "blocked_claims": ["unique_diagnosis"],
                "context_evidence_ids": [fault["evidence_id"], repair["evidence_id"]],
            },
        }
        model_answer = (
            "### 诊断结论\n"
            "文档记录该字段组合对应 **ICU 输入过压钳位失效**。[^1]\n\n"
            "### 处理建议\n"
            "文档建议检查输入采样与钳位器件，确认异常后更换输入调理模块。[^2]\n\n"
            "- HMC：`91810001**0001`（LCN `91810001` / HAMC `**` / HSC `0001`）。[^1]\n"
            "- 数据状态：合成测试数据 / 非运行资料。[^1]"
        )
        citation_repair = json.dumps(
            {
                "units": [
                    {
                        "unit_id": "u1",
                        "bindings": [{"citation_id": 1, "quote": "ICU 输入过压钳位失效"}],
                    },
                    {
                        "unit_id": "u2",
                        "bindings": [
                            {
                                "citation_id": 2,
                                "quote": "检查输入采样与钳位器件，确认异常后更换输入调理模块",
                            }
                        ],
                    },
                    {
                        "unit_id": "u3",
                        "bindings": [{"citation_id": 1, "quote": "91810001**0001"}],
                    },
                    {
                        "unit_id": "u4",
                        "bindings": [{"citation_id": 1, "quote": "合成测试数据 / 非运行资料"}],
                    },
                ]
            },
            ensure_ascii=False,
        )
        with (
            patch.object(
                answer_module.llm_service,
                "_call_model",
                side_effect=[model_answer, citation_repair],
            ) as call,
            patch.object(
                answer_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(answer_module.answer_node(state))

        answer = result["final_answer"]
        self.assertEqual(2, call.call_count)
        prompt = call.call_args_list[0].args[0][0]["content"]
        self.assertIn("91810001**0001", prompt)
        self.assertIn("ICU 输入过压钳位失效", prompt)
        self.assertIn("检查输入采样与钳位器件", prompt)
        self.assertIn("合成测试数据 / 非运行资料", prompt)
        self.assertIn("91810001**0001", answer)
        self.assertIn("ICU 输入过压钳位失效", answer)
        self.assertIn("更换输入调理模块", answer)
        self.assertIn("[^1]", answer)
        self.assertIn("[^2]", answer)
        self.assertIn("合成测试数据 / 非运行资料", answer)
        self.assertNotIn("待复核（未绑定已批准直接证据）", answer)
        self.assertNotIn("下一步需要确认", answer)
        self.assertEqual(["hmc_fault", "maintenance_action"], [item["type"] for item in result["citations"]])
        self.assertEqual("合成测试数据 / 非运行资料", result["citations"][0]["document_metadata"]["data_status"])

    def test_no_gate_question_removes_model_authored_empty_next_step_section(self):
        answer = answer_module._canonicalize_clarification_section(
            "### HMC 查询结果\n已找到记录。\n\n### 下一步需要确认\n无\n\n### 适用性说明\n仅用于检索。",
            {"safety_decision": {"questions": []}},
        )

        self.assertNotIn("下一步需要确认", answer)
        self.assertNotIn("\n无\n", f"\n{answer}\n")
        self.assertIn("### 适用性说明", answer)

    def test_answer_normalizes_escaped_markdown_and_replaces_duplicate_next_steps(self):
        first_question = "请补充缺失的 HAMC。"
        second_question = "请确认适用构型和有效文档版本。"
        state = {
            "intent": "exact_code",
            "query": "LCN 91810001 HSC 0001 故障如何修复",
            "messages": [],
            "evidence_registry": [],
            "candidate_faults": [],
            "diagnostic_support": 0.2,
            "safety_decision": {
                "status": "clarify",
                "can_claim_unique": False,
                "questions": [first_question, second_question],
                "approved_evidence_ids": [],
                "blocked_evidence_ids": [],
                "blocked_claims": ["incomplete_hmc_code"],
                "context_evidence_ids": [],
            },
        }
        escaped = (
            "\\### 初步判断\\\n"
            "当前仅能核对字段。\\\n\n"
            "\\**下一步需要确认\\**\n"
            "1\\. 请提供更多代码。\n"
            "2\\. 请提供更多背景。\n\n"
            "\\**适用性说明\\**\n"
            "仅用于知识检索。\n\n"
            "\\### 下一步需要确认\n"
            "\\- 这是重复章节。"
        )
        with (
            patch.object(answer_module.llm_service, "_call_model", return_value=escaped),
            patch.object(
                answer_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = _run(answer_module.answer_node(state))

        answer = result["final_answer"]
        self.assertNotIn("\\", answer)
        self.assertEqual(1, answer.count("下一步需要确认"))
        self.assertEqual(1, answer.count(first_question))
        self.assertEqual(1, answer.count(second_question))
        self.assertNotIn("这是重复章节", answer)
        self.assertNotIn("请提供更多代码", answer)
        self.assertIn(answer_module._APPLICABILITY_NOTICE, answer)
        self.assertNotIn("仅用于知识检索", answer)

    def test_blocked_status_without_failure_marker_does_not_claim_source_unavailable(self):
        state = {
            "intent": "exact_code",
            "safety_decision": {
                "status": "blocked",
                "can_claim_unique": False,
                "questions": [],
                "blocked_claims": ["exact_hmc_fact"],
            },
        }
        answer = answer_module._enforce_answer_safety(
            "权威事实源当前不可用，本回答不能作为正式诊断结论。\n"
            "HMC 精确查询已完成但未找到匹配记录。",
            state,
            architecture_query=False,
        )
        self.assertNotIn("权威事实源当前不可用", answer)
        self.assertNotIn("PostgreSQL 权威事实查询失败", answer)


if __name__ == "__main__":
    unittest.main()
