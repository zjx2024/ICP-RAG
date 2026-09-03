import unittest
from decimal import Decimal

from backend.icp.structured_graph import (
    StructuredGraphError,
    business_node_id,
    build_fault_history_graph,
    build_fmeca_graph,
    build_formal_diagram_graph,
    build_function_locator_graph,
    build_hmc_graph,
    build_mfl_graph,
    build_structured_graph,
    document_text_node_id,
)


def _nodes(graph, label=None):
    values = graph["vertices"]
    return [item for item in values if label is None or item["label"] == label]


def _edges(graph, label=None):
    values = graph["edges"]
    return [item for item in values if label is None or item["label"] == label]


class StructuredGraphIdentityTests(unittest.TestCase):
    def test_business_ids_are_cross_document_but_case_and_leading_zero_sensitive(self):
        self.assertEqual(
            business_node_id("hmc_code", "91810001**0001"),
            business_node_id("hmc_code", "91810001**0001"),
        )
        self.assertNotEqual(
            business_node_id("hmc_code", "91810001**0001"),
            business_node_id("hmc_code", "91810001**1"),
        )
        self.assertNotEqual(
            business_node_id("api", "GetStatus"),
            business_node_id("api", "getstatus"),
        )

    def test_free_text_is_scoped_by_document_and_field_kind(self):
        left = document_text_node_id("doc-a", "hmc_fault_desc", "输入过压")
        same = document_text_node_id("doc-a", "hmc_fault_desc", "输入过压")
        other_doc = document_text_node_id("doc-b", "hmc_fault_desc", "输入过压")
        other_field = document_text_node_id("doc-a", "mfl_module_fault", "输入过压")

        self.assertEqual(left, same)
        self.assertNotEqual(left, other_doc)
        self.assertNotEqual(left, other_field)

    def test_row_document_mismatch_fails_closed(self):
        with self.assertRaisesRegex(StructuredGraphError, "belongs to doc_id"):
            build_hmc_graph(
                "doc-a",
                [
                    {
                        "doc_id": "doc-b",
                        "variant_id": "variant-1",
                        "hmc_code": "CODE-1",
                        "fault_desc": "故障",
                        "repair_method": "维修",
                    }
                ],
            )


class FunctionLocatorStructuredGraphTests(unittest.TestCase):
    def test_only_explicit_ids_create_hierarchy_and_interface_edges(self):
        graph = build_function_locator_graph(
            "doc-func",
            components=[
                {
                    "doc_id": "doc-func",
                    "component_id": "SYS-1",
                    "component_type": "system",
                    "name": "综合系统",
                },
                {
                    "doc_id": "doc-func",
                    "component_id": "MOD-1",
                    "parent_component_id": "SYS-1",
                    "component_type": "module",
                    "name": "显示模块",
                },
            ],
            functions=[
                {
                    "doc_id": "doc-func",
                    "function_id": "FUNC-1",
                    "module_id": "MOD-1",
                    "function_name": "生成画面",
                    "inputs": ["航电数据"],
                    "outputs": ["视频帧"],
                },
                {
                    "doc_id": "doc-func",
                    "function_id": "FUNC-NAME-ONLY",
                    "module_name": "显示模块",
                    "function_name": "名称不能用于连边",
                },
            ],
            interfaces=[
                {
                    "doc_id": "doc-func",
                    "interface_id": "IF-1",
                    "source_component_id": "MOD-1",
                    "target_component_id": "SYS-1",
                    "relation_type": "data_flow",
                    "signal_name": "显示状态",
                }
            ],
        )

        self.assertEqual(2, len(_nodes(graph, "Component")))
        self.assertEqual(2, len(_nodes(graph, "Function")))
        self.assertEqual(1, len(_nodes(graph, "Interface")))
        self.assertEqual(1, len(_edges(graph, "contains_component")))
        self.assertEqual(1, len(_edges(graph, "provides_function")))
        self.assertEqual(1, len(_edges(graph, "interface_source")))
        self.assertEqual(1, len(_edges(graph, "interface_target")))
        self.assertEqual(1, len(_edges(graph, "has_input")))
        self.assertEqual(1, len(_edges(graph, "has_output")))

        name_only_id = business_node_id("function", "FUNC-NAME-ONLY")
        self.assertFalse(
            any(
                edge["target"] == name_only_id
                and edge["label"] == "provides_function"
                for edge in graph["edges"]
            )
        )

    def test_business_id_duplicates_are_order_independent(self):
        components = [
            {
                "component_id": "MOD-1",
                "component_type": "module",
                "name": "显示模块",
                "source_ref": "source-a",
            },
            {
                "component_id": "MOD-1",
                "component_type": "module",
                "name": "显示处理模块",
                "source_ref": "source-b",
            },
        ]
        left = build_function_locator_graph("doc-func", components=components)
        right = build_function_locator_graph("doc-func", components=reversed(components))

        self.assertEqual(left, right)
        self.assertEqual(1, len(_nodes(left, "Component")))
        node = _nodes(left, "Component")[0]
        self.assertEqual(
            ["显示处理模块", "显示模块"],
            node["properties"]["name_alternatives"],
        )
        self.assertTrue(
            any(item["code"] == "explicit_id_property_conflict" for item in left["warnings"])
        )


class TableStructuredGraphTests(unittest.TestCase):
    def test_hmc_exact_code_dedupes_but_preserves_source_variants(self):
        rows = [
            {
                "doc_id": "doc-hmc",
                "variant_id": "variant-a",
                "hmc_code": "91810001**0001",
                "lcn": "91810001",
                "hamc": "**",
                "hsc": "0001",
                "fault_desc": "ICU 输入过压钳位失效",
                "repair_method": "更换输入调理模块",
                "source_ref": "table=1;row=2",
            },
            {
                "doc_id": "doc-hmc",
                "variant_id": "variant-b",
                "hmc_code": "91810001**0001",
                "lcn": "91810001",
                "hamc": "**",
                "hsc": "0001",
                "fault_desc": "ICU 输入过压钳位失效",
                "repair_method": "检查钳位器件后更换模块",
                "source_ref": "table=2;row=3",
            },
        ]

        left = build_hmc_graph("doc-hmc", rows)
        right = build_hmc_graph("doc-hmc", reversed(rows))

        self.assertEqual(left, right)
        self.assertEqual(1, len(_nodes(left, "HMCCode")))
        code = _nodes(left, "HMCCode")[0]
        self.assertEqual(["variant-a", "variant-b"], code["properties"]["variant_ids"])
        fault_edges = _edges(left, "describes_fault")
        self.assertEqual(1, len(fault_edges))
        self.assertEqual(
            2, len(fault_edges[0]["properties"]["occurrence_qualifiers"])
        )
        self.assertEqual(2, len(_edges(left, "has_repair_method")))
        self.assertEqual(3, len(_nodes(left, "TextValue")))

    def test_same_hmc_business_id_but_same_text_in_two_documents_has_distinct_text_nodes(self):
        row = {
            "variant_id": "variant-a",
            "hmc_code": "HMC-001",
            "fault_desc": "通信中断",
            "repair_method": "检查接口",
        }
        left = build_hmc_graph("doc-a", [row])
        right = build_hmc_graph("doc-b", [row])

        self.assertEqual(
            _nodes(left, "HMCCode")[0]["id"],
            _nodes(right, "HMCCode")[0]["id"],
        )
        self.assertTrue(
            set(item["id"] for item in _nodes(left, "TextValue")).isdisjoint(
                item["id"] for item in _nodes(right, "TextValue")
            )
        )

    def test_fmeca_projects_only_declared_columns_and_preserves_decimal(self):
        graph = build_fmeca_graph(
            "doc-fmeca",
            [
                {
                    "doc_id": "doc-fmeca",
                    "code": "FM-001",
                    "fault_mode": "画面丢失",
                    "fault_cause": "输出链路中断",
                    "local_impact": "单屏无显示",
                    "final_impact": "任务信息不可见",
                    "severity": "II",
                    "probability": Decimal("0.000010000000000000"),
                    "detect_method": "BIT 告警",
                }
            ],
        )

        record = _nodes(graph, "FMECARecord")[0]
        self.assertEqual("0.000010000000000000", record["properties"]["probability"])
        self.assertEqual(1, len(_edges(graph, "has_fault_mode")))
        self.assertEqual(1, len(_edges(graph, "has_fault_cause")))
        self.assertEqual(1, len(_edges(graph, "has_local_impact")))
        self.assertEqual(1, len(_edges(graph, "has_final_impact")))
        self.assertEqual(1, len(_edges(graph, "has_detection_method")))

    def test_mfl_api_name_is_an_explicit_business_identity(self):
        rows = [
            {
                "mfl_code": "MFL-001",
                "module_fault": "显示模块异常",
                "api_name": "GetDisplayStatus",
                "api_method": "GET /display/status",
            },
            {
                "mfl_code": "MFL-002",
                "module_fault": "显示输出异常",
                "api_name": "GetDisplayStatus",
                "api_method": "GET /display/status",
            },
        ]
        graph = build_mfl_graph("doc-mfl", rows)

        self.assertEqual(2, len(_nodes(graph, "MFLRecord")))
        self.assertEqual(1, len(_nodes(graph, "API")))
        self.assertEqual(2, len(_edges(graph, "reports_via_api")))
        method_edges = _edges(graph, "has_api_method")
        self.assertEqual(1, len(method_edges))
        self.assertEqual(
            2, len(method_edges[0]["properties"]["occurrence_qualifiers"])
        )

    def test_identical_text_in_fmeca_and_mfl_is_not_semantically_merged(self):
        fmeca = build_fmeca_graph(
            "doc-shared",
            [{"code": "FM-1", "fault_mode": "显示异常"}],
        )
        mfl = build_mfl_graph(
            "doc-shared",
            [{"mfl_code": "MFL-1", "module_fault": "显示异常"}],
        )
        fmeca_text = _nodes(fmeca, "TextValue")[0]["id"]
        mfl_text = _nodes(mfl, "TextValue")[0]["id"]
        self.assertNotEqual(fmeca_text, mfl_text)

    def test_fmeca_product_flag_bridges_only_explicit_function_ids(self):
        graph = build_fmeca_graph(
            "doc-fmeca",
            [
                {
                    "code": "FM-001",
                    "product_flag": "FUNC-101（显示管理）；通用处理功能",
                    "fault_mode": "显示异常",
                }
            ],
        )

        function = next(
            item
            for item in _nodes(graph, "Function")
            if item["properties"].get("business_id") == "FUNC-101"
        )
        self.assertEqual(business_node_id("function", "FUNC-101"), function["id"])
        bridges = [
            item
            for item in _edges(graph, "has_product_flag")
            if item["target"] == function["id"]
        ]
        self.assertEqual(1, len(bridges))
        self.assertEqual("explicit_function_id", bridges[0]["properties"]["reference_kind"])


class FaultHistoryStructuredGraphTests(unittest.TestCase):
    def test_explicit_code_references_share_business_ids_with_source_types(self):
        history = build_fault_history_graph(
            "doc-history",
            [
                {
                    "case_id": "CASE-001",
                    "fault_symptom": "画面闪烁",
                    "related_module": "显示模块",
                    "hmc_code": "HMC-001",
                    "fmeca_code": "FM-001",
                    "fta_ref": "FTA-001",
                    "rbd_ref": "RBD-001",
                    "mfl_code": "MFL-001",
                    "steps": [
                        {"step_no": 1, "content": "检查连接器"},
                        {"step_no": 2, "content": "更换显示模块"},
                    ],
                }
            ],
        )
        hmc = build_hmc_graph(
            "doc-hmc",
            [
                {
                    "variant_id": "v1",
                    "hmc_code": "HMC-001",
                    "fault_desc": "画面闪烁",
                    "repair_method": "更换显示模块",
                }
            ],
        )
        fmeca = build_fmeca_graph(
            "doc-fmeca", [{"code": "FM-001", "fault_mode": "画面闪烁"}]
        )
        mfl = build_mfl_graph(
            "doc-mfl", [{"mfl_code": "MFL-001", "module_fault": "显示异常"}]
        )

        by_label = {item["label"]: item for item in history["vertices"]}
        self.assertEqual(_nodes(hmc, "HMCCode")[0]["id"], by_label["HMCCode"]["id"])
        self.assertEqual(
            _nodes(fmeca, "FMECARecord")[0]["id"],
            by_label["FMECARecord"]["id"],
        )
        self.assertEqual(
            _nodes(mfl, "MFLRecord")[0]["id"],
            by_label["MFLRecord"]["id"],
        )
        self.assertEqual(1, len(_edges(history, "references_hmc")))
        self.assertEqual(1, len(_edges(history, "references_fmeca")))
        self.assertEqual(1, len(_edges(history, "references_fta")))
        self.assertEqual(1, len(_edges(history, "references_rbd")))
        self.assertEqual(1, len(_edges(history, "references_mfl")))
        self.assertEqual(2, len(_edges(history, "has_diagnosis_step")))

        # related_module is an untyped name mention, not an inferred Component.
        self.assertFalse(_nodes(history, "Component"))
        mention_edge = _edges(history, "mentions_module")[0]
        self.assertTrue(mention_edge["target"].startswith("sg:t:"))

    def test_repeated_step_text_dedupes_only_the_text_node_not_ordered_edges(self):
        graph = build_fault_history_graph(
            "doc-history",
            [
                {
                    "case_id": "CASE-001",
                    "fault_symptom": "异常",
                    "steps": ["重新上电", "重新上电"],
                }
            ],
        )

        step_nodes = [
            item
            for item in _nodes(graph, "TextValue")
            if item["properties"]["value_kind"] == "fault_history_step"
        ]
        self.assertEqual(1, len(step_nodes))
        self.assertEqual(2, len(_nodes(graph, "FaultHistoryStep")))
        self.assertEqual(2, len(_edges(graph, "has_diagnosis_step")))
        self.assertEqual(2, len(_edges(graph, "has_step_content")))
        self.assertEqual(1, len(_edges(graph, "next_step")))

    def test_all_document_links_and_typed_related_ids_are_projected_once(self):
        graph = build_fault_history_graph(
            "doc-history",
            [
                {
                    "case_id": "CASE-001",
                    "hmc_code": "HMC-001",
                    "fmeca_code": "FM-001",
                    "fta_ref": "FTA-001",
                    "rbd_ref": "RBD-001",
                    "mfl_code": "MFL-001",
                    "related_module": "显示模块 MOD-101；名称只有显示模块",
                    "related_function": "关联 FUNC-101，但显示管理只是名称",
                    "document_links": {
                        "hmc_codes": ["HMC-001", "HMC-002"],
                        "fmeca_refs": ["FM-001", "FM-002"],
                        "fta_refs": ["FTA-001", "EVT-001-BE-01"],
                        "rbd_refs": ["RBD-001", "RBD-002"],
                        "mfl_refs": ["MFL-001", "MFL-002"],
                        "function_refs": ["FUNC-101", "FUNC-102", "显示管理"],
                    },
                }
            ],
        )

        self.assertEqual(2, len(_edges(graph, "references_hmc")))
        self.assertEqual(2, len(_edges(graph, "references_fmeca")))
        self.assertEqual(2, len(_edges(graph, "references_fta")))
        self.assertEqual(2, len(_edges(graph, "references_rbd")))
        self.assertEqual(2, len(_edges(graph, "references_mfl")))

        case_id = business_node_id("fault_history_case", "CASE-001")
        exact_function_targets = {
            edge["target"]
            for edge in _edges(graph, "mentions_function")
            if edge["source"] == case_id
            and edge["properties"].get("reference_kind") == "explicit_function_id"
        }
        self.assertEqual(
            {
                business_node_id("function", "FUNC-101"),
                business_node_id("function", "FUNC-102"),
            },
            exact_function_targets,
        )
        exact_module = next(
            edge
            for edge in _edges(graph, "mentions_module")
            if edge["properties"].get("reference_kind") == "explicit_module_id"
        )
        self.assertEqual(business_node_id("component", "MOD-101"), exact_module["target"])
        self.assertIn(
            business_node_id("fta_event", "EVT-001-BE-01"),
            {item["id"] for item in _nodes(graph, "FTAEvent")},
        )


class FormalDiagramStructuredGraphTests(unittest.TestCase):
    @staticmethod
    def _revision(doc_type, graph, **overrides):
        return {
            "doc_id": f"doc-{doc_type}",
            "diagram_type": doc_type,
            "diagram_id": f"diagram-{doc_type}",
            "diagram_key": "FTA-SYN-001" if doc_type == "fta" else "RBD:自动标题",
            "title": "正式工程图",
            "revision_id": f"revision-{doc_type}",
            "revision_no": 2,
            "canonical_hash": "a" * 64,
            "review_status": "pending",
            "validation_status": "auto_validated",
            "validation_tier": "deterministic_structure",
            "projection_status": "published",
            "is_active": True,
            "canonical_graph": graph,
            **overrides,
        }

    def test_fta_only_projects_formal_active_revision_and_bridges_ids(self):
        formal = self._revision(
            "fta",
            {
                "root_fault_id": "top",
                "metadata": {
                    "business_key": "FTA-SYN-001",
                    "linked_function": "FUNC-201 综合处理功能",
                    "source_refs": ["table=2;row=1"],
                },
                "vertices": [
                    {
                        "id": "top",
                        "label": "TopEvent",
                        "properties": {
                            "name": "系统无法启动",
                            "business_id": "FTA-SYN-001",
                            "event_id": "FTA-SYN-001",
                        },
                    },
                    {
                        "id": "basic",
                        "label": "BasicEvent",
                        "properties": {
                            "name": "电源异常",
                            "event_id": "EVT-SYN-001-BE-01",
                            "business_id": "EVT-SYN-001-BE-01",
                            "failure_mode_id": "FM-SYN-001",
                            "source_ref": "table=3;row=2",
                        },
                    },
                ],
                "edges": [
                    {
                        "id": "cause-1",
                        "label": "causes",
                        "source": "top",
                        "target": "basic",
                        "properties": {
                            "gate_type": {"value": "OR"},
                            "source_ref": "table=4;row=2",
                        },
                    }
                ],
            },
        )
        provisional = self._revision(
            "fta",
            {
                "root_fault_id": "excluded",
                "vertices": [
                    {"id": "excluded", "label": "TopEvent", "properties": {"name": "不得进入"}}
                ],
                "edges": [],
            },
            revision_id="revision-provisional",
            validation_status="provisional",
        )

        graph = build_formal_diagram_graph("fta", "doc-fta", [formal, provisional])

        occurrences = [
            item
            for item in _nodes(graph, "FTAEvent")
            if item["properties"].get("identity_scope") == "document_occurrence"
        ]
        self.assertEqual(2, len(occurrences))
        self.assertTrue(all("business_id" not in item["properties"] for item in occurrences))
        self.assertEqual(1, len(_nodes(graph, "FTAGate")))
        self.assertEqual("OR", _nodes(graph, "FTAGate")[0]["properties"]["gate_type"])
        self.assertEqual(1, len(_edges(graph, "decomposed_by")))
        self.assertEqual(1, len(_edges(graph, "has_input_event")))
        self.assertEqual(1, len(_edges(graph, "represents_fault_mode")))
        self.assertEqual(1, len(_edges(graph, "implements_function")))
        self.assertIn(
            business_node_id("fmeca_record", "FM-SYN-001"),
            {item["id"] for item in _nodes(graph, "FMECARecord")},
        )
        self.assertFalse(
            any(
                item["properties"].get("revision_id") == "revision-provisional"
                for item in graph["vertices"]
            )
        )
        self.assertTrue(
            any(
                item["record_id"] == "revision-provisional"
                and item["code"] == "diagram_revision_excluded"
                for item in graph["warnings"]
            )
        )
        topology_edge = _edges(graph, "decomposed_by")[0]
        self.assertEqual("revision-fta", topology_edge["properties"]["revision_id"])
        self.assertEqual("auto_validated", topology_edge["properties"]["validation_status"])

    def test_rbd_keeps_occurrences_and_does_not_guess_business_id_from_title(self):
        revision = self._revision(
            "rbd",
            {
                "metadata": {
                    "business_key": "RBD:自动标题",
                    "title": "RBD-SYN-999 看起来像编号但只是标题",
                    "source_refs": ["paragraph=8"],
                },
                "vertices": [
                    {"id": "start", "label": "Start", "properties": {"name": "START"}},
                    {
                        "id": "block-a",
                        "label": "ReliabilityBlock",
                        "properties": {
                            "name": "输入模块",
                            "module_id": "MOD-301",
                            "function_id": "FUNC-301",
                        },
                    },
                    {"id": "end", "label": "End", "properties": {"name": "END"}},
                ],
                "edges": [
                    {"id": "e1", "label": "series", "source": "start", "target": "block-a"},
                    {"id": "e2", "label": "series", "source": "block-a", "target": "end"},
                ],
            },
            title="RBD-SYN-999 看起来像编号但只是标题",
        )

        graph = build_formal_diagram_graph("rbd", "doc-rbd", [revision])

        self.assertEqual(1, len(_nodes(graph, "RBDBlock")))
        self.assertEqual(2, len(_nodes(graph, "RBDJunction")))
        self.assertEqual(2, len(_edges(graph, "next_block")))
        self.assertEqual(1, len(_edges(graph, "represents_module")))
        self.assertEqual(1, len(_edges(graph, "implements_function")))
        diagram_nodes = _nodes(graph, "RBDDiagramRef")
        self.assertEqual(1, len(diagram_nodes))
        self.assertFalse(any(item["properties"].get("business_id") for item in diagram_nodes))
        self.assertNotIn(
            business_node_id("rbd_diagram", "RBD-SYN-999"),
            {item["id"] for item in graph["vertices"]},
        )


class StructuredGraphDispatcherTests(unittest.TestCase):
    def test_dispatcher_requires_explicit_function_locator_groups(self):
        with self.assertRaisesRegex(StructuredGraphError, "must map"):
            build_structured_graph("function_locator", "doc-1", [])

        graph = build_structured_graph(
            "function_locator",
            "doc-1",
            {
                "components": [
                    {
                        "component_id": "SYS-1",
                        "component_type": "system",
                        "name": "系统",
                    }
                ],
                "functions": [],
                "interfaces": [],
            },
        )
        self.assertEqual("function_locator", graph["doc_type"])
        self.assertEqual(1, len(graph["vertices"]))

    def test_dispatcher_rejects_unknown_type(self):
        with self.assertRaisesRegex(StructuredGraphError, "unsupported doc_type"):
            build_structured_graph("unknown", "doc-1", [])


if __name__ == "__main__":
    unittest.main()
