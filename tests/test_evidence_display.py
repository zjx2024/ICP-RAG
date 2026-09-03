import unittest

from backend.agent.evidence_display import (
    deterministic_graph_source_ref,
    summarize_diagram_locator,
    summarize_fta_calculation,
    summarize_graph_structure,
    summarize_rbd_calculation,
)


class StructuredEvidenceDisplayTests(unittest.TestCase):
    def test_rbd_summary_uses_vertices_and_bounds_function_blocks(self):
        vertices = [
            {"id": "start", "label": "Start", "properties": {"name": "START"}},
            {"id": "end", "label": "End", "properties": {"name": "END"}},
            *[
                {
                    "id": f"b-{index}",
                    "label": "ReliabilityBlock",
                    "properties": {"name": f"模块 {index}"},
                }
                for index in range(1, 17)
            ],
        ]
        summary = summarize_graph_structure(
            {"vertices": vertices, "edges": [{"source": "start", "target": "b-1"}]},
            diagram_type="rbd",
            title="任务处理链路",
        )

        self.assertIn("RBD 结构：任务处理链路", summary)
        self.assertIn("START → END", summary)
        self.assertIn("18 个节点、1 条关系", summary)
        self.assertIn("模块 1", summary)
        # 超过有界完整呈现上限（12）时仍按调用方 limit 节选并声明总数。
        self.assertIn("（共 16 项）", summary)
        self.assertNotIn("模块 7", summary)
        self.assertNotIn("串联", summary)
        self.assertNotIn("并联", summary)

    def test_fta_basic_event_excerpt_includes_bounded_descriptions_and_model_note(self):
        graph = {
            "root_fault_id": "top",
            "metadata": {
                "modeling_assumption": "A、B 总线作为冗余通道；网关为串联单点。",
            },
            "vertices": [
                {"id": "top", "label": "TopEvent", "properties": {"name": "通信中断"}},
                *[
                    {
                        "id": f"e-{index}",
                        "label": "BasicEvent",
                        "properties": {
                            "name": f"事件 {index}",
                            "description": f"明确原因说明 {index} " + "很长" * 60,
                        },
                    }
                    for index in range(1, 16)
                ],
            ],
            "edges": [],
        }
        summary = summarize_graph_structure(
            graph,
            diagram_type="fta",
            title="通信中断",
        )

        # 15 个事件超过有界完整呈现上限，保持节选语义。
        self.assertIn("基本事件（节选）：", summary)
        self.assertNotIn("关键基本事件", summary)
        self.assertIn("事件 1（明确原因说明 1", summary)
        self.assertIn("（共 15 项）", summary)
        self.assertNotIn("事件 6（", summary)
        self.assertIn("建模说明：A、B 总线作为冗余通道；网关为串联单点。", summary)

    def test_rbd_summary_displays_only_explicit_success_and_structure_notes(self):
        summary = summarize_graph_structure(
            {
                "metadata": {
                    "success_criterion": ["至少一条路径成功", "输出端可达"],
                    "structure": "A/B 两路冗余后汇合。",
                },
                "vertices": [],
                "edges": [],
            },
            diagram_type="rbd",
            title="冗余链路",
        )

        self.assertIn("成功判据：至少一条路径成功；输出端可达", summary)
        self.assertIn("结构说明：A/B 两路冗余后汇合。", summary)

    def test_fta_calculation_maps_cut_ids_to_names_and_states_missing_inputs(self):
        graph = {
            "vertices": [
                {"id": "a", "label": "BasicEvent", "properties": {"name": "电源闪断"}},
                {"id": "b", "label": "BasicEvent", "properties": {"name": "复位卡滞"}},
            ]
        }
        summary = summarize_fta_calculation(
            {
                "title": "重复复位故障树",
                "top_event": "系统重复复位",
                "minimal_cut_set_count": 1,
                "minimal_cut_sets": [["a", "b"]],
                "top_event_probability": None,
                "missing_probability_nodes": ["电源闪断"],
            },
            graph=graph,
        )

        self.assertIn("电源闪断 + 复位卡滞", summary)
        self.assertIn("未形成概率结论，缺少参数：电源闪断", summary)
        self.assertNotIn("{", summary)

    def test_fta_structure_does_not_repeat_title_when_it_is_the_top_event(self):
        summary = summarize_graph_structure(
            {
                "root_fault_id": "top",
                "vertices": [
                    {
                        "id": "top",
                        "label": "TopEvent",
                        "properties": {"name": "系统重复复位"},
                    }
                ],
                "edges": [],
            },
            diagram_type="fta",
            title="系统重复复位",
        )
        self.assertEqual(1, summary.count("系统重复复位"))

    def test_rbd_calculation_summary_is_readable_when_paths_are_truncated(self):
        summary = summarize_rbd_calculation(
            {
                "title": "双通道链路",
                "success_path_count": 2,
                "success_paths": [["A 通道"], ["B 通道"]],
                "system_reliability": None,
                "path_limit_reached": True,
                "calculation_limit_reason": "成功路径超过安全枚举上限",
            }
        )

        self.assertIn("已枚举成功路径 2 条", summary)
        self.assertIn("A 通道", summary)
        self.assertIn("未形成可靠度结论：成功路径超过安全枚举上限", summary)
        self.assertNotIn("{", summary)

    def test_invalid_metadata_is_fail_safe_and_not_a_source_locator(self):
        graph = {"metadata": "invalid", "vertices": [], "edges": []}
        self.assertEqual("", deterministic_graph_source_ref(graph))
        self.assertIn(
            "结构包含 0 个节点、0 条关系",
            summarize_graph_structure(graph, diagram_type="fta", title="空图"),
        )

    def test_search_locator_summary_does_not_expose_indexed_structure_json(self):
        summary = summarize_diagram_locator(
            doc_type="fta",
            title="外部数据通信中断",
        )
        self.assertEqual(
            "FTA 故障树检索线索：外部数据通信中断。详细结构需按来源定位回查当前图修订。",
            summary,
        )


if __name__ == "__main__":
    unittest.main()
