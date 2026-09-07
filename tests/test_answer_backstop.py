"""发布边界确定性保底渲染的单元测试。

保底行只转写已批准记录的字段值与确定性计算摘要；触发条件由本轮字段
契约和计算合同决定，不依赖用户措辞。
"""
import unittest

from backend.agent.nodes.answer_backstop import (
    _api_attribute_present,
    _calculation_present,
    _facet_present,
    _mfl_backstop_lines,
    answer_backstop,
)


def _mfl_fields(
    *,
    module_failure="输出接口模块：画面数据输出中断",
    record_code="MFL-SYN-049",
    api_reportable=True,
    api_name="GetOIMFaultStatus",
    api_method="事件订阅并按需查询",
    api_usage="读取输出通道状态字并结合环路自检结果判定。",
    maintenance_action="检查输出接口连接与驱动信号，确认后更换输出接口模块。",
    sur_failure="SUR-OIM-02状态字第3位异常",
    subcircuit_fault="输出编码格式错误。",
    api_reportable_raw=None,
):
    fields = {
        "module_failure": module_failure,
        "record_code": record_code,
        "sur_failure": sur_failure,
        "subcircuit_fault": subcircuit_fault,
        "maintenance_action": maintenance_action,
    }
    if api_reportable is not None:
        fields["api_reportable"] = api_reportable
    if api_reportable_raw is not None:
        fields["api_reportable_raw"] = api_reportable_raw
    if api_name is not None:
        fields["api_name"] = api_name
    if api_method is not None:
        fields["api_method"] = api_method
    if api_usage is not None:
        fields["api_usage"] = api_usage
    return fields


def _ledger_item(doc_type="mfl", record_id="MFL-SYN-049", **field_overrides):
    return {
        "doc_type": doc_type,
        "record_id": record_id,
        "metadata": {"answer_fields": _mfl_fields(**field_overrides)},
    }


class ApiAttributePresenceTests(unittest.TestCase):
    def test_reportable_row_requires_api_name(self):
        fields = _mfl_fields()
        self.assertFalse(_api_attribute_present("该行建议检查连接。", fields))
        self.assertTrue(
            _api_attribute_present("该行可通过 GetOIMFaultStatus 上报。", fields)
        )

    def test_not_reportable_row_requires_explicit_marker(self):
        fields = _mfl_fields(api_reportable=False, api_name="—",
                             api_reportable_raw="否")
        self.assertFalse(_api_attribute_present("建议检查输出连接。", fields))
        self.assertTrue(
            _api_attribute_present("该行不可直接上报，需现场核查。", fields)
        )

    def test_missing_reportable_field_is_treated_as_present(self):
        fields = _mfl_fields(api_reportable=None, api_name=None,
                             api_reportable_raw=None)
        self.assertTrue(_api_attribute_present("", fields))


class FacetPresenceTests(unittest.TestCase):
    def test_unreferenced_record_is_not_present(self):
        self.assertFalse(_facet_present("", "maintenance_action", _mfl_fields()))

    def test_action_value_presence(self):
        fields = _mfl_fields()
        self.assertTrue(
            _facet_present("建议检查输出接口连接与驱动信号后更换模块。[^1]",
                           "maintenance_action", fields)
        )
        self.assertFalse(
            _facet_present("建议检查供电与接地连接。[^1]", "maintenance_action", fields)
        )

    def test_locator_accepts_any_level_value(self):
        fields = _mfl_fields()
        self.assertTrue(
            _facet_present("SUR-OIM-02状态字第3位异常。[^1]",
                           "maintenance_locator", fields)
        )

    def test_empty_facet_values_are_present(self):
        fields = {
            "maintenance_action": "检查输出接口连接与驱动信号，确认后更换输出接口模块。",
        }
        self.assertTrue(_facet_present("无关文本。[^1]", "maintenance_locator", fields))


class MflBackstopLineTests(unittest.TestCase):
    def test_contract_row_without_citation_is_fully_rendered(self):
        ledger = {3: _ledger_item()}
        lines = _mfl_backstop_lines(
            "资料没有给出处理建议。", ledger,
            ["maintenance_action", "maintenance_api"],
        )
        self.assertEqual(len(lines), 1)
        text = lines[0]["text"]
        self.assertIn("[^3]", text)
        self.assertIn("资料建议处置：", text)
        self.assertIn("API可直接上报：是", text)
        self.assertIn("GetOIMFaultStatus", text)
        self.assertIn("（MFL-SYN-049）", text)

    def test_no_contract_fields_means_no_lines(self):
        ledger = {3: _ledger_item()}
        self.assertEqual(
            _mfl_backstop_lines("正文。", ledger, ["fault_cause"]), []
        )

    def test_consumed_row_only_backstops_missing_api_attribute(self):
        ledger = {3: _ledger_item()}
        answer = (
            "建议检查输出接口连接与驱动信号，确认后更换输出接口模块。[^3]"
        )
        lines = _mfl_backstop_lines(
            answer, ledger, ["maintenance_action", "maintenance_locator"]
        )
        self.assertEqual(len(lines), 1)
        self.assertIn("该行可通过API直接上报（GetOIMFaultStatus）", lines[0]["text"])
        self.assertNotIn("资料建议处置：", lines[0]["text"])

    def test_not_reportable_row_gets_negative_api_statement(self):
        ledger = {2: _ledger_item(record_code="MFL-SYN-051", api_reportable=False,
                                  api_name="—", api_reportable_raw="否")}
        lines = _mfl_backstop_lines(
            "建议核查接口时钟与模式编码，确认后更换。[^2]", ledger,
            ["maintenance_action"],
        )
        self.assertEqual(len(lines), 1)
        self.assertIn("API可直接上报：否", lines[0]["text"])

    def test_non_mfl_records_are_ignored(self):
        ledger = {1: _ledger_item(doc_type="fmeca", record_id="FM-SYN-002")}
        self.assertEqual(
            _mfl_backstop_lines("正文。", ledger, ["maintenance_action"]), []
        )

    def test_line_budget_is_bounded(self):
        ledger = {
            index: _ledger_item(record_code=f"MFL-SYN-{index:03d}")
            for index in range(1, 21)
        }
        lines = _mfl_backstop_lines("正文。", ledger, ["maintenance_action"])
        self.assertLessEqual(len(lines), 12)


class CalculationPresenceTests(unittest.TestCase):
    def _calc_item(self, results=None, minimal_cut_sets=None, claim="rbd_calculation"):
        metadata = {"calculation": {"results": results or {}}}
        content = {"minimal_cut_sets": minimal_cut_sets or []}
        import json as _json
        return {
            "claim_type": claim,
            "metadata": metadata,
            "content": _json.dumps(content, ensure_ascii=False),
        }

    def test_exact_reliability_value_marks_present(self):
        item = self._calc_item(results={"system_reliability": 0.99951007})
        self.assertTrue(
            _calculation_present("系统可靠度约 0.99951007。", item)
        )

    def test_rounded_value_is_not_present(self):
        item = self._calc_item(results={"system_reliability": 0.99951007})
        self.assertFalse(_calculation_present("系统可靠度约 0.9995。", item))

    def test_joint_cut_set_members_must_all_appear(self):
        item = self._calc_item(
            results={"top_event_probability": None},
            minimal_cut_sets=[["TCM闭环失效", "风扇反馈丢失"], ["看门狗超时"]],
            claim="fta_calculation",
        )
        self.assertFalse(
            _calculation_present("最小割集包含看门狗超时。", item)
        )
        self.assertTrue(
            _calculation_present(
                "联合割集为 TCM闭环失效 加 风扇反馈丢失；另有看门狗超时。", item
            )
        )


class AnswerBackstopTests(unittest.TestCase):
    def _state(self, fields, calculation_requests=None):
        return {
            "query_frame": {"required_answer_fields": fields},
            "calculation_requests": calculation_requests or [],
        }

    def test_no_backstop_when_everything_consumed(self):
        ledger = {3: _ledger_item()}
        citations = [{"id": 3}]
        answer = (
            "建议检查输出接口连接与驱动信号，确认后更换输出接口模块，"
            "该行可通过 GetOIMFaultStatus 直接上报。[^3]"
        )
        result, report = answer_backstop(
            answer, self._state(["maintenance_action"]), ledger, citations
        )
        self.assertEqual(result, answer)
        self.assertEqual(report, {})

    def test_calculation_summary_appended_with_citation(self):
        item = {
            "claim_type": "rbd_calculation",
            "record_id": "RBD-REV-1",
            "display_summary": "RBD 确定性计算：供电通路；系统可靠度：0.99951007。",
            "metadata": {"calculation": {"results": {"system_reliability": 0.99951007}}},
            "content": "{}",
        }
        ledger = {2: item}
        result, report = answer_backstop(
            "四条通路中显示通路最低。", self._state([], ["rbd_reliability"]),
            ledger, [{"id": 2}],
        )
        self.assertIn("0.99951007", result)
        self.assertIn("[^2]", result)
        self.assertIn("补充确定性计算结果", result)
        self.assertEqual(report["calculation"][0]["record_id"], "RBD-REV-1")

    def test_citation_ids_outside_prompt_window_are_ignored(self):
        ledger = {30: _ledger_item()}
        result, report = answer_backstop(
            "正文。", self._state(["maintenance_action"]), ledger, [{"id": 1}]
        )
        self.assertEqual(result, "正文。")
        self.assertEqual(report, {})

    def test_calculation_requires_current_turn_request(self):
        item = {
            "claim_type": "rbd_calculation",
            "display_summary": "RBD 确定性计算：供电通路；系统可靠度：0.99951007。",
            "metadata": {"calculation": {"results": {"system_reliability": 0.99951007}}},
            "content": "{}",
        }
        result, report = answer_backstop(
            "正文。", self._state([]), {2: item}, [{"id": 2}]
        )
        self.assertEqual(result, "正文。")
        self.assertEqual(report, {})


if __name__ == "__main__":
    unittest.main()
