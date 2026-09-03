import unittest
from datetime import datetime

from backend.agent.nodes import answer as answer_module
from backend.agent.tools import icp_diagnosis as diagnosis_module


class DirectEvidenceAnswerFieldTests(unittest.TestCase):
    def test_function_locator_exposes_hierarchy_function_and_interface_fields(self):
        function = diagnosis_module._direct_evidence(
            "function_locator",
            [
                {
                    "doc_id": "doc-function",
                    "content_type": "function_record",
                    "function_id": "FUNC-330",
                    "system_name": "综合信息处理与显示系统",
                    "subsystem_name": "数据接口与通信子系统",
                    "module_name": "IFC",
                    "function_name": "外部总线适配",
                    "description": "接收外部数据并完成协议转换",
                    "inputs": ["外部总线数据"],
                    "outputs": ["规范化业务数据"],
                }
            ],
        )[0]
        fields = function["metadata"]["answer_fields"]
        self.assertEqual("FUNC-330", fields["function_id"])
        self.assertEqual("数据接口与通信子系统", fields["subsystem_name"])
        self.assertEqual("IFC", fields["module_name"])
        self.assertEqual(["外部总线数据"], fields["inputs"])
        self.assertEqual(["规范化业务数据"], fields["outputs"])

        interface = diagnosis_module._direct_evidence(
            "function_locator",
            [
                {
                    "doc_id": "doc-function",
                    "content_type": "function_interface",
                    "interface_id": "IF-330-01",
                    "source_component_id": "IFC",
                    "target_component_id": "GPP",
                    "relation_type": "sends_to",
                    "signal_name": "规范化业务数据",
                    "direction": "输出",
                    "protocol": "内部消息",
                }
            ],
        )[0]
        interface_fields = interface["metadata"]["answer_fields"]
        self.assertEqual("IFC", interface_fields["source_component_id"])
        self.assertEqual("GPP", interface_fields["target_component_id"])
        self.assertEqual("sends_to", interface_fields["relation_type"])

    def test_fault_history_exposes_time_cause_steps_solution_and_result(self):
        evidence = diagnosis_module._direct_evidence(
            "fault_history_query",
            [
                {
                    "doc_id": "doc-history",
                    "doc_name": "故障历史.xlsx",
                    "case_id": "CASE-001",
                    "occurred_at": datetime(2026, 8, 28, 9, 30),
                    "mission_phase": "巡航任务阶段",
                    "device_type": "显示处理单元",
                    "related_module": "显示输出模块",
                    "related_function": "FUNC-220",
                    "personnel": "维修员甲",
                    "fault_symptom": "显示链路间歇中断",
                    "root_cause": "连接器接触电阻增大",
                    "diagnosis_method": "测量链路阻抗并进行振动复现",
                    "final_solution": "清洁连接器并更换失效触点",
                    "verification_result": "连续运行四小时未复现",
                    "preventive_action": "定期检查连接器锁紧状态",
                    "hmc_code": "91810001**0001",
                    "fmeca_code": "FM-002",
                    "fta_ref": "FTA-SYN-001",
                    "mfl_code": "MFL-009",
                    "document_links": {
                        "hmc_codes": ["91810001**0001"],
                        "fmeca_refs": ["FM-002"],
                        "fta_refs": ["FTA-SYN-001", "EVT-SYN-001-BE-01"],
                        "mfl_refs": ["MFL-009"],
                    },
                    "steps": [
                        {"step_no": 1, "content": "断电后检查连接器外观"},
                        {"step_no": 2, "content": "测量并对比各触点阻抗"},
                    ],
                    "source_ref": "故障历史.xlsx#row=8",
                }
            ],
        )[0]

        summary = evidence["display_summary"]
        self.assertEqual(summary, evidence["content"])
        self.assertIn("发生时间：2026-08-28 09:30:00", summary)
        self.assertIn("任务阶段/工作模式：巡航任务阶段", summary)
        self.assertIn("设备类型：显示处理单元", summary)
        self.assertIn("关联模块：显示输出模块", summary)
        self.assertIn("关联功能：FUNC-220", summary)
        self.assertIn("排故人员：维修员甲", summary)
        self.assertIn("根因：连接器接触电阻增大", summary)
        self.assertIn("排故方法：测量链路阻抗并进行振动复现", summary)
        self.assertIn("操作步骤：\n1. 断电后检查连接器外观", summary)
        self.assertIn("2. 测量并对比各触点阻抗", summary)
        self.assertIn("最终方案：清洁连接器并更换失效触点", summary)
        self.assertIn("验证结果：连续运行四小时未复现", summary)
        self.assertIn("预防措施：定期检查连接器锁紧状态", summary)
        self.assertIn("关联 HMC：91810001**0001", summary)
        self.assertIn("关联 FMECA：FM-002", summary)
        self.assertIn("关联 FTA：FTA-SYN-001；EVT-SYN-001-BE-01", summary)
        self.assertIn("关联 MFL：MFL-009", summary)

        answer_fields = evidence["metadata"]["answer_fields"]
        self.assertEqual("巡航任务阶段", answer_fields["mission_phase"])
        self.assertEqual("显示处理单元", answer_fields["device_type"])
        self.assertEqual("显示输出模块", answer_fields["related_module"])
        self.assertEqual("FUNC-220", answer_fields["related_function"])
        self.assertEqual("维修员甲", answer_fields["operator"])
        self.assertEqual(
            ["FTA-SYN-001", "EVT-SYN-001-BE-01"],
            answer_fields["explicit_references"]["fta"],
        )

        payload = answer_module._evidence_payload(evidence)
        self.assertEqual(summary, payload["content"])
        self.assertEqual(summary, payload["display_summary"])
        context = answer_module._format_context(
            {
                "evidence_registry": [evidence],
                "context_evidence_ids": [evidence["evidence_id"]],
                "safety_decision": {
                    "approved_evidence_ids": [evidence["evidence_id"]],
                    "blocked_evidence_ids": [],
                    "context_evidence_ids": [evidence["evidence_id"]],
                },
            }
        )
        self.assertIn("发生时间：2026-08-28 09:30:00", context)
        self.assertIn("1. 断电后检查连接器外观", context)
        self.assertIn("最终方案：清洁连接器并更换失效触点", context)
        self.assertIn("验证结果：连续运行四小时未复现", context)

    def test_hmc_fmeca_and_mfl_expose_readable_cause_and_maintenance_fields(self):
        hmc = diagnosis_module._direct_evidence(
            "hmc_query",
            [
                {
                    "doc_id": "doc-hmc",
                    "hmc_code": "91810001**0001",
                    "fault_desc": "ICU 输入过压钳位失效",
                    "repair_method": "检查钳位器件，异常时更换输入调理模块",
                }
            ],
        )
        self.assertEqual("故障现象：ICU 输入过压钳位失效", hmc[0]["display_summary"])
        self.assertEqual(
            "维修措施：检查钳位器件，异常时更换输入调理模块",
            hmc[1]["display_summary"],
        )
        self.assertEqual(
            hmc[1]["display_summary"],
            answer_module._evidence_payload(hmc[1])["content"],
        )
        self.assertEqual(
            "检查钳位器件，异常时更换输入调理模块",
            hmc[1]["metadata"]["answer_fields"]["maintenance_action"],
        )

        fmeca = diagnosis_module._direct_evidence(
            "fmeca_query",
            [
                {
                    "doc_id": "doc-fmeca",
                    "code": "FM-002",
                    "product_flag": "ICU / FUNC-110",
                    "func_management": "外部电源输入调理",
                    "fault_mode": "输入滤波性能退化",
                    "fault_cause": "滤波电容老化",
                    "mission_phase": "持续运行后",
                    "local_impact": "纹波增大",
                    "final_impact": "接口数据间歇异常",
                    "probability": "0.025",
                    "probability_raw": "2.5%",
                    "probability_level": "B",
                    "detect_method": "测量电容 ESR 与输入纹波",
                    "design_improve": "提高电容寿命裕量",
                    "compensate_action": "定期检测 ESR，超限后更换",
                }
            ],
        )[0]
        self.assertIn("产品/功能标志：ICU / FUNC-110", fmeca["display_summary"])
        self.assertIn("功能：外部电源输入调理", fmeca["display_summary"])
        self.assertIn("故障原因：滤波电容老化", fmeca["display_summary"])
        self.assertIn("任务阶段与工作方式：持续运行后", fmeca["display_summary"])
        self.assertIn("检测方法：测量电容 ESR 与输入纹波", fmeca["display_summary"])
        self.assertIn("设计改进措施：提高电容寿命裕量", fmeca["display_summary"])
        self.assertIn("使用补偿措施：定期检测 ESR，超限后更换", fmeca["display_summary"])
        self.assertIn("故障概率：2.5%", fmeca["display_summary"])
        self.assertIn("故障概率等级：B", fmeca["display_summary"])
        self.assertEqual(
            {
                "record_code": "FM-002",
                "product_flag": "ICU / FUNC-110",
                "function": "外部电源输入调理",
                "failure_mode": "输入滤波性能退化",
                "fault_cause": "滤波电容老化",
                "mission_phase": "持续运行后",
                "local_impact": "纹波增大",
                "final_impact": "接口数据间歇异常",
                "failure_probability": "2.5%",
                "failure_probability_normalized": "0.025",
                "failure_probability_level": "B",
                "detection_method": "测量电容 ESR 与输入纹波",
                "design_improvement": "提高电容寿命裕量",
                "compensation_action": "定期检测 ESR，超限后更换",
            },
            fmeca["metadata"]["answer_fields"],
        )

        mfl = diagnosis_module._direct_evidence(
            "mfl_query",
            [
                {
                    "doc_id": "doc-mfl",
                    "mfl_code": "MFL-009",
                    "module_fault": "输入调理模块故障",
                    "sur_fault": "采样通道异常",
                    "subcircuit_fault": "滤波电路性能退化",
                    "api_reportable": False,
                    "api_reportable_raw": "N",
                    "api_name": "read_input_status",
                    "api_method": "读取输入状态字",
                    "api_usage": "先读取状态，再执行复测",
                    "platform_test_status": "已完成",
                    "platform_test": False,
                    "platform_test_raw": "未通过",
                    "maintenance_action": "检查滤波电容，超限后更换模块",
                }
            ],
        )[0]
        self.assertIn("SUR板故障模式：采样通道异常", mfl["display_summary"])
        self.assertIn("API可直接上报：否", mfl["display_summary"])
        self.assertIn("API上报原始值：N", mfl["display_summary"])
        self.assertIn("API调用方法：读取输入状态字", mfl["display_summary"])
        self.assertIn("平台级实验结果：否", mfl["display_summary"])
        self.assertIn("平台级实验原始值：未通过", mfl["display_summary"])
        self.assertIn("维修措施：检查滤波电容，超限后更换模块", mfl["display_summary"])
        self.assertIs(False, mfl["metadata"]["answer_fields"]["api_reportable"])
        self.assertEqual("N", mfl["metadata"]["answer_fields"]["api_reportable_raw"])
        self.assertIs(False, mfl["metadata"]["answer_fields"]["platform_test"])
        self.assertEqual(
            mfl["display_summary"],
            answer_module._evidence_payload(mfl)["content"],
        )

    def test_history_legacy_keys_are_used_but_missing_references_are_not_invented(self):
        evidence = diagnosis_module._direct_evidence(
            "fault_history_query",
            [
                {
                    "doc_id": "doc-legacy-history",
                    "case_id": "CASE-LEGACY",
                    "task_stage": "地面联调",
                    "equipment_type": "接口控制器",
                    "module_name": "接口模块",
                    "function_name": "数据转发",
                    "operator": "值班工程师",
                    "fault_symptom": "外部链路无数据",
                    "root_cause": "连接器松动",
                }
            ],
        )[0]

        summary = evidence["display_summary"]
        self.assertIn("任务阶段/工作模式：地面联调", summary)
        self.assertIn("设备类型：接口控制器", summary)
        self.assertIn("关联模块：接口模块", summary)
        self.assertIn("关联功能：数据转发", summary)
        self.assertIn("排故人员：值班工程师", summary)
        self.assertNotIn("关联 HMC", summary)
        self.assertNotIn("explicit_references", evidence["metadata"]["answer_fields"])


if __name__ == "__main__":
    unittest.main()
