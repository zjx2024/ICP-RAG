import asyncio
import json
import unittest
from unittest.mock import patch

from backend.agent.contracts import make_evidence
from backend.agent.nodes import answer as answer_module


def _run(coroutine):
    return asyncio.run(coroutine)


def _approved_state(*evidence, intent, targets, query="用户问题"):
    items = list(evidence)
    ids = [item["evidence_id"] for item in items]
    return {
        "query": query,
        "intent": intent,
        "query_targets": [{"intent": value} for value in targets],
        "evidence_registry": items,
        "context_evidence_ids": list(ids),
        "safety_decision": {
            "approved_evidence_ids": list(ids),
            "context_evidence_ids": list(ids),
        },
    }


def _with_required_fields(state, *fields):
    state["query_frame"] = {
        "coverage_scope": "focused_detail",
        "required_answer_fields": list(fields),
    }
    return state


def _data_output_parallel_vertices():
    return [
        {
            "id": "data-a",
            "properties": {
                "connection_type": "parallel",
                "branch_id": "root.data:branch:1",
                "module_id": "MOD-DATA-A",
                "name": "A路数据输出",
            },
        },
        {
            "id": "data-b",
            "properties": {
                "connection_type": "parallel",
                "branch_id": "root.data:branch:2",
                "module_id": "MOD-DATA-B",
                "name": "B路数据输出",
            },
        },
    ]


def _authorize_data_output_topology(state):
    state["topology_dependency"] = {
        "required": True,
        "subject": "A路数据输出",
        "subject_kind": "data_path",
        "failure_state": "失效",
        "question_aspects": ["continued_operation"],
        "canonical_business_ids": ["MOD-DATA-A"],
    }
    return state


def _evidence(
    *,
    doc_type,
    record_id,
    claim_type,
    answer_fields=None,
    content="仅供测试的来源正文",
    source_kind="postgres",
    authority="authoritative_record",
):
    return make_evidence(
        source_kind=source_kind,
        authority=authority,
        evidence_level="L1_DIRECT" if source_kind == "postgres" else "L2_CORROBORATED",
        claim_type=claim_type,
        doc_type=doc_type,
        doc_id=f"doc-{doc_type}",
        record_id=record_id,
        title=record_id,
        content=content,
        display_summary=content,
        review_status="auto_validated" if source_kind == "hugegraph" else "not_required",
        metadata={"answer_fields": answer_fields or {}},
    )


class AnswerTargetCompletenessReviewTests(unittest.TestCase):
    def test_focused_severity_requires_every_same_grade_fmeca_record(self):
        first = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-002",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "输入滤波性能退化",
                "local_impact": "输入纹波增大",
                "severity": "III",
            },
        )
        second = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-003",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "储能器件内阻升高",
                "local_impact": "瞬态支撑能力下降",
                "severity": "III",
            },
        )
        state = _with_required_fields(
            _approved_state(
                first,
                second,
                intent="diagnosis_followup",
                targets=["diagnosis_followup"],
                query="这种异常是什么级别？",
            ),
            "severity",
        )

        missing, manifest = answer_module._target_completion_manifest(
            "可能出现输入纹波增大和瞬态支撑能力下降。[^1][^2]", state
        )

        self.assertEqual(["severity"], missing)
        self.assertEqual(
            {"FM-SYN-002", "FM-SYN-003"},
            {
                item["record_id"]
                for item in manifest
                if "severity" in item["supports"]
            },
        )

        complete_missing, _ = answer_module._target_completion_manifest(
            "两种故障模式的严酷度均为 III。[^1][^2]", state
        )

        self.assertEqual([], complete_missing)

    def test_focused_severity_with_different_grades_requires_record_binding(self):
        first = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-002",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "输入滤波性能退化",
                "severity": "II",
            },
        )
        second = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-003",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "储能器件内阻升高",
                "severity": "III",
            },
        )
        state = _with_required_fields(
            _approved_state(
                first,
                second,
                intent="diagnosis_followup",
                targets=["diagnosis_followup"],
                query="它们分别是什么级别？",
            ),
            "severity",
        )

        ambiguous_missing, _ = answer_module._target_completion_manifest(
            "两条记录的严酷度分别为 II 和 III。[^1][^2]", state
        )
        bound_missing, _ = answer_module._target_completion_manifest(
            "FM-SYN-002 的严酷度为 II。[^1]\n"
            "FM-SYN-003 的严酷度为 III。[^2]",
            state,
        )

        self.assertEqual(["severity"], ambiguous_missing)
        self.assertEqual([], bound_missing)

    def test_focused_severity_is_not_satisfied_by_impact_only(self):
        fmeca = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-002",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "输入滤波性能退化",
                "local_impact": "输入纹波增大",
                "final_impact": "系统间歇复位",
                "severity": "III",
            },
        )
        state = _with_required_fields(
            _approved_state(
                fmeca,
                intent="diagnosis_followup",
                targets=["diagnosis_followup"],
                query="该异常级别？",
            ),
            "severity",
        )

        missing, _ = answer_module._target_completion_manifest(
            "局部影响是输入纹波增大，最终影响是系统间歇复位。[^1]", state
        )

        self.assertEqual(["severity"], missing)

    def test_multi_dimension_diagnosis_does_not_turn_top_k_into_n_of_n_manifest(self):
        first = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-002",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "输入滤波性能退化",
                "fault_cause": "滤波电容老化",
                "local_impact": "输入纹波增大",
                "final_impact": "系统间歇复位",
                "severity": "III",
                "detection_method": "读取输入电压状态字",
            },
        )
        second = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-003",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "储能器件内阻升高",
                "fault_cause": "电解液老化",
                "local_impact": "瞬态支撑能力下降",
                "final_impact": "高负载时复位",
                "severity": "III",
                "detection_method": "测量等效串联电阻",
            },
        )
        maintenance = _evidence(
            doc_type="mfl",
            record_id="MFL-SYN-041",
            claim_type="maintenance_locator",
            answer_fields={
                "record_code": "MFL-SYN-041",
                "module_failure": "输入调理模块故障",
                "maintenance_action": "检查采样链路后更换异常模块",
            },
        )
        history = _evidence(
            doc_type="fault_history",
            record_id="CASE-SYN-017",
            claim_type="historical_case",
            answer_fields={
                "case_id": "CASE-SYN-017",
                "occurred_at": "2025-01-12 09:20",
                "root_cause": "输入钳位器件性能退化",
                "steps": ["检查采样链路", "更换输入调理模块"],
                "final_solution": "更换输入调理模块并复测",
                "verification_result": "连续运行 72 小时正常",
            },
        )
        state = _approved_state(
            first,
            second,
            maintenance,
            history,
            intent="cause_analysis",
            targets=[
                "cause_analysis",
                "impact_analysis",
                "maintenance_query",
                "history_query",
            ],
            query="请综合说明候选原因、影响、处理方法和历史案例",
        )
        state["query_frame"] = {
            "coverage_scope": "focused_detail",
            "required_answer_fields": [
                "failure_mode",
                "fault_cause",
                "local_impact",
                "final_impact",
                "severity",
                "detection_method",
                "maintenance_action",
                "history_time",
                "history_root",
                "history_steps",
                "history_solution",
                "history_result",
            ],
        }

        self.assertFalse(answer_module.focused_answer_contract(state))
        self.assertEqual(
            ([], []),
            answer_module._target_completion_manifest("", state),
        )

    def test_focused_history_solution_uses_same_generic_contract(self):
        history = _evidence(
            doc_type="fault_history",
            record_id="CASE-SYN-017",
            claim_type="fault_history_case",
            answer_fields={
                "root_cause": "输入钳位器件性能退化",
                "final_solution": "更换输入调理模块并复测",
            },
        )
        state = _with_required_fields(
            _approved_state(
                history,
                intent="history_query",
                targets=["history_query"],
                query="以前最后是怎么解决的？",
            ),
            "history_solution",
        )

        missing, _ = answer_module._target_completion_manifest(
            "历史案例确认根因为输入钳位器件性能退化。[^1]", state
        )
        complete_missing, _ = answer_module._target_completion_manifest(
            "历史最终方案是更换输入调理模块并复测。[^1]", state
        )

        self.assertEqual(["history_solution"], missing)
        self.assertEqual([], complete_missing)

    def test_focused_entity_name_uses_same_generic_contract(self):
        component = _evidence(
            doc_type="function_locator",
            record_id="MOD-PWR-ICU",
            claim_type="component_fact",
            answer_fields={
                "component_id": "MOD-PWR-ICU",
                "component_name": "输入调理模块",
                "component_type": "module",
                "parent_component_id": "SUB-PWR",
            },
        )
        state = _with_required_fields(
            _approved_state(
                component,
                intent="entity_profile",
                targets=["entity_profile"],
                query="MOD-PWR-ICU 是什么？",
            ),
            "record_name",
        )

        missing, _ = answer_module._target_completion_manifest(
            "`MOD-PWR-ICU` 位于 SUB-PWR。[^1]", state
        )
        complete_missing, _ = answer_module._target_completion_manifest(
            "`MOD-PWR-ICU` 的标准名称是输入调理模块。[^1]", state
        )

        self.assertEqual(["record_name"], missing)
        self.assertEqual([], complete_missing)

    def test_mfl_record_identity_accepts_each_supported_failure_level(self):
        failure_fields = (
            ("module_failure", "协议转换模块故障"),
            ("sur_failure", "SUR 接口板收发异常"),
            ("subcircuit_failure", "转换队列配置异常"),
        )
        for index, (field_name, field_value) in enumerate(failure_fields, start=1):
            with self.subTest(field_name=field_name):
                record_code = f"MFL-SYN-04{index}"
                mfl = _evidence(
                    doc_type="mfl",
                    record_id=record_code,
                    claim_type="maintenance_locator",
                    answer_fields={
                        "record_code": record_code,
                        field_name: field_value,
                    },
                )
                state = _with_required_fields(
                    _approved_state(
                        mfl,
                        intent="entity_profile",
                        targets=["entity_profile"],
                        query=f"{record_code} 是什么？",
                    ),
                    "record_identity",
                )

                missing, _ = answer_module._target_completion_manifest(
                    f"记录代码是 `{record_code}`。[^1]", state
                )
                complete_missing, _ = answer_module._target_completion_manifest(
                    f"`{record_code}` 对应{field_value}。[^1]", state
                )

                self.assertEqual(["record_identity"], missing)
                self.assertEqual([], complete_missing)

    def test_fault_history_record_identity_accepts_case_id_and_symptom(self):
        history = _evidence(
            doc_type="fault_history",
            record_id="CASE-SYN-017",
            claim_type="fault_history_case",
            answer_fields={
                "case_id": "CASE-SYN-017",
                "fault_symptom": "上电后输入电压反复波动",
            },
        )
        state = _with_required_fields(
            _approved_state(
                history,
                intent="history_query",
                targets=["history_query"],
                query="CASE-SYN-017 是什么记录？",
            ),
            "record_identity",
        )

        missing, _ = answer_module._target_completion_manifest(
            "案例编号是 `CASE-SYN-017`。[^1]", state
        )
        complete_missing, _ = answer_module._target_completion_manifest(
            "`CASE-SYN-017` 记录的故障现象是上电后输入电压反复波动。[^1]",
            state,
        )

        self.assertEqual(["record_identity"], missing)
        self.assertEqual([], complete_missing)

    def test_architecture_receiver_natural_flow_wording_is_not_appended_again(self):
        manifest = [
            {
                "citation_id": 1,
                "record_id": "MOD-CHANNEL-A",
                "supports": ["architecture_receivers"],
                "answer_fields": {
                    "component_id": "MOD-CHANNEL-A",
                    "component_name": "A通道总线控制模块",
                    "abbreviation": "BCA",
                    "description": "接收外部总线数据。",
                },
            }
        ]
        answer = "外部数据先进入 A通道总线控制模块（BCA / MOD-CHANNEL-A）。[^1]"

        self.assertTrue(
            answer_module._target_facet_is_present(
                answer, "architecture_receivers", manifest
            )
        )
        self.assertEqual(
            answer,
            answer_module._append_missing_architecture_roles(
                answer, ["architecture_receivers"], manifest
            ),
        )

    def test_rbd_structure_requires_every_explicit_user_facing_field(self):
        graph = {
            "diagram_type": "rbd",
            "metadata": {
                "success_criterion": "A/B 任一路可用即可维持数据输出",
                "structure": "A/B 两路并联后与输出网关串联",
                "redundancy_structure": "A/B 两路冗余",
                "single_failure_boundary": "任一路失效时，另一条路径仍可维持数据输出",
                "missing_parameters": ["输出网关故障率"],
            },
            "vertices": _data_output_parallel_vertices(),
            "edges": [],
        }
        rbd = _evidence(
            doc_type="rbd",
            record_id="RBD-DATA-OUTPUT",
            claim_type="rbd_structure",
            answer_fields={},
            content=json.dumps(graph, ensure_ascii=False),
            authority="validated_relation",
        )
        state = _authorize_data_output_topology(
            _approved_state(
                rbd,
                intent="reliability_query",
                targets=["reliability_query"],
                query="A路数据输出坏了还能不能输出，当前还缺什么参数？",
            )
        )

        missing, manifest = answer_module._target_completion_manifest(
            "该模型包含冗余设计。[^1]", state
        )

        self.assertEqual(
            [
                "rbd_success_criterion",
                "rbd_path_structure",
                "rbd_redundancy_structure",
                "rbd_single_failure_boundary",
                "rbd_missing_parameters",
            ],
            missing,
        )
        self.assertEqual("A/B 两路并联后与输出网关串联", manifest[0]["answer_fields"]["structure_description"])

    def test_rbd_structure_editor_fills_explicit_fields_without_calculating(self):
        graph = {
            "diagram_type": "rbd",
            "metadata": {
                "success_criterion": "A/B 任一路可用即可维持数据输出",
                "structure": "A/B 两路并联后与输出网关串联",
                "redundancy_structure": "A/B 两路冗余",
                "single_failure_boundary": "任一路失效时，另一条路径仍可维持数据输出",
                "missing_parameters": ["输出网关故障率"],
                "unapproved_note": "不得进入编辑器的内部备注",
            },
            "vertices": _data_output_parallel_vertices(),
            "edges": [],
        }
        rbd = _evidence(
            doc_type="rbd",
            record_id="RBD-DATA-OUTPUT",
            claim_type="rbd_structure",
            answer_fields={},
            content=json.dumps(graph, ensure_ascii=False),
            authority="validated_relation",
        )
        state = _authorize_data_output_topology(
            _approved_state(
                rbd,
                intent="reliability_query",
                targets=["reliability_query"],
                query="A路数据输出坏了还能不能输出，当前还缺什么参数？",
            )
        )
        draft = "该模型包含冗余设计。[^1]"
        complete = (
            "成功判据是 A/B 任一路可用即可维持数据输出；结构为 A/B 两路并联后与输出网关串联，"
            "其冗余结构是 A/B 两路冗余。[^1]\n\n"
            "能力边界：任一路失效时，另一条路径仍可维持数据输出；"
            "缺失参数为输出网关故障率。[^1]"
        )

        with patch.object(answer_module.llm_service, "_call_model", return_value=complete) as call:
            result = _run(
                answer_module._review_answer_target_completeness(draft, state, None)
            )

        self.assertEqual(complete, result)
        call.assert_called_once()
        prompt = call.call_args.args[0][0]["content"]
        self.assertIn("输出网关故障率", prompt)
        self.assertNotIn("系统可靠度", prompt)
        self.assertNotIn("不得进入编辑器的内部备注", prompt)

    def test_rbd_structure_does_not_infer_unstated_failure_boundary_or_gaps(self):
        graph = {
            "diagram_type": "rbd",
            "metadata": {"success_criterion": "输出端可达"},
            "vertices": [],
            "edges": [],
        }
        rbd = _evidence(
            doc_type="rbd",
            record_id="RBD-MINIMAL",
            claim_type="rbd_structure",
            answer_fields={},
            content=json.dumps(graph, ensure_ascii=False),
            authority="validated_relation",
        )
        state = _approved_state(
            rbd,
            intent="reliability_query",
            targets=["reliability_query"],
        )

        missing, manifest = answer_module._target_completion_manifest("输出端可达。[^1]", state)

        self.assertEqual([], missing)
        supports = {facet for item in manifest for facet in item["supports"]}
        self.assertNotIn("rbd_single_failure_boundary", supports)
        self.assertNotIn("rbd_missing_parameters", supports)

    def test_numeric_rbd_request_is_left_to_calculation_contract(self):
        graph = {
            "diagram_type": "rbd",
            "metadata": {
                "success_criterion": "输出端可达",
                "structure": "A/B 两路并联",
            },
            "vertices": [],
            "edges": [],
        }
        rbd = _evidence(
            doc_type="rbd",
            record_id="RBD-NUMERIC",
            claim_type="rbd_structure",
            answer_fields={},
            content=json.dumps(graph, ensure_ascii=False),
            authority="validated_relation",
        )
        state = _approved_state(
            rbd,
            intent="reliability_query",
            targets=["reliability_query"],
        )
        state["calculation_requests"] = ["rbd_reliability"]

        self.assertEqual(([], []), answer_module._target_completion_manifest("", state))

    def test_exact_hmc_requires_supported_identity_symptom_and_action(self):
        hmc = _evidence(
            doc_type="hmc",
            record_id="91810001**0001",
            claim_type="hmc_fault",
            answer_fields={
                "hmc_code": "91810001**0001",
                "lcn": "91810001",
                "hamc": "**",
                "hsc": "0001",
                "fault_symptom": "ICU 输入过压钳位失效",
                "maintenance_action": "检查输入采样与钳位器件",
            },
        )
        state = _approved_state(
            hmc,
            intent="exact_code",
            targets=["exact_code"],
            query="91810001-**-0001 是什么，怎么处理？",
        )

        missing, manifest = answer_module._target_completion_manifest(
            "文档记录的故障是 ICU 输入过压钳位失效。[^1]",
            state,
        )

        self.assertEqual(["hmc_identity", "maintenance_action"], missing)
        self.assertEqual({"hmc_identity", "hmc_symptom", "maintenance_action"}, {
            facet for item in manifest for facet in item["supports"]
        })
        self.assertEqual("91810001**0001", manifest[0]["answer_fields"]["hmc_code"])

    def test_one_hmc_component_does_not_count_as_complete_identity(self):
        hmc = _evidence(
            doc_type="hmc",
            record_id="91810001**0001",
            claim_type="hmc_fault",
            answer_fields={
                "hmc_code": "91810001**0001",
                "lcn": "91810001",
                "hamc": "**",
                "hsc": "0001",
                "fault_symptom": "ICU 输入过压钳位失效",
            },
        )
        state = _approved_state(
            hmc,
            intent="exact_code",
            targets=["exact_code"],
        )

        missing, _manifest = answer_module._target_completion_manifest(
            "HSC 是 `0001`，故障现象为 ICU 输入过压钳位失效。[^1]", state
        )

        self.assertEqual(["hmc_identity"], missing)

    def test_maintenance_target_covers_locator_action_and_api_when_supported(self):
        mfl = _evidence(
            doc_type="mfl",
            record_id="MFL-SYN-041",
            claim_type="maintenance_locator",
            answer_fields={
                "module_failure": "协议转换模块故障",
                "subcircuit_failure": "转换队列配置异常",
                "maintenance_action": "检查转换队列和配置，异常时更换协议转换模块",
                "api_reportable": True,
                "api_name": "GetFrameStatus",
                "api_method": "读取转换队列状态",
            },
        )
        state = _approved_state(
            mfl,
            intent="maintenance_query",
            targets=["maintenance_query"],
            query="这类问题怎么定位和处理，有没有状态接口？",
        )

        missing, _manifest = answer_module._target_completion_manifest(
            "建议检查转换队列和配置，异常时更换协议转换模块。[^1]",
            state,
        )

        self.assertEqual(["maintenance_locator", "maintenance_api"], missing)

    def test_generic_api_mention_does_not_hide_available_name_and_method(self):
        mfl = _evidence(
            doc_type="mfl",
            record_id="MFL-SYN-041",
            claim_type="maintenance_locator",
            answer_fields={
                "maintenance_action": "检查转换队列和配置",
                "api_reportable": True,
                "api_name": "GetFrameStatus",
                "api_method": "读取转换队列状态",
            },
        )
        state = _approved_state(
            mfl,
            intent="maintenance_query",
            targets=["maintenance_query"],
        )

        missing, _manifest = answer_module._target_completion_manifest(
            "建议检查转换队列和配置；文档还记载了可用 API。[^1]", state
        )

        self.assertEqual(["maintenance_api"], missing)

    def test_architecture_and_exact_code_multitargets_are_both_checked(self):
        hmc = _evidence(
            doc_type="hmc",
            record_id="91810001**0001",
            claim_type="hmc_fault",
            answer_fields={
                "hmc_code": "91810001**0001",
                "fault_symptom": "ICU 输入过压钳位失效",
                "maintenance_action": "检查输入采样与钳位器件",
            },
        )
        function = _evidence(
            doc_type="function_locator",
            record_id="FUNC-330",
            claim_type="function_fact",
            answer_fields={
                "system_name": "综合信息处理与显示系统",
                "subsystem_name": "数据接口与通信子系统",
                "module_name": "IFC",
                "function_name": "外部总线适配",
                "description": "接收外部数据并完成协议转换",
            },
        )
        relation = _evidence(
            doc_type="function_locator",
            record_id="PATH-IFC-GPP",
            claim_type="relation_path",
            source_kind="hugegraph",
            authority="validated_relation",
            answer_fields={
                "source_entity": "IFC",
                "relation": "sends_to",
                "target_entity": "GPP",
                "path_summary": "IFC —[sends_to]→ GPP",
            },
        )
        state = _approved_state(
            hmc,
            function,
            relation,
            intent="architecture_query",
            targets=["architecture_query", "exact_code"],
            query="这个 HMC 属于哪个模块，并通过什么关系连到处理单元？",
        )
        draft = (
            "`91810001**0001` 记录 ICU 输入过压钳位失效，"
            "处理时检查输入采样与钳位器件。[^1]"
        )

        missing, _manifest = answer_module._target_completion_manifest(draft, state)

        self.assertEqual(
            ["architecture_hierarchy", "architecture_function", "relation_path"],
            missing,
        )

    def test_relation_requires_path_not_just_one_endpoint(self):
        relation = _evidence(
            doc_type="function_locator",
            record_id="PATH-IFC-GPP",
            claim_type="relation_path",
            source_kind="hugegraph",
            authority="validated_relation",
            answer_fields={
                "source_entity": "IFC",
                "relation": "sends_to",
                "target_entity": "GPP",
                "path_summary": "IFC —[sends_to]→ GPP",
            },
        )
        state = _approved_state(
            relation,
            intent="architecture_query",
            targets=["architecture_query"],
        )

        missing, _manifest = answer_module._target_completion_manifest(
            "起点是 IFC。[^1]", state
        )

        self.assertEqual(["relation_path"], missing)

    def test_plural_architecture_roles_require_each_receiver_and_function_record(self):
        bca = _evidence(
            doc_type="function_locator",
            record_id="MOD-CHANNEL-A",
            claim_type="component_fact",
            answer_fields={
                "component_id": "MOD-CHANNEL-A",
                "component_name": "A通道总线控制模块",
                "component_type": "module",
                "abbreviation": "BCA",
                "parent_component_id": "SUB-DATA",
                "description": "接收与发送A通道外部总线数据。",
            },
        )
        bcb = _evidence(
            doc_type="function_locator",
            record_id="MOD-CHANNEL-B",
            claim_type="component_fact",
            answer_fields={
                "component_id": "MOD-CHANNEL-B",
                "component_name": "B通道总线控制模块",
                "component_type": "module",
                "abbreviation": "BCB",
                "parent_component_id": "SUB-DATA",
                "description": "接收与发送B通道外部总线数据。",
            },
        )
        subsystem = _evidence(
            doc_type="function_locator",
            record_id="SUB-DATA",
            claim_type="component_fact",
            answer_fields={
                "component_id": "SUB-DATA",
                "component_name": "数据接口与通信子系统",
                "component_type": "subsystem",
                "abbreviation": "IFC",
                "description": "完成双通道总线收发、网络接入和协议转换。",
            },
        )
        adapter = _evidence(
            doc_type="function_locator",
            record_id="MOD-PROTOCOL",
            claim_type="component_fact",
            answer_fields={
                "component_id": "MOD-PROTOCOL",
                "component_name": "协议转换模块",
                "component_type": "module",
                "abbreviation": "PCU",
                "description": "在内部数据格式与外部协议之间进行转换。",
            },
        )
        adapter_function = _evidence(
            doc_type="function_locator",
            record_id="FUNC-ADAPT",
            claim_type="function_fact",
            answer_fields={
                "function_id": "FUNC-ADAPT",
                "function_name": "协议映射与帧转换",
                "module_name": "协议转换模块",
                "description": "在内部数据格式与外部协议之间进行转换。",
            },
        )
        downstream = _evidence(
            doc_type="function_locator",
            record_id="SUB-PROCESSOR",
            claim_type="component_fact",
            answer_fields={
                "component_id": "SUB-PROCESSOR",
                "component_name": "通用处理子系统",
                "component_type": "subsystem",
                "abbreviation": "GPP",
                "description": "完成任务计算与数据处理。",
            },
        )
        state = _approved_state(
            bca,
            bcb,
            subsystem,
            adapter,
            adapter_function,
            downstream,
            intent="architecture_query",
            targets=["architecture_query"],
            query="外部总线送入的信息由哪些单元接收，谁负责格式适配，之后交给哪个处理子系统？",
        )
        draft = (
            "协议转换模块（PCU）负责格式适配，在内部数据格式与外部协议之间进行转换。[^4]\n\n"
            "适配后的信息送入通用处理子系统。[^6]"
        )

        missing, manifest = answer_module._target_completion_manifest(draft, state)

        self.assertIn("architecture_receivers", missing)
        self.assertIn("architecture_receiver_context", missing)
        self.assertIn("architecture_responsibility_function", missing)
        self.assertNotIn("architecture_responsible_component", missing)
        self.assertIn("architecture_downstream", missing)
        receiver_records = {
            item["record_id"]
            for item in manifest
            if "architecture_receivers" in item["supports"]
        }
        self.assertEqual({"MOD-CHANNEL-A", "MOD-CHANNEL-B"}, receiver_records)

        completed = answer_module._append_missing_architecture_roles(
            draft, missing, manifest
        )
        self.assertIn("A通道总线控制模块", completed)
        self.assertIn("B通道总线控制模块", completed)
        self.assertIn("数据接口与通信子系统", completed)
        self.assertIn("FUNC-ADAPT", completed)
        self.assertIn("SUB-PROCESSOR", completed)
        remaining, _ = answer_module._target_completion_manifest(completed, state)
        self.assertFalse(
            answer_module._ARCHITECTURE_ROLE_FACETS & set(remaining),
            remaining,
        )

    def test_closed_architecture_inventory_requires_every_direct_member(self):
        names = (
            ("SUB-PWR", "电源与配电子系统", "PWR"),
            ("SUB-GPP", "通用处理子系统", "GPP"),
            ("SUB-IFC", "数据接口与通信子系统", "IFC"),
            ("SUB-DSP", "显示处理与输出子系统", "DSP"),
            ("SUB-HMS", "健康监测与记录子系统", "HMS"),
            ("SUB-TCS", "时钟与热管理子系统", "TCS"),
        )
        evidence = [
            _evidence(
                doc_type="function_locator",
                record_id=component_id,
                claim_type="component_fact",
                answer_fields={
                    "component_id": component_id,
                    "component_name": name,
                    "component_type": "subsystem",
                    "abbreviation": abbreviation,
                    "parent_component_id": "SYN-ICP-9000",
                    "description": f"{name}的权威职责。",
                    "collection_closed": True,
                    "collection_parent_id": "SYN-ICP-9000",
                    "collection_parent_name": "综合信息处理与显示系统",
                    "collection_member_kind": "subsystem",
                    "collection_member_count": 6,
                    "collection_role": "member",
                    "collection_member_id": component_id,
                    "collection_member_name": name,
                },
            )
            for component_id, name, abbreviation in names
        ]
        state = _approved_state(
            *evidence,
            intent="architecture_query",
            targets=["architecture_query"],
            query="请介绍整套系统架构",
        )
        draft = "\n".join(
            f"- **{name}**。[^{index}]"
            for index, (_component_id, name, _abbreviation) in enumerate(
                names[:4], start=1
            )
        )

        missing, manifest = answer_module._target_completion_manifest(draft, state)

        self.assertIn("closed_collection_members", missing)
        completed = answer_module._append_missing_architecture_roles(
            draft, missing, manifest
        )
        self.assertIn("共 6 个子系统", completed)
        for _component_id, name, _abbreviation in names:
            self.assertIn(name, completed)
        remaining, _ = answer_module._target_completion_manifest(completed, state)
        self.assertNotIn("closed_collection_members", remaining)

    def test_closed_architecture_inventory_rejects_wrong_declared_count(self):
        names = (
            ("SUB-A", "甲子系统"),
            ("SUB-B", "乙子系统"),
            ("SUB-C", "丙子系统"),
        )
        evidence = [
            _evidence(
                doc_type="function_locator",
                record_id=component_id,
                claim_type="component_fact",
                answer_fields={
                    "component_id": component_id,
                    "component_name": name,
                    "component_type": "subsystem",
                    "parent_component_id": "SYS-1",
                    "collection_closed": True,
                    "collection_parent_id": "SYS-1",
                    "collection_parent_name": "测试系统",
                    "collection_member_kind": "subsystem",
                    "collection_member_count": 3,
                    "collection_role": "member",
                    "collection_member_id": component_id,
                    "collection_member_name": name,
                },
            )
            for component_id, name in names
        ]
        state = _approved_state(
            *evidence,
            intent="architecture_query",
            targets=["architecture_query"],
        )
        wrong = "系统共有 2 个子系统：" + "、".join(
            f"{name}[^{index}]"
            for index, (_component_id, name) in enumerate(names, start=1)
        )

        missing, manifest = answer_module._target_completion_manifest(wrong, state)

        self.assertIn("closed_collection_members", missing)
        completed = answer_module._append_missing_architecture_roles(
            wrong, missing, manifest
        )
        self.assertNotIn("共有 2 个子系统", completed)
        self.assertIn("共 3 个子系统", completed)

    def test_evidence_window_reserves_complete_architecture_inventory(self):
        generic = [
            _evidence(
                doc_type="function_locator",
                record_id=f"FUNC-{index:03d}",
                claim_type="function_fact",
                answer_fields={
                    "function_id": f"FUNC-{index:03d}",
                    "function_name": f"一般功能{index}",
                    "description": "普通功能记录",
                },
            )
            for index in range(20)
        ]
        inventory = [
            _evidence(
                doc_type="function_locator",
                record_id=f"SUB-{index}",
                claim_type="component_fact",
                answer_fields={
                    "component_id": f"SUB-{index}",
                    "component_name": f"第{index}子系统",
                    "component_type": "subsystem",
                    "parent_component_id": "SYS-1",
                    "collection_closed": True,
                    "collection_parent_id": "SYS-1",
                    "collection_parent_name": "测试系统",
                    "collection_member_kind": "subsystem",
                    "collection_member_count": 6,
                    "collection_role": "member",
                    "collection_member_id": f"SUB-{index}",
                    "collection_member_name": f"第{index}子系统",
                },
            )
            for index in range(1, 7)
        ]
        for item in generic:
            item["answer_rerank_score"] = 1.0
        for item in inventory:
            item["answer_rerank_score"] = -1.0
        state = _approved_state(
            *generic,
            *inventory,
            intent="architecture_query",
            targets=["architecture_query"],
        )

        selected = answer_module._select_evidence_window(
            state, [*generic, *inventory], limit=10
        )

        selected_ids = {item["record_id"] for item in selected}
        self.assertTrue({f"SUB-{index}" for index in range(1, 7)} <= selected_ids)

    def test_unstructured_or_unapproved_content_cannot_create_completion_material(self):
        unstructured = _evidence(
            doc_type="mfl",
            record_id="MFL-RAW",
            claim_type="maintenance_locator",
            answer_fields={},
            content="检查某模块并调用 SecretApi；这段没有获准 answer_fields。",
        )
        unapproved = _evidence(
            doc_type="mfl",
            record_id="MFL-BLOCKED",
            claim_type="maintenance_locator",
            answer_fields={
                "maintenance_action": "执行不应获准的动作",
                "api_name": "BlockedApi",
            },
        )
        state = _approved_state(
            unstructured,
            intent="maintenance_query",
            targets=["maintenance_query"],
        )
        state["evidence_registry"].append(unapproved)
        state["context_evidence_ids"].append(unapproved["evidence_id"])
        state["safety_decision"]["context_evidence_ids"].append(
            unapproved["evidence_id"]
        )

        missing, manifest = answer_module._target_completion_manifest("当前没有处理信息。", state)

        self.assertEqual([], missing)
        self.assertEqual([], manifest)
        with patch.object(
            answer_module.llm_service,
            "_call_model",
            side_effect=AssertionError("无结构化获准字段时不应调用补全模型"),
        ):
            result = _run(
                answer_module._review_answer_target_completeness(
                    "当前没有处理信息。", state, None
                )
            )
        self.assertEqual("当前没有处理信息。", result)

    def test_bounded_editor_receives_only_answer_fields_and_can_fill_missing_targets(self):
        mfl = _evidence(
            doc_type="mfl",
            record_id="MFL-SYN-041",
            claim_type="maintenance_locator",
            answer_fields={
                "module_failure": "协议转换模块故障",
                "maintenance_action": "检查转换队列和配置",
                "api_name": "GetFrameStatus",
                "api_method": "读取转换队列状态",
            },
            content="原始正文中有不应传给补全器的 OTHER_INTERNAL_VALUE",
        )
        state = _approved_state(
            mfl,
            intent="maintenance_query",
            targets=["maintenance_query"],
        )
        draft = "建议检查转换队列和配置。[^1]"
        complete = (
            "维修定位为协议转换模块故障；建议检查转换队列和配置。[^1]\n\n"
            "可调用 `GetFrameStatus` 读取转换队列状态。[^1]"
        )

        with patch.object(answer_module.llm_service, "_call_model", return_value=complete) as call:
            result = _run(
                answer_module._review_answer_target_completeness(draft, state, None)
            )

        self.assertEqual(complete, result)
        self.assertLessEqual(call.call_count, 2)
        prompt = call.call_args.args[0][0]["content"]
        self.assertIn("GetFrameStatus", prompt)
        self.assertIn("协议转换模块故障", prompt)
        self.assertNotIn("OTHER_INTERNAL_VALUE", prompt)

    def test_preflight_lists_every_supported_explicit_target(self):
        hmc = _evidence(
            doc_type="hmc",
            record_id="91810001**0001",
            claim_type="hmc_fault",
            answer_fields={
                "hmc_code": "91810001**0001",
                "fault_symptom": "ICU 输入过压钳位失效",
                "maintenance_action": "检查输入采样与钳位器件",
            },
        )
        function = _evidence(
            doc_type="function_locator",
            record_id="FUNC-330",
            claim_type="function_fact",
            answer_fields={
                "module_name": "IFC",
                "function_name": "外部总线适配",
            },
        )
        state = _approved_state(
            hmc,
            function,
            intent="architecture_query",
            targets=["architecture_query", "exact_code"],
        )

        instruction = answer_module._target_preflight_instruction(state)

        self.assertIn("HMC代码/组成", instruction)
        self.assertIn("HMC故障现象", instruction)
        self.assertIn("文档处置", instruction)
        self.assertIn("架构层级", instruction)
        self.assertIn("功能/职责", instruction)

    def test_focused_finalizer_rewrites_missing_scalar_from_json_contract(self):
        first = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-002",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "输入滤波性能退化",
                "severity": "III",
            },
        )
        second = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-003",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "欠压检测误告警",
                "severity": "III",
            },
        )
        state = _with_required_fields(
            _approved_state(
                first,
                second,
                intent="impact_analysis",
                targets=["impact_analysis"],
                query="只说当前候选的严酷度",
            ),
            "severity",
        )
        missing, manifest = answer_module._target_completion_manifest(
            "当前候选需要继续核对。", state
        )
        organized = (
            "- **欠压检测误告警、输入滤波性能退化**的严酷度均为 "
            "**III**。[^2][^1]"
        )

        with patch.object(
            answer_module.llm_service,
            "_call_model",
            return_value=json.dumps(
                {
                    "ordered_citation_ids": [2, 1],
                    "answer": "严酷度 III，必须立即更换输入模块。[^1]",
                },
                ensure_ascii=False,
            ),
        ) as call:
            result = _run(
                answer_module._finalize_focused_target_answer(
                    "当前候选需要继续核对。",
                    state,
                    None,
                    missing,
                    manifest,
                )
            )

        self.assertEqual(
            "当前候选需要继续核对。\n\n" + organized,
            result,
        )
        self.assertNotIn("更换输入模块", result)
        self.assertEqual([], answer_module._target_completion_manifest(result, state)[0])
        prompt = call.call_args.args[0][0]["content"]
        self.assertIn("FM-SYN-002（输入滤波性能退化）", prompt)
        self.assertIn("FM-SYN-003（欠压检测误告警）", prompt)

    def test_focused_finalizer_selects_narrative_subset_and_preserves_rbd(self):
        matching = _evidence(
            doc_type="fmeca",
            record_id="FM-PWR-A",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "A路电源损坏",
                "local_impact": "A路供电能力丧失",
                "final_impact": "由冗余供电路径维持关键功能",
            },
        )
        unrelated = _evidence(
            doc_type="fmeca",
            record_id="FM-PWR-OTHER",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "输入滤波性能退化",
                "local_impact": "输入纹波增大",
                "final_impact": "接口数据可能间歇异常",
            },
        )
        rbd = _evidence(
            doc_type="rbd",
            record_id="RBD-PWR-001",
            claim_type="rbd_structure",
            answer_fields={
                "success_criterion": "A/B两路电源中至少一路正常即可维持关键功能",
            },
            source_kind="hugegraph",
            authority="validated_relation",
        )
        state = _with_required_fields(
            _approved_state(
                matching,
                unrelated,
                rbd,
                intent="impact_analysis",
                targets=["impact_analysis", "reliability_query"],
                query="若A路电源损坏，会有什么影响？",
            ),
            "local_impact",
            "final_impact",
        )
        draft = "RBD 成功判据为 A/B两路电源中至少一路正常即可维持关键功能。[^3]"
        missing, manifest = answer_module._target_completion_manifest(draft, state)

        with patch.object(
            answer_module.llm_service,
            "_call_model",
            return_value=json.dumps({"ordered_citation_ids": [1]}),
        ):
            result = _run(
                answer_module._finalize_focused_target_answer(
                    draft, state, None, missing, manifest
                )
            )

        self.assertIn(draft, result)
        self.assertIn("A路供电能力丧失", result)
        self.assertIn("由冗余供电路径维持关键功能", result)
        self.assertIn("[^1]", result)
        self.assertEqual(1, result.count("- **A路电源损坏**："))
        self.assertNotIn("输入纹波增大", result)
        self.assertNotIn("接口数据可能间歇异常", result)
        self.assertEqual([], answer_module._target_completion_manifest(result, state)[0])

    def test_focused_renderer_merges_narrative_fields_and_deduplicates_values(self):
        rendered = answer_module._render_focused_target_manifest(
            [
                {
                    "citation_id": 1,
                    "record_id": "FM-PWR-A",
                    "record_name": "A路电源损坏",
                    "supports": ["local_impact", "final_impact"],
                    "answer_fields": {
                        "local_impact": "A路供电能力丧失。",
                        "final_impact": "关键功能转由冗余路径维持。",
                    },
                },
                {
                    "citation_id": 2,
                    "record_id": "RBD-PWR-001",
                    "record_name": "双路供电成功判据",
                    "supports": [
                        "rbd_success_criterion",
                        "rbd_single_failure_boundary",
                    ],
                    "answer_fields": {
                        "success_criterion": "A/B两路电源中至少一路正常即可维持关键功能。",
                        "single_failure_boundary": "A/B两路电源中至少一路正常即可维持关键功能。",
                    },
                },
            ],
            [
                "local_impact",
                "final_impact",
                "rbd_success_criterion",
                "rbd_single_failure_boundary",
            ],
            [1, 2],
        )

        self.assertEqual(1, rendered.count("- **A路电源损坏**："))
        self.assertIn("局部影响：A路供电能力丧失", rendered)
        self.assertIn("最终影响：关键功能转由冗余路径维持", rendered)
        self.assertEqual(
            1,
            rendered.count("A/B两路电源中至少一路正常即可维持关键功能"),
        )
        self.assertEqual(1, rendered.count("- **双路供电成功判据**："))
        self.assertNotIn("。。", rendered)
        self.assertNotIn("。；", rendered)

    def test_focused_finalizer_keeps_scalar_all_record_requirement(self):
        first = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-002",
            claim_type="fmeca_failure_mode",
            answer_fields={"failure_mode": "输入滤波性能退化", "severity": "II"},
        )
        second = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-003",
            claim_type="fmeca_failure_mode",
            answer_fields={"failure_mode": "储能器件内阻升高", "severity": "III"},
        )
        state = _with_required_fields(
            _approved_state(
                first,
                second,
                intent="impact_analysis",
                targets=["impact_analysis"],
                query="这些候选分别是什么级别？",
            ),
            "severity",
        )
        draft = "当前候选需要继续核对。"
        missing, manifest = answer_module._target_completion_manifest(draft, state)

        with patch.object(
            answer_module.llm_service,
            "_call_model",
            return_value=json.dumps({"ordered_citation_ids": [1]}),
        ):
            result = _run(
                answer_module._finalize_focused_target_answer(
                    draft, state, None, missing, manifest
                )
            )

        self.assertEqual(draft, result)

    def test_focused_finalizer_rejects_unknown_or_incomplete_selection(self):
        complete = _evidence(
            doc_type="fmeca",
            record_id="FM-PWR-A",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "A路电源损坏",
                "local_impact": "A路供电能力丧失",
                "final_impact": "关键功能转由冗余路径维持",
            },
        )
        complete_state = _with_required_fields(
            _approved_state(
                complete,
                intent="impact_analysis",
                targets=["impact_analysis"],
                query="A路损坏有什么影响？",
            ),
            "local_impact",
            "final_impact",
        )
        draft = "当前影响尚未整理。"
        missing, manifest = answer_module._target_completion_manifest(
            draft, complete_state
        )
        with patch.object(
            answer_module.llm_service,
            "_call_model",
            return_value=json.dumps({"ordered_citation_ids": [99]}),
        ):
            unknown_result = _run(
                answer_module._finalize_focused_target_answer(
                    draft, complete_state, None, missing, manifest
                )
            )
        self.assertEqual(draft, unknown_result)

        local_only = _evidence(
            doc_type="fmeca",
            record_id="FM-PWR-LOCAL",
            claim_type="fmeca_failure_mode",
            answer_fields={"local_impact": "A路供电能力丧失"},
        )
        final_only = _evidence(
            doc_type="fmeca",
            record_id="FM-PWR-FINAL",
            claim_type="fmeca_failure_mode",
            answer_fields={"final_impact": "关键功能转由冗余路径维持"},
        )
        split_state = _with_required_fields(
            _approved_state(
                local_only,
                final_only,
                intent="impact_analysis",
                targets=["impact_analysis"],
                query="A路损坏有什么影响？",
            ),
            "local_impact",
            "final_impact",
        )
        split_missing, split_manifest = answer_module._target_completion_manifest(
            draft, split_state
        )
        with patch.object(
            answer_module.llm_service,
            "_call_model",
            return_value=json.dumps({"ordered_citation_ids": [1]}),
        ):
            incomplete_result = _run(
                answer_module._finalize_focused_target_answer(
                    draft, split_state, None, split_missing, split_manifest
                )
            )
        self.assertEqual(draft, incomplete_result)

    def test_focused_finalizer_uses_single_record_fallback_on_model_failure(self):
        fmeca = _evidence(
            doc_type="fmeca",
            record_id="FM-PWR-A",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "A路电源损坏",
                "local_impact": "A路供电能力丧失",
                "final_impact": "关键功能转由冗余路径维持",
            },
        )
        state = _with_required_fields(
            _approved_state(
                fmeca,
                intent="impact_analysis",
                targets=["impact_analysis"],
                query="A路损坏有什么影响？",
            ),
            "local_impact",
            "final_impact",
        )
        draft = "RBD 资料明确写明任一路正常即可维持功能。[^2]"
        missing, manifest = answer_module._target_completion_manifest(draft, state)

        for model_result in ("不是 JSON", asyncio.TimeoutError()):
            with self.subTest(model_result=type(model_result).__name__):
                kwargs = (
                    {"side_effect": model_result}
                    if isinstance(model_result, BaseException)
                    else {"return_value": model_result}
                )
                with patch.object(
                    answer_module.llm_service,
                    "_call_model",
                    **kwargs,
                ):
                    result = _run(
                        answer_module._finalize_focused_target_answer(
                            draft, state, None, missing, manifest
                        )
                    )

                self.assertIn(draft, result)
                self.assertIn("A路供电能力丧失", result)
                self.assertIn("关键功能转由冗余路径维持", result)
                self.assertEqual(
                    [], answer_module._target_completion_manifest(result, state)[0]
                )

    def test_focused_record_identity_removes_unrequested_history_details(self):
        history = _evidence(
            doc_type="fault_history",
            record_id="CASE-SYN-017",
            claim_type="historical_case",
            answer_fields={
                "case_id": "CASE-SYN-017",
                "title": "输入调理模块故障案例",
                "fault_symptom": "上电后输入电压反复波动",
                "occurred_at": "2025-01-12 09:20",
                "root_cause": "输入钳位器件性能退化",
                "steps": ["检查输入采样", "更换输入调理模块"],
                "final_solution": "更换输入调理模块并复测",
                "verification_result": "连续运行 72 小时正常",
            },
        )
        state = _with_required_fields(
            _approved_state(
                history,
                intent="history_query",
                targets=["history_query"],
                query="CASE-SYN-017 是什么记录？",
            ),
            "record_identity",
        )
        draft = (
            "`CASE-SYN-017` 是输入调理模块故障案例，记录的故障现象是"
            "上电后输入电压反复波动。[^1]\n"
            "发生时间为 2025-01-12 09:20，根因是输入钳位器件性能退化；"
            "处理时先检查输入采样，再更换输入调理模块，最终连续运行 72 小时正常。[^1]"
        )

        result = answer_module._constrain_focused_answer_fields(draft, state)

        self.assertIn("CASE-SYN-017", result)
        self.assertIn("输入调理模块故障案例", result)
        self.assertIn("上电后输入电压反复波动", result)
        for unrequested in (
            "2025-01-12 09:20",
            "输入钳位器件性能退化",
            "检查输入采样",
            "更换输入调理模块并复测",
            "连续运行 72 小时正常",
        ):
            self.assertNotIn(unrequested, result)

    def test_focused_history_solution_removes_other_history_fields(self):
        history = _evidence(
            doc_type="fault_history",
            record_id="CASE-SYN-017",
            claim_type="historical_case",
            answer_fields={
                "case_id": "CASE-SYN-017",
                "title": "输入调理模块故障案例",
                "occurred_at": "2025-01-12 09:20",
                "root_cause": "输入钳位器件性能退化",
                "steps": ["检查输入采样", "更换输入调理模块"],
                "final_solution": "更换输入调理模块并复测",
                "verification_result": "连续运行 72 小时正常",
            },
        )
        state = _with_required_fields(
            _approved_state(
                history,
                intent="history_query",
                targets=["history_query"],
                query="这个历史案例最后怎么解决？",
            ),
            "history_solution",
        )
        draft = (
            "CASE-SYN-017 发生于 2025-01-12 09:20，根因是输入钳位器件性能退化，"
            "处理步骤包括检查输入采样和更换输入调理模块；"
            "最终方案是更换输入调理模块并复测，连续运行 72 小时正常。[^1]"
        )

        result = answer_module._constrain_focused_answer_fields(draft, state)

        self.assertIn("输入调理模块故障案例", result)
        self.assertIn("更换输入调理模块并复测", result)
        for unrequested in (
            "2025-01-12 09:20",
            "输入钳位器件性能退化",
            "检查输入采样",
            "连续运行 72 小时正常",
        ):
            self.assertNotIn(unrequested, result)

    def test_focused_severity_removes_other_fmeca_fields(self):
        fmeca = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-002",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "输入滤波性能退化",
                "fault_cause": "滤波电容老化",
                "local_impact": "输入纹波增大",
                "final_impact": "系统间歇复位",
                "severity": "III",
                "detection_method": "读取输入电压状态字",
                "compensation_action": "限制高负载运行",
            },
        )
        state = _with_required_fields(
            _approved_state(
                fmeca,
                intent="impact_analysis",
                targets=["impact_analysis"],
                query="该异常是什么级别？",
            ),
            "severity",
        )
        draft = (
            "输入滤波性能退化的原因是滤波电容老化，局部影响为输入纹波增大，"
            "最终可能导致系统间歇复位，严酷度为 III；检测时读取输入电压状态字，"
            "并限制高负载运行。[^1]"
        )

        result = answer_module._constrain_focused_answer_fields(draft, state)

        self.assertIn("输入滤波性能退化", result)
        self.assertIn("严酷度", result)
        self.assertIn("III", result)
        for unrequested in (
            "滤波电容老化",
            "输入纹波增大",
            "系统间歇复位",
            "读取输入电压状态字",
            "限制高负载运行",
        ):
            self.assertNotIn(unrequested, result)

    def test_focused_constraint_preserves_already_scoped_draft_verbatim(self):
        fmeca = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-002",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "输入滤波性能退化",
                "fault_cause": "滤波电容老化",
                "local_impact": "输入纹波增大",
                "severity": "III",
            },
        )
        state = _with_required_fields(
            _approved_state(
                fmeca,
                intent="impact_analysis",
                targets=["impact_analysis"],
                query="只说严酷度",
            ),
            "severity",
        )
        draft = "输入滤波性能退化的严酷度为 **III**。[^1]"

        result = answer_module._constrain_focused_answer_fields(draft, state)

        self.assertEqual(draft, result)

    def test_executed_fmeca_outside_plan_still_creates_cause_obligation(self):
        """adaptive_tools 补执行的 fmeca_query 产生已批准记录时，fault_cause
        仍是答案义务，即使该工具不在最小 tools_plan 里。"""
        fmeca_row = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-031",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "A通道接收超时",
                "fault_cause": "接收器件失效、线路异常或时序配置错误。",
            },
        )
        state = _approved_state(
            fmeca_row,
            intent="cause_analysis",
            targets=["cause_analysis"],
            query="内部报文已生成但外部设备拿不到数据，按可能断开的环节说明原因",
        )
        state["query_frame"] = {
            "coverage_scope": "relation_path",
            "required_answer_fields": ["fault_cause", "structural_cause_chain"],
            "diagnostic_closure": {"required": True},
        }
        state["tools_plan"] = [
            {"tool": "function_locator"},
            {"tool": "fta_query"},
            {"tool": "hybrid_search"},
        ]
        state["tool_results"] = [
            {
                "tool": "fmeca_query",
                "status": "ok",
                "evidence_ids": [fmeca_row["evidence_id"]],
            }
        ]
        self.assertFalse(answer_module.focused_answer_contract(state))

        missing, manifest = answer_module._target_completion_manifest(
            "当前只描述了FTA结构链，未按环节给出对应原因。", state
        )

        self.assertIn("fault_cause", missing)
        fm_items = [item for item in manifest if item.get("doc_type") == "fmeca"]
        self.assertTrue(fm_items)
        self.assertTrue(
            all(
                "fault_cause" in set(item.get("supports") or [])
                for item in fm_items
            )
        )

    def test_executed_fmeca_without_approved_evidence_adds_no_obligation(self):
        """空跑/未批准的工具不得为对应文档族制造字段义务。"""
        fmeca_row = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-031",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "failure_mode": "A通道接收超时",
                "fault_cause": "接收器件失效、线路异常或时序配置错误。",
            },
        )
        state = _approved_state(
            fmeca_row,
            intent="cause_analysis",
            targets=["cause_analysis"],
            query="内部报文已生成但外部设备拿不到数据，按可能断开的环节说明原因",
        )
        state["query_frame"] = {
            "coverage_scope": "relation_path",
            "required_answer_fields": ["fault_cause", "structural_cause_chain"],
            "diagnostic_closure": {"required": True},
        }
        state["tools_plan"] = [{"tool": "fta_query"}]
        state["tool_results"] = [
            {
                "tool": "fmeca_query",
                "status": "ok",
                "evidence_ids": ["ev-not-approved"],
            }
        ]

        missing, manifest = answer_module._target_completion_manifest("", state)

        # 没有已批准 FMECA 证据：义务集合不因空跑工具扩张，清单保持为空。
        self.assertEqual(([], []), (missing, manifest))

    def test_filter_keeps_fmeca_leaf_of_approved_fta_tree(self):
        """已批准 FTA 树的叶子故障模式即使没有对象 ID 交集也不被 fmeca_ids
        边界检查删除；未关联的 FMECA 行仍按原规则过滤。"""
        fta_item = _evidence(
            doc_type="fta",
            record_id="FTA-SYN-003",
            claim_type="fta_structure",
            answer_fields={
                "basic_events": [
                    "{'name': 'A通道接收超时', 'description': '接收器件失效。'}"
                ]
            },
            source_kind="hugegraph",
            authority="validated_relation",
            content="FTA 结构：系统无法与外部设备交换有效业务数据",
        )
        leaf_fmeca = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-031",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "product_flag": "BCA",
                "failure_mode": "A通道接收超时",
                "fault_cause": "接收器件失效、线路异常或时序配置错误。",
            },
        )
        unrelated_fmeca = _evidence(
            doc_type="fmeca",
            record_id="FM-SYN-999",
            claim_type="fmeca_failure_mode",
            answer_fields={
                "product_flag": "OTHER",
                "failure_mode": "温度采集满量程",
                "fault_cause": "采集前端损坏。",
            },
        )
        state = _approved_state(
            fta_item,
            leaf_fmeca,
            unrelated_fmeca,
            intent="cause_analysis",
            targets=["cause_analysis"],
            query="外部设备拿不到数据，按可能断开的环节说明原因",
        )
        state["topology_dependency"] = {
            "required": True,
            "subject": "外部数据链路",
            "subject_kind": "data_path",
            "failure_state": "失效",
            "canonical_business_ids": ["FUNC-999"],
        }

        retained = answer_module._filter_answer_domain_evidence(
            state, [fta_item, leaf_fmeca, unrelated_fmeca]
        )
        retained_ids = {item["evidence_id"] for item in retained}

        self.assertIn(leaf_fmeca["evidence_id"], retained_ids)
        self.assertNotIn(unrelated_fmeca["evidence_id"], retained_ids)


if __name__ == "__main__":
    unittest.main()
