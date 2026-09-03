import unittest

from backend.agent.tool_policy import (
    detect_calculation_requests,
    extract_hmc_components,
    resolve_tool_policy,
)
from backend.agent.tools.icp_calculation import _mission_time


class ToolPolicyTests(unittest.TestCase):
    def test_business_id_prefix_is_not_mistaken_for_a_named_source_request(self):
        decision = resolve_tool_policy(
            intent="entity_profile",
            query="MFL-SYN-041 是什么",
            raw_user_query="MFL-SYN-041 是什么",
            query_targets=[
                {"intent": "entity_profile", "required_fields": ["record_identity"]}
            ],
            required_answer_fields=["record_identity"],
            coverage_scope="focused_detail",
        )

        self.assertEqual([], decision["requested_source_tools"])
        self.assertEqual([], decision["source_coverage_fields"])

    def test_complete_cause_collection_uses_architecture_and_fmeca_truth(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="这个子系统有哪些原因？",
            proposed_plan=[],
            coverage_scope="complete_collection",
            collection_member_kind="cause",
        )

        self.assertIn("function_locator", decision["allowed_tools"])
        self.assertIn("fmeca_query", decision["allowed_tools"])
        self.assertNotIn("fta_analyzer", decision["allowed_tools"])

    def test_complete_failure_mode_collection_keeps_membership_truth_separate_from_context(self):
        decision = resolve_tool_policy(
            intent="impact_analysis",
            query="任意完整集合问法",
            proposed_plan=[
                {"tool": "fmeca_query"},
                {"tool": "hmc_query"},
                {"tool": "mfl_query"},
                {"tool": "fault_history_query"},
            ],
            coverage_scope="complete_collection",
            collection_member_kind="failure_mode",
            query_targets=[
                {"intent": "impact_analysis"},
                {"intent": "maintenance_query", "required_fields": ["maintenance_action"]},
            ],
            required_answer_fields=["closed_collection_members", "maintenance_action"],
        )

        # Function Locator + FMECA are forced because only their PostgreSQL
        # records can prove the collection is closed.  The explicitly asked
        # maintenance dimension must not be discarded merely because it shares
        # a turn with that closed-set contract.
        self.assertTrue(
            {"function_locator", "fmeca_query"}.issubset(
                set(decision["allowed_tools"])
            )
        )
        self.assertTrue(
            {"hmc_query", "mfl_query"}.issubset(set(decision["allowed_tools"]))
        )

    def test_relation_contract_keeps_fta_locator_and_graph_when_planner_starts_with_fmeca(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="外发数据为什么会中断",
            query_targets=[
                {"intent": "cause_analysis", "required_fields": ["fault_cause"]}
            ],
            required_answer_fields=["fault_cause"],
            coverage_scope="focused_detail",
            relation_expansion_required=True,
            proposed_plan=[{"tool": "fmeca_query", "reason": "直接原因"}],
        )

        self.assertTrue(
            {"fmeca_query", "function_locator", "fta_query", "graph_traversal"}
            .issubset(set(decision["allowed_tools"]))
        )
        self.assertTrue(
            any("语义关系合同" in note for note in decision["notes"])
        )

    def test_architecture_policy_rejects_fault_and_unknown_tools(self):
        decision = resolve_tool_policy(
            intent="architecture_query",
            query="系统由哪些模块组成？",
            proposed_plan=[
                {"tool": "fta_query", "reason": "误规划"},
                {"tool": "drop_database", "reason": "未知工具"},
                {"tool": "fault_reasoning", "reason": "兼容别名"},
            ],
        )
        self.assertEqual(
            ["function_locator", "hybrid_search", "evidence_synthesis"],
            decision["allowed_tools"],
        )
        self.assertEqual(["function_locator"], decision["hybrid_doc_types"])
        self.assertEqual(["fta_query", "drop_database"], decision["rejected_tools"])

    def test_entity_profile_can_resolve_non_architecture_business_record(self):
        decision = resolve_tool_policy(
            intent="entity_profile",
            query="这个业务记录是什么",
            query_targets=[
                {"intent": "entity_profile", "required_fields": ["record_identity"]}
            ],
            required_answer_fields=["record_identity"],
            coverage_scope="focused_detail",
            proposed_plan=[{"tool": "fmeca_query"}],
        )

        self.assertEqual(
            ["fmeca_query", "evidence_synthesis"], decision["allowed_tools"]
        )
        self.assertIn("mfl_query", decision["authorized_tools"])
        self.assertIn("fault_history_query", decision["authorized_tools"])

    def test_semantic_fault_location_keeps_mfl_without_internal_tool_wording(self):
        decision = resolve_tool_policy(
            intent="function_location",
            query="这个问题具体会落在哪块板、哪个功能环节？",
            query_targets=[
                {
                    "intent": "function_location",
                    "required_fields": ["fault_location", "maintenance_locator"],
                }
            ],
            required_answer_fields=["fault_location", "maintenance_locator"],
            coverage_scope="focused_detail",
            proposed_plan=[{"tool": "function_locator"}],
        )

        # ``mfl_query`` comes from the semantic field owner, not from a raw
        # phrase such as “MFL” or “维修清单” in the user question. The first
        # function lookup stays minimal; adaptive completion may invoke MFL if
        # that lookup cannot already prove the requested maintenance level.
        self.assertIn("function_locator", decision["allowed_tools"])
        self.assertIn("mfl_query", decision["authorized_tools"])

    def test_current_incident_closure_does_not_reduce_cause_to_one_table(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="运行时反复重启，可能是什么原因？",
            query_targets=[
                {"intent": "cause_analysis", "required_fields": ["fault_cause"]}
            ],
            required_answer_fields=["fault_cause"],
            coverage_scope="focused_detail",
            diagnostic_closure={"required": True, "reason": "当前现场异常需要排查闭环"},
            proposed_plan=[{"tool": "fmeca_query"}],
        )

        # The first query remains minimal, but the adaptive stage keeps the
        # documented handling/history sources available for closure.
        self.assertEqual(
            ["fmeca_query", "evidence_synthesis"], decision["allowed_tools"]
        )
        self.assertTrue(
            {"mfl_query", "hmc_query", "fault_history_query"}.issubset(
                set(decision["authorized_tools"])
            )
        )

    def test_explicit_business_id_forces_prefix_agnostic_identity_probes(self):
        expected_probes = {
            "function_locator",
            "hmc_query",
            "fmeca_query",
            "mfl_query",
            "fault_history_query",
        }

        for business_id in (
            "FM-SYN-002",
            "MFL-SYN-041",
            "HIS-SYN-020",
            "MOD-PWR-PSUA",
            "XX-UNCLASSIFIED-9",
        ):
            with self.subTest(business_id=business_id):
                decision = resolve_tool_policy(
                    intent="entity_profile",
                    query=f"{business_id} 是什么",
                    raw_user_query=f"{business_id} 是什么",
                    query_targets=[
                        {
                            "intent": "entity_profile",
                            "required_fields": ["record_identity"],
                        }
                    ],
                    required_answer_fields=["record_identity"],
                    coverage_scope="focused_detail",
                    # Reproduce the failure mode: the planner assumes every
                    # code is an architecture object.
                    proposed_plan=[{"tool": "function_locator"}],
                )

                self.assertTrue(
                    expected_probes.issubset(decision["allowed_tools"])
                )
                self.assertEqual(
                    expected_probes,
                    set(decision["allowed_tools"]) - {"evidence_synthesis"},
                )
                self.assertTrue(
                    any(
                        "不依据前缀" in note
                        for note in decision["notes"]
                    )
                )

    def test_exact_code_requires_postgres_hmc_query(self):
        decision = resolve_tool_policy(
            intent="exact_code",
            query="查询 31810001-**-0001",
            proposed_plan=[],
        )
        self.assertEqual(
            ["hmc_query", "evidence_synthesis"],
            decision["allowed_tools"],
        )
        self.assertEqual([], decision["hybrid_doc_types"])

    def test_labelled_hmc_triplet_forces_exact_policy(self):
        query = "查询LCN=31810001，HAMC=**，HSC=0001"
        self.assertEqual(
            {
                "lcn": "31810001",
                "hamc": "**",
                "hsc": "0001",
                "hmc_code": "31810001**0001",
            },
            extract_hmc_components(query),
        )
        decision = resolve_tool_policy(
            intent="symptom_diagnosis",
            query=query,
            proposed_plan=[{"tool": "hybrid_search"}],
        )
        self.assertEqual("exact_code", decision["intent"])
        self.assertIn("hmc_query", decision["allowed_tools"])

    def test_complete_hmc_defers_model_selected_hybrid_search_until_exact_miss(self):
        decision = resolve_tool_policy(
            intent="exact_code",
            query="查询 HMC 91810001-**-0001 的故障现象",
            proposed_plan=[
                {"tool": "hmc_query", "reason": "先查权威记录"},
                {"tool": "hybrid_search", "reason": "补充原文"},
            ],
        )
        self.assertEqual(
            ["hmc_query", "evidence_synthesis"],
            decision["allowed_tools"],
        )
        self.assertIn("hybrid_search", decision["authorized_tools"])
        self.assertEqual([], decision["hybrid_doc_types"])
        self.assertTrue(any("先执行权威精确查询" in note for note in decision["notes"]))

    def test_labelled_lcn_hsc_without_hamc_is_not_misparsed_as_compact_code(self):
        query = "LCN 91810001 HSC 0001 故障如何修复"
        self.assertEqual(
            {"lcn": "91810001", "hsc": "0001"},
            extract_hmc_components(query),
        )
        decision = resolve_tool_policy(
            intent="maintenance_query",
            query=query,
            proposed_plan=[],
        )
        self.assertEqual("maintenance_query", decision["intent"])
        self.assertIn("exact_code", [item["intent"] for item in decision["query_targets"]])
        self.assertIn("hmc_query", decision["allowed_tools"])

    def test_hmc_labels_accept_colons_equals_and_fullwidth_code_strings(self):
        self.assertEqual(
            {
                "lcn": "00123456",
                "hamc": "**",
                "hsc": "0001",
                "hmc_code": "00123456**0001",
            },
            extract_hmc_components("LCN：００１２３４５６ HAMC=＊＊ HSC 1"),
        )

    def test_incomplete_labelled_input_does_not_fall_through_to_compact_parser(self):
        self.assertEqual({}, extract_hmc_components("LCN 91810001 HAMC"))

    def test_compact_complete_hmc_still_parses(self):
        self.assertEqual(
            {
                "lcn": "31810001",
                "hamc": "**",
                "hsc": "0001",
                "hmc_code": "31810001**0001",
            },
            extract_hmc_components("查询 31810001-**-0001"),
        )

    def test_symptom_policy_limits_hybrid_search_to_selected_types(self):
        decision = resolve_tool_policy(
            intent="symptom_diagnosis",
            query="运行中反复重启",
            proposed_plan=[{"tool": "mfl_query"}, {"tool": "rbd_query"}],
        )
        self.assertIn("mfl_query", decision["allowed_tools"])
        self.assertNotIn("rbd_query", decision["allowed_tools"])
        self.assertNotIn("hybrid_search", decision["allowed_tools"])
        self.assertEqual([], decision["hybrid_doc_types"])

    def test_model_selected_query_is_not_expanded_to_fixed_intent_bundle(self):
        decision = resolve_tool_policy(
            intent="impact_analysis",
            query="滤波电容老化会有什么影响？",
            proposed_plan=[{"tool": "fmeca_query", "reason": "查询影响记录"}],
        )
        self.assertEqual(
            ["fmeca_query", "evidence_synthesis"],
            decision["allowed_tools"],
        )
        self.assertIn("hybrid_search", decision["authorized_tools"])
        self.assertIn("graph_traversal", decision["authorized_tools"])
        self.assertIn("fta_query", decision["authorized_tools"])
        self.assertIn("rbd_query", decision["authorized_tools"])

    def test_maintenance_policy_keeps_fmeca_authorized_for_adaptive_completion(self):
        decision = resolve_tool_policy(
            intent="maintenance_query",
            query="这个故障怎么处理？",
            proposed_plan=[{"tool": "mfl_query", "reason": "先查维修动作"}],
        )

        self.assertEqual(
            ["mfl_query", "evidence_synthesis"],
            decision["allowed_tools"],
        )
        self.assertIn("fmeca_query", decision["authorized_tools"])

    def test_impact_planner_can_select_propagation_tools_without_fixed_full_scan(self):
        decision = resolve_tool_policy(
            intent="impact_analysis",
            query="公共时基慢慢漂了，会牵连哪些能力？",
            proposed_plan=[
                {"tool": "fmeca_query", "reason": "查直接影响"},
                {"tool": "fta_query", "reason": "查故障传播结构"},
                {"tool": "function_locator", "reason": "定位被牵连的功能"},
                {"tool": "graph_traversal", "reason": "沿已绑定实体查跨文档路径"},
            ],
        )
        self.assertEqual(
            [
                "function_locator",
                "fmeca_query",
                "fta_query",
                "graph_traversal",
                "evidence_synthesis",
            ],
            decision["allowed_tools"],
        )
        self.assertNotIn("rbd_query", decision["allowed_tools"])
        self.assertNotIn("hybrid_search", decision["allowed_tools"])

    def test_impact_and_reliability_targets_query_rbd_but_do_not_auto_calculate(self):
        decision = resolve_tool_policy(
            intent="impact_analysis",
            query="两路中一路异常、另一路正常，继续运行有什么风险？",
            raw_user_query="两路中一路异常、另一路正常，继续运行有什么风险？",
            query_targets=[
                {"intent": "impact_analysis"},
                {"intent": "reliability_query"},
            ],
            proposed_plan=[
                {"tool": "fmeca_query"},
                {"tool": "rbd_query"},
                {"tool": "rbd_calculator"},
            ],
        )
        self.assertEqual(
            ["impact_analysis", "reliability_query"],
            [item["intent"] for item in decision["query_targets"]],
        )
        self.assertIn("fmeca_query", decision["allowed_tools"])
        self.assertIn("rbd_query", decision["allowed_tools"])
        self.assertNotIn("rbd_calculator", decision["allowed_tools"])
        self.assertIn("rbd_calculator", decision["rejected_tools"])
        self.assertEqual([], decision["calculation_requests"])

    def test_impact_rbd_calculation_still_requires_original_numeric_request(self):
        rewritten_only = resolve_tool_policy(
            intent="impact_analysis",
            query="改写后的检索语句：计算可靠度",
            raw_user_query="两路中一路异常，另一路正常有什么风险？",
            query_targets=[{"intent": "reliability_query"}],
            proposed_plan=[
                {"tool": "rbd_query"},
                {"tool": "rbd_calculator"},
            ],
        )
        self.assertIn("rbd_query", rewritten_only["allowed_tools"])
        self.assertNotIn("rbd_calculator", rewritten_only["allowed_tools"])
        self.assertIn("rbd_calculator", rewritten_only["rejected_tools"])

        explicit = resolve_tool_policy(
            intent="impact_analysis",
            query="检索用改写",
            raw_user_query="一路失效后，请按 10 小时任务计算系统成功概率",
            query_targets=[{"intent": "reliability_query"}],
            proposed_plan=[{"tool": "rbd_query"}],
        )
        self.assertEqual(["rbd_reliability"], explicit["calculation_requests"])
        self.assertIn("rbd_query", explicit["allowed_tools"])
        self.assertIn("rbd_calculator", explicit["allowed_tools"])

    def test_hybrid_only_plan_receives_target_scoped_document_types(self):
        decision = resolve_tool_policy(
            intent="impact_analysis",
            query="滤波电容老化的影响",
            proposed_plan=[{"tool": "hybrid_search"}],
        )
        self.assertEqual(
            ["hybrid_search", "evidence_synthesis"],
            decision["allowed_tools"],
        )
        self.assertEqual(
            ["function_locator", "hmc", "fmeca", "mfl", "fault_history"],
            decision["hybrid_doc_types"],
        )

    def test_cause_policy_requires_fta_without_user_document_terms(self):
        generic = resolve_tool_policy(intent="cause_analysis", query="为什么会黑屏？")
        self.assertIn("fta_query", generic["allowed_tools"])
        self.assertIn("fta", generic["hybrid_doc_types"])
        self.assertNotIn("fta_analyzer", generic["allowed_tools"])
        self.assertNotIn("graph_traversal", generic["allowed_tools"])

    def test_explicit_rbd_source_request_is_not_dropped_from_cause_analysis(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="导致模板与数据存储模块损坏的原因",
            raw_user_query="导致模板与数据存储模块损坏，你去查 RBD。",
            query_targets=[
                {"intent": "cause_analysis", "required_fields": ["fault_cause"]}
            ],
            required_answer_fields=["fault_cause"],
            coverage_scope="focused_detail",
            proposed_plan=[
                {"tool": "function_locator"},
                {"tool": "fmeca_query"},
                {"tool": "fta_query"},
            ],
        )

        # RBD is a user-mandated source-coverage query here, not a fabricated
        # reliability intent and not permission to calculate a probability.
        self.assertIn("rbd_query", decision["allowed_tools"])
        self.assertIn("rbd_query", decision["authorized_tools"])
        self.assertIn("rbd_query", decision["requested_source_tools"])
        self.assertIn("rbd_success_criterion", decision["source_coverage_fields"])
        self.assertIn("rbd_path_structure", decision["source_coverage_fields"])
        self.assertNotIn("rbd_calculator", decision["allowed_tools"])
        rbd_plan = next(
            item for item in decision["tools_plan"] if item["tool"] == "rbd_query"
        )
        self.assertIn("用户明确要求", rbd_plan["reason"])

    def test_planner_selected_graph_is_retained_for_post_retrieval_gate(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="这个故障通过哪些功能关联到模块？",
            proposed_plan=[
                {"tool": "fmeca_query"},
                {"tool": "graph_traversal"},
            ],
        )
        self.assertIn("graph_traversal", decision["allowed_tools"])

    def test_cause_plan_without_initial_graph_still_authorizes_adaptive_expansion(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="系统运行中总是无故反复重启，通常是什么原因？",
            proposed_plan=[{"tool": "fmeca_query"}, {"tool": "fta_query"}],
        )
        self.assertNotIn("graph_traversal", decision["allowed_tools"])
        self.assertIn("graph_traversal", decision["authorized_tools"])

    def test_planner_failure_uses_scoped_fallback_without_graph_or_calculators(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="运行中反复重启",
            proposed_plan=[{"tool": "graph_traversal"}],
            planner_failed=True,
        )
        self.assertEqual(
            ["fmeca_query", "fta_query", "hybrid_search", "evidence_synthesis"],
            decision["allowed_tools"],
        )
        # 规划失败时不应信任失败输出中的图工具建议。
        self.assertNotIn("graph_traversal", decision["allowed_tools"])

    def test_policy_does_not_classify_intent_from_raw_words(self):
        decision = resolve_tool_policy(
            intent="impact_analysis",
            query="这个怎么修，以前有没有类似情况？",
            proposed_plan=[],
        )
        self.assertEqual(["impact_analysis"], [item["intent"] for item in decision["query_targets"]])
        self.assertNotIn("mfl_query", decision["allowed_tools"])
        self.assertNotIn("fault_history_query", decision["allowed_tools"])

    def test_planner_multi_targets_are_merged_without_keyword_rules(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="任意措辞",
            query_targets=[
                {"intent": "cause_analysis"},
                {"intent": "maintenance_query"},
                {"intent": "history_query"},
            ],
        )
        for tool in (
            "hmc_query", "fmeca_query", "fta_query", "mfl_query",
            "fault_history_query", "hybrid_search",
        ):
            self.assertIn(tool, decision["allowed_tools"])

    def test_focused_severity_rejects_planner_selected_repair_and_history_tools(self):
        decision = resolve_tool_policy(
            intent="impact_analysis",
            query="该异常级别",
            query_targets=[
                {"intent": "impact_analysis", "required_fields": ["severity"]}
            ],
            required_answer_fields=["severity"],
            coverage_scope="focused_detail",
            proposed_plan=[
                {"tool": "function_locator", "reason": "不必要地重新定位对象"},
                {"tool": "fmeca_query", "reason": "读取严酷度字段"},
                {"tool": "hybrid_search", "reason": "不必要地同时宽召回"},
                {"tool": "hmc_query", "reason": "错误扩展到 HMC"},
                {"tool": "mfl_query", "reason": "错误扩展到维修"},
                {"tool": "fault_history_query", "reason": "错误扩展到历史"},
            ],
        )

        self.assertEqual(
            ["fmeca_query", "evidence_synthesis"],
            decision["allowed_tools"],
        )
        self.assertEqual(
            [
                "function_locator",
                "hmc_query",
                "mfl_query",
                "fault_history_query",
            ],
            decision["rejected_tools"],
        )
        self.assertNotIn("hybrid_search", decision["allowed_tools"])
        self.assertIn("hybrid_search", decision["authorized_tools"])
        self.assertNotIn("hmc_query", decision["authorized_tools"])
        self.assertNotIn("mfl_query", decision["authorized_tools"])
        self.assertNotIn("fault_history_query", decision["authorized_tools"])

    def test_initial_symptom_diagnosis_is_not_reduced_to_one_field_toolset(self):
        decision = resolve_tool_policy(
            intent="symptom_diagnosis",
            query="设备运行中偶发自动复位",
            query_targets=[
                {"intent": "symptom_diagnosis", "required_fields": ["failure_mode"]}
            ],
            required_answer_fields=["failure_mode"],
            coverage_scope="focused_detail",
            proposed_plan=[{"tool": "fmeca_query"}],
        )

        self.assertIn("hmc_query", decision["authorized_tools"])
        self.assertIn("mfl_query", decision["authorized_tools"])
        self.assertIn("fault_history_query", decision["authorized_tools"])

    def test_duplicate_query_targets_merge_required_fields_without_loss(self):
        decision = resolve_tool_policy(
            intent="impact_analysis",
            query="继续追问当前候选",
            query_targets=[
                {"intent": "impact_analysis", "required_fields": ["severity"]},
                {
                    "intent": "impact_analysis",
                    "required_fields": ["failure_probability_level"],
                },
            ],
            required_answer_fields=["final_impact"],
            coverage_scope="focused_detail",
            proposed_plan=[{"tool": "fmeca_query"}],
        )

        self.assertEqual(1, len(decision["query_targets"]))
        self.assertEqual(
            ["final_impact", "severity", "failure_probability_level"],
            decision["query_targets"][0]["required_fields"],
        )

    def test_rewritten_query_cannot_authorize_a_calculator(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="计算顶事件概率并分析原因",
            raw_user_query="分析反复重启的原因",
            proposed_plan=[{"tool": "fta_analyzer"}],
        )
        self.assertNotIn("fta_analyzer", decision["allowed_tools"])
        self.assertIn("fta_analyzer", decision["rejected_tools"])

    def test_calculator_requires_explicit_numeric_request(self):
        narrative = resolve_tool_policy(
            intent="reliability_query",
            query="说明 RBD 的串并联结构",
            proposed_plan=[{"tool": "rbd_calculator"}],
        )
        numeric = resolve_tool_policy(
            intent="reliability_query",
            query="任务时间 10 小时时，计算系统可靠度是多少？",
            proposed_plan=[],
        )
        self.assertNotIn("rbd_calculator", narrative["allowed_tools"])
        self.assertIn("rbd_calculator", narrative["rejected_tools"])
        self.assertIn("rbd_calculator", numeric["allowed_tools"])

    def test_rbd_recompute_paraphrase_authorizes_deterministic_calculator(self):
        query = "按资料里的失效率复算十小时任务的成功概率，并找出最低的一项"
        self.assertEqual(["rbd_reliability"], detect_calculation_requests(query))
        self.assertEqual(10.0, _mission_time(query))
        self.assertEqual(25.0, _mission_time("复算二十五小时任务可靠度"))
        decision = resolve_tool_policy(
            intent="reliability_query",
            query=query,
            proposed_plan=[{"tool": "rbd_query"}],
        )
        self.assertIn("rbd_calculator", decision["allowed_tools"])

    def test_unknown_intent_has_visible_hybrid_fallback(self):
        decision = resolve_tool_policy(intent="invented_intent", query="故障")
        self.assertEqual("hybrid", decision["intent"])
        self.assertTrue(decision["notes"])
        self.assertEqual(
            ["function_locator", "hmc", "fmeca", "fta", "rbd", "mfl", "fault_history"],
            decision["hybrid_doc_types"],
        )


if __name__ == "__main__":
    unittest.main()
