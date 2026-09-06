import unittest

from backend.agent.tool_policy import (
    extract_hmc_components,
    normalize_calculation_authorizations,
    resolve_tool_policy,
)
from backend.agent.tools.icp_calculation import _mission_time


class ToolPolicyTests(unittest.TestCase):
    def test_focused_direct_query_retains_only_same_source_semantic_fallback(self):
        for field, direct, doc_type in (("maintenance_api", "mfl_query", "mfl"), ("local_impact", "fmeca_query", "fmeca")):
            policy = resolve_tool_policy(intent="entity_profile", query="查 MOD-CASE-31 的该维度", raw_user_query="查 MOD-CASE-31 的该维度",
                proposed_plan=[{"tool": direct}], required_answer_fields=[field], coverage_scope="focused_detail")
            self.assertIn(direct, policy["allowed_tools"])
            self.assertNotIn("hybrid_search", policy["allowed_tools"])
            self.assertIn("hybrid_search", policy["authorized_tools"])
            self.assertEqual([doc_type], policy["authorized_doc_types"])

    def test_natural_description_has_same_source_recall_before_answering(self):
        for query in ("外面的接收端收不到内部生成的内容", "Where could the output interruption be?"):
            policy = resolve_tool_policy(intent="maintenance_query", query=query, raw_user_query=query,
                proposed_plan=[{"tool": "mfl_query"}], required_answer_fields=["maintenance_locator", "maintenance_api"])
            self.assertIn("hybrid_search", policy["allowed_tools"])
            self.assertEqual(["mfl"], policy["hybrid_doc_types"])
            self.assertNotIn("hmc_query", policy["allowed_tools"])

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
            query_targets=[{
                "intent": "cause_analysis",
                "required_fields": ["closed_collection_members"],
            }],
            required_answer_fields=["closed_collection_members"],
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

    def test_relation_contract_keeps_explicit_fta_field_locator_and_graph(self):
        query = "外发数据为什么会中断"
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query=query,
            raw_user_query=query,
            query_targets=[
                {
                    "intent": "cause_analysis",
                    "required_fields": ["fault_cause", "structural_cause_chain"],
                }
            ],
            required_answer_fields=["fault_cause", "structural_cause_chain"],
            coverage_scope="focused_detail",
            relation_expansion={
                "needed": True,
                "relation_kind": "structural_cause_chain",
                "reference_mode": "current_query",
                "basis": "外发数据为什么会中断",
                "reason": "需要核对跨文档结构链",
            },
            proposed_plan=[
                {"tool": "fmeca_query", "reason": "直接原因"},
                {"tool": "fta_query", "reason": "结构原因链"},
            ],
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
            query_targets=[{
                "intent": "architecture_query",
                "required_fields": ["architecture_hierarchy"],
            }],
            required_answer_fields=["architecture_hierarchy"],
            coverage_scope="focused_detail",
            proposed_plan=[
                {"tool": "function_locator", "reason": "读取层级"},
                {"tool": "hybrid_search", "reason": "结构化未命中时补查"},
                {"tool": "fta_query", "reason": "误规划"},
                {"tool": "drop_database", "reason": "未知工具"},
                {"tool": "fault_reasoning", "reason": "兼容别名"},
            ],
        )
        self.assertEqual(
            ["function_locator", "hybrid_search", "evidence_synthesis"],
            decision["allowed_tools"],
        )
        self.assertIn("hybrid_search", decision["authorized_tools"])
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
            ["fmeca_query", "hybrid_search", "evidence_synthesis"], decision["allowed_tools"]
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
            diagnostic_closure={
                "required": True,
                "required_dimensions": ["maintenance_action", "history_result"],
                "basis": "运行时反复重启，可能是什么原因",
                "reason": "当前现场异常需要排查闭环",
            },
            proposed_plan=[{"tool": "fmeca_query"}],
        )

        # The first query remains minimal, but the adaptive stage keeps the
        # documented handling/history sources available for closure.
        self.assertEqual(
            ["fmeca_query", "mfl_query", "hybrid_search", "evidence_synthesis"],
            decision["allowed_tools"],
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
        self.assertEqual(
            ["hybrid_search", "evidence_synthesis"], decision["allowed_tools"]
        )
        self.assertIn("mfl_query", decision["rejected_tools"])
        self.assertNotIn("rbd_query", decision["allowed_tools"])
        self.assertEqual(
            [
                "function_locator", "hmc", "fmeca", "fta", "rbd", "mfl",
                "fault_history",
            ],
            decision["hybrid_doc_types"],
        )

    def test_model_selected_source_without_any_execution_contract_uses_generic_fallback(self):
        decision = resolve_tool_policy(
            intent="impact_analysis",
            query="滤波电容老化会有什么影响？",
            proposed_plan=[{"tool": "fmeca_query", "reason": "查询影响记录"}],
        )
        self.assertEqual(
            ["hybrid_search", "evidence_synthesis"],
            decision["allowed_tools"],
        )
        self.assertIn("hybrid_search", decision["authorized_tools"])
        self.assertIn("fmeca_query", decision["rejected_tools"])
        self.assertNotIn("graph_traversal", decision["authorized_tools"])
        self.assertNotIn("fta_query", decision["authorized_tools"])
        self.assertNotIn("rbd_query", decision["authorized_tools"])

    def test_canonical_maintenance_field_authorizes_only_its_owner(self):
        decision = resolve_tool_policy(
            intent="maintenance_query",
            query="这个故障定位到哪个维修条目？",
            query_targets=[{
                "intent": "maintenance_query",
                "required_fields": ["maintenance_locator"],
            }],
            required_answer_fields=["maintenance_locator"],
            coverage_scope="focused_detail",
            proposed_plan=[{"tool": "mfl_query", "reason": "查询维修定位"}],
        )

        self.assertEqual(
            ["mfl_query", "hybrid_search", "evidence_synthesis"],
            decision["allowed_tools"],
        )
        self.assertNotIn("fmeca_query", decision["authorized_tools"])
        self.assertEqual(["mfl"], decision["hybrid_doc_types"])

    def test_no_contract_rejects_multiple_direct_families_and_runs_one_generic_recall(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="一个没有结构合同的模型输出",
            proposed_plan=[
                {"tool": "fmeca_query"},
                {"tool": "fta_query"},
                {"tool": "fault_history_query"},
            ],
        )

        self.assertEqual(
            ["hybrid_search", "evidence_synthesis"], decision["allowed_tools"]
        )
        self.assertEqual(
            {"fmeca_query", "fta_query", "fault_history_query"},
            set(decision["rejected_tools"]),
        )
        self.assertEqual(
            ["function_locator", "hmc", "fmeca", "fta", "rbd", "mfl", "fault_history"],
            decision["hybrid_doc_types"],
        )

    def test_selected_anchor_is_an_execution_contract_for_its_source(self):
        decision = resolve_tool_policy(
            intent="maintenance_query",
            query="它还有哪些已记录状态",
            selected_anchor_ids=["ev-mfl"],
            selected_anchor_doc_types=["mfl"],
            proposed_plan=[{"tool": "mfl_query"}],
        )

        self.assertIn("mfl_query", decision["allowed_tools"])
        self.assertNotIn("hybrid_search", decision["allowed_tools"])

    def test_one_selected_anchor_does_not_unlock_neighbour_source_proposals(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="继续核对它",
            selected_anchor_ids=["ev-rbd"],
            selected_anchor_doc_types=["rbd"],
            proposed_plan=[
                {"tool": "rbd_query"},
                {"tool": "fmeca_query"},
                {"tool": "fault_history_query"},
            ],
        )

        self.assertEqual(
            ["rbd_query", "evidence_synthesis"], decision["allowed_tools"]
        )
        self.assertEqual(
            {"fmeca_query", "fault_history_query"},
            set(decision["rejected_tools"]),
        )

    def test_impact_planner_can_select_propagation_tools_without_fixed_full_scan(self):
        query = "公共时基慢慢漂了，会牵连哪些能力？"
        decision = resolve_tool_policy(
            intent="impact_analysis",
            query=query,
            raw_user_query=query,
            relation_expansion={
                "needed": True,
                "relation_kind": "impact_propagation",
                "reference_mode": "current_query",
                "basis": "会牵连哪些能力",
            },
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
                {
                    "intent": "impact_analysis",
                    "required_fields": ["local_impact"],
                },
                {
                    "intent": "reliability_query",
                    "required_fields": ["rbd_single_failure_boundary"],
                },
            ],
            required_answer_fields=["local_impact", "rbd_single_failure_boundary"],
            coverage_scope="focused_detail",
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
            query_targets=[{
                "intent": "reliability_query",
                "required_fields": ["rbd_success_criterion"],
            }],
            required_answer_fields=["rbd_success_criterion"],
            coverage_scope="focused_detail",
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
            calculation_authorizations=[
                {
                    "operation": "rbd_reliability",
                    "mode": "calculate",
                    "target": "rbd",
                    "basis": "按 10 小时任务计算系统成功概率",
                }
            ],
            query_targets=[{"intent": "reliability_query"}],
            proposed_plan=[{"tool": "rbd_query"}],
        )
        self.assertEqual(["rbd_reliability"], explicit["calculation_requests"])
        self.assertIn("rbd_query", explicit["allowed_tools"])
        self.assertIn("rbd_calculator", explicit["allowed_tools"])

    def test_hybrid_only_plan_uses_generic_seven_type_scope_without_field_contract(self):
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
            ["function_locator", "hmc", "fmeca", "fta", "rbd", "mfl", "fault_history"],
            decision["hybrid_doc_types"],
        )

    def test_structural_cause_field_requires_fta_without_user_document_terms(self):
        generic = resolve_tool_policy(
            intent="cause_analysis",
            query="为什么会黑屏？",
            query_targets=[{
                "intent": "cause_analysis",
                "required_fields": ["structural_cause_chain"],
            }],
            required_answer_fields=["structural_cause_chain"],
            coverage_scope="focused_detail",
        )
        self.assertIn("fta_query", generic["allowed_tools"])
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
            requested_sources=[{
                "doc_type": "rbd",
                "basis": "RBD",
                "requested_fields": ["rbd_single_failure_boundary"],
                "question_role": "capability",
            }],
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
        self.assertEqual(
            ["rbd_single_failure_boundary"],
            decision["source_coverage_fields"],
        )
        self.assertNotIn("rbd_calculator", decision["allowed_tools"])
        rbd_plan = next(
            item for item in decision["tools_plan"] if item["tool"] == "rbd_query"
        )
        self.assertIn("用户明确要求", rbd_plan["reason"])

    def test_planner_selected_graph_without_grounded_contract_is_rejected(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="这个故障通过哪些功能关联到模块？",
            proposed_plan=[
                {"tool": "fmeca_query"},
                {"tool": "graph_traversal"},
            ],
        )
        self.assertNotIn("graph_traversal", decision["allowed_tools"])
        self.assertNotIn("graph_traversal", decision["authorized_tools"])
        self.assertIn("graph_traversal", decision["rejected_tools"])

    def test_bare_relation_boolean_and_answer_field_do_not_authorize_graph(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="这个现象是什么原因",
            raw_user_query="这个现象是什么原因",
            query_targets=[{
                "intent": "cause_analysis",
                "required_fields": ["structural_cause_chain"],
            }],
            required_answer_fields=["structural_cause_chain"],
            coverage_scope="focused_detail",
            relation_expansion_required=True,
            proposed_plan=[{"tool": "graph_traversal"}],
        )

        self.assertNotIn("graph_traversal", decision["allowed_tools"])
        self.assertNotIn("graph_traversal", decision["authorized_tools"])
        self.assertFalse(decision["relation_expansion"].get("needed", False))

    def test_graph_capable_answer_fields_do_not_open_deferred_graph_execution(self):
        for field in ("relation_path", "impact_propagation", "fault_location"):
            with self.subTest(field=field):
                decision = resolve_tool_policy(
                    intent="architecture_query",
                    query="只读取当前字段",
                    raw_user_query="只读取当前字段",
                    query_targets=[{
                        "intent": "architecture_query",
                        "required_fields": [field],
                    }],
                    required_answer_fields=[field],
                    coverage_scope="focused_detail",
                    proposed_plan=[{"tool": "graph_traversal"}],
                )

                self.assertNotIn("graph_traversal", decision["allowed_tools"])
                self.assertNotIn(
                    "graph_traversal", decision["authorized_tools"]
                )
                self.assertIn("graph_traversal", decision["rejected_tools"])

    def test_relation_contract_requires_closed_kind_and_literal_current_basis(self):
        query = "请核对这个对象与上游模块的连接"
        for relation in (
            {"needed": True},
            {
                "needed": "yes",
                "relation_kind": "relation_path",
                "reference_mode": "current_query",
                "basis": "与上游模块的连接",
            },
            {
                "needed": True,
                "relation_kind": "invented_relation",
                "reference_mode": "current_query",
                "basis": "与上游模块的连接",
            },
            {
                "needed": True,
                "relation_kind": "relation_path",
                "reference_mode": "current_query",
                "basis": "来自改写而非当前问题",
            },
        ):
            with self.subTest(relation=relation):
                decision = resolve_tool_policy(
                    intent="architecture_query",
                    query=query,
                    raw_user_query=query,
                    relation_expansion=relation,
                    proposed_plan=[{"tool": "graph_traversal"}],
                )
                self.assertNotIn("graph_traversal", decision["allowed_tools"])

    def test_selected_anchor_relation_contract_requires_selected_offered_id(self):
        query = "它与上游模块怎样连接"
        contract = {
            "needed": True,
            "relation_kind": "relation_path",
            "reference_mode": "selected_anchor",
            "basis": "它与上游模块怎样连接",
            "selected_anchor_ids": ["ev-real", "ev-invented"],
        }
        accepted = resolve_tool_policy(
            intent="architecture_query",
            query=query,
            raw_user_query=query,
            relation_expansion=contract,
            selected_anchor_ids=["ev-real"],
            selected_anchor_doc_types=["function_locator"],
            proposed_plan=[{"tool": "graph_traversal"}],
        )
        rejected = resolve_tool_policy(
            intent="architecture_query",
            query=query,
            raw_user_query=query,
            relation_expansion=contract,
            selected_anchor_ids=["ev-other"],
            selected_anchor_doc_types=["function_locator"],
            proposed_plan=[{"tool": "graph_traversal"}],
        )

        self.assertIn("graph_traversal", accepted["allowed_tools"])
        self.assertEqual(
            ["ev-real"], accepted["relation_expansion"]["selected_anchor_ids"]
        )
        self.assertNotIn("graph_traversal", rejected["allowed_tools"])

    def test_cause_plan_without_relation_contract_does_not_authorize_graph(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="系统运行中总是无故反复重启，通常是什么原因？",
            proposed_plan=[{"tool": "fmeca_query"}, {"tool": "fta_query"}],
        )
        self.assertNotIn("graph_traversal", decision["allowed_tools"])
        self.assertNotIn("graph_traversal", decision["authorized_tools"])

    def test_planner_failure_uses_scoped_fallback_without_graph_or_calculators(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="运行中反复重启",
            proposed_plan=[{"tool": "graph_traversal"}],
            planner_failed=True,
        )
        self.assertEqual(
            ["hybrid_search", "evidence_synthesis"],
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

    def test_intent_targets_without_fields_do_not_create_a_source_bundle(self):
        decision = resolve_tool_policy(
            intent="cause_analysis",
            query="任意措辞",
            query_targets=[
                {"intent": "cause_analysis"},
                {"intent": "maintenance_query"},
                {"intent": "history_query"},
            ],
        )
        self.assertEqual(
            ["hybrid_search", "evidence_synthesis"], decision["allowed_tools"]
        )
        self.assertFalse(
            {
                "function_locator", "hmc_query", "fmeca_query", "fta_query",
                "rbd_query", "mfl_query", "fault_history_query",
            }
            & set(decision["allowed_tools"])
        )

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
            ["fmeca_query", "hybrid_search", "evidence_synthesis"],
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
        self.assertEqual(["fmeca"], decision["hybrid_doc_types"])
        self.assertIn("hybrid_search", decision["authorized_tools"])
        self.assertNotIn("hmc_query", decision["authorized_tools"])
        self.assertNotIn("mfl_query", decision["authorized_tools"])
        self.assertNotIn("fault_history_query", decision["authorized_tools"])

    def test_symptom_intent_authorizes_only_explicit_failure_mode_sources(self):
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

        self.assertIn("mfl_query", decision["authorized_tools"])
        self.assertNotIn("hmc_query", decision["authorized_tools"])
        self.assertNotIn("fault_history_query", decision["authorized_tools"])

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
            calculation_authorizations=[
                {
                    "operation": "rbd_reliability",
                    "mode": "calculate",
                    "target": "rbd",
                    "basis": "计算系统可靠度是多少",
                }
            ],
            proposed_plan=[],
        )
        self.assertNotIn("rbd_calculator", narrative["allowed_tools"])
        self.assertIn("rbd_calculator", narrative["rejected_tools"])
        self.assertIn("rbd_calculator", numeric["allowed_tools"])

    def test_rbd_recompute_paraphrase_authorizes_deterministic_calculator(self):
        query = "按资料里的失效率复算十小时任务的成功概率，并找出最低的一项"
        authorization = {
            "operation": "rbd_reliability",
            "mode": "calculate",
            "target": "rbd",
            "basis": "复算十小时任务的成功概率",
        }
        self.assertEqual(
            [authorization],
            normalize_calculation_authorizations(
                [authorization], raw_user_query=query
            ),
        )
        self.assertEqual(10.0, _mission_time(query))
        self.assertEqual(25.0, _mission_time("复算二十五小时任务可靠度"))
        decision = resolve_tool_policy(
            intent="reliability_query",
            query=query,
            calculation_authorizations=[authorization],
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
