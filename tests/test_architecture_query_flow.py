import asyncio
import importlib.util
import json
import sys
import types
import unittest
from unittest.mock import patch

# The repository intentionally supports a MockPostgres fallback. Keep this
# focused unit test runnable in the minimal host interpreter where psycopg2 is
# absent, without changing production imports.
if importlib.util.find_spec("psycopg2") is None:
    psycopg2_stub = types.ModuleType("psycopg2")

    def _unavailable_connect(*_args, **_kwargs):
        raise RuntimeError("psycopg2 unavailable in focused test environment")

    psycopg2_stub.connect = _unavailable_connect
    sys.modules["psycopg2"] = psycopg2_stub

from backend.agent.nodes import answer as answer_module
from backend.agent.nodes import plan as plan_module
from backend.agent.tools import icp_diagnosis as diagnosis_module


class ArchitecturePlanTests(unittest.TestCase):
    def test_historical_citation_markers_are_not_reused_in_next_turn_prompts(self):
        messages = [
            {"role": "assistant", "content": "上一轮结论来自旧证据。[^1][^7]"},
            {"role": "user", "content": "继续说明原因"},
        ]

        answer_history = answer_module._format_history(messages)
        plan_history = plan_module._format_history(messages)
        self.assertIn("上一轮结论来自旧证据", answer_history)
        self.assertIn("上一轮结论来自旧证据", plan_history)
        self.assertNotIn("[^1]", answer_history)
        self.assertNotIn("[^7]", answer_history)
        self.assertNotIn("[^1]", plan_history)
        self.assertNotIn("[^7]", plan_history)

    def test_cause_goal_authorizes_bounded_handling_and_history_completion(self):
        target_response = json.dumps(
            {"primary_intent": "cause_analysis", "targets": ["cause_analysis"]},
            ensure_ascii=False,
        )
        overbroad_plan = json.dumps(
            {
                "intent": "symptom_diagnosis",
                "query_targets": [
                    {"intent": "cause_analysis"},
                    {"intent": "maintenance_query", "reason": "推测用户隐含需要"},
                    {"intent": "history_query", "reason": "为了更全面"},
                ],
                "rewritten_query": "系统运行中反复自动重启的可能原因",
                "tools_plan": [
                    {"tool": "fta_query", "reason": "查询原因链"},
                    {"tool": "mfl_query", "reason": "补充已有处置资料"},
                    {"tool": "fault_history_query", "reason": "补充可比历史结果"},
                ],
                "reasoning": "分析潜在原因",
            },
            ensure_ascii=False,
        )
        state = {
            "query": "系统跑着跑着总是自己一遍遍重新启动，这通常是哪里出了问题？",
            "messages": [],
            "memory_enabled": False,
            "iteration": 0,
        }
        with (
            patch.object(
                plan_module.llm_service,
                "_call_model",
                side_effect=[target_response, overbroad_plan],
            ),
            patch.object(
                plan_module.agent_model_role_service,
                "apply_agent_llm_params",
                return_value=None,
            ),
        ):
            result = asyncio.run(plan_module.plan_node(state))

        self.assertEqual("cause_analysis", result["intent"])
        self.assertEqual(
            ["cause_analysis"],
            [item["intent"] for item in result["query_targets"]],
        )
        self.assertIn("fta_query", result["allowed_tools"])
        self.assertIn("mfl_query", result["allowed_tools"])
        self.assertIn("fault_history_query", result["allowed_tools"])

    def test_planner_architecture_output_is_policy_limited(self):
        model_response = json.dumps(
            {
                "intent": "architecture_query",
                "query_targets": [{"intent": "architecture_query"}],
                "rewritten_query": "SYN-ICP-9000 系统组成、层级和上下游接口",
                "entity_keywords": ["SYN-ICP-9000", "综合信息处理与显示系统"],
                "type_keywords": [],
                "query_frame": {"entities": ["SYN-ICP-9000"]},
                "tool_queries": {
                    "function_locator": ["SYN-ICP-9000 系统组成 层级 接口 上下游"],
                    "hybrid_search": ["SYN-ICP-9000 由哪些子系统和模块组成"],
                },
                "tools_plan": [
                    {"tool": "function_locator", "reason": "查询结构化架构事实"},
                    {"tool": "fta_query", "reason": "越界建议应被拒绝"},
                ],
                "reasoning": "这是架构知识查询",
            },
            ensure_ascii=False,
        )
        state = {
            "query": "SYN-ICP-9000 系统由哪些子系统和模块组成？上下游接口是什么？",
            "messages": [],
            "memory_enabled": False,
            "iteration": 0,
        }
        with (
            patch.object(plan_module.llm_service, "_call_model", return_value=model_response),
            patch.object(plan_module.agent_model_role_service, "apply_agent_llm_params", return_value=None),
        ):
            result = asyncio.run(plan_module.plan_node(state))

        self.assertEqual("architecture_query", result["intent"])
        self.assertEqual(
            ["function_locator"],
            [item["tool"] for item in result["tools_plan"]],
        )
        self.assertNotIn("fault_reasoning", [item["tool"] for item in result["tools_plan"]])

    def test_planner_failure_uses_broad_semantic_recall_not_keyword_guessing(self):
        state = {
            "query": "数据接口异常并伴随告警，如何排故？",
            "messages": [],
            "memory_enabled": False,
            "iteration": 0,
        }
        with (
            patch.object(plan_module.llm_service, "_call_model", return_value="not-json"),
            patch.object(plan_module.agent_model_role_service, "apply_agent_llm_params", return_value=None),
        ):
            result = asyncio.run(plan_module.plan_node(state))
        self.assertEqual("hybrid", result["intent"])
        self.assertEqual(
            ["function_locator", "hmc", "fmeca", "fta", "rbd", "mfl", "fault_history"],
            result["hybrid_doc_types"],
        )

    def test_simple_effect_question_is_deterministically_knowledge_explanation(self):
        model_response = json.dumps(
            {
                "intent": "impact_analysis",
                "query_targets": [{"intent": "impact_analysis"}],
                "rewritten_query": "滤波电容老化",
                "entity_keywords": ["滤波电容", "老化"],
                "type_keywords": ["影响"],
                "query_frame": {
                    "entities": ["滤波电容"],
                    "symptoms": ["老化"],
                    "normalized_terms": ["滤波电容老化"],
                },
                "tool_queries": {
                    "fmeca_query": ["滤波电容老化的局部影响和最终影响"],
                    "hybrid_search": ["滤波电容老化会有什么影响"],
                },
                "tools_plan": [{"tool": "fmeca_query", "reason": "查询影响记录"}],
                "reasoning": "用户询问部件状态的影响",
            },
            ensure_ascii=False,
        )
        state = {
            "query": "滤波电容有点老化了，会有什么影响吗？",
            "messages": [],
            "memory_enabled": False,
            "iteration": 0,
        }
        with (
            patch.object(plan_module.llm_service, "_call_model", return_value=model_response),
            patch.object(plan_module.agent_model_role_service, "apply_agent_llm_params", return_value=None),
        ):
            result = asyncio.run(plan_module.plan_node(state))

        self.assertEqual("impact_analysis", result["intent"])
        self.assertEqual("knowledge_explanation", result["answer_mode"])
        self.assertEqual(
            ["fmeca_query"],
            [item["tool"] for item in result["tools_plan"]],
        )


class ArchitectureDiagnosisTests(unittest.TestCase):
    def test_architecture_terms_keep_system_name_without_question_suffix(self):
        terms = diagnosis_module._unique_terms(
            {
                "intent": "architecture_query",
                "query": "综合信息处理与显示系统由哪些模块组成？",
                "rewritten_query": "综合信息处理与显示系统由哪些模块组成？",
                "entity_keywords": ["综合信息处理与显示系统"],
                "query_frame": {"entities": ["综合信息处理与显示系统"]},
            }
        )
        self.assertIn("综合信息处理与显示系统", terms)

    def test_architecture_path_uses_only_function_truth_and_projection(self):
        pg_rows = [
            {
                "doc_id": "doc-1",
                "function_id": "FUNC-001",
                "system_name": "综合信息处理与显示系统",
                "subsystem_name": "信息处理子系统",
                "module_name": "处理模块",
                "function_name": "总线数据处理",
                "description": "接收上游总线数据并向显示模块输出处理结果",
                "aliases": ["数据处理"],
                "failure_symptoms": "",
                "source_ref": "架构说明.docx#4.2",
                "doc_name": "架构说明.docx",
                "document_metadata": {
                    "document_number": "SYN-ICP-DOC-001",
                    "applicable_configuration": "CONFIG-A",
                    "document_version": "V1.0",
                },
            }
        ]
        search_rows = [
            {
                "doc_id": "doc-1",
                "doc_name": "架构说明.docx",
                "chunk_id": "edge-1",
                "title": "处理模块 → 显示模块",
                "content": "系统架构关系: 处理模块 → 显示模块；关系: 数据输出",
                "content_type": "function_architecture_edge",
                "document_metadata": {"document_version": "V1.0"},
                "score": 0.8,
            }
        ]
        state = {
            "intent": "architecture_query",
            "query": "系统由什么组成，上下游接口是什么？",
            "rewritten_query": "综合信息处理与显示系统 组成 上下游接口",
            "entity_keywords": ["综合信息处理与显示系统"],
            "retrieval_top_k": 10,
            "sql_limit": 8,
        }
        with (
            patch.object(diagnosis_module.icp_repository, "search_functions", return_value=pg_rows),
            patch.object(diagnosis_module, "_hybrid_search", return_value=(search_rows, {"reranked": True})),
            patch.object(diagnosis_module, "_enrich_function_sources", return_value=pg_rows),
            patch.object(diagnosis_module, "_query_hmc", side_effect=AssertionError("architecture path queried HMC")),
            patch.object(diagnosis_module, "_query_fmeca", side_effect=AssertionError("architecture path queried FMECA")),
        ):
            result = asyncio.run(diagnosis_module.icp_diagnosis_node(state))

        self.assertEqual([], result["candidate_faults"])
        self.assertEqual(0.0, result["diagnostic_confidence"])
        self.assertEqual("", result["clarification_question"])
        self.assertEqual(
            ["function_locator", "hybrid_search"],
            [item["tool"] for item in result["tool_calls"]],
        )
        self.assertEqual("postgres_authoritative", result["domain_results"][0]["authority"])
        self.assertEqual("CONFIG-A", result["domain_results"][0]["document_metadata"]["applicable_configuration"])
        self.assertIn("系统—子系统—模块—功能/接口证据", result["diagnostic_context"])
        self.assertIn("视觉候选·仅供浏览", result["diagnostic_context"])
        self.assertIn("自动生成 HugeGraph", result["diagnostic_context"])

    def test_architecture_early_return_hydrates_function_component_and_interface_hits(self):
        lexical_rows = [
            {
                "id": 1,
                "doc_id": "doc-arch",
                "function_id": "FUNC-BCA",
                "function_name": "BCA通道处理",
                "module_name": "BCA",
                "description": "处理BCA通道数据",
                "source_ref": "架构说明.docx#BCA",
            }
        ]
        search_rows = [
            {
                "doc_type_kwd": "function_locator",
                "doc_id": "doc-arch",
                "pg_id_kwd": "31",
                "business_id_kwd": "FUNC-330",
                "content_type": "function_record",
                "chunk_id": "func-330",
                "title": "外部数据标准化",
                "content": "功能ID: FUNC-330",
            },
            {
                "doc_type_kwd": "function_locator",
                "doc_id": "doc-arch",
                "pg_id_kwd": "12",
                "business_id_kwd": "SUB-IFC",
                "content_type": "function_component",
                "chunk_id": "sub-ifc",
                "title": "数据接口与通信子系统",
                "content": "组件ID: SUB-IFC",
            },
            {
                "doc_type_kwd": "function_locator",
                "doc_id": "doc-arch",
                "pg_id_kwd": "9",
                "business_id_kwd": "IF-IFC-GPP",
                "content_type": "function_interface",
                "chunk_id": "if-ifc-gpp",
                "title": "IFC到GPP数据流",
                "content": "外部数据经IFC标准化送入GPP",
            },
            {
                "doc_type_kwd": "function_locator",
                "doc_id": "doc-arch",
                "content_type": "function_text",
                "chunk_id": "paragraph-1",
                "title": "数据流说明",
                "content": "外部数据经IFC标准化送入GPP",
            },
        ]
        hydrated_rows = [
            {
                "id": 31,
                "doc_id": "doc-arch",
                "function_id": "FUNC-330",
                "function_name": "外部数据标准化",
                "module_id": "MOD-IFC",
                "module_name": "IFC",
                "description": "接收外部数据并标准化后送往GPP",
                "outputs": ["标准化业务数据"],
                "source_ref": "架构说明.docx#3.3;block=tbl-2;row=4",
                "content_type": "function_record",
            },
            {
                "id": 12,
                "doc_id": "doc-arch",
                "component_id": "SUB-IFC",
                "component_type": "subsystem",
                "name": "数据接口与通信子系统",
                "abbreviation": "IFC",
                "description": "接收并标准化外部数据",
                "source_ref": "架构说明.docx#3.2;block=tbl-1;row=3",
                "content_type": "function_component",
            },
            {
                "id": 9,
                "doc_id": "doc-arch",
                "interface_id": "IF-IFC-GPP",
                "source_component_id": "SUB-IFC",
                "target_component_id": "SUB-GPP",
                "relation_type": "data_flow",
                "signal_name": "标准化业务数据",
                "description": "外部数据经IFC标准化送入GPP",
                "source_ref": "架构说明.docx#3.4;block=tbl-3;row=2",
                "content_type": "function_interface",
            },
        ]
        state = {
            "intent": "architecture_query",
            "query_targets": [{"intent": "architecture_query"}],
            "query": "没有正确传出数据到外设，数据正常经过了哪些模块？",
            "rewritten_query": "外部数据 IFC GPP 数据流",
            "allowed_tools": ["function_locator", "hybrid_search", "evidence_synthesis"],
            "retrieval_top_k": 12,
            "sql_limit": 8,
        }
        with (
            patch.object(diagnosis_module.icp_repository, "search_functions", return_value=lexical_rows),
            patch.object(diagnosis_module, "_hybrid_search", return_value=(search_rows, {"reranked": True})),
            patch.object(
                diagnosis_module.icp_repository,
                "get_records_by_storage_locators",
                return_value=hydrated_rows,
            ) as lookup,
            patch.object(
                diagnosis_module,
                "_enrich_function_sources",
                side_effect=lambda rows, _errors=None: [dict(row) for row in rows],
            ),
        ):
            result = asyncio.run(diagnosis_module.icp_diagnosis_node(state))

        lookup.assert_called_once()
        self.assertEqual("FUNC-330", result["domain_results"][0]["function_id"])
        self.assertEqual(
            {"FUNC-BCA", "FUNC-330", "SUB-IFC", "IF-IFC-GPP"},
            {
                item.get("function_id") or item.get("component_id") or item.get("interface_id")
                for item in result["domain_results"]
                if item["authority"] == "postgres_authoritative"
            },
        )
        approved = [
            item
            for item in result["evidence_registry"]
            if item.get("authority") == "authoritative_record"
        ]
        self.assertEqual(
            {"FUNC-BCA", "FUNC-330", "SUB-IFC", "IF-IFC-GPP"},
            {item["record_id"] for item in approved},
        )
        self.assertIn("外部数据经IFC标准化送入GPP", result["diagnostic_context"])
        paragraph = next(
            item for item in result["evidence_registry"] if item.get("chunk_id") == "paragraph-1"
        )
        self.assertEqual("search_projection", paragraph["authority"])

    def test_architecture_leaf_function_expands_authoritative_family_and_interface(self):
        leaf_function = {
            "id": 31,
            "doc_id": "doc-flow",
            "function_id": "FUNC-LEAF",
            "parent_function_id": "FUNC-FAMILY",
            "function_name": "协议转换",
            "module_id": "MOD-PROTOCOL",
            "module_name": "协议转换模块",
            "description": "把外部协议转换为内部格式",
            "source_ref": "架构说明.docx#function-leaf",
            "content_type": "function_record",
        }
        expanded_rows = [
            {
                "id": 10,
                "doc_id": "doc-flow",
                "component_id": "SUB-DATA",
                "component_type": "subsystem",
                "name": "数据接口子系统",
                "description": "完成外部数据接入与转换",
                "source_ref": "架构说明.docx#subsystem",
                "content_type": "function_component",
            },
            {
                "id": 11,
                "doc_id": "doc-flow",
                "component_id": "MOD-CHANNEL-A",
                "parent_component_id": "SUB-DATA",
                "component_type": "module",
                "name": "A通道模块",
                "source_ref": "架构说明.docx#module-a",
                "content_type": "function_component",
            },
            {
                "id": 12,
                "doc_id": "doc-flow",
                "component_id": "MOD-CHANNEL-B",
                "parent_component_id": "SUB-DATA",
                "component_type": "module",
                "name": "B通道模块",
                "source_ref": "架构说明.docx#module-b",
                "content_type": "function_component",
            },
            {
                "id": 20,
                "doc_id": "doc-flow",
                "interface_id": "IF-DATA-PROCESSOR",
                "source_component_id": "SUB-DATA",
                "target_component_id": "SUB-PROCESSOR",
                "relation_type": "data_flow",
                "signal_name": "标准化业务数据",
                "description": "数据接口子系统将标准化数据送入处理子系统",
                "source_ref": "架构说明.docx#interface",
                "content_type": "function_interface",
            },
            leaf_function,
        ]
        state = {
            "intent": "architecture_query",
            "query_targets": [{"intent": "architecture_query"}],
            "query": "外部设备送来的报文进入处理单元前会经过哪里？",
            "rewritten_query": "外部报文 数据接口 处理单元 数据流",
            "allowed_tools": ["function_locator", "hybrid_search", "evidence_synthesis"],
            "retrieval_top_k": 12,
            "sql_limit": 8,
        }
        with (
            patch.object(diagnosis_module.icp_repository, "search_functions", return_value=[]),
            patch.object(
                diagnosis_module,
                "_hybrid_search",
                return_value=([{
                    "doc_type_kwd": "function_locator",
                    "doc_id": "doc-flow",
                    "pg_id_kwd": "31",
                    "business_id_kwd": "FUNC-LEAF",
                    "content_type": "function_record",
                    "title": "协议转换",
                    "content": "外部协议转换为内部格式",
                }], {"reranked": True}),
            ),
            patch.object(
                diagnosis_module.icp_repository,
                "get_records_by_storage_locators",
                return_value=[leaf_function],
            ),
            patch.object(
                diagnosis_module.icp_repository,
                "expand_function_context",
                return_value=expanded_rows,
            ) as expand,
            patch.object(
                diagnosis_module,
                "_enrich_function_sources",
                side_effect=lambda rows, _errors=None: [dict(row) for row in rows],
            ),
        ):
            result = asyncio.run(diagnosis_module.icp_diagnosis_node(state))

        expand.assert_called_once()
        self.assertEqual("FUNC-LEAF", expand.call_args.args[0][0]["function_id"])
        authoritative_ids = {
            item.get("function_id") or item.get("component_id") or item.get("interface_id")
            for item in result["domain_results"]
            if item["authority"] == "postgres_authoritative"
        }
        self.assertTrue(
            {"FUNC-LEAF", "SUB-DATA", "MOD-CHANNEL-A", "MOD-CHANNEL-B", "IF-DATA-PROCESSOR"}
            <= authoritative_ids
        )
        self.assertIn("标准化数据送入处理子系统", result["diagnostic_context"])
        self.assertTrue(
            {"SUB-DATA", "MOD-CHANNEL-A", "MOD-CHANNEL-B", "IF-DATA-PROCESSOR"}
            <= {item["record_id"] for item in result["evidence_registry"]}
        )

    def test_architecture_primary_intent_does_not_swallow_exact_hmc_subtarget(self):
        hmc_rows = [
            {
                "doc_id": "doc-hmc",
                "hmc_code": "91810001**0001",
                "lcn": "91810001",
                "hamc": "**",
                "hsc": "0001",
                "fault_desc": "ICU 输入过压钳位失效",
                "repair_method": "检查输入采样与钳位器件",
                "match_type": "exact_code",
            }
        ]
        state = {
            "intent": "architecture_query",
            "query": "HMC 91810001-**-0001 通过哪些关系关联到系统模块？",
            "rewritten_query": "HMC 91810001-**-0001 的跨文档模块关系",
            "query_targets": [
                {"intent": "architecture_query"},
                {"intent": "exact_code"},
            ],
            "allowed_tools": ["hmc_query", "graph_traversal", "evidence_synthesis"],
            "retrieval_top_k": 10,
            "sql_limit": 8,
        }
        with (
            patch.object(diagnosis_module, "_query_hmc", return_value=hmc_rows) as hmc_query,
            patch.object(
                diagnosis_module,
                "_enrich_function_sources",
                side_effect=lambda rows, _errors: rows,
            ),
        ):
            result = asyncio.run(diagnosis_module.icp_diagnosis_node(state))

        hmc_query.assert_called_once()
        self.assertIn("hmc_query", [item["tool"] for item in result["tool_calls"]])
        self.assertEqual("exact", result["graph_seed_bindings"][0]["binding_status"])
        self.assertTrue(result["graph_traversal_request"]["seed_ids"])


class ArchitectureAnswerTests(unittest.TestCase):
    def test_answer_uses_architecture_prompt_and_preserves_citation_metadata(self):
        state = {
            "intent": "architecture_query",
            "query": "处理模块的功能和下游是什么？",
            "messages": [],
            "diagnostic_context": "【证据】处理模块向显示模块输出 [^1]",
            "candidate_faults": [{"fault": "不应出现"}],
            "diagnostic_confidence": 0.91,
            "clarification_question": "不应提出故障追问",
            "domain_results": [
                {
                    "source_type": "function_locator_pg",
                    "authority": "postgres_authoritative",
                    "doc_id": "doc-1",
                    "doc_name": "架构说明.docx",
                    "chunk_id": "FUNC-001",
                    "title": "总线数据处理",
                    "content": "处理模块向显示模块输出",
                    "source_ref": "架构说明.docx#4.2",
                    "document_metadata": {"document_version": "V1.0"},
                }
            ],
        }
        with (
            patch.object(answer_module.llm_service, "_call_model", return_value="处理模块负责数据处理，并向显示模块输出。[^1]" ) as call,
            patch.object(answer_module.agent_model_role_service, "apply_agent_llm_params", return_value=None),
        ):
            result = asyncio.run(answer_module.answer_node(state))

        prompt = call.call_args.args[0][0]["content"]
        self.assertIn("系统 → 子系统 → 模块 → 功能", prompt)
        self.assertIn("不要输出候选故障", prompt)
        self.assertNotIn("候选故障：\n", prompt)
        self.assertEqual("V1.0", result["citations"][0]["document_metadata"]["document_version"])
        self.assertEqual("架构说明.docx#4.2", result["citations"][0]["source_ref"])

    def test_unconfigured_llm_does_not_emit_unprocessed_architecture_answer(self):
        state = {
            "intent": "architecture_query",
            "query": "系统有哪些模块？",
            "messages": [],
            "diagnostic_context": "【证据】模块 A [^1]",
            "candidate_faults": [{"fault": "不应出现"}],
            "diagnostic_confidence": 0.9,
            "domain_results": [],
        }
        with (
            patch.object(answer_module.llm_service, "_call_model", return_value="[LLM未配置]"),
            patch.object(answer_module.agent_model_role_service, "apply_agent_llm_params", return_value=None),
        ):
            with self.assertRaises(answer_module.AnswerGenerationError) as caught:
                asyncio.run(answer_module.answer_node(state))

        self.assertIn("未返回有效内容", str(caught.exception))
        self.assertNotIn("模块 A", str(caught.exception))

    def test_unconfirmed_topology_is_deterministically_labeled_pending(self):
        state = {
            "intent": "architecture_query",
            "query": "A 与 B 如何连接？",
            "messages": [],
            "safety_decision": {
                "status": "clarify",
                "can_claim_unique": False,
                "blocked_claims": ["formal_topology"],
                "questions": ["请确认当前有效架构图修订。"],
            },
        }
        with (
            patch.object(answer_module.llm_service, "_call_model", return_value="A 与 B 是已确认连接。"),
            patch.object(answer_module.agent_model_role_service, "apply_agent_llm_params", return_value=None),
        ):
            result = asyncio.run(answer_module.answer_node(state))

        self.assertIn("拓扑边界", result["final_answer"])
        self.assertNotIn("已确认连接", result["final_answer"])
        self.assertNotIn("尚未通过验证的连接", result["final_answer"])
        self.assertIn("请确认当前有效架构图修订", result["final_answer"])


if __name__ == "__main__":
    unittest.main()
