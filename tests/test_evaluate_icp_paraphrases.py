import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import evaluate_icp_paraphrases as evaluator


class EvaluateIcpParaphrasesTest(unittest.TestCase):
    def _evaluate(
        self,
        assertions,
        *,
        answer="",
        doc_types=(),
        used_doc_types=(),
        used_record_ids=(),
        scope_routes=(),
    ):
        return evaluator._evaluate_assertions(
            assertions,
            answer=answer,
            tools=(),
            doc_types=doc_types,
            used_doc_types=used_doc_types,
            used_record_ids=used_record_ids,
            evidence_count=3,
            used_evidence_count=2,
            citation_count=2,
            event_types=("answer", "done"),
            scope_routes=scope_routes,
            duration_seconds=1.0,
            runtime_errors=(),
            done_received=True,
            answer_received=True,
        )

    @staticmethod
    def _by_name(results):
        return {item["name"]: item for item in results}

    def test_citation_dimensions_only_use_compacted_final_citations(self):
        doc_types, record_ids = evaluator._citation_dimensions(
            [
                {"doc_type": "fmeca", "record_id": "FM-SYN-002"},
                {"doc_type": "fault_history", "record_id": "HIS-SYN-001"},
                {"doc_type": "fmeca", "record_id": "FM-SYN-002"},
                {"doc_type": "", "record_id": ""},
            ]
        )

        self.assertEqual(["fault_history", "fmeca"], doc_types)
        self.assertEqual(["FM-SYN-002", "HIS-SYN-001"], record_ids)

    def test_used_source_assertions_do_not_accept_unused_catalog_sources(self):
        results = self._by_name(
            self._evaluate(
                {
                    "required_doc_types": ["fta"],
                    "required_used_doc_types": ["fta"],
                    "required_used_record_ids": ["FTA-SYN-005"],
                },
                doc_types=("fmeca", "fta"),
                used_doc_types=("fmeca",),
                used_record_ids=("FM-SYN-002",),
            )
        )

        self.assertTrue(results["required_doc_types"]["passed"])
        self.assertFalse(results["required_used_doc_types"]["passed"])
        self.assertFalse(results["required_used_record_ids"]["passed"])

    def test_complete_fault_answer_facets_and_used_sources_pass(self):
        answer = (
            "可能原因：输入滤波器老化后纹波升高。\n"
            "影响：可能导致多个模块间歇异常并使冗余能力降低。\n"
            "处理措施：先检查接地和纹波，确认后更换滤波器并复测。\n"
            "历史案例：2025年9月9日曾发生类似现象。"
            "排故方法：检查队列和接口状态；最终方案：恢复受控参数。"
            "验证结果：处置后在压力测试中无丢包。"
        )
        required_facets = [
            "cause",
            "impact",
            "resolution",
            "history_time",
            "history_action",
            "history_result",
        ]
        results = self._by_name(
            self._evaluate(
                {
                    "required_used_doc_types": ["fmeca", "fault_history"],
                    "any_used_record_ids": ["HIS-SYN-013", "HIS-SYN-099"],
                    "forbidden_used_record_ids": ["HIS-SYN-002"],
                    "required_answer_facets": required_facets,
                },
                answer=answer,
                doc_types=("fmeca", "fault_history", "fta"),
                used_doc_types=("fmeca", "fault_history"),
                used_record_ids=("FM-SYN-038", "HIS-SYN-013"),
            )
        )

        self.assertTrue(results["required_used_doc_types"]["passed"])
        self.assertTrue(results["any_used_record_ids"]["passed"])
        self.assertTrue(results["forbidden_used_record_ids"]["passed"])
        self.assertTrue(results["required_answer_facets"]["passed"])
        self.assertEqual(
            sorted(required_facets),
            results["required_answer_facets"]["actual"],
        )

    def test_history_facets_require_both_history_citation_and_answer_details(self):
        answer = "可能原因：连接松动。处理措施：检查连接器，必要时重新压接。"
        results = self._by_name(
            self._evaluate(
                {
                    "required_answer_facets": [
                        "cause",
                        "resolution",
                        "history_time",
                        "history_action",
                        "history_result",
                    ]
                },
                answer=answer,
                used_doc_types=("fault_history",),
                used_record_ids=("HIS-SYN-011",),
            )
        )

        facet_result = results["required_answer_facets"]
        self.assertFalse(facet_result["passed"])
        self.assertEqual(["cause", "resolution"], facet_result["actual"])

    def test_natural_history_sentence_detects_action_and_verification(self):
        answer = (
            "在2025年8月2日的联试中发生过类似情况。"
            "当时通过检查物理层幅值并重新压接连接器修复了故障，"
            "验证连续接收两小时无丢帧。"
        )

        facets = evaluator._detect_answer_facets(
            answer,
            used_doc_types=("fault_history",),
        )

        self.assertIn("history_time", facets)
        self.assertIn("history_action", facets)
        self.assertIn("history_result", facets)

    def test_history_handling_solution_label_counts_as_an_action(self):
        facets = evaluator._detect_answer_facets(
            "历史案例的处理方案：重新压接并固定针脚；验证结果为连续接收两小时无丢帧。",
            used_doc_types=("fault_history",),
        )

        self.assertIn("history_action", facets)
        self.assertIn("history_result", facets)

    def test_hyphenated_history_date_between_chinese_text_is_detected(self):
        facets = evaluator._detect_answer_facets(
            "在2025-09-09的高流量试验中，验证结果显示处置后无丢包。",
            used_doc_types=("fault_history",),
        )

        self.assertIn("history_time", facets)
        self.assertIn("history_result", facets)

    def test_network_integrity_decline_is_detected_as_impact(self):
        facets = evaluator._detect_answer_facets(
            "缓存不足会导致网络转发完整性下降，进而造成任务数据间歇缺失或延迟。",
            used_doc_types=("fmeca",),
        )

        self.assertIn("impact", facets)

    def test_chinese_first_bilingual_labels_are_semantic_facets(self):
        facets = evaluator._detect_answer_facets(
            "**原因 (cause)**：缓存不足。"
            "**影响（impact）**：网络转发完整性下降。",
            used_doc_types=("fmeca",),
        )

        self.assertIn("cause", facets)
        self.assertIn("impact", facets)

    def test_capability_loss_and_unavailable_data_are_detected_as_impact(self):
        facets = evaluator._detect_answer_facets(
            "器件失效会导致局部接收能力丧失，最终表现为部分外部数据不可用。",
            used_doc_types=("fmeca",),
        )

        self.assertIn("impact", facets)

    def test_mechanism_and_combined_handling_label_are_detected(self):
        facets = evaluator._detect_answer_facets(
            "性能下降后的核心机制在于传导干扰扩散。"
            "相似历史的处置与验证：通过抓帧定位问题，重新压接后恢复。",
            used_doc_types=("fault_history",),
        )

        self.assertIn("cause", facets)
        self.assertIn("history_action", facets)

    def test_causal_components_followed_by_leads_to_are_detected(self):
        facets = evaluator._detect_answer_facets(
            "缓存不足、端口拥塞或处理性能下降会导致网络转发完整性下降。",
            used_doc_types=("fmeca",),
        )

        self.assertIn("cause", facets)

    def test_natural_history_root_cause_direction_is_detected(self):
        facets = evaluator._detect_answer_facets(
            "该案例的根因指向物理连接与终端问题，"
            "重新压接后确认故障位于连接器。",
            used_doc_types=("fault_history",),
        )

        self.assertIn("cause", facets)

    def test_unknown_answer_facet_is_rejected(self):
        with self.assertRaisesRegex(evaluator.EvaluationInputError, "unknown_facet"):
            evaluator._normalize_assertions(
                {"required_answer_facets": ["cause", "unknown_facet"]},
                "case.assertions",
            )

    def test_scope_route_assertions_are_evaluated_from_public_stage_events(self):
        results = self._by_name(
            self._evaluate(
                {
                    "required_scope_routes": ["knowledge_base"],
                    "forbidden_scope_routes": ["out_of_scope"],
                },
                scope_routes=("knowledge_base",),
            )
        )

        self.assertTrue(results["required_scope_routes"]["passed"])
        self.assertTrue(results["forbidden_scope_routes"]["passed"])

    def test_multi_turn_fixture_keeps_each_dialogue_in_one_ordered_case(self):
        fixture = (
            Path(__file__).resolve().parents[1]
            / "evals"
            / "icp_multiturn_dialogue_cases.json"
        )
        cases, metadata = evaluator._load_cases(str(fixture))

        self.assertEqual("ICP 智能体多轮对话回归集", metadata["name"])
        self.assertGreaterEqual(len(cases), 15)
        self.assertTrue(all(len(case.turns) >= 2 for case in cases))
        self.assertGreaterEqual(sum(len(case.turns) for case in cases), 35)
        self.assertEqual(
            ["definition", "importance"],
            [turn.turn_id for turn in cases[0].turns],
        )
        capability_case = next(
            case for case in cases if case.case_id == "MT-ROUTER-01-capability-and-scope"
        )
        self.assertEqual(
            ["agent_capability"],
            capability_case.turns[0].assertions["required_scope_routes"],
        )

    def test_multi_turn_runner_reuses_one_session_and_reports_each_turn(self):
        turns = (
            evaluator.EvaluationTurn(
                index=1,
                turn_id="first",
                question="先问模块定义",
                assertions={
                    "required_scope_routes": ["knowledge_base"],
                    "required_tools": ["function_locator"],
                    "required_event_types": ["answer", "done"],
                },
                metadata={},
            ),
            evaluator.EvaluationTurn(
                index=2,
                turn_id="followup",
                question="再问它的重要性",
                assertions={
                    "required_scope_routes": ["knowledge_base"],
                    "required_tools": ["function_locator"],
                    "required_event_types": ["answer", "done"],
                },
                metadata={},
            ),
        )
        case = evaluator.EvaluationCase(
            index=0,
            case_id="dialogue",
            group_id="dialogue",
            variant_id="dialogue",
            question=turns[0].question,
            assertions=turns[-1].assertions,
            metadata={},
            turns=turns,
        )
        config = evaluator.RunnerConfig(
            base_url="http://127.0.0.1:19999",
            agent_id="agent-1",
            user_id="evaluation",
            timeout_seconds=5.0,
            keep_sessions=False,
            include_raw_events=False,
            chat_model_id="",
            embedding_model_id="",
            rerank_model_id="",
        )
        payloads = []

        def request(method, url, *, payload=None, timeout):
            if method == "POST" and url.endswith("/api/session"):
                return {"id": "shared-session"}
            if method == "DELETE" and url.endswith("/api/session/shared-session"):
                return {"deleted": True}
            self.fail(f"unexpected request: {method} {url}")

        def stream(url, payload, *, timeout):
            payloads.append(dict(payload))
            return iter(
                [
                    {
                        "type": "stage",
                        "stage": "scope_routing",
                        "route": "knowledge_base",
                    },
                    {"type": "plan"},
                    {"type": "tool_call", "tool": "function_locator"},
                    {
                        "type": "answer",
                        "content": "模块资料已找到。",
                        "citations": [],
                        "evidence_catalog": [],
                        "used_evidence_ids": [],
                    },
                    {"type": "done"},
                ]
            )

        with patch.object(evaluator, "_request_json", side_effect=request), patch.object(
            evaluator, "_stream_sse", side_effect=stream
        ):
            result = evaluator._run_case(case, config)

        self.assertTrue(result["passed"])
        self.assertEqual(2, result["turn_count"])
        self.assertEqual(["first", "followup"], [turn["turn_id"] for turn in result["turns"]])
        self.assertEqual(["shared-session", "shared-session"], [item["session_id"] for item in payloads])
        self.assertTrue(result["cleanup"]["deleted"])


if __name__ == "__main__":
    unittest.main()
