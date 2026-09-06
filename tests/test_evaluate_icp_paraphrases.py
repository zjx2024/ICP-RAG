import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.request import Request

from scripts import evaluate_icp_paraphrases as evaluator


class EvaluateIcpParaphrasesTest(unittest.TestCase):
    def test_configured_lan_models_bypass_proxy_but_public_hosts_do_not(self):
        opener = Mock()
        with patch.object(evaluator, "build_opener", return_value=opener), patch.object(evaluator, "urlopen") as proxied:
            for url in ("http://127.0.0.1:19999/readyz", "http://192.168.2.8:11434/api/chat", "http://10.1.2.3/api/chat", "http://[fd00::8]/api/chat"):
                evaluator._open_url(Request(url), 2)
            self.assertEqual(4, opener.open.call_count)
            proxied.assert_not_called()
            evaluator._open_url(Request("https://public.example/api/chat"), 2)
            evaluator._open_url(Request("https://8.8.8.8/api/chat"), 2)
            self.assertEqual(2, proxied.call_count)

    def _evaluate(
        self,
        assertions,
        *,
        answer="",
        prior_answer="",
        doc_types=(),
        used_doc_types=(),
        used_record_ids=(),
        scope_routes=(),
        completion_status="",
        delivered_answer_fields=(),
        answer_delivery_manifest=(),
    ):
        return evaluator._evaluate_assertions(
            assertions,
            answer=answer,
            prior_answer=prior_answer,
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
            completion_status=completion_status,
            delivered_answer_fields=delivered_answer_fields,
            answer_delivery_manifest=answer_delivery_manifest,
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

    def test_history_condition_and_recovery_standard_are_distinct_facets(self):
        answer = (
            "相近案例的复现条件：在70 ℃、80%负载条件下连续运行后复现中断。\n"
            "恢复后的复测标准：完成三轮热循环，输出保持稳定且未发生切换。"
        )

        facets = evaluator._detect_answer_facets(
            answer, used_doc_types=("fault_history",)
        )

        self.assertIn("history_condition", facets)
        self.assertIn("history_result", facets)
        self.assertNotIn("history_action", facets)

    def test_allowed_used_records_rejects_unlisted_semantic_neighbor(self):
        results = self._by_name(
            self._evaluate(
                {
                    "allowed_used_doc_types": ["fmeca", "mfl"],
                    "allowed_used_record_ids": ["FM-NAV-1", "MFL-NAV-1"],
                },
                used_doc_types=("fmeca", "mfl"),
                used_record_ids=("FM-NAV-1", "MFL-NAV-1", "MFL-NAV-9"),
            )
        )

        self.assertTrue(results["allowed_used_doc_types"]["passed"])
        self.assertFalse(results["allowed_used_record_ids"]["passed"])

    def test_max_answer_chars_detects_field_dump_even_when_overlap_is_low(self):
        result = self._by_name(
            self._evaluate(
                {"max_answer_chars": 20},
                answer="组件类型、父组件ID、集合角色等字段被逐项转储。",
            )
        )["max_answer_chars"]

        self.assertFalse(result["passed"])
        self.assertGreater(result["actual"], result["expected"])

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

    def test_structured_assertion_schemas_are_not_coerced_to_string_lists(self):
        normalized = evaluator._normalize_assertions(
            {
                "semantic_criteria": [
                    {
                        "id": "importance",
                        "criterion": "解释该能力对当前任务的作用及最先受影响的环节。",
                    }
                ],
                "numeric_expectations": [
                    {
                        "id": "reliability",
                        "value": 0.9989,
                        "absolute_tolerance": 0.00001,
                        "model_identity": "显示输出模型",
                        "business_identity": "revision-7",
                    }
                ],
                "response_obligation_expectations": [
                    {"id": "importance", "status": "not_machine_verified"}
                ],
            },
            "case.assertions",
        )

        self.assertEqual("importance", normalized["semantic_criteria"][0]["id"])
        self.assertEqual(0.9989, normalized["numeric_expectations"][0]["value"])
        self.assertEqual("both", normalized["numeric_expectations"][0]["location"])
        self.assertEqual(
            "not_machine_verified",
            normalized["response_obligation_expectations"][0]["status"],
        )
        with self.assertRaisesRegex(evaluator.EvaluationInputError, "JSON 对象"):
            evaluator._normalize_assertions(
                {"semantic_criteria": ["importance"]}, "case.assertions"
            )
        with self.assertRaisesRegex(evaluator.EvaluationInputError, "有限数值"):
            evaluator._normalize_assertions(
                {
                    "numeric_expectations": [
                        {"id": "bad", "value": "0.9"}
                    ]
                },
                "case.assertions",
            )

    def test_manifest_obligation_delivery_and_completion_assertions(self):
        manifest = (
            {
                "kind": "response_obligation",
                "obligation": "importance",
                "status": "not_machine_verified",
            },
            {"kind": "field_delivery", "field": "responsibility"},
        )
        results = self._by_name(
            self._evaluate(
                {
                    "completion_status": "full",
                    "required_response_obligations": ["importance"],
                    "response_obligation_expectations": [
                        {"id": "importance", "status": "not_machine_verified"}
                    ],
                    "required_delivered_fields": ["responsibility"],
                    "forbidden_delivered_fields": ["local_impact"],
                    "allowed_delivered_fields": ["responsibility"],
                },
                completion_status="full",
                delivered_answer_fields=("responsibility",),
                answer_delivery_manifest=manifest,
            )
        )

        self.assertTrue(results["completion_status"]["passed"])
        self.assertTrue(results["required_response_obligations"]["passed"])
        self.assertTrue(
            results["response_obligation_status:importance"]["passed"]
        )
        self.assertEqual(
            "not_machine_verified",
            results["required_response_obligations"]["actual"]["statuses"][
                "importance"
            ],
        )
        self.assertTrue(results["required_delivered_fields"]["passed"])
        self.assertTrue(results["forbidden_delivered_fields"]["passed"])
        self.assertTrue(results["allowed_delivered_fields"]["passed"])

        missing = self._by_name(
            self._evaluate(
                {"required_response_obligations": ["immediate_impact"]},
                answer_delivery_manifest=manifest,
            )
        )
        self.assertFalse(missing["required_response_obligations"]["passed"])

        wrong_status = self._by_name(
            self._evaluate(
                {
                    "response_obligation_expectations": [
                        {"id": "importance", "status": "fulfilled"}
                    ]
                },
                answer_delivery_manifest=manifest,
            )
        )
        self.assertFalse(wrong_status["response_obligation_status:importance"]["passed"])

    def test_numeric_expectation_checks_answer_tolerance_and_bound_identity(self):
        expectation = evaluator._normalize_numeric_expectations(
            [
                {
                    "id": "display-path",
                    "value": 0.9989,
                    "absolute_tolerance": 0.00001,
                    "model_identity": "显示输出模型",
                    "business_identity": "revision-7",
                }
            ],
            "numeric_expectations",
        )
        structured = [
            {
                "title": "deterministic result：显示输出模型",
                "record_id": "revision-7",
                "content": "computed value = 0.998904",
                "evaluation_citation_id": "1",
            }
        ]

        assertions, report = evaluator._evaluate_numeric_expectations(
            expectation,
            answer="重新计算得到 0.998901。[^1]",
            structured_sources=structured,
        )

        self.assertTrue(assertions[0]["passed"])
        self.assertTrue(report[0]["actual"]["answer_match"])
        self.assertTrue(report[0]["actual"]["structured_match"])

        wrong_identity = [dict(expectation[0], business_identity="revision-8")]
        assertions, _ = evaluator._evaluate_numeric_expectations(
            wrong_identity,
            answer="重新计算得到 0.998901。",
            structured_sources=structured,
        )
        self.assertFalse(assertions[0]["passed"])

    def test_numeric_expectation_does_not_accept_answer_when_structured_result_drifted(self):
        expectation = evaluator._normalize_numeric_expectations(
            [{"id": "value", "value": 0.91, "absolute_tolerance": 0.001}],
            "numeric_expectations",
        )
        assertions, report = evaluator._evaluate_numeric_expectations(
            expectation,
            answer="结果为 0.91。",
            structured_sources=[{"content": "computed value = 0.87"}],
        )

        self.assertFalse(assertions[0]["passed"])
        self.assertTrue(report[0]["actual"]["answer_match"])
        self.assertFalse(report[0]["actual"]["structured_match"])

    def test_numeric_values_swapped_between_models_fail_despite_correct_tool_results(self):
        expectations = evaluator._normalize_numeric_expectations([
            {"id": "a", "value": 0.91, "model_identity": "模型甲", "business_identity": "revision-a"},
            {"id": "b", "value": 0.87, "model_identity": "模型乙", "business_identity": "revision-b"},
        ], "numeric")
        sources = evaluator._numeric_structured_sources([
            {"id": 1, "title": "模型甲", "record_id": "revision-a", "content": "0.91"},
            {"id": 2, "title": "模型乙", "record_id": "revision-b", "content": "0.87"},
        ], [], [])
        for answer in (
            "模型甲为 0.87。[^1]\n模型乙为 0.91。[^2]",
            "模型甲为 0.87[^2]，模型乙为 0.91[^1]。",
            "模型甲、模型乙的结果是 0.87、0.91。",
            "结果是 0.91 和 0.87。",
        ):
            with self.subTest(answer=answer):
                assertions, _ = evaluator._evaluate_numeric_expectations(expectations, answer=answer, structured_sources=sources)
                self.assertFalse(any(item["passed"] for item in assertions))
        for answer in (
            "模型甲为 0.91。[^1]\n模型乙为 0.87。[^2]",
            "模型甲为 0.91[^1]，模型乙为 0.87[^2]。",
            "重新核算：0.91[^1]。\n另一项为 0.87[^2]。",
            "| 模型甲 | 0.91 | [^1] |\n| 模型乙 | 0.87 | [^2] |",
        ):
            with self.subTest(answer=answer):
                assertions, _ = evaluator._evaluate_numeric_expectations(expectations, answer=answer, structured_sources=sources)
                self.assertTrue(all(item["passed"] for item in assertions))

    def test_numeric_catalog_record_id_is_not_a_public_citation(self):
        sources = evaluator._numeric_structured_sources([], [
            {"id": 1, "evaluation_citation_id": "1", "title": "模型甲", "content": "0.91"},
        ], [])
        expectations = evaluator._normalize_numeric_expectations([
            {"id": "a", "value": 0.91, "model_identity": "模型甲"},
        ], "numeric")
        assertions, _ = evaluator._evaluate_numeric_expectations(expectations, answer="结果 0.91[^1]。", structured_sources=sources)
        self.assertFalse(assertions[0]["passed"])

    def test_numeric_result_binds_quantity_parameters_and_units(self):
        expectation = evaluator._normalize_numeric_expectations([{
            "id": "ten-hours", "value": 0.91, "model_identity": "模型甲",
            "result_field": "system_reliability", "result_unit": "1",
            "parameters": {"mission_time": 10, "mission_time_unit": "h"},
        }], "numeric")
        calculation = {
            "parameters": {"mission_time": 10, "mission_time_unit": "h"},
            "results": {"system_reliability": 0.91, "unreliability": 0.09},
            "units": {"system_reliability": "1"},
        }
        source = {"title": "模型甲", "authority": "deterministic_derivation", "metadata": {"calculation": calculation}, "evaluation_citation_id": "1"}
        result, _ = evaluator._evaluate_numeric_expectations(expectation, answer="0.91[^1]", structured_sources=[source])
        self.assertTrue(result[0]["passed"])
        for bad in (
            {**calculation, "parameters": {"mission_time": 20, "mission_time_unit": "h"}},
            {**calculation, "parameters": {"mission_time": 10, "mission_time_unit": "min"}},
            {**calculation, "results": {"system_reliability": 0.09, "unreliability": 0.91}},
            {**calculation, "units": {"system_reliability": "%"}},
        ):
            result, _ = evaluator._evaluate_numeric_expectations(expectation, answer="0.91[^1]", structured_sources=[{**source, "metadata": {"calculation": bad}}])
            self.assertFalse(result[0]["passed"])
        result, _ = evaluator._evaluate_numeric_expectations(expectation, answer="0.91[^1]", structured_sources=[{**source, "authority": "authoritative_record"}])
        self.assertFalse(result[0]["passed"])

    def test_numeric_identity_cannot_bind_to_sibling_tool_preview_value(self):
        sources = evaluator._numeric_structured_sources(
            (),
            (),
            (
                {
                    "tool": "calculator",
                    "results_preview": [
                        {
                            "title": "model-a",
                            "record_id": "record-a",
                            "content": "computed value = 0.80",
                        },
                        {
                            "title": "model-b",
                            "record_id": "record-b",
                            "content": "computed value = 0.95",
                        },
                    ],
                },
            ),
        )
        expectation = evaluator._normalize_numeric_expectations(
            [
                {
                    "id": "bound-value",
                    "value": 0.95,
                    "model_identity": "model-a",
                    "business_identity": "record-a",
                }
            ],
            "numeric_expectations",
        )

        assertions, _ = evaluator._evaluate_numeric_expectations(
            expectation,
            answer="结果为 0.95。",
            structured_sources=sources,
        )

        self.assertEqual(2, len(sources))
        self.assertFalse(assertions[0]["passed"])

    def test_semantic_judge_accepts_natural_rewrite(self):
        class FakeJudge:
            def evaluate(self, *, question, prior_turns, answer, criteria):
                self.seen = (question, prior_turns, answer, criteria)
                return [
                    {
                        "id": "first-impact",
                        "passed": True,
                        "reason": "回答建立了职责、不可用状态与首个任务后果的关系。",
                    }
                ]

        judge = FakeJudge()
        criteria = [
            {
                "id": "first-impact",
                "criterion": "说明该模块为何重要，并指出不可用时最早受损的任务能力。",
            }
        ]
        results, judgement = evaluator._evaluate_semantic_criteria(
            criteria,
            judge=judge,
            question="那它为什么重要？暂时不可用会怎样？",
            prior_turns=[{"question": "它是什么？", "answer": "它负责汇聚传感数据。"}],
            answer="它承接各传感器的数据入口；一旦下线，航迹输入会先中断。",
        )

        self.assertTrue(results[0]["passed"])
        self.assertEqual("completed", judgement["status"])
        self.assertEqual(1, len(judge.seen[1]))

    def test_semantic_keyword_bag_does_not_pass_without_judge_approval(self):
        class FakeJudge:
            def evaluate(self, **_kwargs):
                return [
                    {
                        "id": "first-impact",
                        "passed": False,
                        "reason": "只出现概念词，没有形成对问题的解释。",
                    }
                ]

        results, _ = evaluator._evaluate_semantic_criteria(
            [
                {
                    "id": "first-impact",
                    "criterion": "解释重要性和最先受到的任务影响。",
                }
            ],
            judge=FakeJudge(),
            question="为什么重要，最先影响什么？",
            prior_turns=(),
            answer="重要、任务、影响、最先。",
        )

        self.assertFalse(results[0]["passed"])

        turn = evaluator.EvaluationTurn(
            index=1,
            turn_id="semantic",
            question="为什么重要，最先影响什么？",
            assertions={
                "answer_regex": ["重要", "任务", "影响", "最先"],
                "semantic_criteria": [
                    {
                        "id": "first-impact",
                        "criterion": "解释重要性和最先受到的任务影响。",
                    }
                ],
            },
            metadata={},
        )
        case = evaluator.EvaluationCase(
            index=0,
            case_id="semantic",
            group_id="semantic",
            variant_id="semantic",
            question=turn.question,
            assertions=turn.assertions,
            metadata={},
            turns=(turn,),
        )
        built = evaluator._build_turn_result(
            case,
            turn,
            include_raw_events=False,
            events=({"type": "answer"}, {"type": "done"}),
            runtime_errors=(),
            answer="重要、任务、影响、最先。",
            citations=(),
            evidence_catalog=(),
            used_evidence_ids=(),
            done_received=True,
            answer_received=True,
            started_at="2026-01-01T00:00:00Z",
            duration_seconds=0.1,
            semantic_judge=FakeJudge(),
        )
        by_name = self._by_name(built["assertion_results"])
        self.assertTrue(by_name["answer_regex:重要"]["passed"])
        self.assertFalse(by_name["semantic_criterion:first-impact"]["passed"])
        self.assertFalse(built["passed"])

    def test_semantic_judge_failure_is_fail_closed(self):
        class BrokenJudge:
            def evaluate(self, **_kwargs):
                raise RuntimeError("judge unavailable")

        results, judgement = evaluator._evaluate_semantic_criteria(
            [{"id": "relevance", "criterion": "回答当前追问。"}],
            judge=BrokenJudge(),
            question="继续说明。",
            prior_turns=(),
            answer="一段回答。",
        )

        self.assertFalse(results[0]["passed"])
        self.assertEqual("error", judgement["status"])
        self.assertIn("judge unavailable", judgement["error"])

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

    def test_followup_overlap_rejects_replayed_prior_answer(self):
        prior = (
            "系统共六个子系统：显示处理、通用处理、健康监测、"
            "数据接口、电源与配电、时钟与热管理。"
        )
        replay = "共六个子系统：显示处理、通用处理、健康监测、数据接口、电源与配电、时钟与热管理。"
        focused = "总数是六个；没有漏掉，供电对应电源与配电，显示对应显示处理。"

        replay_result = self._by_name(
            self._evaluate(
                {"max_prior_answer_overlap": 0.7},
                answer=replay,
                prior_answer=prior,
            )
        )["max_prior_answer_overlap"]
        focused_result = self._by_name(
            self._evaluate(
                {"max_prior_answer_overlap": 0.7},
                answer=focused,
                prior_answer=prior,
            )
        )["max_prior_answer_overlap"]

        self.assertFalse(replay_result["passed"])
        self.assertTrue(focused_result["passed"])

    def test_followup_overlap_ratio_is_bounded(self):
        self.assertEqual(
            0.0,
            evaluator._normalize_assertions(
                {"max_prior_answer_overlap": 0}, "case.assertions"
            )["max_prior_answer_overlap"],
        )
        with self.assertRaisesRegex(
            evaluator.EvaluationInputError, "0 到 1"
        ):
            evaluator._normalize_assertions(
                {"max_prior_answer_overlap": 1.1}, "case.assertions"
            )

    def test_turn_result_evaluates_overlap_against_supplied_prior_answer(self):
        case = evaluator.EvaluationCase(
            index=1,
            case_id="MT-GENERIC",
            group_id="generic",
            variant_id="base",
            question="首轮问题",
            assertions={},
            metadata={},
            turns=(),
        )
        turn = evaluator.EvaluationTurn(
            index=2,
            turn_id="followup",
            question="那具体呢？",
            assertions={"max_prior_answer_overlap": 0.7},
            metadata={},
        )

        result = evaluator._build_turn_result(
            case,
            turn,
            include_raw_events=False,
            prior_answer="系统包含供电和显示及处理三个部分。",
            events=({"type": "answer"}, {"type": "done"}),
            runtime_errors=(),
            answer="系统总数是三个，供电和显示都没有遗漏。",
            citations=(),
            evidence_catalog=(),
            used_evidence_ids=(),
            done_received=True,
            answer_received=True,
            started_at="2026-01-01T00:00:00Z",
            duration_seconds=0.1,
        )

        overlap = self._by_name(result["assertion_results"])[
            "max_prior_answer_overlap"
        ]
        self.assertTrue(overlap["passed"])
        self.assertGreater(overlap["actual"], 0)

    def test_named_prior_sentence_check_reports_a_passing_assertion(self):
        result = self._by_name(
            self._evaluate(
                {"answer_not_repeat_prior": ["启动模板"]},
                prior_answer="模块负责保存启动模板和任务数据。",
                answer="最先受影响的是任务初始化。",
            )
        )["answer_not_repeat_prior"]

        self.assertTrue(result["passed"])
        self.assertEqual([], result["actual"])

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
        architecture_case = next(
            case for case in cases if case.case_id == "MT-ARCH-02-complete-system"
        )
        self.assertNotIn("max_prior_answer_overlap", architecture_case.turns[1].assertions)
        self.assertTrue(architecture_case.turns[1].assertions["semantic_criteria"])
        self.assertTrue(all(turn.metadata["resolved_gold_context"] for case in cases for turn in case.turns))
        history_case = next(
            case for case in cases if case.case_id == "MT-HISTORY-01-line-versus-board"
        )
        self.assertEqual(
            ["HIS-SYN-011"],
            history_case.turns[1].assertions["required_used_record_ids"],
        )

    def test_rbd_boundary_fixture_requires_semantic_judgement_not_keyword_bag(self):
        fixture = (
            Path(__file__).resolve().parents[1]
            / "evals"
            / "icp_multiturn_dialogue_cases.json"
        )
        cases, _metadata = evaluator._load_cases(str(fixture))
        case = next(
            item
            for item in cases
            if item.case_id == "MT-RBD-01-redundancy-boundary"
        )
        turn = case.turns[1]
        self.assertNotIn("answer_regex", turn.assertions)
        criteria = turn.assertions["semantic_criteria"]
        self.assertEqual({"current_turn", "source_fidelity", "expected_dimension"}, {item["id"] for item in criteria})
        self.assertTrue(turn.metadata["resolved_gold_context"])
        # Missing a real judge must not silently fall back to keyword credit.
        results, judgement = evaluator._evaluate_semantic_criteria(
            criteria, judge=None, question=turn.question, prior_turns=[],
            answer="成功路径、冗余、根因、维修。",
            reference_context=turn.metadata["resolved_gold_context"],
        )
        self.assertEqual("error", judgement["status"])
        self.assertTrue(all(not item["passed"] for item in results))

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

    def test_runner_captures_delivery_contract_from_answer_and_done_events(self):
        turn = evaluator.EvaluationTurn(
            index=1,
            turn_id="delivery",
            question="请说明当前影响。",
            assertions={
                "completion_status": "partial",
                "required_response_obligations": ["immediate_impact"],
                "required_delivered_fields": ["local_impact"],
            },
            metadata={},
        )
        case = evaluator.EvaluationCase(
            index=0,
            case_id="delivery",
            group_id="delivery",
            variant_id="delivery",
            question=turn.question,
            assertions=turn.assertions,
            metadata={},
            turns=(turn,),
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
        manifest = [
            {
                "kind": "response_obligation",
                "obligation": "immediate_impact",
                "status": "not_machine_verified",
            },
            {"kind": "field_delivery", "field": "local_impact"},
        ]

        def request(method, url, *, payload=None, timeout):
            if method == "POST" and url.endswith("/api/session"):
                return {"id": "delivery-session"}
            if method == "DELETE" and url.endswith("/api/session/delivery-session"):
                return {"deleted": True}
            self.fail(f"unexpected request: {method} {url}")

        def stream(_url, _payload, *, timeout):
            return iter(
                [
                    {
                        "type": "answer",
                        "content": "当前只能确认局部影响。",
                        "unconfirmed_fields": ["final_impact"],
                        "answer_delivery_manifest": manifest,
                    },
                    {
                        "type": "done",
                        "completion_status": "partial",
                        "delivered_answer_fields": ["local_impact"],
                    },
                ]
            )

        with patch.object(evaluator, "_request_json", side_effect=request), patch.object(
            evaluator, "_stream_sse", side_effect=stream
        ):
            result = evaluator._run_case(case, config)

        turn_result = result["turns"][0]
        self.assertTrue(turn_result["passed"])
        self.assertEqual("partial", turn_result["completion_status"])
        self.assertEqual(["final_impact"], turn_result["unconfirmed_fields"])
        self.assertEqual(["local_impact"], turn_result["delivered_answer_fields"])
        self.assertEqual(manifest, turn_result["answer_delivery_manifest"])
        self.assertEqual(
            ["immediate_impact"],
            turn_result["answer_delivery"]["response_obligations"],
        )


class SemanticJudgeConfigurationTests(unittest.TestCase):
    def test_filtered_case_indexes_preserve_original_positions_without_crashing(self):
        cases = [evaluator.EvaluationCase(i, f"case-{i}", "g", "v", "问题", {}, {}) for i in (2, 8, 15)]
        args = evaluator.build_parser().parse_args(["cases.json", "--case-id", "case-8", "--case-id", "case-15"])
        def result(case, _config):
            return {"index": case.index, "case_id": case.case_id, "group_id": "g",
                    "passed": True, "assertion_results": [], "runtime_errors": []}
        with patch.object(evaluator, "_load_cases", return_value=(cases, {})), patch.object(
            evaluator, "_resolve_agent", return_value={"id": "agent"}
        ), patch.object(evaluator, "_resolve_judge_config", return_value=None), patch.object(
            evaluator, "_run_case", side_effect=result
        ):
            report, code = evaluator.run(args)
        self.assertEqual(0, code)
        self.assertEqual([8, 15], [case["index"] for case in report["cases"]])
        self.assertEqual(["case-8", "case-15"], report["input"]["selected_case_ids"])

    def test_case_filter_is_explicit_and_unknown_id_fails_before_service_calls(self):
        args = evaluator.build_parser().parse_args([
            "evals/icp_multiturn_dialogue_cases.json", "--case-id", "missing-case",
        ])
        self.assertEqual(["missing-case"], args.case_id)
        with patch.object(evaluator, "_resolve_agent") as agent, self.assertRaises(evaluator.EvaluationInputError):
            evaluator.run(args)
        agent.assert_not_called()

    def test_current_ollama_model_is_discovered_without_exposing_secret(self):
        args = evaluator.build_parser().parse_args(["cases.json"])
        payload = {
            "chat_model": {
                "model_id": "model-1",
                "model_type": "llm",
                "provider": "ollama",
                "model_name": "judge-model",
                "base_url": "http://ollama:11434",
                "api_key": "******",
            }
        }
        with patch.object(evaluator, "_request_json", return_value=payload) as request:
            config = evaluator._resolve_judge_config(
                "http://127.0.0.1:19999", args, required=True
            )

        self.assertEqual("ollama", config.provider)
        self.assertEqual("http://ollama:11434/api/chat", config.endpoint)
        self.assertEqual("", config.api_key)
        self.assertFalse(config.public_dict()["api_key_configured"])
        self.assertNotIn("api_key", config.public_dict())
        request.assert_called_once()

    def test_complete_openai_compatible_cli_override_needs_no_discovery(self):
        args = evaluator.build_parser().parse_args(
            [
                "cases.json",
                "--judge-provider",
                "openai-compatible",
                "--judge-model",
                "judge-model",
                "--judge-base-url",
                "http://model-server:8000/v1",
                "--judge-api-key",
                "secret",
            ]
        )
        with patch.object(evaluator, "_request_json") as request:
            config = evaluator._resolve_judge_config(
                "http://127.0.0.1:19999", args, required=True
            )

        self.assertEqual("openai_compatible", config.provider)
        self.assertEqual(
            "http://model-server:8000/v1/chat/completions", config.endpoint
        )
        self.assertTrue(config.public_dict()["api_key_configured"])
        request.assert_not_called()

    def test_judge_endpoint_rejects_embedded_secret_material(self):
        with self.assertRaisesRegex(evaluator.EvaluationInputError, "不得携带凭据"):
            evaluator._judge_api_endpoint(
                "openai_compatible",
                "",
                "https://user:secret@example.invalid/v1/chat/completions",
            )
        with self.assertRaisesRegex(evaluator.EvaluationInputError, "不得携带凭据"):
            evaluator._judge_api_endpoint(
                "openai_compatible",
                "",
                "https://example.invalid/v1/chat/completions?api_key=secret",
            )

    def test_openai_compatible_judge_requires_strict_json_content(self):
        config = evaluator.JudgeConfig(
            provider="openai_compatible",
            model_name="judge-model",
            endpoint="http://model-server/v1/chat/completions",
            api_key="",
            timeout_seconds=5.0,
            source="cli",
        )
        criteria = [{"id": "relevance", "criterion": "回答当前问题。"}]
        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"criteria":[{"id":"relevance","passed":true,'
                            '"reason":"直接回应。","checks":[{"aspect":"当前问题","requirement_kind":"positive",'
                            '"answer_quotes":["这是直接回应。"],"passed":true,"reason":"已经回答"}]}]}'
                        )
                    }
                }
            ]
        }
        with patch.object(evaluator, "_request_json", return_value=response) as request:
            decisions = evaluator.SemanticJudge(config).evaluate(
                question="请继续。",
                prior_turns=(),
                answer="这是直接回应。",
                criteria=criteria,
            )

        self.assertTrue(decisions[0]["passed"])
        sent = request.call_args.kwargs
        self.assertEqual(0, sent["payload"]["temperature"])
        self.assertNotIn("Authorization", sent["extra_headers"])

        invalid = {
            "choices": [
                {
                    "message": {
                        "content": "```json\n" + response["choices"][0]["message"]["content"] + "\n```"
                    }
                }
            ]
        }
        with patch.object(evaluator, "_request_json", return_value=invalid):
            with self.assertRaisesRegex(evaluator.ApiError, "严格 JSON"):
                evaluator.SemanticJudge(config).evaluate(
                    question="请继续。",
                    prior_turns=(),
                    answer="这是直接回应。",
                    criteria=criteria,
                )

    def test_ollama_judge_uses_json_mode_and_validates_response(self):
        config = evaluator.JudgeConfig(
            provider="ollama",
            model_name="judge-model",
            endpoint="http://ollama:11434/api/chat",
            api_key="",
            timeout_seconds=5.0,
            source="current_chat_model",
        )
        criteria = [{"id": "relevance", "criterion": "回答当前问题。"}]
        response = {
            "message": {
                "content": (
                    '{"criteria":[{"id":"relevance","passed":false,'
                    '"reason":"没有回应当前追问。","checks":[{"aspect":"当前追问","requirement_kind":"positive",'
                    '"answer_quotes":[],"passed":false,"reason":"未交付新增内容"}]}]}'
                )
            }
        }
        with patch.object(evaluator, "_request_json", return_value=response) as request:
            decisions = evaluator.SemanticJudge(config).evaluate(
                question="请继续。",
                prior_turns=(),
                answer="重复上一轮内容。",
                criteria=criteria,
            )

        self.assertFalse(decisions[0]["passed"])
        sent = request.call_args.kwargs["payload"]
        self.assertEqual("object", sent["format"]["type"])
        self.assertEqual(1, sent["format"]["properties"]["criteria"]["maxItems"])
        self.assertFalse(sent["think"])
        self.assertEqual(8192, sent["options"]["num_predict"])
        self.assertFalse(sent["stream"])


class EvaluationGateTests(unittest.TestCase):
    def test_breakdowns_do_not_report_judge_failure_as_answer_failure(self):
        breakdowns = evaluator._evaluation_breakdowns([{
            "metadata": {"category": "multi-turn"},
            "turns": [
                {"passed": True, "duration_seconds": 1, "semantic_judgement": {"status": "completed", "criteria": [{"passed": True}]}},
                {"passed": False, "duration_seconds": 3, "semantic_judgement": {"status": "completed", "criteria": [{"passed": False}]}},
                {"passed": False, "duration_seconds": 5, "semantic_judgement": {"status": "error"}},
                {"passed": False, "runtime_errors": [{"kind": "timeout"}]},
            ],
        }])
        overall = breakdowns["overall"]
        self.assertEqual(4, overall["total_turns"])
        self.assertEqual(2, overall["evaluable_turns"])
        self.assertEqual(0.5, overall["assertion_pass_rate"])
        self.assertEqual(1, overall["runtime_error_turns"])
        self.assertEqual(1, overall["judge_error_turns"])
        self.assertEqual(3, overall["latency_p50_seconds"])
        self.assertEqual(3, breakdowns["by_turn"]["followup"]["total_turns"])
        self.assertEqual(0.0, breakdowns["by_turn"]["followup"]["assertion_pass_rate"])

    """B2：CI 门槛参数（--gate / --gate-turn）的判定与解析。"""

    def test_gate_violation_none_when_above_threshold(self):
        self.assertIsNone(evaluator._gate_violation(0.90, 0.80, 0.95, 0.90))
        self.assertIsNone(evaluator._gate_violation(0.80, None, 0.90, None))

    def test_gate_violation_reports_case_or_turn_gate(self):
        message = evaluator._gate_violation(0.70, 0.80, 0.90, None)
        self.assertIn("用例通过率", message)
        self.assertIn("0.7000", message)
        self.assertIn("0.8000", message)
        turn_message = evaluator._gate_violation(0.90, 0.80, 0.60, 0.70)
        self.assertIn("轮次通过率", turn_message)

    def test_gate_threshold_accepts_bounded_and_rejects_range(self):
        self.assertEqual(0.85, evaluator._gate_threshold("0.85"))
        self.assertEqual(1.0, evaluator._gate_threshold("1"))
        with self.assertRaises(evaluator.argparse.ArgumentTypeError):
            evaluator._gate_threshold("1.2")
        with self.assertRaises(evaluator.argparse.ArgumentTypeError):
            evaluator._gate_threshold("-0.1")
        with self.assertRaises(evaluator.argparse.ArgumentTypeError):
            evaluator._gate_threshold("abc")

    def test_parser_exposes_gate_flags(self):
        parser = evaluator.build_parser()
        args = parser.parse_args(
            ["cases.json", "--gate", "0.9", "--gate-turn", "0.95"]
        )
        self.assertEqual(0.9, args.gate)
        self.assertEqual(0.95, args.gate_turn)
        bare = parser.parse_args(["cases.json"])
        self.assertIsNone(bare.gate)
        self.assertIsNone(bare.gate_turn)

    def test_run_uses_configured_threshold_instead_of_implicit_full_pass(self):
        cases = tuple(
            evaluator.EvaluationCase(
                index=index,
                case_id=f"case-{index}",
                group_id="gate",
                variant_id=f"variant-{index}",
                question="question",
                assertions={},
                metadata={},
                turns=(),
            )
            for index in range(2)
        )

        def case_result(case, _config):
            passed = case.index == 0
            return {
                "index": case.index,
                "case_id": case.case_id,
                "group_id": case.group_id,
                "variant_id": case.variant_id,
                "passed": passed,
                "cleanup": {},
                "tools": [],
                "evidence": {
                    "used_evidence_ids": [],
                    "catalog_evidence_ids": [],
                },
            }

        parser = evaluator.build_parser()
        with patch.object(
            evaluator, "_load_cases", return_value=(cases, {"name": "gate-test"})
        ), patch.object(
            evaluator, "_resolve_agent", return_value={"id": "agent-1"}
        ), patch.object(evaluator, "_run_case", side_effect=case_result):
            strict_report, strict_exit = evaluator.run(
                parser.parse_args(["ignored.json"])
            )
            passing_report, passing_exit = evaluator.run(
                parser.parse_args(["ignored.json", "--gate", "0.5"])
            )
            failing_report, failing_exit = evaluator.run(
                parser.parse_args(["ignored.json", "--gate", "0.75"])
            )

        self.assertEqual(0.5, strict_report["summary"]["pass_rate"])
        self.assertEqual(1, strict_exit)
        self.assertIsNone(passing_report["gate_failure"])
        self.assertEqual(0, passing_exit)
        self.assertIsNotNone(failing_report["gate_failure"])
        self.assertEqual(1, failing_exit)


if __name__ == "__main__":
    unittest.main()
