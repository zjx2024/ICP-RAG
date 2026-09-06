import asyncio
import unittest
from unittest.mock import patch

from backend.agent.nodes.answer import _context_evidence
from backend.agent.nodes.answer import _with_answer_evidence_rerank


def _evidence(evidence_id: str, title: str, content: str) -> dict:
    return {
        "evidence_id": evidence_id,
        "doc_type": "fmeca",
        "record_id": evidence_id,
        "title": title,
        "content": content,
        "authority": "authoritative_record",
        "evidence_level": "L1_DIRECT",
    }


class AnswerEvidenceRerankTests(unittest.TestCase):
    def test_answer_window_keeps_only_explicitly_linked_fmeca_mfl_family(self):
        primary = _evidence(
            "e-fm-002",
            "输入滤波性能退化",
            "输入纹波和传导干扰增大，多个子系统出现间歇性异常。",
        )
        primary["match_type"] = "semantic_id_hydration"
        primary["answer_rerank_score"] = 0.65
        primary["metadata"] = {
            "answer_fields": {
                "record_code": "FM-SYN-002",
                "product_flag": "ICU / FUNC-110",
                "failure_mode": "输入滤波性能退化",
                "explicit_references": {"mfl": ["MFL-SYN-002"]},
            }
        }
        unrelated = _evidence(
            "e-fm-011",
            "过流保护误跳闸",
            "正常负载被隔离，局部功能中断并可能反复重启。",
        )
        unrelated["match_type"] = "semantic_id_hydration"
        unrelated["answer_rerank_score"] = 0.003
        unrelated["metadata"] = {
            "answer_fields": {
                "record_code": "FM-SYN-011",
                "product_flag": "PDU / FUNC-130",
                "failure_mode": "过流保护误跳闸",
            }
        }
        linked_mfl = _evidence(
            "e-mfl-002",
            "MFL-SYN-002",
            "检查连接、接地和滤波器件。",
        )
        linked_mfl["doc_type"] = "mfl"
        linked_mfl["record_id"] = "MFL-SYN-002"
        linked_mfl["claim_type"] = "maintenance_locator"
        linked_mfl["metadata"] = {
            "answer_fields": {
                "record_code": "MFL-SYN-002",
                "module_failure": "输入调理模块：输入滤波性能退化",
            }
        }
        unrelated_mfl = _evidence(
            "e-mfl-011",
            "MFL-SYN-011",
            "检查保护门限和跳闸前波形。",
        )
        unrelated_mfl["doc_type"] = "mfl"
        unrelated_mfl["record_id"] = "MFL-SYN-011"
        unrelated_mfl["claim_type"] = "maintenance_locator"
        unrelated_mfl["metadata"] = {
            "answer_fields": {
                "record_code": "MFL-SYN-011",
                "module_failure": "配电模块：过流保护误跳闸",
            }
        }
        evidence = [primary, unrelated, linked_mfl, unrelated_mfl]
        ids = [item["evidence_id"] for item in evidence]
        state = {
            "query": "输入端滤波元件退化后为什么多处零星异常？",
            "intent": "impact_analysis",
            "query_targets": [{"intent": "impact_analysis"}],
            "evidence_registry": evidence,
            "context_evidence_ids": ids,
            "safety_decision": {
                "approved_evidence_ids": ids,
                "context_evidence_ids": ids,
            },
        }

        selected = _context_evidence(state)

        self.assertEqual(
            {"e-fm-002", "e-mfl-002"},
            {item["evidence_id"] for item in selected},
        )

    def test_rerank_orders_answer_window_without_dropping_audit_evidence(self):
        state = {
            "agent_id": "agent-1",
            "query": "外部设备收不到本机发出的业务数据",
            "intent": "cause_analysis",
            "evidence_registry": [
                _evidence("e-display", "显示输出", "外部显示器无画面"),
                _evidence("e-fan", "风扇接口", "风扇状态接口异常"),
                _evidence("e-network", "网络发送", "对外业务数据发送中断"),
                _evidence("e-history", "历史处置", "调整网络队列后恢复发送"),
            ],
            "safety_decision": {
                "approved_evidence_ids": [
                    "e-display",
                    "e-fan",
                    "e-network",
                    "e-history",
                ],
            },
        }
        ranked = [(2, 0.96), (3, 0.91), (0, 0.12), (1, 0.03)]
        with patch(
            "backend.agent.nodes.answer.agent_model_role_service.resolve_rerank_model",
            return_value=({"model_name": "local-reranker"}, "agent_role"),
        ), patch(
            "backend.agent.nodes.answer.rerank_service.rerank",
            return_value=ranked,
        ):
            updated = asyncio.run(_with_answer_evidence_rerank(state))

        self.assertEqual(4, len(updated["evidence_registry"]))
        self.assertEqual("local-reranker", updated["answer_evidence_rerank"]["model"])
        selected = _context_evidence(updated)
        self.assertEqual(
            ["e-network", "e-history", "e-display", "e-fan"],
            [item["evidence_id"] for item in selected],
        )

    def test_strong_history_exact_ids_narrow_single_fault_family(self):
        target = _evidence(
            "e-fm-038", "突发丢包", "高负载下缓存不足导致成片丢包。"
        )
        target["record_id"] = "FM-SYN-038"
        target["match_type"] = "semantic_id_hydration"
        target["answer_rerank_score"] = 0.72
        target["metadata"] = {
            "answer_fields": {
                "record_code": "FM-SYN-038",
                "product_flag": "NGW / FUNC-320",
                "failure_mode": "突发丢包",
            }
        }
        neighbour = _evidence(
            "e-fm-039", "端口链路抖动", "连接器接触不良导致链路抖动。"
        )
        neighbour["record_id"] = "FM-SYN-039"
        neighbour["match_type"] = "semantic_id_hydration"
        neighbour["answer_rerank_score"] = 0.69
        neighbour["metadata"] = {
            "answer_fields": {
                "record_code": "FM-SYN-039",
                "product_flag": "NGW / FUNC-320",
                "failure_mode": "端口链路抖动",
            }
        }

        def mfl(evidence_id, record_id, mode):
            item = _evidence(evidence_id, record_id, f"模块故障模式：{mode}")
            item["doc_type"] = "mfl"
            item["record_id"] = record_id
            item["match_type"] = "semantic_id_hydration"
            item["metadata"] = {
                "answer_fields": {
                    "record_code": record_id,
                    "module_failure": f"网络网关模块：{mode}",
                }
            }
            return item

        target_mfl = mfl("e-mfl-038", "MFL-SYN-038", "突发丢包")
        neighbour_mfl = mfl("e-mfl-039", "MFL-SYN-039", "端口链路抖动")
        history = _evidence(
            "e-history-013", "HIS-SYN-013", "高负载下丢包，恢复队列配置后正常。"
        )
        history["doc_type"] = "fault_history"
        history["record_id"] = "HIS-SYN-013"
        history["match_type"] = "semantic_id_hydration"
        history["retrieval_score"] = 0.9
        history["metadata"] = {
            "answer_fields": {
                "explicit_references": {
                    "fmeca": ["FM-SYN-038"],
                    "mfl": ["MFL-SYN-038"],
                }
            }
        }
        evidence = [target, neighbour, target_mfl, neighbour_mfl, history]
        ids = [item["evidence_id"] for item in evidence]
        state = {
            "query": "网络一忙就成批丢包，负载下降后恢复",
            "intent": "cause_analysis",
            "query_targets": [{"intent": "cause_analysis"}],
            "evidence_registry": evidence,
            "context_evidence_ids": ids,
            "safety_decision": {
                "approved_evidence_ids": ids,
                "context_evidence_ids": ids,
            },
        }

        selected = _context_evidence(state)

        self.assertEqual(
            {"e-fm-038", "e-mfl-038", "e-history-013"},
            {item["evidence_id"] for item in selected},
        )

    def test_query_direction_words_do_not_delete_approved_history(self):
        current = _evidence(
            "e-egress",
            "对外发送链路",
            "内部报文已经产生，但外部设备无法接收。",
        )
        opposite = _evidence(
            "e-ingress-history",
            "A 通道历史",
            "外部设备持续发送数据，但系统 A 通道接收计数长期不变。",
        )
        state = {
            "query": "内部已经产生报文，但外接设备一直收不到",
            "intent": "cause_analysis",
            "evidence_registry": [current, opposite],
            "context_evidence_ids": [current["evidence_id"], opposite["evidence_id"]],
            "safety_decision": {
                "approved_evidence_ids": [current["evidence_id"], opposite["evidence_id"]],
                "context_evidence_ids": [current["evidence_id"], opposite["evidence_id"]],
            },
        }

        self.assertEqual(
            ["e-egress", "e-ingress-history"],
            [item["evidence_id"] for item in _context_evidence(state)],
        )

    def test_outbound_wording_alone_does_not_delete_receive_mode(self):
        outbound = _evidence(
            "e-egress",
            "A通道发送中断",
            "A通道对外业务数据无法送达，系统依赖B通道。",
        )
        inbound = _evidence(
            "e-ingress",
            "A通道接收超时",
            "A通道总线接收能力丧失，部分外部输入数据不可用。",
        )
        state = {
            "query": "内部数据已经正常生成，但A通道对外发送链路断了，外部设备还能收到吗？",
            "intent": "impact_analysis",
            "evidence_registry": [outbound, inbound],
            "context_evidence_ids": [outbound["evidence_id"], inbound["evidence_id"]],
            "safety_decision": {
                "approved_evidence_ids": [outbound["evidence_id"], inbound["evidence_id"]],
                "context_evidence_ids": [outbound["evidence_id"], inbound["evidence_id"]],
            },
        }

        self.assertEqual(
            ["e-egress", "e-ingress"],
            [item["evidence_id"] for item in _context_evidence(state)],
        )

    def test_display_question_keeps_display_specific_evidence(self):
        display = _evidence("e-display", "显示输出", "外部显示设备无画面")
        state = {
            "query": "外接屏没有画面，显示输出哪里可能故障？",
            "evidence_registry": [display],
        }

        self.assertEqual([display], _context_evidence(state))

    def test_api_wording_display_output_queue_is_not_display_domain(self):
        history = _evidence(
            "e-network-history",
            "网络突发丢包历史",
            "GetNGWFaultStatus 显示输出队列占用率 100%，丢包集中在突发窗口。",
        )
        state = {
            "query": "网络一忙就成批丢包，负载下降后恢复",
            "evidence_registry": [history],
        }

        self.assertEqual([history], _context_evidence(state))

    def test_rerank_failure_preserves_original_state(self):
        state = {
            "agent_id": "agent-1",
            "query": "设备异常",
            "evidence_registry": [
                _evidence(f"e-{index}", f"证据{index}", f"内容{index}")
                for index in range(4)
            ],
        }
        with patch(
            "backend.agent.nodes.answer.agent_model_role_service.resolve_rerank_model",
            return_value=({"model_name": "local-reranker"}, "agent_role"),
        ), patch(
            "backend.agent.nodes.answer.rerank_service.rerank",
            side_effect=RuntimeError("offline"),
        ):
            updated = asyncio.run(_with_answer_evidence_rerank(state))

        self.assertIs(state, updated)
        self.assertNotIn("answer_evidence_rerank", updated)


if __name__ == "__main__":
    unittest.main()
