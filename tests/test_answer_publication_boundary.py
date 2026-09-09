"""Approved source fields must not become unchecked post-review statements."""
import asyncio
import copy
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

from backend.agent.nodes import answer
from backend.agent.contracts import make_evidence


class PublicationBoundaryTests(unittest.TestCase):
    def test_final_text_is_reviewed_text_even_when_mfl_fields_are_unconsumed(self):
        for wording in ("不可直接上报，但支持主动查询。", "状态由调用方读取，不能自动推送。", "未确认该报码与维修清单的直接对应。"):
            item = make_evidence(source_kind="postgres", authority="authoritative_record",
                evidence_level="L1_DIRECT", claim_type="mfl_fact", doc_type="mfl",
                doc_id="document-71", record_id="record-39", title="状态查询",
                content="主动查询接口", metadata={"answer_fields": {
                    "record_code": "record-39", "module_failure": "装置：反馈异常",
                    "api_reportable": False, "api_name": "Read_Diagnostic_Status",
                    "api_method": "调用方主动查询", "maintenance_action": "确认后检查连接"}})
            state = {"query": "状态从哪里读取？", "evidence_registry": [item],
                "context_evidence_ids": [item["evidence_id"]],
                "query_frame": {"required_answer_fields": ["maintenance_api", "maintenance_action"]},
                "safety_decision": {"approved_evidence_ids": [item["evidence_id"]]}}
            with self.subTest(wording=wording), ExitStack() as stack:
                stack.enter_context(patch.object(answer, "_with_answer_evidence_rerank", new=AsyncMock(side_effect=lambda x: x)))
                stack.enter_context(patch.object(answer.agent_model_role_service, "apply_agent_llm_params", return_value={}))
                stack.enter_context(patch.object(answer, "_generate_answer_response", new=AsyncMock(return_value=wording)))
                review = stack.enter_context(patch.object(answer, "_review_answer", new=AsyncMock(return_value={
                    "text": wording, "status": "partial", "unconfirmed_goals": [state["query"]], "manifest": []})))
                result = asyncio.run(answer.answer_node(copy.deepcopy(state)))
                self.assertEqual(wording, result["final_answer"])
                self.assertEqual(1, review.await_count)
                self.assertNotIn("没有可用的状态接口", result["final_answer"])
                self.assertNotIn("确认后检查连接", result["final_answer"])


if __name__ == "__main__":
    unittest.main()
