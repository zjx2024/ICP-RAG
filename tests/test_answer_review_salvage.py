# v3/v4 validator assertions below are offline compatibility; live v5 counterparts are in test_answer_integrated_review.
"""Final-attempt salvage: prune critic-rejected units, publish partial.

Contracts only, no model quality.  These tests pin the structural protections:
protocol failures never salvage, pruned shells reject, clean reviews are
unaffected, and pruned publications demote every goal to unconfirmed.
"""
import asyncio
import copy
import json
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

from backend.agent import answer_observation
from scripts import legacy_answer_review as protocol
from backend.agent.contracts import make_evidence
from backend.agent.nodes import answer
from backend.agent.nodes.answer_request import _answer_task_context

LINE1 = "输出更新时间滞后，最先受影响的是显示刷新。"
LINE2 = "这与相邻模块的共享总线负载有关。"
DRAFT = LINE1 + "\n" + LINE2
SOURCE = "e1:/answer_fields/local_impact"


def state_for(doc_type="fmeca", **changes):
    item = make_evidence(source_kind="postgres", authority="authoritative_record", evidence_level="L1_DIRECT",
        claim_type=doc_type + "_fact", doc_type=doc_type, doc_id="document-57", record_id="ROW-731",
        title="本轮对象", content=LINE1, metadata={"answer_fields": {"local_impact": LINE1}})
    item.update(changes)
    return {"query": "为什么它重要？最先受影响的是什么？", "messages": [],
        "evidence_registry": [item], "context_evidence_ids": [item["evidence_id"]],
        "safety_decision": {"approved_evidence_ids": [item["evidence_id"]], "can_claim_unique": False}}


def check(unit_id, *, sources=None, claims=None, issue=""):
    return {"unit_id": unit_id, "sources": sources or ([SOURCE] if not issue else []),
            "claims": claims or ([] if issue else ["source_fact"]),
            "attributed_doc_types": [], "issue": issue}


def clean_unit(unit_id, text):
    return {"unit_id": unit_id, "sources": [SOURCE], "claims": ["source_fact"],
             "attributed_doc_types": [], "issue": ""}


def response(units, goals=None):
    return {"version": 3, "units": units,
            "goals": [{**g, "unit_ids": g.get("unit_ids", [u["unit_id"] for u in units])} for g in goals] if goals is not None
            else [{"goal_id": "q0", "status": "answered", "unit_ids": [u["unit_id"] for u in units], "issue": ""}]}


def prepared(state, draft=DRAFT):
    ledger, conversation, observations = answer._answer_review_material(state)
    material = protocol.prepare_review(draft, ledger=ledger, conversation=conversation,
        observations=observations, task=_answer_task_context(state),
        document_semantics=answer.document_semantics_prompt())
    return material, ledger


def validate(value, state=None, draft=DRAFT, salvage=False):
    state = state or state_for()
    material, ledger = prepared(state, draft)
    return protocol.validate_review(
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
        prepared=material, ledger=ledger, safety=state.get("safety_decision") or {},
        calculation_requests=state.get("calculation_requests") or [], salvage=salvage)


class SalvageValidationTests(unittest.TestCase):
    def test_poison_unit_pruned_publishes_partial(self):
        review = response([clean_unit("u1", LINE1), check("u2", issue="该段断言无来源支持")])
        result = validate(review, salvage=True)
        self.assertTrue(result["salvaged"])
        self.assertEqual(result["pruned_units"], ["u2"])
        self.assertNotIn(LINE2, result["text"])
        self.assertIn(LINE1, result["text"])
        self.assertEqual(result["status"], "partial")

    def test_salvage_off_still_rejects_whole_draft(self):
        review = response([clean_unit("u1", LINE1), check("u2", issue="该段断言无来源支持")])
        with self.assertRaises(protocol.AnswerReviewError):
            validate(review, salvage=False)

    def test_all_units_poisoned_still_rejects(self):
        review = response([check("u1", issue="无支持"), check("u2", issue="无支持")])
        with self.assertRaises(protocol.AnswerReviewError):
            validate(review, salvage=True)

    def test_protocol_failure_never_salvages(self):
        with self.assertRaises(protocol.AnswerReviewError):
            validate("not json", salvage=True)
        incomplete = {"version": 3, "units": [clean_unit("u1", LINE1)], "goals": [
            {"goal_id": "q0", "status": "answered", "issue": ""}]}
        with self.assertRaises(protocol.AnswerReviewError):
            validate(incomplete, salvage=True)

    def test_blocked_claim_unit_is_pruned_not_published(self):
        review = response([
            clean_unit("u1", LINE1),
            {"unit_id": "u2", "sources": [SOURCE], "claims": ["unique_diagnosis"],
             "attributed_doc_types": [], "issue": ""},
        ])
        result = validate(review, salvage=True)
        self.assertEqual(result["pruned_units"], ["u2"])
        self.assertNotIn(LINE2, result["text"])
        self.assertNotIn("unique", result["text"])

    def test_no_goal_answered_still_rejects(self):
        review = response([clean_unit("u1", LINE1)],
                          goals=[{"goal_id": "q0", "status": "unanswered", "issue": "资料足够但未回答"}])
        with self.assertRaises(protocol.AnswerReviewError):
            validate(review, salvage=True)

    def test_pruned_publication_demotes_all_goals(self):
        state = state_for()
        review = response([clean_unit("u1", LINE1), check("u2", issue="无支持")])
        result = validate(review, state=state, salvage=True)
        self.assertEqual(result["status"], "partial")
        self.assertIn(state["query"], result["unconfirmed_goals"])
        self.assertTrue(all(m["status"] != "answered" for m in result["manifest"]))

    def test_goal_issue_only_publishes_partial_without_pruning(self):
        review = response([clean_unit("u1", LINE1), clean_unit("u2", LINE2)],
                          goals=[{"goal_id": "q0", "status": "answered", "issue": ""}])
        # q0 answered but with a residual issue note: no unit to prune.
        review["goals"][0]["issue"] = "次要维度未展开"
        result = validate(review, salvage=True)
        self.assertTrue(result["salvaged"])
        self.assertEqual(result["pruned_units"], [])
        self.assertIn(LINE1, result["text"])
        self.assertIn(LINE2, result["text"])
        self.assertEqual(result["status"], "partial")

    def test_clean_review_unaffected_by_salvage_flag(self):
        review = response([clean_unit("u1", LINE1), clean_unit("u2", LINE2)])
        strict = validate(review, salvage=False)
        salvage = validate(review, salvage=True)
        self.assertFalse(strict["salvaged"])
        self.assertFalse(salvage["salvaged"])
        self.assertEqual(strict["text"], salvage["text"])
        self.assertEqual(strict["status"], "full")
        self.assertEqual(salvage["status"], "full")

    def test_heading_shell_surviving_rejects(self):
        heading = "## 处理建议"
        draft = heading + "\n" + LINE2
        review = response([
            {"unit_id": "u1", "sources": [], "claims": ["presentation"],
             "attributed_doc_types": [], "issue": ""},
            check("u2", issue="无支持"),
        ])
        with self.assertRaises(protocol.AnswerReviewError):
            validate(review, draft=draft, salvage=True)

    def test_surviving_citations_rerendered_after_pruning(self):
        review = response([clean_unit("u1", LINE1), check("u2", issue="无支持")])
        result = validate(review, salvage=True)
        # The surviving unit keeps a rendered citation marker bound to its record.
        self.assertIn("[^1]", result["text"])
        self.assertTrue(result["text"].strip().endswith("[^1]"))
