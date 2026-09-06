"""Generic scoped rereads and conversation progress; no evaluated case IDs."""
import asyncio
import unittest
from unittest.mock import MagicMock, patch

from backend.agent.tools import icp_diagnosis as diagnosis
from backend.agent.nodes.plan import _selected_anchor_delivery_contracts, _offered_conversation_anchors
from backend.agent.answer_context_binding import conversation_review_context
from backend.agent.evidence_packets import query_coverage
from backend.agent.history_focus import select_history_parent_cases
from backend.icp.repository import IcpRepository


class BusinessSelectorTests(unittest.TestCase):
    def read(self, doc_type, selector, rows):
        conn = MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        columns = list(rows[0]) if rows else ["id"]
        cursor.description = [(key,) for key in columns]
        if doc_type == "fault_history" and len(rows) == 1:
            cursor.fetchall.side_effect = [[tuple(row[key] for key in columns) for row in rows], [(1, "initial"), (2, "verify")]]
        else:
            cursor.fetchall.return_value = [tuple(row[key] for key in columns) for row in rows]
        with patch("backend.icp.repository.postgres_client.conn", conn):
            result = IcpRepository().get_records_by_business_selectors(doc_type, [selector])
        sql, params = cursor.execute.call_args_list[0].args
        self.assertEqual(sql.count("%s"), len(params))
        self.assertNotIn("ILIKE", sql)
        self.assertIn("LIMIT 2", sql)
        self.assertIn("d.status IN ('indexed','kg_pending','kg_done')", sql)
        return result, sql, params

    def test_all_row_families_use_document_scoped_bound_parameters(self):
        for doc_type, column in [("fmeca", "code"), ("mfl", "mfl_code"), ("fault_history", "case_id")]:
            # Includes quotes, percent and non-code-like IDs: the DB contract
            # is equality and scope, not a vocabulary of synthetic prefixes.
            for record in ["entry-581", "现场记录甲", "x' OR 1=1 -- %"]:
                with self.subTest(doc_type=doc_type, record=record):
                    result, sql, params = self.read(doc_type, {"doc_id": "revision-document", "record_id": record},
                                                    [{"id": 73, "doc_id": "revision-document", column: record}])
                    self.assertEqual(("revision-document", record), params)
                    self.assertNotIn(record, sql)
                    self.assertEqual("found", result["reads"][0]["status"])
                    self.assertEqual("exact_conversation_reference", result["rows"][0]["match_type"])
                    self.assertFalse(result["rows"][0]["record_read"]["original_document_read"])
                    if doc_type == "fault_history":
                        self.assertEqual(["initial", "verify"], result["rows"][0]["steps"])

    def test_empty_and_ambiguous_are_not_unique_or_source_wide_absence(self):
        for rows, status in [([], "not_found"), ([{"id": 1}, {"id": 2}], "ambiguous")]:
            result, _, _ = self.read("mfl", {"doc_id": "d", "record_id": "entry"}, rows)
            self.assertEqual([], result["rows"])
            self.assertEqual(status, result["reads"][0]["status"])
            self.assertEqual("selected_identity_in_active_document", result["reads"][0]["scope"])

    def test_hmc_variant_and_complete_components_keep_zero_and_wildcard(self):
        selectors = [{"doc_id": "d", "variant_id": "v-05"},
                     {"doc_id": "d", "lcn": "00456789", "hamc": "**", "hsc": "0005", "source_ref": "row-02"}]
        for selector in selectors:
            result, _, params = self.read("hmc", selector, [{"id": 1}])
            self.assertEqual(tuple(selector.values()), params)
            self.assertEqual(selector, result["rows"][0]["hmc_anchor_selector"])

    def test_invalid_and_unbounded_selectors_are_rejected(self):
        conn = MagicMock()
        with patch("backend.icp.repository.postgres_client.conn", conn):
            for doc_type, selectors in [("fta", []), ("mfl", [{}]), ("hmc", [{"doc_id": "d", "lcn": "00456789", "hsc": "0005"}]),
                                        ("mfl", [{"doc_id": "d", "record_id": "x"}] * 13)]:
                with self.subTest(doc_type=doc_type), self.assertRaises(ValueError):
                    IcpRepository().get_records_by_business_selectors(doc_type, selectors)
        conn.cursor.return_value.__enter__.return_value.execute.assert_not_called()

    def test_sql_error_propagates_instead_of_becoming_empty(self):
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value.execute.side_effect = RuntimeError("database unavailable")
        with patch("backend.icp.repository.postgres_client.conn", conn), self.assertRaisesRegex(RuntimeError, "unavailable"):
            IcpRepository().get_records_by_business_selectors("mfl", [{"doc_id": "d", "record_id": "x"}])


class ConversationRereadTests(unittest.TestCase):
    @staticmethod
    def state(doc_type="fault_history", record="CASE-073"):
        tool = {"fault_history": "fault_history_query", "mfl": "mfl_query", "fmeca": "fmeca_query"}[doc_type]
        return {"query": "它是怎么得到这个结论的？", "allowed_tools": [tool],
                "dialogue_context": {"status": "resolved", "context_used": True},
                "selected_conversation_anchors": [{"doc_type": doc_type, "doc_id": "older-document", "record_id": record}]}

    def test_history_and_mfl_node_reread_instead_of_newest_keyword_neighbors(self):
        for doc_type, record, column in [("fault_history", "CASE-073", "case_id"), ("mfl", "ACTION-022", "mfl_code")]:
            state = self.state(doc_type, record)
            row = {"id": 11, "doc_id": "older-document", column: record,
                   "fault_symptom": "observed", "module_fault": "observed", "match_type": "exact_conversation_reference"}
            read = {"selector": {"doc_id": "older-document", "record_id": record}, "status": "found"}
            with patch.object(diagnosis.icp_repository, "get_records_by_business_selectors", return_value={"rows": [row], "reads": [read]}) as exact, \
                 patch.object(diagnosis.icp_repository, "search_history") as broad_history, \
                 patch.object(diagnosis, "_query_mfl") as broad_mfl, \
                 patch.object(diagnosis, "_enrich_function_sources", side_effect=lambda rows, errors=None: list(rows)):
                result = asyncio.run(diagnosis.icp_diagnosis_node(state))
            exact.assert_called_once_with(doc_type, [{"doc_id": "older-document", "record_id": record}])
            broad_history.assert_not_called()
            broad_mfl.assert_not_called()
            self.assertIn(record, {item["record_id"] for item in result["evidence_registry"]})
            coverage = query_coverage(result["tool_results"])
            self.assertIn([read], [item.get("anchor_reads") for item in coverage])

    def test_empty_anchor_reread_does_not_fall_back_to_a_neighbor(self):
        state = self.state()
        with patch.object(diagnosis.icp_repository, "get_records_by_business_selectors", return_value={"rows": [], "reads": [{"status": "not_found"}]}), \
             patch.object(diagnosis.icp_repository, "search_history") as broad:
            result = asyncio.run(diagnosis.icp_diagnosis_node(state))
        broad.assert_not_called()
        self.assertFalse(result["evidence_registry"])
        self.assertEqual("not_found", result["tool_results"][0]["data"]["anchor_reads"][0]["status"])

    def test_current_correction_and_unrelated_turn_do_not_reread_old_identity(self):
        for change in [{"query": "Check CASE-991 instead"},
                       {"dialogue_context": {"status": "standalone", "context_used": False}}]:
            self.assertEqual([], diagnosis._conversation_record_selectors({**self.state(), **change}, "fault_history"))

    def test_collection_is_not_clamped_to_previous_member(self):
        state = self.state("fmeca", "MODE-073")
        state["query_frame"] = {"coverage_scope": "complete_collection"}
        self.assertEqual([], diagnosis._conversation_record_selectors(state, "fmeca"))

    def test_scoped_identity_dedup_preserves_comparison_documents(self):
        state = self.state()
        anchor = state["selected_conversation_anchors"][0]
        state["selected_conversation_anchors"] = [anchor, dict(anchor), {**anchor, "doc_id": "other-revision"}]
        selectors = diagnosis._conversation_record_selectors(state, "fault_history")
        self.assertEqual(2, len(selectors))
        self.assertEqual({"older-document", "other-revision"}, {s["doc_id"] for s in selectors})

    def test_unscoped_citation_cannot_mint_exact_authority(self):
        state = self.state()
        state["selected_conversation_anchors"][0].pop("doc_id")
        self.assertEqual([], diagnosis._conversation_record_selectors(state, "fault_history"))

    def test_history_parent_does_not_link_same_code_from_another_document(self):
        state = self.state()
        for rows in [[{"case_id": "CASE-073", "doc_id": "different"}],
                     [{"case_id": "CASE-073", "doc_id": doc} for doc in ["older-document", "different"]]]:
            selection = select_history_parent_cases(state, rows)
            self.assertTrue(selection.conflict)
            self.assertFalse(selection.authorizes_exact_links)


class GoalProgressTests(unittest.TestCase):
    def test_anchor_free_turns_do_not_evict_an_available_earlier_identity(self):
        anchor = {"anchor_id": "earlier", "doc_id": "d", "record_id": "MODE-504"}
        messages = [{"role": "assistant", "id": "origin", "conversation_anchors": [anchor]}]
        messages += [{"role": "assistant", "content": "范围说明"} for _ in range(9)]
        offered = _offered_conversation_anchors(messages)
        self.assertEqual("earlier", offered[0]["anchor_id"])
        self.assertEqual("origin", offered[0]["origin_message_id"])

    def test_anchor_budget_still_bounds_recent_identities_without_old_prose(self):
        messages = [{"role": "assistant", "content": "not an anchor",
                     "conversation_anchors": [{"anchor_id": str(i), "record_id": str(i)}]} for i in range(30)]
        offered = _offered_conversation_anchors(messages, limit=12)
        self.assertEqual([str(i) for i in range(29, 17, -1)], [a["anchor_id"] for a in offered])
        self.assertTrue(all("content" not in a for a in offered))

    def test_progress_is_turn_and_evidence_scoped_including_unresolved_goals(self):
        entries = [{"kind": "goal_delivery", "goal": "why it matters", "status": "answered", "evidence_ids": ["e1"]},
                   {"kind": "goal_delivery", "goal": "verification standard", "status": "unanswered", "evidence_ids": []},
                   {"kind": "goal_delivery", "goal": "other object", "status": "answered", "evidence_ids": ["e2"]}]
        messages = [{"role": "assistant", "id": "turn-old", "answer_delivery_manifest": entries},
                    {"role": "assistant", "id": "turn-selected", "answer_delivery_manifest": entries}]
        selected = [{"origin_message_id": "turn-selected", "evidence_id": "e1"}]
        progress = _selected_anchor_delivery_contracts(messages, {"context_used": True}, selected)
        self.assertEqual(["why it matters", "verification standard"], [p["goal"] for p in progress])
        self.assertEqual({"turn-selected"}, {p["origin_message_id"] for p in progress})
        self.assertEqual([], _selected_anchor_delivery_contracts(messages, {"context_used": False}, selected))

    def test_answer_context_preserves_prior_text_and_delivery_without_fact_authority(self):
        state = {"query": "explain that further", "messages": [{"role": "assistant", "content": "Earlier text [^8]",
                  "answer_delivery_manifest": [{"kind": "goal_delivery", "goal": "impact", "status": "partial", "evidence_ids": ["e1"]}]}]}
        context = conversation_review_context(state)
        self.assertEqual("Earlier text [^8]", context["messages"][0]["content"])
        self.assertEqual("partial", context["messages"][0]["prior_delivery"][0]["status"])
        self.assertEqual(state["query"], context["query"])


if __name__ == "__main__":
    unittest.main()
