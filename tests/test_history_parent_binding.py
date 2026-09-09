import unittest

from backend.agent.history_focus import (
    history_parent_binding,
    history_parent_selection_from_binding,
    merge_history_parent_bindings,
    resolve_history_parent_cases,
    select_history_parent_cases,
)


class HistoryParentBindingTests(unittest.TestCase):
    def test_explicit_case_id_authorizes_only_that_parent(self):
        rows = [
            {"id": 1, "case_id": "CASE-A", "match_type": "structured_query"},
            {"id": 2, "case_id": "CASE-B", "match_type": "semantic_id_hydration"},
        ]
        state = {
            "query": "核对 CASE-A 的恢复记录",
            "query_targets": [{"intent": "history_query"}],
            "query_frame": {"original_query": "核对 CASE-A 的恢复记录"},
        }

        selection = select_history_parent_cases(state, rows)

        self.assertEqual(frozenset({"case-a"}), selection.selected_case_ids)
        self.assertTrue(selection.authorizes_exact_links)
        self.assertEqual("identity", history_parent_binding(selection, rows)["authority"])

    def test_current_explicit_case_id_outvotes_an_older_history_anchor(self):
        rows = [
            {"id": 1, "case_id": "CASE-OLD", "match_type": "exact_conversation_reference"},
            {"id": 2, "case_id": "INCIDENT-NEW", "match_type": "structured_query"},
        ]
        state = {
            "query": "核对 INCIDENT-NEW 的恢复记录",
            "query_targets": [{"intent": "history_query"}],
            "query_frame": {"original_query": "核对 INCIDENT-NEW 的恢复记录"},
            "selected_conversation_anchors": [
                {"doc_type": "fault_history", "record_id": "CASE-OLD"}
            ],
        }

        selection = select_history_parent_cases(state, rows)

        self.assertEqual(
            frozenset({"incident-new"}), selection.selected_case_ids
        )
        self.assertEqual("current_query_id", selection.source)
        self.assertTrue(selection.authorizes_exact_links)

    def test_unresolved_current_id_in_another_namespace_blocks_old_anchor(self):
        rows = [
            {"id": 1, "case_id": "CASE-OLD", "match_type": "exact_conversation_reference"}
        ]
        state = {
            "query": "核对 INCIDENT-NEW 的恢复记录",
            "query_targets": [{"intent": "history_query"}],
            "query_frame": {"original_query": "核对 INCIDENT-NEW 的恢复记录"},
            "selected_conversation_anchors": [
                {"doc_type": "fault_history", "record_id": "CASE-OLD"}
            ],
        }

        selection = select_history_parent_cases(state, rows)

        self.assertEqual(frozenset(), selection.selected_case_ids)
        self.assertEqual("current_query_unresolved_id", selection.source)
        self.assertTrue(selection.conflict)
        self.assertFalse(selection.authorizes_exact_links)

    def test_selected_nonhistory_id_does_not_masquerade_as_history_conflict(self):
        rows = [
            {"id": 1, "case_id": "CASE-OLD", "match_type": "exact_conversation_reference"}
        ]
        state = {
            "query": "针对 FM-GEN-302 再看这个历史案例",
            "query_targets": [{"intent": "history_query"}],
            "query_frame": {
                "original_query": "针对 FM-GEN-302 再看这个历史案例"
            },
            "selected_conversation_anchors": [
                {"doc_type": "fault_history", "record_id": "CASE-OLD"},
                {"doc_type": "fmeca", "record_id": "FM-GEN-302"},
            ],
        }

        selection = select_history_parent_cases(state, rows)

        self.assertEqual(frozenset({"case-old"}), selection.selected_case_ids)
        self.assertEqual("conversation_anchor", selection.source)
        self.assertTrue(selection.authorizes_exact_links)

    def test_semantic_winner_is_candidate_not_exact_identity(self):
        rows = [
            {
                "id": 11,
                "case_id": "CASE-RX",
                "match_type": "semantic_id_hydration",
                "retrieval_match_score": 0.95,
                "fault_symptom": "输入接收计数停止",
            },
            {
                "id": 12,
                "case_id": "CASE-DSP",
                "match_type": "semantic_id_hydration",
                "retrieval_match_score": 0.2,
                "fault_symptom": "显示画面停止更新",
            },
        ]
        state = {
            "query": "以前输入接收计数停止的案例如何恢复",
            "query_targets": [{"intent": "history_query"}],
            "query_frame": {
                "original_query": "以前输入接收计数停止的案例如何恢复"
            },
        }

        selection = select_history_parent_cases(state, rows)
        binding = history_parent_binding(selection, rows)

        self.assertEqual(frozenset({"case-rx", "case-dsp"}), selection.selected_case_ids)
        self.assertFalse(selection.authorizes_exact_links)
        self.assertIn(binding["authority"], {"candidate", "ambiguous"})
        self.assertFalse(binding["authorizes_exact_links"])

    def test_structured_query_singleton_is_not_an_identity_claim(self):
        rows = [
            {"id": 21, "case_id": "CASE-ONE", "match_type": "structured_query"}
        ]

        selection = select_history_parent_cases(
            {
                "query": "找一个相近案例",
                "query_targets": [{"intent": "history_query"}],
            },
            rows,
        )

        self.assertEqual(frozenset({"case-one"}), selection.selected_case_ids)
        self.assertFalse(selection.authorizes_exact_links)
        self.assertEqual("candidate_singleton", selection.source)

    def test_verified_singleton_can_authorize_stored_links(self):
        rows = [
            {
                "id": 31,
                "case_id": "CASE-BOUND",
                "match_type": "exact_conversation_reference",
            }
        ]

        selection = select_history_parent_cases(
            {
                "query": "它最后怎样验证",
                "query_targets": [{"intent": "history_query"}],
            },
            rows,
        )

        self.assertTrue(selection.authorizes_exact_links)
        self.assertEqual("verified_singleton", selection.source)

    def test_answer_reuses_binding_instead_of_reselecting_higher_score(self):
        tool_rows = [
            {
                "id": 41,
                "case_id": "CASE-KEPT",
                "match_type": "exact_conversation_reference",
            }
        ]
        initial = select_history_parent_cases(
            {
                "query": "这个案例如何恢复",
                "query_targets": [{"intent": "history_query"}],
            },
            tool_rows,
        )
        state = {
            "query": "这个案例如何恢复",
            "query_targets": [{"intent": "history_query"}],
            "history_parent_binding": history_parent_binding(initial, tool_rows),
        }
        answer_rows = [
            {
                "id": 41,
                "case_id": "CASE-KEPT",
                "match_type": "exact_conversation_reference",
                "answer_rerank_score": 0.1,
            },
            {
                "id": 42,
                "case_id": "CASE-NEIGHBOR",
                "match_type": "semantic_id_hydration",
                "answer_rerank_score": 0.99,
            },
        ]

        resolved = resolve_history_parent_cases(state, answer_rows)

        self.assertEqual(frozenset({"case-kept"}), resolved.selected_case_ids)
        self.assertTrue(resolved.authorizes_exact_links)

    def test_stale_binding_fails_closed(self):
        state = {
            "history_parent_binding": {
                "version": 1,
                "case_ids": ["CASE-MISSING"],
                "source": "current_query_id",
                "authority": "identity",
                "authorizes_exact_links": True,
            }
        }

        resolved = history_parent_selection_from_binding(
            state,
            [{"id": 51, "case_id": "CASE-OTHER"}],
        )

        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.conflict)
        self.assertFalse(resolved.authorizes_exact_links)
        self.assertEqual(frozenset(), resolved.selected_case_ids)

    def test_forged_identity_boolean_cannot_upgrade_semantic_parent(self):
        state = {
            "query": "找一个相近案例",
            "query_targets": [{"intent": "history_query"}],
            "history_parent_binding": {
                "version": 1,
                "case_ids": ["CASE-SEMANTIC"],
                "row_ids": ["52"],
                "source": "current_query_id",
                "authority": "identity",
                "authorizes_exact_links": True,
            },
        }

        resolved = history_parent_selection_from_binding(
            state,
            [{
                "id": 52,
                "case_id": "CASE-SEMANTIC",
                "match_type": "semantic_id_hydration",
                "retrieval_match_score": 0.99,
            }],
        )

        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.conflict)
        self.assertFalse(resolved.authorizes_exact_links)

    def test_identity_binding_must_name_an_available_parent_row(self):
        state = {
            "query": "核对 CASE-A 的恢复记录",
            "query_targets": [{"intent": "history_query"}],
            "history_parent_binding": {
                "version": 1,
                "case_ids": ["CASE-A"],
                "row_ids": ["999"],
                "source": "current_query_id",
                "authority": "identity",
                "authorizes_exact_links": True,
            },
        }

        resolved = history_parent_selection_from_binding(
            state,
            [{"id": 53, "case_id": "CASE-A", "match_type": "structured_query"}],
        )

        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.conflict)
        self.assertFalse(resolved.authorizes_exact_links)

    def test_binding_cannot_forge_comparison_to_authorize_two_parents(self):
        rows = [
            {"id": 54, "case_id": "CASE-A", "match_type": "exact_record_identity"},
            {"id": 55, "case_id": "CASE-B", "match_type": "exact_record_identity"},
        ]
        state = {
            "query": "继续看这个案例",
            "query_targets": [{"intent": "history_query"}],
            "history_parent_binding": {
                "version": 1,
                "case_ids": ["CASE-A", "CASE-B"],
                "row_ids": ["54", "55"],
                "source": "verified_records",
                "authority": "identity",
                "comparison": True,
                "authorizes_exact_links": True,
            },
        }

        resolved = history_parent_selection_from_binding(state, rows)

        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.ambiguous)
        self.assertFalse(resolved.comparison)
        self.assertFalse(resolved.authorizes_exact_links)

    def test_adaptive_candidate_cannot_override_verified_identity(self):
        verified = {
            "version": 1,
            "case_ids": ["case-bound"],
            "row_ids": ["61"],
            "source": "conversation_anchor",
            "authority": "identity",
            "comparison": False,
            "ambiguous": False,
            "conflict": False,
            "authorizes_exact_links": True,
        }
        candidate = {
            "version": 1,
            "case_ids": ["case-neighbor"],
            "row_ids": ["62"],
            "source": "retrieval_candidates",
            "authority": "candidate",
            "comparison": False,
            "ambiguous": False,
            "conflict": False,
            "authorizes_exact_links": False,
        }

        self.assertEqual(
            verified,
            merge_history_parent_bindings(verified, candidate),
        )

    def test_conflicting_adaptive_identities_fail_closed(self):
        first = {
            "version": 1,
            "case_ids": ["case-a"],
            "authority": "identity",
            "authorizes_exact_links": True,
        }
        second = {
            "version": 1,
            "case_ids": ["case-b"],
            "authority": "identity",
            "authorizes_exact_links": True,
        }

        merged = merge_history_parent_bindings(first, second)

        self.assertTrue(merged["conflict"])
        self.assertFalse(merged["authorizes_exact_links"])
        self.assertEqual([], merged["case_ids"])


if __name__ == "__main__":
    unittest.main()
