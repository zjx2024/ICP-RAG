import copy
import subprocess
import sys
import unittest
from pathlib import Path

from backend.agent.candidate_ranking import rank_candidates
from backend.agent.ranking_policy import CandidateRankingWeights
from backend.agent.tools.icp_diagnosis import _candidate_faults, _direct_evidence


class CandidateRecordIdentityTests(unittest.TestCase):
    def fmeca(self, code="FM-A", doc="doc-a", label="相同故障描述", **extra):
        return {"code": code, "doc_id": doc, "fault_mode": label, **extra}

    def test_same_name_different_records_are_not_merged(self):
        for label in ("异常", "事件历史记录", "identical observation", "成功路径/冗余"):
            with self.subTest(label=label):
                rows = [self.fmeca("FM-A", label=label), self.fmeca("FM-B", label=label)]
                result = _candidate_faults([], rows, [])
                self.assertEqual(2, len(result))
                self.assertEqual(2, len({item["candidate_id"] for item in result}))
                self.assertEqual([.25, .25], [item["score"] for item in result])

    def test_same_code_in_different_documents_keeps_own_evidence(self):
        rows = [self.fmeca(doc="doc-a"), self.fmeca(doc="doc-b")]
        evidence = _direct_evidence("fmeca_query", rows)
        result = _candidate_faults([], rows, [], evidence)
        self.assertEqual(2, len(result))
        for candidate in result:
            own = {item["evidence_id"] for item in evidence if item["doc_id"] == candidate["record_identity"]["doc_id"]}
            self.assertTrue(candidate["supporting_evidence_ids"])
            self.assertTrue(set(candidate["supporting_evidence_ids"]).issubset(own))

    def test_configuration_and_revision_are_part_of_identity(self):
        for key in ("applicable_configuration", "document_version"):
            rows = [self.fmeca(document_metadata={key: "A"}), self.fmeca(document_metadata={key: "B"})]
            result = _candidate_faults([], rows, [])
            self.assertEqual(2, len(result))
        result = _candidate_faults([], [self.fmeca(revision_id="r1"), self.fmeca(revision_id="r2")], [])
        self.assertEqual(2, len(result))

    def test_same_label_across_document_families_is_not_an_explicit_relation(self):
        result = _candidate_faults([{"hmc_code": "0099", "doc_id": "h", "fault_desc": "异常"}],
                                   [self.fmeca(label="异常")], [])
        self.assertEqual(2, len(result))
        self.assertEqual({"hmc", "fmeca"}, {item["record_identity"]["doc_type"] for item in result})

    def test_hmc_variants_keep_their_own_fault_evidence(self):
        rows = [{"hmc_code": "0099", "doc_id": "d", "fault_desc": "相同",
                 "variant_id": variant, "source_ref": variant} for variant in ("a", "b")]
        evidence = _direct_evidence("hmc_query", rows)
        result = _candidate_faults(rows, [], [], evidence)
        self.assertEqual(2, len(result))
        for candidate in result:
            own = {item["evidence_id"] for item in evidence
                   if item["metadata"]["variant_id"] == candidate["record_identity"]["variant_id"]}
            self.assertEqual(own, set(candidate["supporting_evidence_ids"]))
        search = {"kb_id": "hmc", "title": "0099", "doc_id": "d", "chunk_id": "s", "score": 1}
        ambiguous = _candidate_faults(rows, [], [search])
        self.assertEqual(3, len(ambiguous))
        self.assertTrue(all(item["score"] == .3 for item in ambiguous if item["authoritative_support"]))
        specific = _candidate_faults(rows, [], [{**search, "variant_id": "a"}])
        self.assertEqual(2, len(specific))
        self.assertEqual({"a": .55, "b": .3},
                         {item["record_identity"]["variant_id"]: item["score"] for item in specific})

    def test_explicit_row_scope_and_identity_reach_answer_context(self):
        from backend.agent.nodes.answer import _format_context

        rows = [self.fmeca(scope={"configuration": value}) for value in ("A", "B")]
        result = _candidate_faults([], rows, [])
        self.assertEqual(2, len(result))
        evidence = _direct_evidence("fmeca_query", rows)
        ids = [item["evidence_id"] for item in evidence]
        rendered = _format_context({"query": "这两种构型有何区别？",
            "evidence_registry": evidence, "context_evidence_ids": ids,
            "safety_decision": {"approved_evidence_ids": ids}})
        self.assertIn('"configuration": "A"', rendered)
        self.assertIn('"configuration": "B"', rendered)
        self.assertIn('"record_id": "FM-A"', rendered)

    def test_duplicate_rows_and_search_fragments_do_not_inflate_scores(self):
        row = self.fmeca()
        search = {"kb_id": "fmeca", "doc_id": "doc-a", "title": "FM-A", "score": .8, "chunk_id": "c1"}
        single = _candidate_faults([], [row], [search])
        duplicate = _candidate_faults([], [row, copy.deepcopy(row)], [search, {**search, "chunk_id": "c2"}])
        self.assertEqual(1, len(duplicate))
        self.assertEqual(.47, single[0]["score"])
        self.assertEqual(single[0]["score"], duplicate[0]["score"])
        self.assertEqual(single[0]["candidate_id"], duplicate[0]["candidate_id"])

    def test_ambiguous_code_without_document_does_not_select_a_parent(self):
        rows = [self.fmeca(doc="a"), self.fmeca(doc="b")]
        result = _candidate_faults([], rows, [{"kb_id": "fmeca", "title": "FM-A", "chunk_id": "s", "score": 1}])
        self.assertEqual(3, len(result))
        self.assertEqual([.25, .25], [item["score"] for item in result if item["authoritative_support"]])
        self.assertEqual(1, sum(not item["authoritative_support"] for item in result))

    def test_wrong_document_and_configuration_cannot_reinforce_a_record(self):
        row = self.fmeca(document_metadata={"applicable_configuration": "A"})
        for extra in ({"doc_id": "wrong"}, {"document_metadata": {"applicable_configuration": "B"}}):
            search = {"kb_id": "fmeca", "title": "FM-A", "chunk_id": "s", "score": 1, **extra}
            result = _candidate_faults([], [row], [search])
            self.assertEqual(.25, next(item["score"] for item in result if item["authoritative_support"]))

    def test_unknown_identity_never_uses_label_to_merge_or_grant_exactness(self):
        result = rank_candidates(
            [{"fault_desc": "相同", "match_type": "exact_code"}, {"fault_desc": "相同"}], [], [], [],
            weights=CandidateRankingWeights(), structured_hmc_matches=frozenset({"exact_code"}),
        )
        self.assertEqual(2, len(result))
        self.assertFalse(any(item["exact_hmc_match"] or item["authoritative_support"] for item in result))

    def test_leading_zero_identity_and_case_are_preserved(self):
        rows = [{"hmc_code": code, "fault_desc": "相同", "doc_id": "d"} for code in ("0001", "001", "Ab", "AB")]
        result = _candidate_faults(rows, [], [])
        self.assertEqual({"0001", "001", "Ab", "AB"}, {item["record_identity"]["record_id"] for item in result})

    def test_order_changes_do_not_change_known_identity_or_scores(self):
        rows = [self.fmeca("FM-A"), self.fmeca("FM-B")]
        before = copy.deepcopy(rows)
        self.assertEqual(_candidate_faults([], rows, []), _candidate_faults([], list(reversed(rows)), []))
        self.assertEqual(before, rows)

    def test_rewording_changes_display_not_record_identity_or_support(self):
        before = _candidate_faults([], [self.fmeca(label="间歇现象")], [])
        after = _candidate_faults([], [self.fmeca(label="observation rephrased freely")], [])
        self.assertEqual(before[0]["candidate_id"], after[0]["candidate_id"])
        self.assertEqual(before[0]["score"], after[0]["score"])
        self.assertNotEqual(before[0]["fault"], after[0]["fault"])

    def test_ranker_import_does_not_load_storage_or_orchestration(self):
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; import backend.agent.candidate_ranking; "
            "assert not any(name.startswith(('backend.storage', 'backend.agent.nodes', "
            "'backend.agent.tools')) for name in sys.modules)",
        ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)
        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
