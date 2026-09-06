import unittest
import asyncio
from unittest.mock import AsyncMock, patch

from backend.agent.current_model_selection import model_candidates, validate_selection
from backend.agent.rbd_contract import bind_rbd_models, extract_explicit_rbd_ids, validate_rbd_authority
from tests.test_rbd_authority_contract import _row, _evidence
from backend.agent.tools import icp_diagnosis


class SemanticRbdAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.query = "请看负责机上能量持续供应的那一条，不用通信那条。"
        self.rows = [_row("rev-77", diagram_key="RBD-OMEGA", doc_id="source-41"),
                     _row("rev-89", diagram_key="RBD-SIGMA", doc_id="source-52")]
        self.rows[0]["image_title"] = "机内电源连续性"
        self.rows[1]["image_title"] = "通信可用性"
        self.selection = validate_selection({
            "status": "selected", "question_basis": "负责机上能量持续供应的那一条",
            "selections": [{"doc_id": "source-41", "revision_id": "rev-77",
                            "matched_title": "机内电源连续性"}],
        }, model_candidates(self.rows, "rbd"), query=self.query)

    def bind(self, **kwargs):
        return bind_rbd_models(self.rows, required=True, raw_user_query=self.query,
                               current_selection=self.selection, **kwargs)

    def test_natural_language_selection_is_revalidated_at_consumers(self):
        rows, binding = self.bind()
        self.assertEqual([self.rows[0]], rows)
        for models in (rows, [_evidence(row) for row in rows]):
            self.assertTrue(validate_rbd_authority(
                binding, models=models, raw_user_query=self.query)["valid"])
        self.assertNotIn("calculation_authorizations", binding)

    def test_document_scope_survives_row_to_evidence_and_changed_scope_still_fails(self):
        self.rows[0]["document_metadata"] = {"configuration": "BUILD-Q", "document_version": "V8"}
        candidates = model_candidates(self.rows, "rbd")
        self.assertEqual("BUILD-Q", candidates[0]["configuration"])
        self.assertEqual("V8", candidates[0]["document_version"])
        rows, binding = self.bind()
        self.assertEqual(["build-q"], binding["model_bindings"][0]["configurations"])
        item = _evidence(rows[0])
        item["metadata"].update(self.rows[0]["document_metadata"])
        self.assertTrue(validate_rbd_authority(binding, models=[item], raw_user_query=self.query)["valid"])
        item["metadata"]["configuration"] = "BUILD-R"
        self.assertFalse(validate_rbd_authority(binding, models=[item], raw_user_query=self.query)["valid"])

    def test_old_turn_or_changed_source_cannot_reuse_selection(self):
        _, binding = self.bind()
        self.assertFalse(validate_rbd_authority(binding, models=[self.rows[0]],
                                               raw_user_query="改为看通信")["valid"])
        for field, value in (("doc_id", "other"), ("revision_id", "other"),
                             ("validation_status", "provisional"), ("is_active", False)):
            with self.subTest(field=field):
                changed = {**self.rows[0], field: value}
                self.assertFalse(validate_rbd_authority(binding, models=[changed],
                                                       raw_user_query=self.query)["valid"])

    def test_missing_selection_provenance_cannot_pass(self):
        _, binding = self.bind()
        binding.pop("current_selection")
        self.assertFalse(validate_rbd_authority(
            binding, models=[self.rows[0]], raw_user_query=self.query)["valid"])

    def test_selected_provisional_model_is_still_not_formal(self):
        self.rows[0]["validation_status"] = "provisional"
        _, binding = self.bind()
        self.assertFalse(binding["bound"])

    def test_explicit_current_identifier_overrides_semantic_selection(self):
        self.query += "更正：RBD-SIGMA"
        rows, binding = self.bind(explicit_rbd_ids=["RBD-SIGMA"])
        self.assertEqual([self.rows[1]], rows)
        self.assertEqual("exact_rbd_id", binding["basis"])

    def test_semantic_multiple_selection_needs_each_offered_identity(self):
        self.selection["selections"].append(model_candidates(self.rows, "rbd")[1])
        rows, binding = self.bind()
        self.assertEqual(self.rows, rows)
        self.assertEqual("explicit_model_set", binding["basis"])
        self.assertTrue(validate_rbd_authority(binding, models=rows,
                                             raw_user_query=self.query)["valid"])

    def test_unrepresented_or_ambiguous_selection_is_not_top_one(self):
        for status in ("ambiguous", "not_represented"):
            self.selection = {**self.selection, "status": status, "selections": []}
            rows, binding = self.bind()
            self.assertEqual([], rows)
            self.assertFalse(binding["bound"])

    def test_internal_identifier_substring_is_not_an_explicit_rbd_id(self):
        self.assertEqual([], extract_explicit_rbd_ids("检查 revision-rbd-omega"))
        self.assertEqual(["revision-rbd-omega"],
                         extract_explicit_rbd_ids("revision_id: revision-rbd-omega"))

    def test_diagnosis_carries_current_read_selection_to_formal_binding(self):
        with patch.object(icp_diagnosis, "_query_diagrams", return_value=self.rows), patch.object(
            icp_diagnosis, "select_current_models", new_callable=AsyncMock,
            return_value=self.selection,
        ) as selector:
            result = asyncio.run(icp_diagnosis.icp_diagnosis_node({
                "query": self.query, "allowed_tools": ["rbd_query"],
            }))
        selector.assert_awaited_once()
        binding = result["rbd_query_binding"]
        self.assertTrue(binding["bound"])
        self.assertEqual("current_model_selection", binding["basis"])
        self.assertTrue(validate_rbd_authority(binding, models=[self.rows[0]],
                                             raw_user_query=self.query)["valid"])

    def _anchored_diagnosis(self, query, selections, *, status="selected"):
        anchors = [{"doc_type": "rbd", "doc_id": row["doc_id"],
                    "record_id": row["revision_id"]} for row in self.rows]
        selection = validate_selection({
            "status": status, "question_basis": query, "selections": [
                {"doc_id": row["doc_id"], "revision_id": row["revision_id"],
                 "matched_title": row["image_title"]} for row in selections
            ],
        }, model_candidates(self.rows, "rbd"), query=query)
        with patch.object(icp_diagnosis, "_query_diagrams", return_value=self.rows), patch.object(
            icp_diagnosis, "select_current_models", new_callable=AsyncMock,
            return_value=selection,
        ) as selector:
            result = asyncio.run(icp_diagnosis.icp_diagnosis_node({
                "query": query, "allowed_tools": ["rbd_query"],
                "selected_conversation_anchors": anchors,
            }))
        return result["rbd_query_binding"], selector

    def test_multi_anchor_continuation_without_compare_label_uses_current_semantics(self):
        for query in ("把刚才那些分别算一下。", "逐个给出它们在八小时后的结果。"):
            with self.subTest(query=query):
                binding, selector = self._anchored_diagnosis(query, self.rows)
                selector.assert_awaited_once()
                self.assertEqual("explicit_model_set", binding["basis"])
                self.assertTrue(validate_rbd_authority(
                    binding, models=self.rows, raw_user_query=query)["valid"])
                self.assertNotIn("calculation_authorizations", binding)

    def test_multi_anchor_singular_ambiguity_does_not_authorize_all_or_top_one(self):
        binding, selector = self._anchored_diagnosis(
            "那这一条的结果呢？", [], status="ambiguous")
        selector.assert_awaited_once()
        self.assertFalse(binding["bound"])
        self.assertTrue(binding["ambiguous"])

    def test_multi_anchor_continuation_can_narrow_to_one_current_model(self):
        query = "只继续看负责供能的那个。"
        binding, selector = self._anchored_diagnosis(query, self.rows[:1])
        selector.assert_awaited_once()
        self.assertTrue(binding["bound"])
        self.assertEqual([f"revision_id:{self.rows[0]['revision_id']}"], binding["record_ids"])

    def test_explicit_correction_skips_semantic_multi_anchor_selection(self):
        binding, selector = self._anchored_diagnosis("更正：只看 RBD-SIGMA。", self.rows[:1])
        selector.assert_not_awaited()
        self.assertTrue(binding["bound"])
        self.assertEqual("exact_rbd_id", binding["basis"])
        self.assertEqual([f"revision_id:{self.rows[1]['revision_id']}"], binding["record_ids"])
