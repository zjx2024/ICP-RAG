import unittest

from backend.agent.current_model_selection import model_candidates, selection_matches_model, validate_selection


class CurrentModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.query = "让那条带备路的供电链完成整个值班，具体该用哪个模型？"
        self.row = {"doc_id": "source-17", "revision_id": "revision-93", "diagram_id": "diagram-2",
                    "diagram_key": "model-71", "image_title": "机内电源连续性", "is_active": True,
                    "validation_status": "auto_validated"}
        self.proposal = {"status": "selected", "question_basis": "那条带备路的供电链",
                         "selections": [{"doc_id": "source-17", "revision_id": "revision-93", "matched_title": "机内电源连续性"}],
                         "reason": "current semantic selection, not an identity merge"}

    def validate(self, proposal=None, rows=None):
        return validate_selection(proposal or self.proposal, model_candidates(rows or [self.row], "rbd"), query=self.query)

    def test_paraphrase_can_choose_current_model_without_fixed_words_or_typed_id(self):
        result = self.validate()
        self.assertEqual("selected", result["status"])
        self.assertTrue(selection_matches_model(result, self.row, query=self.query))
        self.assertNotIn("calculation_authorizations", result)

    def test_current_question_change_invalidates_old_selection(self):
        self.assertFalse(selection_matches_model(self.validate(), self.row, query="改查另一条链"))

    def test_selection_cannot_cross_document_or_revision(self):
        for field in ("doc_id", "revision_id", "diagram_id", "diagram_key"):
            with self.subTest(field=field):
                self.assertFalse(selection_matches_model(self.validate(), {**self.row, field: "other"}, query=self.query))

    def test_model_may_not_invent_identity_title_or_parameter(self):
        for change in ({"doc_id": "other"}, {"revision_id": "other"}, {"matched_title": "other"}, {"mission_time": 20}):
            proposal = {**self.proposal, "selections": [{**self.proposal["selections"][0], **change}]}
            with self.subTest(change=change):
                self.assertEqual({}, self.validate(proposal))

    def test_duplicate_candidate_or_selection_is_not_unique(self):
        self.assertEqual(1, len(model_candidates([self.row, self.row], "rbd")))
        self.assertEqual([], model_candidates([self.row, {**self.row, "image_title": "conflicting"}], "rbd"))
        self.assertEqual({}, self.validate({**self.proposal, "selections": self.proposal["selections"] * 2}))

    def test_ambiguity_or_unrepresented_subcomponent_does_not_grant_a_model(self):
        for status in ("ambiguous", "not_represented"):
            proposal = {**self.proposal, "status": status, "selections": []}
            result = self.validate(proposal)
            self.assertFalse(selection_matches_model(result, self.row, query=self.query))
            self.assertEqual({}, self.validate({**proposal, "selections": self.proposal["selections"]}))

    def test_formal_status_is_not_upgraded_by_selection(self):
        provisional = {**self.row, "validation_status": "provisional"}
        result = self.validate(rows=[provisional])
        self.assertEqual("provisional", result["selections"][0]["validation_status"])

    def test_basis_must_come_from_the_current_question(self):
        self.assertEqual({}, self.validate({**self.proposal, "question_basis": "来自旧问题的字串"}))
