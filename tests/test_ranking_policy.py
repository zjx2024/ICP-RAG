import os
import unittest
from unittest.mock import patch

from backend.agent.ranking_policy import CandidateRankingWeights, candidate_ranking_weights
from backend.agent.tools.icp_diagnosis import _candidate_faults


class RankingPolicyTests(unittest.TestCase):
    def test_default_scores_preserve_existing_behavior(self):
        hmc = [{"hmc_code": "A", "fault_desc": "精确对象", "match_type": "exact_code"},
               {"hmc_code": "B", "fault_desc": "组件对象", "match_type": "exact_components"},
               {"hmc_code": "C", "fault_desc": "条件对象", "match_type": "keyword"}]
        fmeca = [{"code": "F", "fault_mode": "模式"}]
        search = [{"kb_id": "fmeca", "title": "F", "score": 0.8}]
        result = _candidate_faults(hmc, fmeca, search, weights=CandidateRankingWeights())
        self.assertEqual({"精确对象": .55, "组件对象": .45, "条件对象": .3, "模式": .47}, {r["fault"]: r["score"] for r in result})

    def test_extreme_weights_cannot_promote_search_to_exact_authority(self):
        rows = [{"kb_id": "hmc", "title": "近似对象", "chunk_id": str(i), "score": 1} for i in range(6)]
        result = _candidate_faults([{"hmc_code": "A", "fault_desc": "精确对象", "match_type": "exact_code"}], [], rows,
            weights=CandidateRankingWeights(hmc_exact=0, search_base=1, search_score=1))
        self.assertEqual("精确对象", result[0]["fault"])
        self.assertFalse(result[1]["authoritative_support"])
        self.assertFalse(result[1]["exact_hmc_match"])
        self.assertEqual(.99, result[1]["score"])

    def test_environment_overrides_are_explicit_validated_and_auditable(self):
        with patch.dict(os.environ, {"ICP_CANDIDATE_RANKING_WEIGHTS": '{"fmeca":0.35}'}):
            weights = candidate_ranking_weights()
        self.assertEqual(.35, weights.fmeca)
        self.assertEqual(.55, weights.hmc_exact)
        self.assertEqual(.35, weights.public_dict()["weights"]["fmeca"])
        for raw in ('[]', '{"allow_unique":true}', '{"fmeca":NaN}', '{"fmeca":true}', '{"fmeca":2}', '{"fmeca":-1}'):
            with self.subTest(raw=raw), patch.dict(os.environ, {"ICP_CANDIDATE_RANKING_WEIGHTS": raw}), self.assertRaises(ValueError):
                candidate_ranking_weights()


if __name__ == "__main__":
    unittest.main()
