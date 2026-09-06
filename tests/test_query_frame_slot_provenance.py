import unittest

from backend.agent.query_understanding import build_query_frame


class QueryFrameSlotProvenanceTests(unittest.TestCase):
    def test_unbacked_temporal_interpretation_is_discarded(self):
        frame = build_query_frame(
            query="滤波电容用久后性能变差，会带来什么后果？",
            planner_frame={
                "operating_stage": "持续运行后",
                "operating_stage_basis": "用久后",
                "temporal_pattern": "持续发生",
                "temporal_pattern_basis": "",
            },
        )

        self.assertEqual("持续运行后", frame["operating_stage"])
        self.assertEqual("用久后", frame["operating_stage_basis"])
        self.assertEqual("未知", frame["temporal_pattern"])
        self.assertEqual("", frame["temporal_pattern_basis"])

    def test_explicit_occurrence_basis_is_retained(self):
        frame = build_query_frame(
            query="运行中反复出现短时掉电",
            planner_frame={
                "operating_stage": "运行中",
                "operating_stage_basis": "运行中",
                "temporal_pattern": "重复发生",
                "temporal_pattern_basis": "反复出现",
            },
        )

        self.assertEqual("运行中", frame["operating_stage"])
        self.assertEqual("重复发生", frame["temporal_pattern"])
        self.assertEqual("反复出现", frame["temporal_pattern_basis"])

    def test_rewrite_only_basis_cannot_become_a_filter(self):
        frame = build_query_frame(
            query="电源有问题",
            rewritten_query="高负载时电源间歇失效",
            planner_frame={
                "operating_stage": "高负载",
                "operating_stage_basis": "高负载时",
                "temporal_pattern": "间歇发生",
                "temporal_pattern_basis": "间歇失效",
            },
        )

        self.assertEqual("未知", frame["operating_stage"])
        self.assertEqual("未知", frame["temporal_pattern"])

    def test_semantic_slot_is_owned_by_the_planning_model(self):
        frame = build_query_frame(
            query="运行中反复出现短时掉电",
            planner_frame={
                "operating_stage": "高负载",
                "operating_stage_basis": "运行中",
                "temporal_pattern": "首次发生",
                "temporal_pattern_basis": "反复出现",
            },
        )

        # Python validates the enum and its verbatim provenance.  It does not
        # maintain a second phrase dictionary to overrule the model's meaning.
        self.assertEqual("高负载", frame["operating_stage"])
        self.assertEqual("运行中", frame["operating_stage_basis"])
        self.assertEqual("首次发生", frame["temporal_pattern"])
        self.assertEqual("反复出现", frame["temporal_pattern_basis"])

    def test_single_character_basis_cannot_authorize_a_filter(self):
        frame = build_query_frame(
            query="运行中首次出现短时掉电",
            planner_frame={
                "operating_stage": "运行中",
                "operating_stage_basis": "运",
                "temporal_pattern": "首次发生",
                "temporal_pattern_basis": "首",
            },
        )

        self.assertEqual("未知", frame["operating_stage"])
        self.assertEqual("", frame["operating_stage_basis"])
        self.assertEqual("未知", frame["temporal_pattern"])
        self.assertEqual("", frame["temporal_pattern_basis"])

    def test_no_temporal_slot_is_inferred_from_operating_duration(self):
        frame = build_query_frame(
            query="设备持续运行后电容性能变差",
            planner_frame={
                "operating_stage": "持续运行后",
                "operating_stage_basis": "持续运行后",
                "temporal_pattern": "未知",
                "temporal_pattern_basis": "",
            },
        )

        self.assertEqual("持续运行后", frame["operating_stage"])
        self.assertEqual("未知", frame["temporal_pattern"])
        self.assertEqual("", frame["temporal_pattern_basis"])


if __name__ == "__main__":
    unittest.main()
