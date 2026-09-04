"""answer_presence 共享在场原语的单元测试（C1 批 1）。"""
import unittest

from backend.agent.nodes import answer_presence as presence


class NormalizePresenceTextTests(unittest.TestCase):
    def test_strips_punctuation_and_whitespace(self):
        self.assertEqual(
            presence.normalize_presence_text("接收器件失效、线路异常或时序配置错误。"),
            "接收器件失效线路异常或时序配置错误",
        )

    def test_casefold_ascii(self):
        self.assertEqual(
            presence.normalize_presence_text("GetBCAFaultStatus"),
            "getbcafaultstatus",
        )

    def test_empty_returns_empty(self):
        self.assertEqual(presence.normalize_presence_text(""), "")


class NormalizedSubstringPresentTests(unittest.TestCase):
    def test_long_substring_hit(self):
        answer = presence.normalize_presence_text(
            "故障原因：滤网堵塞、叶片污染、风道受阻或叶片损伤。"
        )
        self.assertTrue(
            presence.normalized_substring_present("滤网堵塞、叶片污染", answer)
        )

    def test_short_fragment_rejected(self):
        answer = presence.normalize_presence_text("检查A通道连接。")
        self.assertFalse(presence.normalized_substring_present("检查", answer))  # 3 字符 < 6


class SemanticTokensTests(unittest.TestCase):
    def test_tokens_skip_stopwords(self):
        tokens = presence.semantic_tokens("系统出现故障，接收器件失效")
        self.assertNotIn("系统", tokens)
        self.assertIn("接收器件失效", tokens)

    def test_binding_context_markers_removed(self):
        tokens = presence.semantic_tokens("【历史案例上下文】更换模块")
        self.assertNotIn("【历史案例上下文】", tokens)
        self.assertIn("更换模块", tokens or {"更换模块"})


class ApprovedIdSetTests(unittest.TestCase):
    def test_from_evidence_dicts(self):
        items = [
            {"evidence_id": "ev-1", "title": "a"},
            {"evidence_id": "", "title": "b"},
            {"evidence_id": "ev-2"},
        ]
        self.assertEqual(
            presence.approved_id_set(items),
            {"ev-1", "ev-2"},
        )

    def test_from_raw_strings(self):
        self.assertEqual(
            presence.approved_id_set(["a", "", "b", None]),
            {"a", "b"},
        )


if __name__ == "__main__":
    unittest.main()