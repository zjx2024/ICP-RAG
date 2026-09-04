"""后处理函数：Python dict 字面量清洗、重复 bullet 去重与清单外壳剥离的单元测试。"""
import unittest

from backend.agent.nodes import answer as answer_module

_sanitize = answer_module._sanitize_prompt_echo_literals
_deduplicate = answer_module._deduplicate_repeated_answer_lines
_strip_manifest = answer_module._strip_manifest_echo_lines


class SanitizePromptEchoLiteralsTests(unittest.TestCase):
    def test_single_quote_name_description_shape(self):
        self.assertEqual(
            _sanitize("{'name': 'B通道发送占用', 'description': '状态机卡滞。'}"),
            "B通道发送占用（状态机卡滞。）",
        )

    def test_double_quote_shape(self):
        self.assertEqual(
            _sanitize('{"name": "A通道接收超时", "description": "接收器件失效。"}'),
            "A通道接收超时（接收器件失效。）",
        )

    def test_multiple_literals_in_text(self):
        text = (
            "基本事件：{'name': 'A', 'description': 'x。'}；"
            "{'name': 'B', 'description': 'y。'}。[^1]"
        )
        self.assertEqual(_sanitize(text), "基本事件：A（x。）；B（y。）。[^1]")

    def test_description_optional_keeps_name(self):
        self.assertEqual(
            _sanitize("{'name': 'A', 'description': ''}"),
            "A",
        )

    def test_no_literal_untouched(self):
        text = "FTA 结构：包含 14 个节点；逻辑门：AND。"
        self.assertEqual(_sanitize(text), text)


class DeduplicateRepeatedAnswerLinesTests(unittest.TestCase):
    def test_drops_identical_bullets_with_different_markers(self):
        answer = (
            "- BCB B通道发送占用：隔离并复位B通道；不能恢复时更换模块。[^7]\n"
            "- BCB B通道发送占用：隔离并复位B通道；不能恢复时更换模块。[^8]\n"
            "- BCA A通道接收超时：检查A通道连接。[^1]"
        )
        self.assertEqual(
            _deduplicate(answer),
            (
                "- BCB B通道发送占用：隔离并复位B通道；不能恢复时更换模块。[^7]\n"
                "- BCA A通道接收超时：检查A通道连接。[^1]"
            ),
        )

    def test_distinct_bullets_all_kept(self):
        answer = (
            "- A通道接收超时：维修措施一。[^1]\n"
            "- A通道发送占用：维修措施二。[^2]"
        )
        self.assertEqual(_deduplicate(answer), answer)

    def test_non_bullet_paragraphs_untouched(self):
        answer = (
            "当前已批准证据不足以支持可直接陈述的事实结论。\n"
            "- 唯一条目。[^1]\n"
            "实际处置以适用构型为准。"
        )
        self.assertEqual(_deduplicate(answer), answer)


class StripManifestEchoLinesTests(unittest.TestCase):
    def test_removes_manifest_shell_and_children(self):
        answer = (
            "- 证据 [^2]｜fmeca｜记录 FM-SYN-072（风量衰减）｜可补 故障原因\n"
            "  - 故障原因：滤网堵塞、叶片污染、风道受阻或叶片损伤。[^2]\n"
            "- 正常叙述：风量衰减会导致局部热应力加剧。[^3]"
        )
        self.assertEqual(
            _strip_manifest(answer),
            "- 正常叙述：风量衰减会导致局部热应力加剧。[^3]",
        )

    def test_removes_manifest_shell_without_citation_marker(self):
        answer = (
            "故障原因[^1]\n"
            "- 证据｜fmeca｜记录 X｜可支持 故障原因\n"
            "  - 故障原因：内容。[^1]"
        )
        self.assertEqual(_strip_manifest(answer), "故障原因[^1]")

    def test_normal_lines_untouched(self):
        answer = "- 系统无法与外部设备交换数据：结构原因链已列出。[^1]"
        self.assertEqual(_strip_manifest(answer), answer)


if __name__ == "__main__":
    unittest.main()