"""B1 全链路 tracing 单元测试。"""
import json
import os
import tempfile
import unittest

from backend.agent import tracing


class TracingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_dir = getattr(tracing.config, "TRACE_DIR", None)
        tracing.config.TRACE_DIR = self._tmp.name
        # 重置进程级聚合，保证用例相互独立
        with tracing._AGG_LOCK:
            for key in tracing._AGG:
                tracing._AGG[key] = 0

    def tearDown(self):
        tracing.config.TRACE_DIR = self._old_dir
        self._tmp.cleanup()

    def test_begin_record_end_writes_jsonl_and_clears_context(self):
        trace_id = tracing.begin_turn(session_id="s1", query="外接设备收不到数据")
        tracing.record_llm(template_name="plan", duration_ms=120.5, ok=True, model_name="qwen")
        tracing.record_tool(tool_name="function_locator", authorized="approved", evidence_count=3)
        tracing.record_degradation("planner_json", detail="json parse")
        tracing.record_degradation("adaptive_fallback", detail="route unusable")
        summary = tracing.end_turn()

        self.assertEqual(summary["llm_calls"], 1)
        self.assertEqual(summary["tool_events"], 1)
        self.assertEqual(summary["degradations"]["planner_json"], 1)
        self.assertEqual(summary["degradations"]["adaptive_fallback"], 1)
        self.assertIsNone(tracing.peek(), "end_turn 后必须清空上下文")

        path = os.path.join(self._tmp.name, f"{trace_id}.jsonl")
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        self.assertEqual(rows[0]["trace_id"], trace_id)
        self.assertNotIn("type", rows[0], "summary 行不带 type 字段")  # summary 行有 trace_id
        event_types = [row.get("type") for row in rows[1:]]
        self.assertIn("llm", event_types)
        self.assertIn("tool", event_types)
        self.assertIn("degradation", event_types)

        # 回放接口
        replay = tracing.get_trace(trace_id)
        self.assertEqual(replay["trace_id"], trace_id)
        self.assertEqual(len(replay["events"]), 4)

    def test_trace_without_begin_is_noop_for_context_but_counts_aggregate(self):
        before = dict(tracing._AGG)
        tracing.record_llm(template_name="t", duration_ms=1, ok=False, model_name="m")
        after = dict(tracing._AGG)
        self.assertEqual(after["llm_calls"], before["llm_calls"] + 1)
        self.assertEqual(after["llm_failures"], before["llm_failures"] + 1)
        self.assertIsNone(tracing.end_turn(), "无活跃 trace 时 end_turn 返回 None")
        self.assertEqual(os.listdir(self._tmp.name), [], "无活跃 trace 时不得落盘")

    def test_degradation_counts_even_without_trace(self):
        before = dict(tracing._AGG)
        tracing.record_degradation("retrieval_projection", detail="projection failed")
        tracing.record_degradation("unknown_kind", detail="ignored")
        after = dict(tracing._AGG)
        self.assertEqual(after["deg_retrieval_projection"], before["deg_retrieval_projection"] + 1)
        self.assertNotIn("deg_unknown_kind", after)

    def test_aggregate_stats_rates(self):
        trace_id = tracing.begin_turn(session_id="s1", query="q1")
        tracing.record_llm(template_name="a", duration_ms=1, ok=True)
        tracing.record_llm(template_name="b", duration_ms=1, ok=True)
        tracing.end_turn()

        stats = tracing.aggregate_stats()
        self.assertEqual(stats["turns"], 1)
        self.assertEqual(stats["llm_calls"], 2)
        self.assertEqual(stats["avg_llm_calls_per_turn"], 2.0)
        self.assertEqual(stats["llm_failure_rate"], 0.0)

        # list_traces 只返回 summary 首行
        listed = tracing.list_traces(limit=5)
        self.assertTrue(any(item["trace_id"] == trace_id for item in listed))
        self.assertNotIn("events", listed[0])

    def test_contextvar_isolation_between_turns(self):
        first = tracing.begin_turn(session_id="a", query="q1")
        second = tracing.begin_turn(session_id="b", query="q2")  # 覆盖上下文
        self.assertNotEqual(first, second)
        self.assertEqual(tracing.peek().session_id, "b")
        tracing.record_llm(template_name="x", duration_ms=1, ok=True)
        summary_b = tracing.end_turn()
        self.assertEqual(summary_b["session_id"], "b")
        self.assertIsNone(tracing.peek())


if __name__ == "__main__":
    unittest.main()