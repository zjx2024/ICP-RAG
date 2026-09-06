import unittest

from backend.agent.state import create_initial_state


class AgentStateLifecycleTests(unittest.TestCase):
    def make_state(self, history=None):
        return create_initial_state("本轮问题", "agent", "", history=history, memory_enabled=False)

    def test_fresh_turns_do_not_share_mutable_containers(self):
        first, second = self.make_state(), self.make_state()
        for key, value in first.items():
            if isinstance(value, (list, dict)):
                with self.subTest(key=key):
                    self.assertIsNot(value, second[key])
        first["evidence_registry"].append({"evidence_id": "old"})
        self.assertEqual([], second["evidence_registry"])

    def test_history_is_copied_including_nested_delivery_and_identity_context(self):
        history = [{"role": "assistant", "content": "前轮答案", "conversation_anchors": [{"record_id": "REC-41"}],
                    "answer_delivery_manifest": [{"field": "local_impact", "record_ids": ["REC-41"]}]}]
        first, second = self.make_state(history), self.make_state(history)
        first["messages"][0]["conversation_anchors"][0]["record_id"] = "REC-73"
        first["messages"][0]["answer_delivery_manifest"][0]["record_ids"].append("REC-73")
        self.assertEqual(history, second["messages"])
        self.assertEqual(["REC-41"], history[0]["answer_delivery_manifest"][0]["record_ids"])

    def test_restored_history_does_not_grant_current_calculation_or_approval(self):
        state = self.make_state([{"role": "assistant", "content": "旧结果", "calculation_authorizations": [{"operation": "rbd_reliability"}],
                                  "approved_evidence_ids": ["old"], "_reviewed_context_units": ["旧结论"]}])
        self.assertEqual([], state["calculation_authorizations"])
        self.assertEqual([], state["approved_evidence_ids"])
        self.assertEqual([], state["evidence_registry"])
        self.assertNotIn("_reviewed_context_units", state)


if __name__ == "__main__":
    unittest.main()
