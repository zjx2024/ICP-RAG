import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

module = importlib.import_module("backend.core.llm_service")


class OllamaOptionTranslationTests(unittest.TestCase):
    def test_persisted_and_parser_options_reach_supported_chatollama_fields(self):
        with patch.object(module, "ChatOllama") as chat:
            module._build_ollama_chat(
                base_url="http://ollama:11434",
                model_name="qwen3.6:latest",
                temperature=0,
                max_tokens=99999,
                num_predict=4096,
                timeout=120,
                top_p=0.1,
                reasoning=False,
                response_format="json",
            )

        chat.assert_called_once_with(
            base_url="http://ollama:11434",
            model="qwen3.6:latest",
            temperature=0.0,
            top_p=0.1,
            num_predict=4096,
            client_kwargs={"timeout": 120.0},
            reasoning=False,
            format="json",
        )

    def test_non_positive_limits_fail_before_network_client_construction(self):
        with patch.object(module, "ChatOllama") as chat:
            with self.assertRaisesRegex(ValueError, "num_predict"):
                module._build_ollama_chat(
                    base_url="http://ollama:11434",
                    model_name="qwen3.6:latest",
                    num_predict=0,
                )
        chat.assert_not_called()

    def test_parser_can_force_non_streaming_invoke_for_total_request_timeout(self):
        client = module.LLMClient.__new__(module.LLMClient)
        chat = Mock()
        chat.invoke.return_value = SimpleNamespace(content="{}")
        with patch.object(module, "_build_ollama_chat", return_value=chat):
            result = client._call_ollama_with_config(
                [{"role": "user", "content": "extract"}],
                {
                    "base_url": "http://ollama:11434",
                    "model_name": "qwen3.6:latest",
                    "stream": False,
                },
            )

        self.assertEqual(result, "{}")
        chat.invoke.assert_called_once()
        self.assertEqual(chat.invoke.call_args.kwargs, {"stream": False})


if __name__ == "__main__":
    unittest.main()
