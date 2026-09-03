import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from backend.api.routers import types as types_module
from backend.services import model_service


def _model(model_type="llm"):
    return {
        "name": "本地模型",
        "model_type": model_type,
        "provider": "ollama",
        "api_key": "",
        "api_url": "http://127.0.0.1:11434/api/chat",
        "model_name": "qwen-local",
        "base_url": "http://127.0.0.1:11434",
        "temperature": 0.1,
        "max_tokens": 2048,
        "top_p": 0.9,
        "timeout": 120,
    }


class OptionalLlmResolverTests(unittest.TestCase):
    def test_missing_default_is_optional_only_for_optional_resolver(self):
        with patch.object(
            model_service.postgres_client,
            "get_default_model_by_type",
            return_value=None,
        ):
            self.assertIsNone(model_service.resolve_optional_llm_model_config(None))
            with self.assertRaises(HTTPException) as captured:
                model_service.resolve_llm_model_config(None)
        self.assertEqual(400, captured.exception.status_code)

    def test_default_llm_is_returned_when_available(self):
        default = _model()
        with patch.object(
            model_service.postgres_client,
            "get_default_model_by_type",
            return_value=default,
        ):
            config = model_service.resolve_optional_llm_model_config(None)
        self.assertEqual("ollama", config["provider"])
        self.assertEqual("qwen-local", config["model_name"])

    def test_explicit_model_id_keeps_not_found_and_type_validation(self):
        with patch.object(model_service.postgres_client, "get_model", return_value=None):
            with self.assertRaises(HTTPException) as missing:
                model_service.resolve_optional_llm_model_config("missing-id")
        self.assertEqual(404, missing.exception.status_code)

        with patch.object(
            model_service.postgres_client,
            "get_model",
            return_value=_model("embedding"),
        ):
            with self.assertRaises(HTTPException) as wrong_type:
                model_service.resolve_optional_llm_model_config("embedding-id")
        self.assertEqual(400, wrong_type.exception.status_code)


class IcpParsePreflightTests(unittest.TestCase):
    @staticmethod
    def _close_background(coroutine):
        # The route only schedules the task. Close the mocked coroutine so the
        # unit test does not leak a never-awaited coroutine.
        coroutine.close()
        return object()

    def test_no_default_models_still_schedules_deterministic_icp_parse(self):
        document = {
            "doc_id": "doc-1",
            "doc_type": "hmc",
            "file_name": "HMC.xlsx",
            "status": "uploaded",
        }
        background = AsyncMock(return_value=None)
        with (
            patch.object(types_module, "_get_icp_document", return_value=document),
            patch.object(
                types_module,
                "resolve_optional_llm_model_config",
                return_value=None,
            ) as resolver,
            patch.object(
                model_service,
                "require_embedding_model_configured",
                side_effect=AssertionError("ICP parse must not require embedding preflight"),
            ) as embedding_preflight,
            patch.object(
                types_module.postgres_client,
                "transition_icp_document_status",
                return_value=True,
            ),
            patch.object(
                types_module.task_manager,
                "create_task",
                new=AsyncMock(return_value="task-1"),
            ),
            patch.object(types_module, "parse_icp_document_background", background),
            patch.object(
                types_module.asyncio,
                "create_task",
                side_effect=self._close_background,
            ),
        ):
            result = asyncio.run(
                types_module.parse_icp_document("hmc", "doc-1", {})
            )

        self.assertEqual("parsing", result["status"])
        resolver.assert_called_once_with(None)
        embedding_preflight.assert_not_called()
        background.assert_called_once_with("task-1", "doc-1", None)

    def test_explicit_invalid_model_stops_before_state_transition(self):
        document = {
            "doc_id": "doc-1",
            "doc_type": "hmc",
            "file_name": "HMC.xlsx",
            "status": "uploaded",
        }
        invalid = HTTPException(status_code=404, detail="模型不存在: missing-id")
        with (
            patch.object(types_module, "_get_icp_document", return_value=document),
            patch.object(
                types_module,
                "resolve_optional_llm_model_config",
                side_effect=invalid,
            ),
            patch.object(
                types_module.postgres_client,
                "transition_icp_document_status",
            ) as transition,
            patch.object(
                types_module.task_manager,
                "create_task",
                new=AsyncMock(),
            ) as create_task,
        ):
            with self.assertRaises(HTTPException) as captured:
                asyncio.run(
                    types_module.parse_icp_document(
                        "hmc",
                        "doc-1",
                        {"model_id": "missing-id"},
                    )
                )

        self.assertEqual(404, captured.exception.status_code)
        transition.assert_not_called()
        create_task.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
