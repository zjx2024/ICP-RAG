import unittest
from unittest.mock import AsyncMock, patch

# Follow the application's aggregate import order to avoid the retained legacy
# documents-router/service compatibility cycle during isolated test imports.
import backend.api  # noqa: F401
from backend.services import kg_service


class _TaskManager:
    def __init__(self):
        self.statuses = []
        self.progress = []
        self.result = None
        self.error = None

    async def update_status(self, task_id, status):
        self.statuses.append(status)

    async def update_progress(self, task_id, progress, message, **kwargs):
        self.progress.append((progress, message))

    async def set_result(self, task_id, result):
        self.result = result

    async def set_error(self, task_id, error):
        self.error = error


class _Postgres:
    def __init__(self):
        self.statuses = []

    def update_icp_document_status(self, doc_id, status):
        self.statuses.append((doc_id, status))


class _Repository:
    def __init__(self, revision=None, validation=None):
        self.revision = revision
        self.validation = validation or {
            "errors": [],
            "quarantined_revision_ids": [],
            "results": [],
        }

    def auto_validate_document(self, doc_id, diagram_type=None):
        return dict(self.validation)

    def list_diagrams_by_document(self, doc_id):
        return (
            [{"diagram_id": "diagram-1", "diagram_type": self.revision["diagram_type"]}]
            if self.revision
            else []
        )

    def list_revisions(self, diagram_id):
        return [dict(self.revision)] if self.revision else []


class AutomaticGraphPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_table_document_projects_without_diagram_or_human_gate(self):
        tasks = _TaskManager()
        postgres = _Postgres()
        structured = {
            "doc_id": "doc-1",
            "doc_type": "hmc",
            "status": "published",
            "vertices": 4,
            "edges": 3,
            "warnings": [],
        }
        with patch.object(kg_service, "task_manager", tasks), patch.object(
            kg_service, "postgres_client", postgres
        ), patch(
            "backend.icp.structured_projection.project_structured_document",
            return_value=structured,
        ):
            await kg_service.generate_icp_graph_background(
                "task-1", "doc-1", "hmc", "hmc.xlsx"
            )

        self.assertEqual("completed", tasks.statuses[-1])
        self.assertEqual("kg_done", tasks.result["status"])
        self.assertEqual(4, tasks.result["vertices"])
        self.assertEqual(("doc-1", "kg_done"), postgres.statuses[-1])

    async def test_auto_validated_diagram_is_forwarded_to_verified_projector(self):
        revision = {
            "revision_id": "revision-1",
            "diagram_id": "diagram-1",
            "diagram_type": "fta",
            "validation_status": "auto_validated",
            "review_status": "pending",
            "projection_status": "not_projected",
            "is_active": False,
        }
        repository = _Repository(
            revision,
            {
                "errors": [],
                "quarantined_revision_ids": [],
                "results": [revision],
            },
        )
        downstream = AsyncMock()
        with patch.object(kg_service, "task_manager", _TaskManager()), patch.object(
            kg_service, "postgres_client", _Postgres()
        ), patch.object(kg_service, "diagram_repository", repository), patch.object(
            kg_service, "generate_icp_kg_background", downstream
        ), patch(
            "backend.icp.structured_projection.project_structured_document",
            return_value={"vertices": 1, "edges": 0, "warnings": []},
        ):
            await kg_service.generate_icp_graph_background(
                "task-1", "doc-1", "fta", "fta.docx"
            )

        downstream.assert_awaited_once()
        self.assertEqual(
            ["revision-1"], downstream.await_args.kwargs["revision_ids"]
        )

    async def test_quarantined_visual_diagram_finishes_without_fabricating_edges(self):
        revision = {
            "revision_id": "revision-1",
            "diagram_id": "diagram-1",
            "diagram_type": "rbd",
            "validation_status": "quarantined",
            "review_status": "pending",
            "projection_status": "not_projected",
            "is_active": False,
        }
        repository = _Repository(
            revision,
            {
                "errors": [],
                "quarantined_revision_ids": ["revision-1"],
                "results": [revision],
            },
        )
        tasks = _TaskManager()
        postgres = _Postgres()
        downstream = AsyncMock()
        with patch.object(kg_service, "task_manager", tasks), patch.object(
            kg_service, "postgres_client", postgres
        ), patch.object(kg_service, "diagram_repository", repository), patch.object(
            kg_service, "generate_icp_kg_background", downstream
        ), patch(
            "backend.icp.structured_projection.project_structured_document",
            return_value={"vertices": 1, "edges": 0, "warnings": []},
        ):
            await kg_service.generate_icp_graph_background(
                "task-1", "doc-1", "rbd", "rbd.docx"
            )

        downstream.assert_not_awaited()
        self.assertEqual("completed", tasks.statuses[-1])
        self.assertEqual("kg_done", tasks.result["status"])
        self.assertEqual(
            ["revision-1"],
            tasks.result["automatic_validation"]["quarantined_revision_ids"],
        )
        self.assertEqual(("doc-1", "kg_done"), postgres.statuses[-1])


if __name__ == "__main__":
    unittest.main()
