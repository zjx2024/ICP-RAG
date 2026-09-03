import unittest
from unittest.mock import patch

from fastapi import HTTPException

from backend.api.routers import icp_domain as icp_domain_router
from backend.icp.graph_authority import GraphRelationAuthority


def _request(**updates):
    value = {
        "authorization_key": "edge:physical-1",
        "physical_edge_id": "physical-1",
        "logical_edge_id": "sg:e:1",
        "relation": "contains_component",
        "source_logical_id": "sg:b:system:1",
        "target_logical_id": "sg:b:module:1",
        "source_physical_id": "icp:ce:entity-system-a",
        "target_physical_id": "icp:ce:entity-module-a",
        "kb_id": "icp_structured",
        "doc_ids": ["doc-1"],
        "revision_id": "",
    }
    value.update(updates)
    return value


def _assertion(**updates):
    value = {
        "assertion_id": "assertion-1",
        "assertion_status": "asserted",
        "relation_type": "contains",
        "source_entity_id": "entity-system-a",
        "target_entity_id": "entity-module-a",
        "logical_edge_ids": ["sg:e:1"],
        "source_ref": "架构.docx#table=1;row=2",
        "provenance": {
            "source_relation": "contains_component",
            "source_logical_id": "sg:b:system:1",
            "target_logical_id": "sg:b:module:1",
        },
    }
    value.update(updates)
    return value


class _EntityRepository:
    def __init__(self, assertions=None, *, failure=None):
        self.assertions = dict(assertions or {})
        self.failure = failure
        self.calls = []

    def find_relation_by_logical_edge_id(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        return self.assertions.get((kwargs["doc_id"], kwargs["logical_edge_id"]))


class GraphRelationAuthorityTests(unittest.TestCase):
    def _authority(self, *, assertions=None, documents=None, revisions=None, failure=None):
        repository = _EntityRepository(assertions, failure=failure)
        documents = dict(documents or {})
        revisions = dict(revisions or {})
        return (
            GraphRelationAuthority(
                entity_repository=repository,
                document_loader=lambda doc_id: documents.get(doc_id),
                revision_loader=lambda revision_id: revisions.get(revision_id),
            ),
            repository,
        )

    def test_structured_edge_requires_exact_current_assertion_and_endpoints(self):
        authority, repository = self._authority(
            assertions={("doc-1", "sg:e:1"): _assertion()},
            documents={
                "doc-1": {
                    "doc_id": "doc-1",
                    "doc_type": "function_locator",
                    "status": "kg_done",
                }
            },
        )

        decision = authority([_request()])["edge:physical-1"]

        self.assertTrue(decision["authorized"])
        self.assertEqual(["assertion-1"], decision["assertion_ids"])
        self.assertEqual(["架构.docx#table=1;row=2"], decision["source_refs"])
        self.assertEqual(1, len(repository.calls))

        wrong_endpoint = authority(
            [_request(target_physical_id="icp:ce:entity-module-other")]
        )["edge:physical-1"]
        self.assertFalse(wrong_endpoint["authorized"])

    def test_merged_logical_markers_require_one_covering_assertion(self):
        merged_assertion = _assertion(
            logical_edge_ids=["sg:e:1", "sg:e:2"]
        )
        authority, repository = self._authority(
            assertions={
                ("doc-1", "sg:e:1"): merged_assertion,
                ("doc-1", "sg:e:2"): merged_assertion,
            },
            documents={
                "doc-1": {
                    "doc_id": "doc-1",
                    "doc_type": "function_locator",
                    "status": "kg_done",
                }
            },
        )
        request = _request(
            logical_edge_id="",
            logical_edge_ids=["sg:e:1", "sg:e:2"],
            logical_edge_ids_by_doc={"doc-1": ["sg:e:1", "sg:e:2"]},
        )

        accepted = authority([request])["edge:physical-1"]
        forged = authority(
            [
                {
                    **request,
                    "logical_edge_ids": ["sg:e:1", "sg:e:2", "sg:e:forged"],
                    "logical_edge_ids_by_doc": {
                        "doc-1": ["sg:e:1", "sg:e:2", "sg:e:forged"]
                    },
                }
            ]
        )["edge:physical-1"]

        self.assertTrue(accepted["authorized"])
        self.assertFalse(forged["authorized"])
        self.assertGreaterEqual(len(repository.calls), 2)

    def test_merged_cross_document_markers_are_authorized_per_owner(self):
        assertion_a = _assertion(
            assertion_id="assertion-a", logical_edge_ids=["sg:e:a"]
        )
        assertion_b = _assertion(
            assertion_id="assertion-b", logical_edge_ids=["sg:e:b"]
        )
        authority, _ = self._authority(
            assertions={
                ("doc-a", "sg:e:a"): assertion_a,
                ("doc-b", "sg:e:b"): assertion_b,
            },
            documents={
                "doc-a": {
                    "doc_id": "doc-a",
                    "doc_type": "function_locator",
                    "status": "kg_done",
                },
                "doc-b": {
                    "doc_id": "doc-b",
                    "doc_type": "function_locator",
                    "status": "kg_done",
                },
            },
        )

        decision = authority(
            [
                _request(
                    logical_edge_id="",
                    doc_ids=["doc-a", "doc-b"],
                    logical_edge_ids=["sg:e:a", "sg:e:b"],
                    logical_edge_ids_by_doc={
                        "doc-a": ["sg:e:a"],
                        "doc-b": ["sg:e:b"],
                    },
                )
            ]
        )["edge:physical-1"]

        self.assertTrue(decision["authorized"])
        self.assertEqual(["assertion-a", "assertion-b"], decision["assertion_ids"])

    def test_every_claimed_document_must_exist_be_ready_and_assert_the_edge(self):
        authority, _ = self._authority(
            assertions={("doc-1", "sg:e:1"): _assertion()},
            documents={
                "doc-1": {
                    "doc_id": "doc-1",
                    "doc_type": "function_locator",
                    "status": "kg_done",
                },
                "doc-2": {
                    "doc_id": "doc-2",
                    "doc_type": "function_locator",
                    "status": "failed",
                },
            },
        )

        failed = authority([_request(doc_ids=["doc-1", "doc-2"])])[
            "edge:physical-1"
        ]
        missing = authority([_request(doc_ids=["doc-missing"])])[
            "edge:physical-1"
        ]

        self.assertFalse(failed["authorized"])
        self.assertFalse(missing["authorized"])

    def test_only_kg_done_document_can_authorize_a_relation(self):
        for status in ("indexed", "kg_pending", "failed"):
            with self.subTest(status=status):
                authority, _ = self._authority(
                    assertions={("doc-1", "sg:e:1"): _assertion()},
                    documents={
                        "doc-1": {
                            "doc_id": "doc-1",
                            "doc_type": "function_locator",
                            "status": status,
                        }
                    },
                )
                decision = authority([_request()])["edge:physical-1"]
                self.assertFalse(decision["authorized"])

        ready, _ = self._authority(
            assertions={("doc-1", "sg:e:1"): _assertion()},
            documents={
                "doc-1": {
                    "doc_id": "doc-1",
                    "doc_type": "function_locator",
                    "status": "kg_done",
                }
            },
        )
        self.assertTrue(ready([_request()])["edge:physical-1"]["authorized"])

    def test_structured_revision_comes_from_assertion_and_request_can_only_match(self):
        assertion = _assertion(
            provenance={
                **_assertion()["provenance"],
                "revision_id": "rev-authoritative",
            }
        )
        authoritative_revision = {
            "revision_id": "rev-authoritative",
            "doc_id": "doc-1",
            "diagram_type": "fta",
            "is_active": True,
            "projection_status": "published",
            "validation_status": "auto_validated",
        }
        attacker_revision = {
            **authoritative_revision,
            "revision_id": "rev-from-request",
        }
        authority, _ = self._authority(
            assertions={("doc-1", "sg:e:1"): assertion},
            documents={
                "doc-1": {
                    "doc_id": "doc-1",
                    "doc_type": "fta",
                    "status": "kg_done",
                }
            },
            revisions={
                "rev-authoritative": authoritative_revision,
                "rev-from-request": attacker_revision,
            },
        )

        no_projection_claim = authority([_request(revision_id="")])[
            "edge:physical-1"
        ]
        matching_claim = authority(
            [_request(revision_id="rev-authoritative")]
        )["edge:physical-1"]
        forged_claim = authority([_request(revision_id="rev-from-request")])[
            "edge:physical-1"
        ]

        self.assertTrue(no_projection_claim["authorized"])
        self.assertEqual("rev-authoritative", no_projection_claim["revision_id"])
        self.assertEqual(["rev-authoritative"], no_projection_claim["revision_ids"])
        self.assertTrue(matching_claim["authorized"])
        self.assertFalse(forged_claim["authorized"])

    def test_request_revision_cannot_fill_missing_assertion_provenance(self):
        authority, _ = self._authority(
            assertions={("doc-1", "sg:e:1"): _assertion()},
            documents={
                "doc-1": {
                    "doc_id": "doc-1",
                    "doc_type": "function_locator",
                    "status": "kg_done",
                }
            },
            # Even a real revision row must not turn an untrusted HugeGraph
            # marker into assertion provenance.
            revisions={
                "rev-from-request": {
                    "revision_id": "rev-from-request",
                    "doc_id": "doc-1",
                    "diagram_type": "fta",
                    "is_active": True,
                    "projection_status": "published",
                    "validation_status": "auto_validated",
                }
            },
        )

        decision = authority([_request(revision_id="rev-from-request")])[
            "edge:physical-1"
        ]

        self.assertFalse(decision["authorized"])

    def test_legacy_engineering_edge_requires_exact_active_formal_revision(self):
        base_revision = {
            "revision_id": "rev-1",
            "doc_id": "doc-fta",
            "diagram_type": "fta",
            "is_active": True,
            "projection_status": "published",
            "validation_status": "auto_validated",
            "canonical_graph": {
                "edges": [
                    {
                        "id": "e1",
                        "label": "causes",
                        "source": "top",
                        "target": "basic",
                        "properties": {"source_ref": "FTA.docx#figure=2"},
                    }
                ]
            },
        }
        authority, _ = self._authority(
            documents={
                "doc-fta": {
                    "doc_id": "doc-fta",
                    "doc_type": "fta",
                    "status": "kg_done",
                }
            },
            revisions={"rev-1": base_revision},
        )
        request = _request(
            logical_edge_id="e1",
            relation="causes",
            source_logical_id="top",
            target_logical_id="basic",
            kb_id="fta",
            doc_ids=["doc-fta"],
            revision_id="rev-1",
        )

        decision = authority([request])["edge:physical-1"]

        self.assertTrue(decision["authorized"])
        self.assertEqual("auto_validated", decision["validation_status"])
        self.assertEqual(["FTA.docx#figure=2"], decision["source_refs"])

    def test_provisional_quarantined_inactive_and_stale_revisions_are_denied(self):
        document = {
            "doc-id": None,
            "doc_id": "doc-fta",
            "doc_type": "fta",
            "status": "kg_done",
        }
        request = _request(
            logical_edge_id="e1",
            relation="causes",
            source_logical_id="top",
            target_logical_id="basic",
            kb_id="fta",
            doc_ids=["doc-fta"],
            revision_id="rev-1",
        )
        base = {
            "revision_id": "rev-1",
            "doc_id": "doc-fta",
            "diagram_type": "fta",
            "is_active": True,
            "projection_status": "published",
            "validation_status": "auto_validated",
            "canonical_graph": {
                "edges": [
                    {
                        "id": "e1",
                        "label": "causes",
                        "source": "top",
                        "target": "basic",
                    }
                ]
            },
        }
        variants = (
            {"validation_status": "provisional"},
            {"validation_status": "quarantined"},
            {"is_active": False},
            {"projection_status": "stale"},
            {"canonical_graph": {"edges": []}},
        )
        for update in variants:
            with self.subTest(update=update):
                revision = {**base, **update}
                authority, _ = self._authority(
                    documents={"doc-fta": document},
                    revisions={"rev-1": revision},
                )
                decision = authority([request])["edge:physical-1"]
                self.assertFalse(decision["authorized"])

    def test_fta_structured_assertion_also_requires_formal_revision_provenance(self):
        assertion = _assertion(
            provenance={
                **_assertion()["provenance"],
                "revision_id": "rev-1",
            }
        )
        revision = {
            "revision_id": "rev-1",
            "doc_id": "doc-1",
            "diagram_type": "fta",
            "is_active": True,
            "projection_status": "published",
            "validation_status": "provisional",
        }
        authority, _ = self._authority(
            assertions={("doc-1", "sg:e:1"): assertion},
            documents={
                "doc-1": {
                    "doc_id": "doc-1",
                    "doc_type": "fta",
                    "status": "kg_done",
                }
            },
            revisions={"rev-1": revision},
        )

        decision = authority([_request()])["edge:physical-1"]

        self.assertFalse(decision["authorized"])

    def test_authority_backend_exception_is_not_converted_into_authorization(self):
        authority, _ = self._authority(
            documents={
                "doc-1": {
                    "doc_id": "doc-1",
                    "doc_type": "function_locator",
                    "status": "kg_done",
                }
            },
            failure=RuntimeError("PostgreSQL unavailable"),
        )

        with self.assertRaisesRegex(RuntimeError, "PostgreSQL unavailable"):
            authority([_request()])


class GraphTraversalApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_api_forces_reauthorization_and_preserves_core_limits(self):
        captured = {}

        def traverse(request, **kwargs):
            captured.update(request)
            captured["callback"] = kwargs.get("reauthorize_relations")
            return {
                "status": "empty",
                "nodes": [],
                "edges": [],
                "paths": [],
                "evidence": [],
                "diagnostics": {"warnings": [], "errors": []},
            }

        with patch.object(icp_domain_router, "run_graph_traversal", side_effect=traverse):
            result = await icp_domain_router.traverse_authoritative_graph(
                {
                    "mode": "neighbors",
                    "seed_ids": ["sg:b:1"],
                    "kb_ids": ["icp_structured"],
                    "require_relation_reauthorization": False,
                }
            )

        self.assertEqual("empty", result["status"])
        self.assertTrue(captured["require_relation_reauthorization"])
        self.assertIs(captured["callback"], icp_domain_router.reauthorize_graph_relations)

    async def test_api_rejects_unknown_projection_and_maps_blocked_request(self):
        with self.assertRaises(HTTPException) as unknown:
            await icp_domain_router.traverse_authoritative_graph(
                {"seed_ids": ["x"], "kb_ids": ["untrusted"]}
            )
        self.assertEqual(422, unknown.exception.status_code)

        blocked_result = {
            "status": "blocked",
            "diagnostics": {"errors": ["max_depth 必须在 1..5 范围内"]},
        }
        with patch.object(
            icp_domain_router, "run_graph_traversal", return_value=blocked_result
        ):
            with self.assertRaises(HTTPException) as blocked:
                await icp_domain_router.traverse_authoritative_graph(
                    {"seed_ids": ["x"], "max_depth": 99}
                )
        self.assertEqual(422, blocked.exception.status_code)
        self.assertIn("1..5", blocked.exception.detail)


if __name__ == "__main__":
    unittest.main()
