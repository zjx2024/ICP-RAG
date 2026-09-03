import threading
import time
import unittest

from backend.icp.entity_resolution import (
    EntityResolutionRepository,
    UnsafeResolutionError,
    business_entity_key,
    cosine_similarity,
    document_entity_key,
    normalize_business_id,
)


class _ScriptedCursor:
    def __init__(self, connection):
        self.connection = connection
        self.description = None
        self._rows = []
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.connection.executions.append((" ".join(str(sql).split()), params))
        response = self.connection.responses.pop(0) if self.connection.responses else {}
        self._rows = list(response.get("rows", []))
        self.description = response.get("description")
        self.rowcount = response.get("rowcount", len(self._rows))

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    def fetchall(self):
        rows, self._rows = self._rows, []
        return rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _ScriptedConnection:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.executions = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return _ScriptedCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class _OverlapDetectingCursor(_ScriptedCursor):
    def execute(self, sql, params=None):
        with self.connection.state_lock:
            self.connection.active_executions += 1
            self.connection.max_active_executions = max(
                self.connection.max_active_executions,
                self.connection.active_executions,
            )
        try:
            time.sleep(0.02)
            super().execute(sql, params)
        finally:
            with self.connection.state_lock:
                self.connection.active_executions -= 1


class _OverlapDetectingConnection(_ScriptedConnection):
    def __init__(self):
        super().__init__()
        self.state_lock = threading.Lock()
        self.active_executions = 0
        self.max_active_executions = 0

    def cursor(self):
        return _OverlapDetectingCursor(self)


class EntityIdentityTests(unittest.TestCase):
    def test_business_identifier_normalization_preserves_identity_semantics(self):
        self.assertEqual("91810001 ** 0001", normalize_business_id(" 91810001  **  0001 "))
        self.assertEqual("abc-01", normalize_business_id("abc-01"))
        self.assertNotEqual(normalize_business_id("abc-01"), normalize_business_id("ABC-01"))

    def test_business_key_is_scope_aware_and_document_text_never_crosses_documents(self):
        first = business_entity_key("Module", "ICU", {"configuration": "A"})
        second = business_entity_key("Module", "ICU", {"configuration": "B"})
        self.assertNotEqual(first, second)

        doc_a = document_entity_key("doc-a", "FaultCause", "fault_cause", "电容老化")
        doc_b = document_entity_key("doc-b", "FaultCause", "fault_cause", "电容老化")
        self.assertNotEqual(doc_a, doc_b)

    def test_cosine_similarity_is_pure_and_rejects_invalid_vectors(self):
        self.assertAlmostEqual(1.0, cosine_similarity([1.0, 0.0], [2.0, 0.0]))
        self.assertAlmostEqual(0.0, cosine_similarity([1.0, 0.0], [0.0, 1.0]))
        with self.assertRaises(ValueError):
            cosine_similarity([0.0, 0.0], [1.0, 0.0])


class EntityResolutionSchemaTests(unittest.TestCase):
    def test_init_tables_is_idempotent_and_separates_candidates_from_assertions(self):
        connection = _ScriptedConnection()
        repository = EntityResolutionRepository(connection)

        repository.init_tables()
        execution_count = len(connection.executions)
        repository.init_tables()

        sql = "\n".join(statement for statement, _ in connection.executions)
        self.assertGreater(execution_count, 10)
        self.assertEqual(execution_count, len(connection.executions))
        self.assertEqual(1, connection.commits)
        self.assertIn("icp_canonical_entities", sql)
        self.assertIn("icp_entity_mentions", sql)
        self.assertIn("icp_entity_resolution_candidates", sql)
        self.assertIn("icp_relation_assertions", sql)
        self.assertIn("candidate_semantic", sql)
        self.assertIn(
            "UNIQUE (doc_id, relation_type, source_entity_id, target_entity_id)", sql
        )
        self.assertIn("logical_ids", sql)
        self.assertIn("icp_purge_orphan_entity_after_delete", sql)
        self.assertIn("trg_icp_purge_entity_after_mention_delete", sql)
        self.assertIn("trg_icp_purge_entity_after_assertion_delete", sql)
        self.assertIn("NOT EXISTS ( SELECT 1 FROM icp_entity_mentions", sql)
        self.assertIn("NOT EXISTS ( SELECT 1 FROM icp_relation_assertions", sql)


class EntityResolutionOperationTests(unittest.TestCase):
    def _repository(self, responses):
        connection = _ScriptedConnection(responses)
        ids = iter([f"id-{index}" for index in range(1, 20)])
        repository = EntityResolutionRepository(
            connection, id_factory=lambda: next(ids)
        )
        repository._initialized = True
        return repository, connection

    def test_business_entity_uses_exact_id_identity(self):
        returned = {
            "entity_id": "ent-exact",
            "entity_type": "HMCCode",
            "business_id": "91810001**0001",
            "identity_method": "business_id",
            "logical_ids": '["sg:b:hmc:1"]',
        }
        repository, connection = self._repository([{"rows": [returned]}])

        entity = repository.upsert_business_entity(
            entity_type="HMCCode",
            business_id="91810001**0001",
            logical_id="sg:b:hmc:1",
        )

        self.assertEqual("ent-exact", entity["entity_id"])
        self.assertEqual(["sg:b:hmc:1"], entity["logical_ids"])
        sql, params = connection.executions[0]
        self.assertIn("identity_method", sql)
        self.assertIn("'business_id'", sql)
        self.assertEqual("91810001**0001", params[4])

    def test_similarity_vector_only_creates_candidate_and_never_resolves_mention(self):
        returned = {
            "candidate_id": "candidate-1",
            "mention_id": "mention-1",
            "candidate_entity_id": "entity-1",
            "candidate_status": "candidate_semantic",
            "combined_score": 1.0,
        }
        repository, connection = self._repository([{"rows": [returned]}])

        candidate = repository.add_similarity_candidate(
            mention_id="mention-1",
            candidate_entity_id="entity-1",
            mention_embedding=[1.0, 0.0],
            candidate_embedding=[2.0, 0.0],
            resolution_method="embedding",
            model_version="bge-m3",
        )

        self.assertEqual("candidate_semantic", candidate["candidate_status"])
        self.assertEqual(1, len(connection.executions))
        sql, params = connection.executions[0]
        self.assertTrue(sql.startswith("INSERT INTO icp_entity_resolution_candidates"))
        self.assertNotIn("UPDATE icp_entity_mentions", sql)
        self.assertEqual(1.0, params[6])

    def test_similarity_cannot_be_promoted_by_formal_resolution_api(self):
        repository, connection = self._repository([])

        with self.assertRaises(UnsafeResolutionError):
            repository._resolve_mention(
                mention_id="mention-1",
                entity_id="entity-1",
                method="hybrid_similarity",
            )

        self.assertEqual([], connection.executions)

    def test_similarity_cannot_assert_a_relation(self):
        repository, connection = self._repository([])

        with self.assertRaises(UnsafeResolutionError):
            repository.upsert_relation_assertion(
                doc_id="doc-1",
                relation_type="caused_by",
                source_entity_id="fault-1",
                target_entity_id="cause-1",
                assertion_method="embedding",
            )

        self.assertEqual([], connection.executions)

    def test_explicit_reference_is_a_formal_resolution(self):
        returned = {
            "mention_id": "mention-1",
            "resolved_entity_id": "entity-1",
            "resolution_status": "resolved_explicit_reference",
            "resolution_method": "explicit_reference",
        }
        repository, connection = self._repository([{"rows": [returned]}])

        result = repository.resolve_mention_by_explicit_reference(
            mention_id="mention-1",
            entity_id="entity-1",
            reference_value="FM-SYN-002",
            source_ref="table=2;row=3",
        )

        self.assertEqual("resolved_explicit_reference", result["resolution_status"])
        sql, params = connection.executions[0]
        self.assertTrue(sql.startswith("UPDATE icp_entity_mentions AS mention"))
        self.assertEqual("explicit_reference", params[1])

    def test_logical_id_and_logical_edge_queries_keep_pg_readback(self):
        repository, connection = self._repository(
            [
                {"rows": [{"entity_id": "entity-1", "logical_ids": ["sg:b:1"]}]},
                {"rows": [{"assertion_id": "assertion-1"}]},
            ]
        )

        entities = repository.find_entities_by_logical_id("sg:b:1")
        edge = repository.find_relation_by_logical_edge_id(
            doc_id="doc-1", logical_edge_id="sg:e:1"
        )

        self.assertEqual("entity-1", entities[0]["entity_id"])
        self.assertEqual("assertion-1", edge["assertion_id"])
        self.assertIn("logical_ids ? %s", connection.executions[0][0])
        self.assertIn("logical_edge_ids ? %s", connection.executions[1][0])
        self.assertEqual(1, connection.commits)

    def test_candidate_targets_require_authority_from_another_document(self):
        repository, connection = self._repository([{"rows": []}])

        repository.list_candidate_target_entities(
            entity_type="Symptom", exclude_doc_id="doc-current"
        )

        sql, params = connection.executions[0]
        self.assertIn("EXISTS ( SELECT 1 FROM icp_entity_mentions", sql)
        self.assertIn("EXISTS ( SELECT 1 FROM icp_relation_assertions", sql)
        self.assertIn("assertion.assertion_status='asserted'", sql)
        self.assertNotIn("origin_doc_id", sql)
        self.assertEqual(("Symptom", "doc-current", "doc-current", 5000), params)

    def test_orphan_purge_is_document_delete_repair_hook(self):
        repository, connection = self._repository([{"rowcount": 2}])

        deleted = repository.purge_orphan_entities()

        self.assertEqual(2, deleted)
        sql, params = connection.executions[0]
        self.assertIsNone(params)
        self.assertIn("DELETE FROM icp_canonical_entities", sql)
        self.assertIn("NOT EXISTS ( SELECT 1 FROM icp_entity_mentions", sql)
        self.assertIn("NOT EXISTS ( SELECT 1 FROM icp_relation_assertions", sql)
        self.assertEqual(1, connection.commits)

    def test_repository_lock_prevents_shared_connection_interleaving(self):
        connection = _OverlapDetectingConnection()
        repository = EntityResolutionRepository(connection)
        repository._initialized = True
        start = threading.Barrier(3)
        failures = []

        def query(entity_id):
            try:
                start.wait()
                repository.get_entity(entity_id)
            except Exception as exc:  # pragma: no cover - failure detail only
                failures.append(exc)

        threads = [
            threading.Thread(target=query, args=("entity-1",)),
            threading.Thread(target=query, args=("entity-2",)),
        ]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join(timeout=2)

        self.assertFalse(failures)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(1, connection.max_active_executions)


class _RecordingSyncRepository(EntityResolutionRepository):
    def __init__(self, responses=None):
        super().__init__(_ScriptedConnection(responses))
        self._initialized = True
        self.business_entities = []
        self.document_entities = []
        self.mentions = []
        self.formal_resolutions = []
        self.document_resolutions = []
        self.assertions = []

    def upsert_business_entity(self, **kwargs):
        self.business_entities.append(kwargs)
        return {
            "entity_id": "entity:" + kwargs["logical_id"],
            "entity_type": kwargs["entity_type"],
        }

    def upsert_document_scoped_entity(self, **kwargs):
        self.document_entities.append(kwargs)
        return {
            "entity_id": "entity:" + kwargs["logical_id"],
            "entity_type": kwargs["entity_type"],
        }

    def upsert_mention(self, **kwargs):
        self.mentions.append(kwargs)
        return {"mention_id": f"mention-{len(self.mentions)}"}

    def _resolve_mention(self, **kwargs):
        self.formal_resolutions.append(kwargs)
        return {"mention_id": kwargs["mention_id"]}

    def resolve_mention_document_scoped(self, **kwargs):
        self.document_resolutions.append(kwargs)
        return {"mention_id": kwargs["mention_id"]}

    def upsert_relation_assertion(self, **kwargs):
        self.assertions.append(kwargs)
        return {"assertion_id": f"assertion-{len(self.assertions)}"}


class StructuredGraphSyncTests(unittest.TestCase):
    def test_sync_graph_preserves_logical_ids_and_never_similarity_merges_text(self):
        graph = {
            "doc_id": "doc-1",
            "doc_type": "hmc",
            "vertices": [
                {
                    "id": "sg:b:hmc:1",
                    "label": "HMCCode",
                    "properties": {
                        "business_id": "91810001**0001",
                        "name": "91810001**0001",
                        "source_refs": ["table=1;row=2"],
                    },
                },
                {
                    "id": "sg:t:hmc_fault_desc:1",
                    "label": "TextValue",
                    "properties": {
                        "value_kind": "hmc_fault_desc",
                        "text": "ICU 输入过压钳位失效",
                        "source_refs": ["table=1;row=2"],
                    },
                },
                {
                    "id": "sg:b:document:1",
                    "label": "Document",
                    "properties": {"business_id": "doc-1", "name": "HMC.docx"},
                },
            ],
            "edges": [
                {
                    "id": "sg:e:describes_fault:1",
                    "label": "describes_fault",
                    "source": "sg:b:hmc:1",
                    "target": "sg:t:hmc_fault_desc:1",
                    "properties": {
                        "record_key": "91810001**0001",
                        "occurrence_qualifiers": ["row-2"],
                        "source_refs": ["table=1;row=2"],
                    },
                },
                {
                    "id": "sg:e:extracted_from:1",
                    "label": "extracted_from",
                    "source": "sg:b:hmc:1",
                    "target": "sg:b:document:1",
                    "properties": {},
                },
            ],
        }
        repository = _RecordingSyncRepository()

        result = repository.sync_graph(
            doc_id="doc-1",
            doc_type="hmc",
            graph=graph,
            document_metadata={"system_code": "SYN-ICP-9000", "config": "A"},
        )

        self.assertEqual(3, result["entity_count"])
        self.assertEqual(2, result["assertion_count"])
        self.assertEqual(2, len(repository.business_entities))
        self.assertEqual(1, len(repository.document_entities))
        self.assertEqual("Symptom", repository.document_entities[0]["entity_type"])
        self.assertEqual("sg:t:hmc_fault_desc:1", repository.document_entities[0]["logical_id"])
        self.assertEqual(2, len(repository.formal_resolutions))
        self.assertEqual(1, len(repository.document_resolutions))
        self.assertEqual("describes", repository.assertions[0]["relation_type"])
        self.assertEqual(
            "sg:e:describes_fault:1",
            repository.assertions[0]["provenance"]["logical_edge_id"],
        )
        sql = "\n".join(statement for statement, _ in repository.connection.executions)
        self.assertIn("DELETE FROM icp_relation_assertions WHERE doc_id=%s", sql)
        self.assertIn("DELETE FROM icp_entity_mentions WHERE doc_id=%s", sql)
        self.assertIn("SET logical_ids = COALESCE", sql)
        self.assertIn("DELETE FROM icp_canonical_entities", sql)
        self.assertEqual(1, repository.connection.commits)

    def test_formal_diagram_revision_sets_assertion_method_from_validation(self):
        expected_methods = {
            "auto_validated": "auto_validated_diagram",
            "manual_confirmed": "manual_confirmed_diagram",
        }
        for validation_status, expected_method in expected_methods.items():
            with self.subTest(validation_status=validation_status):
                repository = _RecordingSyncRepository()
                graph = {
                    "doc_id": "doc-fta",
                    "doc_type": "fta",
                    "vertices": [
                        {
                            "id": "top",
                            "label": "FTAEvent",
                            "properties": {"name": "顶事件"},
                        },
                        {
                            "id": "gate",
                            "label": "FTAGate",
                            "properties": {"name": "OR"},
                        },
                    ],
                    "edges": [
                        {
                            "id": "diagram-edge",
                            "label": "decomposed_by",
                            "source": "top",
                            "target": "gate",
                            "properties": {
                                "revision_id": f"revision-{validation_status}",
                                "validation_status": validation_status,
                            },
                        }
                    ],
                }

                repository.sync_graph(
                    doc_id="doc-fta", doc_type="fta", graph=graph
                )

                self.assertEqual(
                    expected_method,
                    repository.assertions[0]["assertion_method"],
                )
                self.assertEqual(
                    f"revision-{validation_status}",
                    repository.assertions[0]["provenance"]["source_properties"][
                        "revision_id"
                    ],
                )

    def test_reparse_removes_stale_fields_and_recomputes_logical_ids(self):
        repository = _RecordingSyncRepository(
            [
                {"rows": [{"entity_id": "entity:removed-field"}]},
                {"rowcount": 2},
                {"rowcount": 3},
                {"rowcount": 1},
                {"rowcount": 1},
            ]
        )
        graph = {
            "doc_id": "doc-1",
            "doc_type": "hmc",
            "vertices": [
                {
                    "id": "sg:b:hmc:current",
                    "label": "HMCCode",
                    "properties": {
                        "business_id": "91810001**0001",
                        "name": "91810001**0001",
                    },
                }
            ],
            "edges": [],
        }

        result = repository.sync_graph(doc_id="doc-1", doc_type="hmc", graph=graph)

        self.assertEqual(3, result["stale_mention_count"])
        self.assertEqual(2, result["stale_assertion_count"])
        self.assertEqual(1, result["garbage_collected_entity_count"])
        self.assertEqual(1, len(repository.mentions))
        self.assertEqual(
            "sg:b:hmc:current", repository.mentions[0]["attributes"]["logical_id"]
        )
        update_sql, update_params = repository.connection.executions[-2]
        self.assertIn("SET logical_ids = COALESCE", update_sql)
        self.assertEqual(
            ["entity:removed-field", "entity:sg:b:hmc:current"],
            update_params[0],
        )

    def test_failed_reparse_rolls_back_snapshot_deletions(self):
        repository = _RecordingSyncRepository()
        graph = {
            "doc_id": "doc-1",
            "doc_type": "hmc",
            "vertices": [
                {
                    "id": "sg:b:hmc:1",
                    "label": "HMCCode",
                    "properties": {"business_id": "91810001**0001"},
                }
            ],
            "edges": [
                {
                    "id": "bad-edge",
                    "label": "describes_fault",
                    "source": "sg:b:hmc:1",
                    "target": "missing",
                }
            ],
        }

        with self.assertRaises(ValueError):
            repository.sync_graph(doc_id="doc-1", doc_type="hmc", graph=graph)

        sql = "\n".join(statement for statement, _ in repository.connection.executions)
        self.assertIn("DELETE FROM icp_relation_assertions WHERE doc_id=%s", sql)
        self.assertIn("DELETE FROM icp_entity_mentions WHERE doc_id=%s", sql)
        self.assertEqual(0, repository.connection.commits)
        self.assertEqual(1, repository.connection.rollbacks)


if __name__ == "__main__":
    unittest.main()
