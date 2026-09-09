#!/usr/bin/env python3
"""Run a safe, repeatable synthetic seven-document ingestion smoke test.

The files under ``test-docs`` are generated parser fixtures.  A successful run
only demonstrates that the configured local stack can ingest those fixtures;
it does not validate real maintenance data or an airworthiness workflow.

Safety properties:

* readiness, the exact seven-type registry and Dify route mounting are checked
  before the first upload;
* an existing document is never uploaded again when its SHA-256 (when exposed
  by the server) or its type-local filename matches;
* existing records are never deleted, and filename-only matches in an
  unparsed/failed state are not modified because the current API does not
  expose the source artifact hash;
* parsing must schedule and finish the automatic HugeGraph projection for all
  seven document types; the smoke never calls manual confirm/reject endpoints;
* newly parsed architecture/FTA/RBD revisions must finish as
  ``auto_validated``, ``provisional`` or ``quarantined``.  Publishable revisions
  must be active after read-back, while quarantined revisions remain isolated;
* ``--cleanup-created`` can delete only document IDs returned by uploads made
  by the current process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


EXPECTED_DOCUMENTS: Tuple[Tuple[str, str], ...] = (
    ("function_locator", "01_系统架构与功能说明.docx"),
    ("hmc", "02_HMC故障代码手册.docx"),
    ("fmeca", "03_FMECA分析报告.docx"),
    ("fta", "04_FTA故障树分析报告.docx"),
    ("rbd", "05_RBD可靠性框图分析报告.docx"),
    ("mfl", "06_MFL维修故障清单.docx"),
    ("fault_history", "07_故障历史查询文档.docx"),
)
EXPECTED_TYPES = tuple(doc_type for doc_type, _ in EXPECTED_DOCUMENTS)

READY_DOCUMENT_STATUSES = {"kg_done"}
ACTIVE_DOCUMENT_STATUSES = {"parsing", "generating_kg", "deleting"}
TERMINAL_TASK_STATUSES = {"completed", "failed", "cancelled"}
DIAGRAM_DOCUMENT_TYPES = frozenset({"function_locator", "fta", "rbd"})
AUTOMATIC_DIAGRAM_STATUSES = frozenset(
    {"auto_validated", "provisional", "quarantined"}
)
PUBLISHABLE_DIAGRAM_STATUSES = frozenset(
    {"auto_validated", "provisional", "manual_confirmed"}
)
REQUIRED_POSITIVE_STAT = {
    "function_locator": "functions_written",
    "hmc": "rows_ok",
    "fmeca": "rows_ok",
    "fta": "present_diagrams",
    "rbd": "present_diagrams",
    "mfl": "rows_ok",
    "fault_history": "cases",
}
SUMMARY_STATS = (
    "components_written",
    "functions_written",
    "interfaces_written",
    "rows_ok",
    "rows_failed",
    "cases",
    "steps",
    "images",
    "graph_count",
    "nodes",
    "edges",
    "present_diagrams",
    "review_pending",
    "review_status_counts",
    "projection_status_counts",
    "vectors",
    "projection_rebuild_required",
)


class SmokeError(RuntimeError):
    """The smoke run cannot safely continue or a required check failed."""


class ApiError(SmokeError):
    def __init__(self, method: str, path: str, status: Optional[int], detail: str):
        self.method = method
        self.path = path
        self.status = status
        self.detail = detail
        status_text = str(status) if status is not None else "transport"
        super().__init__(f"{method} {path} failed ({status_text}): {detail}")


@dataclass(frozen=True)
class Fixture:
    doc_type: str
    path: Path
    sha256: str

    @property
    def file_name(self) -> str:
        return self.path.name


class JsonApiClient:
    """Small stdlib-only JSON/multipart client used by the smoke runner."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        parsed = urlsplit(base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("--base-url must be an absolute http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("--base-url must not contain credentials")
        if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("--base-url must contain only scheme, host and optional port")
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        self.timeout = timeout
        # This smoke validates the explicitly supplied service, normally a
        # loopback endpoint.  urllib otherwise honors HTTP(S)_PROXY and can
        # turn a healthy local readiness response into a proxy-generated 502.
        self._opener = build_opener(ProxyHandler({}))

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[bytes] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Any:
        request = Request(
            self.base_url + path,
            data=body,
            headers={"Accept": "application/json", **dict(headers or {})},
            method=method,
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            raw = exc.read()
            detail = _response_detail(raw)
            raise ApiError(method, path, exc.code, detail) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ApiError(method, path, None, str(exc)) from exc
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(method, path, None, "response was not valid UTF-8 JSON") from exc

    def get(self, path: str) -> Any:
        return self._request("GET", path)

    def post_json(self, path: str, payload: Mapping[str, Any]) -> Any:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self._request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json"},
        )

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    def upload_file(self, path: str, file_path: Path) -> Any:
        boundary = f"icp-smoke-{uuid.uuid4().hex}"
        content_type = (
            mimetypes.guess_type(file_path.name)[0]
            or "application/octet-stream"
        )
        # Starlette/python-multipart currently ignores RFC 5987 ``filename*``
        # when a plain filename is also present.  Browsers send the UTF-8
        # filename directly, and Starlette decodes it with the multipart
        # charset; doing the same keeps the server-side Chinese evidence name.
        disposition = (
            f'Content-Disposition: form-data; name="files"; '
            f'filename="{file_path.name}"\r\n'
        )
        body = b"".join(
            (
                f"--{boundary}\r\n".encode("ascii"),
                disposition.encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
                file_path.read_bytes(),
                b"\r\n",
                f"--{boundary}--\r\n".encode("ascii"),
            )
        )
        return self._request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}; charset=utf-8"
            },
        )


def _response_detail(raw: bytes) -> str:
    if not raw:
        return "empty response"
    try:
        payload = json.loads(raw.decode("utf-8"))
        if isinstance(payload, dict):
            value = payload.get("detail") or payload.get("error") or payload
        else:
            value = payload
        return str(value)[:1000]
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw.decode("utf-8", errors="replace")[:1000]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixtures(test_docs: Path) -> List[Fixture]:
    root = test_docs.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise SmokeError(f"test-docs path is not a directory: {root}")
    fixtures: List[Fixture] = []
    seen_hashes: Dict[str, str] = {}
    for doc_type, file_name in EXPECTED_DOCUMENTS:
        path = root / file_name
        if not path.is_file():
            raise SmokeError(f"missing synthetic fixture for {doc_type}: {path}")
        if path.suffix.casefold() != ".docx":
            raise SmokeError(f"synthetic fixture is not a .docx file: {path}")
        sha256 = _sha256_file(path)
        previous = seen_hashes.get(sha256)
        if previous:
            raise SmokeError(
                f"fixtures {previous!r} and {file_name!r} have the same SHA-256"
            )
        seen_hashes[sha256] = file_name
        fixtures.append(Fixture(doc_type=doc_type, path=path, sha256=sha256))
    return fixtures


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _remote_sha256(document: Mapping[str, Any]) -> Optional[str]:
    metadata = _as_dict(document.get("metadata"))
    candidates = (
        document.get("sha256"),
        document.get("file_sha256"),
        document.get("content_sha256"),
        metadata.get("sha256"),
        metadata.get("file_sha256"),
        metadata.get("content_sha256"),
    )
    for value in candidates:
        normalized = str(value or "").strip().casefold()
        if len(normalized) == 64 and all(char in "0123456789abcdef" for char in normalized):
            return normalized
    return None


def _projection_failures(stats: Mapping[str, Any]) -> int:
    failures = stats.get("projection_failed")
    if isinstance(failures, (list, tuple, set, dict)):
        return len(failures)
    if failures:
        try:
            return int(failures)
        except (TypeError, ValueError):
            return 1
    return 0


class SmokeRunner:
    def __init__(
        self,
        client: Any,
        fixtures: Sequence[Fixture],
        *,
        poll_interval: float = 1.0,
        task_timeout: float = 900.0,
        cleanup_created: bool = False,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ):
        self.client = client
        self.fixtures = list(fixtures)
        self.poll_interval = poll_interval
        self.task_timeout = task_timeout
        self.cleanup_created = cleanup_created
        self.sleep = sleep
        self.monotonic = monotonic
        self.created: List[Tuple[str, str]] = []

    def run(self, *, base_url: str, test_docs: str) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "kind": "synthetic_icp_ingestion_smoke",
            "ok": False,
            "base_url": base_url,
            "test_docs": test_docs,
            "checks": {},
            "documents": [],
            "safety": {
                "synthetic_fixtures_only": True,
                "manual_graph_review_actions": 0,
                "explicit_graph_generation_actions": 0,
                "automatic_graph_tasks_observed": 0,
                "cleanup_created_requested": self.cleanup_created,
                "created_doc_ids": [],
                "cleanup": [],
            },
            "errors": [],
        }
        preflight_ok = False
        try:
            self._preflight(summary["checks"])
            preflight_ok = True
            for fixture in self.fixtures:
                try:
                    result = self._process_fixture(fixture)
                except Exception as exc:  # keep other types independently observable
                    result = {
                        "doc_type": fixture.doc_type,
                        "file_name": fixture.file_name,
                        "sha256": fixture.sha256,
                        "ok": False,
                        "error": str(exc),
                    }
                summary["documents"].append(result)
                if result.get("graph_task_id"):
                    summary["safety"]["automatic_graph_tasks_observed"] += 1
                if not result.get("ok"):
                    summary["errors"].append(
                        f"{fixture.doc_type}: {result.get('error') or '; '.join(result.get('issues') or [])}"
                    )
            try:
                summary["checks"]["readiness_final"] = self._check_readiness()
            except Exception as exc:
                summary["checks"]["readiness_final"] = {"ok": False, "error": str(exc)}
                summary["errors"].append(f"final readiness: {exc}")
        except Exception as exc:
            summary["errors"].append(f"preflight: {exc}")
        finally:
            summary["safety"]["created_doc_ids"] = [doc_id for _, doc_id in self.created]
            if self.cleanup_created:
                cleanup_results = self._cleanup_created()
                summary["safety"]["cleanup"] = cleanup_results
                for result in cleanup_results:
                    if not result.get("ok"):
                        summary["errors"].append(
                            f"cleanup {result.get('doc_type')}/{result.get('doc_id')}: {result.get('error')}"
                        )

        documents_ok = (
            len(summary["documents"]) == len(EXPECTED_DOCUMENTS)
            and all(item.get("ok") for item in summary["documents"])
        )
        summary["ok"] = bool(preflight_ok and documents_ok and not summary["errors"])
        return summary

    def _preflight(self, checks: Dict[str, Any]) -> None:
        checks["readiness_initial"] = self._check_readiness()

        registry = self.client.get("/api/v1/document-types")
        if not isinstance(registry, list):
            raise SmokeError("document type registry was not a JSON list")
        actual_types = [str(item.get("doc_type") or "") for item in registry if isinstance(item, dict)]
        if len(actual_types) != 7 or set(actual_types) != set(EXPECTED_TYPES):
            raise SmokeError(
                f"active registry must contain exactly the seven ICP types; got {actual_types}"
            )
        checks["document_types"] = {"ok": True, "types": actual_types}

        openapi = self.client.get("/openapi.json")
        paths = _as_dict(openapi).get("paths")
        if not isinstance(paths, dict):
            raise SmokeError("OpenAPI document did not contain a paths object")
        # Dify 与通用检索链路已退役；改为确认 ICP 前端依赖的任务监控路由仍挂载。
        required_routes = {
            "/api/v1/knowledge-bases/tasks/{task_id}/status": "get",
            "/api/v1/knowledge-bases/tasks/{task_id}/cancel": "post",
        }
        for path, method in required_routes.items():
            operations = paths.get(path)
            if not isinstance(operations, dict) or method not in operations:
                raise SmokeError(f"route missing from OpenAPI: {method.upper()} {path}")
        checks["task_routes"] = {
            "ok": True,
            "routes": sorted(required_routes),
        }

    def _check_readiness(self) -> Dict[str, Any]:
        payload = self.client.get("/readyz")
        if not isinstance(payload, dict) or payload.get("status") != "ready":
            raise SmokeError(f"service is not ready: {payload}")
        checks = payload.get("checks")
        if isinstance(checks, dict):
            failed = [name for name, value in checks.items() if not _as_dict(value).get("ok")]
            if failed:
                raise SmokeError(f"readiness dependencies failed: {failed}")
        return {"ok": True, "dependencies": sorted(checks) if isinstance(checks, dict) else []}

    def _process_fixture(self, fixture: Fixture) -> Dict[str, Any]:
        documents = self.client.get(f"/api/v1/document-types/{fixture.doc_type}/documents")
        if not isinstance(documents, list):
            raise SmokeError("document list response was not a JSON list")
        existing, basis, candidate_count = self._select_existing(fixture, documents)
        task_info: Optional[Dict[str, Any]] = None
        if existing is None:
            upload = self.client.upload_file(
                f"/api/v1/document-types/{fixture.doc_type}/documents/upload",
                fixture.path,
            )
            if not isinstance(upload, list) or len(upload) != 1 or not isinstance(upload[0], dict):
                raise SmokeError(f"upload returned an unexpected payload: {upload}")
            document = upload[0]
            doc_id = str(document.get("doc_id") or "")
            if not doc_id:
                raise SmokeError("upload response did not contain doc_id")
            # Record the deletion allow-list immediately after the server has
            # acknowledged creation, even if a later response-contract check
            # fails.  Cleanup can therefore never leak a just-created record
            # merely because its response was malformed.
            self.created.append((fixture.doc_type, doc_id))
            if str(document.get("file_name") or "") != fixture.file_name:
                raise SmokeError("server did not preserve the synthetic fixture filename")
            origin = "created"
            task_info = self._start_and_wait(fixture.doc_type, doc_id)
        else:
            document = existing
            doc_id = str(document.get("doc_id") or "")
            if not doc_id:
                raise SmokeError("existing document did not contain doc_id")
            origin = "reused"
            status = str(document.get("status") or "")
            if status in READY_DOCUMENT_STATUSES:
                pass
            elif basis == "sha256" and status in {"uploaded", "failed"}:
                task_info = self._start_and_wait(fixture.doc_type, doc_id)
            elif status in ACTIVE_DOCUMENT_STATUSES:
                raise SmokeError(
                    f"matching existing document {doc_id} is active ({status}); no action taken"
                )
            else:
                hash_note = (
                    "server SHA-256 matched"
                    if basis == "sha256"
                    else "server API does not expose the source SHA-256"
                )
                raise SmokeError(
                    f"matching existing document {doc_id} is {status or 'unknown'}; "
                    f"{hash_note}, so it was not modified"
                )

        parse_result = self.client.get(
            f"/api/v1/document-types/{fixture.doc_type}/documents/{doc_id}/parse-result"
        )
        issues, stats = self._validate_parse_result(
            fixture.doc_type,
            parse_result,
        )
        graph_result = _as_dict((task_info or {}).get("graph_result"))
        issues.extend(
            self._validate_graph_result(
                graph_result,
                require_task=task_info is not None,
            )
        )

        diagram_revisions: List[Mapping[str, Any]] = []
        validation_counts: Dict[str, int] = {}
        projection_counts: Dict[str, int] = {}
        if fixture.doc_type in DIAGRAM_DOCUMENT_TYPES:
            payload = self.client.get(
                f"/api/v1/document-types/{fixture.doc_type}/documents/{doc_id}/diagrams"
            )
            if isinstance(payload, list):
                diagram_revisions = [item for item in payload if isinstance(item, Mapping)]
            else:
                issues.append("diagram revision response was not a JSON list")
            diagram_issues, validation_counts, projection_counts = (
                self._validate_diagram_revisions(
                    fixture.doc_type,
                    diagram_revisions,
                    newly_created=origin == "created",
                )
            )
            issues.extend(diagram_issues)

        parse_task_id = str((task_info or {}).get("parse_task_id") or "") or None
        graph_task_id = str((task_info or {}).get("graph_task_id") or "") or None
        item: Dict[str, Any] = {
            "doc_type": fixture.doc_type,
            "file_name": fixture.file_name,
            "sha256": fixture.sha256,
            "doc_id": doc_id,
            "origin": origin,
            "duplicate_basis": basis,
            "duplicate_candidates": candidate_count,
            "task_id": parse_task_id,
            "task_status": "completed" if parse_task_id else "not_started_reused_ready",
            "graph_task_id": graph_task_id,
            "graph_task_status": "completed" if graph_task_id else "not_started_reused_ready",
            "document_status": parse_result.get("status") if isinstance(parse_result, dict) else None,
            "parse_count": parse_result.get("parse_count") if isinstance(parse_result, dict) else None,
            "stats": {key: stats[key] for key in SUMMARY_STATS if key in stats},
            "projection_failures": _projection_failures(stats),
            "diagram_revisions": len(diagram_revisions),
            "diagram_validation_status_counts": validation_counts,
            "diagram_projection_status_counts": projection_counts,
            "ok": not issues,
            "issues": issues,
        }
        if issues:
            item["error"] = "; ".join(issues)
        return item

    @staticmethod
    def _select_existing(
        fixture: Fixture,
        documents: Iterable[Mapping[str, Any]],
    ) -> Tuple[Optional[Mapping[str, Any]], Optional[str], int]:
        docs = [item for item in documents if isinstance(item, Mapping)]
        hash_matches = [item for item in docs if _remote_sha256(item) == fixture.sha256]
        if hash_matches:
            return hash_matches[0], "sha256", len(hash_matches)

        name_matches = [item for item in docs if str(item.get("file_name") or "") == fixture.file_name]
        if not name_matches:
            return None, None, 0
        if len(name_matches) > 1:
            raise SmokeError(
                f"found {len(name_matches)} existing {fixture.doc_type} documents named "
                f"{fixture.file_name!r}; source SHA-256 is unavailable/does not match, so reuse is ambiguous"
            )
        remote_hash = _remote_sha256(name_matches[0])
        if remote_hash and remote_hash != fixture.sha256:
            raise SmokeError(
                f"existing {fixture.file_name!r} has a different SHA-256; no upload or mutation performed"
            )
        return name_matches[0], "file_name_server_sha_unavailable", 1

    def _start_and_wait(self, doc_type: str, doc_id: str) -> Dict[str, Any]:
        response = self.client.post_json(
            f"/api/v1/document-types/{doc_type}/documents/{doc_id}/parse",
            {},
        )
        if not isinstance(response, dict) or not response.get("task_id"):
            raise SmokeError(f"parse start returned an unexpected payload: {response}")
        parse_task_id = str(response["task_id"])
        parse_task = self._wait_for_task(parse_task_id, label="parse")
        parse_result = _as_dict(parse_task.get("result"))
        graph_task_id = str(parse_result.get("graph_task_id") or "").strip()
        if not graph_task_id:
            raise SmokeError(
                f"parse task {parse_task_id} completed without scheduling the required automatic graph task"
            )
        graph_task = self._wait_for_task(graph_task_id, label="automatic graph")
        return {
            "parse_task_id": parse_task_id,
            "graph_task_id": graph_task_id,
            "graph_result": _as_dict(graph_task.get("result")),
        }

    def _wait_for_task(self, task_id: str, *, label: str) -> Dict[str, Any]:
        deadline = self.monotonic() + self.task_timeout
        while True:
            task = self.client.get(f"/api/v1/knowledge-bases/tasks/{task_id}/status")
            if not isinstance(task, dict):
                raise SmokeError("task status response was not a JSON object")
            status = str(task.get("status") or "")
            if status in TERMINAL_TASK_STATUSES:
                if status != "completed":
                    error = str(task.get("error") or "")[:1000]
                    raise SmokeError(f"{label} task {task_id} ended as {status}: {error}")
                if int(task.get("progress") or 0) < 100:
                    raise SmokeError(f"{label} task {task_id} completed below 100% progress")
                # TaskManager records ``completed`` immediately before attaching
                # the result payload.  Poll through that short state transition
                # so graph_task_id and read-back results cannot be lost to a race.
                if isinstance(task.get("result"), dict):
                    return task
            if self.monotonic() >= deadline:
                raise SmokeError(
                    f"{label} task {task_id} did not complete with a result within "
                    f"{self.task_timeout:g}s"
                )
            self.sleep(self.poll_interval)

    @staticmethod
    def _validate_parse_result(
        doc_type: str,
        payload: Any,
    ) -> Tuple[List[str], Dict[str, Any]]:
        issues: List[str] = []
        if not isinstance(payload, dict):
            return ["parse-result was not a JSON object"], {}
        status = str(payload.get("status") or "")
        if status != "kg_done":
            issues.append(
                f"document status is {status or 'missing'}, expected kg_done after automatic graph projection"
            )
        try:
            parse_count = int(payload.get("parse_count") or 0)
        except (TypeError, ValueError):
            parse_count = 0
        if parse_count < 1:
            issues.append("parse_count is below 1")
        stats = _as_dict(payload.get("last_result"))
        required_key = REQUIRED_POSITIVE_STAT[doc_type]
        required_value = stats.get(required_key)
        if isinstance(required_value, bool) or not isinstance(required_value, (int, float)) or required_value <= 0:
            issues.append(f"parse statistic {required_key} must be positive")
        failures = _projection_failures(stats)
        if failures:
            issues.append(f"OpenSearch projection reported {failures} failure(s)")
        if stats.get("projection_rebuild_required"):
            issues.append("OpenSearch projection is marked for rebuild")
        return issues, stats

    @staticmethod
    def _validate_graph_result(
        payload: Mapping[str, Any],
        *,
        require_task: bool,
    ) -> List[str]:
        if not require_task:
            return []
        issues: List[str] = []
        if not payload:
            return ["automatic graph task did not return a result"]
        if str(payload.get("status") or "") != "kg_done":
            issues.append("automatic graph task did not finish with document status kg_done")
        structured = payload.get("structured_projection")
        if not isinstance(structured, Mapping):
            issues.append("automatic graph task did not report the structured projection")
        elif str(structured.get("status") or "") != "published":
            issues.append("structured Document/explicit-relation projection was not published")
        return issues

    @staticmethod
    def _validate_diagram_revisions(
        doc_type: str,
        revisions: Sequence[Mapping[str, Any]],
        *,
        newly_created: bool,
    ) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
        issues: List[str] = []
        validation_counts: Dict[str, int] = {}
        projection_counts: Dict[str, int] = {}
        if doc_type in {"fta", "rbd"} and not revisions:
            issues.append(f"{doc_type.upper()} parse reported diagrams but returned no latest revisions")
            return issues, validation_counts, projection_counts

        allowed = set(AUTOMATIC_DIAGRAM_STATUSES)
        if not newly_created:
            # Historical manually confirmed revisions remain supported, but a
            # fresh synthetic parse must prove the automatic three-way policy.
            allowed.add("manual_confirmed")

        for revision in revisions:
            revision_id = str(revision.get("revision_id") or "unknown")
            validation_status = str(revision.get("validation_status") or "")
            projection_status = str(revision.get("projection_status") or "")
            is_active = bool(revision.get("is_active"))
            validation_counts[validation_status or "missing"] = (
                validation_counts.get(validation_status or "missing", 0) + 1
            )
            projection_counts[projection_status or "missing"] = (
                projection_counts.get(projection_status or "missing", 0) + 1
            )

            if validation_status not in allowed:
                issues.append(
                    f"diagram revision {revision_id} has non-terminal validation status "
                    f"{validation_status or 'missing'}"
                )
                continue
            if validation_status in PUBLISHABLE_DIAGRAM_STATUSES:
                if projection_status != "published" or not is_active:
                    issues.append(
                        f"publishable diagram revision {revision_id} was not published and active"
                    )
            elif projection_status == "published" or is_active:
                issues.append(
                    f"quarantined diagram revision {revision_id} leaked into the active projection"
                )
        return issues, validation_counts, projection_counts

    def _cleanup_created(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        # Reverse order mirrors resource creation and never derives targets from
        # a list endpoint, filename, wildcard or server response other than the
        # exact upload-time doc_id kept in memory.
        for doc_type, doc_id in reversed(self.created):
            try:
                response = self.client.delete(
                    f"/api/v1/document-types/{doc_type}/documents/{doc_id}"
                )
                status = response.get("status") if isinstance(response, dict) else None
                if status not in {"success", "success_with_warnings"}:
                    raise SmokeError(f"unexpected cleanup response: {response}")
                results.append(
                    {
                        "doc_type": doc_type,
                        "doc_id": doc_id,
                        "ok": True,
                        "status": status,
                        "warnings": response.get("cleanup_errors", []) if isinstance(response, dict) else [],
                    }
                )
            except Exception as exc:
                results.append(
                    {"doc_type": doc_type, "doc_id": doc_id, "ok": False, "error": str(exc)}
                )
        return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "安全执行七类合成测试文档上传、解析和自动关系投影端到端 smoke；"
            "不会调用人工确认、拒绝或手工发布接口。"
        ),
        epilog=(
            "默认复用唯一的同类型同名已解析记录，不删除任何数据。"
            "只有显式传入 --cleanup-created 才会删除本进程本次上传返回的 doc_id；"
            "合成测试结果不代表真实工程文档或适航维修验证。"
        ),
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:19999",
        help="ICP-RAG 服务源地址（默认: %(default)s）",
    )
    parser.add_argument(
        "--test-docs",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "test-docs",
        help="包含七个固定合成 DOCX 的目录（默认: 仓库 test-docs）",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="单次 HTTP 请求超时秒数（默认: %(default)s）",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=900.0,
        help="每个解析任务最长等待秒数（默认: %(default)s）",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="任务状态轮询间隔秒数（默认: %(default)s）",
    )
    parser.add_argument(
        "--cleanup-created",
        action="store_true",
        help="显式删除且仅删除本进程本次成功上传的记录；不会删除复用记录",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="缩进 JSON；默认输出单行精简 JSON",
    )
    return parser


def _positive_float(parser: argparse.ArgumentParser, name: str, value: float) -> float:
    if value <= 0:
        parser.error(f"{name} must be greater than zero")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _positive_float(parser, "--request-timeout", args.request_timeout)
    _positive_float(parser, "--task-timeout", args.task_timeout)
    _positive_float(parser, "--poll-interval", args.poll_interval)

    try:
        fixtures = load_fixtures(args.test_docs)
        client = JsonApiClient(args.base_url, timeout=args.request_timeout)
        runner = SmokeRunner(
            client,
            fixtures,
            poll_interval=args.poll_interval,
            task_timeout=args.task_timeout,
            cleanup_created=args.cleanup_created,
        )
        summary = runner.run(base_url=client.base_url, test_docs=str(args.test_docs.resolve()))
    except Exception as exc:
        summary = {
            "kind": "synthetic_icp_ingestion_smoke",
            "ok": False,
            "base_url": args.base_url,
            "test_docs": str(args.test_docs),
            "checks": {},
            "documents": [],
            "safety": {
                "synthetic_fixtures_only": True,
                "manual_graph_review_actions": 0,
                "explicit_graph_generation_actions": 0,
                "automatic_graph_tasks_observed": 0,
                "cleanup_created_requested": args.cleanup_created,
                "created_doc_ids": [],
                "cleanup": [],
            },
            "errors": [str(exc)],
        }

    if args.pretty:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
