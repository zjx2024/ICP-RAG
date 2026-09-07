#!/usr/bin/env python3
"""Capture a reproducible, read-only ICP evaluation baseline manifest.

The collector fingerprints source inputs and, unless ``--offline`` is used,
observes the local HTTP services used by an evaluation run.  It never talks
to PostgreSQL directly and only issues GET requests.  Probe failures are data
in the manifest rather than fatal errors, so an offline or partially started
stack does not hide the local source fingerprints.

The output deliberately excludes credentials.  Sensitive response keys are
recursively redacted, URL credentials/query strings are removed, Dify's
server-info response body is never retained, and no process environment is
copied into the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableSequence, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener


SCHEMA_VERSION = "icp-eval-baseline/v1"
DEFAULT_APP_URL = "http://127.0.0.1:19999"
DEFAULT_OPENSEARCH_URL = "http://127.0.0.1:9201"
DEFAULT_HUGEGRAPH_URL = "http://127.0.0.1:8082"
DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_HTTP_BYTES = 8 * 1024 * 1024
REDACTED = "***REDACTED***"

EXPECTED_DOCUMENTS: Tuple[Tuple[str, str], ...] = (
    ("function_locator", "01_系统架构与功能说明.docx"),
    ("hmc", "02_HMC故障代码手册.docx"),
    ("fmeca", "03_FMECA分析报告.docx"),
    ("fta", "04_FTA故障树分析报告.docx"),
    ("rbd", "05_RBD可靠性框图分析报告.docx"),
    ("mfl", "06_MFL维修故障清单.docx"),
    ("fault_history", "07_故障历史查询文档.docx"),
)

EVALUATION_FILES: Tuple[Tuple[str, str], ...] = (
    ("canary_19", "evals/icp_multiturn_dialogue_cases.json"),
    ("regression_positioning", "evals/icp_multiturn_regression_positioning.json"),
    ("regression_analysis", "evals/icp_multiturn_regression_analysis.json"),
    ("regression_boundary", "evals/icp_multiturn_regression_boundary.json"),
    ("expression_holdout", "evals/icp_expression_holdout.json"),
)

PARSER_RELIABILITY_FILES: Tuple[str, ...] = (
    "backend/core/parsers/registry.py",
    "backend/core/parsers/excel_row.py",
    "backend/core/parsers/diagram.py",
    "backend/core/parsers/diagram_extraction.py",
    "backend/core/parsers/fault_history.py",
    "backend/core/parsers/function_architecture.py",
    "backend/core/parsers/function_architecture_indexed.py",
    "backend/core/parsers/rbd_expression.py",
    "backend/core/parsers/table_engine.py",
    "backend/icp/document_types.py",
    "backend/icp/image_analysis.py",
    "backend/icp/reliability.py",
)

AGENT_FILES: Tuple[str, ...] = (
    "backend/agent/graph.py",
    "backend/agent/icp_defaults.py",
    "backend/agent/contracts.py",
    "backend/agent/answer_contract.py",
    "backend/agent/history_focus.py",
    "backend/agent/query_understanding.py",
    "backend/agent/rbd_contract.py",
    "backend/agent/current_model_selection.py",
    "backend/agent/model_selection_service.py",
    "backend/agent/citation_binding.py",
    "backend/agent/answer_context_binding.py",
    "backend/agent/answer_review.py",
    "backend/agent/tracing.py",
    "backend/core/llm_service.py",
    "backend/agent/clarification_contract.py",
    "backend/agent/state.py",
    "backend/agent/tool_policy.py",
    "backend/agent/api/chat_router.py",
    "backend/agent/nodes/scope_router.py",
    "backend/agent/nodes/plan.py",
    "backend/agent/nodes/adaptive_tools.py",
    "backend/agent/nodes/evidence_gate.py",
    "backend/agent/nodes/answer.py",
    "backend/agent/nodes/answer_backstop.py",
    "backend/agent/nodes/answer_presence.py",
    "backend/agent/nodes/answer_claim_units.py",
    "backend/agent/nodes/answer_collections.py",
    "backend/agent/nodes/answer_domain_scope.py",
    "backend/agent/nodes/answer_evidence_context.py",
    "backend/agent/nodes/answer_evidence_window.py",
    "backend/agent/nodes/answer_fault_rules.py",
    "backend/agent/nodes/answer_field_catalog.py",
    "backend/agent/nodes/answer_history_rules.py",
    "backend/agent/nodes/answer_payload.py",
    "backend/agent/nodes/answer_record_scope.py",
    "backend/agent/nodes/answer_request.py",
    "backend/agent/nodes/answer_source_fields.py",
    "backend/agent/hmc_identity.py",
    "backend/agent/nodes/answer_structure_views.py",
    "backend/agent/nodes/answer_target_presence.py",
    "backend/agent/nodes/answer_text.py",
    "backend/agent/tools/icp_diagnosis.py",
    "backend/agent/tools/icp_calculation.py",
)

EVALUATOR_FILES: Tuple[str, ...] = (
    "scripts/evaluate_icp_paraphrases.py",
    "scripts/capture_icp_eval_baseline.py",
    "scripts/eval_source_refs.py",
    "scripts/audit_icp_eval_suite.py",
    "scripts/calibrate_icp_judge.py",
    "scripts/run_icp_eval_campaign.py",
    "scripts/diagnose_icp_architecture.py",
    "evals/icp_eval_gold_sources.json",
    "evals/icp_judge_calibration.json",
)

PARSE_STAT_FIELDS = frozenset(
    {
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
        "projection_failed",
    }
)

_SENSITIVE_EXACT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "password",
        "passwd",
        "token",
        "access_token",
        "refresh_token",
        "bearer_token",
        "id_token",
        "secret",
        "client_secret",
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
    }
)


class ProbeError(RuntimeError):
    """A read-only HTTP observation could not be completed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_timezone() -> str:
    now = datetime.now().astimezone()
    return str(now.tzinfo or "")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _aggregate_file_entries(entries: Iterable[Mapping[str, Any]]) -> str:
    canonical = [
        {
            "exists": bool(item.get("exists")),
            "path": str(item.get("path") or ""),
            "sha256": item.get("sha256"),
            "size_bytes": item.get("size_bytes"),
        }
        for item in sorted(entries, key=lambda value: str(value.get("path") or ""))
    ]
    return _sha256_bytes(_canonical_json_bytes(canonical))


def _error(
    errors: MutableSequence[Dict[str, Any]],
    component: str,
    operation: str,
    exc: BaseException | str,
) -> None:
    message = str(exc).strip() or type(exc).__name__
    errors.append(
        {
            "component": component,
            "operation": operation,
            "kind": type(exc).__name__ if isinstance(exc, BaseException) else "error",
            "message": message[:500],
        }
    )


def _file_entry(
    root: Path,
    relative_path: str,
    errors: MutableSequence[Dict[str, Any]],
    *,
    component: str,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"path": relative_path, "exists": False}
    path = root / relative_path
    try:
        stat = path.stat()
        if not path.is_file():
            raise OSError("path is not a regular file")
        entry.update(
            {
                "exists": True,
                "size_bytes": stat.st_size,
                "sha256": _sha256_file(path),
            }
        )
    except (OSError, ValueError) as exc:
        # Keep host-specific absolute paths out of a portable manifest.
        safe_message = "missing or inaccessible regular file"
        entry["error"] = safe_message
        _error(errors, component, relative_path, safe_message)
    return entry


def _evaluation_shape(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = payload if isinstance(payload, Mapping) else {}
    raw_cases = payload if isinstance(payload, list) else root.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("evaluation JSON must be an array or contain a cases array")

    turn_count = 0
    for case in raw_cases:
        if not isinstance(case, Mapping):
            continue
        turns = case.get("turns")
        if isinstance(turns, list):
            turn_count += len(turns)
            continue
        questions = case.get("questions")
        if isinstance(questions, list):
            turn_count += len(questions)
            continue
        if str(case.get("question") or case.get("query") or "").strip():
            turn_count += 1 + len(
                case.get("paraphrases") if isinstance(case.get("paraphrases"), list) else []
            )

    return {
        "name": str(root.get("name") or "") or None,
        "version": str(root.get("version") or "") or None,
        "cases": len(raw_cases),
        "turns": turn_count,
    }


def _collect_fixtures(root: Path, errors: MutableSequence[Dict[str, Any]]) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    for doc_type, filename in EXPECTED_DOCUMENTS:
        entry = _file_entry(
            root,
            f"test-docs/{filename}",
            errors,
            component="fixtures",
        )
        entry["doc_type"] = doc_type
        files.append(entry)
    return {
        "kind": "synthetic_test_documents",
        "synthetic_only": True,
        "hash_algorithm": "sha256",
        "aggregate_algorithm": "sha256-canonical-file-list-v1",
        "aggregate_sha256": _aggregate_file_entries(files),
        "files": files,
    }


def _collect_evaluations(root: Path, errors: MutableSequence[Dict[str, Any]]) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    for suite_id, relative_path in EVALUATION_FILES:
        entry = _file_entry(root, relative_path, errors, component="evaluation_suites")
        entry["suite_id"] = suite_id
        if entry.get("exists"):
            try:
                entry.update(_evaluation_shape(root / relative_path))
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                entry["shape_error"] = str(exc)[:500]
                _error(errors, "evaluation_suites", relative_path, exc)
        files.append(entry)
    regression_files = [item for item in files if item["suite_id"].startswith("regression_")]
    return {
        "hash_algorithm": "sha256",
        "aggregate_algorithm": "sha256-canonical-file-list-v1",
        "aggregate_sha256": _aggregate_file_entries(files),
        "regression_100_aggregate_sha256": _aggregate_file_entries(regression_files),
        "files": files,
    }


def _collect_code_group(
    root: Path,
    paths: Sequence[str],
    errors: MutableSequence[Dict[str, Any]],
    component: str,
) -> Dict[str, Any]:
    files = [
        _file_entry(root, relative_path, errors, component=component)
        for relative_path in paths
    ]
    return {
        "hash_algorithm": "sha256",
        "aggregate_algorithm": "sha256-canonical-file-list-v1",
        "aggregate_sha256": _aggregate_file_entries(files),
        "files": files,
    }


def _run_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    binary: bool = False,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        list(argv),
        cwd=str(cwd),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )


def _git_text(
    root: Path,
    argv: Sequence[str],
    errors: MutableSequence[Dict[str, Any]],
    operation: str,
    *,
    required: bool = True,
) -> Optional[str]:
    try:
        result = _run_command(("git", *argv), cwd=root)
    except OSError as exc:
        if required:
            _error(errors, "git", operation, exc)
        return None
    if result.returncode != 0:
        if required:
            detail = str(result.stderr or "git command failed").strip()
            _error(errors, "git", operation, detail)
        return None
    # Porcelain status uses its first two columns as data.  ``strip()`` would
    # turn an unstaged `` M path`` entry into ``M path`` and falsely report it
    # as staged when that entry happens to be the first output line.
    return str(result.stdout).rstrip("\r\n")


def _collect_git(root: Path, errors: MutableSequence[Dict[str, Any]]) -> Dict[str, Any]:
    head = _git_text(root, ("rev-parse", "HEAD"), errors, "head")
    if not head:
        return {"available": False}
    branch = _git_text(root, ("branch", "--show-current"), errors, "branch")
    tree = _git_text(root, ("rev-parse", "HEAD^{tree}"), errors, "tree")
    upstream = _git_text(
        root,
        ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"),
        errors,
        "upstream",
        required=False,
    )
    ahead: Optional[int] = None
    behind: Optional[int] = None
    if upstream:
        counts = _git_text(
            root,
            ("rev-list", "--left-right", "--count", f"HEAD...{upstream}"),
            errors,
            "ahead_behind",
            required=False,
        )
        if counts:
            try:
                ahead_text, behind_text = counts.split()
                ahead, behind = int(ahead_text), int(behind_text)
            except (TypeError, ValueError) as exc:
                _error(errors, "git", "ahead_behind", exc)

    status_text = _git_text(
        root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        errors,
        "status",
    )
    status_entries = status_text.splitlines() if status_text else []
    tracked_entries = [item for item in status_entries if not item.startswith(("??", "!!"))]
    untracked_entries = [item for item in status_entries if item.startswith("??")]
    staged_count = sum(bool(item and item[0] not in {" ", "?", "!"}) for item in tracked_entries)
    worktree_count = sum(len(item) > 1 and item[1] not in {" ", "?", "!"} for item in tracked_entries)

    diff_sha256: Optional[str] = None
    try:
        diff = _run_command(
            ("git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"),
            cwd=root,
            binary=True,
        )
        if diff.returncode == 0:
            diff_sha256 = _sha256_bytes(bytes(diff.stdout))
        else:
            _error(errors, "git", "tracked_diff", "git diff failed")
    except OSError as exc:
        _error(errors, "git", "tracked_diff", exc)

    return {
        "available": True,
        "branch": branch,
        "head": head,
        "tree": tree,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "dirty": bool(status_entries),
        "status": {
            "entries": status_entries,
            "tracked_changed": len(tracked_entries),
            "untracked": len(untracked_entries),
            "staged": staged_count,
            "worktree_changed": worktree_count,
            "sha256": _sha256_bytes((status_text or "").encode("utf-8")),
        },
        "tracked_diff": {
            "command": "git diff --binary --no-ext-diff HEAD --",
            "sha256": diff_sha256,
        },
    }


def _tool_version(command: str, root: Path) -> Dict[str, Any]:
    try:
        result = _run_command((command, "--version"), cwd=root)
    except OSError as exc:
        return {"available": False, "error": type(exc).__name__}
    output = str(result.stdout or result.stderr or "").strip().splitlines()
    return {
        "available": result.returncode == 0,
        "version": output[0][:200] if output else None,
        **({"exit_code": result.returncode} if result.returncode else {}),
    }


def _collect_environment(root: Path) -> Dict[str, Any]:
    return {
        "captured_with": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
        },
        "node": _tool_version("node", root),
        "npm": _tool_version("npm", root),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "timezone": _local_timezone(),
    }


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _is_sensitive_key(value: Any) -> bool:
    key = _normalized_key(value)
    if key in _SENSITIVE_EXACT_KEYS:
        return True
    return (
        "password" in key
        or "passwd" in key
        or "api_key" in key
        or "apikey" in key
        or "secret" in key
        or (key.endswith("token") and not key.endswith("tokens"))
    )


def _redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return value
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = host
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except ValueError:
        return "invalid-http-url"


def _scrub_sensitive(value: Any, *, parent_key: str = "") -> Any:
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if _is_sensitive_key(key):
                result[key] = None if item is None else REDACTED
            else:
                result[key] = _scrub_sensitive(item, parent_key=key)
        return result
    if isinstance(value, (list, tuple)):
        return [_scrub_sensitive(item, parent_key=parent_key) for item in value]
    if isinstance(value, str) and "url" in _normalized_key(parent_key):
        return _redact_url(value)
    return value


def _normalize_base_url(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must be an absolute http(s) origin")
    if parsed.username or parsed.password:
        raise ValueError("URL must not contain credentials")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("URL must contain only scheme, host and optional port")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("URL contains an invalid port") from exc
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


class JsonGetClient:
    """Small no-proxy, GET-only JSON client with bounded response bodies."""

    def __init__(self, base_url: str, timeout_seconds: float):
        self.base_url = _normalize_base_url(base_url)
        self.timeout_seconds = timeout_seconds
        self._opener = build_opener(ProxyHandler({}))

    def get(self, path: str) -> Tuple[int, Any]:
        if not path.startswith("/"):
            raise ValueError("probe path must start with /")
        request = Request(
            self.base_url + path,
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            response = self._opener.open(request, timeout=self.timeout_seconds)
        except HTTPError as exc:
            # Retain status, not an error body which may echo deployment data.
            return int(exc.code), None
        except (URLError, TimeoutError, OSError) as exc:
            raise ProbeError(str(exc)) from exc
        with response:
            body = response.read(MAX_HTTP_BYTES + 1)
            status = int(getattr(response, "status", response.getcode()))
        if len(body) > MAX_HTTP_BYTES:
            raise ProbeError(f"response exceeds {MAX_HTTP_BYTES} bytes")
        if not body:
            return status, None
        try:
            return status, json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ProbeError("response was not valid UTF-8 JSON") from exc


def _safe_get(
    client: JsonGetClient,
    path: str,
    errors: MutableSequence[Dict[str, Any]],
    component: str,
) -> Tuple[bool, Any, Optional[int]]:
    try:
        status, payload = client.get(path)
    except (ProbeError, ValueError) as exc:
        _error(errors, component, path, exc)
        return False, None, None
    ok = 200 <= status < 300
    if not ok:
        _error(errors, component, path, f"HTTP {status}")
    return ok, payload, status


def _readiness_summary(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    checks = payload.get("checks")
    return {
        "status": payload.get("status"),
        "checks": {
            str(name): {"ok": bool(value.get("ok"))}
            for name, value in sorted(
                checks.items() if isinstance(checks, Mapping) else [],
                key=lambda item: str(item[0]),
            )
            if isinstance(value, Mapping)
        },
    }


def _document_hash(document: Mapping[str, Any]) -> Optional[str]:
    metadata = document.get("metadata")
    candidates: List[Any] = [
        document.get("sha256"),
        document.get("file_sha256"),
        document.get("content_sha256"),
    ]
    if isinstance(metadata, Mapping):
        candidates.extend(
            [
                metadata.get("sha256"),
                metadata.get("file_sha256"),
                metadata.get("content_sha256"),
            ]
        )
    for value in candidates:
        text = str(value or "").strip().casefold()
        if len(text) == 64 and all(char in "0123456789abcdef" for char in text):
            return text
    return None


def _document_summary(document: Mapping[str, Any], doc_type: str) -> Dict[str, Any]:
    return {
        "doc_id": str(document.get("doc_id") or "") or None,
        "doc_type": str(document.get("doc_type") or doc_type),
        "file_name": str(document.get("file_name") or "") or None,
        "status": str(document.get("status") or "") or None,
        "parse_count": document.get("parse_count"),
        "source_sha256": _document_hash(document),
    }


def _parse_result_summary(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    last_result = payload.get("last_result")
    stats = {
        str(key): _scrub_sensitive(value, parent_key=str(key))
        for key, value in (last_result.items() if isinstance(last_result, Mapping) else [])
        if str(key) in PARSE_STAT_FIELDS
    }
    return {
        "status": str(payload.get("status") or "") or None,
        "parse_count": payload.get("parse_count"),
        "stats": stats,
    }


def _probe_application(
    base_url: str,
    timeout_seconds: float,
    errors: MutableSequence[Dict[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"configured_url": "invalid", "reachable": False}
    try:
        client = JsonGetClient(base_url, timeout_seconds)
    except ValueError as exc:
        _error(errors, "application", "configure", exc)
        return result
    result["configured_url"] = _normalize_base_url(base_url)

    ok, payload, status = _safe_get(client, "/readyz", errors, "application")
    result["readyz"] = {
        "ok": ok,
        "status_code": status,
        **_readiness_summary(payload),
    }
    result["reachable"] = status is not None

    ok, registry, status = _safe_get(
        client, "/api/v1/document-types", errors, "application"
    )
    registry_items = registry if isinstance(registry, list) else []
    actual_types = sorted(
        {
            str(item.get("doc_type") or "")
            for item in registry_items
            if isinstance(item, Mapping) and item.get("doc_type")
        }
    )
    expected_types = [doc_type for doc_type, _ in EXPECTED_DOCUMENTS]
    result["document_types"] = {
        "ok": ok and set(actual_types) == set(expected_types) and len(actual_types) == 7,
        "status_code": status,
        "actual": actual_types,
        "expected": expected_types,
    }

    documents_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for doc_type in expected_types:
        path = f"/api/v1/document-types/{quote(doc_type, safe='')}/documents"
        docs_ok, docs_payload, _ = _safe_get(client, path, errors, "application")
        summaries: List[Dict[str, Any]] = []
        if docs_ok and isinstance(docs_payload, list):
            for raw_document in docs_payload:
                if not isinstance(raw_document, Mapping):
                    continue
                summary = _document_summary(raw_document, doc_type)
                doc_id = summary.get("doc_id")
                if doc_id:
                    parse_path = (
                        f"/api/v1/document-types/{quote(doc_type, safe='')}/documents/"
                        f"{quote(str(doc_id), safe='')}/parse-result"
                    )
                    parse_ok, parse_payload, parse_status = _safe_get(
                        client, parse_path, errors, "application"
                    )
                    summary["parse_result"] = {
                        "ok": parse_ok,
                        "status_code": parse_status,
                        **_parse_result_summary(parse_payload),
                    }
                summaries.append(summary)
        summaries.sort(key=lambda item: (str(item.get("file_name") or ""), str(item.get("doc_id") or "")))
        documents_by_type[doc_type] = summaries
    result["documents"] = documents_by_type

    models_ok, models_payload, models_status = _safe_get(
        client, "/api/v1/models/current-config", errors, "application"
    )
    result["models"] = {
        "ok": models_ok,
        "status_code": models_status,
        "configuration": _scrub_sensitive(models_payload) if models_ok else None,
    }

    openapi_ok, openapi, _ = _safe_get(client, "/openapi.json", errors, "application")
    paths = openapi.get("paths") if isinstance(openapi, Mapping) else None
    retrieval_registered = bool(
        isinstance(paths, Mapping)
        and isinstance(paths.get("/api/v1/dify/retrieval"), Mapping)
        and "post" in paths["/api/v1/dify/retrieval"]
    )
    dify_ok, _dify_payload, dify_status = _safe_get(
        client, "/api/v1/dify/server-info", errors, "application"
    )
    result["dify"] = {
        "server_info_get": {"ok": dify_ok, "status_code": dify_status},
        "retrieval_post_registered": openapi_ok and retrieval_registered,
        "response_body_retained": False,
    }
    return result


def _probe_opensearch(
    base_url: str,
    timeout_seconds: float,
    errors: MutableSequence[Dict[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"configured_url": "invalid", "reachable": False}
    try:
        client = JsonGetClient(base_url, timeout_seconds)
    except ValueError as exc:
        _error(errors, "opensearch", "configure", exc)
        return result
    result["configured_url"] = _normalize_base_url(base_url)

    health_ok, health, health_status = _safe_get(
        client, "/_cluster/health", errors, "opensearch"
    )
    health_fields = {}
    if isinstance(health, Mapping):
        for key in (
            "cluster_name",
            "status",
            "number_of_nodes",
            "number_of_data_nodes",
            "active_primary_shards",
            "active_shards",
            "unassigned_shards",
        ):
            if key in health:
                health_fields[key] = health[key]
    result["health"] = {"ok": health_ok, "status_code": health_status, **health_fields}
    result["reachable"] = health_status is not None

    alias_ok, aliases_payload, alias_status = _safe_get(
        client, "/_alias", errors, "opensearch"
    )
    aliases: List[Dict[str, str]] = []
    if isinstance(aliases_payload, Mapping):
        for index_name, raw in aliases_payload.items():
            raw_aliases = raw.get("aliases") if isinstance(raw, Mapping) else None
            if isinstance(raw_aliases, Mapping):
                aliases.extend(
                    {"alias": str(alias), "index": str(index_name)}
                    for alias in raw_aliases
                )
    aliases.sort(key=lambda item: (item["alias"], item["index"]))
    result["aliases"] = {"ok": alias_ok, "status_code": alias_status, "items": aliases}

    indices_ok, indices_payload, indices_status = _safe_get(
        client,
        "/_cat/indices?format=json&h=index,uuid,docs.count&expand_wildcards=all",
        errors,
        "opensearch",
    )
    indices: List[Dict[str, Any]] = []
    if isinstance(indices_payload, list):
        for raw in indices_payload:
            if not isinstance(raw, Mapping):
                continue
            count = raw.get("docs.count")
            try:
                count = int(count) if count not in {None, ""} else None
            except (TypeError, ValueError):
                count = None
            indices.append(
                {
                    "index": str(raw.get("index") or ""),
                    "uuid": str(raw.get("uuid") or "") or None,
                    "document_count": count,
                }
            )
    indices.sort(key=lambda item: item["index"])
    result["indices"] = {
        "ok": indices_ok,
        "status_code": indices_status,
        "items": indices,
    }
    return result


def _version_fields(value: Any, prefix: str = "") -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if "version" in _normalized_key(key) and isinstance(item, (str, int, float, bool)):
                fields[path] = item
            fields.update(_version_fields(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            fields.update(_version_fields(item, f"{prefix}[{index}]"))
    return fields


def _probe_hugegraph(
    base_url: str,
    timeout_seconds: float,
    errors: MutableSequence[Dict[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"configured_url": "invalid", "reachable": False}
    try:
        client = JsonGetClient(base_url, timeout_seconds)
    except ValueError as exc:
        _error(errors, "hugegraph", "configure", exc)
        return result
    result["configured_url"] = _normalize_base_url(base_url)
    ok, payload, status = _safe_get(client, "/versions", errors, "hugegraph")
    result.update(
        {
            "reachable": status is not None,
            "version": {
                "ok": ok,
                "status_code": status,
                "fields": _version_fields(payload) if ok else {},
            },
        }
    )
    return result


def _collect_services(
    *,
    offline: bool,
    base_url: str,
    opensearch_url: str,
    hugegraph_url: str,
    timeout_seconds: float,
    errors: MutableSequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if offline:
        return {
            "mode": "offline",
            "http_methods_used": [],
            "postgres_direct_connection": False,
            "skipped": ["application", "opensearch", "hugegraph"],
        }
    return {
        "mode": "read_only_http",
        "http_methods_used": ["GET"],
        "postgres_direct_connection": False,
        "application": _probe_application(base_url, timeout_seconds, errors),
        "opensearch": _probe_opensearch(opensearch_url, timeout_seconds, errors),
        "hugegraph": _probe_hugegraph(hugegraph_url, timeout_seconds, errors),
    }


def capture_baseline(
    repo_root: Path,
    *,
    offline: bool = False,
    base_url: str = DEFAULT_APP_URL,
    opensearch_url: str = DEFAULT_OPENSEARCH_URL,
    hugegraph_url: str = DEFAULT_HUGEGRAPH_URL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Return a complete baseline manifest without mutating repository/services."""

    root = repo_root.expanduser().resolve()
    errors: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": _utc_now(),
        "collector": {
            "read_only": True,
            "stdlib_only": True,
            "secrets_recorded": False,
            "repo_root_recorded": False,
        },
        "source": _collect_git(root, errors),
        "environment": _collect_environment(root),
        "fixtures": _collect_fixtures(root, errors),
        "evaluation_suites": _collect_evaluations(root, errors),
        "code_inputs": {
            # Curated role groups explain changes; the complete source group
            # also catches repository/config/transport files outside those
            # groups, which can change behavior just as much as a prompt.
            "application_source": _collect_code_group(
                root,
                tuple(sorted(str(path.relative_to(root)) for path in (root / "backend").rglob("*.py"))),
                errors,
                "code.application_source",
            ),
            "parser_reliability": _collect_code_group(
                root,
                PARSER_RELIABILITY_FILES,
                errors,
                "code.parser_reliability",
            ),
            "agent_and_prompts": _collect_code_group(
                root,
                AGENT_FILES,
                errors,
                "code.agent_and_prompts",
            ),
            "evaluator": _collect_code_group(
                root,
                EVALUATOR_FILES,
                errors,
                "code.evaluator",
            ),
        },
    }
    manifest["services"] = _collect_services(
        offline=offline,
        base_url=base_url,
        opensearch_url=opensearch_url,
        hugegraph_url=hugegraph_url,
        timeout_seconds=timeout_seconds,
        errors=errors,
    )
    manifest["errors"] = errors
    manifest["status"] = "complete" if not errors else "complete_with_errors"
    return manifest


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读采集 ICP 评测所需的代码、数据和服务基线指纹。",
        epilog=(
            "采集器不直连 PostgreSQL、不发送写请求，也不记录凭据。"
            "外部服务不可达时仍输出本地指纹，并把失败写入 errors。"
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="仓库根目录（默认由脚本位置确定）",
    )
    parser.add_argument("--base-url", default=DEFAULT_APP_URL, help="ICP API origin")
    parser.add_argument(
        "--opensearch-url", default=DEFAULT_OPENSEARCH_URL, help="OpenSearch origin"
    )
    parser.add_argument(
        "--hugegraph-url", default=DEFAULT_HUGEGRAPH_URL, help="HugeGraph origin"
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="每个只读 HTTP 请求的超时秒数（默认 %(default)s）",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="仅采集本地 Git、运行时、文档、评测集和代码指纹",
    )
    parser.add_argument("-o", "--output", default="-", help="输出 JSON；- 表示 stdout")
    parser.add_argument("--compact", action="store_true", help="输出紧凑 JSON")
    return parser


def _write_manifest(manifest: Mapping[str, Any], output: str, *, compact: bool) -> None:
    if compact:
        text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
    text += "\n"
    if output == "-":
        sys.stdout.write(text)
        return

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = capture_baseline(
        args.repo_root,
        offline=bool(args.offline),
        base_url=str(args.base_url),
        opensearch_url=str(args.opensearch_url),
        hugegraph_url=str(args.hugegraph_url),
        timeout_seconds=float(args.timeout_seconds),
    )
    _write_manifest(manifest, str(args.output), compact=bool(args.compact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
