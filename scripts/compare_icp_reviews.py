"""Frozen offline old/new review comparison. No retrieval, sessions or retries."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
import tarfile
import types

import requests

from backend.agent import answer_review
from backend.agent.answer_observation import fingerprint, observe_answers, public_model
from backend.agent.document_semantics import document_semantics_prompt
from backend.agent.model_runtime_policy import bounded_model_transport, turn_budget
from backend.core.llm_service import llm_service
from scripts import legacy_answer_review
from scripts.diagnose_icp_architecture import write_private, read_json


def load_inputs(path, *, include_saved=True):
    payload = read_json(Path(path))
    samples = copy.deepcopy(payload["samples"])
    for source in payload.get("saved_snapshots", []) if include_saved else []:
        saved = read_json(Path(source["path"]))
        if fingerprint({k: v for k, v in saved.items() if k != "sha256"}) != saved["sha256"]:
            raise ValueError("Saved snapshot changed")
        prepared = saved["prepared"]
        messages, observation_groups = [], {}
        for sid, value in prepared["sources"].items():
            if value["kind"] == "conversation":
                messages.append({"message_id": sid.removeprefix("history:"), "role": value["role"],
                                 "content": value["quote"], "truncated": value.get("truncated", False),
                                 "prior_delivery": value.get("prior_delivery", [])})
            elif value["kind"] == "observation":
                _, identity, path = sid.split(":", 2)
                group = observation_groups.setdefault(identity, {
                    "observation_id": identity, "category": value["category"], "fields": {}})
                if group["category"] != value["category"] or path in group["fields"]:
                    raise ValueError("Observation identity collision")
                group["fields"][path] = value["quote"]
        observations = list(observation_groups.values())
        reconstructed = legacy_answer_review.prepare_review(prepared["draft"], ledger={},
            conversation={"messages": messages}, observations=observations,
            task={}, document_semantics="")["sources"]
        original_observations = {sid: value for sid, value in prepared["sources"].items() if value["kind"] == "observation"}
        if original_observations != {sid: value for sid, value in reconstructed.items() if value["kind"] == "observation"}:
            raise ValueError("Observation round-trip changed source coordinates or values")
        samples.append({
            "id": Path(source["path"]).stem, "expected_release": source["expected_release"], "basis": source["basis"],
            "question": prepared["goals"]["q0"], "history": {"messages": messages}, "observations": observations,
            "ledger": saved["ledger"], "calculation_requests": saved["calculation_requests"],
            "safety": saved["safety"], "draft": prepared["draft"], "snapshot_sha256": saved["sha256"],
        })
    for sample in samples:
        sample["ledger"] = {int(k): v for k, v in sample["ledger"].items()}
    if len({s["id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate samples")
    return payload, samples


def archived_review(path):
    """Load an explicitly supplied local baseline, never model-provided code."""
    with tarfile.open(path, 'r:gz') as archive:
        source = archive.extractfile('backend/agent/answer_review.py').read().decode()
    module = types.ModuleType('scripts._offline_reference_review')
    exec(compile(source, str(path) + ':answer_review.py', 'exec'), module.__dict__)
    return module, fingerprint(source)


def review_once(sample, protocol, model, *, condition=None, reasoning=None):
    if reasoning is None:
        # Current replay follows serving policy unless the experiment explicitly
        # overrides it. Archived baselines retain their historical non-thinking policy.
        from backend.agent.nodes.answer import _prepare_answer_review_model
        reasoning = (_prepare_answer_review_model(model).get("reasoning", False)
                     if protocol is answer_review else False)
    ledger = sample["ledger"]
    options = {"compact_sources": True} if protocol is legacy_answer_review else {}
    prepared = protocol.prepare_review(sample["draft"], ledger=ledger, conversation=sample["history"],
        observations=sample["observations"], task={"original_question": sample["question"]},
        document_semantics=document_semantics_prompt(), **options)
    payload = {**protocol.review_input(prepared, ledger), "safety": sample["safety"],
               "calculation_requests": sample["calculation_requests"]}
    schema = (protocol.review_schema(prepared["source_aliases"], unit_ids=[u["unit_id"] for u in prepared["units"]],
              goal_ids=list(prepared["goals"])) if protocol is legacy_answer_review else protocol.review_schema())
    config = dict(model, temperature=0, stream=False, reasoning=reasoning, format=schema)
    prompt = protocol.REVIEW_INSTRUCTION + "\n" + json.dumps(payload, ensure_ascii=False, default=str)
    messages = (protocol.review_messages(payload) if hasattr(protocol, "review_messages")
                else [{"role": "user", "content": prompt}])
    started = time.monotonic()
    raw = ""
    status, result, error = "system_error", None, None
    with observe_answers() as captured:
        try:
            with turn_budget():
                raw = llm_service.call_model(messages, bounded_model_transport(config))
                result = protocol.validate_review(raw, prepared=prepared, ledger=ledger,
                    safety=sample["safety"], calculation_requests=sample["calculation_requests"])
            status = "released"
        except protocol.AnswerReviewError as exc:
            status = "rejected" if exc.code == "review_rejected" else "system_error"
            error = {"code": exc.code, "feedback": exc.feedback}
        except Exception as exc:
            error = {"code": type(exc).__name__}
    return {
        "condition": condition or ("old" if protocol is legacy_answer_review else "new"),
        "status": status, "raw": raw, "result": result, "error": error,
        "duration_seconds": time.monotonic() - started, "model_calls": captured.model_calls,
        "prompt_sha256": fingerprint(prompt), "prepared": prepared,
        "messages": messages, "messages_sha256": fingerprint(messages),
        "requested_model": public_model(config),
    }


def summary(results):
    out = {}
    for condition in ("old", "new"):
        rows = [row for row in results if row["condition"] == condition]
        out[condition] = {
            "attempted": len(rows),
            "positive": sum(row["expected_release"] for row in rows),
            "negative": sum(not row["expected_release"] for row in rows),
            "correct_release": sum(row["expected_release"] and row["status"] == "released" for row in rows),
            "false_release": sum(not row["expected_release"] and row["status"] == "released" for row in rows),
            "false_rejection": sum(row["expected_release"] and row["status"] == "rejected" for row in rows),
            "correct_rejection": sum(not row["expected_release"] and row["status"] == "rejected" for row in rows),
            "system_error": sum(row["status"] == "system_error" for row in rows),
            "seconds": round(sum(row["duration_seconds"] for row in rows), 3),
            "output_chars": sum(len(row["raw"]) for row in rows),
            "token_usage": [call.get("token_usage") for row in rows for call in row["model_calls"] if call.get("token_usage")],
        }
    return out


def stability(results):
    """Describe repeats without treating majority voting as a release policy."""
    groups = {}
    for row in results:
        key = (row['condition'], row['sample_id'])
        groups.setdefault(key, []).append(row['status'])
    return [{'condition': condition, 'sample_id': sample, 'statuses': statuses,
             'changed': len(set(statuses)) > 1}
            for (condition, sample), statuses in groups.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="evals/icp_gate_review_inputs.json")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:19999")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--reference-archive", help="Optional local pre-change v5 source archive; defaults to legacy v4")
    parser.add_argument("--compare-reasoning", action="store_true", help="Same current reviewer, only reasoning false/true differs")
    parser.add_argument("--additional-inputs", action="append", default=[])
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    if args.repetitions < 1 or (args.compare_reasoning and args.reference_archive):
        parser.error("Positive repetitions required; reasoning comparison cannot also change the reviewer")
    reference, reference_hash = (archived_review(args.reference_archive) if args.reference_archive
                                 else (legacy_answer_review, None))
    if args.compare_reasoning:
        reference = answer_review
    payload, samples = load_inputs(args.inputs)
    for path in args.additional_inputs:
        samples.extend(load_inputs(path)[1])
    if len({sample['id'] for sample in samples}) != len(samples):
        raise ValueError("Duplicate samples across input files")
    if args.validate_only:
        print(json.dumps({"samples": len(samples), "positive": sum(s["expected_release"] for s in samples)}))
        return 0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    baseline = read_json(Path(args.baseline))["services"]["application"]["models"]["configuration"]["chat_model"]
    response = requests.get(args.base_url + "/api/v1/models/current-config", timeout=30)
    response.raise_for_status()
    model = response.json()["chat_model"]
    if args.compare_reasoning and model.get("provider") != "ollama":
        raise ValueError("This controlled reasoning comparison requires the existing Ollama model")
    for key in ("model_id", "provider", "model_name", "base_url"):
        if model[key] != baseline[key]:
            raise ValueError("Model changed from baseline")
    capture = read_json(Path(payload["generation_captures"][0]))
    frozen = next(i for i in capture["observation"]["inputs"] if i["stage"] == "generation")["model"]
    if frozen["model_id"] != model["model_id"]:
        raise ValueError("Captured generation uses another model")
    model.update({key: value for key, value in frozen.items() if key not in ("model_id", "model_name", "provider")})
    code_paths = [Path(answer_review.__file__), Path(legacy_answer_review.__file__),
                  Path("backend/agent/citation_binding.py"), Path("backend/agent/nodes/answer.py"),
                  Path(__file__), Path("backend/core/llm_service.py"), Path("backend/agent/model_runtime_policy.py")]
    hashes = {str(path): fingerprint(path.read_text()) for path in code_paths}
    write_private(output / "inputs.json", {"samples": samples, "model": public_model(model), "source_hashes": hashes,
                                          "reference_archive": args.reference_archive, "reference_sha256": reference_hash,
                                          "compare_reasoning": args.compare_reasoning, "repetitions": args.repetitions,
                                          "input_paths": [args.inputs, *args.additional_inputs]})
    results = []
    schedule = [(repeat, index, sample) for repeat in range(args.repetitions) for index, sample in enumerate(samples)]
    for repeat, index, sample in schedule:
        conditions = [("old", reference, False),
                      ("new", answer_review, True if args.compare_reasoning else None)]
        if (index + repeat) % 2:
            conditions.reverse()
        for condition, protocol, reasoning in conditions:
            if hashes != {str(path): fingerprint(path.read_text()) for path in code_paths}:
                raise ValueError("Implementation changed during fixed-input comparison")
            result = {"sample_id": sample["id"], "input_sha256": fingerprint(sample),
                      "expected_release": sample["expected_release"], "repeat": repeat + 1,
                      **review_once(sample, protocol, model, condition=condition, reasoning=reasoning)}
            results.append(result)
            write_private(output / f"review-{len(results):03d}.json", result)
            print(f"[{len(results)}/{len(schedule)*2}] repeat={repeat+1} {result['condition']} {sample['id']} {result['status']}", flush=True)
    report = {"schema_version": "icp-review-comparison/v1", "summary": summary(results),
              "stability": stability(results), "results": results}
    write_private(output / "report.json", report)
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    new = report["summary"]["new"]
    return int(bool(new["false_release"] or new["false_rejection"] or new["system_error"]))


if __name__ == "__main__":
    raise SystemExit(main())
