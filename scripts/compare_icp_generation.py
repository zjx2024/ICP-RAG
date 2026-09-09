"""Paired archived/current answer nodes on frozen generation inputs, offline."""
from __future__ import annotations

import argparse
import asyncio
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tarfile
import time
import types
from unittest.mock import AsyncMock, patch

import requests

from backend.agent.answer_observation import fingerprint, observe_answers, public_model
from backend.agent.model_runtime_policy import turn_budget
from backend.agent.nodes import answer
from backend.storage.postgres_client import postgres_client
from scripts.diagnose_icp_architecture import write_private, read_json, restore_generation
from scripts import evaluate_icp_paraphrases as evaluator


def publication_checks(result, assertions):
    """Reuse original citation assertions; offline runs cannot claim SSE/tool checks."""
    keys = {"min_citations", "required_used_doc_types", "any_used_doc_types",
            "forbidden_used_doc_types", "allowed_used_doc_types", "required_used_record_ids",
            "any_used_record_ids", "forbidden_used_record_ids", "allowed_used_record_ids"}
    selected = {key: value for key, value in assertions.items() if key in keys}
    citations = result.get("citations") or []
    doc_types, records = evaluator._citation_dimensions(citations)
    checks = evaluator._evaluate_assertions(
        selected, answer=result.get("final_answer", ""), tools=[], doc_types=[],
        used_doc_types=doc_types, used_record_ids=records, evidence_count=0,
        used_evidence_count=len(citations), citation_count=len(citations), event_types=[],
        duration_seconds=result.get("duration_seconds", 0), runtime_errors=[],
        done_received=False, answer_received=bool(result.get("final_answer")),
    )
    return [check for check in checks if check["name"] in selected]


def archived_node(path):
    # Only an explicitly supplied local source snapshot, never model-generated code.
    with tarfile.open(path, "r:gz") as archive:
        source = archive.extractfile("backend/agent/nodes/answer.py").read().decode()
    module = types.ModuleType("scripts._offline_archived_answer")
    source = source.replace("from backend.agent.answer_review import", "from scripts.legacy_answer_review import")
    exec(compile(source, str(path) + ":answer.py", "exec"), module.__dict__)
    return module


def same_facts(generation, module):
    state = copy.deepcopy(generation["state"])
    context = module._format_context(state)
    # Earlier captures predate the pre-batch projection (e.g. usage_scope and
    # HMC symptom/action IDs). Freeze ONE common pre-batch projection from the
    # saved raw state for BOTH conditions; never compare with old capture scores.
    if context != answer._format_context(state) or module._extract_citations(state) != answer._extract_citations(state):
        raise ValueError("Old/new evidence projections differ")
    if "system_prompt" not in state:
        # Recover only the exact archived template slot, not current settings.
        marker = "<captured-system-slot>"
        template = module._build_answer_prompt({**state, "system_prompt": marker}, context)
        prefix, suffix = template.split(marker)
        if not generation["prompt"].startswith(prefix) or not generation["prompt"].endswith(suffix):
            raise ValueError("Cannot reconstruct captured system prompt")
        state["system_prompt"] = generation["prompt"][len(prefix):len(generation["prompt"]) - len(suffix)]
    return state


async def run_node(module, state, model, *, reasoning=None, expected_prompt=None):
    started = time.monotonic()
    role_models = []
    with ExitStack() as overrides, observe_answers() as capture, turn_budget(), \
            patch.object(module, "_with_answer_evidence_rerank", new=AsyncMock(side_effect=lambda value: value)), \
            patch.object(postgres_client, "get_model", return_value=model), \
            patch.object(module.agent_model_role_service, "get_model_by_role", return_value=model), \
            patch.object(module.agent_model_role_service, "apply_agent_llm_params", return_value=model):
        if expected_prompt is not None:
            generate = module._generate_answer_response
            first = True
            async def checked_generation(prompt, *args, **kwargs):
                nonlocal first
                if first and prompt != expected_prompt:
                    raise ValueError("Writer input changed before inference")
                first = False
                return await generate(prompt, *args, **kwargs)
            overrides.enter_context(patch.object(module, "_generate_answer_response", checked_generation))
        if reasoning is not None:
            prepare_writer = module._prepare_answer_model
            prepare_reviewer = module._prepare_answer_review_model
            def writer_model(value):
                prepared = {**prepare_writer(value), "reasoning": reasoning}
                role_models.append({"role": "writer", "model": public_model(prepared)})
                return prepared
            def reviewer_model(value):
                prepared = prepare_reviewer(value)
                if prepared.get("reasoning") is not True:
                    raise ValueError("Both conditions require review thinking enabled")
                role_models.append({"role": "reviewer", "model": public_model(prepared)})
                return prepared
            overrides.enter_context(patch.object(module, "_prepare_answer_model", writer_model))
            overrides.enter_context(patch.object(module, "_prepare_answer_review_model", reviewer_model))
        try:
            result = await module.answer_node(copy.deepcopy(state))
            status, error = "ok", None
        except Exception as exc:
            result, status = {}, "runtime_error"
            cause = exc.__cause__ or exc
            error = {"type": type(exc).__name__, "code": getattr(cause, "code", type(cause).__name__)}
    return {"status": status, "error": error, **result, "observation": capture.as_dict(),
            "role_models": role_models,
            "duration_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="evals/icp_gate_review_inputs.json")
    parser.add_argument("--archive")
    parser.add_argument("--compare-reasoning", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:19999")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if bool(args.archive) == args.compare_reasoning:
        parser.error("Choose exactly one of --archive or --compare-reasoning")
    old = answer if args.compare_reasoning else archived_node(Path(args.archive))
    captures = [read_json(Path(p)) for p in read_json(Path(args.inputs))["generation_captures"]]
    generation_inputs = []
    for capture in captures:
        generation = next(i for i in capture["observation"]["inputs"] if i["stage"] == "generation")
        if args.compare_reasoning:
            generation = restore_generation(generation)
            state = copy.deepcopy(generation["state"])
            if (answer._format_context(state) != generation["context"]
                    or answer._build_answer_prompt(state, generation["context"]) != generation["prompt"]
                    or answer._extract_citations(state) != generation["citations"]):
                raise ValueError("Current generation input no longer matches frozen capture")
        else:
            state = same_facts(generation, old)
        if answer._format_context(state) != old._format_context(state):
            raise ValueError("Old/new source view differs")
        generation_inputs.append((capture, generation, state))
    if args.validate_only:
        print(json.dumps({"pairs": len(generation_inputs), "same_facts": True}))
        return 0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = requests.get(args.base_url + "/api/v1/models/current-config", timeout=30)
    config.raise_for_status()
    registered = config.json()["chat_model"]
    cases, _ = evaluator._load_cases("evals/icp_multiturn_dialogue_cases.json")
    targets = {(case.case_id, turn.index): turn for case in cases for turn in case.turns}
    judge = evaluator.SemanticJudge(evaluator._resolve_judge_config(args.base_url, args, required=True))
    source_paths = [Path(answer.__file__), Path("backend/agent/answer_review.py"),
                    Path("backend/agent/citation_binding.py"), Path("scripts/legacy_answer_review.py"),
                    Path(__file__), Path("backend/core/llm_service.py"), Path("backend/agent/model_runtime_policy.py")]
    hashes = {str(p): fingerprint(p.read_text()) for p in source_paths}
    write_private(output / "inputs.json", {"captures": captures, "source_hashes": hashes,
        "archive": args.archive, "judge": judge.config.public_dict(), "compare_reasoning": args.compare_reasoning,
        "projection": "exact_current_capture" if args.compare_reasoning else "same_prebatch_projection_of_frozen_raw_state_not_historical_display",
        "common_inputs": [{"state": state, "context": old._format_context(state),
                           "citations": old._extract_citations(state)} for _, _, state in generation_inputs]})
    results = []
    for index, (capture, generation, state) in enumerate(generation_inputs):
        model = {**registered, **generation["model"]}
        if model["model_id"] != registered["model_id"]:
            raise ValueError("Captured model no longer current")
        target = targets[(capture["case_id"], capture["turn_index"])]
        if target.question != state["query"]:
            raise ValueError("Original question changed")
        conditions = [("old", old, False if args.compare_reasoning else None),
                      ("new", answer, True if args.compare_reasoning else None)]
        if index % 2:
            conditions.reverse()
        for condition, module, reasoning in conditions:
            if hashes != {str(p): fingerprint(p.read_text()) for p in source_paths}:
                raise ValueError("Code changed during comparison")
            result = {"condition": condition,
                      "case_id": capture["case_id"], "turn_index": capture["turn_index"],
                      "input_sha256": fingerprint({"state": state, "context": old._format_context(state),
                                                    "citations": old._extract_citations(state), "model": public_model(model)}),
                      **asyncio.run(run_node(module, state, model, reasoning=reasoning,
                                             expected_prompt=generation["prompt"] if args.compare_reasoning else None))}
            if result["status"] == "ok":
                messages = [m for m in state["messages"] if m.get("role") in {"user", "assistant"}]
                prior = [{"question": messages[i]["content"], "answer": messages[i+1]["content"]}
                         for i in range(0, len(messages)-1, 2) if messages[i+1]["role"] == "assistant"]
                _, judgement = evaluator._evaluate_semantic_criteria(
                    target.assertions["semantic_criteria"], judge=judge, question=target.question,
                    prior_turns=prior, answer=result["final_answer"],
                    reference_context=target.metadata.get("resolved_gold_context") or [],
                    answer_citations=result["citations"])
                result["judgement"] = judgement
                result["core_passed"] = judgement["status"] == "completed" and all(c["passed"] for c in judgement["criteria"])
            result["publication_checks"] = publication_checks(result, target.assertions)
            result["automated_passed"] = bool(result.get("core_passed")) and all(
                check["passed"] for check in result["publication_checks"])
            results.append(result)
            write_private(output / f"generation-{len(results):02d}.json", result)
            print(f"[{len(results)}/{len(captures)*2}] {result['condition']} {result['case_id']} {result['status']} semantic={result.get('core_passed')}", flush=True)
    report = {"schema_version": "icp-generation-comparison/v1", "results": results,
              "end_to_end_score": None, "same_model_self_judge": True}
    write_private(output / "report.json", report)
    return int(any(r["condition"] == "new" and not r.get("automated_passed") for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
