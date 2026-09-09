"""Re-score every saved answer without invoking or modifying the diagnosis agent.

Run with python -m scripts.rejudge_icp_report. Original reports are immutable;
the new report retains all original turns, observations and runtime failures.
The suite may change grading criteria, never the saved questions or answers.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from scripts import evaluate_icp_paraphrases as evaluator


def rejudge_case(record, expected, judge):
    if record.get("case_id") != expected.case_id or len(record.get("turns") or []) != len(expected.turns):
        raise ValueError("saved case identity or turn count does not match suite")
    result = copy.deepcopy(record)
    previous = []
    for saved, turn in zip(result["turns"], expected.turns):
        if saved.get("question") != turn.question or saved.get("turn_id") != turn.turn_id:
            raise ValueError("saved question identity changed; regrading is not a new run")
        facts = saved.get("evidence") or {}
        answer = saved.get("answer") or ""
        citations = saved.get("citations") or []
        errors = saved.get("runtime_errors") or []
        event_types = saved.get("event_types") or []
        checks = evaluator._evaluate_assertions(
            turn.assertions, answer=answer, prior_answer=previous[-1]["answer"] if previous else "",
            tools=saved.get("tools") or [], doc_types=facts.get("doc_types") or [],
            used_doc_types=facts.get("used_doc_types") or [], used_record_ids=facts.get("used_record_ids") or [],
            evidence_count=len(facts.get("catalog") or []), used_evidence_count=len(facts.get("used_evidence_ids") or []),
            citation_count=len(citations), event_types=event_types, scope_routes=saved.get("scope_routes") or [],
            duration_seconds=saved.get("duration_seconds") or 0, runtime_errors=errors,
            done_received="done" in event_types, answer_received=bool(answer) and "answer" in event_types,
            completion_status=saved.get("completion_status") or "",
            delivered_answer_fields=saved.get("delivered_answer_fields") or [],
            answer_delivery_manifest=saved.get("answer_delivery_manifest") or [],
        )
        numeric, numeric_reports = evaluator._evaluate_numeric_expectations(
            turn.assertions.get("numeric_expectations", []), answer=answer,
            structured_sources=evaluator._numeric_structured_sources(citations, facts.get("catalog") or [], saved.get("tool_calls") or []),
        )
        semantic, judgement = evaluator._evaluate_semantic_criteria(
            turn.assertions.get("semantic_criteria", []), judge=judge, question=turn.question,
            prior_turns=previous, answer=answer, answer_citations=citations,
            reference_context=turn.metadata.get("resolved_gold_context") or [],
        )
        checks.extend(numeric)
        checks.extend(semantic)
        saved.update(assertion_results=checks, assertions=dict(turn.assertions),
                     numeric_expectation_results=numeric_reports, semantic_judgement=judgement,
                     passed=not errors and all(check["passed"] for check in checks))
        previous.append({"question": turn.question, "answer": answer})
    result["passed"] = all(turn["passed"] for turn in result["turns"])
    # Keep compatibility projections consistent with the last regraded turn.
    for key in ("assertions", "assertion_results", "numeric_expectation_results", "semantic_judgement"):
        result[key] = result["turns"][-1][key]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--suite", default="evals/icp_multiturn_dialogue_cases.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=evaluator.DEFAULT_BASE_URL)
    parser.add_argument("--judge-timeout-seconds", type=float, default=60)
    for field in ("model-id", "provider", "model", "base-url", "api-url", "api-key"):
        parser.add_argument(f"--judge-{field}", default="")
    args = parser.parse_args()
    source, target = Path(args.report), Path(args.output)
    if source.resolve() == target.resolve() or target.exists():
        raise ValueError("choose a new output; original and prior regrades cannot be overwritten")
    raw = source.read_bytes()
    original = json.loads(raw)
    cases, _ = evaluator._load_cases(args.suite)
    by_id = {case.case_id: case for case in cases}
    saved = original.get("cases") or []
    ids = [row["case_id"] for row in saved]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError("regrade all suite cases, not only failures or an unfinished checkpoint")
    config = evaluator._resolve_judge_config(args.base_url, args, required=True)
    judge = evaluator.SemanticJudge(config)
    result = copy.deepcopy(original)
    result["cases"] = []
    result["evaluation_complete"] = False
    result["run_status"] = "regrading"
    result["summary"] = {"total": len(saved), "completed_cases": 0, "status": "regrading"}
    result.setdefault("configuration", {})["semantic_judge"] = config.public_dict()
    result["regrade"] = {"status": "running", "source_report_sha256": hashlib.sha256(raw).hexdigest(),
        "source_report": str(source), "original_judge": (original.get("configuration") or {}).get("semantic_judge"),
        "suite_sha256": hashlib.sha256(Path(args.suite).read_bytes()).hexdigest(),
        "judge": config.public_dict(), "new_agent_requests": 0, "started_at": evaluator._utc_now()}
    for index, record in enumerate(saved, 1):
        graded = rejudge_case(record, by_id[record["case_id"]], judge)
        result["cases"].append(graded)
        result["summary"]["completed_cases"] = index
        print(f"[{index}/{len(saved)}] {'PASS' if graded['passed'] else 'FAIL'} {graded['case_id']}", flush=True)
        evaluator._write_report(result, str(target) + ".checkpoint.json", False)
    turns = [turn for record in result["cases"] for turn in record["turns"]]
    passed = sum(record["passed"] for record in result["cases"])
    turn_passed = sum(turn["passed"] for turn in turns)
    result["summary"] = {**original.get("summary", {}), "total": len(saved), "passed": passed,
        "failed": len(saved) - passed, "pass_rate": round(passed / len(saved), 4),
        "total_turns": len(turns), "passed_turns": turn_passed, "failed_turns": len(turns) - turn_passed,
        "turn_pass_rate": round(turn_passed / len(turns), 4)}
    result["breakdowns"] = evaluator._evaluation_breakdowns(result["cases"])
    result["groups"] = evaluator._group_summaries(result["cases"])
    overall = result["breakdowns"]["overall"]
    result["evaluation_complete"] = not (overall["runtime_error_turns"] or overall["judge_error_turns"] or result["summary"].get("cleanup_failures"))
    result["regrade"].update(status="finished", finished_at=evaluator._utc_now())
    result["run_status"] = "finished"
    evaluator._write_report(result, str(target), False)
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0 if result["evaluation_complete"] and passed == len(saved) else 1


if __name__ == "__main__":
    raise SystemExit(main())
