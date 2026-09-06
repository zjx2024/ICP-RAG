"""Run the three reviewed regression suites sequentially and archive each repeat.

This is an explicit live-service task, not part of ordinary CI/unittest. A new
output directory is mandatory. Failed answers do not skip the remaining suites;
configuration/archive failures or changed code/data stop the campaign. Reports
from different repeats are never pooled into a misleading larger sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from scripts import audit_icp_eval_suite as audit
from scripts import capture_icp_eval_baseline as baseline
from scripts import evaluate_icp_paraphrases as evaluator

ROOT = Path(__file__).resolve().parents[1]


def live_configuration(manifest: dict[str, Any], agent: dict[str, Any]) -> str:
    """Freeze selected agent, model configuration and active document versions."""
    application = manifest.get("services", {}).get("application", {})
    models = application.get("models", {})
    if not application.get("reachable") or not models.get("ok"):
        raise ValueError("Live application and model configuration are required")
    payload = {"models": models.get("configuration"), "documents": application.get("documents"),
               "agent": baseline._scrub_sensitive(agent)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def quality_gate(summary: dict[str, Any]) -> dict[str, Any]:
    """Predeclared quality targets; errors never shrink the success denominator."""
    overall = summary.get("breakdowns", {}).get("overall", {})
    judged = overall.get("semantic_judged_turns", 0)
    semantic_rate = overall.get("semantic_passed_turns", 0) / judged if judged else 0.0
    passed = bool(summary.get("evaluation_complete") and judged == summary.get("total_turns")
                  and semantic_rate >= 0.90 and summary.get("pass_rate", 0) >= 0.80
                  and not overall.get("runtime_error_turns") and not overall.get("judge_error_turns"))
    return {"passed": passed, "semantic_turn_target": 0.90, "complete_case_target": 0.80,
            "semantic_turn_rate": round(semantic_rate, 4),
            "note": "Automated synthetic quality gate only; safety expert review and comparable cost remain separate."}


def selected_runtime_configuration(args: argparse.Namespace) -> dict[str, Any]:
    agent = evaluator._resolve_agent(args.base_url, args.agent_id, 30.0)
    judge_args = evaluator.build_parser().parse_args([
        str(ROOT / audit.DEFAULT_SUITES[0]), "--judge-model-id", args.judge_model_id,
        "--judge-timeout-seconds", str(args.judge_timeout_seconds),
    ])
    judge = evaluator._resolve_judge_config(args.base_url, judge_args, required=True)
    return {"agent": agent, "judge": judge.public_dict()}


def collect_trace_costs(report: dict[str, Any], *, base_url: str) -> dict[str, Any]:
    """Read summary metrics for this suite's own sessions, never draft text."""
    try:
        traces = evaluator._request_json("GET", evaluator._api_url(base_url, "/api/traces?limit=1000"), timeout=30.0)
        if not isinstance(traces, list):
            raise ValueError("Trace summary response is not a list")
        cases = {case["session_id"]: case for case in report["cases"] if case.get("session_id")}
        rows = []
        for trace in traces:
            case = cases.get(trace.get("session_id")) if isinstance(trace, dict) else None
            if not case:
                continue
            matching = [turn for turn in case["turns"] if turn.get("question") == trace.get("query")]
            rows.append({"case_id": case["case_id"], "trace_id": trace.get("trace_id"),
                         "turn_ids": [turn.get("turn_id") for turn in matching],
                         "llm_calls": trace.get("llm_calls"), "llm_failures": trace.get("llm_failures"),
                         "duration_ms": trace.get("duration_ms")})
        return {"status": "observed" if rows else "unavailable", "traces": rows,
                "planned_turns": sum(len(case["turns"]) for case in report["cases"]),
                "observed_traces": len(rows),
                "total_llm_calls": sum(row["llm_calls"] for row in rows if isinstance(row["llm_calls"], int)),
                "comparison": "No same-condition cost improvement inferred; failed/skipped turns reported separately."}
    except (ValueError, OSError, evaluator.ApiError) as exc:
        return {"status": "unavailable", "error": str(exc), "comparison": "not_verified"}


def calibration_status(path: str, judge: dict[str, Any]) -> dict[str, Any]:
    if not path:
        return {"qualified": False, "status": "not_provided"}
    raw = Path(path).read_bytes()
    report = json.loads(raw)
    samples = report.get("cases") or []
    same_judge = all(report.get("judge", {}).get(key) == judge.get(key)
                     for key in ("model_id", "model_name", "protocol_version", "timeout_seconds"))
    qualified = bool(samples and same_judge and report.get("total") == len(samples)
                     and all(item.get("passed") is True and item.get("judgement", {}).get("status") == "completed"
                             for item in samples))
    return {"qualified": qualified, "status": "reviewed", "same_judge": same_judge,
            "total": len(samples), "passed": sum(item.get("passed") is True for item in samples),
            "sha256": hashlib.sha256(raw).hexdigest(), "path": str(Path(path).resolve())}


def input_fingerprint(manifest: dict[str, Any]) -> dict[str, str]:
    groups = {**manifest["code_inputs"], "fixtures": manifest["fixtures"],
              "evaluation_suites": manifest["evaluation_suites"]}
    result = {key: value.get("aggregate_sha256", "") for key, value in groups.items()}
    if not all(result.values()):
        raise ValueError("Baseline is missing an input fingerprint")
    return result


def repeat_summary(reports: list[dict[str, Any]], *, expected_suites: int) -> dict[str, Any]:
    if len(reports) != expected_suites:
        return {"run_status": "in_progress", "completed_suites": len(reports)}
    cases = [case for report in reports for case in report["cases"]]
    identities = [case["case_id"] for case in cases]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate case identity across campaign suites")
    turns = [turn for case in cases for turn in case["turns"]]
    passed = sum(case.get("passed") is True for case in cases)
    passed_turns = sum(turn.get("passed") is True for turn in turns)
    return {
        "run_status": "finished",
        "evaluation_complete": all(report.get("evaluation_complete") is True for report in reports),
        "total": len(cases), "passed": passed, "total_turns": len(turns),
        "passed_turns": passed_turns,
        "pass_rate": round(passed / len(cases), 4) if cases else 0.0,
        "turn_pass_rate": round(passed_turns / len(turns), 4) if turns else 0.0,
        "breakdowns": evaluator._evaluation_breakdowns(cases),
    }


def run(args: argparse.Namespace) -> int:
    suite_paths = [str(ROOT / relative) for relative in audit.DEFAULT_SUITES]
    audited = audit.audit_suites(suite_paths)
    if not audited["ready_for_gold_review"]:
        raise ValueError("Regression suite audit is not ready")
    groups = {"regression_100": suite_paths}
    if args.acceptance:
        groups = {"baseline_19": [str(ROOT / "evals/icp_multiturn_dialogue_cases.json")],
                  **groups, "expression_holdout": [str(ROOT / "evals/icp_expression_holdout.json")]}
        for paths in groups.values():
            for path in paths:
                cases, _ = evaluator._load_cases(path)
                if any(not turn.assertions.get("semantic_criteria") for case in cases for turn in case.turns):
                    raise ValueError("Acceptance requires semantic criteria on every turn")
    output = Path(args.output_dir).resolve()
    # Refuse even an empty existing directory: the caller must deliberately
    # choose a fresh run identity, never overwrite an earlier baseline/report.
    output.mkdir(parents=True, exist_ok=False)
    before = baseline.capture_baseline(ROOT, base_url=args.base_url)
    evaluator._write_report(before, str(output / "before.json"), False)
    fingerprint = input_fingerprint(before)
    runtime_fingerprint = ""
    calibration = {"qualified": False, "status": "not_provided"}
    if args.acceptance:
        runtime_configuration = selected_runtime_configuration(args)
        runtime_fingerprint = live_configuration(before, runtime_configuration)
        calibration = calibration_status(args.judge_calibration, runtime_configuration.get("judge", {}))
    evaluator._write_report(audited, str(output / "suite-audit.json"), False)
    campaign: dict[str, Any] = {
        "schema_version": "icp-eval-campaign/v1", "run_status": "in_progress",
        "purpose": "synthetic_regression_diagnostic", "requested_repeats": args.repeats,
        "cases_per_repeat": audited["cases"], "input_fingerprint": fingerprint,
        "runs": [], "repeat_summaries": [],
        "acceptance": args.acceptance, "live_configuration_fingerprint": runtime_fingerprint,
        "suite_groups": groups,
        "judge_calibration": calibration,
    }

    def save() -> None:
        evaluator._write_report(campaign, str(output / "campaign.json"), False)

    def verify_inputs() -> None:
        current = baseline.capture_baseline(ROOT, offline=True)
        if input_fingerprint(current) != fingerprint:
            raise ValueError("Code, gold or source inputs changed during evaluation")
        if args.acceptance:
            if args.judge_calibration and hashlib.sha256(Path(args.judge_calibration).read_bytes()).hexdigest() != calibration.get("sha256"):
                raise ValueError("Judge calibration report changed during evaluation")
            current_live = baseline.capture_baseline(ROOT, base_url=args.base_url)
            if live_configuration(current_live, selected_runtime_configuration(args)) != runtime_fingerprint:
                raise ValueError("Agent, model configuration or source documents changed during evaluation")

    any_failed = False
    save()
    try:
        for repeat in range(1, args.repeats + 1):
            reports_by_group: dict[str, list[dict[str, Any]]] = {name: [] for name in groups}
            for group_name, path in [(name, path) for name, paths in groups.items() for path in paths]:
                verify_inputs()
                suite_name = Path(path).stem
                target = output / f"repeat-{repeat}-{suite_name}.json"
                command = [path, "--base-url", args.base_url,
                           "--concurrency", str(args.concurrency),
                           "--judge-timeout-seconds", str(args.judge_timeout_seconds),
                           "--include-raw-events", "-o", str(target)]
                if args.agent_id:
                    command += ["--agent-id", args.agent_id]
                if args.judge_model_id:
                    command += ["--judge-model-id", args.judge_model_id]
                print(f"Starting repeat {repeat}/{args.repeats}: {suite_name}", flush=True)
                code = evaluator.main(command)
                if code not in (0, 1) or not target.is_file():
                    raise ValueError(f"Suite did not archive a final report (exit {code})")
                report = json.loads(target.read_text(encoding="utf-8"))
                if args.acceptance:
                    costs = collect_trace_costs(report, base_url=args.base_url)
                    evaluator._write_report(costs, str(target.with_suffix(".costs.json")), False)
                verify_inputs()
                reports_by_group[group_name].append(report)
                any_failed = any_failed or code != 0
                campaign["runs"].append({"repeat": repeat, "suite": suite_name, "group": group_name,
                                         "report": target.name, "exit_code": code})
                save()
            summaries = {name: repeat_summary(reports_by_group[name], expected_suites=len(paths))
                         for name, paths in groups.items()}
            if args.acceptance:
                for summary in summaries.values():
                    summary["quality_gate"] = quality_gate(summary)
                    gate = summary["quality_gate"]
                    gate["metric_targets_met"] = gate["passed"]
                    gate["judge_calibrated"] = calibration["qualified"]
                    gate["passed"] = gate["passed"] and calibration["qualified"]
                campaign["repeat_summaries"].append({"repeat": repeat, "groups": summaries,
                    "evaluation_complete": all(s["evaluation_complete"] for s in summaries.values())})
            else:
                campaign["repeat_summaries"].append({"repeat": repeat, **summaries["regression_100"]})
            save()
        after = baseline.capture_baseline(ROOT, base_url=args.base_url)
        evaluator._write_report(after, str(output / "after.json"), False)
        if input_fingerprint(after) != fingerprint:
            raise ValueError("Inputs changed before campaign completion")
        if args.acceptance:
            if live_configuration(after, selected_runtime_configuration(args)) != runtime_fingerprint:
                raise ValueError("Live configuration changed before campaign completion")
            campaign["automated_quality_passed"] = calibration["qualified"] and all(
                summary["quality_gate"]["passed"] for repeat in campaign["repeat_summaries"]
                for summary in repeat["groups"].values())
        campaign["run_status"] = "finished"
        campaign["evaluation_complete"] = all(
            item["evaluation_complete"] for item in campaign["repeat_summaries"])
        save()
        return 1 if any_failed or (args.acceptance and not campaign["automated_quality_passed"]) else 0
    except (ValueError, OSError, KeyError, evaluator.ApiError, KeyboardInterrupt) as exc:
        campaign["run_status"] = "interrupted"
        campaign["evaluation_complete"] = False
        campaign["failure"] = {"kind": type(exc).__name__, "message": str(exc)}
        save()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default=evaluator.DEFAULT_BASE_URL)
    parser.add_argument("--agent-id", default="")
    parser.add_argument("--judge-model-id", default="", help="Optional independently calibrated judge; does not change the agent default")
    parser.add_argument("--judge-calibration", default="", help="Frozen calibration report; unqualified judges allow diagnostics, never acceptance")
    parser.add_argument("--concurrency", type=evaluator._positive_int, default=1)
    parser.add_argument("--repeats", type=evaluator._positive_int, default=1)
    parser.add_argument("--judge-timeout-seconds", type=evaluator._positive_float, default=120.0)
    parser.add_argument("--acceptance", action="store_true",
                        help="Run 19/40, 100/216 and expression holdout separately on each repeat; verify live configuration")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, KeyError, evaluator.ApiError) as exc:
        print(f"Campaign stopped: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
