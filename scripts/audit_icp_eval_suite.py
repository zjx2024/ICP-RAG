"""Offline structural/coverage audit for multi-turn evals, not a gold judge."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from scripts import evaluate_icp_paraphrases as evaluator


DEFAULT_SUITES = (
    "evals/icp_multiturn_regression_positioning.json",
    "evals/icp_multiturn_regression_analysis.json",
    "evals/icp_multiturn_regression_boundary.json",
)
REQUIRED_SCENARIOS = frozenset({
    "exact_code", "partial_code", "vague_symptom", "specific_fault",
    "related_question", "out_of_scope", "no_evidence", "clarification_recovery",
    "nonadjacent_reference", "topic_return", "object_correction",
    "parameter_correction", "neighbor_isolation", "multi_object_comparison",
    "summary_verification", "history_window_boundary",
})


def audit_suites(paths: Sequence[str]) -> dict[str, Any]:
    case_ids: set[str] = set()
    categories: Counter[str] = Counter()
    lengths: Counter[int] = Counter()
    scenarios: Counter[str] = Counter()
    issues: list[dict[str, str]] = []
    files = []
    totals: Counter[str] = Counter()
    for path in paths:
        cases, _ = evaluator._load_cases(path)
        files.append({"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "cases": len(cases)})
        for case in cases:
            if case.case_id in case_ids:
                raise evaluator.EvaluationInputError(f"duplicate case across files: {case.case_id}")
            case_ids.add(case.case_id)
            categories[str(case.metadata.get("category") or "uncategorized")] += 1
            lengths[len(case.turns)] += 1
            declared = case.metadata.get("scenarios", [])
            if not isinstance(declared, list) or not all(isinstance(item, str) for item in declared):
                raise evaluator.EvaluationInputError(f"invalid scenarios: {case.case_id}")
            scenarios.update(set(declared))
            if not case.metadata.get("gold_sources"):
                issues.append({"case_id": case.case_id, "kind": "gold_provenance_not_registered"})
            for turn in case.turns:
                assertions = turn.assertions
                totals["turns"] += 1
                for name in ("semantic_criteria", "numeric_expectations", "required_used_record_ids", "any_used_record_ids", "answer_regex"):
                    totals[name] += bool(assertions.get(name))
                totals["wide_or_regex"] += any("|" in value for value in assertions.get("answer_regex", []))
                if not assertions.get("semantic_criteria"):
                    issues.append({"case_id": case.case_id, "turn_id": turn.turn_id, "kind": "semantic_criteria_missing"})
    missing = sorted(REQUIRED_SCENARIOS - set(scenarios))
    long_dialogue_missing = not any(length >= 3 for length in lengths)
    return {
        "schema_version": "icp-eval-audit/v1", "files": files,
        "cases": len(case_ids), "counts": dict(totals),
        "turn_lengths": {str(key): value for key, value in sorted(lengths.items())},
        "categories": dict(sorted(categories.items())),
        "declared_scenarios": dict(sorted(scenarios.items())),
        "missing_declared_scenarios": missing,
        "long_dialogue_missing": long_dialogue_missing,
        "issues": issues,
        "ready_for_gold_review": not issues and not missing and not long_dialogue_missing,
        "note": "场景标签须由人工核对题目；结构检查不能证明 gold 正确，也不能替代真实评测。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="*", default=list(DEFAULT_SUITES))
    parser.add_argument("--output")
    parser.add_argument("--strict-quality", action="store_true")
    args = parser.parse_args()
    report = audit_suites(args.suites)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cases", "counts", "turn_lengths", "long_dialogue_missing", "ready_for_gold_review")}, ensure_ascii=False))
    return 1 if args.strict_quality and not report["ready_for_gold_review"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
