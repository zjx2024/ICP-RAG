"""Calibrate the real semantic judge against fixed, source-contained examples.

Run with ``python -m scripts.calibrate_icp_judge``.  This does not call the
diagnosis agent, mutate models or create sessions.  --validate-only is offline.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from scripts import evaluate_icp_paraphrases as evaluator


def load_samples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    seen = set()
    for case in payload["cases"]:
        criteria = evaluator._normalize_semantic_criteria(case["criteria"], case["id"])
        for answer in case["answers"]:
            identity = f"{case['id']}/{answer['id']}"
            if identity in seen or set(answer["expected"]) != {item["id"] for item in criteria}:
                raise ValueError(f"duplicate sample or incomplete gold: {identity}")
            if not all(isinstance(value, bool) for value in answer["expected"].values()):
                raise ValueError(f"expected decisions must be booleans: {identity}")
            if not case.get("question") or not answer.get("text"):
                raise ValueError(f"empty question/answer: {identity}")
            seen.add(identity)
            samples.append({
                "id": identity, "question": case["question"],
                "prior_turns": case.get("prior_turns", []), "answer": answer["text"],
                "criteria": criteria, "expected": answer["expected"],
                "reference_context": case.get("reference_context", []),
                "answer_citations": answer.get("citations", []),
            })
    if not samples:
        raise ValueError("empty calibration set")
    return samples


def evaluate_sample(sample: dict[str, Any], judge: Any) -> dict[str, Any]:
    _, judgement = evaluator._evaluate_semantic_criteria(
        sample["criteria"], judge=judge, question=sample["question"],
        prior_turns=sample["prior_turns"], answer=sample["answer"],
        reference_context=sample.get("reference_context", []),
        answer_citations=sample.get("answer_citations", []),
    )
    actual = {item["id"]: item.get("passed") for item in judgement.get("criteria", [])}
    return {
        "id": sample["id"], "expected": sample["expected"], "judgement": judgement,
        "passed": judgement["status"] == "completed" and actual == sample["expected"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="evals/icp_judge_calibration.json")
    parser.add_argument("--base-url", default=evaluator.DEFAULT_BASE_URL)
    parser.add_argument("--output", default=".artifacts/evals/judge-calibration.json")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--concurrency", type=evaluator._positive_int, default=2)
    for field in ("model-id", "provider", "model", "base-url", "api-url", "api-key"):
        parser.add_argument(f"--judge-{field}", default="")
    parser.add_argument("--judge-timeout-seconds", type=evaluator._positive_float, default=60.0)
    args = parser.parse_args()
    samples = load_samples(Path(args.cases))
    if args.validate_only:
        print(json.dumps({"samples": len(samples), "valid": True}))
        return 0
    config = evaluator._resolve_judge_config(args.base_url, args, required=True)
    judge = evaluator.SemanticJudge(config)
    results = []
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(samples))) as pool:
        futures = [pool.submit(evaluate_sample, sample, judge) for sample in samples]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{len(results)}/{len(samples)}] {'PASS' if result['passed'] else 'FAIL'} {result['id']}", flush=True)
            if not result["passed"]:
                print(json.dumps(result["judgement"], ensure_ascii=False), flush=True)
    report = {
        "schema_version": "icp-judge-calibration/v1", "judge": config.public_dict(),
        "total": len(results), "passed": sum(item["passed"] for item in results),
        "cases": sorted(results, key=lambda item: item["id"]),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["total"] == report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
