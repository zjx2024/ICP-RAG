"""Replay saved review exchanges without retrieval, generation or model calls."""
import argparse
import json
from pathlib import Path

from backend.agent.answer_observation import fingerprint
from backend.agent import answer_review
from scripts import legacy_answer_review


def replay(snapshot):
    recorded = {key: value for key, value in snapshot.items() if key != "sha256"}
    if snapshot.get("schema_version") != "icp-review-exchange/v1" or fingerprint(recorded) != snapshot.get("sha256"):
        raise ValueError("Review snapshot is incomplete or changed")
    if snapshot.get("transport"):
        # An incomplete provider completion never reached the validator. Even
        # a coincidentally parseable prefix cannot be replayed as its verdict.
        return {"snapshot_sha256": snapshot["sha256"], "model_calls": 0,
                "same_verdict": False, "validation_executed": False,
                "replay_status": "requires_complete_model_output", "error": snapshot["error"]}
    protocol = answer_review if snapshot["prepared"].get("review_version") == 5 else legacy_answer_review
    options = {} if protocol is answer_review else {"salvage": snapshot["salvage"]}
    try:
        result = protocol.validate_review(snapshot["raw"], prepared=snapshot["prepared"],
                                 ledger={int(k): v for k, v in snapshot["ledger"].items()},
                                 safety=snapshot["safety"], calculation_requests=snapshot["calculation_requests"],
                                 **options)
        current = {"result": result, "error": None}
    except protocol.AnswerReviewError as exc:
        current = {"result": None, "error": {"code": exc.code, "feedback": exc.feedback}}
    return {"snapshot_sha256": snapshot["sha256"], "model_calls": 0,
            "same_verdict": current == {key: snapshot[key] for key in ("result", "error")}, **current}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="Private review file or diagnostic capture JSON")
    args = parser.parse_args()
    payload = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    snapshots = [payload] if payload.get("schema_version") == "icp-review-exchange/v1" else payload["observation"]["reviews"]
    if not snapshots:
        raise ValueError("No review exchanges recorded; this is not a successful replay")
    results = [replay(snapshot) for snapshot in snapshots]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return int(not all(result["same_verdict"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
