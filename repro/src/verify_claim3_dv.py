"""Independent process verifier for the Claim 3 data-valuation audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from claim3_dv_contract import evidence_errors, load_rows, paired_tests


def main() -> None:
    raw_path, result_path = map(Path, sys.argv[1:3])
    rows = load_rows(raw_path)
    errors = evidence_errors(rows)
    recomputed = paired_tests(rows) if not errors else []
    result = json.loads(result_path.read_text())
    expected = "FALSIFIED" if any(test["counterexample"] for test in recomputed) else "BLOCKED"
    if result.get("verdict") != expected:
        errors.append(f"verdict {result.get('verdict')} disagrees with raw evidence {expected}")
    reported = {
        (item["task_id"], item["budget"], item["baseline"])
        for item in result.get("counterexamples", [])
    }
    observed = {
        (item["task_id"], item["budget"], item["baseline"])
        for item in recomputed
        if item["counterexample"]
    }
    if reported != observed:
        errors.append("reported counterexample cells disagree with raw-data recomputation")
    checker = {
        "verifier_exit": 0 if not errors else 1,
        "errors": errors,
        "raw_rows": len(rows),
        "recomputed_comparisons": len(recomputed),
        "recomputed_counterexamples": sorted(observed),
    }
    output = result_path.parent / "independent_checker.json"
    output.write_text(json.dumps(checker, indent=2) + "\n")
    print(json.dumps(checker, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

