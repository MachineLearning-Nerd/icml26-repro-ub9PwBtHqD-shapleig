"""Independent reconstruction of the executed partial Claim 2 scope."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from claim2_partial_contract import BUDGETS, TASKS


def main() -> None:
    raw_path, summary_path = map(Path, sys.argv[1:3])
    with raw_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors: list[str] = []
    cells: dict[tuple[str, int, int], dict] = {}
    for row in rows:
        key = (row["task_id"], int(row["replicate"]), int(row["budget"]))
        if key in cells:
            errors.append(f"duplicate cell {key}")
        cells[key] = row
    expected = {
        (task, replicate, budget)
        for task in TASKS
        for replicate in range(1, 31)
        for budget in BUDGETS
    }
    if set(cells) != expected:
        errors.append(f"expected {len(expected)} exact cells, got {len(cells)}")

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != "ShaplEIG":
            errors.append("non-ShaplEIG method present")
        value = float(row["mse"])
        if not math.isfinite(value) or value < 0:
            errors.append("invalid MSE")
        grouped[(row["task_id"], int(row["budget"]))].append(value)

    supplied = json.loads(summary_path.read_text())
    supplied_cells = {
        (row["task_id"], int(row["budget"])): row
        for row in supplied["by_task_budget"]
    }
    for key, values in grouped.items():
        row = supplied_cells.get(key)
        if row is None:
            errors.append(f"missing aggregate {key}")
            continue
        mean = sum(values) / len(values)
        if not math.isclose(mean, float(row["mean_mse"]), rel_tol=1e-12):
            errors.append(f"mean mismatch {key}")
        if int(row["replicates"]) != 30:
            errors.append(f"replicate mismatch {key}")

    result = {
        "verifier": "claim_2_partial_scope",
        "raw_rows": len(rows),
        "tasks": len({row["task_id"] for row in rows}),
        "families": len({row["family"] for row in rows}),
        "replicates_per_task": 30,
        "maximum_budget": max(int(row["budget"]) for row in rows),
        "aggregate_cells": len(grouped),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

