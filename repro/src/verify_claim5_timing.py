"""Independent parser/checker for the partial Claim 5 timing evidence."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from claim5_timing_contract import TASKS, archive_sizes


def main() -> None:
    raw_path, summary_path = map(Path, sys.argv[1:3])
    with raw_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors: list[str] = []
    cells: dict[tuple[str, int, int], dict] = {}
    for row in rows:
        key = (row["task_id"], int(row["replicate"]), int(row["archive_size"]))
        if key in cells:
            errors.append(f"duplicate cell {key}")
        cells[key] = row
    expected = {
        (task, replicate, archive)
        for task, p in TASKS.items()
        for replicate in range(1, 6)
        for archive in archive_sizes(p)
    }
    if set(cells) != expected:
        errors.append(f"expected {len(expected)} exact cells, got {len(cells)}")

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["task_id"], int(row["archive_size"]))].append(row)
        p = TASKS.get(row["task_id"])
        if p is None or int(row["players"]) != p:
            errors.append("task/player mismatch")
            continue
        if int(row["candidate_count"]) != (1 << p) - int(row["archive_size"]):
            errors.append("candidate set is not the complete unevaluated grid")
        for field in ("hp_fit_seconds", "eig_seconds"):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0:
                errors.append(f"invalid {field}")

    supplied = json.loads(summary_path.read_text())
    rebuilt = []
    for (task, archive), group in sorted(grouped.items()):
        rebuilt.append(
            {
                "task_id": task,
                "archive_size": archive,
                "hp_fit_mean_seconds": sum(float(r["hp_fit_seconds"]) for r in group)
                / len(group),
                "eig_mean_seconds": sum(float(r["eig_seconds"]) for r in group)
                / len(group),
            }
        )
    supplied_cells = {
        (row["task_id"], int(row["archive_size"])): row
        for row in supplied["by_task_archive"]
    }
    for row in rebuilt:
        supplied_row = supplied_cells.get((row["task_id"], row["archive_size"]))
        if supplied_row is None:
            errors.append("missing supplied aggregate")
            continue
        for field in ("hp_fit_mean_seconds", "eig_mean_seconds"):
            if not math.isclose(row[field], float(supplied_row[field]), rel_tol=1e-12):
                errors.append(f"aggregate mismatch: {field}")

    result = {
        "verifier": "claim_5_partial_timing",
        "raw_rows": len(rows),
        "aggregate_cells": len(rebuilt),
        "tasks": len({r["task_id"] for r in rows}),
        "maximum_players": max(int(r["players"]) for r in rows),
        "maximum_archive_size": max(int(r["archive_size"]) for r in rows),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

