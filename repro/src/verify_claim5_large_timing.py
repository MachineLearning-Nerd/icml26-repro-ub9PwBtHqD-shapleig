"""Independent parser for the source-matched large-game timing CSV."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

from claim5_large_timing_contract import (
    TASKS,
    archive_sizes,
    expected_candidate_count,
)


def main() -> None:
    raw_path, summary_path = map(Path, sys.argv[1:3])
    with raw_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors: list[str] = []
    cells = {}
    for row in rows:
        key = (row["task_id"], int(row["replicate"]), int(row["archive_size"]))
        if key in cells:
            errors.append(f"duplicate cell {key}")
        cells[key] = row
    expected = {
        (task, 1, archive)
        for task, (_, p) in TASKS.items()
        for archive in archive_sizes(p)
    }
    if set(cells) != expected:
        errors.append(f"expected {len(expected)} exact cells, got {len(cells)}")
    for row in rows:
        task = row["task_id"]
        if task not in TASKS:
            errors.append(f"unknown task {task}")
            continue
        p = TASKS[task][1]
        archive = int(row["archive_size"])
        if int(row["players"]) != p:
            errors.append(f"{task}: player mismatch")
        if int(row["candidate_count"]) != expected_candidate_count(p, archive):
            errors.append(f"{task}: candidate trajectory mismatch")
        for field in ("hp_fit_seconds", "eig_seconds", "maximum_eig"):
            if not math.isfinite(float(row[field])):
                errors.append(f"{task}: invalid {field}")
    supplied = json.loads(summary_path.read_text())
    if supplied["tasks"] != 3 or supplied["maximum_players"] != 101:
        errors.append("summary does not cover all three exact large tasks")
    if supplied["maximum_archive_size"] != 512:
        errors.append("summary does not reach budget 512")
    if supplied["source_candidate_pool"] != 1024:
        errors.append("summary candidate pool differs from author config")
    maxima = supplied["observed_maxima"]
    if not math.isclose(
        maxima["hp_fit_seconds"],
        max(float(row["hp_fit_seconds"]) for row in rows),
        rel_tol=1e-12,
    ):
        errors.append("GP-fit maximum is not reproducible from raw rows")
    if not math.isclose(
        maxima["eig_seconds"],
        max(float(row["eig_seconds"]) for row in rows),
        rel_tol=1e-12,
    ):
        errors.append("EIG maximum is not reproducible from raw rows")
    result = {
        "verifier": "claim_5_large_timing",
        "raw_rows": len(rows),
        "tasks": len({row["task_id"] for row in rows}),
        "maximum_players": max(int(row["players"]) for row in rows),
        "maximum_archive_size": max(int(row["archive_size"]) for row in rows),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
