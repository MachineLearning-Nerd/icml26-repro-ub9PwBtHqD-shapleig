"""Structural contract for the partial Claim 5 timing matrix."""
from __future__ import annotations

import math


TASKS = {
    "le_vit9_imagenet": 9,
    "dv_rf_bike_sharing": 10,
    "dv_gb_bike_sharing": 10,
    "dv_gb_california_housing": 10,
}
REPLICATES = range(1, 6)


def archive_sizes(p: int) -> tuple[int, ...]:
    return (p + 1, 16, 32, 64)


def timing_errors(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    expected = {
        (task, replicate, archive)
        for task, p in TASKS.items()
        for replicate in REPLICATES
        for archive in archive_sizes(p)
    }
    observed = {
        (str(row["task_id"]), int(row["replicate"]), int(row["archive_size"]))
        for row in rows
    }
    if len(rows) != len(expected):
        errors.append(f"expected {len(expected)} rows, got {len(rows)}")
    if observed != expected:
        errors.append("task/replicate/archive grid mismatch")
    for row in rows:
        task = str(row["task_id"])
        if task not in TASKS:
            errors.append(f"unknown task {task}")
            continue
        p = TASKS[task]
        archive = int(row["archive_size"])
        if int(row["players"]) != p:
            errors.append(f"{task}: wrong player count")
        if int(row["candidate_count"]) != (1 << p) - archive:
            errors.append(f"{task}/{archive}: wrong complete candidate count")
        for field in ("hp_fit_seconds", "eig_seconds"):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{task}/{archive}: invalid {field}")
        if int(row["fit_restarts"]) != 5:
            errors.append(f"{task}/{archive}: expected five fit restarts")
    return errors

