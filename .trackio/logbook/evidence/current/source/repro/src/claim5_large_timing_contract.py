"""Machine-checkable structure for the large-game Claim 5 timing audit."""
from __future__ import annotations

import math


TASKS = {
    "le_rf_corrgroups60": ("corrgroups60", 60),
    "le_rf_nhanes": ("nhanesi", 79),
    "le_rf_crime": ("communitiesandcrime", 101),
}
ARCHIVE_TARGET = 512
CANDIDATE_POOL = 1024
REPLICATES = (1,)


def archive_sizes(p: int) -> tuple[int, int]:
    return (p + 1, ARCHIVE_TARGET)


def expected_candidate_count(p: int, archive: int) -> int:
    return CANDIDATE_POOL - (archive - (p + 1))


def large_timing_errors(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    expected = {
        (task, replicate, archive)
        for task, (_, p) in TASKS.items()
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
        errors.append("large-task/replicate/archive grid mismatch")
    for row in rows:
        task = str(row["task_id"])
        if task not in TASKS:
            errors.append(f"unknown task {task}")
            continue
        p = TASKS[task][1]
        archive = int(row["archive_size"])
        if int(row["players"]) != p:
            errors.append(f"{task}: wrong player count")
        if int(row["candidate_count"]) != expected_candidate_count(p, archive):
            errors.append(f"{task}/{archive}: wrong remaining candidate count")
        if int(row["source_candidate_pool"]) != CANDIDATE_POOL:
            errors.append(f"{task}/{archive}: source candidate pool is not 1024")
        if int(row["fit_restarts"]) != 5:
            errors.append(f"{task}/{archive}: expected five GP restarts")
        for field in ("hp_fit_seconds", "eig_seconds", "maximum_eig"):
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                errors.append(f"{task}/{archive}: invalid {field}")
        if float(row["hp_fit_seconds"]) <= 0 or float(row["eig_seconds"]) <= 0:
            errors.append(f"{task}/{archive}: non-positive timing")
    return errors
