"""Structural contract for the executed partial Claim 2 task subset."""
from __future__ import annotations

import math


TASKS = {
    "dv_rf_bike_sharing": ("data_valuation", 10),
    "dv_gb_bike_sharing": ("data_valuation", 10),
    "dv_gb_california_housing": ("data_valuation", 10),
    "le_vit9_imagenet": ("local_explanation", 9),
}
BUDGETS = (16, 24, 32, 48, 64)


def executed_subset_errors(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    expected = {
        (task, replicate, budget)
        for task in TASKS
        for replicate in range(1, 31)
        for budget in BUDGETS
    }
    observed = {
        (str(row["task_id"]), int(row["replicate"]), int(row["budget"]))
        for row in rows
    }
    if len(rows) != len(expected):
        errors.append(f"expected {len(expected)} rows, got {len(rows)}")
    if observed != expected:
        errors.append("task/replicate/budget grid mismatch")
    for row in rows:
        task = str(row["task_id"])
        if task not in TASKS:
            errors.append(f"unknown task {task}")
            continue
        family, players = TASKS[task]
        if str(row["family"]) != family:
            errors.append(f"{task}: wrong family")
        if int(row["players"]) != players:
            errors.append(f"{task}: wrong player count")
        if str(row["method"]) != "ShaplEIG":
            errors.append(f"{task}: non-ShaplEIG row")
        mse = float(row["mse"])
        if not math.isfinite(mse) or mse < 0:
            errors.append(f"{task}: invalid MSE")
        if not str(row["data_sha256"]):
            errors.append(f"{task}: missing data hash")
    for task in TASKS:
        hashes = {
            str(row["data_sha256"])
            for row in rows
            if str(row["task_id"]) == task
        }
        if len(hashes) != 30:
            errors.append(f"{task}: expected 30 distinct game hashes, got {len(hashes)}")
    return errors

