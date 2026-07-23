"""Independent structural and statistical checks for the Claim 3 DV audit."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


TASKS = {
    "dv_rf_bike_sharing",
    "dv_gb_bike_sharing",
    "dv_gb_california_housing",
}
METHODS = {
    "ShaplEIG",
    "KernelSHAP",
    "LeverageSHAP",
    "PermutationSampling",
    "RegressionMSR",
}
BUDGETS = [16, 24, 32, 48, 64]


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["replicate"] = int(row["replicate"])
        row["budget"] = int(row["budget"])
        row["mse"] = float(row["mse"])
    return rows


def evidence_errors(rows: list[dict]) -> list[str]:
    errors = []
    expected = len(TASKS) * 30 * len(BUDGETS) * len(METHODS)
    if len(rows) != expected:
        errors.append(f"expected {expected} rows, got {len(rows)}")
    keys = [
        (row["task_id"], row["replicate"], row["budget"], row["method"])
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        errors.append("duplicate task/replicate/budget/method cells")
    if {row["task_id"] for row in rows} != TASKS:
        errors.append("exact three-task set is incomplete")
    if {row["method"] for row in rows} != METHODS:
        errors.append("named method set is incomplete")
    if sorted({row["budget"] for row in rows}) != BUDGETS:
        errors.append("budget grid differs from the registered early-budget audit")
    for task_id in TASKS:
        reps = {row["replicate"] for row in rows if row["task_id"] == task_id}
        if reps != set(range(1, 31)):
            errors.append(f"{task_id}: matched seeds 1..30 are incomplete")
    if any(not math.isfinite(row["mse"]) or row["mse"] < 0 for row in rows):
        errors.append("MSE must be finite and nonnegative")
    return errors


def paired_tests(rows: list[dict]) -> list[dict]:
    cells = {
        (row["task_id"], row["replicate"], row["budget"], row["method"]): row["mse"]
        for row in rows
    }
    tests = []
    for task_id in sorted(TASKS):
        for budget in BUDGETS:
            shapleig = np.array(
                [cells[task_id, rep, budget, "ShaplEIG"] for rep in range(1, 31)]
            )
            for baseline in sorted(METHODS - {"ShaplEIG"}):
                other = np.array(
                    [cells[task_id, rep, budget, baseline] for rep in range(1, 31)]
                )
                eps = np.finfo(float).tiny
                log_ratio = np.log(np.maximum(shapleig, eps)) - np.log(
                    np.maximum(other, eps)
                )
                p = float(
                    wilcoxon(
                        log_ratio,
                        alternative="greater",
                        zero_method="wilcox",
                    ).pvalue
                )
                tests.append(
                    {
                        "task_id": task_id,
                        "budget": budget,
                        "baseline": baseline,
                        "shapleig_mean_mse": float(np.mean(shapleig)),
                        "baseline_mean_mse": float(np.mean(other)),
                        "geometric_mean_ratio_shapleig_over_baseline": float(
                            np.exp(np.mean(log_ratio))
                        ),
                        "one_sided_wilcoxon_p": p,
                        "_log_ratio": log_ratio,
                    }
                )
    # Holm correction over the complete registered 3 x 5 x 4 search.
    order = sorted(range(len(tests)), key=lambda index: tests[index]["one_sided_wilcoxon_p"])
    adjusted = np.ones(len(tests))
    running = 0.0
    total = len(tests)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * tests[index]["one_sided_wilcoxon_p"])
        running = max(running, candidate)
        adjusted[index] = running
    for index, test in enumerate(tests):
        # Independent deterministic bootstrap of matched replicate log-ratios.
        rng = np.random.default_rng(260602247 + index)
        log_ratio = test.pop("_log_ratio")
        indices = rng.integers(0, len(log_ratio), size=(20000, len(log_ratio)))
        ratios = np.exp(np.mean(log_ratio[indices], axis=1))
        test["bootstrap_ratio_95ci"] = [
            float(np.quantile(ratios, 0.025)),
            float(np.quantile(ratios, 0.975)),
        ]
        test["holm_adjusted_p"] = float(adjusted[index])
        test["counterexample"] = bool(
            test["shapleig_mean_mse"] > test["baseline_mean_mse"]
            and test["geometric_mean_ratio_shapleig_over_baseline"] > 1
            and test["bootstrap_ratio_95ci"][0] > 1
            and test["holm_adjusted_p"] < 0.05
        )
    return tests

