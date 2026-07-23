"""Decomposed timings on an exact, explicitly partial Claim 5 subset."""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from claim5_timing_contract import TASKS, archive_sizes, timing_errors
from fetch_dv10_games import fetch_all
from fetch_vit9_games import fetch as fetch_vit9
from real_application import (
    DATA as VIT9_DATA,
    fit_gp,
    gp_state,
    grid,
    leverage_sample,
    load_game,
)
from shapleig import shapley_matrix


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / ".openresearch" / "artifacts" / "claim_5"
DV_DATA = ROOT / "data" / "claim3_dv10"
RAW = ART / "partial_timing_raw.csv"
SUMMARY = ART / "partial_timing_summary.json"
UPSTREAM = "162ce44fe380c7c11b959fc85206b5dcdeddff58"


def _sem(values: list[float]) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _observed(p: int, archive: int, seed: int) -> list[int]:
    selected = list(dict.fromkeys(leverage_sample(p, archive, seed).tolist()))
    if len(selected) < archive:
        present = set(selected)
        rng = np.random.default_rng(seed + 100_000 * p + archive)
        selected.extend(
            int(value)
            for value in rng.permutation(1 << p)
            if int(value) not in present
        )
    return selected[:archive]


def _task_paths() -> dict[str, list[Path]]:
    vit9 = fetch_vit9(VIT9_DATA)[:5]
    manifest = fetch_all(DV_DATA)
    paths = {"le_vit9_imagenet": [Path(path) for path in vit9]}
    for task in TASKS:
        if task == "le_vit9_imagenet":
            continue
        paths[task] = [
            Path(record["local_path"])
            for record in manifest["tasks"][task][:5]
        ]
    return paths


def _time_cell(task: str, path: Path, replicate: int, archive: int) -> dict:
    p = TASKS[task]
    X, values = load_game(path, expected_p=p)
    A = shapley_matrix(p)
    observed = _observed(p, archive, replicate)
    prior_mode = math.exp(math.sqrt(2) + 0.5 * math.log(p) - 3)
    start_ell = np.full(p, prior_mode)

    started = time.perf_counter()
    ell, outputscale, fitted_mean = fit_gp(
        X[observed],
        values[observed],
        start_ell,
        ard=True,
        seed=replicate * 1000 + archive,
        restarts=5,
    )
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    _, Q, C, variance = gp_state(
        X,
        values,
        observed,
        ell,
        A,
        outputscale=outputscale,
        fitted_mean=fitted_mean,
    )
    rho = np.sum(Q * np.linalg.solve(C, Q), axis=0) / variance
    scores = -0.5 * np.log1p(-np.clip(rho, 0, 1 - 1e-12))
    scores[np.asarray(observed, dtype=int)] = -np.inf
    best = int(np.argmax(scores))
    eig_seconds = time.perf_counter() - started

    return {
        "task_id": task,
        "replicate": replicate,
        "players": p,
        "archive_size": archive,
        "candidate_count": (1 << p) - archive,
        "fit_restarts": 5,
        "hp_fit_seconds": fit_seconds,
        "eig_seconds": eig_seconds,
        "selected_candidate": best,
        "maximum_eig": float(scores[best]),
        "source_file": path.name,
    }


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    task_paths = _task_paths()
    rows: list[dict] = []
    started = time.perf_counter()
    for task, p in TASKS.items():
        for replicate, path in enumerate(task_paths[task], 1):
            for archive in archive_sizes(p):
                row = _time_cell(task, path, replicate, archive)
                rows.append(row)
                print(
                    f"claim5 timing {task} rep={replicate}/5 "
                    f"t={archive} fit={row['hp_fit_seconds']:.6f}s "
                    f"eig={row['eig_seconds']:.6f}s",
                    flush=True,
                )
    errors = timing_errors(rows)
    if errors:
        raise AssertionError(errors)

    with RAW.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["task_id"], row["archive_size"])].append(row)
    aggregates = []
    for (task, archive), group in sorted(groups.items()):
        fit_values = [float(row["hp_fit_seconds"]) for row in group]
        eig_values = [float(row["eig_seconds"]) for row in group]
        aggregates.append(
            {
                "task_id": task,
                "players": TASKS[task],
                "archive_size": archive,
                "replicates": len(group),
                "candidate_count": int(group[0]["candidate_count"]),
                "hp_fit_mean_seconds": float(np.mean(fit_values)),
                "hp_fit_sem_seconds": _sem(fit_values),
                "hp_fit_max_seconds": max(fit_values),
                "eig_mean_seconds": float(np.mean(eig_values)),
                "eig_sem_seconds": _sem(eig_values),
                "eig_max_seconds": max(eig_values),
            }
        )
    summary = {
        "assessment": "TOY",
        "full_claim_verdict": "BLOCKED",
        "tasks": 4,
        "paper_tasks": 15,
        "families": ["data_valuation", "local_explanation"],
        "paper_families": 4,
        "players": [9, 10],
        "paper_player_range": [8, 101],
        "maximum_archive_size": 64,
        "paper_maximum_budget": 512,
        "candidate_policy": "complete unevaluated coalition grid",
        "maximum_candidate_pool": 1024,
        "implementation": (
            "independent NumPy/SciPy port of the pinned author Hamming-GP, "
            "five-restart MAP fit, and vectorized Shapley-EIG equations"
        ),
        "source": f"slds-lmu/shapleig@{UPSTREAM}",
        "by_task_archive": aggregates,
        "observed_maxima": {
            "hp_fit_seconds": max(float(row["hp_fit_seconds"]) for row in rows),
            "eig_seconds": max(float(row["eig_seconds"]) for row in rows),
        },
        "paper_small_game_bounds_seconds": {"hp_fit": 120, "eig": 1},
        "all_observed_eig_below_one_second": all(
            float(row["eig_seconds"]) < 1 for row in rows
        ),
        "all_observed_fit_below_120_seconds": all(
            float(row["hp_fit_seconds"]) < 120 for row in rows
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "fixed_command": "uv sync --frozen && uv run python repro/src/reproduce.py",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "limitations": [
            "Four exact tasks are timed, not all 15.",
            "The measured p=9-10 range does not test the paper's p>16 timing claim.",
            "The frozen NumPy/SciPy port is mathematically source-matched but is not the author Torch/BoTorch runtime.",
            "Archive designs are deterministic matched leverage samples, not complete sequential trajectories.",
        ],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    mutated = copy.deepcopy(rows)
    mutated.pop()
    negative = {
        "mutation": "delete one task/replicate/archive timing cell",
        "detected": bool(timing_errors(mutated)),
        "errors": timing_errors(mutated),
    }
    (ART / "timing_negative_control.json").write_text(
        json.dumps(negative, indent=2) + "\n"
    )
    if not negative["detected"]:
        raise AssertionError("timing missing-cell negative control was not detected")

    subprocess.run(
        [sys.executable, "repro/src/verify_claim5_timing.py", str(RAW), str(SUMMARY)],
        cwd=ROOT,
        check=True,
    )
    independent = {
        "verifier_exit": 0,
        "raw_rows": len(rows),
        "expected_rows": 80,
        "aggregate_cells": len(aggregates),
        "negative_control_detected": negative["detected"],
    }
    (ART / "timing_independent_checker.json").write_text(
        json.dumps(independent, indent=2) + "\n"
    )

    record_path = ART / "result.json"
    record = json.loads(record_path.read_text())
    record["partial_timing_evidence"] = summary
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    (ART / "EVAL.md").write_text(
        f"""# Claim 5 evaluation

Verdict: BLOCKED

Partial assessment: TOY

The exact public subset now contains 80 decomposed timing cells: four tasks,
five games per task, and four archive sizes per game. Every measurement times
the five-restart Hamming-GP MAP fit separately from a complete-candidate
vectorized EIG call. The largest observed fit was
`{summary['observed_maxima']['hp_fit_seconds']:.6f}` seconds and the largest EIG
call was `{summary['observed_maxima']['eig_seconds']:.6f}` seconds on this
8-logical-CPU Apple-arm64 host. Both align with the paper's small-game upper
bounds on this subset.

This is partial rather than full verification: it covers p=9-10, four of 15
tasks, archives through 64 rather than 512, and an independent NumPy/SciPy port
rather than the authors' Torch/BoTorch runtime. It does not test the reported
~25-minute refits for p>16. A separate parser rebuilds all aggregates from the
raw CSV, and deleting one timing cell is detected.
"""
    )
    print("=== CLAIM_5_PARTIAL_TIMING ===")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

