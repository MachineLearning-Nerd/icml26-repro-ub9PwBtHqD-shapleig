"""Source-matched decomposed timings for the three exact large tree games."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from shapiq.approximator.sampling import CoalitionSampler

from claim5_large_timing_contract import (
    CANDIDATE_POOL,
    TASKS,
    archive_sizes,
    large_timing_errors,
)
from complexity import efficient_eig_batch
from real_application import fit_gp
from run_scope_audit import build_tree_game


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / ".openresearch" / "artifacts" / "claim_5"
RAW = ART / "large_timing_raw.csv"
SUMMARY = ART / "large_timing_summary.json"
UPSTREAM = "162ce44fe380c7c11b959fc85206b5dcdeddff58"


def _source_design(p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Port the author's scalable ShapleyApplication candidate construction."""
    sampler = CoalitionSampler(
        n_players=p,
        sampling_weights=np.ones(p + 1),
        pairing_trick=True,
        random_state=seed,
    )
    initial_sampler = deepcopy(sampler)
    initial_sampler.sample(p + 1)
    initial = initial_sampler.coalitions_matrix.astype(np.int8)
    sampler.sample(p + 1 + CANDIDATE_POOL)
    sampled = sampler.coalitions_matrix.astype(np.int8)
    already_initial = (
        (sampled[:, None, :] == initial[None, :, :]).all(axis=2).any(axis=1)
    )
    candidates = sampled[~already_initial][:CANDIDATE_POOL]
    if initial.shape != (p + 1, p):
        raise AssertionError(f"initial design shape {initial.shape}, expected {(p + 1, p)}")
    if candidates.shape != (CANDIDATE_POOL, p):
        raise AssertionError(
            f"candidate shape {candidates.shape}, expected {(CANDIDATE_POOL, p)}"
        )
    return initial, candidates


def _time_task(task_id: str, dataset_name: str, p: int, replicate: int) -> list[dict]:
    game, _, depth = build_tree_game(dataset_name, p, replicate)
    initial, candidate_pool = _source_design(p, replicate)
    full_archive = np.concatenate(
        [initial, candidate_pool[: 512 - len(initial)]], axis=0
    )
    full_values = np.asarray(
        game.value_function(full_archive.astype(bool)), dtype=float
    ).reshape(-1)
    if full_values.shape != (512,) or not np.all(np.isfinite(full_values)):
        raise AssertionError(f"{task_id}: invalid 512-coalition game evaluation")
    rows = []
    prior_mode = math.exp(math.sqrt(2) + 0.5 * math.log(p) - 3)
    for archive in archive_sizes(p):
        observed = full_archive[:archive]
        values = full_values[:archive]
        selected = archive - len(initial)
        candidates = candidate_pool[selected:]
        start_ell = np.full(p, prior_mode)

        started = time.perf_counter()
        ell, outputscale, _ = fit_gp(
            observed,
            values,
            start_ell,
            ard=True,
            seed=replicate * 1000 + archive,
            restarts=5,
        )
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        eig = efficient_eig_batch(
            observed,
            candidates,
            ell,
            outputscale=outputscale,
            noise=1e-6,
        )
        eig_seconds = time.perf_counter() - started
        best = int(np.argmax(eig.scores))
        rows.append(
            {
                "task_id": task_id,
                "dataset": dataset_name,
                "replicate": replicate,
                "players": p,
                "archive_size": archive,
                "source_candidate_pool": CANDIDATE_POOL,
                "candidate_count": len(candidates),
                "fit_restarts": 5,
                "hp_fit_seconds": fit_seconds,
                "eig_seconds": eig_seconds,
                "maximum_eig": float(eig.scores[best]),
                "selected_candidate_index": best,
                "rf_estimators": 10,
                "rf_depth": depth,
                "candidate_sha256": hashlib.sha256(candidates.tobytes()).hexdigest(),
                "aka_operation_bound": eig.aka_operations,
                "eig_workspace_bytes": eig.efficient_peak_bytes,
            }
        )
        print(
            f"claim5 large timing {task_id} seed={replicate} t={archive} "
            f"candidates={len(candidates)} fit={fit_seconds:.6f}s "
            f"eig={eig_seconds:.6f}s",
            flush=True,
        )
    return rows


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []
    for task_id, (dataset_name, p) in TASKS.items():
        rows.extend(_time_task(task_id, dataset_name, p, replicate=1))
    errors = large_timing_errors(rows)
    if errors:
        raise AssertionError(errors)
    with RAW.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "verdict": "BLOCKED",
        "partial_assessment": "SOURCE_MATCHED_LARGE_SUBSET",
        "tasks": 3,
        "paper_tasks": 15,
        "families": ["local_explanation"],
        "paper_families": 4,
        "players": [60, 79, 101],
        "maximum_players": 101,
        "archive_sizes": {task: list(archive_sizes(p)) for task, (_, p) in TASKS.items()},
        "maximum_archive_size": 512,
        "source_candidate_pool": CANDIDATE_POOL,
        "candidate_policy": (
            "author scalable-mode uniform-cardinality PolySHAP sampler with "
            "pairing trick; fixed 1024-coalition subset"
        ),
        "fit_policy": "five-restart cold-start ARD Hamming-GP MAP fit",
        "source": f"slds-lmu/shapleig@{UPSTREAM}",
        "source_configurations": [
            "shapleig_crv_tree_corrgroup.yaml",
            "shapleig_crv_tree_nhanesi.yaml",
            "shapleig_crv_tree_crime.yaml",
        ],
        "raw_rows": len(rows),
        "observed_maxima": {
            "hp_fit_seconds": max(float(row["hp_fit_seconds"]) for row in rows),
            "eig_seconds": max(float(row["eig_seconds"]) for row in rows),
        },
        "threshold_checks": {
            "all_eig_below_30_seconds": all(
                float(row["eig_seconds"]) < 30 for row in rows
            ),
            "all_fit_below_1500_seconds": all(
                float(row["hp_fit_seconds"]) < 1500 for row in rows
            ),
        },
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
            "This directly covers the three paper tasks above 16 players, not all 15 tasks.",
            "One deterministic paper seed is measured; the paper averages 30 repetitions with SEM.",
            "The NumPy/SciPy port is equation- and configuration-matched, but it is not the author Torch/BoTorch runtime.",
            "The t=512 archive uses the deterministic prefix of the author-sampled candidate pool, not a full 451/432/410-step EIG-selected trajectory.",
        ],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    mutated = copy.deepcopy(rows)
    mutated.pop()
    negative = {
        "mutation": "delete the p=101, t=512 timing cell",
        "detected": bool(large_timing_errors(mutated)),
        "errors": large_timing_errors(mutated),
    }
    (ART / "large_timing_negative_control.json").write_text(
        json.dumps(negative, indent=2) + "\n"
    )
    if not negative["detected"]:
        raise AssertionError("large-timing negative control was not detected")
    subprocess.run(
        [
            sys.executable,
            "repro/src/verify_claim5_large_timing.py",
            str(RAW),
            str(SUMMARY),
        ],
        cwd=ROOT,
        check=True,
    )
    checker = {
        "verifier_exit": 0,
        "raw_rows": len(rows),
        "maximum_players": 101,
        "maximum_archive_size": 512,
        "negative_control_detected": True,
    }
    (ART / "large_timing_independent_checker.json").write_text(
        json.dumps(checker, indent=2) + "\n"
    )

    record_path = ART / "result.json"
    record = json.loads(record_path.read_text())
    record["verdict"] = "BLOCKED"
    record["large_game_timing_evidence"] = summary
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    (ART / "EVAL.md").write_text(
        f"""# Claim 5 evaluation

Verdict: BLOCKED

The judge's missing large-game regime is now directly measured on the paper's
three exact tree tasks: p=60, 79, and 101, at the source initial archive and
budget 512. The candidate construction matches the pinned author scalable
configuration (1,024 PolySHAP-sampled coalitions with pairing), and GP fitting
and vectorized EIG are timed separately. The largest observed GP fit was
`{summary['observed_maxima']['hp_fit_seconds']:.6f}` seconds and the largest
EIG call was `{summary['observed_maxima']['eig_seconds']:.6f}` seconds.

This remains BLOCKED for the exact full claim because it covers three of 15
tasks, one repetition rather than the paper's repetition means and SEM, and an
independent NumPy/SciPy implementation rather than the author's Torch/BoTorch
runtime. It is faithful large-regime evidence, not full verification.
"""
    )
    print("=== CLAIM_5_LARGE_TIMING ===")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
