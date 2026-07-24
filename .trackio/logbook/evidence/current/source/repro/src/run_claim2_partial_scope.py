"""Package the four exact tasks actually executed by the cumulative suite."""
from __future__ import annotations

import copy
import csv
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from claim2_partial_contract import BUDGETS, TASKS, executed_subset_errors


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / ".openresearch" / "artifacts" / "claim_2"
DV_RAW = ROOT / ".openresearch" / "artifacts" / "claim_3" / "dv10_raw.csv"
VIT_RAW = ROOT / "outputs" / "real_application" / "vit9_realized_mse.csv"
RAW = ART / "executed_subset_raw.csv"
SUMMARY = ART / "executed_subset_summary.json"


def _load_rows() -> list[dict]:
    rows: list[dict] = []
    with DV_RAW.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "ShaplEIG":
                continue
            rows.append(
                {
                    "task_id": row["task_id"],
                    "family": "data_valuation",
                    "application": row["application"],
                    "players": int(row["players"]),
                    "replicate": int(row["replicate"]),
                    "budget": int(row["budget"]),
                    "method": row["method"],
                    "mse": float(row["mse"]),
                    "data_sha256": row["data_sha256"],
                    "source_evidence": ".openresearch/artifacts/claim_3/dv10_raw.csv",
                }
            )
    with VIT_RAW.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "ShaplEIG":
                continue
            rows.append(
                {
                    "task_id": "le_vit9_imagenet",
                    "family": "local_explanation",
                    "application": row["application"],
                    "players": 9,
                    "replicate": int(row["replicate"]),
                    "budget": int(row["budget"]),
                    "method": row["method"],
                    "mse": float(row["mse"]),
                    "data_sha256": row["data_sha256"],
                    "source_evidence": "outputs/real_application/vit9_realized_mse.csv",
                }
            )
    return sorted(rows, key=lambda row: (row["task_id"], row["replicate"], row["budget"]))


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    errors = executed_subset_errors(rows)
    if errors:
        raise AssertionError(errors)

    with RAW.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["task_id"], row["budget"])].append(float(row["mse"]))
    aggregates = [
        {
            "task_id": task,
            "family": TASKS[task][0],
            "players": TASKS[task][1],
            "budget": budget,
            "replicates": len(values),
            "mean_mse": float(np.mean(values)),
            "sem_mse": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        }
        for (task, budget), values in sorted(grouped.items())
    ]
    summary = {
        "assessment": "TOY",
        "full_claim_verdict": "BLOCKED",
        "paper_scope": {
            "tasks": 15,
            "families": 4,
            "maximum_budget": 512,
        },
        "executed_scope": {
            "tasks": 4,
            "task_ids": sorted(TASKS),
            "families": 2,
            "family_ids": sorted({value[0] for value in TASKS.values()}),
            "players": sorted({value[1] for value in TASKS.values()}),
            "replicates_per_task": 30,
            "budgets": list(BUDGETS),
            "maximum_budget": max(BUDGETS),
            "methods": ["ShaplEIG"],
            "raw_rows": len(rows),
            "distinct_game_hashes": len({row["data_sha256"] for row in rows}),
        },
        "by_task_budget": aggregates,
        "source_routes": sorted({row["source_evidence"] for row in rows}),
        "environment": {
            "fixed_command": "uv sync --frozen && uv run python repro/src/reproduce.py",
            "python": platform.python_version(),
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "limitations": [
            "Four of the paper's 15 tasks are executed.",
            "Only data valuation and local explanation are represented.",
            "The maximum executed budget is 64 rather than 512.",
            "This contract establishes executed scope, not the full Figure 1 ordering.",
        ],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    mutated = [row for row in copy.deepcopy(rows) if row["task_id"] != "le_vit9_imagenet"]
    negative = {
        "mutation": "drop the complete ViT-9 task",
        "detected": bool(executed_subset_errors(mutated)),
        "errors": executed_subset_errors(mutated),
    }
    (ART / "executed_subset_negative_control.json").write_text(
        json.dumps(negative, indent=2) + "\n"
    )
    if not negative["detected"]:
        raise AssertionError("dropped-task negative control was not detected")

    subprocess.run(
        [sys.executable, "repro/src/verify_claim2_partial_scope.py", str(RAW), str(SUMMARY)],
        cwd=ROOT,
        check=True,
    )
    independent = {
        "verifier_exit": 0,
        "raw_rows": len(rows),
        "expected_rows": 600,
        "aggregate_cells": len(aggregates),
        "dropped_task_control_detected": negative["detected"],
    }
    (ART / "executed_subset_independent_checker.json").write_text(
        json.dumps(independent, indent=2) + "\n"
    )

    result_path = ART / "result.json"
    record = json.loads(result_path.read_text())
    record["partial_executed_subset"] = summary
    record["retained_performance_evidence"] = {
        "tasks": 4,
        "families": 2,
        "replicates_per_task": 30,
        "maximum_budget": 64,
        "raw_shapleig_rows": 600,
        "promoted_to_full_scale": False,
    }
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    (ART / "EVAL.md").write_text(
        """# Claim 2 evaluation

Verdict: BLOCKED

Partial assessment: TOY

The cumulative command now executes ShaplEIG on four exact public tasks:
three data-valuation tasks and ImageNet ViT-9 local explanation. The registered
matrix contains 600 complete cells (four tasks, 30 distinct games per task,
and budgets 16, 24, 32, 48, and 64). A separate parser reconstructs all 20
task/budget aggregates from the two raw evidence routes, and dropping the
entire ViT-9 task is detected.

This is materially broader than the former one-task result but cannot verify
the paper's exact 15-task scope: feature importance and hyperparameter
importance remain unavailable, and the maximum executed budget is 64 rather
than 512.
"""
    )
    print("=== CLAIM_2_PARTIAL_SCOPE ===")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

