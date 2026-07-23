"""Exact three-task data-valuation counterexample search for Claim 3."""
from __future__ import annotations

import copy
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from claim3_dv_contract import BUDGETS, METHODS, evidence_errors, paired_tests
from fetch_dv10_games import COMMIT, TASKS, fetch_all
from real_application import (
    baseline_estimates,
    gp_run,
    leverage_estimate,
    load_game,
)
from shapleig import shapley_matrix


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "dv10"
ART = ROOT / ".openresearch" / "artifacts" / "claim_3"
RAW = ART / "dv10_raw.csv"


def main() -> None:
    started = time.perf_counter()
    ART.mkdir(parents=True, exist_ok=True)
    manifest = fetch_all(DATA)
    rows = []
    selected_methods = METHODS - {"ShaplEIG"}
    for task_id, task_spec in TASKS.items():
        records = manifest["tasks"][task_id]
        for replicate, record in enumerate(records, 1):
            _, values = load_game(Path(record["local_path"]), expected_p=10)
            truth = shapley_matrix(10) @ values
            shapleig = gp_run(
                values,
                "eig",
                BUDGETS,
                replicate,
                adaptive=True,
                ard=True,
                refit_interval=1,
            )
            for budget in BUDGETS:
                estimates = {
                    name: estimate
                    for name, estimate in baseline_estimates(
                        values, budget, replicate
                    ).items()
                    if name in selected_methods
                }
                estimates["LeverageSHAP"] = leverage_estimate(
                    values, budget, replicate
                )
                estimates["ShaplEIG"] = shapleig[budget]
                if set(estimates) != METHODS:
                    raise AssertionError(f"method mismatch: {set(estimates)}")
                for method, estimate in estimates.items():
                    rows.append(
                        {
                            "task_id": task_id,
                            "application": task_spec["application"],
                            "players": 10,
                            "replicate": replicate,
                            "seed": replicate,
                            "budget": budget,
                            "method": method,
                            "mse": float(np.mean((estimate - truth) ** 2)),
                            "efficiency_error": float(
                                abs(estimate.sum() - (values[-1] - values[0]))
                            ),
                            "data_sha256": record["sha256"],
                            "source_path": record["source_path"],
                        }
                    )
            print(f"claim3 {task_id} replicate {replicate}/30 complete", flush=True)

    with RAW.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    errors = evidence_errors(copy.deepcopy(rows))
    if errors:
        raise AssertionError(errors)
    tests = paired_tests(copy.deepcopy(rows))
    counterexamples = [
        {key: value for key, value in test.items()}
        for test in tests
        if test["counterexample"]
    ]
    verdict = "FALSIFIED" if counterexamples else "BLOCKED"

    # A missing matched observation must be rejected by the structural checker.
    mutated = rows[:-1]
    negative = {
        "mutation": "delete one exact task/seed/budget/method observation",
        "detected": bool(evidence_errors(mutated)),
        "errors": evidence_errors(mutated),
    }
    if not negative["detected"]:
        raise AssertionError("negative control was not detected")

    result = {
        "verdict": verdict,
        "paper_claim": "ShaplEIG has lower MSE than the four named baselines across the 15 Figure 1 tasks and varying budgets.",
        "audit_scope": {
            "exact_tasks": list(TASKS),
            "families": ["data_valuation"],
            "players": 10,
            "replicates_per_task": 30,
            "matched_seeds": list(range(1, 31)),
            "budgets": BUDGETS,
            "methods": sorted(METHODS),
            "candidate_optimizer": "exhaustive",
            "initial_design": "LeverageSHAP, p+1",
            "gp_refit_interval": 1,
            "map_restarts": 5,
            "source_commit": COMMIT,
        },
        "multiplicity": {
            "registered_comparisons": len(tests),
            "correction": "Holm family-wise correction over all 60 task/budget/baseline comparisons",
            "counterexample_rule": "mean MSE and geometric-mean paired ratio >1, bootstrap ratio 95% CI lower bound >1, Holm-adjusted one-sided Wilcoxon p<0.05",
        },
        "counterexamples": counterexamples,
        "all_tests": tests,
        "negative_control": negative,
        "environment": {
            "fixed_command": "uv sync --frozen && uv run python repro/src/reproduce.py",
            "python": platform.python_version(),
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "logical_cpus": __import__("os").cpu_count(),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "This registered search covers three exact tasks, not the full 15-task matrix.",
            "The frozen Python 3.12 NumPy/SciPy reimplementation is an explicit environment deviation from the authors' Python 3.11 Torch/GPyTorch stack.",
            "A FALSIFIED verdict is issued only if an exact-task counterexample survives matched uncertainty and family-wise multiplicity correction.",
        ],
    }
    durable_manifest = copy.deepcopy(manifest)
    for records in durable_manifest["tasks"].values():
        for record in records:
            record.pop("local_path", None)
    (ART / "dv10_source_manifest.json").write_text(
        json.dumps(durable_manifest, indent=2) + "\n"
    )
    (ART / "dv10_statistics.json").write_text(json.dumps(tests, indent=2) + "\n")
    (ART / "negative_control.json").write_text(json.dumps(negative, indent=2) + "\n")
    (ART / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    subprocess.run(
        [sys.executable, "repro/src/verify_claim3_dv.py", str(RAW), str(ART / "result.json")],
        cwd=ROOT,
        check=True,
    )
    (ART / "claim_contract.json").write_text(
        json.dumps(
            {
                "claim_id": 3,
                "verified": "all 15 exact tasks, paper budgets, repetitions, and named baselines favor ShaplEIG under matched realized-MSE uncertainty",
                "falsified": "at least one exact paper task/configuration yields a named-baseline win surviving the pre-registered paired bootstrap, Wilcoxon, and Holm criteria",
                "blocked": "neither the complete 15-task matrix nor a robust exact-task counterexample is available",
                "allowed_verdicts": ["VERIFIED", "FALSIFIED", "BLOCKED"],
            },
            indent=2,
        )
        + "\n"
    )
    (ART / "method.md").write_text(
        """# Claim 3 method

This audit runs all 30 pinned public games for each of the three exact
10-player data-valuation tasks. It uses the author configuration's leverage
initial design (`p+1`), weighted-Hamming GP, five MAP restarts, refitting every
iteration, exhaustive candidate search, seeds 1–30, and the four named
baselines. Errors are realized MSE against exhaustive 1024-coalition Shapley
values. The registered early-budget grid is 16, 24, 32, 48, and 64.

For every task/budget/baseline cell, inference is paired over the 30 games.
A counterexample requires arithmetic and geometric mean MSE to favor the
baseline, a 20,000-resample paired bootstrap ratio interval wholly above one,
and a one-sided paired Wilcoxon test surviving Holm correction over all 60
registered comparisons.
"""
    )
    (ART / "source_audit.md").write_text(
        """# Claim 3 source audit

- Paper: arXiv:2606.02247v1, Figure 1 and Section 5.2.
- Public data: `mmschlk/shapiq` commit
  `799cfd0f2c32f17446130247a7ac3519e68cce82`.
- Author code: `slds-lmu/shapleig` commit
  `162ce44fe380c7c11b959fc85206b5dcdeddff58`.
- Exact configuration:
  `src/xac/experiments/conf/shapleig_crv_shapiq_dv_10p.yaml`.
- Quantifiers tested here: three named data-valuation tasks, 30 repetitions,
  exact 10-player games, four named baselines, and five early budgets.
"""
    )
    (ART / "EVAL.md").write_text(
        f"""# Claim 3 evaluation

Verdict: {verdict}

The exact three-task data-valuation search produced {len(counterexamples)}
counterexample cells under the registered family-wise statistical rule.
See `dv10_raw.csv`, `dv10_statistics.json`, `independent_checker.json`, and
`negative_control.json`. This does not claim completion of the 15-task matrix.
"""
    )
    print("=== CLAIM_3_DV_AUDIT ===")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
