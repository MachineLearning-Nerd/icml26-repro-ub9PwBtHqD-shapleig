"""Fixed cumulative reproduction entry point.

Every experiment node runs this file through the project-level command. Child
nodes may extend the checks in committed code, but the command and locked
environment remain unchanged.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / ".openresearch" / "artifacts" / "baseline"
OUT = ROOT / "outputs"


def run(*args: str) -> None:
    print(f"$ {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def verify_cumulative_results() -> dict:
    summary = json.loads((OUT / "summary.json").read_text())
    real = json.loads((OUT / "real_application" / "summary.json").read_text())
    stats = json.loads(
        (OUT / "real_application" / "statistical_analysis.json").read_text()
    )
    parity = json.loads(
        (OUT / "real_application" / "upstream_parity.json").read_text()
    )

    c1 = summary["claims"]["C1_closed_form_EIG"]
    if not c1["matches_mc"] or c1["posterior_cov_schur_err"] >= 1e-9:
        raise AssertionError("closed-form EIG regression failed")

    comparisons = stats["comparisons"]["ShaplEIG"]
    for baseline in (
        "KernelSHAP",
        "LeverageSHAP",
        "PermutationSampling",
        "RegressionMSR",
    ):
        result = comparisons[baseline]
        if not (
            result["geometric_mean_mse_ratio"] < 1
            and result["bootstrap_ratio_95ci"][1] < 1
            and result["wilcoxon_two_sided_p"] < 5e-5
        ):
            raise AssertionError(f"SOTA regression failed against {baseline}")

    aggregate = real["aggregate_mean_mse"]
    uncertainty_wins = [
        budget
        for budget in ("16", "24", "32", "48", "64")
        if aggregate[budget]["GP-uncertainty"] < aggregate[budget]["ShaplEIG"]
    ]
    if uncertainty_wins != ["16", "24", "32", "48"]:
        raise AssertionError(
            "retained GP-uncertainty falsification no longer reproduces: "
            f"{uncertainty_wins}"
        )
    if not parity["pass"]:
        raise AssertionError("author-code baseline parity failed")

    return {
        "claim_1_formula_regression": "PASS",
        "claim_3_one_task_regression": "PASS",
        "claim_4_falsification_regression": "PASS",
        "claim_4_gp_uncertainty_winning_budgets": uncertainty_wins,
        "upstream_parity": "PASS",
        "scope": {
            "task_families": 1,
            "tasks": 1,
            "replicates": real["replicates"],
            "maximum_budget": max(real["budgets"]),
        },
    }


def main() -> None:
    started = time.perf_counter()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")

    run(sys.executable, "repro/src/fetch_vit9_games.py")
    run(sys.executable, "-m", "pytest", "-q", "repro/tests")
    run(sys.executable, "repro/src/run_claims.py")
    run(
        sys.executable,
        "repro/src/real_application.py",
        "--seeds",
        "30",
        "--budgets",
        "16",
        "24",
        "32",
        "48",
        "64",
    )
    run(sys.executable, "repro/src/analyze_real_results.py")
    run(sys.executable, "repro/src/plot_real_results.py")
    run(sys.executable, "repro/src/verify_upstream_parity.py")
    run(sys.executable, "repro/src/run_claim4_audit.py")
    run(sys.executable, "repro/src/run_complexity_claim.py")
    run(sys.executable, "repro/src/run_claim5_audit.py")
    run(sys.executable, "repro/src/run_scope_audit.py")
    run(sys.executable, "repro/src/run_claim3_dv_audit.py")
    run(sys.executable, "repro/src/build_report_assets.py")

    result = verify_cumulative_results()
    claim_1 = json.loads(
        (ROOT / ".openresearch" / "artifacts" / "claim_1" / "result.json").read_text()
    )
    result["claim_1_complexity"] = claim_1["verdict"]
    claim_5 = json.loads(
        (ROOT / ".openresearch" / "artifacts" / "claim_5" / "result.json").read_text()
    )
    result["claim_5_overhead"] = claim_5["verdict"]
    scope_audit = json.loads(
        (ROOT / ".openresearch" / "artifacts" / "claim_2" / "result.json").read_text()
    )
    result["claim_2_scope"] = scope_audit["claim_2_verdict"]
    claim_3 = json.loads(
        (ROOT / ".openresearch" / "artifacts" / "claim_3" / "result.json").read_text()
    )
    result["claim_3_full_scale"] = claim_3["verdict"]
    claim_4 = json.loads(
        (ROOT / ".openresearch" / "artifacts" / "claim_4" / "result.json").read_text()
    )
    result["claim_4_ablation"] = claim_4["verdict"]
    if claim_4["verdict"] != "FALSIFIED":
        raise AssertionError("Claim 4 falsification package regressed")
    result.update(
        {
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "elapsed_seconds": time.perf_counter() - started,
            "fixed_command": "uv sync --frozen && uv run python repro/src/reproduce.py",
        }
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    eval_lines = [
        "# Baseline cumulative evaluation",
        "",
        "Verdict: PASS (judged-state regression only; not full-scale evidence)",
        "",
        f"- Git SHA: `{result['git_sha']}`",
        f"- Python: `{result['python']}`",
        f"- Runtime: `{result['elapsed_seconds']:.3f}` seconds",
        "- Scope: one task family, one task, 30 repetitions, maximum budget 64.",
        "- Claim 4 retained falsification: GP-uncertainty wins at budgets "
        + ", ".join(map(str, result["claim_4_gp_uncertainty_winning_budgets"]))
        + ".",
        "",
    ]
    (ARTIFACTS / "EVAL.md").write_text("\n".join(eval_lines))
    print("=== OPENRESEARCH_EVAL ===")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
