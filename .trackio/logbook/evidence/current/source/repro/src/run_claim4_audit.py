"""Package and verify the retained Figure 3 ablation falsification."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from claim4_contract import EXPECTED_BUDGETS, EXPECTED_METHODS, core_claim4_errors
from verify_claim4_ablation import recompute, verify


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs" / "real_application" / "vit9_realized_mse.csv"
ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_4"
RAW = ARTIFACT / "ablation_raw.csv"


def source_rows() -> list[dict[str, str]]:
    with SOURCE.open() as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["method"] in EXPECTED_METHODS
        ]
    rows.sort(key=lambda row: (
        int(row["replicate"]),
        int(row["budget"]),
        row["method"],
    ))
    return rows


def main() -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    rows = source_rows()
    with RAW.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    observed = recompute(RAW)
    record = {
        "claim": (
            "EIG acquisition outperforms GP+Random, GP+Leverage Score "
            "Sampling, and GP+Uncertainty Sampling"
        ),
        "verdict": "FALSIFIED",
        "reason": (
            "On the paper's official 30 ViT-9 games, GP+Uncertainty has the "
            "better mean rank and lower arithmetic-mean MSE at four of five "
            "reported budgets; ShaplEIG wins only at budget 64."
        ),
        "evidence": {
            "task": "ImageNet ViT-9 local explanation",
            "replicates": 30,
            "budgets": EXPECTED_BUDGETS,
            "methods": EXPECTED_METHODS,
            **observed,
        },
        "source": {
            "paper_url": "https://ar5iv.labs.arxiv.org/html/2606.02247#A4.F3",
            "paper_html_sha256": (
                "81809d39fa180bab590aa3eb2c3a0c37ac3d17df20c91ac47dc4f805c981dff1"
            ),
            "raw_source": "outputs/real_application/vit9_realized_mse.csv",
            "raw_artifact_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        },
        "environment": {
            "fixed_command": "uv sync --frozen && uv run python repro/src/reproduce.py",
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "deterministic_seeds": list(range(1, 31)),
        },
        "negative_control": {
            "mutation": "swap ShaplEIG and GP-uncertainty method labels in raw rows",
            "detected": False,
        },
    }
    mutated_rows = copy.deepcopy(rows)
    for row in mutated_rows:
        if row["method"] == "ShaplEIG":
            row["method"] = "GP-uncertainty"
        elif row["method"] == "GP-uncertainty":
            row["method"] = "ShaplEIG"
    with tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mutated_rows[0]))
        writer.writeheader()
        writer.writerows(mutated_rows)
        handle.flush()
        record["negative_control"]["detected"] = bool(
            verify(record, Path(handle.name))
        )
    contract_errors = core_claim4_errors(record)
    if contract_errors:
        raise AssertionError(contract_errors)
    errors = verify(record, RAW)
    if errors:
        raise AssertionError(errors)

    (ARTIFACT / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    checker = {
        "implementation": "independent stdlib CSV raw-grid recomputation",
        "errors": verify(record, RAW),
        "raw_sha256": record["source"]["raw_artifact_sha256"],
        "raw_rows": observed["raw_rows"],
    }
    (ARTIFACT / "independent_checker.json").write_text(
        json.dumps(checker, indent=2) + "\n"
    )
    (ARTIFACT / "negative_control.json").write_text(
        json.dumps(record["negative_control"], indent=2) + "\n"
    )
    (ARTIFACT / "EVAL.md").write_text(
        """# Claim 4 evaluation

Verdict: FALSIFIED

The exact official ViT-9 ablation grid contains 30 games, five budgets, and
the four Figure 3 GP variants. GP+Uncertainty has a better within-ablation
mean rank than ShaplEIG and lower arithmetic-mean MSE at budgets 16, 24, 32,
and 48. ShaplEIG is lower only at budget 64. A separate CSV parser recomputes
the full grid, aggregates, and ranks. Swapping the two method labels makes
the evidence inconsistent and is rejected.

This is a direct counterexample on one of the paper's own Figure 3 tasks. It
does not assert that GP+Uncertainty dominates on every task or metric.
"""
    )
    subprocess.run(
        [
            sys.executable,
            "repro/src/verify_claim4_ablation.py",
            str(ARTIFACT / "result.json"),
            str(RAW),
        ],
        cwd=ROOT,
        check=True,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
