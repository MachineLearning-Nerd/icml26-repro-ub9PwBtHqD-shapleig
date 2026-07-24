"""Audit whether the paper's exact per-iteration timing experiment can run."""
from __future__ import annotations

import csv
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

from claim5_audit import REQUIRED_AUTHOR_MODULES, blocking_reasons


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / ".openresearch" / "artifacts" / "claim_5"

AUTHOR_VERSIONS = {
    "torch": "2.9.1",
    "botorch": "0.14.0",
    "gpytorch": "1.14",
    "hydra-core": "1.3.2",
    "yahpo-gym": "1.0.2",
    "tabpfn": "6.2.0",
    "shapiq": "1.4.1",
    "numpy": "1.26.4",
    "scipy": "1.16.3",
}
AUTHOR_SOURCE = {
    "git_sha": "162ce44fe380c7c11b959fc85206b5dcdeddff58",
    "pyproject_sha256": "3a840a9dc6e876f7d442a56d65b385c43971279334f9473619ba18f425faf878",
    "poetry_lock_sha256": "faa76605a00528255840634fac74caa1cb975c79546705158d7d4a627faed2c0",
    "experiment_runner_sha256": "ac7da7040f54be7c5973e9f44e04dee8dd91d1611c9073d36e6ee452c2659108",
    "acquisition_functions_sha256": "42039d0dbe871e9f9a9b6e582fc53c96b32ff3999f615d86d2a84e9ed0b2c219",
    "gp_surrogate_sha256": "147748a61e4672956e344ac23a8304fa2e3fcb17bb0afbb6146378844f887d41",
}
TASK_PLAYERS = [10, 8, 8, 10, 10, 10, 16, 16, 8, 60, 79, 101, 14, 9, 16]


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    available = [
        name for name in REQUIRED_AUTHOR_MODULES if importlib.util.find_spec(name)
    ]
    missing = [name for name in REQUIRED_AUTHOR_MODULES if name not in available]
    required_paths = {
        "yahpo_metadata": "data/yahpo_surrogates/yahpo_data",
        "tabpfn_tid15": "data/shapiq_games/tabpfn/tid15",
        "tabpfn_tid37": "data/shapiq_games/tabpfn/tid37",
        "tabpfn_diabetes_regression": "data/shapiq_games/tabpfn/tiddiabreg",
    }
    path_status = {key: (ROOT / value).exists() for key, value in required_paths.items()}
    with (
        ROOT / ".openresearch" / "artifacts" / "claim_1" / "scaling.csv"
    ).open(newline="") as handle:
        scaling = list(csv.DictReader(handle))
    p101 = next(
        row
        for row in scaling
        if row["route"] == "efficient_esp" and row["p"] == "101"
    )
    audit = {
        "python_contract": {
            "author": ">=3.11,<3.12",
            "frozen_reproduction": "==3.12.11",
            "compatible": False,
        },
        "module_audit": {
            "required": list(REQUIRED_AUTHOR_MODULES),
            "available": available,
            "missing": missing,
        },
        "author_locked_versions": AUTHOR_VERSIONS,
        "required_external_data": {
            "paths": required_paths,
            "present": path_status,
            "all_present": all(path_status.values()),
        },
    }
    negative_audit = json.loads(json.dumps(audit))
    negative_audit["python_contract"]["compatible"] = True
    negative_audit["module_audit"]["missing"] = []
    negative_audit["required_external_data"]["all_present"] = True
    record = {
        "claim": "Figures 5-6 per-iteration computational overhead",
        "verdict": "BLOCKED",
        "source": AUTHOR_SOURCE,
        "paper_scope": {
            "task_count": 15,
            "task_players_in_table_order": TASK_PLAYERS,
            "maximum_players": max(TASK_PLAYERS),
            "maximum_budget": 512,
            "reported_small_game_hp_fit_upper_seconds": 120,
            "reported_large_game_hp_fit_upper_seconds": 1500,
            "reported_small_game_eig_upper_seconds": 1,
            "reported_large_game_eig_upper_seconds": 30,
            "timed_blocks": ["gp.fit()", "EIGFunctionProperty(candidate_set, ...)"],
        },
        "prerequisite_audit": audit,
        "blocking_reasons": blocking_reasons(audit),
        "non_equivalent_diagnostic": {
            "description": "single-candidate NumPy ESP computation at p=101",
            "seconds": float(p101["seconds_median"]),
            "note": "This is not the paper's vectorized 1024-candidate per-iteration time.",
            "treated_as_claim_evidence": False,
        },
        "negative_control": {
            "mutation": "mark Python, modules, and manual data as available",
            "unblocked_mutation_rejected": len(blocking_reasons(negative_audit)) == 0,
        },
        "environment": {
            "fixed_command": "uv sync --frozen && uv run python repro/src/reproduce.py",
            "python": platform.python_version(),
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    }
    (ART / "dependency_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (ART / "negative_control.json").write_text(
        json.dumps(record["negative_control"], indent=2) + "\n"
    )
    (ART / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    subprocess.run(
        [sys.executable, "repro/src/verify_claim5_overhead.py", str(ART / "result.json")],
        cwd=ROOT,
        check=True,
    )
    (ART / "independent_checker.json").write_text(
        json.dumps(
            {
                "source_hashes_recorded": len(AUTHOR_SOURCE) == 6,
                "live_blocking_reasons": record["blocking_reasons"],
                "verifier_exit": 0,
            },
            indent=2,
        )
        + "\n"
    )
    (ART / "EVAL.md").write_text(
        """# Claim 5 evaluation

Verdict: BLOCKED

The exact paper measurement cannot be run under the frozen reproduction
environment. The authors time `gp.fit()` and their vectorized
`EIGFunctionProperty` implementation with a Python 3.11-only Poetry lock,
BoTorch, GPyTorch, Torch, Hydra, and manually provisioned YAHPO/TabPFN data.
The fixed baseline is Python 3.12.11 and contains none of those runtime modules
or manual datasets.

The polynomial Claim 1 timing is deliberately not counted: it measures one
candidate and does not reproduce the paper's vectorized candidate-set call or
its GP fitting block. Changing the lock now would violate the campaign's fixed
environment contract. This is a reproducible blocker, not evidence for or
against the reported 30-second and 25-minute measurements.
"""
    )
    print("=== CLAIM_5_OVERHEAD ===")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
