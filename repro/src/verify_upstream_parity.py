"""Check local SOTA baselines against the authors' pinned source byte-for-byte."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from fetch_vit9_games import fetch
from real_application import (DATA, OUT, game_callable, leverage_estimate,
                              load_game, regression_msr_estimate, sv_from_result)

UPSTREAM = "162ce44fe380c7c11b959fc85206b5dcdeddff58"
BASE = f"https://raw.githubusercontent.com/slds-lmu/shapleig/{UPSTREAM}/src/xac/applications"


def load_remote(module: str):
    payload = urlopen(f"{BASE}/{module}.py").read()
    path = Path(tempfile.gettempdir()) / f"shapleig-{UPSTREAM}-{module}.py"
    path.write_bytes(payload)
    spec = importlib.util.spec_from_file_location(f"upstream_{module}", path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def main():
    fetch(DATA)
    poly, rmsr = load_remote("polyshap"), load_remote("regressionMSR")
    frontier = poly.ExplanationFrontierGenerator(N=list(range(9))).generate_kadd(max_order=1)
    rows = []
    cases = [(1, 16), (1, 32), (1, 64), (12, 16), (16, 16)]
    for seed, budget in cases:
        _, values = load_game(DATA / f"model_name=vit_9_patches_{seed}.npz")
        game = game_callable(values)
        official_lev = poly.PolySHAP(
            n=9, explanation_frontier=frontier, sampling_weights=np.ones(10),
            pairing_trick=True, random_state=seed,
        ).approximate(budget=budget, game=game)
        official_reg = rmsr.RegressionMSR(
            n=9, pairing_trick=True, replacement=False,
            sampling_weights=np.ones(10), random_state=seed,
        ).approximate(budget=budget, game=game)
        rows.append({
            "replicate": seed,
            "budget": budget,
            "leverage_max_abs_diff": float(np.max(np.abs(
                sv_from_result(official_lev, 9) - leverage_estimate(values, budget, seed)
            ))),
            "regression_msr_max_abs_diff": float(np.max(np.abs(
                sv_from_result(official_reg, 9) - regression_msr_estimate(values, budget, seed)
            ))),
        })
    result = {
        "upstream": f"slds-lmu/shapleig@{UPSTREAM}",
        "budgets": rows,
        "pass": all(max(r["leverage_max_abs_diff"], r["regression_msr_max_abs_diff"]) < 1e-12
                    for r in rows),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "upstream_parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
