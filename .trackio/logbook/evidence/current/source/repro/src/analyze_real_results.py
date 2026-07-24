"""Paired multi-budget statistics for the 30-game real-application run."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "real_application"


def main():
    with (OUT / "vit9_realized_mse.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    values = {(int(r["replicate"]), int(r["budget"]), r["method"]): float(r["mse"])
              for r in rows}
    replicates = sorted({k[0] for k in values})
    budgets = sorted({k[1] for k in values})
    methods = sorted({k[2] for k in values})
    expected = len(replicates) * len(budgets) * len(methods)
    if len(values) != expected:
        raise RuntimeError(f"incomplete result grid: {len(values)} != {expected}")

    rng = np.random.default_rng(260602247)
    competitors = [m for m in methods if m not in {
        "ShaplEIG", "ShaplEIG-fixed", "GP-random", "GP-uncertainty", "GP-LeverageSHAP"
    }]
    comparisons = defaultdict(dict)
    for proposed in ("ShaplEIG", "ShaplEIG-fixed"):
        for other in competitors:
            deltas = np.array([
                np.mean([np.log10(values[r, b, proposed]) - np.log10(values[r, b, other])
                         for b in budgets])
                for r in replicates
            ])
            samples = rng.choice(deltas, size=(20000, len(deltas)), replace=True).mean(axis=1)
            comparisons[proposed][other] = {
                "geometric_mean_mse_ratio": float(10 ** deltas.mean()),
                "bootstrap_ratio_95ci": [float(10 ** np.quantile(samples, 0.025)),
                                          float(10 ** np.quantile(samples, 0.975))],
                "replicate_auc_win_fraction": float(np.mean(deltas < 0)),
                "wilcoxon_two_sided_p": float(wilcoxon(deltas).pvalue),
            }

    mean_ranks = {}
    for method in methods:
        ranks = []
        for r in replicates:
            for b in budgets:
                ordered = sorted(methods, key=lambda m: values[r, b, m])
                ranks.append(ordered.index(method) + 1)
        mean_ranks[method] = float(np.mean(ranks))
    result = {
        "analysis": "paired mean log10(MSE) across five matched budgets per official game",
        "replicates": len(replicates), "budgets": budgets,
        "bootstrap_resamples": 20000, "comparisons": dict(comparisons),
        "mean_rank_over_replicate_budget_cells": dict(sorted(mean_ranks.items(), key=lambda x: x[1])),
    }
    (OUT / "statistical_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
