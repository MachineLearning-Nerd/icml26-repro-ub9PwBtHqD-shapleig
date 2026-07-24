"""Generate aggregate tables and uncertainty-aware plots from realized MSE."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "real_application"


def main():
    with (OUT / "vit9_realized_mse.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"], int(row["budget"])].append(float(row["mse"]))
    summary_rows = []
    for (method, budget), vals in sorted(grouped.items()):
        array = np.asarray(vals)
        logs = np.log(array)
        log_se = logs.std(ddof=1) / np.sqrt(len(array))
        summary_rows.append({"method": method, "budget": budget, "replicates": len(array),
                             "mean_mse": array.mean(), "median_mse": np.median(array),
                             "geometric_mean_mse": np.exp(logs.mean()),
                             "geometric_95ci_low": np.exp(logs.mean() - 1.96 * log_se),
                             "geometric_95ci_high": np.exp(logs.mean() + 1.96 * log_se),
                             "standard_error": array.std(ddof=1) / np.sqrt(len(array))})
    with (OUT / "aggregate_mse.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summary_rows)

    panels = {
        "sota_mse_curve.png": ["ShaplEIG", "KernelSHAP", "LeverageSHAP", "RegressionMSR",
                               "UnbiasedKernelSHAP", "SVARM", "PermutationSampling"],
        "gp_ablation_curve.png": ["ShaplEIG", "ShaplEIG-fixed", "GP-random",
                                  "GP-uncertainty", "GP-LeverageSHAP"],
    }
    for filename, methods in panels.items():
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for method in methods:
            selected = [r for r in summary_rows if r["method"] == method]
            x = np.array([r["budget"] for r in selected])
            y = np.array([r["geometric_mean_mse"] for r in selected])
            low = np.array([r["geometric_95ci_low"] for r in selected])
            high = np.array([r["geometric_95ci_high"] for r in selected])
            ax.plot(x, y, marker="o", linewidth=2, label=method)
            ax.fill_between(x, low, high, alpha=0.12)
        ax.set_yscale("log")
        ax.set_xlabel("Coalition-evaluation budget")
        ax.set_ylabel("Realized Shapley MSE (geometric mean, 95% CI)")
        ax.set_title("ImageNet/ViT-9 local explanation — 30 official games")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=180)
        plt.close(fig)
    print(f"wrote {len(summary_rows)} aggregate rows and {len(panels)} plots")


if __name__ == "__main__":
    main()
