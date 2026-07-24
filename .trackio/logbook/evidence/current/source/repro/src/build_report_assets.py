"""Build the evidence figures used by the public reproduction report."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "reports" / "shapleig-claims" / "images"
CLAIMS = ROOT / ".openresearch" / "artifacts"
REAL = ROOT / "outputs" / "real_application"

COLORS = {
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#059669",
    "amber": "#d97706",
    "red": "#dc2626",
    "slate": "#475569",
    "light": "#e2e8f0",
}


def finish(fig: plt.Figure, name: str) -> Path:
    path = IMAGES / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def claim3_dv_counterexample() -> Path:
    result = json.loads((CLAIMS / "claim_3" / "result.json").read_text())
    rows = result["counterexamples"]
    labels = {
        "dv_rf_bike_sharing": "Bike / RF",
        "dv_gb_bike_sharing": "Bike / GB",
        "dv_gb_california_housing": "California / GB",
    }
    rows = sorted(rows, key=lambda row: labels[row["task_id"]])
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    ax.bar(
        x - width / 2,
        [row["shapleig_mean_mse"] for row in rows],
        width,
        color=COLORS["blue"],
        label="ShaplEIG",
    )
    ax.bar(
        x + width / 2,
        [row["baseline_mean_mse"] for row in rows],
        width,
        color=COLORS["red"],
        label="Regression MSR",
    )
    for index, row in enumerate(rows):
        ratio = row["geometric_mean_ratio_shapleig_over_baseline"]
        low, high = row["bootstrap_ratio_95ci"]
        ax.text(
            index,
            max(row["shapleig_mean_mse"], row["baseline_mean_mse"]) + 0.00035,
            f"{ratio:.2f}×\n[{low:.2f}, {high:.2f}]",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_xticks(x, [labels[row["task_id"]] for row in rows])
    ax.set_ylabel("Arithmetic mean realized Shapley MSE")
    ax.set_ylim(0, 0.0145)
    ax.set_title(
        "Claim 3 counterexample at budget 16 (30 matched games per task)",
        pad=12,
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return finish(fig, "claim3_dv_counterexample.png")


def claim4_ablation() -> Path:
    evidence = json.loads((CLAIMS / "claim_4" / "result.json").read_text())["evidence"]
    budgets = evidence["budgets"]
    means = evidence["aggregate_mean_mse"]
    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    styles = {
        "ShaplEIG": (COLORS["blue"], "o", 2.8),
        "GP-uncertainty": (COLORS["red"], "s", 2.8),
        "GP-random": (COLORS["slate"], "^", 1.8),
        "GP-LeverageSHAP": (COLORS["amber"], "D", 1.8),
    }
    for method, (color, marker, width) in styles.items():
        ax.plot(
            budgets,
            [means[str(budget)][method] for budget in budgets],
            label=method,
            color=color,
            marker=marker,
            linewidth=width,
            markersize=6,
        )
    for budget in evidence["gp_uncertainty_lower_mean_mse_budgets"]:
        ax.axvspan(budget - 1.0, budget + 1.0, color=COLORS["red"], alpha=0.06)
    ax.set_yscale("log")
    ax.set_xlabel("Coalition-evaluation budget")
    ax.set_ylabel("Arithmetic mean realized Shapley MSE (log scale)")
    ax.set_title("Claim 4 counterexample: GP+Uncertainty wins 4 of 5 budgets")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return finish(fig, "claim4_ablation.png")


def claim1_scaling() -> Path:
    with (CLAIMS / "claim_1" / "scaling.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    efficient = [row for row in rows if row["route"] == "efficient_esp"]
    naive = [row for row in rows if row["route"] == "naive_explicit"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    axes[0].loglog(
        [int(row["p"]) for row in efficient],
        [float(row["operation_count"]) for row in efficient],
        "o-",
        color=COLORS["blue"],
        label="closed-form operation model",
    )
    axes[0].set_title("Closed form follows the predicted quartic cost")
    axes[0].set_xlabel("players p")
    axes[0].set_ylabel("modeled operations")
    axes[0].grid(True, which="both", alpha=0.2)
    axes[0].text(
        0.05,
        0.88,
        "slope = 3.970\n$R^2$ = 0.999993",
        transform=axes[0].transAxes,
        va="top",
    )
    axes[1].semilogy(
        [int(row["p"]) for row in efficient],
        [float(row["peak_bytes_model"]) / 2**20 for row in efficient],
        "o-",
        color=COLORS["green"],
        label="closed form",
    )
    axes[1].semilogy(
        [int(row["p"]) for row in naive],
        [float(row["peak_bytes_model"]) / 2**20 for row in naive],
        "s-",
        color=COLORS["red"],
        label="explicit $2^p$ grid",
    )
    axes[1].set_title("The explicit route becomes exponentially impractical")
    axes[1].set_xlabel("players p")
    axes[1].set_ylabel("modeled peak memory (MiB)")
    axes[1].grid(True, which="both", alpha=0.2)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return finish(fig, "claim1_scaling.png")


def scope_coverage() -> Path:
    with (CLAIMS / "claim_2" / "task_inventory.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    families = [
        "feature_importance",
        "data_valuation",
        "hyperparameter_importance",
        "local_explanation",
    ]
    labels = ["Feature\nimportance", "Data\nvaluation", "Hyperparameter\nimportance", "Local\nexplanation"]
    runnable = Counter(row["family"] for row in rows if row["route"] != "blocked")
    blocked = Counter(row["family"] for row in rows if row["route"] == "blocked")
    x = np.arange(len(families))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x, [runnable[f] for f in families], color=COLORS["green"], label="exact route available")
    ax.bar(
        x,
        [blocked[f] for f in families],
        bottom=[runnable[f] for f in families],
        color=COLORS["amber"],
        label="blocked exact route",
    )
    for index, family in enumerate(families):
        total = runnable[family] + blocked[family]
        ax.text(index, total + 0.08, str(total), ha="center", fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 7)
    ax.set_ylabel("paper tasks")
    ax.set_title("Claims 2–3: 9 of 15 exact tasks are runnable")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return finish(fig, "scope_coverage.png")


def retained_sota() -> Path:
    with (REAL / "aggregate_mse.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    wanted = ["ShaplEIG", "KernelSHAP", "LeverageSHAP", "PermutationSampling", "RegressionMSR"]
    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    palette = [COLORS["blue"], COLORS["red"], COLORS["amber"], COLORS["slate"], COLORS["cyan"]]
    for method, color in zip(wanted, palette, strict=True):
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["budget"]),
        )
        ax.plot(
            [int(row["budget"]) for row in selected],
            [float(row["geometric_mean_mse"]) for row in selected],
            marker="o",
            linewidth=2.5 if method == "ShaplEIG" else 1.7,
            color=color,
            label=method,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Coalition-evaluation budget")
    ax.set_ylabel("Geometric mean realized Shapley MSE")
    ax.set_title("Retained evidence: one official ViT-9 task, not the 15-task claim")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return finish(fig, "claim3_retained_task.png")


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    paths = [
        claim3_dv_counterexample(),
        claim4_ablation(),
        claim1_scaling(),
        scope_coverage(),
        retained_sota(),
    ]
    manifest = {
        path.relative_to(ROOT).as_posix(): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in paths
    }
    manifest_path = IMAGES.parent / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
