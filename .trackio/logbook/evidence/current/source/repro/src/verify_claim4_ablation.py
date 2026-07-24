"""Independent raw-data verifier for the Figure 3 falsification."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from claim4_contract import EXPECTED_BUDGETS, EXPECTED_METHODS, core_claim4_errors


def recompute(raw_path: Path) -> dict:
    with raw_path.open() as handle:
        rows = list(csv.DictReader(handle))
    values: dict[tuple[int, int, str], float] = {}
    sums: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (int(row["replicate"]), int(row["budget"]), row["method"])
        if key in values:
            raise ValueError(f"duplicate raw key: {key}")
        value = float(row["mse"])
        values[key] = value
        sums[(key[1], key[2])].append(value)
    expected = {
        (replicate, budget, method)
        for replicate in range(1, 31)
        for budget in EXPECTED_BUDGETS
        for method in EXPECTED_METHODS
    }
    if set(values) != expected:
        raise ValueError(
            f"incomplete raw grid: missing={len(expected - set(values))}, "
            f"extra={len(set(values) - expected)}"
        )
    means = {
        str(budget): {
            method: sum(sums[(budget, method)]) / len(sums[(budget, method)])
            for method in EXPECTED_METHODS
        }
        for budget in EXPECTED_BUDGETS
    }
    uncertainty_wins = [
        budget
        for budget in EXPECTED_BUDGETS
        if means[str(budget)]["GP-uncertainty"]
        < means[str(budget)]["ShaplEIG"]
    ]
    shapleig_wins = [
        budget
        for budget in EXPECTED_BUDGETS
        if means[str(budget)]["ShaplEIG"]
        < means[str(budget)]["GP-uncertainty"]
    ]
    ranks: dict[str, list[int]] = defaultdict(list)
    for replicate in range(1, 31):
        for budget in EXPECTED_BUDGETS:
            ordered = sorted(
                EXPECTED_METHODS,
                key=lambda method: values[(replicate, budget, method)],
            )
            for index, method in enumerate(ordered, start=1):
                ranks[method].append(index)
    mean_ranks = {
        method: sum(method_ranks) / len(method_ranks)
        for method, method_ranks in ranks.items()
    }
    return {
        "raw_rows": len(rows),
        "aggregate_mean_mse": means,
        "gp_uncertainty_lower_mean_mse_budgets": uncertainty_wins,
        "shapleig_lower_mean_mse_budgets": shapleig_wins,
        "mean_rank_within_four_variants": mean_ranks,
    }


def verify(record: dict, raw_path: Path) -> list[str]:
    errors = core_claim4_errors(record)
    try:
        observed = recompute(raw_path)
    except (KeyError, TypeError, ValueError) as exc:
        return errors + [str(exc)]
    evidence = record["evidence"]
    for key in (
        "raw_rows",
        "aggregate_mean_mse",
        "gp_uncertainty_lower_mean_mse_budgets",
        "shapleig_lower_mean_mse_budgets",
        "mean_rank_within_four_variants",
    ):
        if observed[key] != evidence[key]:
            errors.append(f"raw recomputation mismatch: {key}")
    return errors


def main() -> None:
    record = json.loads(Path(sys.argv[1]).read_text())
    errors = verify(record, Path(sys.argv[2]))
    print(
        json.dumps(
            {
                "verifier": "claim_4_ablation_raw_recomputation",
                "errors": errors,
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
