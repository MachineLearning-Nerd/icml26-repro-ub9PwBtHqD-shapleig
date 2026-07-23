"""Contract checks for the Figure 3 GP-surrogate ablation claim."""
from __future__ import annotations


EXPECTED_BUDGETS = [16, 24, 32, 48, 64]
EXPECTED_METHODS = [
    "GP-LeverageSHAP",
    "GP-random",
    "GP-uncertainty",
    "ShaplEIG",
]


def core_claim4_errors(record: dict) -> list[str]:
    """Return contract violations without reading any external evidence."""
    errors: list[str] = []
    if record.get("verdict") != "FALSIFIED":
        errors.append("the exact Figure 3 claim must remain FALSIFIED")
    evidence = record.get("evidence", {})
    if evidence.get("task") != "ImageNet ViT-9 local explanation":
        errors.append("wrong official ablation task")
    if evidence.get("replicates") != 30:
        errors.append("expected all 30 official game replicates")
    if evidence.get("budgets") != EXPECTED_BUDGETS:
        errors.append("expected the five evaluated ablation budgets")
    if evidence.get("methods") != EXPECTED_METHODS:
        errors.append("expected exactly the four Figure 3 GP variants")
    if evidence.get("raw_rows") != 600:
        errors.append("expected a complete 30 x 5 x 4 raw grid")
    if evidence.get("gp_uncertainty_lower_mean_mse_budgets") != [16, 24, 32, 48]:
        errors.append("GP+Uncertainty no longer wins four arithmetic-mean curves")
    if evidence.get("shapleig_lower_mean_mse_budgets") != [64]:
        errors.append("ShaplEIG no longer wins only at budget 64")
    ranks = evidence.get("mean_rank_within_four_variants", {})
    if not (
        ranks.get("GP-uncertainty", float("inf"))
        < ranks.get("ShaplEIG", float("-inf"))
    ):
        errors.append("GP+Uncertainty must have the better within-ablation mean rank")
    if not record.get("negative_control", {}).get("detected"):
        errors.append("label-swap negative control was not detected")
    return errors
