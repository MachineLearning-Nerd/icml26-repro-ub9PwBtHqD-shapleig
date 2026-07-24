import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from claim4_contract import core_claim4_errors


def valid_record():
    return {
        "verdict": "FALSIFIED",
        "evidence": {
            "task": "ImageNet ViT-9 local explanation",
            "replicates": 30,
            "budgets": [16, 24, 32, 48, 64],
            "methods": [
                "GP-LeverageSHAP",
                "GP-random",
                "GP-uncertainty",
                "ShaplEIG",
            ],
            "raw_rows": 600,
            "gp_uncertainty_lower_mean_mse_budgets": [16, 24, 32, 48],
            "shapleig_lower_mean_mse_budgets": [64],
            "mean_rank_within_four_variants": {
                "GP-uncertainty": 2.1,
                "ShaplEIG": 2.25,
            },
        },
        "negative_control": {"detected": True},
    }


def test_valid_record_passes():
    assert core_claim4_errors(valid_record()) == []


def test_mutated_winner_is_rejected():
    record = valid_record()
    record["evidence"]["gp_uncertainty_lower_mean_mse_budgets"] = []
    assert core_claim4_errors(record)
