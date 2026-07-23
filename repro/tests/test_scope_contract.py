"""Tests for the exact 15-task scope contract."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scope_contract import EXPECTED_MISSING, core_scope_errors


def complete_record():
    return {
        "paper_scope": {"tasks": 15, "families": 4, "maximum_budget": 512},
        "availability": {
            "runnable_tasks": 9,
            "blocked_tasks": 6,
            "blocked_task_ids": sorted(EXPECTED_MISSING),
            "precomputed_repetitions": {"example": 30},
            "tree_constructor_smoke": {
                "tree": {
                    "passed": True,
                    "expected_players": 101,
                    "observed_players": 101,
                }
            },
        },
    }


def test_complete_scope_inventory_passes():
    assert core_scope_errors(complete_record()) == []


def test_deleted_public_repetition_is_detected():
    record = complete_record()
    record["availability"]["precomputed_repetitions"]["example"] = 29
    assert core_scope_errors(record)
