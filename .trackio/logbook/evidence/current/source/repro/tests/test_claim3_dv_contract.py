from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from claim3_dv_contract import BUDGETS, METHODS, TASKS, evidence_errors


def complete_rows() -> list[dict]:
    return [
        {
            "task_id": task_id,
            "replicate": replicate,
            "budget": budget,
            "method": method,
            "mse": 1.0,
        }
        for task_id in TASKS
        for replicate in range(1, 31)
        for budget in BUDGETS
        for method in METHODS
    ]


def test_complete_registered_grid_passes():
    assert evidence_errors(complete_rows()) == []


def test_missing_matched_cell_is_detected():
    assert evidence_errors(complete_rows()[:-1])
