import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from claim2_partial_contract import BUDGETS, TASKS, executed_subset_errors


def valid_rows():
    rows = []
    for task, (family, players) in TASKS.items():
        for replicate in range(1, 31):
            for budget in BUDGETS:
                rows.append(
                    {
                        "task_id": task,
                        "family": family,
                        "players": players,
                        "replicate": replicate,
                        "budget": budget,
                        "method": "ShaplEIG",
                        "mse": 0.1,
                        "data_sha256": f"{task}-{replicate}",
                    }
                )
    return rows


def test_complete_executed_subset_passes():
    assert executed_subset_errors(valid_rows()) == []


def test_dropped_task_fails():
    rows = [row for row in valid_rows() if row["task_id"] != "le_vit9_imagenet"]
    assert executed_subset_errors(rows)

