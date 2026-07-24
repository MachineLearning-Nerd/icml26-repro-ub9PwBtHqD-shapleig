import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from claim5_timing_contract import TASKS, archive_sizes, timing_errors


def valid_rows():
    return [
        {
            "task_id": task,
            "replicate": replicate,
            "players": p,
            "archive_size": archive,
            "candidate_count": (1 << p) - archive,
            "fit_restarts": 5,
            "hp_fit_seconds": 0.1,
            "eig_seconds": 0.01,
        }
        for task, p in TASKS.items()
        for replicate in range(1, 6)
        for archive in archive_sizes(p)
    ]


def test_complete_timing_grid_passes():
    assert timing_errors(valid_rows()) == []


def test_missing_timing_cell_fails():
    rows = valid_rows()
    rows.pop()
    assert timing_errors(rows)
