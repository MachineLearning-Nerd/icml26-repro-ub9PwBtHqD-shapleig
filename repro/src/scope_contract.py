"""Shared contract checks for Claims 2 and 3."""
from __future__ import annotations


EXPECTED_MISSING = {
    "fi_tabpfn_diabetes_regression",
    "fi_tabpfn_diabetes",
    "fi_tabpfn_breast_cancer",
    "hpi_xgboost_chess",
    "hpi_xgboost_thyroid",
    "hpi_lcbench_jasmine",
}


def core_scope_errors(record: dict) -> list[str]:
    errors: list[str] = []
    scope = record["paper_scope"]
    availability = record["availability"]
    if scope["tasks"] != 15 or scope["families"] != 4:
        errors.append("Table 1 scope is not 15 tasks across four families")
    if scope["maximum_budget"] != 512:
        errors.append("maximum paper budget is not 512")
    if availability["runnable_tasks"] != 9:
        errors.append("pinned public/frozen environment inventory is not 9 runnable tasks")
    if availability["blocked_tasks"] != 6:
        errors.append("pinned inventory is not 6 blocked tasks")
    if set(availability["blocked_task_ids"]) != EXPECTED_MISSING:
        errors.append("unexpected set of blocked tasks")
    for task_id, count in availability["precomputed_repetitions"].items():
        if count != 30:
            errors.append(f"{task_id} has {count} public repetitions, expected 30")
    for task_id, smoke in availability["tree_constructor_smoke"].items():
        if smoke["passed"] is not True:
            errors.append(f"{task_id} exact constructor smoke failed")
        if smoke["observed_players"] != smoke["expected_players"]:
            errors.append(f"{task_id} player count differs from Table 1")
    return errors
