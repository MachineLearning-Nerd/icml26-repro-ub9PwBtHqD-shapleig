"""Prerequisite logic for the exact Claim 5 timing reproduction."""
from __future__ import annotations


REQUIRED_AUTHOR_MODULES = (
    "torch",
    "gpytorch",
    "botorch",
    "hydra",
    "yahpo_gym",
    "tabpfn",
)


def blocking_reasons(audit: dict) -> list[str]:
    reasons: list[str] = []
    if audit["python_contract"]["compatible"] is not True:
        reasons.append("author and frozen Python contracts are incompatible")
    missing = audit["module_audit"]["missing"]
    if missing:
        reasons.append("missing author runtime modules: " + ", ".join(missing))
    if audit["required_external_data"]["all_present"] is not True:
        reasons.append("manually provisioned author task data are incomplete")
    return reasons
