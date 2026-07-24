"""Machine verifier for the exact Claim 5 blockage contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from claim5_audit import blocking_reasons


def verify(record: dict) -> list[str]:
    errors: list[str] = []
    reasons = blocking_reasons(record["prerequisite_audit"])
    if record["verdict"] != "BLOCKED":
        errors.append("Claim 5 must remain BLOCKED without exact author timing evidence")
    if not reasons:
        errors.append("BLOCKED verdict has no live prerequisite blocker")
    if record["paper_scope"]["task_count"] != 15:
        errors.append("paper timing scope was not audited as 15 tasks")
    if record["paper_scope"]["maximum_players"] != 101:
        errors.append("paper maximum game size was not audited")
    if record["paper_scope"]["maximum_budget"] != 512:
        errors.append("paper maximum evaluation budget was not audited")
    if record["non_equivalent_diagnostic"]["treated_as_claim_evidence"]:
        errors.append("single-candidate ESP diagnostic was promoted to Claim 5 evidence")
    if not record["negative_control"]["unblocked_mutation_rejected"]:
        errors.append("negative control did not reject a vacuous BLOCKED verdict")
    return errors


def main() -> None:
    record = json.loads(Path(sys.argv[1]).read_text())
    errors = verify(record)
    print(json.dumps({"verifier": "claim_5", "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
