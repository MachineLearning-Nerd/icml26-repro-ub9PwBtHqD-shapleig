"""Machine verifier for Claims 2-3 exact scope and availability."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scope_contract import core_scope_errors


def verify(record: dict) -> list[str]:
    errors = core_scope_errors(record)
    if record["claim_2_verdict"] != "BLOCKED":
        errors.append("Claim 2 cannot pass with six inaccessible exact tasks")
    if record["claim_3_verdict"] != "BLOCKED":
        errors.append("Claim 3 cannot pass without all 15 performance evaluations")
    if record["retained_performance_evidence"]["tasks"] != 1:
        errors.append("retained judged performance scope was not kept at one task")
    if record["retained_performance_evidence"]["maximum_budget"] != 64:
        errors.append("retained judged performance budget was not kept at 64")
    if record["retained_performance_evidence"]["promoted_to_full_scale"]:
        errors.append("one-task result was promoted to full-scale evidence")
    if not record["negative_control"]["missing_manifest_entry_detected"]:
        errors.append("manifest-deletion negative control was not detected")
    return errors


def main() -> None:
    record = json.loads(Path(sys.argv[1]).read_text())
    errors = verify(record)
    print(json.dumps({"verifier": "claims_2_3", "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
