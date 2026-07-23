"""Machine verifier for the Claim 1 complexity contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def verify(record: dict) -> list[str]:
    errors = []
    if record["independent_checker"]["maximum_absolute_error"] > 1e-7:
        errors.append("efficient EIG disagrees with explicit covariance route")
    if record["efficient_scaling"]["maximum_p_completed"] < 101:
        errors.append("efficient route did not reach the paper's maximum p=101")
    if record["efficient_scaling"]["operation_count_loglog_slope"] > 4.35:
        errors.append("executable ESP operation count grows faster than p^4")
    if record["efficient_scaling"]["operation_count_loglog_slope"] < 3.5:
        errors.append("operation-count audit did not exercise the p^4 term")
    if record["naive_scaling"]["minimum_exponential_r2"] < 0.90:
        errors.append("naive route did not display exponential memory growth")
    if record["naive_scaling"]["largest_completed_p"] < 9:
        errors.append("naive route was not checked beyond toy p=4")
    if record["negative_control"]["corruption_detected"] is not True:
        errors.append("negative control did not fail as intended")
    return errors


def main() -> None:
    path = Path(sys.argv[1])
    record = json.loads(path.read_text())
    errors = verify(record)
    print(json.dumps({"verifier": "claim_1", "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
