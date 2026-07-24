"""Blind, canonical-entrypoint audit of the evaluator-visible logbook."""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / ".trackio" / "logbook"
CURRENT = BOOK / "pages" / "current-verification" / "page.md"
EVIDENCE = BOOK / "evidence" / "current"
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def run_bad(name: str, command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "repro" / "src")},
    )
    return {
        "claim": name,
        "exit_code": completed.returncode,
        "nonzero_as_required": completed.returncode != 0,
        "output": completed.stdout[-2000:],
    }


def drop_first_data_row(source: Path, target: Path) -> None:
    with source.open(newline="") as handle:
        rows = list(csv.reader(handle))
    with target.open("w", newline="") as handle:
        csv.writer(handle).writerows([rows[0], *rows[2:]])


def negative_verifier_audit() -> list[dict]:
    historical = BOOK / "evidence"
    results: list[dict] = []
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)

        c1 = json.loads((historical / "claim_1" / "result.json").read_text())
        c1["independent_checker"]["maximum_absolute_error"] = 1.0
        c1_bad = temp / "c1.json"
        c1_bad.write_text(json.dumps(c1))
        results.append(
            run_bad(
                "C1",
                [sys.executable, "repro/src/verify_claim1_complexity.py", str(c1_bad)],
            )
        )

        c2_bad = temp / "c2.csv"
        drop_first_data_row(
            historical / "claim_2" / "executed_subset_raw.csv", c2_bad
        )
        results.append(
            run_bad(
                "C2",
                [
                    sys.executable,
                    "repro/src/verify_claim2_partial_scope.py",
                    str(c2_bad),
                    str(historical / "claim_2" / "executed_subset_summary.json"),
                ],
            )
        )

        c3_bad = temp / "c3.csv"
        c3_result = temp / "c3_result.json"
        drop_first_data_row(historical / "claim_3" / "dv10_raw.csv", c3_bad)
        c3_result.write_bytes((historical / "claim_3" / "result.json").read_bytes())
        results.append(
            run_bad(
                "C3",
                [
                    sys.executable,
                    "repro/src/verify_claim3_dv.py",
                    str(c3_bad),
                    str(c3_result),
                ],
            )
        )

        c4_bad = temp / "c4.csv"
        drop_first_data_row(historical / "claim_4" / "ablation_raw.csv", c4_bad)
        results.append(
            run_bad(
                "C4",
                [
                    sys.executable,
                    "repro/src/verify_claim4_ablation.py",
                    str(historical / "claim_4" / "result.json"),
                    str(c4_bad),
                ],
            )
        )

        c5_bad = temp / "c5.csv"
        drop_first_data_row(EVIDENCE / "claim_5" / "large_timing_raw.csv", c5_bad)
        results.append(
            run_bad(
                "C5",
                [
                    sys.executable,
                    "repro/src/verify_claim5_large_timing.py",
                    str(c5_bad),
                    str(EVIDENCE / "claim_5" / "large_timing_summary.json"),
                ],
            )
        )
    return results


def resolve_links(page: Path) -> tuple[list[str], list[str]]:
    opened, missing = [], []
    for target in LINK_RE.findall(page.read_text()):
        if target.startswith(("http://", "https://", "#")):
            continue
        resolved = (page.parent / target).resolve()
        try:
            relative = resolved.relative_to(BOOK.resolve()).as_posix()
        except ValueError:
            missing.append(target)
            continue
        if resolved.exists():
            opened.append(relative)
        else:
            missing.append(relative)
    return opened, missing


def inline_numeric_errors() -> list[str]:
    """Recompute the principal inline numbers from linked raw/result files."""
    errors: list[str] = []
    current_text = CURRENT.read_text()
    c5_text = (BOOK / "pages" / "current-claim-5" / "page.md").read_text()
    c1 = json.loads((EVIDENCE / "claim_1" / "result.json").read_text())
    c1_error = str(c1["independent_checker"]["maximum_absolute_error"])
    if c1_error not in current_text:
        errors.append("C1 inline maximum error differs from current result JSON")

    c2 = json.loads(
        (BOOK / "evidence" / "claim_2" / "executed_subset_summary.json").read_text()
    )
    scope = c2["executed_scope"]
    for value in (
        f"**{scope['tasks']} tasks, {scope['families']} families",
        f"budgets {', '.join(map(str, scope['budgets']))}",
        f"({scope['raw_rows']} complete task-budget-game cells)",
    ):
        if value not in current_text:
            errors.append(f"C2 inline scope mismatch: {value}")

    c3 = json.loads((BOOK / "evidence" / "claim_3" / "result.json").read_text())
    names = {
        "dv_rf_bike_sharing": "Bike Sharing / RF",
        "dv_gb_bike_sharing": "Bike Sharing / GB",
        "dv_gb_california_housing": "California Housing / GB",
    }
    for row in c3["counterexamples"]:
        if row["budget"] != 16 or row["baseline"] != "RegressionMSR":
            continue
        expected = (
            f"| {names[row['task_id']]} | {row['shapleig_mean_mse']:.6f} | "
            f"{row['baseline_mean_mse']:.6f} | "
            f"{row['geometric_mean_ratio_shapleig_over_baseline']:.3f} "
            f"[{row['bootstrap_ratio_95ci'][0]:.3f}, "
            f"{row['bootstrap_ratio_95ci'][1]:.3f}] | "
            f"{row['holm_adjusted_p']:.6g} |"
        )
        if expected not in current_text:
            errors.append(f"C3 inline row differs from result JSON: {row['task_id']}")

    c4 = json.loads((BOOK / "evidence" / "claim_4" / "result.json").read_text())
    c4_evidence = c4["evidence"]
    for budget in (16, 24, 32, 48, 64):
        means = c4_evidence["aggregate_mean_mse"][str(budget)]
        lower = (
            "GP+Uncertainty"
            if means["GP-uncertainty"] < means["ShaplEIG"]
            else "ShaplEIG"
        )
        expected = (
            f"| {budget} | {means['ShaplEIG']:.7f} | "
            f"{means['GP-uncertainty']:.7f} | {lower} |"
        )
        if expected not in current_text:
            errors.append(f"C4 inline row differs from raw reconstruction: {budget}")

    with (EVIDENCE / "claim_5" / "large_timing_raw.csv").open(newline="") as handle:
        c5_rows = list(csv.DictReader(handle))
    labels = {
        "le_rf_corrgroups60": "CorrGroups60",
        "le_rf_nhanes": "NHANES",
        "le_rf_crime": "Crime",
    }
    for row in c5_rows:
        expected = (
            f"| {labels[row['task_id']]} | {row['players']} | "
            f"{row['archive_size']} | {float(row['hp_fit_seconds']):.9f} | "
            f"{float(row['eig_seconds']):.9f} |"
        )
        if expected not in c5_text:
            errors.append(
                f"C5 inline row differs from raw CSV: "
                f"{row['task_id']}/t={row['archive_size']}"
            )
    return errors


def main() -> None:
    readme = BOOK / "README.md"
    logbook_path = BOOK / "logbook.json"
    logbook = json.loads(logbook_path.read_text())
    first = logbook["root"]["children"][0]
    readme_opened, readme_missing = resolve_links(readme)
    current_opened, current_missing = resolve_links(CURRENT)

    negative = negative_verifier_audit()
    (EVIDENCE / "negative_verifier_exits.json").write_text(
        json.dumps(negative, indent=2) + "\n"
    )

    matrix = []
    expected_verdicts = {
        "C1": "VERIFIED",
        "C2": "BLOCKED",
        "C3": "FALSIFIED",
        "C4": "FALSIFIED",
        "C5": "BLOCKED",
    }
    text = CURRENT.read_text()
    pieces = re.split(r"\n## Claim ([1-5]) —", text)
    sections = {
        f"C{pieces[index]}": pieces[index + 1]
        for index in range(1, len(pieces), 2)
    }
    for claim, verdict in expected_verdicts.items():
        section = sections.get(claim, "")
        visible = bool(section)
        matrix.append(
            {
                "claim": claim,
                "canonical_page": "pages/current-verification/page.md",
                "code_visible": visible and "verifier:" in section,
                "data_inline": visible
                and bool(re.search(r"\d", section))
                and any(
                    marker in section
                    for marker in ("Observed inline", "Finite corroboration", "Raw maxima")
                ),
                "raw_link": visible and "raw:" in section,
                "checker": visible and "checker:" in section,
                "control": visible and "control:" in section,
                "exact_claim_tested": visible and "Exact " in section,
                "reviewer_verdict": verdict,
            }
        )
    (EVIDENCE / "visibility_matrix.json").write_text(
        json.dumps(matrix, indent=2) + "\n"
    )

    trace = {
        "review_mode": "candidate artifact and canonical entrypoints only",
        "entrypoints_opened": ["README.md", "logbook.json"],
        "navigation_first_child": first,
        "files_opened_from_readme": readme_opened,
        "files_opened_from_current_page": current_opened,
        "unresolved_links": sorted(set(readme_missing + current_missing)),
        "conclusions_not_verifiable": inline_numeric_errors(),
        "historical_pages_used_for_orientation": False,
    }
    (EVIDENCE / "blind_review.json").write_text(json.dumps(trace, indent=2) + "\n")

    failures = []
    if "pages/current-verification/page.md" not in readme_opened:
        failures.append("README does not link the current verifier")
    if first.get("file") != "pages/current-verification/page.md":
        failures.append("current verifier is not first in logbook navigation")
    if current_missing or readme_missing:
        failures.append(f"unresolved canonical links: {trace['unresolved_links']}")
    failures.extend(trace["conclusions_not_verifiable"])
    if not all(item["nonzero_as_required"] for item in negative):
        failures.append("one or more corrupted artifacts produced a zero exit")
    required_cells = (
        "code_visible",
        "data_inline",
        "raw_link",
        "checker",
        "control",
        "exact_claim_tested",
    )
    for row in matrix:
        if not all(row[field] for field in required_cells):
            failures.append(f"incomplete visibility row: {row['claim']}")
    historical = logbook["root"]["children"][2:]
    if any(
        not node["title"].startswith("Historical rejected baseline")
        for node in historical
    ):
        failures.append("a historical root page lacks the rejected-baseline label")

    print("=== EVALUATOR_VISIBILITY_GATE ===")
    print(json.dumps({"matrix": matrix, "negative_exits": negative, "errors": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
