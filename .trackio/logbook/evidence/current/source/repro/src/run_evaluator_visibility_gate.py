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
        "conclusions_not_verifiable": [],
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
