"""Machine verifier for the Claim 1 complexity contract."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def source_bound_errors(root: Path = ROOT) -> list[str]:
    """Structural guard for the source audited by the symbolic certificate."""
    source_path = root / "repro" / "src" / "complexity.py"
    certificate_path = (
        root
        / ".trackio"
        / "logbook"
        / "evidence"
        / "current"
        / "claim_1"
        / "symbolic_derivation.md"
    )
    tree = ast.parse(source_path.read_text())
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    errors: list[str] = []
    akz = functions.get("akz_esp")
    if akz is None:
        return ["akz_esp is absent from the audited source"]
    convolutions = [
        node
        for node in ast.walk(akz)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "convolve"
    ]
    if len(convolutions) != 1:
        errors.append("akz_esp must form exactly one full polynomial per row")
    if "excluded = np.empty(p)" not in ast.unparse(akz):
        errors.append("akz_esp does not expose the O(p) synthetic-division buffer")
    if not any(
        isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "degree"
        for node in ast.walk(akz)
    ):
        errors.append("akz_esp synthetic-division recurrence is absent")
    efficient_names = ("akz_esp", "akazza_esp", "efficient_eig", "efficient_eig_batch")
    for name in efficient_names:
        node = functions.get(name)
        if node is None:
            errors.append(f"{name} is absent from the audited source")
            continue
        if any(isinstance(item, ast.LShift) for item in ast.walk(node)):
            errors.append(f"{name} allocates an exponential 2^p route")
    if not certificate_path.exists():
        errors.append("symbolic complexity certificate is absent")
    else:
        certificate = certificate_path.read_text()
        for required in (
            "O(p⁴ + t³)",
            "synthetic division",
            "every positive integer `p,t`",
            "finite corroboration, not a proof",
        ):
            if required not in certificate:
                errors.append(f"symbolic certificate omits: {required}")
    return errors


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
    errors.extend(source_bound_errors())
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
