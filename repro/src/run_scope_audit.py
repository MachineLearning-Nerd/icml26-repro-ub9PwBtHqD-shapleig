"""Audit all 15 paper tasks and smoke-test exact public tree tasks."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from scope_contract import core_scope_errors


ROOT = Path(__file__).resolve().parents[2]
ART2 = ROOT / ".openresearch" / "artifacts" / "claim_2"
ART3 = ROOT / ".openresearch" / "artifacts" / "claim_3"
SHAPIQ_COMMIT = "799cfd0f2c32f17446130247a7ac3519e68cce82"
TREE_API = (
    "https://api.github.com/repos/mmschlk/shapiq/git/trees/"
    f"{SHAPIQ_COMMIT}?recursive=1"
)

TASKS = [
    ("fi_tabpfn_diabetes_regression", "feature_importance", "TabPFN", "Diabetes (Reg.)", 10, 100, "blocked"),
    ("fi_tabpfn_diabetes", "feature_importance", "TabPFN", "Diabetes", 8, 100, "blocked"),
    ("fi_tabpfn_breast_cancer", "feature_importance", "TabPFN", "Breast Cancer", 8, 100, "blocked"),
    ("dv_rf_bike_sharing", "data_valuation", "RF", "Bike Sharing", 10, 30, "precomputed"),
    ("dv_gb_bike_sharing", "data_valuation", "GB", "Bike Sharing", 10, 30, "precomputed"),
    ("dv_gb_california_housing", "data_valuation", "GB", "California Housing", 10, 30, "precomputed"),
    ("hpi_xgboost_chess", "hyperparameter_importance", "XGBoost", "Chess", 16, 100, "blocked"),
    ("hpi_xgboost_thyroid", "hyperparameter_importance", "XGBoost", "Thyroid", 16, 100, "blocked"),
    ("hpi_lcbench_jasmine", "hyperparameter_importance", "LCBench", "Jasmine", 8, 100, "blocked"),
    ("le_rf_corrgroups60", "local_explanation", "RF", "CorrGroups60", 60, 30, "tree"),
    ("le_rf_nhanes", "local_explanation", "RF", "NHANES", 79, 30, "tree"),
    ("le_rf_crime", "local_explanation", "RF", "Crime", 101, 30, "tree"),
    ("le_resnet_imagenet", "local_explanation", "ResNet", "ImageNet", 14, 30, "precomputed"),
    ("le_vit9_imagenet", "local_explanation", "ViT (9 patches)", "ImageNet", 9, 30, "precomputed"),
    ("le_vit16_imagenet", "local_explanation", "ViT (16 patches)", "ImageNet", 16, 30, "precomputed"),
]

PRECOMPUTED = {
    "dv_rf_bike_sharing": (
        "data/precomputed_games/BikeSharing_DatasetValuation_Game/10/",
        "model_name=random_forest_",
    ),
    "dv_gb_bike_sharing": (
        "data/precomputed_games/BikeSharing_DatasetValuation_Game/10/",
        "model_name=gradient_boosting_",
    ),
    "dv_gb_california_housing": (
        "data/precomputed_games/CaliforniaHousing_DatasetValuation_Game/10/",
        "model_name=gradient_boosting_",
    ),
    "le_resnet_imagenet": (
        "data/precomputed_games/ImageClassifier_Game/14/",
        "model_name=resnet_18_",
    ),
    "le_vit9_imagenet": (
        "data/precomputed_games/ImageClassifier_Game/9/",
        "model_name=vit_9_",
    ),
    "le_vit16_imagenet": (
        "data/precomputed_games/ImageClassifier_Game/16/",
        "model_name=vit_16_",
    ),
}

TREE_TASKS = {
    "le_rf_corrgroups60": ("corrgroups60", 60),
    "le_rf_nhanes": ("nhanesi", 79),
    "le_rf_crime": ("communitiesandcrime", 101),
}


def public_manifest() -> tuple[bytes, list[str]]:
    request = Request(TREE_API, headers={"User-Agent": "OpenResearch-Reproduction/1.0"})
    payload = urlopen(request, timeout=60).read()
    tree = json.loads(payload)
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree response is truncated")
    paths = [item["path"] for item in tree["tree"] if item["type"] == "blob"]
    return payload, paths


def build_tree_game(dataset_name: str, expected_p: int, seed: int):
    import shap
    from sklearn.ensemble import RandomForestRegressor
    from shapiq_games.benchmark.treeshapiq_xai.base import TreeSHAPIQXAI

    dataset_X, dataset_y = getattr(shap.datasets, dataset_name)()
    observed_p = int(dataset_X.shape[1])
    if observed_p != expected_p:
        raise AssertionError(f"{dataset_name}: expected {expected_p}, got {observed_p}")
    x_train = dataset_X.drop(dataset_X.index[[seed]]).to_numpy()
    y_array = np.asarray(dataset_y)
    y_train = np.delete(y_array, seed, axis=0)
    x_explain = dataset_X.iloc[[seed]].to_numpy()
    game = None
    used_depth = None
    for max_depth in (32, 16, 8, 4):
        model = RandomForestRegressor(
            n_estimators=10, random_state=seed, max_depth=max_depth
        )
        model.fit(x_train, y_train.ravel())
        try:
            game = TreeSHAPIQXAI(x=x_explain, tree_model=model, normalize=False)
            used_depth = max_depth
            break
        except np.linalg.LinAlgError:
            continue
    if game is None:
        raise RuntimeError(f"{dataset_name}: every author depth fallback failed")
    return game, dataset_X, used_depth


def construct_tree_task(dataset_name: str, expected_p: int) -> dict:
    started = time.perf_counter()
    seed = 1
    game, dataset_X, used_depth = build_tree_game(dataset_name, expected_p, seed)
    coalitions = np.stack(
        [np.zeros(expected_p, dtype=bool), np.ones(expected_p, dtype=bool)]
    )
    values = np.asarray(game.value_function(coalitions), dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise AssertionError(f"{dataset_name}: non-finite smoke values")
    return {
        "passed": True,
        "expected_players": expected_p,
        "observed_players": int(dataset_X.shape[1]),
        "rows": int(dataset_X.shape[0]),
        "seed": seed,
        "rf_estimators": 10,
        "max_depth_used": used_depth,
        "empty_value": float(values[0]),
        "grand_value": float(values[1]),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    ART2.mkdir(parents=True, exist_ok=True)
    ART3.mkdir(parents=True, exist_ok=True)
    payload, paths = public_manifest()
    repetitions = {}
    for task_id, (prefix, name_prefix) in PRECOMPUTED.items():
        repetitions[task_id] = sum(
            path.startswith(prefix)
            and Path(path).name.startswith(name_prefix)
            and path.endswith(".npz")
            for path in paths
        )
    tree_smoke = {}
    for task_id, (dataset_name, expected_p) in TREE_TASKS.items():
        tree_smoke[task_id] = construct_tree_task(dataset_name, expected_p)
        print(f"scope audit {task_id}: {tree_smoke[task_id]}", flush=True)

    blocked_ids = [task[0] for task in TASKS if task[-1] == "blocked"]
    record = {
        "claim_2_verdict": "BLOCKED",
        "claim_3_verdict": "BLOCKED",
        "paper_scope": {
            "tasks": len(TASKS),
            "families": len({task[1] for task in TASKS}),
            "task_players": [task[4] for task in TASKS],
            "maximum_budget": 512,
        },
        "availability": {
            "runnable_tasks": len(TASKS) - len(blocked_ids),
            "blocked_tasks": len(blocked_ids),
            "blocked_task_ids": blocked_ids,
            "runnable_families": ["data_valuation", "local_explanation"],
            "precomputed_repetitions": repetitions,
            "tree_constructor_smoke": tree_smoke,
            "public_manifest": {
                "url": TREE_API,
                "commit": SHAPIQ_COMMIT,
                "response_sha256": hashlib.sha256(payload).hexdigest(),
                "blob_paths": len(paths),
            },
        },
        "retained_performance_evidence": {
            "tasks": 1,
            "families": 1,
            "replicates": 30,
            "maximum_budget": 64,
            "promoted_to_full_scale": False,
        },
        "blocking_reasons": {
            "feature_importance": "author-only TabPFN precomputations and absent TabPFN runtime",
            "hyperparameter_importance": "absent YAHPO runtime and manually provisioned surrogate metadata",
            "claim_3": "no 15-task matched performance matrix exists under the frozen contract",
        },
        "environment": {
            "fixed_command": "uv sync --frozen && uv run python repro/src/reproduce.py",
            "python": platform.python_version(),
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    }
    mutated = copy.deepcopy(record)
    first = next(iter(mutated["availability"]["precomputed_repetitions"]))
    mutated["availability"]["precomputed_repetitions"][first] = 29
    record["negative_control"] = {
        "mutation": f"delete one public manifest entry from {first}",
        "missing_manifest_entry_detected": bool(core_scope_errors(mutated)),
    }
    errors = core_scope_errors(record)
    if errors:
        raise AssertionError(errors)

    inventory_path = ART2 / "task_inventory.csv"
    with inventory_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["task_id", "family", "model", "dataset", "players", "repetitions", "route"]
        )
        writer.writerows(TASKS)
    (ART2 / "public_manifest_summary.json").write_text(
        json.dumps(record["availability"], indent=2) + "\n"
    )
    (ART2 / "tree_constructor_smoke.json").write_text(
        json.dumps(tree_smoke, indent=2) + "\n"
    )
    (ART2 / "negative_control.json").write_text(
        json.dumps(record["negative_control"], indent=2) + "\n"
    )
    (ART2 / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    (ART3 / "result.json").write_text(
        json.dumps(
            {
                "verdict": "BLOCKED",
                "scope_audit": "../claim_2/result.json",
                "retained_performance_evidence": record["retained_performance_evidence"],
                "reason": record["blocking_reasons"]["claim_3"],
            },
            indent=2,
        )
        + "\n"
    )
    subprocess.run(
        [sys.executable, "repro/src/verify_claim23_scope.py", str(ART2 / "result.json")],
        cwd=ROOT,
        check=True,
    )
    checker = {"verifier_exit": 0, "errors": [], "task_rows": len(TASKS)}
    (ART2 / "independent_checker.json").write_text(json.dumps(checker, indent=2) + "\n")
    (ART3 / "independent_checker.json").write_text(json.dumps(checker, indent=2) + "\n")
    (ART2 / "EVAL.md").write_text(
        """# Claim 2 evaluation

Verdict: BLOCKED

The paper scope is confirmed as 15 tasks, four families, 8–101 players, 30 or
100 repetitions, and budgets through 512. At the pinned public source, all 30
files exist for six precomputed DV/LE tasks; three large TreeSHAP task
constructors also execute in the frozen environment. The three TabPFN feature
importance and three YAHPO hyperparameter-importance tasks cannot execute
exactly because their author-required inputs/runtimes are absent.

Nine runnable tasks do not verify a 15-task claim.
"""
    )
    (ART3 / "EVAL.md").write_text(
        """# Claim 3 evaluation

Verdict: BLOCKED

The retained judged evidence remains one ImageNet/ViT-9 task, 30 repetitions,
and budgets 16–64. The task-availability audit does not create performance
measurements, and the one-task result is not promoted to the paper's 15-task,
budget-512 comparison. Six exact tasks remain inaccessible under the immutable
environment contract.
"""
    )
    print("=== CLAIMS_2_3_SCOPE ===")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
