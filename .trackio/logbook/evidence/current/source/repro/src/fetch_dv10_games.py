"""Fetch the three exact public 10-player data-valuation task suites."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


COMMIT = "799cfd0f2c32f17446130247a7ac3519e68cce82"
TREE_API = f"https://api.github.com/repos/mmschlk/shapiq/git/trees/{COMMIT}?recursive=1"
RAW_BASE = f"https://raw.githubusercontent.com/mmschlk/shapiq/{COMMIT}/"
USER_AGENT = "OpenResearch-Reproduction/1.0"

TASKS = {
    "dv_rf_bike_sharing": {
        "prefix": "data/precomputed_games/BikeSharing_DatasetValuation_Game/10/",
        "filename_prefix": "model_name=random_forest_",
        "application": "Bike Sharing data valuation / random forest",
    },
    "dv_gb_bike_sharing": {
        "prefix": "data/precomputed_games/BikeSharing_DatasetValuation_Game/10/",
        "filename_prefix": "model_name=gradient_boosting_",
        "application": "Bike Sharing data valuation / gradient boosting",
    },
    "dv_gb_california_housing": {
        "prefix": "data/precomputed_games/CaliforniaHousing_DatasetValuation_Game/10/",
        "filename_prefix": "model_name=gradient_boosting_",
        "application": "California Housing data valuation / gradient boosting",
    },
}


def _get(url: str) -> bytes:
    return urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=120).read()


def fetch_all(root: Path) -> dict[str, list[dict]]:
    tree_payload = _get(TREE_API)
    tree = json.loads(tree_payload)
    if tree.get("truncated"):
        raise RuntimeError("pinned GitHub tree response is truncated")
    blob_paths = [item["path"] for item in tree["tree"] if item["type"] == "blob"]
    manifest: dict[str, list[dict]] = {}
    for task_id, spec in TASKS.items():
        selected = sorted(
            (
                path
                for path in blob_paths
                if path.startswith(spec["prefix"])
                and Path(path).name.startswith(spec["filename_prefix"])
                and path.endswith(".npz")
            ),
            key=lambda path: int(Path(path).stem.rsplit("_", 1)[-1]),
        )
        if len(selected) != 30:
            raise AssertionError(f"{task_id}: expected 30 pinned files, got {len(selected)}")
        task_dir = root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for source_path in selected:
            destination = task_dir / Path(source_path).name
            payload = destination.read_bytes() if destination.exists() else _get(
                RAW_BASE + quote(source_path, safe="/=")
            )
            digest = hashlib.sha256(payload).hexdigest()
            if not destination.exists():
                destination.write_bytes(payload)
            records.append(
                {
                    "source_path": source_path,
                    "local_path": str(destination),
                    "sha256": digest,
                    "bytes": len(payload),
                }
            )
        manifest[task_id] = records
    return {
        "source_commit": COMMIT,
        "tree_api": TREE_API,
        "tree_response_sha256": hashlib.sha256(tree_payload).hexdigest(),
        "tasks": manifest,
    }

