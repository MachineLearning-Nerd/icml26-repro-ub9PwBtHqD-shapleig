"""Validate the additive, text-only Hugging Face Space release candidate."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / ".trackio" / "logbook"
PROTECTED_MANIFEST = (
    ROOT / ".openresearch" / "protected" / "judged_space_85ca787_manifest.sha256"
)
PUBLISHED_MANIFEST = (
    ROOT / ".openresearch" / "protected" / "published_space_969796f_manifest.sha256"
)
RELEASE = ROOT / ".openresearch" / "release"
ALLOWLIST = RELEASE / "hf_upload_allowlist.txt"
UPLOAD_MANIFEST = RELEASE / "hf_upload_manifest.sha256"
SUBSET_CHECK = RELEASE / "space_subset_check.json"
JUDGED_REVISION = "85ca787e52cd4ba933883116d010d919bfe54fe7"
PUBLISHED_REVISION = "969796fae0abc9817bb4fe1c584db110afceb77b"
MUTABLE_ENTRYPOINTS = {"README.md", "logbook.json"}
SECRET_PATTERN = re.compile(
    rb"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|"
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    rb"(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"][^'\"]{8,})",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        manifest[relative.removeprefix("./")] = digest
    return manifest


def main() -> None:
    protected = read_manifest(PROTECTED_MANIFEST)
    published = read_manifest(PUBLISHED_MANIFEST)
    candidate_paths = {
        path.relative_to(CANDIDATE).as_posix(): path
        for path in CANDIDATE.rglob("*")
        if path.is_file()
    }
    old_paths = set(protected)
    missing = sorted(old_paths - set(candidate_paths))
    changed_old = sorted(
        relative
        for relative, expected in protected.items()
        if relative not in MUTABLE_ENTRYPOINTS
        and relative in candidate_paths
        and sha256(candidate_paths[relative]) != expected
    )
    missing_published = sorted(set(published) - set(candidate_paths))
    changed_published = sorted(
        relative
        for relative, expected in published.items()
        if relative not in MUTABLE_ENTRYPOINTS
        and relative in candidate_paths
        and sha256(candidate_paths[relative]) != expected
    )

    logbook = json.loads(candidate_paths["logbook.json"].read_text())

    def referenced_files(node: dict) -> set[str]:
        found = {node["file"]}
        for child in node.get("children", []):
            found |= referenced_files(child)
        return found

    historical_pages = {
        relative for relative in old_paths if relative.startswith("pages/")
    }
    missing_historical_references = sorted(
        historical_pages - referenced_files(logbook["root"])
    )

    allowlist = [
        line.strip() for line in ALLOWLIST.read_text().splitlines() if line.strip()
    ]
    allowlist_set = set(allowlist)
    unsafe_allowlist = sorted(
        relative
        for relative in allowlist
        if relative.startswith("/") or ".." in Path(relative).parts
    )
    expected_uploads = {
        relative
        for relative, path in candidate_paths.items()
        if relative not in published or sha256(path) != published[relative]
    }
    allowlist_mismatch = sorted(expected_uploads ^ allowlist_set)
    utf8_errors, nul_files, json_errors, secret_files = [], [], [], []
    manifest_lines = []
    for relative in allowlist:
        path = candidate_paths.get(relative)
        if path is None:
            utf8_errors.append(f"missing:{relative}")
            continue
        payload = path.read_bytes()
        if b"\0" in payload:
            nul_files.append(relative)
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            utf8_errors.append(relative)
            continue
        if relative.endswith(".json"):
            try:
                json.loads(text)
            except json.JSONDecodeError:
                json_errors.append(relative)
        if SECRET_PATTERN.search(payload):
            secret_files.append(relative)
        manifest_lines.append(f"{hashlib.sha256(payload).hexdigest()}  {relative}")
    UPLOAD_MANIFEST.write_text("\n".join(manifest_lines) + "\n")

    result = {
        "judged_revision": JUDGED_REVISION,
        "old_path_count": len(old_paths),
        "candidate_path_count": len(candidate_paths),
        "old_file_set_is_subset": not missing,
        "missing_old_paths": missing,
        "unchanged_old_paths_except_mutable_entrypoints": not changed_old,
        "changed_old_paths_except_mutable_entrypoints": changed_old,
        "published_revision": PUBLISHED_REVISION,
        "published_path_count": len(published),
        "published_file_set_is_subset": not missing_published,
        "missing_published_paths": missing_published,
        "unchanged_published_paths_except_mutable_entrypoints": not changed_published,
        "changed_published_paths_except_mutable_entrypoints": changed_published,
        "logbook_json_modified_additively": not missing_historical_references,
        "missing_historical_page_references": missing_historical_references,
        "upload_path_count": len(allowlist),
        "upload_allowlist_matches_changed_text_set": not allowlist_mismatch,
        "upload_allowlist_mismatch": allowlist_mismatch,
        "unsafe_allowlist_paths": unsafe_allowlist,
        "all_uploads_utf8": not utf8_errors,
        "utf8_errors": utf8_errors,
        "nul_files": nul_files,
        "all_json_valid": not json_errors,
        "json_errors": json_errors,
        "secret_pattern_file_count": len(secret_files),
        "secret_pattern_files": secret_files,
    }
    SUBSET_CHECK.write_text(json.dumps(result, indent=2) + "\n")
    failures = [
        missing,
        changed_old,
        missing_published,
        changed_published,
        missing_historical_references,
        allowlist_mismatch,
        unsafe_allowlist,
        utf8_errors,
        nul_files,
        json_errors,
        secret_files,
    ]
    print("=== RELEASE_CANDIDATE_VALIDATION ===")
    print(json.dumps(result, indent=2))
    if any(failures):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
