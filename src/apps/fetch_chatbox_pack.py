#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_manifest() -> dict:
    manifest_path = PROJECT_ROOT / "ai" / "runtime" / "chatbox" / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _bundle_entry(bundle_id: str) -> dict:
    manifest = _load_manifest()
    bundles = manifest.get("bundles")
    if not isinstance(bundles, list):
        raise SystemExit("Chatbox manifest has no bundle list.")

    if not bundle_id:
        bundle_id = str(manifest.get("default_bundle") or "").strip()
    for bundle in bundles:
        if isinstance(bundle, dict) and str(bundle.get("bundle_id") or "") == bundle_id:
            return bundle
    raise SystemExit(f"Bundle not found: {bundle_id}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch an externally hosted chatbox .rftmwpk into ai/runtime/chatbox/packs/")
    ap.add_argument("--bundle-id", default="", help="Bundle id from ai/runtime/chatbox/manifest.json")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing local pack")
    args = ap.parse_args()

    bundle = _bundle_entry(args.bundle_id)
    download = bundle.get("pack_download") if isinstance(bundle.get("pack_download"), dict) else {}
    direct_url = str(download.get("direct_url") or "").strip()
    filename = str(download.get("filename") or Path(str(bundle.get("pack_path") or "")).name).strip()
    expected_sha = str(download.get("pack_provenance_sha256") or "").strip().lower()

    if not direct_url:
        raise SystemExit("No external pack download URL is recorded for this bundle.")

    target_path = PROJECT_ROOT / "ai" / "runtime" / "chatbox" / "packs" / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not args.force:
        print(f"Pack already exists: {target_path}")
        if expected_sha:
            print(f"Current SHA-256: {_sha256(target_path)}")
        return

    print(f"Downloading: {direct_url}")
    print(f"Target: {target_path}")
    urllib.request.urlretrieve(direct_url, target_path)

    if expected_sha:
        actual_sha = _sha256(target_path).lower()
        print(f"SHA-256: {actual_sha}")
        if actual_sha != expected_sha:
            raise SystemExit(
                "Downloaded pack hash does not match manifest provenance hash. "
                f"expected={expected_sha} actual={actual_sha}"
            )

    print("Pack ready.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
