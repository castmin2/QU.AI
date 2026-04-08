#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.apps.rftmw_pack_discovery import default_bundle_manifest, read_rftmw_pack_header


MODEL_METADATA_FILENAMES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "added_tokens.json",
    "preprocessor_config.json",
}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", (text or "").strip()).strip("-._")
    return slug or "rftmw-chatbox-bundle"


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return default_bundle_manifest(bundle_dir=path.parent)
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_model_metadata(model_dir: Path, destination_dir: Path) -> list[str]:
    copied: list[str] = []
    destination_dir.mkdir(parents=True, exist_ok=True)

    for src in model_dir.rglob("*"):
        if not src.is_file():
            continue
        if src.name not in MODEL_METADATA_FILENAMES:
            continue
        rel = src.relative_to(model_dir)
        dest = destination_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(str(rel).replace("\\", "/"))

    return sorted(copied)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage a repo-portable qshll_chatbox RFTMW bundle (pack + tokenizer/config metadata)."
    )
    ap.add_argument("--pack", required=True, help="Path to the .rftmwpk file")
    ap.add_argument(
        "--model-dir",
        required=True,
        help="Local model directory containing config/tokenizer files (no base weights are copied)",
    )
    ap.add_argument("--bundle-dir", default=str(Path("ai") / "runtime" / "chatbox"))
    ap.add_argument("--bundle-id", default="", help="Optional stable bundle id")
    ap.add_argument("--set-default", action="store_true", help="Mark this bundle as the default manifest entry")
    ap.add_argument("--force", action="store_true", help="Overwrite existing pack/model metadata destination")
    args = ap.parse_args()

    pack_path = Path(args.pack).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    bundle_dir = Path(args.bundle_dir).resolve()
    manifest_path = bundle_dir / "manifest.json"

    if not pack_path.exists() or not pack_path.is_file():
        raise SystemExit(f"--pack not found: {pack_path}")
    if not model_dir.exists() or not model_dir.is_dir():
        raise SystemExit(f"--model-dir not found: {model_dir}")

    header = read_rftmw_pack_header(pack_path)
    meta = header.get("meta") if isinstance(header.get("meta"), dict) else {}
    provenance = meta.get("provenance") if isinstance(meta.get("provenance"), dict) else {}
    model_name = str(meta.get("model_name") or provenance.get("model_name") or model_dir.name).strip()
    state_hash = str(provenance.get("state_dict_hash_sha256") or "").strip()
    fp32_size_bytes = int(meta.get("fp32_size_bytes", 0) or 0)

    bundle_id = args.bundle_id.strip() or _slugify(model_name or pack_path.stem)
    packs_dir = bundle_dir / "packs"
    models_dir = bundle_dir / "models"
    target_pack_path = packs_dir / f"{bundle_id}.rftmwpk"
    target_model_dir = models_dir / bundle_id

    packs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if target_pack_path.exists() and not args.force:
        raise SystemExit(f"Target pack already exists: {target_pack_path} (use --force to overwrite)")
    if target_model_dir.exists() and any(target_model_dir.iterdir()) and not args.force:
        raise SystemExit(f"Target model metadata dir already exists: {target_model_dir} (use --force to overwrite)")

    if args.force and target_model_dir.exists():
        shutil.rmtree(target_model_dir, ignore_errors=True)

    shutil.copy2(pack_path, target_pack_path)
    copied_files = _copy_model_metadata(model_dir, target_model_dir)
    if not copied_files:
        raise SystemExit(
            "No tokenizer/config files were copied from --model-dir. "
            "Expected files like config.json, tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, or tokenizer.model."
        )

    manifest = _load_manifest(manifest_path)
    bundles = manifest.get("bundles")
    if not isinstance(bundles, list):
        bundles = []

    rel_pack_path = str(target_pack_path.relative_to(bundle_dir)).replace("\\", "/")
    rel_model_path = str(target_model_dir.relative_to(bundle_dir)).replace("\\", "/")

    bundle_entry = {
        "bundle_id": bundle_id,
        "model_name": model_name,
        "pack_path": rel_pack_path,
        "model_path": rel_model_path,
        "state_dict_hash_sha256": state_hash,
        "fp32_size_bytes": fp32_size_bytes,
        "copied_model_files": copied_files,
    }

    bundles = [b for b in bundles if not isinstance(b, dict) or str(b.get("bundle_id") or "") != bundle_id]
    bundles.append(bundle_entry)
    bundles.sort(key=lambda b: str(b.get("bundle_id") or "").lower())

    manifest["format_version"] = 1
    manifest["bundle_root"] = str(bundle_dir)
    manifest["bundles"] = bundles
    if args.set_default or not str(manifest.get("default_bundle") or "").strip():
        manifest["default_bundle"] = bundle_id

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Bundle staged: {bundle_id}")
    print(f"Manifest: {manifest_path}")
    print(f"Pack: {target_pack_path}")
    print(f"Model metadata dir: {target_model_dir}")
    print(f"Copied files: {len(copied_files)}")
    print("")
    print("Run chatbox with the compressed bundle:")
    print("  qshellchatbox.ps1")
    print("")
    print("Recommended committed paths:")
    print(f"  {rel_pack_path}")
    print(f"  {rel_model_path}")


if __name__ == "__main__":
    main()
