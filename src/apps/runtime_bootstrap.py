#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHATBOX_MANIFEST_PATH = PROJECT_ROOT / "ai" / "runtime" / "chatbox" / "manifest.json"

_ORACLE_URL_ENV_KEYS = (
    "ORACLE_OBJECT_PREAUTH_URL",
    "OCI_OBJECT_PREAUTH_URL",
)
_DIRECT_URL_ENV_KEYS = (
    "QUAI_PACK_DIRECT_URL",
)


def _env_first(*keys: str) -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return ""


def _env_int(key: str, default: int = 0) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_chatbox_manifest() -> dict[str, Any]:
    return json.loads(CHATBOX_MANIFEST_PATH.read_text(encoding="utf-8"))


def resolve_bundle_entry(bundle_id: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_chatbox_manifest()
    bundles = manifest.get("bundles")
    if not isinstance(bundles, list):
        raise SystemExit("Chatbox manifest has no bundle list.")

    if not bundle_id:
        bundle_id = str(manifest.get("default_bundle") or "").strip()
    for bundle in bundles:
        if isinstance(bundle, dict) and str(bundle.get("bundle_id") or "").strip() == bundle_id:
            return manifest, copy.deepcopy(bundle)
    raise SystemExit(f"Bundle not found: {bundle_id}")


def resolve_pack_download(bundle: dict[str, Any]) -> dict[str, Any]:
    download = copy.deepcopy(bundle.get("pack_download") if isinstance(bundle.get("pack_download"), dict) else {})
    oracle_url = _env_first(*_ORACLE_URL_ENV_KEYS)
    direct_url = _env_first(*_DIRECT_URL_ENV_KEYS)
    override_url = oracle_url or direct_url

    if override_url:
        download["direct_url"] = override_url
        download["share_url"] = _env_first("QUAI_PACK_SHARE_URL") or override_url
        download["provider"] = (
            _env_first("QUAI_PACK_PROVIDER")
            or ("oracle_object_storage" if oracle_url else "external_http")
        )

    if not download.get("share_url") and download.get("direct_url"):
        download["share_url"] = str(download.get("direct_url"))

    filename = _env_first("QUAI_PACK_FILENAME")
    if filename:
        download["filename"] = filename
    elif not download.get("filename"):
        download["filename"] = Path(str(bundle.get("pack_path") or "pack.rftmwpk")).name

    size_bytes = _env_int("QUAI_PACK_SIZE_BYTES")
    if size_bytes > 0:
        download["size_bytes"] = size_bytes

    for field, env_key in (
        ("pack_file_sha256", "QUAI_PACK_FILE_SHA256"),
        ("gguf_sha256", "QUAI_PACK_GGUF_SHA256"),
        ("pack_provenance_sha256", "QUAI_PACK_PROVENANCE_SHA256"),
        ("meta_path", "QUAI_PACK_META_PATH"),
    ):
        value = _env_first(env_key)
        if value:
            download[field] = value

    return download


def build_effective_bundle(bundle_id: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, bundle = resolve_bundle_entry(bundle_id)
    bundle["pack_download"] = resolve_pack_download(bundle)
    return manifest, bundle


def bundle_target_path(bundle: dict[str, Any]) -> Path:
    pack_path = str(bundle.get("pack_path") or "").strip()
    return (PROJECT_ROOT / "ai" / "runtime" / "chatbox" / pack_path).resolve()


def bundle_meta_path(bundle: dict[str, Any]) -> Path:
    download = bundle.get("pack_download") if isinstance(bundle.get("pack_download"), dict) else {}
    meta_rel = str(download.get("meta_path") or "").strip()
    if meta_rel:
        return (PROJECT_ROOT / "ai" / "runtime" / "chatbox" / meta_rel).resolve()
    target = bundle_target_path(bundle)
    return target.with_suffix(target.suffix + ".meta.json")


def runtime_bootstrap_snapshot(bundle_id: str = "") -> dict[str, Any]:
    manifest, bundle = build_effective_bundle(bundle_id)
    download = bundle.get("pack_download") if isinstance(bundle.get("pack_download"), dict) else {}
    target_path = bundle_target_path(bundle)
    target_exists = target_path.exists()
    actual_size = target_path.stat().st_size if target_exists else 0
    expected_size = int(download.get("size_bytes", 0) or 0)

    return {
        "project": "QU.AI",
        "default_bundle": str(manifest.get("default_bundle") or "").strip(),
        "bundle": bundle,
        "paths": {
            "manifest": str(CHATBOX_MANIFEST_PATH),
            "pack": str(target_path),
            "meta": str(bundle_meta_path(bundle)),
        },
        "pack_status": {
            "exists": target_exists,
            "actual_size": actual_size,
            "expected_size": expected_size,
            "size_matches": (expected_size <= 0 or actual_size == expected_size),
        },
        "oracle_override_active": bool(_env_first(*_ORACLE_URL_ENV_KEYS)),
        "runtime_env": {
            "prefetch_default_pack": os.getenv("QUAI_PREFETCH_DEFAULT_PACK", "0"),
            "runtime_host": os.getenv("QUAI_RUNTIME_HOST", "0.0.0.0"),
            "runtime_port": os.getenv("QUAI_RUNTIME_PORT", "8787"),
        },
    }
