from __future__ import annotations

import json
import os
import struct
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


_PACK_MAGIC = b"RFTMWPK1"


@dataclass(frozen=True)
class RFTMWPackCandidate:
    path: str
    model_name: str
    runtime_model_id: str
    display_name: str
    state_dict_hash_sha256: str
    fp32_size_bytes: int
    modified_time: float
    source: str
    priority: int


def read_rftmw_pack_header(path: str | os.PathLike[str]) -> dict:
    pack_path = Path(path)
    with pack_path.open("rb") as f:
        magic = f.read(8)
        if magic != _PACK_MAGIC:
            raise ValueError(f"Invalid pack magic: {magic!r}")
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len).decode("utf-8"))
    return header if isinstance(header, dict) else {}


def _candidate_roots() -> List[Path]:
    roots: List[Path] = []

    exact_path = os.getenv("QUANTONIUM_RFTMW_CACHE_PATH", "").strip()
    if exact_path:
        roots.append(Path(exact_path))

    cache_dir = os.getenv("QUANTONIUM_RFTMW_CACHE_DIR", "").strip()
    if cache_dir:
        roots.append(Path(cache_dir))

    extra_roots = os.getenv("QUANTONIUM_RFTMW_DISCOVERY_ROOTS", "").strip()
    if extra_roots:
        for raw in extra_roots.split(os.pathsep):
            raw = raw.strip()
            if raw:
                roots.append(Path(raw))

    roots.extend(
        [
            Path("ai") / "cache" / "rftmw",
            Path("artifacts"),
        ]
    )

    deduped: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve(strict=False)).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _manifest_paths() -> List[Path]:
    paths = [
        Path("ai") / "runtime" / "chatbox" / "manifest.json",
        Path("artifacts") / "chatbox_rftmw_manifest.json",
    ]
    deduped: List[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve(strict=False)).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _discover_manifest_packs() -> List[RFTMWPackCandidate]:
    candidates: List[RFTMWPackCandidate] = []

    for manifest_path in _manifest_paths():
        if not manifest_path.exists() or not manifest_path.is_file():
            continue

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        bundles = manifest.get("bundles")
        if not isinstance(bundles, list):
            continue

        base_dir = manifest_path.parent
        default_bundle = str(manifest.get("default_bundle") or "").strip()
        for bundle in bundles:
            if not isinstance(bundle, dict):
                continue

            rel_pack_path = str(bundle.get("pack_path") or "").strip()
            rel_model_path = str(bundle.get("model_path") or "").strip()
            if not rel_pack_path or not rel_model_path:
                continue

            pack_path = (base_dir / rel_pack_path).resolve()
            model_path = (base_dir / rel_model_path).resolve()
            if not pack_path.exists() or not model_path.exists():
                continue

            model_name = str(bundle.get("model_name") or bundle.get("bundle_id") or model_path.name).strip()
            state_hash = str(bundle.get("state_dict_hash_sha256") or "").strip()
            fp32_size_bytes = int(bundle.get("fp32_size_bytes", 0) or 0)

            if not state_hash or not fp32_size_bytes:
                try:
                    header = read_rftmw_pack_header(pack_path)
                except Exception:
                    header = {}
                meta = header.get("meta") if isinstance(header.get("meta"), dict) else {}
                provenance = meta.get("provenance") if isinstance(meta.get("provenance"), dict) else {}
                if not state_hash:
                    state_hash = str(provenance.get("state_dict_hash_sha256") or "").strip()
                if not fp32_size_bytes:
                    fp32_size_bytes = int(meta.get("fp32_size_bytes", 0) or 0)
                if not model_name:
                    model_name = str(meta.get("model_name") or provenance.get("model_name") or model_path.name).strip()

            try:
                modified_time = pack_path.stat().st_mtime
            except Exception:
                modified_time = 0.0

            candidates.append(
                RFTMWPackCandidate(
                    path=str(pack_path),
                    model_name=model_name,
                    runtime_model_id=str(model_path),
                    display_name=model_name or model_path.name,
                    state_dict_hash_sha256=state_hash,
                    fp32_size_bytes=fp32_size_bytes,
                    modified_time=modified_time,
                    source=f"bundle_manifest:{manifest_path.name}",
                    priority=0 if str(bundle.get("bundle_id") or "").strip() == default_bundle else 1,
                )
            )

    return candidates


def _iter_pack_paths() -> Iterable[Path]:
    seen: set[str] = set()
    for root in _candidate_roots():
        if root.is_file():
            if root.suffix.lower() == ".rftmwpk":
                key = str(root.resolve(strict=False)).lower()
                if key not in seen:
                    seen.add(key)
                    yield root
            continue

        if not root.exists() or not root.is_dir():
            continue

        for pack_path in root.rglob("*.rftmwpk"):
            key = str(pack_path.resolve(strict=False)).lower()
            if key in seen:
                continue
            seen.add(key)
            yield pack_path


def discover_rftmw_packs(*, max_results: int = 8) -> List[RFTMWPackCandidate]:
    candidates: List[RFTMWPackCandidate] = _discover_manifest_packs()
    seen_paths = {str(Path(c.path).resolve(strict=False)).lower() for c in candidates}

    for pack_path in _iter_pack_paths():
        pack_key = str(Path(pack_path).resolve(strict=False)).lower()
        if pack_key in seen_paths:
            continue
        try:
            header = read_rftmw_pack_header(pack_path)
        except Exception:
            continue

        meta = header.get("meta") if isinstance(header.get("meta"), dict) else {}
        provenance = meta.get("provenance") if isinstance(meta.get("provenance"), dict) else {}
        model_name = str(meta.get("model_name") or provenance.get("model_name") or "").strip()
        state_hash = str(provenance.get("state_dict_hash_sha256") or "").strip()
        fp32_size_bytes = int(meta.get("fp32_size_bytes", 0) or 0)

        try:
            modified_time = pack_path.stat().st_mtime
        except Exception:
            modified_time = 0.0

        candidates.append(
            RFTMWPackCandidate(
                path=str(pack_path),
                model_name=model_name,
                runtime_model_id=model_name,
                display_name=model_name or pack_path.stem,
                state_dict_hash_sha256=state_hash,
                fp32_size_bytes=fp32_size_bytes,
                modified_time=modified_time,
                source="pack_scan",
                priority=10,
            )
        )
        seen_paths.add(pack_key)

    preferred_path = os.getenv("QUANTONIUM_RFTMW_CACHE_PATH", "").strip().lower()
    candidates.sort(
        key=lambda c: (
            c.priority,
            0 if c.source.startswith("bundle_manifest:") else 1,
            0 if preferred_path and str(c.path).strip().lower() == preferred_path else 1,
            -c.modified_time,
            c.path.lower(),
        )
    )
    return candidates[:max_results]


def default_bundle_manifest(*, bundle_dir: str | os.PathLike[str] = Path("ai") / "runtime" / "chatbox") -> dict:
    return {
        "format_version": 1,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bundles": [],
        "default_bundle": "",
        "bundle_root": str(Path(bundle_dir)),
        "note": "Commit this manifest alongside packs/ and models/ so qshll_chatbox can boot from repo-local RFTMW assets.",
    }
