#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accelerate import init_empty_weights
from gguf import GGUFReader, dequantize
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_gguf_pytorch_utils import (
    TENSOR_PROCESSORS,
    TensorProcessor,
    get_gguf_hf_weights_map,
    load_gguf_checkpoint,
)

from quantonium_os_src.engine.rftmw_memory import RFTMWMemoryLayer
from src.apps.topological_chat_space import build_topological_chat_context
from src.apps.rftmw_pack_discovery import default_bundle_manifest, read_rftmw_pack_header


def _slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in text.strip()).strip("-._") or "ollama-rftmw-bundle"


def _ollama_manifest_path(model_tag: str) -> Path:
    parts = [p for p in model_tag.strip().split(":") if p]
    if len(parts) != 2:
        raise SystemExit(f"Expected Ollama model tag like 'gemma3:1b' or 'qwen2.5-coder:3b', got: {model_tag!r}")
    family, variant = parts
    return Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library" / family / variant


def _resolve_ollama_blob(model_tag: str) -> Path:
    manifest_path = _ollama_manifest_path(model_tag)
    if not manifest_path.exists():
        raise SystemExit(f"Ollama manifest not found for {model_tag}: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for layer in manifest.get("layers", []):
        if str(layer.get("mediaType") or "") == "application/vnd.ollama.image.model":
            digest = str(layer.get("digest") or "").strip()
            if digest.startswith("sha256:"):
                blob_name = digest.replace("sha256:", "sha256-")
                blob_path = Path.home() / ".ollama" / "models" / "blobs" / blob_name
                if blob_path.exists():
                    return blob_path
    raise SystemExit(f"Could not resolve model blob for {model_tag} from {manifest_path}")


def _write_manifest(
    *,
    bundle_dir: Path,
    bundle_id: str,
    model_name: str,
    target_pack_path: Path,
    target_model_dir: Path,
    header: dict,
) -> Path:
    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = default_bundle_manifest(bundle_dir=bundle_dir)

    bundles = manifest.get("bundles")
    if not isinstance(bundles, list):
        bundles = []

    meta = header.get("meta") if isinstance(header.get("meta"), dict) else {}
    provenance = meta.get("provenance") if isinstance(meta.get("provenance"), dict) else {}

    rel_pack_path = str(target_pack_path.relative_to(bundle_dir)).replace("\\", "/")
    rel_model_path = str(target_model_dir.relative_to(bundle_dir)).replace("\\", "/")
    entry = {
        "bundle_id": bundle_id,
        "model_name": model_name,
        "pack_path": rel_pack_path,
        "model_path": rel_model_path,
        "state_dict_hash_sha256": str(provenance.get("state_dict_hash_sha256") or ""),
        "fp32_size_bytes": int(meta.get("fp32_size_bytes", 0) or 0),
        "copied_model_files": sorted(str(p.relative_to(target_model_dir)).replace("\\", "/") for p in target_model_dir.rglob("*") if p.is_file()),
    }
    topo_meta = meta.get("topological_storage") if isinstance(meta.get("topological_storage"), dict) else {}
    if topo_meta:
        entry["topological_storage"] = topo_meta

    bundles = [b for b in bundles if not isinstance(b, dict) or str(b.get("bundle_id") or "") != bundle_id]
    bundles.append(entry)
    bundles.sort(key=lambda b: str(b.get("bundle_id") or "").lower())

    manifest["format_version"] = 1
    manifest["bundle_root"] = str(bundle_dir)
    manifest["bundles"] = bundles
    manifest["default_bundle"] = bundle_id
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _build_topological_storage_metadata(bundle_id: str, model_tag: str, state_hash: str) -> dict:
    token = f"{model_tag}|{state_hash[:32]}"
    context = build_topological_chat_context(token)
    return {
        "storage_class": "TopologicalChatContext",
        "bundle_id": bundle_id,
        "prompt_sha256": context.prompt_sha256,
        "surface_type": context.surface_type,
        "nu": int(context.nu),
        "nv": int(context.nv),
        "vertex_count": int(context.vertex_count),
        "edge_count": int(context.edge_count),
        "face_count": int(context.face_count),
        "euler_characteristic": int(context.euler_characteristic),
        "orientable": bool(context.orientable),
        "genus": context.genus,
        "crosscap_number": context.crosscap_number,
        "scalar_holonomy": {
            "real": float(np.real(context.scalar_holonomy)),
            "imag": float(np.imag(context.scalar_holonomy)),
            "abs": float(abs(context.scalar_holonomy)),
            "arg": float(np.angle(context.scalar_holonomy)),
        },
        "su2_trace_holonomy": {
            "real": float(np.real(context.su2_trace_holonomy)),
            "imag": float(np.imag(context.su2_trace_holonomy)),
            "abs": float(abs(context.su2_trace_holonomy)),
            "arg": float(np.angle(context.su2_trace_holonomy)),
        },
        "quasi_periodic_phase_response": {
            "real": float(np.real(context.quasi_periodic_phase_response)),
            "imag": float(np.imag(context.quasi_periodic_phase_response)),
            "abs": float(abs(context.quasi_periodic_phase_response)),
            "arg": float(np.angle(context.quasi_periodic_phase_response)),
        },
        "loop_interference": float(context.loop_interference),
        "laplacian_spectrum_min": float(context.laplacian_spectrum_min),
        "laplacian_spectrum_max": float(context.laplacian_spectrum_max),
        "connection_spectrum_min": float(context.connection_spectrum_min),
        "connection_spectrum_max": float(context.connection_spectrum_max),
    }


def _save_model_metadata(model_root: Path, gguf_name: str, target_model_dir: Path) -> AutoConfig:
    cfg = AutoConfig.from_pretrained(str(model_root), gguf_file=gguf_name)
    tok = AutoTokenizer.from_pretrained(str(model_root), gguf_file=gguf_name)
    tok.save_pretrained(target_model_dir)
    cfg.save_pretrained(target_model_dir)
    try:
        from transformers import GenerationConfig

        gen_cfg = GenerationConfig.from_model_config(cfg)
        gen_cfg.save_pretrained(target_model_dir)
    except Exception:
        pass
    return cfg


def _iter_mapped_gguf_tensors(gguf_path: Path, cfg: AutoConfig):
    parsed_meta = load_gguf_checkpoint(str(gguf_path), return_tensors=False)
    model_type = str(parsed_meta.get("config", {}).get("model_type") or getattr(cfg, "model_type", "") or "").strip()
    processor_cls = TENSOR_PROCESSORS.get(model_type, TensorProcessor)
    processor = processor_cls(config=parsed_meta.get("config", {}))

    with init_empty_weights():
        meta_model = AutoModelForCausalLM.from_config(cfg)
    tensor_key_mapping = get_gguf_hf_weights_map(meta_model, processor, model_type=model_type)

    reader = GGUFReader(str(gguf_path))
    parsed_parameters = {"config": parsed_meta.get("config", {}), "tensors": {}}

    yielded: set[str] = set()
    for tensor in reader.tensors:
        weights = dequantize(tensor.data, tensor.tensor_type)
        result = processor.process(
            weights=weights,
            name=tensor.name,
            tensor_key_mapping=tensor_key_mapping,
            parsed_parameters=parsed_parameters,
        )
        name = result.name
        if not name or name not in tensor_key_mapping:
            continue
        hf_name = tensor_key_mapping[name]
        yielded.add(hf_name)
        yield hf_name, np.asarray(result.weights)

    pending_tensors = parsed_parameters.get("tensors", {})
    if isinstance(pending_tensors, dict):
        for name, tensor in pending_tensors.items():
            if name in yielded:
                continue
            if hasattr(tensor, "detach"):
                arr = tensor.detach().cpu().numpy()
            else:
                arr = np.asarray(tensor)
            yield name, np.asarray(arr)


def _build_pack_from_gguf(
    *,
    gguf_path: Path,
    cfg: AutoConfig,
    target_pack_path: Path,
    model_tag: str,
    entropy_threshold: float,
    keep_ratio: float,
    max_rft_error: float,
    max_rft_elems: int,
    layer_limit: int | None,
) -> None:
    memory = RFTMWMemoryLayer(
        entropy_threshold=entropy_threshold,
        weight_keep_ratio=keep_ratio,
        kv_keep_ratio=min(keep_ratio, 0.40),
        max_rft_error=max_rft_error,
        max_rft_elements=max_rft_elems,
    )

    state_hash = hashlib.sha256()
    parameter_count = 0
    parameter_shapes: dict[str, list[int]] = {}
    fp32_size_bytes = 0
    tensors_to_ingest: list[tuple[str, np.ndarray]] = []

    for idx, (name, arr) in enumerate(_iter_mapped_gguf_tensors(gguf_path, cfg)):
        if layer_limit is not None and idx >= layer_limit:
            break
        arr32 = np.asarray(arr, dtype=np.float32)
        tensors_to_ingest.append((name, arr32))
        state_hash.update(name.encode("utf-8"))
        state_hash.update(arr32.tobytes())
        parameter_count += int(arr32.size)
        parameter_shapes[name] = list(arr32.shape)
        fp32_size_bytes += int(arr32.nbytes)

    memory.ingest_named_tensors(tensors_to_ingest, layer_limit=layer_limit, verbose=True)
    provenance = {
        "model_name": model_tag,
        "state_dict_hash_sha256": state_hash.hexdigest(),
        "parameter_count": parameter_count,
        "parameter_shapes": parameter_shapes,
        "fp32_size_bytes": fp32_size_bytes,
        "gguf_source_path": str(gguf_path),
        "build_path": "direct_gguf_tensor_ingest",
        "three_distance_router": True,
    }
    extra_meta = {
        "model_name": model_tag,
        "fp32_size_bytes": fp32_size_bytes,
        "provenance": provenance,
        "topological_storage": _build_topological_storage_metadata(
            target_pack_path.stem,
            model_tag,
            provenance["state_dict_hash_sha256"],
        ),
    }
    memory.save_pack(str(target_pack_path), extra_meta=extra_meta)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a repo-local RFTMW chatbox bundle from an installed Ollama GGUF blob.")
    ap.add_argument("--ollama-model", required=True, help="Installed Ollama model tag, for example gemma3:1b or qwen2.5-coder:3b")
    ap.add_argument("--blob", default="", help="Optional exact GGUF blob path. If omitted, resolve it from the local Ollama manifest.")
    ap.add_argument("--bundle-dir", default=str(Path("ai") / "runtime" / "chatbox"))
    ap.add_argument("--bundle-id", default="", help="Optional stable bundle id")
    ap.add_argument("--entropy-threshold", type=float, default=0.40)
    ap.add_argument("--keep-ratio", type=float, default=0.30)
    ap.add_argument("--max-rft-error", type=float, default=0.08)
    ap.add_argument("--max-rft-elems", type=int, default=2_000_000)
    ap.add_argument("--layer-limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    model_tag = args.ollama_model.strip()
    blob_path = Path(args.blob).expanduser().resolve() if args.blob.strip() else _resolve_ollama_blob(model_tag)
    if not blob_path.exists():
        raise SystemExit(f"GGUF blob not found: {blob_path}")

    bundle_dir = (PROJECT_ROOT / args.bundle_dir).resolve()
    bundle_id = args.bundle_id.strip() or _slugify(model_tag)
    packs_dir = bundle_dir / "packs"
    models_dir = bundle_dir / "models"
    target_pack_path = packs_dir / f"{bundle_id}.rftmwpk"
    target_model_dir = models_dir / bundle_id

    packs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if target_pack_path.exists():
        if not args.force:
            raise SystemExit(f"Target pack already exists: {target_pack_path} (use --force to overwrite)")
        target_pack_path.unlink()
    if target_model_dir.exists() and args.force:
        shutil.rmtree(target_model_dir, ignore_errors=True)
    target_model_dir.mkdir(parents=True, exist_ok=True)

    temp_root = PROJECT_ROOT / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{bundle_id}_gguf_", dir=str(temp_root), ignore_cleanup_errors=True) as tmpdir:
        tmp_dir = Path(tmpdir)
        gguf_name = "model.gguf"
        local_gguf_path = tmp_dir / gguf_name
        print(f"Copying local GGUF blob to workspace temp: {blob_path}")
        shutil.copy2(blob_path, local_gguf_path)
        cfg = _save_model_metadata(tmp_dir, gguf_name, target_model_dir)
        _build_pack_from_gguf(
            gguf_path=local_gguf_path,
            cfg=cfg,
            target_pack_path=target_pack_path,
            model_tag=model_tag,
            entropy_threshold=args.entropy_threshold,
            keep_ratio=args.keep_ratio,
            max_rft_error=args.max_rft_error,
            max_rft_elems=args.max_rft_elems,
            layer_limit=args.layer_limit,
        )

    header = read_rftmw_pack_header(target_pack_path)
    model_name = args.ollama_model
    manifest_path = _write_manifest(
        bundle_dir=bundle_dir,
        bundle_id=bundle_id,
        model_name=model_name,
        target_pack_path=target_pack_path,
        target_model_dir=target_model_dir,
        header=header,
    )

    print("")
    print(f"Bundle ready: {bundle_id}")
    print(f"Pack: {target_pack_path}")
    print(f"Model metadata: {target_model_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
