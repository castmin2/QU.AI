#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Local-only runtime profiles for the quantonium chat stack.

These profiles encode the repo's intended routing guidance:

- RFTMW and Ollama are the only interactive backends exposed in the chatbox
- all inference remains local-only
- model families get conservative CPU-first compression defaults
- RoPE/GQA families are tagged with the repo's RFT KV-routing guidance
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime as _dt
import os
from pathlib import Path
from typing import Dict, List
import json

from src.apps.ai_model_wrapper import discover_local_models
from src.apps.ollama_client import discover_ollama_models


@dataclass(frozen=True)
class LocalRuntimeProfile:
    backend: str
    model_id: str
    profile_name: str
    architecture: str
    recommended_variant: str
    local_only: bool
    topological_space: bool
    weight_keep_ratio: float
    kv_keep_ratio: float
    entropy_threshold: float
    max_rft_elems: int
    max_new_tokens: int
    note: str

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _normalized(model_id: str) -> str:
    return (model_id or "").strip().lower()


def _parse_param_hint_billions(model_id: str) -> float:
    text = _normalized(model_id)
    for token in ("0.5b", "1b", "1.1b", "1.3b", "2b", "2.7b", "3b", "6b", "7b", "8b", "12b", "14b", "24b", "27b", "30b", "32b"):
        if token in text:
            try:
                return float(token.replace("b", ""))
            except ValueError:
                continue
    return 0.0


def profile_for_model(model_id: str, backend: str) -> LocalRuntimeProfile:
    model = _normalized(model_id)
    backend = _normalized(backend)
    size_b = _parse_param_hint_billions(model_id)

    weight_keep_ratio = 0.30
    kv_keep_ratio = 0.30
    entropy_threshold = 0.40
    max_rft_elems = 2_000_000
    max_new_tokens = 120
    architecture = "unknown"
    recommended_variant = "op_rft_golden"
    profile_name = "local_cpu_baseline"
    note = "Local-only baseline profile."

    if any(name in model for name in ("gpt2", "dialogpt", "distilgpt2", "gpt-neo", "gptneo", "gpt-j", "gptj")):
        architecture = "absolute_mha"
        recommended_variant = "op_rft_golden"
        kv_keep_ratio = 0.40
        profile_name = "absolute_mha_intelligence_dividend"
        note = (
            "Absolute-position models are routed to the golden profile. "
            "Repo benchmarks report positive perplexity deltas at 40% retention."
        )
    elif any(name in model for name in ("qwen", "gemma", "llama", "mistral", "tinyllama", "phi")):
        architecture = "rope_gqa_or_rope_like"
        recommended_variant = "pat_rft_rope_pure"
        kv_keep_ratio = 0.40
        profile_name = "rope_gqa_kv_route"
        note = (
            "RoPE/GQA-like families are routed to the rope-pure KV profile. "
            "Repo guidance favors this path around 40% retention, with geometric fallback below 30%."
        )
        if size_b >= 7.0:
            kv_keep_ratio = 0.20
            weight_keep_ratio = 0.25
            max_new_tokens = 96
            profile_name = "rope_gqa_cpu_constrained"
            recommended_variant = "op_rft_geometric"
            note = (
                "Large RoPE-family model on a CPU-first path: use more aggressive retention "
                "to reduce RAM and startup pressure."
            )

    if backend == "ollama":
        # Ollama already serves quantized GGUF locally; keep the note and topology flags,
        # but do not imply weight repacking is active inside the Ollama runtime.
        return LocalRuntimeProfile(
            backend=backend,
            model_id=model_id,
            profile_name=f"ollama_{profile_name}",
            architecture=architecture,
            recommended_variant=recommended_variant,
            local_only=True,
            topological_space=True,
            weight_keep_ratio=weight_keep_ratio,
            kv_keep_ratio=kv_keep_ratio,
            entropy_threshold=entropy_threshold,
            max_rft_elems=max_rft_elems,
            max_new_tokens=max_new_tokens,
            note=(
                note
                + " Ollama keeps local GGUF weights on-device; this profile records the RFT/KV intent "
                + "for retrieval and future proofs-side compression work."
            ),
        )

    return LocalRuntimeProfile(
        backend=backend or "rftmw",
        model_id=model_id,
        profile_name=profile_name,
        architecture=architecture,
        recommended_variant=recommended_variant,
        local_only=True,
        topological_space=True,
        weight_keep_ratio=weight_keep_ratio,
        kv_keep_ratio=kv_keep_ratio,
        entropy_threshold=entropy_threshold,
        max_rft_elems=max_rft_elems,
        max_new_tokens=max_new_tokens,
        note=note,
    )


def apply_profile_env(profile: LocalRuntimeProfile) -> None:
    os.environ["QUANTONIUM_LOCAL_ONLY"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["QUANTONIUM_TOPOLOGICAL_SPACE"] = "1" if profile.topological_space else "0"
    os.environ["QUANTONIUM_RFTMW_KEEP_RATIO"] = f"{profile.weight_keep_ratio:.4f}"
    os.environ["QUANTONIUM_RFTMW_KV_KEEP_RATIO"] = f"{profile.kv_keep_ratio:.4f}"
    os.environ["QUANTONIUM_RFTMW_ENTROPY_THRESHOLD"] = f"{profile.entropy_threshold:.4f}"
    os.environ["QUANTONIUM_RFTMW_MAX_RFT_ELEMS"] = str(profile.max_rft_elems)
    os.environ["QUANTONIUM_RFTMW_MAX_NEW_TOKENS"] = str(profile.max_new_tokens)
    os.environ["QUANTONIUM_RFTMW_PROFILE_VARIANT"] = profile.recommended_variant
    os.environ["QUANTONIUM_LOCAL_PROFILE_NAME"] = profile.profile_name


def local_runtime_manifest() -> Dict[str, object]:
    ollama_profiles = [
        profile_for_model(candidate.model_id, "ollama").as_dict()
        for candidate in discover_ollama_models(include_embeddings=True)
    ]
    rftmw_profiles = [
        profile_for_model(candidate.model_id, "rftmw").as_dict()
        for candidate in discover_local_models()
    ]
    return {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy": {
            "interactive_backends": ["rftmw", "ollama"],
            "local_only": True,
            "topological_space_default": True,
            "public_repo_push_allowed": False,
            "note": (
                "This manifest is intended as local provenance and should be mirrored into the "
                "private proofs workspace if that repo is available."
            ),
        },
        "ollama_models": ollama_profiles,
        "rftmw_source_models": rftmw_profiles,
    }


def write_local_runtime_manifest(path: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(local_runtime_manifest(), indent=2), encoding="utf-8")
    return out
