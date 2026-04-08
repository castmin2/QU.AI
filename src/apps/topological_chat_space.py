#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Whitebox topological context adapter for the chatbox.

This module keeps the chat path grounded in the repo's mesh/RFT stack by:

- deterministically embedding prompt bytes into a phi-structured state vector
- attaching that state to the audited torus mesh used by the topology tests
- exposing measurable observables from TopologicalGraphKernel

It does not claim quantum hardware behavior. The output is classical,
deterministic, and suitable for logging or prompt augmentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Dict

import numpy as np

from algorithms.rft.quantum_inspired.surface_topology import PHI
from algorithms.rft.quantum_inspired.topological_graph_kernel import TopologicalGraphKernel


@dataclass(frozen=True)
class TopologicalChatContext:
    prompt_sha256: str
    surface_type: str
    nu: int
    nv: int
    vertex_count: int
    edge_count: int
    face_count: int
    euler_characteristic: int
    orientable: bool
    genus: int | None
    crosscap_number: int | None
    scalar_holonomy: complex
    su2_trace_holonomy: complex
    quasi_periodic_phase_response: complex
    loop_interference: float
    laplacian_spectrum_min: float
    laplacian_spectrum_max: float
    connection_spectrum_min: float
    connection_spectrum_max: float

    def as_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["scalar_holonomy"] = {
            "real": float(np.real(self.scalar_holonomy)),
            "imag": float(np.imag(self.scalar_holonomy)),
            "abs": float(abs(self.scalar_holonomy)),
            "arg": float(np.angle(self.scalar_holonomy)),
        }
        data["su2_trace_holonomy"] = {
            "real": float(np.real(self.su2_trace_holonomy)),
            "imag": float(np.imag(self.su2_trace_holonomy)),
            "abs": float(abs(self.su2_trace_holonomy)),
            "arg": float(np.angle(self.su2_trace_holonomy)),
        }
        data["quasi_periodic_phase_response"] = {
            "real": float(np.real(self.quasi_periodic_phase_response)),
            "imag": float(np.imag(self.quasi_periodic_phase_response)),
            "abs": float(abs(self.quasi_periodic_phase_response)),
            "arg": float(np.angle(self.quasi_periodic_phase_response)),
        }
        return data


def prompt_to_phi_state(prompt: str, size: int) -> np.ndarray:
    """
    Deterministically map text into a fixed-length complex state vector.

    The mapping is intentionally simple and auditable:
    - UTF-8 bytes accumulate onto a fixed torus vertex grid
    - magnitudes are byte-scaled
    - phases follow golden-ratio progressions
    - a SHA-256 digest seeds low-amplitude fill to avoid dead vertices
    """

    if size <= 0:
        raise ValueError("State size must be positive.")

    encoded = (prompt or "").encode("utf-8", errors="replace")
    if not encoded:
        encoded = b" "

    state = np.zeros(size, dtype=np.complex128)
    two_pi = 2.0 * math.pi

    for idx, value in enumerate(encoded):
        target = idx % size
        magnitude = (float(value) + 1.0) / 256.0
        theta = two_pi * (((idx + 1) * PHI) % 1.0)
        state[target] += magnitude * np.exp(1j * theta)

    digest = hashlib.sha256(encoded).digest()
    for idx in range(size):
        if abs(state[idx]) > 0.0:
            continue
        value = digest[idx % len(digest)]
        magnitude = (float(value) + 1.0) / 4096.0
        theta = two_pi * ((PHI * (idx + 1)) % 1.0)
        state[idx] = magnitude * np.exp(1j * theta)

    return state


def build_topological_chat_context(
    prompt: str,
    *,
    nu: int = 8,
    nv: int = 10,
) -> TopologicalChatContext:
    size = nu * nv
    state = prompt_to_phi_state(prompt, size=size)

    kernel = TopologicalGraphKernel.from_torus(nu, nv, state=state, attachment="vertex")
    kernel.initialize_edge_operators(scale=1.0)
    kernel.initialize_vertex_operators(scale=0.25)
    observables = kernel.torus_observables()
    topology = kernel.topology

    prompt_sha256 = hashlib.sha256((prompt or "").encode("utf-8", errors="replace")).hexdigest()
    return TopologicalChatContext(
        prompt_sha256=prompt_sha256,
        surface_type=str(kernel.mesh.metadata.get("surface_type", "torus")),
        nu=nu,
        nv=nv,
        vertex_count=topology.vertex_count,
        edge_count=topology.edge_count,
        face_count=topology.face_count,
        euler_characteristic=topology.euler_characteristic,
        orientable=topology.orientable,
        genus=topology.genus,
        crosscap_number=topology.crosscap_number,
        scalar_holonomy=observables.scalar_holonomy,
        su2_trace_holonomy=observables.su2_trace_holonomy,
        quasi_periodic_phase_response=observables.quasi_periodic_phase_response,
        loop_interference=observables.loop_interference,
        laplacian_spectrum_min=observables.laplacian_spectrum_min,
        laplacian_spectrum_max=observables.laplacian_spectrum_max,
        connection_spectrum_min=observables.connection_spectrum_min,
        connection_spectrum_max=observables.connection_spectrum_max,
    )


def render_topological_context_block(context: TopologicalChatContext) -> str:
    return "\n".join(
        [
            "[Whitebox Topological Context]",
            (
                f"space={context.surface_type}({context.nu},{context.nv}) "
                f"V={context.vertex_count} E={context.edge_count} F={context.face_count} "
                f"chi={context.euler_characteristic} orientable={context.orientable} "
                f"genus={context.genus} crosscap={context.crosscap_number}"
            ),
            f"prompt_sha256={context.prompt_sha256}",
            (
                f"scalar_holonomy_abs={abs(context.scalar_holonomy):.12f} "
                f"scalar_holonomy_arg={np.angle(context.scalar_holonomy):.12f}"
            ),
            (
                f"su2_trace_abs={abs(context.su2_trace_holonomy):.12f} "
                f"su2_trace_arg={np.angle(context.su2_trace_holonomy):.12f}"
            ),
            (
                f"quasi_periodic_abs={abs(context.quasi_periodic_phase_response):.12f} "
                f"quasi_periodic_arg={np.angle(context.quasi_periodic_phase_response):.12f}"
            ),
            f"loop_interference={context.loop_interference:.12f}",
            (
                f"laplacian_spectrum=[{context.laplacian_spectrum_min:.12f},"
                f"{context.laplacian_spectrum_max:.12f}]"
            ),
            (
                f"connection_spectrum=[{context.connection_spectrum_min:.12f},"
                f"{context.connection_spectrum_max:.12f}]"
            ),
            "Interpret these as deterministic classical observables, not as formal proof.",
        ]
    )
