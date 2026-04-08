#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Mesh-backed topological graph kernel.

This module builds a small, auditable transport layer on top of the surface
topology utilities. It is intentionally classical and whitebox:

- state coefficients attach to an explicit mesh
- local operators are concrete 2x2 complex matrices
- path transport is a deterministic matrix product
- observables are numerically measurable and testable
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple
import importlib.util
import sys

import numpy as np


def _load_surface_topology_module():
    try:
        from . import surface_topology as surface_topology_module
        return surface_topology_module
    except Exception:
        module_path = Path(__file__).with_name("surface_topology.py")
        spec = importlib.util.spec_from_file_location("surface_topology", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load surface_topology from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("surface_topology", module)
        spec.loader.exec_module(module)
        return module


_surface_topology = _load_surface_topology_module()
PHI = _surface_topology.PHI
SurfaceMesh = _surface_topology.SurfaceMesh
SurfaceTopology = _surface_topology.SurfaceTopology
build_edge_phase_field = _surface_topology.build_edge_phase_field
canonical_edge = _surface_topology.canonical_edge
compute_surface_topology = _surface_topology.compute_surface_topology
fundamental_torus_cycles = _surface_topology.fundamental_torus_cycles
path_holonomy = _surface_topology.path_holonomy
triangulate_klein_bottle = _surface_topology.triangulate_klein_bottle
triangulate_torus = _surface_topology.triangulate_torus
wilson_loop = _surface_topology.wilson_loop

AttachmentMode = Literal["vertex", "edge", "face"]


def _complex_vector(values: Iterable[complex]) -> np.ndarray:
    vector = np.asarray(list(values), dtype=np.complex128)
    if vector.ndim != 1:
        raise ValueError("State attachment expects a 1D coefficient vector.")
    return vector


def su2_from_phase(theta: float, phase: float = 0.0) -> np.ndarray:
    """
    Construct a deterministic SU(2) matrix from a rotation angle and phase.
    """

    half = theta / 2.0
    c = np.cos(half)
    s = np.sin(half)
    e_pos = np.exp(1j * phase)
    e_neg = np.exp(-1j * phase)
    matrix = np.array(
        [
            [c, -e_pos * s],
            [e_neg * s, c],
        ],
        dtype=np.complex128,
    )
    return matrix


def is_unitary(matrix: np.ndarray, tol: float = 1e-12) -> bool:
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    return np.allclose(matrix.conj().T @ matrix, identity, atol=tol, rtol=0.0)


@dataclass(frozen=True)
class KernelObservables:
    scalar_holonomy: complex
    su2_trace_holonomy: complex
    quasi_periodic_phase_response: complex
    loop_interference: float
    laplacian_spectrum_min: float
    laplacian_spectrum_max: float
    connection_spectrum_min: float
    connection_spectrum_max: float


class TopologicalGraphKernel:
    """
    Mesh-backed state attachment + transport kernel.

    The kernel does not simulate quantum hardware. It provides deterministic
    algebra on a mesh using U(1) edge phases and local SU(2) operators.
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        *,
        state: Optional[Iterable[complex]] = None,
        attachment: AttachmentMode = "vertex",
        edge_phase: Optional[Dict[Tuple[int, int], complex]] = None,
    ) -> None:
        mesh.validate()
        self.mesh = mesh
        self.topology: SurfaceTopology = compute_surface_topology(mesh)
        self.edge_phase = edge_phase or build_edge_phase_field(mesh)
        self.state_attachment_mode: Optional[AttachmentMode] = None
        self.state_vector: Optional[np.ndarray] = None
        self.vertex_state: Dict[int, complex] = {}
        self.edge_state: Dict[Tuple[int, int], complex] = {}
        self.face_state: Dict[int, complex] = {}
        self.edge_operators: Dict[Tuple[int, int], np.ndarray] = {}
        self.vertex_operators: Dict[int, np.ndarray] = {}
        if state is not None:
            self.attach_state(state, attachment=attachment)

    @classmethod
    def from_torus(
        cls,
        nu: int,
        nv: int,
        *,
        state: Optional[Iterable[complex]] = None,
        attachment: AttachmentMode = "vertex",
    ) -> "TopologicalGraphKernel":
        return cls(triangulate_torus(nu, nv), state=state, attachment=attachment)

    @classmethod
    def from_klein_bottle(
        cls,
        nu: int,
        nv: int,
        *,
        state: Optional[Iterable[complex]] = None,
        attachment: AttachmentMode = "vertex",
    ) -> "TopologicalGraphKernel":
        return cls(triangulate_klein_bottle(nu, nv), state=state, attachment=attachment)

    def _expected_length(self, attachment: AttachmentMode) -> int:
        if attachment == "vertex":
            return self.mesh.vertex_count
        if attachment == "edge":
            return len(self.mesh.edges())
        if attachment == "face":
            return self.mesh.face_count
        raise ValueError(f"Unsupported attachment mode: {attachment}")

    def attach_state(
        self,
        state: Iterable[complex],
        *,
        attachment: AttachmentMode = "vertex",
        normalize: bool = True,
    ) -> None:
        vector = _complex_vector(state)
        expected_length = self._expected_length(attachment)
        if len(vector) != expected_length:
            raise ValueError(
                f"Attachment mode {attachment!r} expects {expected_length} coefficients, "
                f"got {len(vector)}."
            )
        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        self.state_attachment_mode = attachment
        self.state_vector = vector
        self.vertex_state = {}
        self.edge_state = {}
        self.face_state = {}

        if attachment == "vertex":
            self.vertex_state = {
                vertex: vector[index]
                for index, vertex in enumerate(sorted(self.mesh.vertices))
            }
        elif attachment == "edge":
            self.edge_state = {
                edge: vector[index]
                for index, edge in enumerate(sorted(self.mesh.edges()))
            }
        else:
            self.face_state = {
                index: vector[index]
                for index in range(self.mesh.face_count)
            }

    def initialize_edge_operators(self, *, scale: float = 1.0, phase_offset: float = 0.0) -> None:
        self.edge_operators = {}
        for edge in sorted(self.mesh.edges()):
            angle = scale * float(np.angle(self.edge_phase[edge]))
            operator = su2_from_phase(angle, phase=phase_offset + angle / max(PHI, 1e-12))
            self.edge_operators[edge] = operator

    def initialize_vertex_operators(self, *, scale: float = 1.0) -> None:
        self.vertex_operators = {}
        for vertex in sorted(self.mesh.vertices):
            angle = scale * 2.0 * np.pi * ((PHI * vertex) % 1.0)
            self.vertex_operators[vertex] = su2_from_phase(angle, phase=angle / (2.0 * np.pi))

    def attach_edge_operator(self, edge: Tuple[int, int], operator: np.ndarray) -> None:
        canonical = canonical_edge(*edge)
        operator = np.asarray(operator, dtype=np.complex128)
        if operator.shape != (2, 2):
            raise ValueError("Edge operator must be a 2x2 matrix.")
        self.edge_operators[canonical] = operator

    def attach_vertex_operator(self, vertex: int, operator: np.ndarray) -> None:
        if vertex not in self.mesh.vertices:
            raise ValueError(f"Unknown vertex: {vertex}")
        operator = np.asarray(operator, dtype=np.complex128)
        if operator.shape != (2, 2):
            raise ValueError("Vertex operator must be a 2x2 matrix.")
        self.vertex_operators[vertex] = operator

    def verify_local_unitarity(self, tol: float = 1e-12) -> Dict[str, float]:
        max_error = 0.0
        total = 0
        for operator in list(self.edge_operators.values()) + list(self.vertex_operators.values()):
            total += 1
            identity = np.eye(2, dtype=np.complex128)
            error = float(np.linalg.norm(operator.conj().T @ operator - identity))
            max_error = max(max_error, error)
            if error > tol:
                raise ValueError(f"Found non-unitary local operator with error {error:.3e}")
        return {"operator_count": total, "max_unitarity_error": max_error}

    def oriented_edge_operator(self, a: int, b: int) -> np.ndarray:
        edge = canonical_edge(a, b)
        if edge not in self.edge_operators:
            raise ValueError(f"No edge operator attached for edge {edge}")
        operator = self.edge_operators[edge]
        return operator if (a, b) == edge else operator.conj().T

    def path_transport(self, path: List[int], *, include_u1_phase: bool = True) -> np.ndarray:
        if len(path) < 2:
            return np.eye(2, dtype=np.complex128)

        transport = np.eye(2, dtype=np.complex128)
        for a, b in zip(path[:-1], path[1:]):
            step = self.oriented_edge_operator(a, b)
            if include_u1_phase:
                phase = self.edge_phase[canonical_edge(a, b)]
                if (a, b) != canonical_edge(a, b):
                    phase = np.conj(phase)
                step = phase * step
            transport = step @ transport
        return transport

    def scalar_cycle_holonomy(self, path: List[int]) -> complex:
        return wilson_loop(path, self.edge_phase)

    def su2_cycle_holonomy(self, path: List[int]) -> complex:
        transport = self.path_transport(path)
        return np.trace(transport) / 2.0

    def quasi_periodic_phase_response(self, path: List[int]) -> complex:
        phases = []
        for vertex in path[:-1] if len(path) > 1 and path[0] == path[-1] else path:
            if self.state_attachment_mode == "vertex" and vertex in self.vertex_state:
                phases.append(np.angle(self.vertex_state[vertex]))
            else:
                phases.append(2.0 * np.pi * ((PHI * vertex) % 1.0))
        if not phases:
            return 1.0 + 0.0j
        phases_array = np.asarray(phases, dtype=float)
        return np.mean(np.exp(1j * PHI * phases_array))

    def cycle_interference(self, path_a: List[int], path_b: List[int]) -> float:
        holonomy_a = self.su2_cycle_holonomy(path_a)
        holonomy_b = self.su2_cycle_holonomy(path_b)
        return float(np.abs(holonomy_a * np.conj(holonomy_b)))

    def combinatorial_laplacian(self) -> np.ndarray:
        laplacian = np.zeros((self.mesh.vertex_count, self.mesh.vertex_count), dtype=np.float64)
        for a, b in self.mesh.edges():
            laplacian[a, a] += 1.0
            laplacian[b, b] += 1.0
            laplacian[a, b] -= 1.0
            laplacian[b, a] -= 1.0
        return laplacian

    def connection_laplacian(self) -> np.ndarray:
        laplacian = np.zeros((self.mesh.vertex_count, self.mesh.vertex_count), dtype=np.complex128)
        for a, b in self.mesh.edges():
            phase = self.edge_phase[(a, b)]
            laplacian[a, a] += 1.0
            laplacian[b, b] += 1.0
            laplacian[a, b] -= phase
            laplacian[b, a] -= np.conj(phase)
        return laplacian

    def laplacian_spectrum(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.combinatorial_laplacian())

    def connection_spectrum(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.connection_laplacian())

    def torus_observables(self) -> KernelObservables:
        metadata = self.mesh.metadata
        if metadata.get("surface_type") != "torus":
            raise ValueError("torus_observables requires a torus mesh.")
        nu = int(metadata["nu"])
        nv = int(metadata["nv"])
        cycles = fundamental_torus_cycles(nu, nv)
        lap_spec = self.laplacian_spectrum()
        conn_spec = self.connection_spectrum()
        return KernelObservables(
            scalar_holonomy=self.scalar_cycle_holonomy(cycles["u_cycle"]),
            su2_trace_holonomy=self.su2_cycle_holonomy(cycles["u_cycle"]),
            quasi_periodic_phase_response=self.quasi_periodic_phase_response(cycles["u_cycle"]),
            loop_interference=self.cycle_interference(cycles["u_cycle"], cycles["v_cycle"]),
            laplacian_spectrum_min=float(np.min(lap_spec)),
            laplacian_spectrum_max=float(np.max(lap_spec)),
            connection_spectrum_min=float(np.min(conn_spec)),
            connection_spectrum_max=float(np.max(conn_spec)),
        )


__all__ = [
    "KernelObservables",
    "TopologicalGraphKernel",
    "SurfaceMesh",
    "SurfaceTopology",
    "is_unitary",
    "su2_from_phase",
]
