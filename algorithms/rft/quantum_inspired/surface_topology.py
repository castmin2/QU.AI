#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Surface topology utilities.

This module provides a mesh-first topology layer with:

- explicit vertices / edges / oriented faces
- incidence relations
- torus and Klein bottle triangulations
- Euler characteristic, orientability, genus / crosscap number
- boundary extraction
- discrete U(1) edge phases and Wilson loops

The implementation stays compatible with the existing public API names used by
the quantum-inspired graph code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple
import cmath
import math

import numpy as np

PHI = (1.0 + math.sqrt(5.0)) / 2.0

VertexID = int
Edge = Tuple[int, int]
Face = Tuple[int, int, int]


def canonical_edge(a: int, b: int) -> Edge:
    """Return the undirected edge key for a simplicial edge."""
    if a == b:
        raise ValueError("Self-loop edge is not allowed in a simplicial surface.")
    return (a, b) if a < b else (b, a)


@dataclass
class SurfaceMesh:
    """
    Pure topological/combinatorial surface mesh with optional coordinates.

    Topology is fully determined by the oriented face list. Coordinates are
    optional geometry attached to vertices for downstream visualization or
    transport experiments.
    """

    vertices: Dict[VertexID, np.ndarray]
    faces: List[Face]
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def edges(self) -> Set[Edge]:
        edges: Set[Edge] = set()
        for i, j, k in self.faces:
            edges.add(canonical_edge(i, j))
            edges.add(canonical_edge(j, k))
            edges.add(canonical_edge(k, i))
        return edges

    def edge_count(self) -> int:
        return len(self.edges())

    def face_edges_oriented(self, face: Face) -> List[Tuple[int, int]]:
        i, j, k = face
        return [(i, j), (j, k), (k, i)]

    def edge_face_incidence(self) -> Dict[Edge, List[int]]:
        incidence: Dict[Edge, List[int]] = {edge: [] for edge in self.edges()}
        for face_index, face in enumerate(self.faces):
            for a, b in self.face_edges_oriented(face):
                incidence[canonical_edge(a, b)].append(face_index)
        return incidence

    def oriented_edge_face_incidence(self) -> Dict[Edge, List[Tuple[int, int]]]:
        """
        Return canonical_edge -> [(face_index, sign), ...].

        sign is +1 when the face uses the edge in canonical direction and -1
        otherwise.
        """

        incidence: Dict[Edge, List[Tuple[int, int]]] = {edge: [] for edge in self.edges()}
        for face_index, face in enumerate(self.faces):
            for a, b in self.face_edges_oriented(face):
                edge = canonical_edge(a, b)
                sign = +1 if (a, b) == edge else -1
                incidence[edge].append((face_index, sign))
        return incidence

    def vertex_neighbors(self) -> Dict[int, Set[int]]:
        neighbors: Dict[int, Set[int]] = {vertex: set() for vertex in self.vertices}
        for a, b in self.edges():
            neighbors[a].add(b)
            neighbors[b].add(a)
        return neighbors

    def boundary_edges(self) -> Set[Edge]:
        return {
            edge
            for edge, face_indices in self.edge_face_incidence().items()
            if len(face_indices) == 1
        }

    def boundary_vertices(self) -> Set[int]:
        vertices: Set[int] = set()
        for a, b in self.boundary_edges():
            vertices.add(a)
            vertices.add(b)
        return vertices

    def is_closed(self) -> bool:
        return not self.boundary_edges()

    def validate(self) -> None:
        if not self.faces:
            raise ValueError("SurfaceMesh must contain at least one face.")

        vertex_ids = set(self.vertices.keys())
        for face_index, (i, j, k) in enumerate(self.faces):
            if len({i, j, k}) != 3:
                raise ValueError(f"Degenerate face at index {face_index}: {(i, j, k)}")
            if i not in vertex_ids or j not in vertex_ids or k not in vertex_ids:
                raise ValueError(f"Face {face_index} references missing vertices.")

        for edge, uses in self.edge_face_incidence().items():
            if len(uses) > 2:
                raise ValueError(
                    f"Non-manifold edge {edge}: used by {len(uses)} faces (expected <= 2)."
                )


# Backwards-compatible public alias used by older modules.
TriangulatedSurface = SurfaceMesh


@dataclass(frozen=True)
class SurfaceTopology:
    vertex_count: int
    edge_count: int
    face_count: int
    euler_characteristic: int
    orientable: bool
    genus: Optional[int]
    crosscap_number: Optional[int]
    boundary_component_count: int
    closed: bool


def _build_face_adjacency(mesh: SurfaceMesh) -> Dict[int, List[Tuple[int, Edge, int, int]]]:
    """
    Build the face adjacency graph.

    Returns:
        face_a -> [(face_b, shared_edge, sign_a, sign_b), ...]
    """

    adjacency: Dict[int, List[Tuple[int, Edge, int, int]]] = {
        face_index: [] for face_index in range(len(mesh.faces))
    }
    for edge, entries in mesh.oriented_edge_face_incidence().items():
        if len(entries) == 2:
            (face_a, sign_a), (face_b, sign_b) = entries
            adjacency[face_a].append((face_b, edge, sign_a, sign_b))
            adjacency[face_b].append((face_a, edge, sign_b, sign_a))
    return adjacency


def _is_orientable(mesh: SurfaceMesh) -> bool:
    """
    Determine orientability by propagating relative face flips across edges.
    """

    adjacency = _build_face_adjacency(mesh)
    if not adjacency:
        return True

    face_flip: Dict[int, int] = {}
    for start in range(len(mesh.faces)):
        if start in face_flip:
            continue

        face_flip[start] = 1
        stack = [start]
        while stack:
            face_index = stack.pop()
            for neighbor, _edge, sign_face, sign_neighbor in adjacency[face_index]:
                required = -face_flip[face_index] * sign_face * sign_neighbor
                if neighbor in face_flip:
                    if face_flip[neighbor] != required:
                        return False
                else:
                    face_flip[neighbor] = required
                    stack.append(neighbor)
    return True


def _count_boundary_components(mesh: SurfaceMesh) -> int:
    boundary_edges = mesh.boundary_edges()
    if not boundary_edges:
        return 0

    adjacency: Dict[int, Set[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    seen: Set[int] = set()
    components = 0
    for start in adjacency:
        if start in seen:
            continue
        components += 1
        stack = [start]
        seen.add(start)
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency[vertex]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return components


def compute_surface_topology(mesh: SurfaceMesh) -> SurfaceTopology:
    mesh.validate()

    vertex_count = mesh.vertex_count
    edge_count = len(mesh.edges())
    face_count = len(mesh.faces)
    euler_characteristic = vertex_count - edge_count + face_count
    closed = mesh.is_closed()
    orientable = _is_orientable(mesh)
    boundary_component_count = _count_boundary_components(mesh)

    genus: Optional[int] = None
    crosscap_number: Optional[int] = None

    if closed:
        if orientable:
            if (2 - euler_characteristic) % 2 == 0:
                genus = (2 - euler_characteristic) // 2
        else:
            crosscap_number = 2 - euler_characteristic

    return SurfaceTopology(
        vertex_count=vertex_count,
        edge_count=edge_count,
        face_count=face_count,
        euler_characteristic=euler_characteristic,
        orientable=orientable,
        genus=genus,
        crosscap_number=crosscap_number,
        boundary_component_count=boundary_component_count,
        closed=closed,
    )


def _torus_xyz(u: float, v: float, R: float = 3.0, r: float = 1.0) -> np.ndarray:
    cos_u, sin_u = math.cos(u), math.sin(u)
    cos_v, sin_v = math.cos(v), math.sin(v)
    x = (R + r * cos_v) * cos_u
    y = (R + r * cos_v) * sin_u
    z = r * sin_v
    return np.array([x, y, z], dtype=float)


def _klein_xyz(u: float, v: float, scale: float = 2.0) -> np.ndarray:
    """Immersion-style Klein bottle coordinates in R^3 for visualization."""

    cos_u, sin_u = math.cos(u), math.sin(u)
    sin_v = math.sin(v)

    if 0.0 <= u < math.pi:
        x = (6 * cos_u * (1 + sin_u) + 4 * (1 - cos_u / 2) * math.cos(v)) / 8.0
        z = (16 * sin_u + 4 * (1 - cos_u / 2) * sin_v) / 8.0
    else:
        x = (
            6 * cos_u * (1 + sin_u)
            + 4 * (1 - cos_u / 2) * math.cos(v + math.pi)
        ) / 8.0
        z = (16 * sin_u) / 8.0

    y = (4 * (1 - cos_u / 2) * sin_v) / 8.0
    return scale * np.array([x, y, z], dtype=float)


def triangulate_torus(nu: int, nv: int, R: float = 3.0, r: float = 1.0) -> SurfaceMesh:
    if nu < 3 or nv < 3:
        raise ValueError("torus resolution must satisfy nu,nv >= 3")

    vertices: Dict[int, np.ndarray] = {}

    def vid(i: int, j: int) -> int:
        return i * nv + j

    for i in range(nu):
        u = 2.0 * math.pi * i / nu
        for j in range(nv):
            v = 2.0 * math.pi * j / nv
            vertices[vid(i, j)] = _torus_xyz(u, v, R=R, r=r)

    faces: List[Face] = []
    for i in range(nu):
        i1 = (i + 1) % nu
        for j in range(nv):
            j1 = (j + 1) % nv
            a = vid(i, j)
            b = vid(i1, j)
            c = vid(i1, j1)
            d = vid(i, j1)
            faces.append((a, b, c))
            faces.append((a, c, d))

    return SurfaceMesh(
        vertices=vertices,
        faces=faces,
        metadata={
            "surface_type": "torus",
            "nu": nu,
            "nv": nv,
            "closed": True,
            "identification": "periodic_u_periodic_v",
        },
    )


def triangulate_klein_bottle(nu: int, nv: int, scale: float = 2.0) -> SurfaceMesh:
    """
    Construct a Klein bottle from a rectangular grid with a twisted seam.
    """

    if nu < 3 or nv < 3:
        raise ValueError("klein bottle resolution must satisfy nu,nv >= 3")

    vertices: Dict[int, np.ndarray] = {}

    def raw_vid(i: int, j: int) -> int:
        return i * nv + j

    for i in range(nu):
        u = 2.0 * math.pi * i / nu
        for j in range(nv):
            v = 2.0 * math.pi * j / nv
            vertices[raw_vid(i, j)] = _klein_xyz(u, v, scale=scale)

    def seam_v(j: int) -> int:
        return (nv - j) % nv

    faces: List[Face] = []
    for i in range(nu):
        i1 = i + 1
        wrap_twist = i1 == nu
        for j in range(nv):
            j1 = (j + 1) % nv

            a = raw_vid(i, j)
            d = raw_vid(i, j1)
            if not wrap_twist:
                b = raw_vid(i1, j)
                c = raw_vid(i1, j1)
            else:
                b = raw_vid(0, seam_v(j))
                c = raw_vid(0, seam_v(j1))

            faces.append((a, b, c))
            faces.append((a, c, d))

    return SurfaceMesh(
        vertices=vertices,
        faces=faces,
        metadata={
            "surface_type": "klein_bottle",
            "nu": nu,
            "nv": nv,
            "closed": True,
            "identification": "periodic_u_twisted_v",
        },
    )


def extract_face_boundary(face: Face) -> List[Tuple[int, int]]:
    i, j, k = face
    return [(i, j), (j, k), (k, i)]


def build_edge_phase_field(
    mesh: SurfaceMesh,
    phi: float = PHI,
    mode: str = "golden_vertex_phase",
) -> Dict[Edge, complex]:
    """
    Assign a U(1) phase to each edge for discrete transport experiments.
    """

    if mode != "golden_vertex_phase":
        raise ValueError(f"Unsupported phase mode: {mode}")

    vertex_phase: Dict[int, complex] = {}
    for vertex in mesh.vertices:
        theta = 2.0 * math.pi * ((phi * vertex) % 1.0)
        vertex_phase[vertex] = cmath.exp(1j * theta)

    edge_phase: Dict[Edge, complex] = {}
    for a, b in mesh.edges():
        edge_phase[(a, b)] = np.conj(vertex_phase[a]) * vertex_phase[b]
    return edge_phase


def path_holonomy(path: List[int], edge_phase: Dict[Edge, complex]) -> complex:
    """Compute discrete parallel transport along a vertex path."""

    if len(path) < 2:
        return 1.0 + 0.0j

    holonomy = 1.0 + 0.0j
    for a, b in zip(path[:-1], path[1:]):
        edge = canonical_edge(a, b)
        if edge not in edge_phase:
            raise ValueError(f"Path uses non-edge {(a, b)}")
        holonomy *= edge_phase[edge] if (a, b) == edge else np.conj(edge_phase[edge])
    return holonomy


def wilson_loop(path: List[int], edge_phase: Dict[Edge, complex]) -> complex:
    if len(path) < 3 or path[0] != path[-1]:
        raise ValueError("Wilson loop path must be closed.")
    return path_holonomy(path, edge_phase)


def fundamental_torus_cycles(nu: int, nv: int) -> Dict[str, List[int]]:
    """Return the standard generator cycles for the torus triangulation."""

    if nu < 3 or nv < 3:
        raise ValueError("Need nu,nv >= 3")

    def vid(i: int, j: int) -> int:
        return i * nv + j

    u_cycle = [vid(i, 0) for i in range(nu)] + [vid(0, 0)]
    v_cycle = [vid(0, j) for j in range(nv)] + [vid(0, 0)]
    return {"u_cycle": u_cycle, "v_cycle": v_cycle}


def summarize_surface(mesh: SurfaceMesh) -> Dict[str, object]:
    topology = compute_surface_topology(mesh)
    return {
        "metadata": dict(mesh.metadata),
        "vertex_count": topology.vertex_count,
        "edge_count": topology.edge_count,
        "face_count": topology.face_count,
        "euler_characteristic": topology.euler_characteristic,
        "orientable": topology.orientable,
        "genus": topology.genus,
        "crosscap_number": topology.crosscap_number,
        "boundary_component_count": topology.boundary_component_count,
        "closed": topology.closed,
    }


def _demo() -> None:
    print("=" * 72)
    print("TOPOLOGY ENGINE DEMO")
    print("=" * 72)

    torus = triangulate_torus(8, 10)
    torus_topology = compute_surface_topology(torus)
    print("Torus:")
    print(
        f" V={torus_topology.vertex_count}, E={torus_topology.edge_count},"
        f" F={torus_topology.face_count}"
    )
    print(
        f" chi={torus_topology.euler_characteristic},"
        f" orientable={torus_topology.orientable}, genus={torus_topology.genus}"
    )

    klein = triangulate_klein_bottle(8, 10)
    klein_topology = compute_surface_topology(klein)
    print("Klein bottle:")
    print(
        f" V={klein_topology.vertex_count}, E={klein_topology.edge_count},"
        f" F={klein_topology.face_count}"
    )
    print(
        f" chi={klein_topology.euler_characteristic},"
        f" orientable={klein_topology.orientable},"
        f" crosscap_number={klein_topology.crosscap_number}"
    )

    edge_phase = build_edge_phase_field(torus)
    cycles = fundamental_torus_cycles(8, 10)
    wilson_u = wilson_loop(cycles["u_cycle"], edge_phase)
    wilson_v = wilson_loop(cycles["v_cycle"], edge_phase)
    print("Torus Wilson loops:")
    print(f" W_u = {wilson_u} |arg|={abs(cmath.phase(wilson_u)):.6f}")
    print(f" W_v = {wilson_v} |arg|={abs(cmath.phase(wilson_v)):.6f}")
    print("=" * 72)


if __name__ == "__main__":
    _demo()
