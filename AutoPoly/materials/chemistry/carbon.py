#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Carbon chemistry: graphene flakes and single-wall nanotubes.

Roles (force-field agnostic):

- "CA": interior sp2 carbon (3 C neighbors),
- "CE": edge carbon carrying one hydrogen (2 C + 1 H),
- "HE": edge hydrogen.

Builders generate the graphitic lattice, cut the requested shape, strip
under-coordinated carbons (fewer than two C neighbors, iteratively) and
passivate every edge carbon with one H along the in-plane bisector of its
two C-C bonds (C-H 1.08 A).  Carbon black is *not* constructed (import an
atomistic/CG structure instead): there is no unique automatically
constructed amorphous carbon-black surface.

Created on 2026-11-13
@author: zwu
"""
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from ...core.exceptions import ValidationError
from ...core.system import logger
from ..crystals import GRAPHENE_A
from ..structure import AtomicStructure
from ..topology import neighbor_pairs

CH_LENGTH = 1.08
CC_CUTOFF = 1.7
BOND_CUTOFFS = {("C", "C"): CC_CUTOFF, ("C", "H"): 1.25}
#: Interlayer spacing of graphite (A).
INTERLAYER = 3.35
#: Graphite density (core volume of imported carbon) (g/cm^3).
BULK_DENSITY = 2.27


def assign_roles(elements: Sequence[str], bonds) -> List[str]:
    nh = [0] * len(elements)
    for i, j in bonds:
        if elements[i] == "H":
            nh[j] += 1
        if elements[j] == "H":
            nh[i] += 1
    return ["HE" if e == "H" else ("CE" if nh[k] else "CA")
            for k, e in enumerate(elements)]


def role_bond_type(role_a: str, role_b: str) -> str:
    return "c_h" if "HE" in (role_a, role_b) else "c_c"


def validate(structure: AtomicStructure, **kwargs) -> Dict[str, object]:
    """sp2 carbon: every C has 3 neighbors (C or H), every H one C."""
    el = structure.elements
    nbrs = structure.neighbors()
    issues = []
    for i, e in enumerate(el):
        ne = [el[j] for j in nbrs[i]]
        if e == "C":
            if len(ne) != 3 or ne.count("H") > 1:
                issues.append(f"C {i} has neighbors {ne} (need 3, <= 1 H)")
        elif e == "H":
            if ne != ["C"]:
                issues.append(f"H {i} has neighbors {ne}")
        else:
            issues.append(f"atom {i}: unexpected element '{e}'")
    if issues:
        raise ValidationError(
            "Carbon structure failed validation:\n  - "
            + "\n  - ".join(issues[:20])
        )
    roles = structure.roles or assign_roles(el, structure.bonds)
    return {"counts": {r: roles.count(r) for r in ("CA", "CE", "HE")},
            "metadata": {"termination": "h_passivated"}}


def core_mass(structure: AtomicStructure) -> float:
    from ..structure import element_mass
    return structure.elements.count("C") * element_mass("C")


def surface_flags(structure: AtomicStructure) -> List[bool]:
    # every atom of a single-layer sheet / SWCNT is exposed
    return [True] * structure.n_atoms


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------
def _graphene_points(xmin, xmax, ymin, ymax, a=GRAPHENE_A):
    """Rectangular-cell graphene lattice points covering a window (z=0)."""
    b = a * math.sqrt(3.0)
    basis = np.array([[0.0, 0.0], [0.5, 1 / 6], [0.5, 0.5], [0.0, 2 / 3]])
    nx0, nx1 = int(math.floor(xmin / a)) - 1, int(math.ceil(xmax / a)) + 1
    ny0, ny1 = int(math.floor(ymin / b)) - 1, int(math.ceil(ymax / b)) + 1
    pts = []
    for i in range(nx0, nx1):
        for j in range(ny0, ny1):
            for fx, fy in basis:
                pts.append(((i + fx) * a, (j + fy) * b))
    pts = np.array(pts)
    m = (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) & \
        (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
    return pts[m]


def _strip_and_passivate(pos: np.ndarray, ch_length: float,
                         axis_normal_fn=None):
    """
    Remove carbons with < 2 C neighbors (iteratively), then cap carbons
    with exactly 2 C neighbors with one H along -(sum of bond unit vectors).
    Returns elements, positions, bonds.
    """
    pos = np.asarray(pos, dtype=float)
    while True:
        i, j, _, _ = neighbor_pairs(pos, CC_CUTOFF)
        cn = np.bincount(np.concatenate([i, j]), minlength=len(pos))
        keep = cn >= 2
        if keep.all():
            break
        pos = pos[keep]
        if len(pos) == 0:
            raise ValidationError("Carbon structure vanished while stripping "
                                  "under-coordinated atoms; enlarge it")
    i, j, _, _ = neighbor_pairs(pos, CC_CUTOFF)
    bonds = [(int(a), int(b)) for a, b in zip(i, j)]
    nbrs = [[] for _ in range(len(pos))]
    for a, b in bonds:
        nbrs[a].append(b)
        nbrs[b].append(a)
    elements = ["C"] * len(pos)
    new_pos = [p for p in pos]
    for k in range(len(pos)):
        if len(nbrs[k]) == 2:
            u = -sum((pos[n] - pos[k]) / np.linalg.norm(pos[n] - pos[k])
                     for n in nbrs[k])
            u /= np.linalg.norm(u)
            h = len(elements)
            elements.append("H")
            new_pos.append(pos[k] + ch_length * u)
            bonds.append((k, h))
        elif len(nbrs[k]) > 3:
            raise ValidationError(f"Carbon atom {k} has {len(nbrs[k])} C "
                                  "neighbors (lattice error)")
    return elements, np.array(new_pos), bonds


def build_graphene_flake(length: float, width: float, layers: int = 1,
                         ch_length: float = CH_LENGTH) -> AtomicStructure:
    """
    H-passivated rectangular graphene flake (zigzag edges along x,
    armchair along y) centered at the origin in the xy plane; ``layers``
    AB-stacked sheets 3.35 A apart.
    """
    if length < 4 * GRAPHENE_A or width < 4 * GRAPHENE_A:
        raise ValidationError("Graphene flake too small (>= ~10 A per side)")
    elements: List[str] = []
    positions = []
    bonds = []
    for layer in range(layers):
        shift = (GRAPHENE_A / math.sqrt(3.0)) if layer % 2 else 0.0
        pts = _graphene_points(-length / 2, length / 2,
                               -width / 2 - shift, width / 2 - shift)
        pts[:, 1] += shift
        z = (layer - (layers - 1) / 2.0) * INTERLAYER
        pos = np.column_stack([pts, np.full(len(pts), z)])
        el, p, b = _strip_and_passivate(pos, ch_length)
        off = len(elements)
        elements += el
        positions.append(p)
        bonds += [(i + off, j + off) for i, j in b]
    arr = np.vstack(positions)
    arr -= arr.mean(axis=0)
    roles = assign_roles(elements, bonds)
    s = AtomicStructure(elements=elements, positions=arr, bonds=bonds,
                        roles=roles, geometry_type="sheet")
    s.surface = surface_flags(s)
    n_c = elements.count("C")
    ext = np.ptp(arr, axis=0)
    area = float(ext[0] * ext[1]) * 2 * layers / 100.0
    s.descriptors = {
        "envelope": {"shape": "box", "radius": float(
            np.linalg.norm(arr, axis=1).max()),
            "center": [0.0, 0.0, 0.0],
            "dimensions": [float(x) for x in ext]},
        "core_volume": float(ext[0] * ext[1] * INTERLAYER * layers),
        "accessible_area_nm2": area,
        "nominal_size": [float(length), float(width)],
    }
    s.provenance = {"builder": "graphene_flake", "a": GRAPHENE_A,
                    "layers": layers, "ch_length": ch_length,
                    "stacking": "AB"}
    s.metadata = {"material_id": "c", "phase": "graphene",
                  "termination": "h_passivated", "n_carbon": n_c,
                  "n_edge_h": elements.count("H")}
    s.annotate_components()
    validate(s)
    logger.info(f"Graphene flake {length}x{width} A x{layers}: "
                f"{n_c} C + {elements.count('H')} edge H")
    return s


def build_nanotube(n: int, m: int, length: float,
                   ch_length: float = CH_LENGTH) -> AtomicStructure:
    """
    Open-ended (n, m) single-wall carbon nanotube along z, centered at the
    origin, ends H-passivated.
    """
    a = GRAPHENE_A
    a1 = np.array([a, 0.0])
    a2 = np.array([a / 2.0, a * math.sqrt(3.0) / 2.0])
    ch = n * a1 + m * a2
    circ = float(np.linalg.norm(ch))
    radius = circ / (2.0 * math.pi)
    if radius < 2.0:
        raise ValidationError(f"Nanotube ({n},{m}) radius {radius:.2f} A too "
                              "small")
    if length < 3 * a:
        raise ValidationError("Nanotube length too small")
    c_hat = ch / circ
    t_hat = np.array([-c_hat[1], c_hat[0]])
    # lattice points (two-atom basis) covering the unrolled rectangle
    corners = np.array([[0, 0], ch, length * t_hat, ch + length * t_hat])
    lo = corners.min(axis=0) - 2 * a
    hi = corners.max(axis=0) + 2 * a
    basis = np.array([[0.0, 0.0], (a1 + a2) / 3.0])
    M = np.column_stack([a1, a2])
    inv = np.linalg.inv(M)
    span = np.array([lo, hi, [lo[0], hi[1]], [hi[0], lo[1]]]) @ inv.T
    i0, j0 = np.floor(span.min(axis=0)).astype(int) - 1
    i1, j1 = np.ceil(span.max(axis=0)).astype(int) + 1
    pts = []
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            for bvec in basis:
                pts.append(i * a1 + j * a2 + bvec)
    pts = np.array(pts)
    u = pts @ c_hat
    v = pts @ t_hat
    tol = 1e-6
    keep = (u >= -tol) & (u < circ - tol) & (v >= 0) & (v <= length)
    u, v = u[keep], v[keep]
    theta = 2.0 * math.pi * u / circ
    pos = np.column_stack([radius * np.cos(theta), radius * np.sin(theta),
                           v - length / 2.0])
    el, p, b = _strip_and_passivate(pos, ch_length)
    # end-cap H: push outward along the tube axis direction of the end
    roles = assign_roles(el, b)
    s = AtomicStructure(elements=el, positions=p, bonds=b, roles=roles,
                        geometry_type="rod")
    s.surface = surface_flags(s)
    n_c = el.count("C")
    s.descriptors = {
        "envelope": {"shape": "capsule", "radius": float(
            np.linalg.norm(p, axis=1).max()), "center": [0.0, 0.0, 0.0],
            "tube_radius": radius, "length": float(np.ptp(p[:, 2]))},
        "core_volume": float(math.pi * (radius + INTERLAYER / 2) ** 2
                             * length - math.pi * max(radius - INTERLAYER / 2,
                                                      0.0) ** 2 * length),
        "accessible_area_nm2": float(2 * math.pi * radius * length / 100.0),
        "chirality": [n, m],
    }
    s.provenance = {"builder": "swcnt", "n": n, "m": m, "a": a,
                    "ch_length": ch_length, "ends": "open, H-passivated"}
    s.metadata = {"material_id": "c", "phase": "cnt",
                  "termination": "h_passivated", "n_carbon": n_c,
                  "n_edge_h": el.count("H"), "tube_radius": radius}
    s.annotate_components()
    validate(s)
    logger.info(f"SWCNT ({n},{m}) L={length} A: radius {radius:.2f} A, "
                f"{n_c} C + {el.count('H')} H")
    return s
