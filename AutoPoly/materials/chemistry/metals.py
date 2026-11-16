#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elemental fcc metals (Au, Ag): clusters and surface metadata.

A metal cluster has a single chemical role per element ("Au", "Ag") and
no covalent bonds: intra-particle interactions of a rigid/frozen particle
are excluded at run time (neigh_modify exclude molecule/intra); a
flexible metal would need a validated many-body or interfacial model and
is not offered.  Coordination numbers (12 in the bulk) are stored as
surface-site metadata ("metal_surface" for CN < 12).

Builders: sphere cut from the fcc lattice (no optional dependency), or
faceted cuboctahedron / octahedron / Wulff shapes through ASE (optional
dependency ``ase``; Wulff construction is used for the cubic fcc lattice
only).

Created on 2026-11-13
@author: zwu
"""
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from ...core.exceptions import ValidationError
from ...core.system import logger
from ..structure import AtomicStructure
from ..topology import neighbor_pairs

METALS = ("Au", "Ag")
BOND_CUTOFFS: Dict = {}
#: Bulk densities (g/cm^3) for imported clusters.
DENSITY = {"Au": 19.30, "Ag": 10.49}
BULK_DENSITY = None  # element specific: see core_volume_of()


def assign_roles(elements: Sequence[str], bonds) -> List[str]:
    return list(elements)


def role_bond_type(role_a: str, role_b: str) -> str:
    return "m_m"


def _coordination(structure: AtomicStructure, element: str) -> np.ndarray:
    from ..crystals import FCC_LATTICE
    a = FCC_LATTICE[element]
    cut = 0.5 * (a / math.sqrt(2.0) + a)
    i, j, _, _ = neighbor_pairs(structure.positions, cut)
    return np.bincount(np.concatenate([i, j]), minlength=structure.n_atoms)


def validate(structure: AtomicStructure, **kwargs) -> Dict[str, object]:
    el = set(structure.elements)
    if len(el) != 1 or not el <= set(METALS):
        raise ValidationError(
            f"Metal cluster must contain one of {METALS} only, got {sorted(el)}"
        )
    if structure.bonds:
        raise ValidationError("Metal clusters carry no covalent bonds")
    element = next(iter(el))
    cn = _coordination(structure, element)
    if np.any(cn < 3):
        raise ValidationError(
            f"{int(np.sum(cn < 3))} metal atoms with fewer than 3 neighbors "
            "(detached adatoms)"
        )
    hist = {f"cn{k}": int(np.sum(cn == k)) for k in sorted(set(cn.tolist()))}
    return {"coordination": hist,
            "metadata": {"n_surface": int(np.sum(cn < 12)),
                         "termination": "bare"}}


def core_mass(structure: AtomicStructure) -> float:
    return structure.total_mass()


def core_volume_of(structure: AtomicStructure) -> float:
    element = structure.elements[0]
    return structure.total_mass() / (DENSITY[element] * 0.602214076)


def surface_flags(structure: AtomicStructure) -> List[bool]:
    cn = _coordination(structure, structure.elements[0])
    return [bool(c < 12) for c in cn]


def _finish(structure: AtomicStructure, element: str, builder: str,
            extra: Dict) -> AtomicStructure:
    structure.positions = structure.positions - structure.positions.mean(axis=0)
    structure.roles = assign_roles(structure.elements, [])
    report = validate(structure)
    structure.surface = surface_flags(structure)
    structure.site_types = ["metal_surface" if f else None
                            for f in structure.surface]
    norms = np.linalg.norm(structure.positions, axis=1)
    structure.normals = np.where(
        np.array(structure.surface)[:, None],
        structure.positions / np.maximum(norms, 1e-9)[:, None], 0.0)
    v = core_volume_of(structure)
    structure.descriptors = {
        "envelope": {"shape": "sphere", "radius": float(norms.max()),
                     "center": [0.0, 0.0, 0.0]},
        "core_volume": float(v),
        "core_radius_equivalent": float((3 * v / (4 * math.pi)) ** (1 / 3)),
        "accessible_area_nm2": float(4 * math.pi * norms.max() ** 2 / 100.0),
        **extra,
    }
    structure.provenance.update({"builder": builder})
    structure.metadata.update({
        "material_id": element.lower(), "phase": "fcc",
        "termination": "bare", "coordination": report["coordination"],
        "n_surface": report["metadata"]["n_surface"],
    })
    structure.component_ids = [0] * structure.n_atoms
    logger.info(f"{element} {builder}: {structure.n_atoms} atoms, "
                f"{report['metadata']['n_surface']} surface atoms")
    return structure


def build_fcc_sphere(element: str, crystal, radius: float,
                     center_offset: Optional[Sequence[float]] = None
                     ) -> AtomicStructure:
    """Sphere cut from the fcc lattice; adatoms with CN < 3 removed."""
    if element not in METALS:
        raise ValidationError(f"Unsupported metal '{element}'")
    if radius < 3.0:
        raise ValidationError(f"Metal sphere radius {radius} A too small")
    el, pos = crystal.replicate_around_origin(radius + 3.0)
    if center_offset is not None:
        pos = pos - np.asarray(center_offset, dtype=float)
    keep = np.linalg.norm(pos, axis=1) <= radius
    pos = pos[keep]
    while True:
        s = AtomicStructure([element] * len(pos), pos)
        cn = _coordination(s, element)
        if np.all(cn >= 3):
            break
        pos = pos[cn >= 3]
    s = AtomicStructure([element] * len(pos), pos, geometry_type="cluster",
                        provenance={"crystal": crystal.name,
                                    "crystal_citation": crystal.citation})
    return _finish(s, element, "fcc_sphere", {"nominal_radius": float(radius)})


def build_fcc_polyhedron(element: str, kind: str, size: float,
                         surface_energies=None) -> AtomicStructure:
    """Faceted fcc cluster via ASE (optional dependency)."""
    try:
        from ase.cluster import Octahedron, wulff_construction
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ValidationError(
            f"shape '{kind}' needs the optional dependency 'ase' "
            "(pip install ase); use shape='sphere' otherwise"
        ) from e
    from ..crystals import FCC_LATTICE
    a = FCC_LATTICE[element]
    nn = a / math.sqrt(2.0)
    if kind in ("cuboctahedron", "octahedron"):
        # edge length (atoms) from the requested radius
        length = max(2, int(round(2 * size / nn)) + 1)
        cutoff = (length - 1) // 2 if kind == "cuboctahedron" else 0
        if kind == "cuboctahedron" and length % 2 == 0:
            length += 1
            cutoff = (length - 1) // 2
        atoms = Octahedron(element, length=length, cutoff=cutoff,
                           latticeconstant=a)
    else:
        n_atoms = max(13, int(4 / 3 * math.pi * size ** 3 * 4 / a ** 3))
        energies = surface_energies or {(1, 1, 1): 1.0, (1, 0, 0): 1.15}
        atoms = wulff_construction(
            element, surfaces=list(energies), energies=list(energies.values()),
            size=n_atoms, structure="fcc", rounding="closest",
            latticeconstant=a)
    s = AtomicStructure([element] * len(atoms), atoms.get_positions(),
                        geometry_type="cluster",
                        provenance={"ase_builder": kind, "lattice": a})
    return _finish(s, element, f"fcc_{kind}", {"nominal_radius": float(size)})
