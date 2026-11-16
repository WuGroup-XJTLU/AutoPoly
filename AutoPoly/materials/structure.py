#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Force-field-agnostic atomic structure for inorganic components.

AtomicStructure is the shared representation behind built-in substrate
slabs and finite nanoparticles (clusters, sheets, rods, imported
structures).  It holds what a structure *is* -- elements, coordinates,
cell/periodicity, bond topology with periodic image shifts, chemical-role
and surface-site annotations, geometry descriptors and provenance -- and
deliberately nothing about how it interacts: charges and Lennard-Jones
parameters are attached later, after force-field selection, as a
ParameterAssignment (see AutoPoly.materials.forcefields).

Conventions:

- ``bonds[k] = (i, j)`` with ``images[k] = (nx, ny, nz)``: atom j is bonded
  to the periodic image of atom j shifted by ``n . cell`` (i.e. the bond
  vector is ``r_j + n . cell - r_i``).  Finite structures have all-zero
  images.
- ``cell`` rows are the lattice vectors (A).  Only axes with ``pbc=True``
  are periodic; non-periodic axes may still carry an (informational) cell
  extent.
- Per-atom annotations (``roles``, ``surface``, ``site_types``,
  ``component_ids``) are optional lists aligned with ``elements``.

Created on 2026-11-13
@author: zwu
"""
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.exceptions import ValidationError

#: Valid geometry types (descriptors, not force-field parameters).
GEOMETRY_TYPES = ("slab", "cluster", "sheet", "rod", "imported")

#: Standard atomic masses (amu) for the elements used by the built-in
#: materials and imported structures (IUPAC 2013 conventional values).
ATOMIC_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.9994, "F": 18.998,
    "Na": 22.990, "Mg": 24.305, "Al": 26.9815, "Si": 28.0855,
    "P": 30.974, "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078,
    "Ti": 47.867, "Fe": 55.845, "Ni": 58.693, "Cu": 63.546,
    "Zn": 65.38, "Zr": 91.224, "Pd": 106.42, "Ag": 107.8682,
    "Pt": 195.084, "Au": 196.96657, "Pb": 207.2,
}


def element_mass(element: str) -> float:
    """Atomic mass (amu) of an element symbol (RDKit table as fallback)."""
    if element in ATOMIC_MASSES:
        return ATOMIC_MASSES[element]
    try:
        from rdkit import Chem
        return float(Chem.GetPeriodicTable().GetAtomicWeight(element))
    except Exception:
        raise ValidationError(
            f"No atomic mass tabulated for element '{element}'"
        ) from None


@dataclass
class AtomicStructure:
    """
    Force-field-agnostic atomic structure (see module docstring).

    Attributes:
        elements: Element symbol per atom.
        positions: (N, 3) Cartesian coordinates in Angstrom.
        bonds: Bonded atom index pairs.
        images: (M, 3) integer periodic image shift per bond (zeros for
                finite structures).
        cell: Optional (3, 3) lattice vectors (rows), Angstrom.
        pbc: Per-axis periodicity flags.
        roles: Optional chemical role per atom (e.g. "Si", "OB", "OH",
               "HO" for silica), assigned by a chemistry adapter.
        surface: Optional per-atom surface flag.
        site_types: Optional per-atom surface-site label (e.g. "silanol").
        normals: Optional (N, 3) outward surface normals (zero for bulk).
        component_ids: Optional connected-component index per atom.
        geometry_type: "slab", "cluster", "sheet", "rod", or "imported".
        descriptors: Geometry descriptors (envelope, core volume,
                     accessible area, ...) -- never force-field data.
        provenance: Source/builder/parameters/citations/seed record.
        metadata: Material identity and chemistry bookkeeping (material
                  id, phase, termination, silanol counts, ...).
    """
    elements: List[str]
    positions: np.ndarray
    bonds: List[Tuple[int, int]] = field(default_factory=list)
    images: np.ndarray = None  # type: ignore[assignment]  # set in __post_init__
    cell: Optional[np.ndarray] = None
    pbc: Tuple[bool, bool, bool] = (False, False, False)
    roles: Optional[List[str]] = None
    surface: Optional[List[bool]] = None
    site_types: Optional[List[Optional[str]]] = None
    normals: Optional[np.ndarray] = None
    component_ids: Optional[List[int]] = None
    geometry_type: str = "cluster"
    descriptors: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.elements = [str(e) for e in self.elements]
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        self.bonds = [(int(i), int(j)) for i, j in self.bonds]
        if self.images is None:
            self.images = np.zeros((len(self.bonds), 3), dtype=int)
        else:
            self.images = np.asarray(self.images, dtype=int).reshape(-1, 3)
        if self.cell is not None:
            self.cell = np.asarray(self.cell, dtype=float).reshape(3, 3)
        p = tuple(self.pbc)
        self.pbc = (bool(p[0]), bool(p[1]), bool(p[2]))
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=float).reshape(-1, 3)
        self.validate()

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def n_atoms(self) -> int:
        return len(self.elements)

    @property
    def n_bonds(self) -> int:
        return len(self.bonds)

    @property
    def is_periodic(self) -> bool:
        return any(self.pbc)

    def element_counts(self) -> Dict[str, int]:
        """Element -> count, in first-appearance order."""
        counts: Dict[str, int] = {}
        for e in self.elements:
            counts[e] = counts.get(e, 0) + 1
        return counts

    def formula(self) -> str:
        """Hill-like formula string (counts in first-appearance order)."""
        return "".join(
            f"{e}{n if n > 1 else ''}" for e, n in self.element_counts().items()
        )

    def masses(self) -> np.ndarray:
        """Per-atom masses (amu)."""
        return np.array([element_mass(e) for e in self.elements])

    def total_mass(self) -> float:
        """Total mass (amu)."""
        return float(self.masses().sum())

    def center_of_mass(self) -> np.ndarray:
        m = self.masses()
        return (self.positions * m[:, None]).sum(axis=0) / m.sum()

    def indices_of(self, element: str) -> np.ndarray:
        return np.array(
            [i for i, e in enumerate(self.elements) if e == element], dtype=int
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """
        Check internal consistency (array shapes, bond indices, PBC).

        Raises:
            ValidationError: On any inconsistency.
        """
        n = self.n_atoms
        if self.positions.shape != (n, 3):
            raise ValidationError(
                f"AtomicStructure: positions shape {self.positions.shape} "
                f"does not match {n} elements"
            )
        if not np.all(np.isfinite(self.positions)):
            raise ValidationError("AtomicStructure: non-finite coordinates")
        if self.images.shape != (len(self.bonds), 3):
            raise ValidationError(
                f"AtomicStructure: {len(self.bonds)} bonds but images shape "
                f"{self.images.shape}"
            )
        seen = set()
        for k, (i, j) in enumerate(self.bonds):
            if not (0 <= i < n and 0 <= j < n):
                raise ValidationError(
                    f"AtomicStructure: bond {k} ({i}, {j}) references an "
                    f"atom outside 0..{n - 1}"
                )
            img = tuple(int(x) for x in self.images[k])
            if i == j and img == (0, 0, 0):
                raise ValidationError(
                    f"AtomicStructure: bond {k} bonds atom {i} to itself"
                )
            key = (min(i, j), max(i, j),
                   img if i <= j else tuple(-x for x in img))
            if key in seen:
                raise ValidationError(
                    f"AtomicStructure: duplicate bond ({i}, {j}) image {img}"
                )
            seen.add(key)
        if any(self.pbc):
            if self.cell is None:
                raise ValidationError(
                    "AtomicStructure: periodic axes require a cell"
                )
            for axis, periodic in enumerate(self.pbc):
                if not periodic and np.any(self.images[:, axis] != 0):
                    raise ValidationError(
                        f"AtomicStructure: bond image shifts along "
                        f"non-periodic axis {'xyz'[axis]}"
                    )
        elif np.any(self.images != 0):
            raise ValidationError(
                "AtomicStructure: finite structure has non-zero bond images"
            )
        for name in ("roles", "surface", "site_types", "component_ids"):
            values = getattr(self, name)
            if values is not None and len(values) != n:
                raise ValidationError(
                    f"AtomicStructure: '{name}' has {len(values)} entries "
                    f"for {n} atoms"
                )
        if self.normals is not None and self.normals.shape != (n, 3):
            raise ValidationError(
                f"AtomicStructure: normals shape {self.normals.shape} does "
                f"not match {n} atoms"
            )
        if self.geometry_type not in GEOMETRY_TYPES:
            raise ValidationError(
                f"AtomicStructure: unknown geometry_type "
                f"'{self.geometry_type}'; choose from {GEOMETRY_TYPES}"
            )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def bond_vectors(self) -> np.ndarray:
        """(M, 3) bond vectors r_j + image.cell - r_i."""
        if not self.bonds:
            return np.zeros((0, 3))
        b = np.asarray(self.bonds, dtype=int)
        vec = self.positions[b[:, 1]] - self.positions[b[:, 0]]
        if self.cell is not None and np.any(self.images):
            vec = vec + self.images @ self.cell
        return vec

    def bond_lengths(self) -> np.ndarray:
        """(M,) bond lengths (A), periodic images applied."""
        return np.linalg.norm(self.bond_vectors(), axis=1)

    def neighbors(self) -> List[List[int]]:
        """Per-atom bonded neighbor index lists (bond order preserved)."""
        nbrs: List[List[int]] = [[] for _ in range(self.n_atoms)]
        for i, j in self.bonds:
            nbrs[i].append(j)
            nbrs[j].append(i)
        return nbrs

    def coordination(self) -> np.ndarray:
        """Per-atom bonded coordination number."""
        coord = np.zeros(self.n_atoms, dtype=int)
        for i, j in self.bonds:
            coord[i] += 1
            coord[j] += 1
        return coord

    def connected_components(self) -> List[int]:
        """Connected-component index per atom (largest component = 0)."""
        from .topology import connected_components
        return connected_components(self.n_atoms, self.bonds)

    def annotate_components(self) -> None:
        """Fill ``component_ids`` from the bond graph."""
        self.component_ids = self.connected_components()

    def translate(self, vector: Sequence[float]) -> None:
        self.positions = self.positions + np.asarray(vector, dtype=float)

    def rotate(self, matrix: np.ndarray, center: Optional[Sequence[float]] = None) -> None:
        """Rotate coordinates (and normals) about ``center`` (default origin)."""
        r = np.asarray(matrix, dtype=float).reshape(3, 3)
        c = np.zeros(3) if center is None else np.asarray(center, dtype=float)
        self.positions = (self.positions - c) @ r.T + c
        if self.normals is not None:
            self.normals = self.normals @ r.T

    def radius_of_gyration(self) -> float:
        com = self.center_of_mass()
        m = self.masses()
        d2 = ((self.positions - com) ** 2).sum(axis=1)
        return float(np.sqrt((m * d2).sum() / m.sum()))

    def bounding_radius(self, center: Optional[Sequence[float]] = None) -> float:
        """Max distance of any atom from ``center`` (default: centroid)."""
        c = (self.positions.mean(axis=0) if center is None
             else np.asarray(center, dtype=float))
        return float(np.linalg.norm(self.positions - c, axis=1).max())

    # ------------------------------------------------------------------
    # Subsetting / editing
    # ------------------------------------------------------------------
    def subset(self, keep: Iterable[int]) -> "AtomicStructure":
        """
        New structure with only the atoms in ``keep`` (sorted, bonds whose
        both ends survive are kept with their images).
        """
        keep_idx = sorted(set(int(k) for k in keep))
        remap = {old: new for new, old in enumerate(keep_idx)}
        bonds, images = [], []
        for (i, j), img in zip(self.bonds, self.images):
            if i in remap and j in remap:
                bonds.append((remap[i], remap[j]))
                images.append(img)

        def pick(values):
            if values is None:
                return None
            return [values[k] for k in keep_idx]

        return AtomicStructure(
            elements=[self.elements[k] for k in keep_idx],
            positions=self.positions[keep_idx] if keep_idx
            else np.zeros((0, 3)),
            bonds=bonds,
            images=np.array(images, dtype=int).reshape(-1, 3),
            cell=None if self.cell is None else self.cell.copy(),
            pbc=self.pbc,
            roles=pick(self.roles),
            surface=pick(self.surface),
            site_types=pick(self.site_types),
            normals=None if self.normals is None
            else self.normals[keep_idx].copy(),
            component_ids=pick(self.component_ids),
            geometry_type=self.geometry_type,
            descriptors=copy.deepcopy(self.descriptors),
            provenance=copy.deepcopy(self.provenance),
            metadata=copy.deepcopy(self.metadata),
        )

    def copy(self) -> "AtomicStructure":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Serialization (JSON-compatible)
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict (round-trips through from_dict)."""
        return {
            "elements": list(self.elements),
            "positions": [[float(x) for x in p] for p in self.positions],
            "bonds": [[int(i), int(j)] for i, j in self.bonds],
            "images": [[int(x) for x in img] for img in self.images],
            "cell": None if self.cell is None
            else [[float(x) for x in row] for row in self.cell],
            "pbc": [bool(p) for p in self.pbc],
            "roles": None if self.roles is None else list(self.roles),
            "surface": None if self.surface is None
            else [bool(s) for s in self.surface],
            "site_types": None if self.site_types is None
            else list(self.site_types),
            "normals": None if self.normals is None
            else [[float(x) for x in n] for n in self.normals],
            "component_ids": None if self.component_ids is None
            else [int(c) for c in self.component_ids],
            "geometry_type": self.geometry_type,
            "descriptors": _jsonable(self.descriptors),
            "provenance": _jsonable(self.provenance),
            "metadata": _jsonable(self.metadata),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AtomicStructure":
        bonds = [tuple(b) for b in data.get("bonds", [])]
        images = data.get("images")
        return AtomicStructure(
            elements=list(data["elements"]),
            positions=np.array(data["positions"], dtype=float).reshape(-1, 3),
            bonds=bonds,
            images=None if images is None
            else np.array(images, dtype=int).reshape(-1, 3),
            cell=None if data.get("cell") is None
            else np.array(data["cell"], dtype=float),
            pbc=tuple(data.get("pbc", (False, False, False))),
            roles=data.get("roles"),
            surface=data.get("surface"),
            site_types=data.get("site_types"),
            normals=None if data.get("normals") is None
            else np.array(data["normals"], dtype=float),
            component_ids=data.get("component_ids"),
            geometry_type=data.get("geometry_type", "cluster"),
            descriptors=dict(data.get("descriptors", {})),
            provenance=dict(data.get("provenance", {})),
            metadata=dict(data.get("metadata", {})),
        )


def _jsonable(value):
    """Recursively convert numpy scalars/arrays/tuples to JSON types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
