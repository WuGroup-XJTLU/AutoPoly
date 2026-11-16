#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure import/export for nanoparticles.

Readers (finite clusters):

- ``.json``: AtomicStructure dict (bonds included; written by
  Nanoparticle.write_structure),
- ``.pdb``: HETATM/ATOM coordinates + CONECT bonds,
- ``.xyz``: coordinates only -- bonds must be perceived explicitly
  (``perceive_bonds=True``) with the material's distance cutoffs.

Every imported structure is validated by the material's chemistry adapter
(valence, stoichiometry, termination) before it can be typed.

Created on 2026-11-13
@author: zwu
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..core.exceptions import ValidationError
from ..materials.structure import AtomicStructure
from ..materials.topology import perceive_bonds as _perceive


def read_structure(path, material, perceive_bonds: Optional[bool] = None
                   ) -> AtomicStructure:
    """Read and validate a cluster file for ``material`` (MaterialSpec)."""
    path = Path(path)
    if not path.is_file():
        raise ValidationError(f"Nanoparticle structure file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        s = AtomicStructure.from_dict(json.loads(path.read_text()))
        has_bonds = True
    elif suffix == ".pdb":
        el, pos, bonds = _read_pdb(path)
        s = AtomicStructure(el, pos, bonds=bonds, geometry_type="imported")
        has_bonds = bool(bonds)
    elif suffix == ".xyz":
        from ..pipeline.unit_coords import read_xyz
        el, pos = read_xyz(path)
        s = AtomicStructure(el, pos, geometry_type="imported")
        has_bonds = False
    else:
        raise ValidationError(
            f"Unsupported nanoparticle file format '{suffix}' (use .json, "
            ".pdb with CONECT records, or .xyz with perceive_bonds=True)"
        )
    if s.is_periodic:
        raise ValidationError(
            "Imported nanoparticles must be finite clusters (no PBC)"
        )
    s.geometry_type = "imported"
    s.provenance.setdefault("source", str(path))
    if not has_bonds and perceive_bonds is not True:
        raise ValidationError(
            f"{path.name} carries no bond topology. XYZ coordinates alone do "
            "not define bonds, charges or parameters: pass "
            "perceive_bonds=True to perceive bonds with the material's "
            "distance cutoffs (validated against its valence rules), or "
            "provide a .pdb with CONECT records / an AtomicStructure .json"
        )
    if perceive_bonds is True and has_bonds:
        raise ValidationError(
            f"{path.name} already defines bonds; perceive_bonds=True would "
            "discard them"
        )
    return prepare_imported(s, material, perceive_bonds=perceive_bonds)


def prepare_imported(s: AtomicStructure, material,
                     perceive_bonds: Optional[bool] = None) -> AtomicStructure:
    """
    Perceive bonds (if requested), assign roles and validate an imported
    structure with the material's chemistry adapter.
    """
    chem = material.chemistry_module()
    if chem is None:
        raise ValidationError(
            f"Material '{material.id}' has no chemistry adapter; imported "
            "structures cannot be validated"
        )
    bad = sorted(set(s.elements) - set(material.elements))
    if bad:
        raise ValidationError(
            f"Imported {material.id} structure contains element(s) {bad} "
            f"not allowed for this material ({list(material.elements)})"
        )
    if perceive_bonds:
        bonds, images = _perceive(s.elements, s.positions, chem.BOND_CUTOFFS)
        s.bonds, s.images = bonds, images
        s.provenance["bonds"] = "perceived (distance cutoffs)"
    else:
        s.provenance.setdefault("bonds", "explicit")
    s.roles = chem.assign_roles(s.elements, s.bonds)
    report = chem.validate(s)
    s.metadata.update(report.get("metadata", {}))
    s.metadata["validation"] = {k: v for k, v in report.items()
                                if k != "metadata"}
    centroid = s.positions.mean(axis=0)
    radii = np.linalg.norm(s.positions - centroid, axis=1)
    surface = getattr(chem, "surface_flags", None)
    if surface is not None:
        s.surface = surface(s)
    s.annotate_components()
    if max(s.component_ids or [0]) > 0:
        raise ValidationError(
            f"Imported structure has {max(s.component_ids) + 1} disconnected "
            "fragments; a nanoparticle must be one connected cluster"
        )
    mass = s.total_mass()
    density = getattr(chem, "BULK_DENSITY", None)
    rg = s.radius_of_gyration()
    if hasattr(chem, "core_volume_of"):
        v = chem.core_volume_of(s)
        s.descriptors.setdefault("core_volume", float(v))
        s.descriptors.setdefault("core_radius_equivalent", float(
            (3.0 * v / (4.0 * np.pi)) ** (1.0 / 3.0)))
    s.descriptors.setdefault("envelope", {
        "shape": "sphere", "radius": float(radii.max()),
        "center": [float(x) for x in centroid],
    })
    s.descriptors.setdefault("radius_of_gyration", rg)
    if density:
        core_volume = chem.core_mass(s) / (density * 0.602214076)
        s.descriptors.setdefault("core_volume", float(core_volume))
        s.descriptors.setdefault("core_radius_equivalent", float(
            (3.0 * core_volume / (4.0 * np.pi)) ** (1.0 / 3.0)))
    else:
        s.descriptors.setdefault(
            "core_volume", float(4.0 / 3.0 * np.pi * (rg * np.sqrt(5 / 3)) ** 3))
    s.descriptors.setdefault("mass", mass)
    return s


def _read_pdb(path: Path) -> Tuple[List[str], np.ndarray, List[Tuple[int, int]]]:
    elements, coords, serials = [], [], []
    conect = []
    for line in path.read_text().splitlines():
        rec = line[:6].strip()
        if rec in ("ATOM", "HETATM"):
            serials.append(int(line[6:11]))
            coords.append([float(line[30:38]), float(line[38:46]),
                           float(line[46:54])])
            el = line[76:78].strip() if len(line) >= 78 else ""
            if not el:
                name = line[12:16].strip()
                el = "".join(c for c in name if c.isalpha())[:2]
                el = el[0].upper() + el[1:].lower() if len(el) > 1 and \
                    el[1].islower() else el[:1].upper()
            elements.append(el[0].upper() + el[1:].lower())
        elif rec == "CONECT":
            fields = [line[i:i + 5] for i in range(6, len(line), 5)]
            ids = [int(f) for f in fields if f.strip()]
            if ids:
                conect.append(ids)
    index = {s: k for k, s in enumerate(serials)}
    bonds = set()
    for ids in conect:
        a = index.get(ids[0])
        if a is None:
            continue
        for b_serial in ids[1:]:
            b = index.get(b_serial)
            if b is not None and b != a:
                bonds.add((min(a, b), max(a, b)))
    return elements, np.array(coords, dtype=float), sorted(bonds)


def write_structure(s: AtomicStructure, path) -> Path:
    """Write .json (lossless), .xyz, or .pdb (+CONECT)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(s.to_dict(), indent=1))
    elif suffix == ".xyz":
        from ..pipeline.unit_coords import write_xyz
        write_xyz(path, s.elements, s.positions, comment=s.formula())
    elif suffix == ".pdb":
        if s.n_atoms > 99999:
            raise ValidationError("PDB export is limited to 99999 atoms")
        lines = []
        for k, (e, (x, y, z)) in enumerate(zip(s.elements, s.positions), 1):
            name = (e + str(k % 1000))[:4]
            lines.append(
                f"HETATM{k:5d} {name:<4s} NP  X   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {e:>2s}"
            )
        nbrs = s.neighbors()
        for k, nb in enumerate(nbrs, 1):
            for chunk in range(0, len(nb), 4):
                ids = "".join(f"{j + 1:5d}" for j in nb[chunk:chunk + 4])
                lines.append(f"CONECT{k:5d}{ids}")
        lines.append("END")
        path.write_text("\n".join(lines) + "\n")
    else:
        raise ValidationError(f"Unsupported export format '{suffix}'")
    return path
