#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared moltemplate serializer for typed inorganic components.

Writes one self-contained moltemplate class (masses, pair coefficients,
atoms, and -- optionally -- exclusion-only bonds) for a typed inorganic
structure: built-in substrate slabs and nanoparticles use the same writer,
so atom/type/bond serialization is identical across components.

Format (stable; the legacy slab .lt output is reproduced byte for byte):

    # header comment lines ...

    ClassName {

      write_once("Data Masses") { @atom:T mass ... }        (sorted types)
      write_once("In Settings") { pair_coeff @atom:T @atom:T [sub] eps sig }
      write("Data Atoms") { $atom:name $mol:. @atom:T q x y z ... }
      write_once("In Settings") { bond_coeff @bond:B [sub] k r0 ... }
      write("Data Bonds") { $bond:bK @bond:B $atom:a $atom:b ... }

    }

Bond coefficients are written as given; built-in inorganic models emit
zero-force-constant bonds whose only purpose is to make ``special_bonds``
exclude intra-component 1-2/1-3/1-4 nonbonded pairs (the component must
then be held frozen or rigid).

Created on 2026-11-13
@author: zwu
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def role_labels(roles: Sequence[str], prefix: str) -> List[str]:
    """
    Unique, deterministic atom names: ``<prefix>_<role>_<n>`` with a
    1-based counter per role in atom order (e.g. "qz_si_1", "qz_oh_7").
    """
    counters: Dict[str, int] = {}
    labels = []
    for role in roles:
        counters[role] = counters.get(role, 0) + 1
        labels.append(f"{prefix}_{role.lower()}_{counters[role]}")
    return labels


def inorganic_lt_text(
    class_name: str,
    labels: Sequence[str],
    type_names: Sequence[str],
    charges: Sequence[float],
    positions: np.ndarray,
    type_masses: Dict[str, float],
    type_lj: Dict[str, Tuple[float, float]],
    header: Sequence[str] = (),
    bonds: Optional[Sequence[Tuple[int, int]]] = None,
    bond_types: Optional[Sequence[str]] = None,
    bond_coeffs: Optional[Dict[str, Tuple[float, float]]] = None,
    pair_substyle: Optional[str] = None,
    bond_substyle: Optional[str] = None,
    mol_ref: str = "$mol:.",
) -> str:
    """
    Render a typed inorganic component as moltemplate .lt text.

    Args:
        class_name: Moltemplate class name.
        labels: Unique atom name per atom.
        type_names: @atom type per atom.
        charges: Charge per atom (e).
        positions: (N, 3) coordinates.
        type_masses / type_lj: per-type mass and (epsilon, sigma).
        header: Comment lines (without leading "# ").
        bonds / bond_types: bonded pairs and the @bond type per bond.
        bond_coeffs: bond type -> (k, r0), in output order.
        pair_substyle / bond_substyle: sub-style keyword for hybrid styles.
        mol_ref: molecule-ID reference written in every atom line.
    """
    lines = [f"# {h}" for h in header]
    lines += ["", f"{class_name} {{", "", '  write_once("Data Masses") {']
    used = sorted(set(type_names))
    for tname in used:
        lines.append(f"    @atom:{tname} {type_masses[tname]:.4f}")
    lines += ["  }", "", '  write_once("In Settings") {']
    sub = f" {pair_substyle}" if pair_substyle else ""
    for tname in used:
        eps, sig = type_lj[tname]
        lines.append(
            f"    pair_coeff @atom:{tname} @atom:{tname}{sub} "
            f"{eps:.6g} {sig:.6g}"
        )
    lines += ["  }", "", '  write("Data Atoms") {']
    for name, tname, q, (x, y, z) in zip(labels, type_names, charges,
                                         positions):
        lines.append(
            f"    $atom:{name} {mol_ref} @atom:{tname} "
            f"{q:.4f} {x:.4f} {y:.4f} {z:.4f}"
        )
    lines += ["  }"]

    if bonds:
        if bond_types is None or bond_coeffs is None:
            raise ValueError("bonds require bond_types and bond_coeffs")
        bsub = f" {bond_substyle}" if bond_substyle else ""
        lines += ["", '  write_once("In Settings") {']
        for btype, (k, r0) in bond_coeffs.items():
            lines.append(f"    bond_coeff @bond:{btype}{bsub} {k} {r0:.4f}")
        lines += ["  }", "", '  write("Data Bonds") {']
        for n, ((i, j), btype) in enumerate(zip(bonds, bond_types), start=1):
            lines.append(
                f"    $bond:b{n} @bond:{btype} "
                f"$atom:{labels[i]} $atom:{labels[j]}"
            )
        lines += ["  }"]

    lines += ["", "}", ""]
    return "\n".join(lines)


def write_inorganic_lt(path, **kwargs) -> Path:
    """Write inorganic_lt_text(**kwargs) to ``path``; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(inorganic_lt_text(**kwargs))
    return path


def mean_bond_lengths(
    positions: np.ndarray,
    bonds: Sequence[Tuple[int, int]],
    bond_types: Sequence[str],
    type_order: Sequence[str],
    lx: Optional[float] = None,
    ly: Optional[float] = None,
    lz: Optional[float] = None,
) -> Dict[str, float]:
    """
    Mean bond length per bond type (minimum image on the given periods),
    in ``type_order``; types without bonds are omitted.
    """
    out: Dict[str, float] = {}
    for tname in type_order:
        d = []
        for (i, j), bt in zip(bonds, bond_types):
            if bt != tname:
                continue
            dv = positions[i] - positions[j]
            if lx is not None:
                dv[0] -= lx * round(dv[0] / lx)
            if ly is not None:
                dv[1] -= ly * round(dv[1] / ly)
            if lz is not None:
                dv[2] -= lz * round(dv[2] / lz)
            d.append(float(np.linalg.norm(dv)))
        if d:
            out[tname] = float(np.mean(d))
    return out
