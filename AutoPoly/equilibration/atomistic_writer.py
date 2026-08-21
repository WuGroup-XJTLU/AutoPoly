# -*- coding: utf-8 -*-
"""
Write LAMMPS data files for typed atomistic systems (inverse of
``atomistic_loader``), used to hand bridged topologies to the LAMMPS
healing proxy.

Created on 2026-08-20
@author: zwu
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .atomistic_loader import AtomisticSystem


def write_lammps_data(
    system: AtomisticSystem,
    positions: np.ndarray,
    bonds: Sequence[Tuple[int, int, int]],
    angles: Sequence[Tuple[int, int, int, int]],
    dihedrals: Sequence[Tuple[int, int, int, int, int]],
    path: str | Path,
    comment: str = "",
) -> Path:
    """
    Write a LAMMPS data file (atom_style full) from an
    ``AtomisticSystem`` metadata object plus explicit positions and
    (possibly edited) topology.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    L = system.box_size
    masses = system.masses
    n_atom_types = max(int(system.type_ids.max()), len(system.pair_eps) - 1,
                       max(masses, default=0))
    n_bond_types = max(len(system.bond_k) - 1,
                       max([t for _, _, t in bonds], default=0))
    n_angle_types = max(len(system.angle_k) - 1,
                        max([t for _, _, _, t in angles], default=0))
    n_dih_types = max(len(system.dih_k) - 1,
                      max([t for _, _, _, _, t in dihedrals], default=0))
    lines = [
        f"LAMMPS data file — AutoPoly HMC bridge. {comment}",
        "",
        f"{n} atoms",
        f"{len(bonds)} bonds",
        f"{len(angles)} angles",
        f"{len(dihedrals)} dihedrals",
        "0 impropers",
        "",
        f"{n_atom_types} atom types",
        f"{n_bond_types} bond types",
        f"{n_angle_types} angle types",
        f"{n_dih_types} dihedral types",
        "0 improper types",
        "",
        f"{-L/2:.10f} {L/2:.10f} xlo xhi",
        f"{-L/2:.10f} {L/2:.10f} ylo yhi",
        f"{-L/2:.10f} {L/2:.10f} zlo zhi",
        "",
        "Masses",
        "",
    ]
    for t in range(1, n_atom_types + 1):
        lines.append(f"{t} {masses.get(t, 12.0)}")
    lines += ["", "Atoms  # full", ""]
    for i in range(n):
        lines.append(
            f"{i + 1} {int(system.mol_ids[i])} {int(system.type_ids[i])} "
            f"{system.charges[i]:.6f} {pos[i, 0]:.10f} {pos[i, 1]:.10f} {pos[i, 2]:.10f}"
        )
    lines += ["", "Bonds", ""]
    for k, (i, j, t) in enumerate(bonds, 1):
        lines.append(f"{k} {t} {i + 1} {j + 1}")
    lines += ["", "Angles", ""]
    for k, (i, j, l, t) in enumerate(angles, 1):
        lines.append(f"{k} {t} {i + 1} {j + 1} {l + 1}")
    lines += ["", "Dihedrals", ""]
    for k, (i, j, l, m, t) in enumerate(dihedrals, 1):
        lines.append(f"{k} {t} {i + 1} {j + 1} {l + 1} {m + 1}")
    lines.append("")
    path.write_text("\n".join(lines))
    return path
