# -*- coding: utf-8 -*-
"""
LAMMPS I/O for equilibrated KG bead-spring melt states.

Convention note: the autopoly_mc kernel applies the pair potential to
ALL pairs including bonded ones (KG convention, ``exclude_bonded=False``).
LAMMPS reproduces this exactly with ``bond_style fene`` — whose bond
term already contains the WCA part (FENE + LJ truncated at 2^(1/6)sigma,
shifted) — together with ``special_bonds lj 0.0 1.0 1.0`` so the pair
style does not double-count 1-2 pairs.

Created on 2026-08-20
@author: zwu
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..core.system import logger
from .state import MeltState

WCA_RCUT = 2.0 ** (1.0 / 6.0)


def write_lammps_data(
    state: MeltState,
    path: str | Path,
    *,
    bond_k: float = 30.0,
    fene_r0: float = 1.5,
    epsilon: float = 1.0,
    sigma: float = 1.0,
    comment: str = "",
) -> Path:
    """
    Write a LAMMPS data file (atom_style molecular) for the melt state.

    All beads share one atom type (mass = epsilon = sigma = 1 units).
    Bond topology comes from the (possibly reordered) chain contours.
    """
    state.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_atoms = state.n_beads
    bonds = state.bonds()
    L = state.box_size

    lines = [
        f"LAMMPS data file — AutoPoly equilibration melt. {comment}",
        "",
        f"{n_atoms} atoms",
        f"{len(bonds)} bonds",
        "0 angles",
        "0 dihedrals",
        "0 impropers",
        "",
        "1 atom types",
        "1 bond types",
        "",
        f"0.0 {L:.10f} xlo xhi",
        f"0.0 {L:.10f} ylo yhi",
        f"0.0 {L:.10f} zlo zhi",
        "",
        "Masses",
        "",
        "1 1.0",
        "",
        "Atoms  # molecular",
        "",
    ]
    # mol id = chain index (1-based), atom type = 1
    for c, chain in enumerate(state.chains, start=1):
        for bead in chain:
            x, y, z = state.positions[bead]
            lines.append(f"{bead + 1} {c} 1 {x:.10f} {y:.10f} {z:.10f}")
    lines += ["", "Bonds", ""]
    for i, (a, b) in enumerate(bonds, start=1):
        lines.append(f"{i} 1 {a + 1} {b + 1}")
    lines.append("")
    path.write_text("\n".join(lines))
    logger.info(f"Wrote LAMMPS data file: {path} ({n_atoms} atoms, "
                f"{len(bonds)} bonds, L={L:.3f})")
    return path


def write_kg_settings(
    path: str | Path,
    *,
    bond_k: float = 30.0,
    fene_r0: float = 1.5,
    epsilon: float = 1.0,
    sigma: float = 1.0,
) -> Path:
    """Write the matching force-field settings include file."""
    path = Path(path)
    rc = WCA_RCUT * sigma
    lines = [
        "# KG FENE+WCA settings matching autopoly_mc (exclude_bonded=False)",
        f"pair_style lj/cut {rc:.10f}",
        "pair_modify shift yes",
        "special_bonds lj 0.0 1.0 1.0   # fene bond term carries the 1-2 WCA",
        f"pair_coeff * * {epsilon} {sigma} {rc:.10f}",
        "",
        "bond_style fene",
        f"bond_coeff * {bond_k} {fene_r0} {epsilon} {sigma}",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def write_kg_run(
    path: str | Path,
    *,
    data_file: str = "system.data",
    settings_file: str = "system.in.settings",
    temperature: float = 1.0,
    run_steps: int = 100_000,
    seed: int = 87287,
    dump_every: int = 10_000,
) -> Path:
    """Write a minimal KG NVT run script (dt=0.006 tau, T=1)."""
    path = Path(path)
    lines = [
        "# KG melt NVT run (LAMMPS units lj)",
        "units lj",
        "atom_style molecular",
        "boundary p p p",
        f"read_data {data_file}",
        f"include {settings_file}",
        "",
        "neighbor 0.4 bin",
        "neigh_modify every 1 delay 0 check yes",
        "",
        f"velocity all create {temperature} {seed} dist gaussian",
        f"fix nvt all nvt temp {temperature} {temperature} 1.0",
        "timestep 0.006",
        "thermo 1000",
        f"dump traj all custom {dump_every} traj.lammpstrj id mol type x y z",
        "dump_modify traj sort id",
        "",
        f"run {run_steps}",
        "write_data final.data nofix",
        "",
    ]
    path.write_text("\n".join(lines))
    return path
