# -*- coding: utf-8 -*-
"""
HMC healing of atomistic bridge moves via LAMMPS (Option B).

One HMC composite step:
  1. kernel proposes a bridge candidate (loose contact gate),
  2. topology edit is applied (junction forms stretched),
  3. LAMMPS refreshes momenta and integrates a short NVE trajectory
     that heals the junction strain,
  4. accept with min(1, (n_fwd/n_rev) * exp(-ΔH/kBT)).

The LAMMPS run uses the same cutoff-only Hamiltonian as the kernel
(lj/cut/coul/cut), so energies are directly comparable.

Created on 2026-08-20
@author: zwu
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.system import logger
from .atomistic_writer import write_lammps_data

LMP_BIN_DEFAULT = str(Path.home() / "miniconda3/envs/lmp/bin/lmp")


@dataclass
class HealResult:
    accepted: bool
    delta_h: float
    pe_final: float
    ke0: float
    n_fwd: int
    n_rev: int
    healed_positions: Optional[np.ndarray] = None
    log_path: Optional[Path] = None


def _parse_data_positions(path: Path, n_atoms: int) -> np.ndarray:
    """Parse the Atoms section of a LAMMPS-written data file."""
    text = path.read_text()
    m = re.search(r"^Atoms[^\n]*\n\n(.*?)(?:\n\n|\Z)", text, re.M | re.S)
    pos = np.zeros((n_atoms, 3))
    for line in m.group(1).strip().splitlines():
        p = line.split()
        pos[int(p[0]) - 1] = [float(p[-6]), float(p[-5]), float(p[-4])]
    return pos


class LammpsHealer:
    """Drives one-off LAMMPS heal trajectories for bridge candidates."""

    def __init__(
        self,
        settings_path: str | Path,
        *,
        lmp_bin: str = LMP_BIN_DEFAULT,
        workdir: Optional[str | Path] = None,
        heal_steps: int = 400,
        timestep_fs: float = 0.25,
        keep_logs: bool = False,
    ) -> None:
        self.settings_path = str(Path(settings_path).resolve())
        self.lmp_bin = lmp_bin
        self.workdir = Path(workdir) if workdir else None
        self.heal_steps = heal_steps
        self.timestep_fs = timestep_fs
        self.keep_logs = keep_logs
        self._counter = 0

    def heal(
        self,
        system,
        positions: np.ndarray,
        bonds,
        angles,
        dihedrals,
        u_old: float,
        temperature_k: float,
        seed: int,
    ) -> HealResult:
        """
        Run one HMC heal trajectory for a bridged topology.

        Args:
            system: AtomisticSystem metadata (types, charges, mol ids, masses).
            positions: pre-heal positions (old geometry, junction stretched).
            bonds/angles/dihedrals: NEW topology (after the bridge edit).
            u_old: kernel-computed total energy of the OLD state (kcal/mol).
            temperature_k: physical temperature for the momenta refresh.
            seed: RNG seed for the velocity create.

        Returns the trajectory unconditionally (positions always parsed);
        the acceptance decision belongs to the caller.
        """
        self._counter += 1
        if self.workdir:
            wd = self.workdir
            wd.mkdir(parents=True, exist_ok=True)
        else:
            wd = Path(tempfile.mkdtemp(prefix="hmc_heal_"))
        data_path = wd / "heal.data"
        out_data = wd / "healed.data"
        write_lammps_data(system, positions, bonds, angles, dihedrals, data_path,
                          comment=f"heal {self._counter}")

        in_lines = [
            "units real",
            "atom_style full",
            "boundary p p p",
            "bond_style harmonic",
            "angle_style harmonic",
            "dihedral_style opls",
            "improper_style cvff",
            "special_bonds lj/coul 0.0 0.0 0.5",
            "pair_style lj/cut/coul/cut 11.0 11.0",
            "pair_modify mix geometric",
            f"read_data {data_path.name}",
            f"include {self.settings_path}",
            "neighbor 2.0 bin",
            "neigh_modify every 1 delay 0 check yes",
            f"velocity all create {temperature_k} {seed} dist gaussian",
            "thermo_style custom step pe ke",
            "thermo 1",
            "variable ke0 equal ke",
            f"timestep {self.timestep_fs}",
            "run 0",
            'print "KE0 ${ke0}"',
            "fix nve all nve",
            f"run {self.heal_steps}",
            "unfix nve",
            f"write_data {out_data.name} nofix",
        ]
        in_path = wd / "in.heal"
        in_path.write_text("\n".join(in_lines) + "\n")
        log_path = wd / "heal.log"
        proc = subprocess.run(
            [self.lmp_bin, "-in", in_path.name, "-log", log_path.name,
             "-screen", "none"],
            cwd=wd, capture_output=True, text=True, timeout=300,
        )
        text = log_path.read_text() if log_path.exists() else ""
        ke0 = None
        pe_final = ke_final = None
        for m in re.finditer(r"KE0\s+([\d.eE+-]+)", text):
            ke0 = float(m.group(1))
        thermo = re.findall(r"^\s*(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$",
                            text, re.M)
        if thermo:
            last = thermo[-1]
            pe_final, ke_final = float(last[1]), float(last[2])
        if ke0 is None or pe_final is None or ke_final is None:
            logger.warning(f"heal trajectory failed to parse: {proc.stderr[:200]}")
            return HealResult(False, float("inf"), float("nan"),
                              float("nan"), 0, 0,
                              log_path=log_path if self.keep_logs else None)

        delta_h = (pe_final + ke_final) - (u_old + ke0)
        healed = _parse_data_positions(out_data, positions.shape[0])
        return HealResult(True, delta_h, pe_final, ke0, 0, 0,
                          healed_positions=healed,
                          log_path=log_path if self.keep_logs else None)
