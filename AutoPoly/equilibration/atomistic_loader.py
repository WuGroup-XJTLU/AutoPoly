# -*- coding: utf-8 -*-
"""
Load an AutoPoly-generated atomistic system (system.data +
system.in.settings) into arrays for the atomistic MC kernel.

Supported subset (what AutoPoly emits for OPLS-AA/GAFF polymers):
  pair styles:   lj/cut/coul/cut (+ lj/charmm/coul/cut, taper ignored,
                 treated as cut at the outer radius — used only for
                 validation-style comparisons)
  bond style:    harmonic           E = K (r - r0)^2      (LAMMPS conv.)
  angle style:   harmonic           E = K (th - th0)^2
  dihedral:      opls               E = sum_n 0.5 K_n (1 + (-1)^{n+1} cos(n phi))
  special bonds: lj/coul a b c      (1-2, 1-3 excluded; 1-4 scaled by c)

Created on 2026-08-20
@author: zwu
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AtomisticSystem:
    """Typed atomistic melt state (LAMMPS `full` atom style)."""

    positions: np.ndarray          # (N, 3)
    mol_ids: np.ndarray            # (N,) molecule/chain id per atom
    type_ids: np.ndarray           # (N,) 1-based LAMMPS atom type
    charges: np.ndarray            # (N,)
    bonds: List[Tuple[int, int, int]]          # (i, j, bond_type) 0-based
    angles: List[Tuple[int, int, int, int]]    # (i, j, k, angle_type)
    dihedrals: List[Tuple[int, int, int, int, int]]  # (i,j,k,l, dih_type)
    box: Tuple[Tuple[float, float], ...]       # ((xlo,xhi),(ylo,yhi),(zlo,zhi))
    # parameter tables (1-indexed by type id; index 0 unused)
    pair_eps: np.ndarray
    pair_sig: np.ndarray
    bond_k: np.ndarray
    bond_r0: np.ndarray
    angle_k: np.ndarray
    angle_t0: np.ndarray           # radians
    dih_k: np.ndarray              # (n_dih_types+1, 4) OPLS K1..K4
    coul_cut: float = 11.0
    lj_cut: float = 11.0
    scale14_lj: float = 0.5
    scale14_coul: float = 0.5
    masses: Dict[int, float] = field(default_factory=dict)  # type -> mass

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[0])

    @property
    def box_size(self) -> float:
        spans = [hi - lo for lo, hi in self.box]
        assert abs(spans[0] - spans[1]) < 1e-9 and abs(spans[0] - spans[2]) < 1e-9, \
            "only cubic boxes supported"
        return spans[0]


def _parse_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        m = re.match(r"^(Masses|Atoms|Bonds|Angles|Dihedrals|Impropers|Velocities|"
                     r"Pair Coeffs|Bond Coeffs|Angle Coeffs|Dihedral Coeffs|"
                     r"Improper Coeffs|Pair Coeffs IJ|BondBond Coeffs|"
                     r"BondAngle Coeffs)\b", s)
        if m and not s[0].isdigit():
            current = m.group(1)
            sections.setdefault(current, [])
            continue
        if current and s[0].isdigit():
            sections[current].append(s)
    return sections


def load_lammps_system(
    data_path: str | Path,
    settings_path: str | Path,
    charges_path: Optional[str | Path] = None,
    coul_cut: float = 11.0,
    lj_cut: float = 11.0,
) -> AtomisticSystem:
    """Parse system.data + system.in.settings (+ optional charges file)."""
    data_path = Path(data_path)
    text = data_path.read_text()

    # box
    box = []
    for axis in ("x", "y", "z"):
        m = re.search(rf"^\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+{axis}lo\s+{axis}hi",
                      text, re.M)
        box.append((float(m.group(1)), float(m.group(2))))

    sec = _parse_sections(text)

    n_types = 0
    m = re.search(r"^\s*(\d+)\s+atom types", text, re.M)
    n_types = int(m.group(1))

    atoms = sec.get("Atoms", [])
    n = len(atoms)
    positions = np.zeros((n, 3))
    mol_ids = np.zeros(n, dtype=np.int64)
    type_ids = np.zeros(n, dtype=np.int64)
    charges = np.zeros(n)
    for line in atoms:
        p = line.split("#")[0].split()
        idx = int(p[0]) - 1
        mol_ids[idx] = int(p[1])
        type_ids[idx] = int(p[2])
        charges[idx] = float(p[3])
        positions[idx] = [float(p[4]), float(p[5]), float(p[6])]

    bonds = [(int(p[2]) - 1, int(p[3]) - 1, int(p[1]))
             for line in sec.get("Bonds", [])
             for p in [line.split("#")[0].split()]]
    angles = [(int(p[2]) - 1, int(p[3]) - 1, int(p[4]) - 1, int(p[1]))
              for line in sec.get("Angles", [])
              for p in [line.split("#")[0].split()]]
    dihedrals = [(int(p[2]) - 1, int(p[3]) - 1, int(p[4]) - 1, int(p[5]) - 1,
                  int(p[1]))
                 for line in sec.get("Dihedrals", [])
                 for p in [line.split("#")[0].split()]]

    # ---- settings ----
    settings = Path(settings_path).read_text()

    masses: Dict[int, float] = {}
    for m in re.finditer(r"^\s*(\d+)\s+([\d.eE+-]+)", sec.get("Masses", []) and "\n".join(sec.get("Masses", [])) or "", re.M):
        masses[int(m.group(1))] = float(m.group(2))

    pair_eps = np.zeros(n_types + 1)
    pair_sig = np.zeros(n_types + 1)
    for m in re.finditer(r"^\s*pair_coeff\s+(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
                         settings, re.M):
        t = int(m.group(1))
        if t <= n_types:
            pair_eps[t] = float(m.group(3))
            pair_sig[t] = float(m.group(4))

    n_bond_types = int(re.search(r"^\s*(\d+)\s+bond types", text, re.M).group(1))
    bond_k = np.zeros(n_bond_types + 1)
    bond_r0 = np.zeros(n_bond_types + 1)
    for m in re.finditer(r"^\s*bond_coeff\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
                         settings, re.M):
        bond_k[int(m.group(1))] = float(m.group(2))
        bond_r0[int(m.group(1))] = float(m.group(3))

    n_ang_types = int(re.search(r"^\s*(\d+)\s+angle types", text, re.M).group(1))
    angle_k = np.zeros(n_ang_types + 1)
    angle_t0 = np.zeros(n_ang_types + 1)
    for m in re.finditer(r"^\s*angle_coeff\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
                         settings, re.M):
        angle_k[int(m.group(1))] = float(m.group(2))
        angle_t0[int(m.group(1))] = np.radians(float(m.group(3)))

    n_dih_types = int(re.search(r"^\s*(\d+)\s+dihedral types", text, re.M).group(1))
    dih_k = np.zeros((n_dih_types + 1, 4))
    for m in re.finditer(
        r"^\s*dihedral_coeff\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
        settings, re.M,
    ):
        t = int(m.group(1))
        dih_k[t] = [float(m.group(i)) for i in range(2, 6)]

    scale14 = (0.5, 0.5)
    m = re.search(r"^\s*special_bonds\s+\S+\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
                  settings, re.M)
    if m:
        scale14 = (float(m.group(3)), float(m.group(3)))
    # lj/coul split form: "special_bonds lj a b c coul a b c"
    m2 = re.search(
        r"special_bonds\s+lj\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)\s+"
        r"coul\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)", settings)
    if m2:
        scale14 = (float(m2.group(1)), float(m2.group(2)))

    return AtomisticSystem(
        positions=positions,
        mol_ids=mol_ids,
        type_ids=type_ids,
        charges=charges,
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        box=tuple(box),
        pair_eps=pair_eps,
        pair_sig=pair_sig,
        bond_k=bond_k,
        bond_r0=bond_r0,
        angle_k=angle_k,
        angle_t0=angle_t0,
        dih_k=dih_k,
        coul_cut=coul_cut,
        lj_cut=lj_cut,
        scale14_lj=scale14[0],
        scale14_coul=scale14[1],
        masses=masses,
    )


def backbone_chains(system: AtomisticSystem) -> List[List[int]]:
    """
    Ordered heavy-atom backbone contour per molecule: collect heavy atoms
    (mass > 4), build their bond subgraph, start from a degree-1 endpoint,
    and walk. Raises on branched heavy-atom graphs (Phase 3 is linear).
    """
    n = system.n_atoms
    heavy = {i for i in range(n) if system.masses.get(int(system.type_ids[i]), 0.0) > 4.0}
    adj: Dict[int, List[int]] = {i: [] for i in heavy}
    for i, j, _ in system.bonds:
        if i in adj and j in adj:
            adj[i].append(j)
            adj[j].append(i)
    chains: List[List[int]] = []
    seen: set = set()
    for start in sorted(heavy):
        if start in seen:
            continue
        # find an endpoint of this connected component
        comp = []
        stack = [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        endpoints = [u for u in comp if len(adj[u]) == 1]
        if not endpoints:
            raise ValueError("cyclic backbone not supported in Phase 3")
        # walk from the first endpoint
        path = [endpoints[0]]
        prev, cur = None, endpoints[0]
        while True:
            nxt = [v for v in adj[cur] if v != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
        chains.append(path)
    return chains
