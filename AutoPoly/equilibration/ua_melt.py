# -*- coding: utf-8 -*-
"""
United-atom (TraPPE-UA) reduction of an atomistic PE melt for the
connectivity anneal stage.

Rationale (measured 2026-08-20): all-atom CBMC double-bridge acceptance
on the C52 x 8 OPLS-AA melt is ~e^-160 even at deep core-softening; the
residual is dominated by the 16 regrown hydrogens (~-9 kT each) plus
discretization penalties, NOT by the LJ (softening saturates). The
Theodorou-lineage connectivity schemes all operated on united-atom
models for this reason. Strategy: anneal connectivity on the carbon-only
UA view (8 regrown atoms per quadmer instead of 24), then re-hydrogenate
(ideal sp3 + local MC) and harden — the Zhang-2014 backmapping pattern.

UA parameters: TraPPE-UA alkane beads (Siepmann 1998/1999):
  CH3: sigma 3.75 A, eps 0.194 kcal/mol
  CH2: sigma 3.95 A, eps 0.091 kcal/mol
  bonds:  (k/2)(r-r0)^2 with k/2 = 300 kcal/mol/A^2? -- we use the
          LAMMPS-convention K(r-r0)^2 with K = 300 (matches the kernel's
          harmonic no-half convention and keeps C-C stiffness close to
          the all-atom 268)
  angles: K(theta-theta0)^2, K = 62.1 kcal/mol/rad^2, theta0 = 114 deg
  torsion (OPLS form sum_n 0.5 K_n (1 + (-1)^{n+1} cos(n phi))):
          K1 = 2.006, K2 = -0.406, K3 = 0.424 kcal/mol (TraPPE-UA c1..c3
          mapped to the kernel's OPLS convention: u = c1(1+cos phi)
          + c2(1-cos 2phi) + c3(1+cos 3phi) => 0.5*K_n = c_n)

Created on 2026-08-20
@author: zwu
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .atomistic_loader import AtomisticSystem

# TraPPE-UA alkane parameters (type 1 = CH3 end, type 2 = CH2 interior)
UA_SIGMA = {1: 3.75, 2: 3.95}          # Angstrom
UA_EPS = {1: 0.194, 2: 0.091}          # kcal/mol
UA_BOND_K = 300.0                       # K(r-r0)^2, kcal/mol/A^2
UA_BOND_R0 = 1.54                       # Angstrom
UA_ANGLE_K = 62.1                       # K(th-th0)^2, kcal/mol/rad^2
UA_ANGLE_T0 = np.radians(114.0)
# TraPPE-UA torsion: u = c1(1+cos p) + c2(1-cos 2p) + c3(1+cos 3p),
# c = (2.006, -0.406, 0.424) kcal/mol (Martin & Siepmann 1998)
UA_DIH_K = np.array([2.006, -0.406, 0.424, 0.0])


def to_united_atom(system: AtomisticSystem) -> AtomisticSystem:
    """Carbon-only UA view of an all-atom PE melt (TraPPE-UA tables).

    Atom indices are renumbered (carbons only, original order kept), so
    the backbone `chains` must be re-derived by the caller's loader path;
    we re-emit them via the AtomisticSystem constructor's mol list and
    the standard backbone_chains() helper.
    """
    heavy_ids: List[int] = [
        i for i in range(system.n_atoms)
        if system.masses.get(int(system.type_ids[i]), 0.0) > 4.0
    ]
    remap: Dict[int, int] = {old: new for new, old in enumerate(heavy_ids)}
    n_ua = len(heavy_ids)

    positions = np.asarray([system.positions[i] for i in heavy_ids])
    mol_ids = np.asarray([system.mol_ids[i] for i in heavy_ids])
    # UA typing by heavy-atom coordination: 1 C neighbor -> CH3 (type 1),
    # 2 -> CH2 (type 2)
    carbon_neighbors: Dict[int, int] = {i: 0 for i in heavy_ids}
    for i, j, _ in system.bonds:
        if i in remap and j in remap:
            carbon_neighbors[i] += 1
            carbon_neighbors[j] += 1
    type_ids = np.asarray([
        1 if carbon_neighbors[i] == 1 else 2 for i in heavy_ids
    ])
    charges = np.zeros(n_ua)

    bonds: List[Tuple[int, int, int]] = []
    angles: List[Tuple[int, int, int, int]] = []
    dihedrals: List[Tuple[int, int, int, int, int]] = []
    for i, j, _ in system.bonds:
        if i in remap and j in remap:
            bonds.append((remap[i], remap[j], 1))
    for i, j, k, _ in system.angles:
        if i in remap and j in remap and k in remap:
            angles.append((remap[i], remap[j], remap[k], 1))
    for i, j, k, l, _ in system.dihedrals:
        if all(x in remap for x in (i, j, k, l)):
            dihedrals.append((remap[i], remap[j], remap[k], remap[l], 1))

    pair_eps = np.array([0.0, UA_EPS[1], UA_EPS[2]])
    pair_sig = np.array([0.0, UA_SIGMA[1], UA_SIGMA[2]])
    masses = {1: 15.035, 2: 14.027}
    return AtomisticSystem(
        positions=positions,
        mol_ids=mol_ids,
        type_ids=type_ids,
        charges=charges,
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        box=system.box,
        pair_eps=pair_eps,
        pair_sig=pair_sig,
        bond_k=np.array([0.0, UA_BOND_K]),
        bond_r0=np.array([0.0, UA_BOND_R0]),
        angle_k=np.array([0.0, UA_ANGLE_K]),
        angle_t0=np.array([0.0, UA_ANGLE_T0]),
        dih_k=np.vstack([np.zeros(4), UA_DIH_K]),
        coul_cut=system.coul_cut,
        lj_cut=system.lj_cut,
        scale14_lj=1.0,   # TraPPE-UA: no 1-4 scaling
        scale14_coul=0.0,
        masses=masses,
    )


# ----------------------------------------------------------------------
# Backmapping: re-hydrogenate a UA melt into the all-atom representation
# ----------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])


def _place_h2(c, n1, n2, l_ch, half_hch, box):
    """Ideal CH2 completion: two H's symmetric about the -bisector of
    the (n1, n2) directions, H-C-H angle 2*half_hch."""
    u1 = _unit(n1 - c)
    u2 = _unit(n2 - c)
    bis = _unit(-(u1 + u2))
    perp = _unit(np.cross(u1, u2))
    d1 = np.cos(half_hch) * bis + np.sin(half_hch) * perp
    d2 = np.cos(half_hch) * bis - np.sin(half_hch) * perp
    return (c + l_ch * d1) % box, (c + l_ch * d2) % box


def _place_h3(c, n1, l_ch, hcc, box):
    """Ideal CH3 completion: three H's at angle hcc to the C->n1
    back-direction, azimuths 0/120/240 deg."""
    u = _unit(n1 - c)          # C -> carbon neighbor
    back = -u
    # any perpendicular frame
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(u, a))
    e2 = _unit(np.cross(u, e1))
    out = []
    for m in range(3):
        phi = np.radians(120.0 * m)
        d = np.cos(hcc) * back + np.sin(hcc) * (np.cos(phi) * e1 + np.sin(phi) * e2)
        out.append((c + l_ch * d) % box)
    return out


def rehydrogenate(ua: AtomisticSystem, aa_ref: AtomisticSystem) -> AtomisticSystem:
    """Rebuild the all-atom system from a UA melt: carbons keep their UA
    positions/types (end CH3 vs interior CH2), hydrogens are placed at
    ideal sp3 geometry and thermalize by local MC afterwards.

    `aa_ref` provides the OPLS-AA parameter tables and type conventions
    (end C / interior C / H) of the original all-atom system.
    """
    # by-example type tables from the reference system
    bond_t: Dict[Tuple[int, int], int] = {}
    for i, j, t in aa_ref.bonds:
        key = tuple(sorted((int(aa_ref.type_ids[i]), int(aa_ref.type_ids[j]))))
        bond_t.setdefault(key, t)
    ang_t: Dict[Tuple[int, int, int], int] = {}
    for i, j, k, t in aa_ref.angles:
        key = (int(aa_ref.type_ids[i]), int(aa_ref.type_ids[j]), int(aa_ref.type_ids[k]))
        ang_t.setdefault(key, t)
        ang_t.setdefault(key[::-1], t)
    dih_t: Dict[Tuple[int, int, int, int], int] = {}
    for i, j, k, l, t in aa_ref.dihedrals:
        key = tuple(int(aa_ref.type_ids[x]) for x in (i, j, k, l))
        dih_t.setdefault(key, t)
        dih_t.setdefault(key[::-1], t)

    h_type = next(t for t, m in aa_ref.masses.items() if m < 4.0)
    ch_bond = bond_t[tuple(sorted((2, h_type)))]
    l_ch = float(aa_ref.bond_r0[ch_bond])
    # H-C-H and H-C-C reference angles
    hch = None
    hcc = None
    for key, t in ang_t.items():
        if key == (h_type, 2, h_type):
            hch = float(aa_ref.angle_t0[t])
        if key in ((h_type, 2, 2), (2, 2, h_type)):
            hcc = float(aa_ref.angle_t0[t])
        if hch is not None and hcc is not None:
            break
    assert hch is not None and hcc is not None, "AA ref must contain C-H angle tables"

    box = float(ua.box_size)
    n_c = ua.n_atoms
    # carbon adjacency (UA bonds)
    adj: Dict[int, List[int]] = {i: [] for i in range(n_c)}
    for i, j, _ in ua.bonds:
        adj[i].append(j)
        adj[j].append(i)

    positions = [np.asarray(p, dtype=float) for p in ua.positions]
    mol_ids = list(ua.mol_ids)
    type_ids = list(ua.type_ids)
    charges = []
    chg_by_type = {t: float(np.mean(aa_ref.charges[aa_ref.type_ids == t])) for t in (1, 2)}
    for t in type_ids:
        charges.append(chg_by_type[int(t)])
    h_charge = float(np.mean(aa_ref.charges[aa_ref.type_ids == h_type]))

    bonds: List[Tuple[int, int, int]] = []
    for i, j, _ in ua.bonds:
        key = tuple(sorted((int(ua.type_ids[i]), int(ua.type_ids[j]))))
        bonds.append((i, j, bond_t[key]))

    # hydrogen placement
    for c in range(n_c):
        nbs = adj[c]
        cpos = np.asarray(ua.positions[c], dtype=float)
        if len(nbs) == 2:
            n1 = np.asarray(ua.positions[nbs[0]], dtype=float)
            n2 = np.asarray(ua.positions[nbs[1]], dtype=float)
            # minimum-image neighbor vectors (bonds may cross the box)
            n1 = cpos + (n1 - cpos + box / 2) % box - box / 2
            n2 = cpos + (n2 - cpos + box / 2) % box - box / 2
            hs = _place_h2(cpos, n1, n2, l_ch, hch / 2.0, box)
        elif len(nbs) == 1:
            n1 = np.asarray(ua.positions[nbs[0]], dtype=float)
            n1 = cpos + (n1 - cpos + box / 2) % box - box / 2
            hs = _place_h3(cpos, n1, l_ch, hcc, box)
        else:
            raise ValueError(f"carbon {c} has {len(nbs)} backbone neighbors")
        for hp in hs:
            h_idx = len(positions)
            positions.append(np.asarray(hp, dtype=float))
            mol_ids.append(mol_ids[c])
            type_ids.append(h_type)
            charges.append(h_charge)
            bonds.append((c, h_idx, ch_bond))

    n_all = len(positions)
    # full adjacency with H's
    adj2: Dict[int, List[int]] = {i: [] for i in range(n_all)}
    for i, j, _ in bonds:
        adj2[i].append(j)
        adj2[j].append(i)

    angles: List[Tuple[int, int, int, int]] = []
    for j in range(n_all):
        nb = adj2[j]
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                i, k = nb[a], nb[b]
                key = (int(type_ids[i]), int(type_ids[j]), int(type_ids[k]))
                t = ang_t.get(key) or ang_t.get(key[::-1])
                assert t is not None, f"no angle type for {key}"
                angles.append((i, j, k, t))
    dihedrals: List[Tuple[int, int, int, int, int]] = []
    for j, k, _ in bonds:
        for i in adj2[j]:
            if i == k:
                continue
            for l in adj2[k]:
                if l == j or l == i:
                    continue
                key = tuple(int(type_ids[x]) for x in (i, j, k, l))
                t = dih_t.get(key) or dih_t.get(key[::-1])
                assert t is not None, f"no dihedral type for {key}"
                dihedrals.append((i, j, k, l, t))

    masses = dict(aa_ref.masses)
    return AtomisticSystem(
        positions=np.asarray(positions),
        mol_ids=np.asarray(mol_ids),
        type_ids=np.asarray(type_ids),
        charges=np.asarray(charges),
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        box=ua.box,
        pair_eps=aa_ref.pair_eps.copy(),
        pair_sig=aa_ref.pair_sig.copy(),
        bond_k=aa_ref.bond_k.copy(),
        bond_r0=aa_ref.bond_r0.copy(),
        angle_k=aa_ref.angle_k.copy(),
        angle_t0=aa_ref.angle_t0.copy(),
        dih_k=aa_ref.dih_k.copy(),
        coul_cut=aa_ref.coul_cut,
        lj_cut=aa_ref.lj_cut,
        scale14_lj=aa_ref.scale14_lj,
        scale14_coul=aa_ref.scale14_coul,
        masses=masses,
    )

