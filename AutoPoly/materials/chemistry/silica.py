#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Silica (SiO2) chemistry: bond perception, roles, termination.

Shared by the crystalline substrate slab builders (AutoPoly.surfaces) and
the silica nanoparticle builders (AutoPoly.nanoparticles).  Everything
here is force-field agnostic: the output is an AtomicStructure carrying
chemical roles

- "Si": tetrahedral silicon (4 O neighbors)
- "OB": bridging oxygen (2 Si neighbors)
- "OH": silanol oxygen (1 Si + 1 H neighbor)
- "HO": silanol hydrogen
- "OG": grafted silanol oxygen (1 Si + a covalent bond to a grafted
        chain; its H was removed by the Si-O-C condensation)

which a silica force-field adapter turns into types/charges.

Slab path (carve-and-functionalize, following mBuild's SilicaInterface
recipe, mosdef-hub/mbuild, BSD-3; adapted to crystalline bulks): tile a
periodic surface cell, cleave along the natural cleavage plane, perceive
Si-O bonds (lateral minimum image), strip stray fragments, optionally
bridge dangling top-face oxygens toward a target silanol density, cap the
remaining dangling oxygens with H.  These functions were extracted
unchanged from surfaces/base.py so the built slabs stay identical.

Finite-cluster path (nanoparticles): see build_silica_cluster().

Created on 2026-08-12 (slab algorithm, surfaces/base.py)
Moved on 2026-11-13
@author: zwu
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.exceptions import ValidationError
from ...core.system import logger
from ..structure import AtomicStructure
from ..topology import infer_images, neighbor_pairs

#: Si-O bond perception cutoff (A); Si-O bonds are ~1.6 A, next shell
#: starts above 2.4 A.
SI_O_BOND_CUTOFF = 1.9
#: Max Si-Si distance (lateral minimum image) for forming a Si-O-Si bridge
#: between dangling oxygens (mbuild: 0.45 nm).
BRIDGE_SI_SI_MAX = 4.5
#: z margin reserved on each slab face for the oxygen coating plus the
#: hydroxyl caps (A).  Si are kept inside [Z_MARGIN, thickness-Z_MARGIN],
#: which guarantees the final physical slab fits inside [0, thickness].
Z_MARGIN = 2.7
#: Default O-H cap length (A) when no force field has been chosen.
DEFAULT_OH_LENGTH = 0.945

#: Silica chemical roles.
SILICA_ROLES = ("Si", "OB", "OH", "HO")
#: Element of each role.
ROLE_ELEMENTS = {"Si": "Si", "OB": "O", "OH": "O", "HO": "H"}
#: Masses per role (amu), as used by the slab .lt writer.
ROLE_MASSES = {"Si": 28.0855, "OB": 15.9994, "OH": 15.9994, "HO": 1.008}

#: Bond perception cutoffs for silica structures.
BOND_CUTOFFS = {("Si", "O"): SI_O_BOND_CUTOFF, ("O", "H"): 1.2}


# ----------------------------------------------------------------------
# Elementary steps (shared by slabs and clusters)
# ----------------------------------------------------------------------
def coordination(n: int, bonds) -> np.ndarray:
    coord = np.zeros(n, dtype=int)
    for i, j in bonds:
        coord[i] += 1
        coord[j] += 1
    return coord


def perceive_si_o_bonds(species: np.ndarray, pos: np.ndarray,
                        lx: Optional[float] = None,
                        ly: Optional[float] = None) -> List[Tuple[int, int]]:
    """
    Si-O bonds by distance cutoff, as (si, o) pairs ordered by (si, o).

    lx/ly: lateral periods for slabs (minimum image in x/y); None for a
    finite cluster.
    """
    periodic = lx is not None and ly is not None
    if periodic:
        span = float(np.ptp(pos[:, 2])) + 10.0 if len(pos) else 10.0
        cell = np.diag([lx, ly, span])
        pbc = (True, True, False)
    else:
        cell, pbc = None, (False, False, False)
    i, j, _, _ = neighbor_pairs(pos, SI_O_BOND_CUTOFF, cell, pbc)
    bonds = []
    for a, b in zip(i, j):
        ea, eb = species[a], species[b]
        if ea == "Si" and eb == "O":
            bonds.append((int(a), int(b)))
        elif ea == "O" and eb == "Si":
            bonds.append((int(b), int(a)))
    bonds.sort()
    return bonds


def strip_strays(species, pos, bonds):
    """Keep the largest connected component of the bond graph."""
    neighbors = {}
    for i, j in bonds:
        neighbors.setdefault(i, set()).add(j)
        neighbors.setdefault(j, set()).add(i)
    seen, components = set(), []
    for start in list(neighbors):
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            stack.extend(neighbors.get(node, ()) - comp)
        seen |= comp
        components.append(comp)
    if not components:
        raise ValidationError("Substrate slab: no bonded network found")
    major = max(components, key=len)
    n_dropped = len(species) - len(major)
    if n_dropped:
        logger.info(f"Slab: stripped {n_dropped} stray atoms")
    keep_idx = sorted(major)
    remap = {old: new for new, old in enumerate(keep_idx)}
    return (
        species[keep_idx],
        pos[keep_idx],
        [(remap[i], remap[j]) for i, j in bonds
         if i in remap and j in remap],
    )


def bridge_dangling(species, pos, bonds, dangling, target, seed,
                    lx=None, ly=None, name="silica"):
    """
    Form Si-O-Si bridges to reduce the dangling-O count to ``target``
    (mbuild SilicaInterface._bridge_dangling_Os).

    A bridge merges two dangling oxygens on different, non-adjacent Si
    closer than BRIDGE_SI_SI_MAX: o1 bonds the second Si, o2 is deleted.
    lx/ly: lateral periods (slabs) or None (finite clusters).

    Returns:
        (species, pos, bonds, dangling) with indices remapped.
    """
    neighbors = {}
    for i, j in bonds:
        neighbors.setdefault(i, set()).add(j)
        neighbors.setdefault(j, set()).add(i)

    if len(dangling) <= target:
        return species, pos, bonds, list(dangling)

    rng = np.random.default_rng(seed)
    species = species.copy()
    pos = pos.copy()
    bonds = list(bonds)
    dangling = list(dangling)
    n_bridges = int((len(dangling) - target) / 2)

    for _ in range(n_bridges):
        bridged = False
        attempts = 0
        while not bridged and attempts < 1000:
            attempts += 1
            o1 = dangling[rng.integers(len(dangling))]
            si1 = next(iter(neighbors[o1]))
            order = rng.permutation(len(dangling))
            for k in order:
                o2 = dangling[k]
                if o2 == o1:
                    continue
                si2 = next(iter(neighbors[o2]))
                if si1 == si2:
                    continue
                if neighbors[si1] & neighbors[si2]:
                    continue
                d = pos[si2] - pos[si1]
                if lx is not None:
                    d[0] -= lx * round(d[0] / lx)
                if ly is not None:
                    d[1] -= ly * round(d[1] / ly)
                if np.linalg.norm(d) < BRIDGE_SI_SI_MAX:
                    bonds.append((o1, si2))
                    neighbors[o1].add(si2)
                    neighbors[si2].add(o1)
                    neighbors[si2].discard(o2)
                    neighbors.pop(o2, None)
                    # delete o2: mark and compact at the end
                    pos[o2] = np.nan
                    species[o2] = "X"
                    dangling.remove(o1)
                    dangling.remove(o2)
                    bridged = True
                    break
        if not bridged:
            logger.warning(
                f"{name} slab: no geometrically valid "
                "Si-O-Si bridge pairs found; the surface stays at "
                "its intrinsic termination. Use a larger oh_density "
                "or accept the intrinsic density."
            )
            break

    keep = species != "X"
    remap = {}
    new_idx = 0
    for old, k in enumerate(keep):
        if k:
            remap[old] = new_idx
            new_idx += 1
    species = species[keep]
    pos = pos[keep]
    bonds = [(remap[i], remap[j]) for i, j in bonds
             if i in remap and j in remap]
    dangling = [remap[i] for i in dangling]
    return species, pos, bonds, dangling


def cap_with_h(species_l, pos_l, bonds_l, o_idx, oh_len, direction=None):
    """
    Cap a dangling oxygen with H.

    Default direction: along the Si->O bond (slab convention).  A custom
    unit ``direction`` may be given (curved surfaces).
    """
    if direction is None:
        si_idx = None
        for i, j in bonds_l:
            if i == o_idx and species_l[j] == "Si":
                si_idx = j
                break
            if j == o_idx and species_l[i] == "Si":
                si_idx = i
                break
        if si_idx is None:
            raise ValidationError("Substrate slab: dangling O without Si")
        direction = pos_l[o_idx] - pos_l[si_idx]
        direction = direction / np.linalg.norm(direction)
    h_idx = len(species_l)
    species_l.append("H")
    pos_l.append(pos_l[o_idx] + oh_len * np.asarray(direction))
    bonds_l.append((o_idx, h_idx))
    return h_idx


def assign_roles(species_l: Sequence[str], bonds_l) -> List[str]:
    """Silica roles from connectivity (Si / HO / OB if >= 2 Si else OH)."""
    si_neighbors = {}
    for i, j in bonds_l:
        if species_l[j] == "Si":
            si_neighbors.setdefault(i, set()).add(j)
        if species_l[i] == "Si":
            si_neighbors.setdefault(j, set()).add(i)
    roles = []
    for idx, elem in enumerate(species_l):
        if elem == "Si":
            roles.append("Si")
        elif elem == "H":
            roles.append("HO")
        elif len(si_neighbors.get(idx, ())) >= 2:
            roles.append("OB")
        else:
            roles.append("OH")
    return roles


def role_bond_type(role_a: str, role_b: str) -> str:
    """Bond-type key for a silica bond: "si_o" or "o_h"."""
    return "o_h" if "HO" in (role_a, role_b) else "si_o"


# ----------------------------------------------------------------------
# Slabs
# ----------------------------------------------------------------------
def tile_surface_cell(cell_fractional, a, b, c, nx, ny, thickness):
    """Replicate the surface cell; z as needed for the cut."""
    nz = int(math.ceil(thickness / c)) + 1
    species, positions = [], []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for elem, (fx, fy, fz) in cell_fractional:
                    species.append(elem)
                    positions.append((
                        (ix + fx) * a,
                        (iy + fy) * b,
                        (iz + fz) * c,
                    ))
    return np.array(species), np.array(positions)


def cleave_slab(species, pos, lx, ly, thickness, name="silica"):
    """
    Choose the z-shift whose cut runs along the natural cleavage plane,
    then keep Si in the interior window and O in the buffer-coated range.

    Candidate shifts align each distinct Si layer with the window bottom.
    The winner keeps every Si tetrahedrally coordinated (given the O
    coating) and breaks the fewest bonds (smallest dangling-O count).
    Raises ValidationError if no candidate works.
    """
    z = pos[:, 2]
    si_mask = species == "Si"
    si_z = np.sort(z[si_mask])
    planes = []
    for zv in si_z:
        if not planes or zv - planes[-1] > 0.2:
            planes.append(zv)

    si_top = thickness - Z_MARGIN
    best = None  # (n_bad, n_dangling, keep)
    for p in planes:
        shift = Z_MARGIN - p
        zz = z + shift
        keep = (
            (si_mask & (zz >= Z_MARGIN - 1e-6) & (zz <= si_top))
            | ((species == "O") & (zz >= Z_MARGIN - 1.65)
               & (zz <= si_top + 1.65))
        )
        if not np.any(keep & si_mask):
            continue
        sub_species = species[keep]
        sub_pos = pos[keep].copy()
        sub_pos[:, 2] = zz[keep]
        bonds = perceive_si_o_bonds(sub_species, sub_pos, lx, ly)
        coord = coordination(len(sub_species), bonds)
        sub_si = np.where(sub_species == "Si")[0]
        n_bad = int(np.sum(coord[sub_si] != 4))
        n_dangling = int(np.sum(coord[sub_species == "O"] != 2))
        candidate = (n_bad, n_dangling, keep)
        if best is None or (n_bad, n_dangling) < best[:2]:
            best = candidate

    if best is None or best[0] > 0:
        raise ValidationError(
            f"{name}: no cleavage plane keeps all Si "
            "tetrahedrally coordinated for thickness "
            f"{thickness} A; increase the slab thickness"
        )
    return best[2]


def build_silica_slab(
    cell_fractional,
    cell_a: float,
    cell_b: float,
    cell_c: float,
    lx_target: float,
    ly_target: float,
    thickness: float,
    oh_density: float = 4.6,
    hydroxylate_bottom: bool = True,
    seed: int = 42,
    oh_length: float = DEFAULT_OH_LENGTH,
    name: str = "silica",
    phase: Optional[str] = None,
) -> AtomicStructure:
    """
    Build a hydroxylated, laterally periodic silica slab (FF agnostic).

    Lateral sizes snap to integer surface cells; coordinates are centered
    laterally at the origin (wrapped into [-l/2, l/2)) and at the slab
    mid-plane in z.  Roles are annotated; metadata records silanol counts,
    cell multiples and the achieved top silanol density.

    Raises:
        ValidationError: If the cleave cannot keep every Si 4-coordinated.
    """
    nx = max(1, int(round(lx_target / cell_a)))
    ny = max(1, int(round(ly_target / cell_b)))
    lx, ly = nx * cell_a, ny * cell_b

    species, pos = tile_surface_cell(cell_fractional, cell_a, cell_b,
                                     cell_c, nx, ny, thickness)
    keep = cleave_slab(species, pos, lx, ly, thickness, name)
    species, pos = species[keep], pos[keep]
    bonds = perceive_si_o_bonds(species, pos, lx, ly)
    species, pos, bonds = strip_strays(species, pos, bonds)

    coord = coordination(len(species), bonds)
    si_idx = np.where(species == "Si")[0]
    if not np.all(coord[si_idx] == 4):
        bad = int(np.sum(coord[si_idx] != 4))
        raise ValidationError(
            f"{name} slab cleave left {bad} Si without "
            "full tetrahedral coordination; increase the slab "
            "thickness"
        )

    z_si_top = pos[si_idx][:, 2].max()
    dangling_top = [
        i for i in np.where(species == "O")[0]
        if coord[i] == 1 and pos[i][2] > z_si_top
    ]

    area_nm2 = lx * ly / 100.0
    target = int(round(oh_density * area_nm2))
    if len(dangling_top) < target:
        logger.warning(
            f"{name} slab has only "
            f"{len(dangling_top)} dangling surface oxygens, "
            f"below the target silanol count {target} "
            f"({oh_density}/nm^2); keeping all"
        )
    species, pos, bonds, dangling_top = bridge_dangling(
        species, pos, bonds, dangling_top, target, seed, lx, ly, name
    )

    # indices were remapped by bridging: recompute coordination and
    # the bottom-face dangling oxygens
    coord = coordination(len(species), bonds)
    dangling_set = set(dangling_top)
    dangling_bottom = [
        i for i in np.where(species == "O")[0]
        if coord[i] == 1 and i not in dangling_set
    ]

    # Hydroxylation: cap remaining dangling oxygens with H
    species_l = list(species)
    pos_l = list(pos)
    bonds_l = list(bonds)
    n_top_capped = 0
    for i in dangling_top:
        cap_with_h(species_l, pos_l, bonds_l, i, oh_length)
        n_top_capped += 1
    n_bottom_capped = 0
    for i in dangling_bottom:
        if hydroxylate_bottom:
            cap_with_h(species_l, pos_l, bonds_l, i, oh_length)
            n_bottom_capped += 1
        # else: leave bare (documented; slab no longer neutral)

    roles = assign_roles(species_l, bonds_l)

    # Center coordinates: lateral at origin, z at the slab mid-plane.
    # Wrap lateral coordinates into [-l/2, l/2): edge hydroxyl caps
    # can point slightly outside the cell (harmless for a periodic
    # slab, but tidy in the data file).
    arr = np.array([np.asarray(p, dtype=float) for p in pos_l])
    arr[:, 0] -= lx / 2.0
    arr[:, 1] -= ly / 2.0
    arr[:, 0] -= lx * np.floor(arr[:, 0] / lx + 0.5)
    arr[:, 1] -= ly * np.floor(arr[:, 1] / ly + 0.5)
    z_mid = 0.5 * (arr[:, 2].min() + arr[:, 2].max())
    arr[:, 2] -= z_mid
    z_extent = float(arr[:, 2].max() - arr[:, 2].min())

    span_z = max(thickness, z_extent) + 20.0
    cell = np.diag([lx, ly, span_z])
    pbc = (True, True, False)
    structure = AtomicStructure(
        elements=[str(e) for e in species_l],
        positions=arr,
        bonds=bonds_l,
        images=infer_images(arr, bonds_l, cell, pbc),
        cell=cell,
        pbc=pbc,
        roles=roles,
        surface=[r in ("OH", "HO") for r in roles],
        site_types=["silanol" if r in ("OH", "HO") else None for r in roles],
        geometry_type="slab",
        descriptors={
            "envelope": {"shape": "slab", "lx": lx, "ly": ly,
                         "thickness": float(thickness)},
            "z_extent": z_extent,
            "accessible_area_nm2": area_nm2 * (2 if hydroxylate_bottom
                                               else 1),
        },
        provenance={
            "builder": "silica_slab",
            "cell_multiples": [nx, ny],
            "seed": int(seed),
            "oh_length": float(oh_length),
        },
        metadata={
            "material_id": "sio2",
            "phase": phase,
            "termination": "hydroxylated",
            "n_silanol_top": n_top_capped,
            "n_silanol_bottom": n_bottom_capped,
            "target_oh_density": float(oh_density),
            "achieved_top_oh_density": n_top_capped / area_nm2,
            "lx": lx,
            "ly": ly,
            "nx": nx,
            "ny": ny,
        },
    )

    logger.info(
        f"{name} slab: {nx}x{ny} cells "
        f"({lx:.3f} x {ly:.3f} A), {structure.n_atoms} atoms, "
        f"z-extent {z_extent:.2f} A, silanols top {n_top_capped} / "
        f"bottom {n_bottom_capped} (achieved top density "
        f"{n_top_capped / area_nm2:.2f}/nm^2, target {oh_density}/nm^2)"
    )
    return structure


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def validate_silica(structure: AtomicStructure,
                    allow_bare_oxygens: bool = False) -> Dict[str, object]:
    """
    Chemical validation of a silica structure (slab or cluster).

    Checks: every Si has 4 O neighbors; every O is bonded to 1-2 Si and
    (silanols) exactly one H; every H is bonded to exactly one O; roles are
    consistent with connectivity; formal-charge neutrality of the SiO2/OH
    composition (2*n_OB + n_OH == 4*n_Si and n_HO == n_OH).

    Returns:
        Report dict (counts, Qn distribution, bond-length range, issues).

    Raises:
        ValidationError: Listing every violated rule.
    """
    el = structure.elements
    nbrs = structure.neighbors()
    issues = []
    q_dist: Dict[str, int] = {}
    for i, e in enumerate(el):
        ne = [el[j] for j in nbrs[i]]
        if e == "Si":
            if len(ne) != 4 or any(x != "O" for x in ne):
                issues.append(f"Si {i} has neighbors {ne} (need 4 O)")
            n_bridging = sum(
                1 for j in nbrs[i]
                if sum(el[k] == "Si" for k in nbrs[j]) >= 2
            )
            q_dist[f"Q{n_bridging}"] = q_dist.get(f"Q{n_bridging}", 0) + 1
        elif e == "O":
            n_si = ne.count("Si")
            n_h = ne.count("H")
            grafted = structure.roles is not None and \
                structure.roles[i] == "OG"
            if not 1 <= n_si <= 2 or n_h > 1 or len(ne) != n_si + n_h:
                issues.append(f"O {i} has neighbors {ne}")
            elif grafted and (n_si != 1 or n_h):
                issues.append(f"grafted O {i} has neighbors {ne}")
            elif n_si == 1 and n_h == 0 and not (allow_bare_oxygens
                                                 or grafted):
                issues.append(f"O {i} is a dangling (unterminated) oxygen")
            elif n_si == 2 and n_h:
                issues.append(f"O {i} is bridging but carries H")
        elif e == "H":
            if ne != ["O"]:
                issues.append(f"H {i} has neighbors {ne} (need 1 O)")
        else:
            issues.append(f"atom {i}: unexpected element '{e}' in silica")
    roles = structure.roles
    if roles is not None:
        expected = assign_roles(el, structure.bonds)
        bad = [i for i, (r, x) in enumerate(zip(roles, expected))
               if r != x and not (r == "OG" and x == "OH")]
        if bad:
            issues.append(
                f"{len(bad)} role annotation(s) inconsistent with "
                f"connectivity (first: atom {bad[0]})"
            )
    else:
        roles = assign_roles(el, structure.bonds)
    n = {r: sum(x == r for x in roles) for r in SILICA_ROLES + ("OG",)}
    if not allow_bare_oxygens:
        if 2 * n["OB"] + n["OH"] + n["OG"] != 4 * n["Si"]:
            issues.append(
                f"stoichiometry: 2*n_OB + n_OH + n_OG = "
                f"{2 * n['OB'] + n['OH'] + n['OG']} != 4*n_Si = {4 * n['Si']}"
            )
        if n["HO"] != n["OH"]:
            issues.append(f"n_HO = {n['HO']} != n_OH = {n['OH']}")
    lengths = structure.bond_lengths()
    report = {
        "counts": n,
        "q_distribution": dict(sorted(q_dist.items())),
        "bond_length_range": [float(lengths.min()), float(lengths.max())]
        if len(lengths) else None,
        "issues": issues,
    }
    if issues:
        raise ValidationError(
            "Silica structure failed chemical validation:\n  - "
            + "\n  - ".join(issues[:20])
            + (f"\n  ... ({len(issues) - 20} more)" if len(issues) > 20
               else "")
        )
    return report


# ----------------------------------------------------------------------
# Finite clusters (nanoparticles)
# ----------------------------------------------------------------------
#: Si-O bond length used when repositioning condensation oxygens (A).
SI_O_LENGTH = 1.62
#: Si-O-H angle for silanol caps on curved surfaces (deg; INTERFACE FF
#: equilibrium value region, ~115-118 deg).
SI_O_H_ANGLE = 116.0
#: Minimum distance of a new cap H / bridge O to non-bonded atoms (A).
CAP_CLEARANCE = 1.7
#: Bulk SiO2 formula-unit mass (amu).
SIO2_UNIT_MASS = 28.0855 + 2 * 15.9994


def build_silica_cluster(
    crystal,
    radius: float,
    oh_density: Optional[float] = None,
    oh_length: float = DEFAULT_OH_LENGTH,
    seed: int = 42,
    center_offset: Optional[Sequence[float]] = None,
    phase: Optional[str] = None,
) -> AtomicStructure:
    """
    Carve a hydroxylated finite silica particle from a bulk crystal.

    Algorithm (finite 3D coordination, no slab logic):

    1. Tile the bulk cell around the origin; keep every Si within
       ``radius`` of the particle center and *all* O bonded to a kept Si
       (every Si stays tetrahedral).
    2. Iteratively remove under-bridged silicons (Q0/Q1: fewer than two
       bridging O) together with their now-unbonded oxygens.
    3. If ``oh_density`` (silanols/nm^2) is below the intrinsic density,
       condense pairs of dangling oxygens on different, non-adjacent Si
       (Si-Si <= BRIDGE_SI_SI_MAX) into Si-O-Si bridges; the bridging O is
       repositioned between the two Si (Si-O = 1.62 A where geometrically
       possible) and must clear neighboring atoms.  Achieved vs target
       density is reported; nothing is forced.
    4. Cap every remaining dangling O with H at the Si-O-H angle
       SI_O_H_ANGLE, rotated about the Si-O axis to point outward and clear
       neighboring atoms.
    5. Validate (validate_silica): 4-coordinated Si, 2-valent O, neutral
       SiO2 + SiOH composition.

    Args:
        crystal: CrystalCell (alpha_quartz / beta_cristobalite).
        radius: Si cut radius (A).
        oh_density: Target silanol density (/nm^2); None = intrinsic.
        oh_length: O-H cap length (A).
        seed: RNG seed (condensation pairing).
        center_offset: Optional particle-center shift inside the crystal
            (A) -- varies the exposed surface sites.
        phase: Phase ID recorded in metadata.

    Returns:
        AtomicStructure (geometry_type "cluster", centered on the cut
        center) with roles, surface flags, radial normals, descriptors and
        provenance.

    Raises:
        ValidationError: Radius too small or chemical validation failure.
    """
    if radius < 4.0:
        raise ValidationError(
            f"Silica particle radius {radius} A is too small (>= 4 A)"
        )
    margin = radius + 4.0
    el, pos = crystal.replicate_around_origin(margin)
    el = np.array(el)
    if center_offset is not None:
        pos = pos - np.asarray(center_offset, dtype=float)
    r = np.linalg.norm(pos, axis=1)
    si_all = np.where((el == "Si") & (r <= radius))[0]
    if len(si_all) == 0:
        raise ValidationError(f"No Si within radius {radius} A")
    o_all = np.where((el == "O") & (r <= radius + SI_O_BOND_CUTOFF + 0.1))[0]
    sub = np.concatenate([si_all, o_all])
    species = el[sub]
    spos = pos[sub]
    bonds = perceive_si_o_bonds(species, spos)
    coord = coordination(len(species), bonds)
    keep = (species == "Si") | (coord > 0)
    species, spos, bonds = _compact(species, spos, bonds, keep)
    n_si0 = int(np.sum(species == "Si"))
    if not np.all(coordination(len(species), bonds)[species == "Si"] == 4):
        raise ValidationError(
            "Silica cluster carve left under-coordinated Si (bulk cell or "
            "cutoff inconsistent)"
        )

    # 2. remove Q0/Q1 silicons iteratively
    n_removed_si = 0
    while True:
        nbrs = _neighbor_sets(len(species), bonds)
        si_idx = np.where(species == "Si")[0]
        bridging = {
            i: sum(1 for o in nbrs[i] if len(nbrs[o]) >= 2) for i in si_idx
        }
        weak = [i for i, nb in bridging.items() if nb < 2]
        if not weak:
            break
        drop = set(weak)
        for i in weak:
            for o in nbrs[i]:
                if nbrs[o] <= drop:
                    drop.add(o)
        keep = np.ones(len(species), dtype=bool)
        keep[list(drop)] = False
        n_removed_si += len(weak)
        species, spos, bonds = _compact(species, spos, bonds, keep)
        if not np.any(species == "Si"):
            raise ValidationError(
                f"Silica cluster of radius {radius} A dissolved during Q1 "
                "removal; use a larger radius"
            )
    # keep the largest connected component
    from ..topology import largest_component
    main = largest_component(len(species), bonds)
    if len(main) < len(species):
        keep = np.zeros(len(species), dtype=bool)
        keep[main] = True
        species, spos, bonds = _compact(species, spos, bonds, keep)

    nbrs = _neighbor_sets(len(species), bonds)
    dangling = [i for i in np.where(species == "O")[0] if len(nbrs[i]) == 1]
    r_o = float(np.mean(np.linalg.norm(spos[dangling], axis=1))) \
        if dangling else float(radius)
    area_nm2 = 4.0 * np.pi * r_o ** 2 / 100.0
    intrinsic = len(dangling) / area_nm2

    # 3. condensation toward the target silanol density
    n_condensed = 0
    max_bridge = 0.0
    if oh_density is not None:
        target = int(round(oh_density * area_nm2))
        species, spos, bonds, dangling, n_condensed, max_bridge = \
            _condense_silanols(species, spos, bonds, dangling, target, seed)
        if len(dangling) > target:
            logger.warning(
                f"Silica particle (R={radius} A): condensation reached "
                f"{len(dangling) / area_nm2:.2f} silanols/nm^2 (target "
                f"{oh_density}/nm^2, intrinsic {intrinsic:.2f}); no further "
                "geometrically valid Si-O-Si bridges"
            )
        elif intrinsic < oh_density:
            logger.warning(
                f"Silica particle (R={radius} A): intrinsic silanol density "
                f"{intrinsic:.2f}/nm^2 is below the target {oh_density}/nm^2"
            )

    # 4. hydroxylation
    species_l = [str(s) for s in species]
    pos_l = [np.asarray(p, dtype=float) for p in spos]
    bonds_l = list(bonds)
    nbrs = _neighbor_sets(len(species_l), bonds_l)
    for o in dangling:
        si = next(iter(nbrs[o]))
        direction = _silanol_h_direction(np.array(pos_l), o, si, nbrs)
        cap_with_h(species_l, pos_l, bonds_l, o, oh_length, direction)

    arr = np.array(pos_l)
    roles = assign_roles(species_l, bonds_l)
    nbrs = _neighbor_sets(len(species_l), bonds_l)
    surface = []
    for i, role in enumerate(roles):
        if role in ("OH", "HO"):
            surface.append(True)
        elif role == "Si":
            surface.append(any(roles[j] == "OH" for j in nbrs[i]))
        else:
            surface.append(False)
    norms = np.linalg.norm(arr, axis=1)
    normals = np.where(np.array(surface)[:, None],
                       arr / np.maximum(norms, 1e-9)[:, None], 0.0)
    n_si = roles.count("Si")
    n_oh = roles.count("OH")
    core_volume = n_si * SIO2_UNIT_MASS / (crystal.density() * 0.602214076)
    structure = AtomicStructure(
        elements=species_l,
        positions=arr,
        bonds=bonds_l,
        roles=roles,
        surface=surface,
        site_types=["silanol" if r in ("OH", "HO") else None for r in roles],
        normals=normals,
        geometry_type="cluster",
        descriptors={
            "envelope": {"shape": "sphere",
                         "radius": float(norms.max()),
                         "center": [0.0, 0.0, 0.0]},
            "nominal_radius": float(radius),
            "silanol_shell_radius": r_o,
            "core_volume": float(core_volume),
            "core_radius_equivalent": float(
                (3.0 * core_volume / (4.0 * np.pi)) ** (1.0 / 3.0)),
            "accessible_area_nm2": float(area_nm2),
        },
        provenance={
            "builder": "silica_cluster",
            "crystal": crystal.name,
            "crystal_citation": crystal.citation,
            "seed": int(seed),
            "center_offset": None if center_offset is None
            else [float(x) for x in center_offset],
            "oh_length": float(oh_length),
            "si_o_h_angle": SI_O_H_ANGLE,
        },
        metadata={
            "material_id": "sio2",
            "phase": phase or crystal.name,
            "termination": "hydroxylated",
            "n_si_carved": n_si0,
            "n_si_removed_q1": n_removed_si,
            "n_condensed": n_condensed,
            "max_bridge_si_o": max_bridge,
            "n_silanol": n_oh,
            "intrinsic_oh_density": intrinsic,
            "target_oh_density": oh_density,
            "achieved_oh_density": n_oh / area_nm2,
        },
    )
    structure.annotate_components()
    report = validate_silica(structure)
    structure.metadata["q_distribution"] = report["q_distribution"]
    logger.info(
        f"Silica particle (R={radius} A, {crystal.name}): "
        f"{structure.n_atoms} atoms ({n_si} Si), {n_oh} silanols "
        f"({n_oh / area_nm2:.2f}/nm^2), Q-distribution "
        f"{report['q_distribution']}"
    )
    return structure


def _neighbor_sets(n, bonds):
    nbrs = [set() for _ in range(n)]
    for i, j in bonds:
        nbrs[i].add(j)
        nbrs[j].add(i)
    return nbrs


def _compact(species, pos, bonds, keep):
    """Keep atoms where keep[i]; remap bonds."""
    keep = np.asarray(keep, dtype=bool)
    remap = {old: new for new, old in enumerate(np.where(keep)[0])}
    return (np.asarray(species)[keep], np.asarray(pos)[keep],
            [(remap[i], remap[j]) for i, j in bonds
             if i in remap and j in remap])


def _condense_silanols(species, pos, bonds, dangling, target, seed):
    """
    Si-O-Si condensation of dangling-O pairs (finite cluster, see
    build_silica_cluster step 3).  Returns the compacted arrays, the
    remaining dangling list, the number of bridges and the longest new
    bridge Si-O distance.
    """
    if len(dangling) <= target:
        return species, pos, bonds, list(dangling), 0, 0.0
    rng = np.random.default_rng(seed)
    species = np.array(species, dtype=object)
    pos = np.array(pos, dtype=float)
    bonds = list(bonds)
    nbrs = _neighbor_sets(len(species), bonds)
    dangling = list(dangling)
    deleted = set()
    n_bridges = 0
    max_bridge = 0.0
    want = (len(dangling) - target) // 2
    order = list(rng.permutation(len(dangling)))
    candidates = [dangling[k] for k in order]
    for o1 in candidates:
        if n_bridges >= want:
            break
        if o1 in deleted or o1 not in dangling:
            continue
        si1 = next(iter(nbrs[o1]))
        best = None
        for o2 in dangling:
            if o2 == o1 or o2 in deleted:
                continue
            si2 = next(iter(nbrs[o2]))
            if si2 == si1 or nbrs[si1] & nbrs[si2]:
                continue
            d = float(np.linalg.norm(pos[si2] - pos[si1]))
            if d < BRIDGE_SI_SI_MAX and (best is None or d < best[0]):
                best = (d, o2, si2)
        if best is None:
            continue
        d, o2, si2 = best
        new = _bridge_position(pos[si1], pos[si2])
        others = [k for k in range(len(species))
                  if k not in deleted and k not in (o1, o2, si1, si2)]
        if others:
            dmin = np.linalg.norm(pos[others] - new, axis=1).min()
            if dmin < CAP_CLEARANCE + 0.5:
                continue
        pos[o1] = new
        bonds.append((o1, si2))
        nbrs[o1].add(si2)
        nbrs[si2].add(o1)
        nbrs[si2].discard(o2)
        nbrs[o2] = set()
        deleted.add(o2)
        dangling.remove(o1)
        dangling.remove(o2)
        n_bridges += 1
        max_bridge = max(max_bridge,
                         float(np.linalg.norm(new - pos[si1])),
                         float(np.linalg.norm(new - pos[si2])))
    keep = np.ones(len(species), dtype=bool)
    keep[list(deleted)] = False
    bonds = [(i, j) for i, j in bonds if i not in deleted and j not in deleted]
    remap = {old: new for new, old in enumerate(np.where(keep)[0])}
    species2, pos2, bonds2 = _compact(species, pos, bonds, keep)
    dangling2 = [remap[i] for i in dangling]
    return (np.array([str(s) for s in species2]), pos2, bonds2, dangling2,
            n_bridges, max_bridge)


def _bridge_position(p1, p2):
    """O position between two Si: Si-O = SI_O_LENGTH where possible,
    displaced outward (away from the particle center) otherwise."""
    mid = 0.5 * (p1 + p2)
    half = 0.5 * float(np.linalg.norm(p2 - p1))
    if half >= SI_O_LENGTH:
        return mid
    axis = (p2 - p1) / (2 * half)
    out = mid - np.dot(mid, axis) * axis
    n = np.linalg.norm(out)
    if n < 1e-6:
        out = np.cross(axis, [0.0, 0.0, 1.0])
        if np.linalg.norm(out) < 1e-6:
            out = np.cross(axis, [0.0, 1.0, 0.0])
        n = np.linalg.norm(out)
    return mid + out / n * np.sqrt(SI_O_LENGTH ** 2 - half ** 2)


def _silanol_h_direction(pos, o, si, nbrs):
    """
    Unit vector O->H with Si-O-H = SI_O_H_ANGLE, rotated about the Si-O
    axis to maximize the outward (radial) component while clearing
    non-bonded atoms; falls back to the most outward clash-free or the
    least-clashing orientation.
    """
    b = pos[o] - pos[si]
    b /= np.linalg.norm(b)
    # perpendicular basis
    ref = np.array([1.0, 0.0, 0.0]) if abs(b[0]) < 0.9 else \
        np.array([0.0, 1.0, 0.0])
    u = np.cross(b, ref)
    u /= np.linalg.norm(u)
    v = np.cross(b, u)
    theta = np.radians(180.0 - SI_O_H_ANGLE)
    radial = pos[o] / max(np.linalg.norm(pos[o]), 1e-9)
    near = np.linalg.norm(pos - pos[o], axis=1) < 4.0
    near[o] = False
    near[si] = False
    near_idx = np.where(near)[0]
    best, best_score = None, -np.inf
    for phi in np.linspace(0.0, 2 * np.pi, 36, endpoint=False):
        d = np.cos(theta) * b + np.sin(theta) * (np.cos(phi) * u
                                                 + np.sin(phi) * v)
        h = pos[o] + DEFAULT_OH_LENGTH * d
        clear = (np.linalg.norm(pos[near_idx] - h, axis=1).min()
                 if len(near_idx) else 10.0)
        score = float(np.dot(d, radial)) + (0.0 if clear >= CAP_CLEARANCE
                                            else -10.0 + clear)
        if score > best_score:
            best, best_score = d, score
    return best


# ----------------------------------------------------------------------
# Chemistry-adapter interface (used by imports and stage-2 typing)
# ----------------------------------------------------------------------
#: Density of amorphous silica (g/cm^3) for the core volume of imported
#: clusters.
BULK_DENSITY = 2.2


def validate(structure: AtomicStructure, **kwargs) -> Dict[str, object]:
    """Chemistry-adapter hook: validate_silica plus silanol density."""
    report = validate_silica(structure, **kwargs)
    roles = structure.roles or assign_roles(structure.elements,
                                            structure.bonds)
    oh = [i for i, r in enumerate(roles) if r == "OH"]
    if oh and not structure.is_periodic:
        c = structure.positions.mean(axis=0)
        r_o = float(np.mean(np.linalg.norm(structure.positions[oh] - c,
                                           axis=1)))
        area = 4.0 * np.pi * r_o ** 2 / 100.0
        report["metadata"] = {
            "n_silanol": len(oh),
            "silanol_shell_radius": r_o,
            "achieved_oh_density": len(oh) / area,
            "q_distribution": report["q_distribution"],
            "termination": "hydroxylated",
        }
    return report


def core_mass(structure: AtomicStructure) -> float:
    """Mass of the SiO2 core (amu): one SiO2 unit per Si."""
    return structure.elements.count("Si") * SIO2_UNIT_MASS


def surface_flags(structure: AtomicStructure) -> List[bool]:
    """Silanol O/H and the Si carrying them are surface atoms."""
    roles = structure.roles or assign_roles(structure.elements,
                                            structure.bonds)
    nbrs = structure.neighbors()
    out = []
    for i, r in enumerate(roles):
        if r in ("OH", "HO"):
            out.append(True)
        elif r == "Si":
            out.append(any(roles[j] == "OH" for j in nbrs[i]))
        else:
            out.append(False)
    return out



# ----------------------------------------------------------------------
# Grafting sites
# ----------------------------------------------------------------------
def graft_site_clearance(structure: AtomicStructure, o: int,
                         bond_length: float = 1.41) -> float:
    """
    Room at the future head-atom position of a silanol graft site: the
    distance from O + bond_length * (O->H direction) to the nearest core
    atom other than the site's own O, H and Si (H atoms counted 0.4 A
    closer, i.e. as smaller obstacles).
    """
    nbrs = structure.neighbors()
    el = structure.elements
    h = next(j for j in nbrs[o] if el[j] == "H")
    si = next(j for j in nbrs[o] if el[j] == "Si")
    pos = structure.positions
    d = pos[h] - pos[o]
    site = pos[o] + bond_length * d / np.linalg.norm(d)
    mask = np.ones(structure.n_atoms, dtype=bool)
    mask[[o, h, si]] = False
    dist = np.linalg.norm(pos[mask] - site, axis=1)
    is_h = np.array([e == "H" for e in el])[mask]
    return float(np.min(np.where(is_h, dist + 0.4, dist)))


def select_graft_sites(structure: AtomicStructure, n: int,
                       seed: int = 0, exclude: Sequence[int] = (),
                       min_clearance: float = 2.4) -> List[int]:
    """
    ``n`` silanol oxygens for covalent grafting, spread over the surface by
    farthest-point sampling (Euclidean on the particle surface), at most
    one graft per silicon (geminal pairs are not both grafted), among the
    sites with room for the head atom (graft_site_clearance >=
    min_clearance) and not in ``exclude``.

    Raises:
        ValidationError: Fewer eligible silanols than requested.
    """
    roles = structure.roles or assign_roles(structure.elements,
                                            structure.bonds)
    nbrs = structure.neighbors()
    excluded = set(exclude)
    candidates = [i for i, r in enumerate(roles) if r == "OH"
                  and i not in excluded
                  and graft_site_clearance(structure, i) >= min_clearance]
    si_of = {o: next(j for j in nbrs[o] if structure.elements[j] == "Si")
             for o in candidates}
    if not candidates and n > 0:
        raise ValidationError(
            "No silanol site has room for a grafted chain head"
        )
    if n <= 0:
        return []
    if n > len(set(si_of.values())):
        raise ValidationError(
            f"Requested {n} grafts but only {len(set(si_of.values()))} "
            "silicons carry graftable silanols; lower the grafting density"
        )
    rng = np.random.default_rng(seed)
    pos = structure.positions
    chosen = [candidates[int(rng.integers(len(candidates)))]]
    used_si = {si_of[chosen[0]]}
    dmin = np.linalg.norm(pos[candidates] - pos[chosen[0]], axis=1)
    while len(chosen) < n:
        order = np.argsort(-dmin)
        pick = None
        for k in order:
            o = candidates[k]
            if si_of[o] not in used_si and o not in chosen:
                pick = o
                break
        if pick is None:
            raise ValidationError("Ran out of graftable silanol sites")
        chosen.append(pick)
        used_si.add(si_of[pick])
        dmin = np.minimum(dmin, np.linalg.norm(pos[candidates] - pos[pick],
                                               axis=1))
    return sorted(chosen)


def graft_anchor_structure(structure: AtomicStructure,
                           sites: Sequence[int]) -> Tuple[AtomicStructure,
                                                          Dict[int, Dict]]:
    """
    Copy of a hydroxylated silica structure in which every site O loses
    its H (role "OG", site type "graft_anchor").

    Returns:
        (structure, anchors): anchors maps the *new* O index to
        {"si": Si index, "direction": unit O->H vector of the removed H,
        "h_position": removed H position}.
    """
    roles = structure.roles or assign_roles(structure.elements,
                                            structure.bonds)
    nbrs = structure.neighbors()
    info = {}
    drop = set()
    for o in sites:
        if roles[o] != "OH":
            raise ValidationError(f"atom {o} is not a silanol oxygen")
        h = next(j for j in nbrs[o] if structure.elements[j] == "H")
        si = next(j for j in nbrs[o] if structure.elements[j] == "Si")
        v = structure.positions[h] - structure.positions[o]
        info[o] = {"si": si, "direction": v / np.linalg.norm(v),
                   "h_position": structure.positions[h].copy()}
        drop.add(h)
    keep = [i for i in range(structure.n_atoms) if i not in drop]
    remap = {old: new for new, old in enumerate(keep)}
    out = structure.subset(keep)
    out.roles = ["OG" if remap_old in info else r
                 for remap_old, r in zip(keep, out.roles)]
    out.site_types = ["graft_anchor" if old in info else st
                      for old, st in zip(keep, out.site_types or
                                         [None] * len(keep))]
    anchors = {remap[o]: {"si": remap[d["si"]], "direction": d["direction"],
                          "h_position": d["h_position"]}
               for o, d in info.items()}
    return out, anchors
