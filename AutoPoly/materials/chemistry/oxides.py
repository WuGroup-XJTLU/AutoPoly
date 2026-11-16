#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metal-oxide chemistry (alumina, titania): finite clusters and validation.

Roles produced (force-field agnostic):

- cation role ("Al" or "Ti"),
- "OB": oxygen without hydrogen (bridging / bulk / dangling-dry),
- "OH" / "HO": hydroxyl oxygen and its hydrogen (one H on the O),
- "OW" / "HW": aqua-ligand oxygen and its hydrogens (two H on the O).

Cluster construction (build_oxide_cluster):

1. Cation-centered carve: keep every cation within the cut radius and all
   O bonded to a kept cation, so every cation keeps its bulk octahedral
   coordination (6 O).
2. Remove loosely attached cations (fewer than ``min_bridging`` O shared
   with other kept cations) iteratively, then O left without a cation;
   keep the largest connected component.
3. Termination:

   - "hydroxylated": protonate the under-coordinated oxygens until the
     cluster is formally neutral (cation +v, O -2, H +1).  Hydrogens go to
     the O with the largest Pauling bond-valence deficit
     (2 - sum(v/CN_cation) - n_H; v/CN = 0.5 for Al(VI), 2/3 for Ti(VI)),
     at most two per O; ties go to the outermost O.  For alumina the CLAYFF
     charges of ao/ob/oh-ho/o*-h* are exactly 0.525 x these formal charges,
     so formal neutrality is CLAYFF neutrality.
   - "stoichiometric" (alumina): dry cut -- the least-coordinated (then
     outermost) oxygens are removed until n_O = v * n_cation / 2; the
     surface cations become under-coordinated.

4. Hydrogens are placed 1.0 A from their O along the direction of
   largest clearance from all other atoms (second H of an aqua ligand: same
   rule, at least ~1.5 A from the first H).

Facet-specific terminations (e.g. rutile(110) bridging O rows) are not
modeled: the termination is facet-agnostic and reported as such.

Created on 2026-11-13
@author: zwu
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...core.exceptions import ValidationError
from ...core.system import logger
from ..structure import AtomicStructure
from ..topology import largest_component, neighbor_pairs

DEFAULT_OH_LENGTH = 1.0


@dataclass(frozen=True)
class OxideRules:
    """Chemistry parameters of one metal oxide."""
    material_id: str
    cation: str
    valence: int                 # formal cation charge
    cation_cn: int               # bulk cation coordination
    anion_cn: int                # bulk O coordination
    cutoff: float                # cation-O bond perception cutoff (A)
    min_bridging: int            # loose-cation removal threshold
    max_h_per_o: int = 2
    bulk_density: float = 4.0    # g/cm^3 (informational; crystal used)

    @property
    def bond_valence(self) -> float:
        return self.valence / self.cation_cn

    @property
    def formula_mass(self) -> float:
        """Mass of one cation's share of the oxide (MO_{v/2}) (amu)."""
        from ..structure import element_mass
        return element_mass(self.cation) + self.valence / 2 * element_mass("O")


RULES: Dict[str, OxideRules] = {
    "al2o3": OxideRules("al2o3", "Al", 3, 6, 4, 2.1, 3, 2, 3.99),
    "tio2": OxideRules("tio2", "Ti", 4, 6, 3, 2.2, 3, 2, 3.9),
}

#: Bond perception cutoffs per material (structure import/validation).
BOND_CUTOFFS = {("Al", "O"): 2.1, ("Ti", "O"): 2.2, ("O", "H"): 1.2}
#: Default bulk density for imported clusters (alumina) (g/cm^3).
BULK_DENSITY = 3.99


def rules_for(structure_or_id) -> OxideRules:
    if isinstance(structure_or_id, str):
        key = structure_or_id
    else:
        els = set(structure_or_id.elements)
        key = "al2o3" if "Al" in els else "tio2" if "Ti" in els else None
    if key not in RULES:
        raise ValidationError(f"No oxide chemistry rules for '{key}'")
    return RULES[key]


# ----------------------------------------------------------------------
# Roles / validation (chemistry-adapter interface)
# ----------------------------------------------------------------------
def assign_roles(elements: Sequence[str], bonds) -> List[str]:
    n_h = [0] * len(elements)
    for i, j in bonds:
        if elements[i] == "H" and elements[j] == "O":
            n_h[j] += 1
        elif elements[j] == "H" and elements[i] == "O":
            n_h[i] += 1
    h_owner = {}
    for i, j in bonds:
        if elements[i] == "H":
            h_owner[i] = j
        elif elements[j] == "H":
            h_owner[j] = i
    roles = []
    for k, e in enumerate(elements):
        if e == "O":
            roles.append({0: "OB", 1: "OH"}.get(n_h[k], "OW"))
        elif e == "H":
            o = h_owner.get(k)
            roles.append("HW" if o is not None and n_h[o] >= 2 else "HO")
        else:
            roles.append(e)
    return roles


def role_bond_type(role_a: str, role_b: str) -> str:
    pair = {role_a, role_b}
    if pair & {"HO", "HW"}:
        return "o_h"
    return "m_o"


def validate(structure: AtomicStructure, **kwargs) -> Dict[str, object]:
    """
    Valence and formal-charge validation of an oxide cluster.

    Checks: every H bonded to exactly one O; every O bonded to >= 1 cation
    and <= 2 H; cation coordination within [3, bulk CN]; formal charge 0.
    """
    rules = rules_for(structure)
    el = structure.elements
    nbrs = structure.neighbors()
    issues = []
    cn_hist: Dict[str, int] = {}
    for i, e in enumerate(el):
        ne = [el[j] for j in nbrs[i]]
        if e == "H":
            if ne != ["O"]:
                issues.append(f"H {i} bonded to {ne}")
        elif e == "O":
            k = ne.count(rules.cation)
            h = ne.count("H")
            if k < 1 or h > 2 or len(ne) != k + h:
                issues.append(f"O {i} has neighbors {ne}")
            cn_hist[f"O_mu{k}"] = cn_hist.get(f"O_mu{k}", 0) + 1
        elif e == rules.cation:
            k = ne.count("O")
            if not 3 <= k <= rules.cation_cn or len(ne) != k:
                issues.append(f"{e} {i} has {k} O neighbors ({ne})")
            cn_hist[f"{e}_cn{k}"] = cn_hist.get(f"{e}_cn{k}", 0) + 1
        else:
            issues.append(f"atom {i}: unexpected element '{e}'")
    n_m = el.count(rules.cation)
    q = rules.valence * n_m - 2 * el.count("O") + el.count("H")
    if q != 0:
        issues.append(f"formal charge {q:+d} (cation +{rules.valence}, "
                      "O -2, H +1)")
    if issues:
        raise ValidationError(
            f"{rules.material_id} structure failed chemical validation:\n"
            "  - " + "\n  - ".join(issues[:20])
            + (f"\n  ... ({len(issues) - 20} more)" if len(issues) > 20
               else "")
        )
    roles = structure.roles or assign_roles(el, structure.bonds)
    return {
        "coordination": dict(sorted(cn_hist.items())),
        "formal_charge": q,
        "counts": {r: roles.count(r) for r in sorted(set(roles))},
        "metadata": {"termination_checked": True},
    }


def core_mass(structure: AtomicStructure) -> float:
    rules = rules_for(structure)
    return structure.elements.count(rules.cation) * rules.formula_mass


def surface_flags(structure: AtomicStructure) -> List[bool]:
    rules = rules_for(structure)
    el = structure.elements
    nbrs = structure.neighbors()
    out = []
    for i, e in enumerate(el):
        if e == "H":
            out.append(True)
        elif e == "O":
            out.append(sum(el[j] == rules.cation for j in nbrs[i])
                       < rules.anion_cn or any(el[j] == "H" for j in nbrs[i]))
        else:
            out.append(sum(el[j] == "O" for j in nbrs[i]) < rules.cation_cn
                       or any(sum(el[x] == "H" for x in nbrs[j]) > 0
                              for j in nbrs[i]))
    return out


# ----------------------------------------------------------------------
# Cluster construction
# ----------------------------------------------------------------------
def _neighbor_sets(n, bonds):
    nbrs = [set() for _ in range(n)]
    for i, j in bonds:
        nbrs[i].add(j)
        nbrs[j].add(i)
    return nbrs


def _cation_o_bonds(species, pos, rules):
    i, j, _, d = neighbor_pairs(pos, rules.cutoff)
    bonds = []
    for a, b in zip(i, j):
        if {species[a], species[b]} == {rules.cation, "O"}:
            bonds.append((int(a), int(b)))
    return bonds


def _compact(species, pos, bonds, keep):
    keep = np.asarray(keep, dtype=bool)
    remap = {old: new for new, old in enumerate(np.where(keep)[0])}
    return (np.asarray(species)[keep], np.asarray(pos)[keep],
            [(remap[i], remap[j]) for i, j in bonds
             if i in remap and j in remap])


def build_oxide_cluster(
    material_id: str,
    crystal,
    radius: float,
    termination: str = "hydroxylated",
    oh_length: float = DEFAULT_OH_LENGTH,
    center_offset: Optional[Sequence[float]] = None,
    phase: Optional[str] = None,
) -> AtomicStructure:
    """
    Finite metal-oxide particle carved from a bulk crystal (see module
    docstring for the algorithm).

    Raises:
        ValidationError: Unknown termination, too small radius, or a
            termination that cannot reach formal neutrality.
    """
    rules = rules_for(material_id)
    if termination not in ("hydroxylated", "stoichiometric"):
        raise ValidationError(
            f"Unknown {material_id} termination '{termination}' "
            "(use 'hydroxylated' or 'stoichiometric')"
        )
    if termination == "stoichiometric" and material_id != "al2o3":
        raise ValidationError(
            "The 'stoichiometric' (dry) termination is only implemented for "
            "alumina"
        )
    if radius < 4.0:
        raise ValidationError(f"Oxide particle radius {radius} A too small")
    el, pos = crystal.replicate_around_origin(radius + 4.0)
    el = np.array(el)
    if center_offset is not None:
        pos = pos - np.asarray(center_offset, dtype=float)
    r = np.linalg.norm(pos, axis=1)
    sel = ((el == rules.cation) & (r <= radius)) | \
        ((el == "O") & (r <= radius + rules.cutoff + 0.1))
    species, spos = el[sel], pos[sel]
    bonds = _cation_o_bonds(species, spos, rules)
    nb = _neighbor_sets(len(species), bonds)
    keep = np.array([s == rules.cation or len(nb[k]) > 0
                     for k, s in enumerate(species)])
    species, spos, bonds = _compact(species, spos, bonds, keep)
    n_carved = int(np.sum(species == rules.cation))
    cn = np.array([len(x) for x in _neighbor_sets(len(species), bonds)])
    if np.any(cn[species == rules.cation] != rules.cation_cn):
        raise ValidationError(
            f"{material_id} carve left cations below bulk coordination "
            "(crystal/cutoff inconsistent)"
        )

    # 2. loosely attached cations
    removed = 0
    while True:
        nb = _neighbor_sets(len(species), bonds)
        loose = []
        for k in np.where(species == rules.cation)[0]:
            bridging = sum(1 for o in nb[k] if len(nb[o]) >= 2)
            if bridging < rules.min_bridging:
                loose.append(k)
        if not loose:
            break
        drop = set(loose)
        for k in loose:
            for o in nb[k]:
                if nb[o] <= drop:
                    drop.add(o)
        keep = np.ones(len(species), dtype=bool)
        keep[list(drop)] = False
        removed += len(loose)
        species, spos, bonds = _compact(species, spos, bonds, keep)
    main = largest_component(len(species), bonds)
    if len(main) < len(species):
        keep = np.zeros(len(species), dtype=bool)
        keep[main] = True
        species, spos, bonds = _compact(species, spos, bonds, keep)
    if not np.any(species == rules.cation):
        raise ValidationError(f"{material_id} particle dissolved; enlarge R")

    species_l = [str(s) for s in species]
    pos_l = [np.asarray(p, dtype=float) for p in spos]
    bonds_l = list(bonds)
    n_m = species_l.count(rules.cation)

    if termination == "stoichiometric":
        species_l, pos_l, bonds_l = _dry_stoichiometric(
            species_l, pos_l, bonds_l, rules)
    else:
        _protonate(species_l, pos_l, bonds_l, rules, oh_length)

    arr = np.array(pos_l)
    roles = assign_roles(species_l, bonds_l)
    structure = AtomicStructure(
        elements=species_l, positions=arr, bonds=bonds_l, roles=roles,
        geometry_type="cluster",
    )
    structure.surface = surface_flags(structure)
    norms = np.linalg.norm(arr, axis=1)
    structure.normals = np.where(
        np.array(structure.surface)[:, None],
        arr / np.maximum(norms, 1e-9)[:, None], 0.0)
    structure.site_types = [
        {"OH": "hydroxyl", "OW": "aqua", "HO": "hydroxyl", "HW": "aqua"}.get(r)
        for r in roles]
    n_m = species_l.count(rules.cation)
    core_volume = n_m * rules.formula_mass / (crystal.density() * 0.602214076)
    o_surface = [k for k, (e, f) in enumerate(zip(species_l, structure.surface))
                 if e == "O" and f]
    r_o = float(np.mean(norms[o_surface])) if o_surface else float(radius)
    area = 4.0 * np.pi * r_o ** 2 / 100.0
    structure.descriptors = {
        "envelope": {"shape": "sphere", "radius": float(norms.max()),
                     "center": [0.0, 0.0, 0.0]},
        "nominal_radius": float(radius),
        "core_volume": float(core_volume),
        "core_radius_equivalent": float(
            (3.0 * core_volume / (4.0 * np.pi)) ** (1.0 / 3.0)),
        "accessible_area_nm2": float(area),
    }
    structure.provenance = {
        "builder": "oxide_cluster",
        "crystal": crystal.name,
        "crystal_citation": crystal.citation,
        "center_offset": None if center_offset is None
        else [float(x) for x in center_offset],
        "oh_length": float(oh_length),
        "termination_model": "bond-valence protonation to formal neutrality"
        if termination == "hydroxylated" else "dry stoichiometric cut",
    }
    n_oh = roles.count("OH")
    n_ow = roles.count("OW")
    structure.metadata = {
        "material_id": material_id,
        "phase": phase or crystal.name,
        "termination": termination,
        "n_cation_carved": n_carved,
        "n_cation_removed_loose": removed,
        "n_hydroxyl": n_oh,
        "n_aqua": n_ow,
        "hydroxyl_density": (n_oh + 2 * n_ow) / area,
        "facet_specific": False,
    }
    structure.annotate_components()
    report = validate(structure)
    structure.metadata["coordination"] = report["coordination"]
    logger.info(
        f"{material_id} particle (R={radius} A, {crystal.name}, "
        f"{termination}): {structure.n_atoms} atoms ({n_m} {rules.cation}), "
        f"{n_oh} OH + {n_ow} OH2 ligands; coordination "
        f"{report['coordination']}"
    )
    return structure


def _protonate(species_l, pos_l, bonds_l, rules: OxideRules, oh_length):
    """Bond-valence-guided protonation to formal neutrality (in place)."""
    n_m = species_l.count(rules.cation)
    n_o = species_l.count("O")
    need = 2 * n_o - rules.valence * n_m
    if need < 0:
        raise ValidationError(
            f"{rules.material_id} cut is cation-rich (needs {need} H); "
            "cannot hydroxylate to neutrality"
        )
    nb = _neighbor_sets(len(species_l), bonds_l)
    o_idx = [k for k, e in enumerate(species_l) if e == "O"]
    k_o = {o: sum(species_l[j] == rules.cation for j in nb[o]) for o in o_idx}
    h_on = {o: 0 for o in o_idx}
    radius = {o: float(np.linalg.norm(pos_l[o])) for o in o_idx}
    for _ in range(need):
        best = max(
            (o for o in o_idx if h_on[o] < rules.max_h_per_o),
            key=lambda o: (2.0 - rules.bond_valence * k_o[o] - h_on[o],
                           radius[o]),
            default=None,
        )
        if best is None:
            raise ValidationError(
                f"{rules.material_id}: cannot place {need} H with at most "
                f"{rules.max_h_per_o} H per O"
            )
        h_on[best] += 1
    arr = np.array(pos_l)
    for o in o_idx:
        placed = []
        for _ in range(h_on[o]):
            d = _clear_direction(arr, o, oh_length, avoid=placed)
            h = len(species_l)
            species_l.append("H")
            p = pos_l[o] + oh_length * d
            pos_l.append(p)
            bonds_l.append((o, h))
            placed.append(p)
            arr = np.vstack([arr, p])


def _clear_direction(arr, o, length, avoid=()):
    """Direction from atom o with the largest clearance at ``length``."""
    from ..topology import clearest_direction
    return clearest_direction(arr, o, length, avoid=avoid)


def _dry_stoichiometric(species_l, pos_l, bonds_l, rules: OxideRules):
    """Remove least-coordinated O (then cation parity) to stoichiometry."""
    species = np.array(species_l)
    pos = np.array(pos_l)
    bonds = list(bonds_l)
    if (rules.valence * int(np.sum(species == rules.cation))) % 2:
        # odd: drop the outermost cation first
        cats = np.where(species == rules.cation)[0]
        far = cats[np.argmax(np.linalg.norm(pos[cats], axis=1))]
        keep = np.ones(len(species), dtype=bool)
        keep[far] = False
        species, pos, bonds = _compact(species, pos, bonds, keep)
    while True:
        n_m = int(np.sum(species == rules.cation))
        n_o = int(np.sum(species == "O"))
        excess = n_o - rules.valence * n_m // 2
        if excess <= 0:
            break
        nb = _neighbor_sets(len(species), bonds)
        o_idx = np.where(species == "O")[0]
        k = np.array([len(nb[o]) for o in o_idx])
        r = np.linalg.norm(pos[o_idx], axis=1)
        order = np.lexsort((-r, k))
        victim = o_idx[order[0]]
        keep = np.ones(len(species), dtype=bool)
        keep[victim] = False
        species, pos, bonds = _compact(species, pos, bonds, keep)
    if excess < 0:
        raise ValidationError(
            f"{rules.material_id} dry cut became O-deficient; adjust radius"
        )
    nb = _neighbor_sets(len(species), bonds)
    bad = [k for k in np.where(species == rules.cation)[0] if len(nb[k]) < 3]
    if bad:
        raise ValidationError(
            f"{rules.material_id} dry cut left {len(bad)} cations with < 3 O; "
            "use the 'hydroxylated' termination or another radius"
        )
    return [str(s) for s in species], [p for p in pos], bonds
