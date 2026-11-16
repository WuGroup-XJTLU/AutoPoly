#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Material qualification matrix: which exact combinations are md_ready.

A combination is ``md_ready`` only when *all* of the following were
independently tested for exactly that model:

    (A) registry metadata       (B) geometry/topology
    (C) surface termination     (D) inorganic parameterization
    (E) polymer cross terms     (F) LAMMPS smoke test

QUALIFIED_COMBINATIONS lists those combinations together with the test
evidence that exercises them (tests/test_nanocomposite_lammps_smoke.py
and friends).  Anything not listed is *not* md_ready: MD preparation of a
nanoparticle fails with an actionable error instead of falling back to
generic organic typing or guessed mixing.  Material-level capability
(``md_ready`` in the registry) is derived from this table, never declared.

Built-in substrate slabs (SubstrateSpec(builder=...)) keep their legacy
behavior: they are reported against this matrix but never blocked.

Created on 2026-11-13
@author: zwu
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..core.exceptions import ValidationError

QUALIFICATION_STAGES = {
    "A": "registry metadata",
    "B": "geometry/topology",
    "C": "surface termination",
    "D": "inorganic parameterization",
    "E": "polymer cross interactions",
    "F": "LAMMPS smoke test",
}

#: Organic host force fields with plain/hybrid 12-6 LJ + long-range
#: Coulomb styles that the explicit cross-term writer supports.
#: (lopls excluded: its polymer typing fails in moltemplate on v2.0, so no
#: lopls combination can be smoke-tested.)
LJ_HOSTS = ("oplsaa", "gaff", "gaff2")


@dataclass(frozen=True)
class QualifiedCombination:
    """
    One md_ready combination.

    Attributes:
        id: Stable identifier (reported in units.json / the FF report).
        material: Material ID.
        phases: Phase IDs covered (structure sources).
        geometry_types: Geometry types covered ("cluster", "sheet", ...).
        surface: Surface-termination ID.
        inorganic_ff: Inorganic adapter name.
        polymer_ffs: Organic host force fields covered.
        motions: Motion treatments covered.
        cross_rule: Rule for organic-inorganic pairs ("host" = follow the
            host force field's mixing rule; otherwise explicit).
        evidence: Test node IDs exercising (B)-(F).
        grafting: Grafting linkers covered ("silyl_ether"); empty = only
            ungrafted particles.
        notes: Caveats.
    """
    id: str
    material: str
    phases: Tuple[str, ...]
    geometry_types: Tuple[str, ...]
    surface: str
    inorganic_ff: str
    polymer_ffs: Tuple[str, ...]
    motions: Tuple[str, ...]
    cross_rule: str
    evidence: Tuple[str, ...] = ()
    grafting: Tuple[str, ...] = ()
    notes: str = ""

    def covers(self, material, phase, geometry_type, surface, inorganic_ff,
               polymer_ff, motion, linker=None) -> bool:
        return (
            material == self.material
            and (phase in self.phases or "*" in self.phases)
            and geometry_type in self.geometry_types
            and surface == self.surface
            and inorganic_ff == self.inorganic_ff
            and polymer_ff in self.polymer_ffs
            and motion in self.motions
            and (linker is None) == (not self.grafting)
            and (linker is None or linker in self.grafting)
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "material": self.material,
            "phases": list(self.phases),
            "geometry_types": list(self.geometry_types),
            "surface": self.surface,
            "inorganic_ff": self.inorganic_ff,
            "polymer_ffs": list(self.polymer_ffs),
            "motions": list(self.motions),
            "cross_rule": self.cross_rule,
            "evidence": list(self.evidence),
            "grafting": list(self.grafting),
            "notes": self.notes,
        }


_SMOKE = "tests/test_nanocomposite_lammps_smoke.py::test_lammps_smoke"

QUALIFIED_COMBINATIONS: List[QualifiedCombination] = [
    QualifiedCombination(
        id="sio2-interface-hydroxylated",
        material="sio2",
        phases=("beta_cristobalite", "alpha_quartz", "amorphous"),
        geometry_types=("cluster", "imported"),
        surface="hydroxylated",
        inorganic_ff="interface_silica",
        polymer_ffs=("oplsaa", "gaff", "gaff2"),
        motions=("rigid", "frozen"),
        cross_rule="host",
        evidence=(f"{_SMOKE}[sio2_ps_oplsaa_rigid]",
                  f"{_SMOKE}[sio2_pmma_oplsaa_rigid]",
                  f"{_SMOKE}[sio2_pe_oplsaa_frozen]",
                  f"{_SMOKE}[sio2_pe_gaff_rigid]",
                  f"{_SMOKE}[sio2_pe_gaff2_rigid]",
                  f"{_SMOKE}[sio2_imported_pe_oplsaa_rigid]"),
        notes="P0-P2 reference: INTERFACE FF v1.5 silica, explicit cross "
              "terms with the host rule (OPLS-AA geometric, GAFF arithmetic).",
    ),
    QualifiedCombination(
        id="sio2-interface-grafted",
        material="sio2",
        phases=("beta_cristobalite", "alpha_quartz"),
        geometry_types=("cluster",),
        surface="hydroxylated",
        inorganic_ff="interface_silica",
        polymer_ffs=("oplsaa",),
        motions=("rigid", "frozen"),
        cross_rule="host",
        grafting=("silyl_ether",),
        evidence=(f"{_SMOKE}[sio2_grafted_pe_oplsaa_rigid]",),
        notes="Si-O-C silyl-ether grafts: OPLS-AA bonded terms (CT-OS, "
              "CT-OS-Si, X-CT-OS, X-CT-CT-OS); Si-O-C-X torsions omitted; "
              "charge conserved on the grafted O.",
    ),
    QualifiedCombination(
        id="sio2-clayff-hydroxylated",
        material="sio2",
        phases=("beta_cristobalite", "alpha_quartz", "amorphous"),
        geometry_types=("cluster", "imported"),
        surface="hydroxylated",
        inorganic_ff="clayff_silica",
        polymer_ffs=("oplsaa",),
        motions=("rigid",),
        cross_rule="arithmetic",
        evidence=(f"{_SMOKE}[sio2clayff_pe_oplsaa_rigid]",),
        notes="CLAYFF Lorentz-Berthelot cross terms with OPLS-AA.",
    ),
    QualifiedCombination(
        id="al2o3-clayff-hydroxylated",
        material="al2o3",
        phases=("alpha_corundum",),
        geometry_types=("cluster",),
        surface="hydroxylated",
        inorganic_ff="clayff_alumina",
        polymer_ffs=("oplsaa",),
        motions=("rigid", "frozen"),
        cross_rule="arithmetic",
        evidence=(f"{_SMOKE}[al2o3_pe_oplsaa_rigid]",
                  f"{_SMOKE}[al2o3_pe_oplsaa_frozen]"),
        notes="Facet-agnostic bond-valence hydroxylation (incl. aqua "
              "ligands); CLAYFF charges = 0.525 x formal charges.",
    ),
    QualifiedCombination(
        id="al2o3-clayff-stoichiometric",
        material="al2o3",
        phases=("alpha_corundum",),
        geometry_types=("cluster",),
        surface="stoichiometric",
        inorganic_ff="clayff_alumina",
        polymer_ffs=("oplsaa",),
        motions=("rigid",),
        cross_rule="arithmetic",
        evidence=(f"{_SMOKE}[al2o3dry_pe_oplsaa_rigid]",),
        notes="Dry stoichiometric cut; under-coordinated surface Al.",
    ),
    QualifiedCombination(
        id="c-oplsaa-graphene",
        material="c",
        phases=("graphene", "cnt"),
        geometry_types=("sheet", "rod"),
        surface="h_passivated",
        inorganic_ff="oplsaa_aromatic_carbon",
        polymer_ffs=("oplsaa",),
        motions=("rigid", "frozen"),
        cross_rule="host",
        evidence=(f"{_SMOKE}[graphene_pe_oplsaa_rigid]",
                  f"{_SMOKE}[graphene_pe_oplsaa_frozen]",
                  f"{_SMOKE}[cnt_pe_oplsaa_rigid]"),
        notes="OPLS-AA aromatic CA/HA parameters: qualified with the "
              "OPLS-AA host only.",
    ),
    QualifiedCombination(
        id="au-interface-lj",
        material="au",
        phases=("fcc",),
        geometry_types=("cluster",),
        surface="bare",
        inorganic_ff="interface_metal_au",
        polymer_ffs=("oplsaa", "gaff2"),
        motions=("rigid",),
        cross_rule="host",
        evidence=(f"{_SMOKE}[au_pe_oplsaa_rigid]",
                  f"{_SMOKE}[au_pe_gaff2_rigid]"),
        notes="Effective 12-6 LJ interfacial model (Heinz 2008); no image "
              "charges; never mixed with EAM.",
    ),
    QualifiedCombination(
        id="ag-interface-lj",
        material="ag",
        phases=("fcc",),
        geometry_types=("cluster",),
        surface="bare",
        inorganic_ff="interface_metal_ag",
        polymer_ffs=("oplsaa",),
        motions=("rigid",),
        cross_rule="host",
        evidence=(f"{_SMOKE}[ag_pe_oplsaa_rigid]",),
        notes="Qualified separately from gold.",
    ),
]


def register_qualification(q: QualifiedCombination) -> None:
    """Add a qualified combination (IDs must be unique)."""
    if any(x.id == q.id for x in QUALIFIED_COMBINATIONS):
        raise ValidationError(f"Qualification '{q.id}' already registered")
    QUALIFIED_COMBINATIONS.append(q)


def find_qualification(material: str, phase: Optional[str],
                       geometry_type: str, surface: str, inorganic_ff: str,
                       polymer_ff: str, motion: str,
                       linker: Optional[str] = None
                       ) -> Optional[QualifiedCombination]:
    """The qualified combination covering these settings, or None."""
    for q in QUALIFIED_COMBINATIONS:
        if q.covers(material, phase or "*", geometry_type, surface,
                    inorganic_ff, polymer_ff, motion, linker):
            return q
    return None


def require_md_ready(material: str, phase: Optional[str],
                     geometry_type: str, surface: str, inorganic_ff: str,
                     polymer_ff: str, motion: str,
                     linker: Optional[str] = None) -> QualifiedCombination:
    """
    Return the qualified combination or raise an actionable error naming
    what is missing.
    """
    q = find_qualification(material, phase, geometry_type, surface,
                           inorganic_ff, polymer_ff, motion, linker)
    if q is not None:
        return q
    near = [x for x in QUALIFIED_COMBINATIONS if x.material == material]
    hint = ""
    if near:
        hint = " Qualified combinations for this material:\n" + "\n".join(
            f"  - {x.id}: phases {list(x.phases)}, geometry "
            f"{list(x.geometry_types)}, surface '{x.surface}', FF "
            f"'{x.inorganic_ff}', polymer FFs {list(x.polymer_ffs)}, "
            f"motions {list(x.motions)}"
            + (f", grafting {list(x.grafting)}" if x.grafting else "")
            for x in near
        )
    else:
        hint = (" No combination of this material is qualified for MD "
                "yet; geometry export (xyz/structure JSON) is still "
                "available.")
    raise ValidationError(
        f"Combination not md_ready: material '{material}', phase "
        f"'{phase}', geometry '{geometry_type}', surface '{surface}', "
        f"inorganic FF '{inorganic_ff}', polymer FF '{polymer_ff}', "
        f"motion '{motion}'"
        + (f", grafting linker '{linker}'" if linker else "") + "." + hint
        + "\nTo proceed with an unvalidated model, supply explicit "
          "parameters and pass allow_unqualified=True (recorded as "
          "'user-qualified' in the force-field report)."
    )


def material_md_ready(material: str) -> bool:
    return any(q.material == material for q in QUALIFIED_COMBINATIONS)


def qualification_matrix() -> List[Dict[str, object]]:
    """
    Per-material stage table (A-F) for docs/CI.

    A: registered; B: geometry_ready; C: surface_ready; D: an inorganic
    adapter is registered; E/F: a qualified combination exists.
    """
    from .registry import MATERIAL_REGISTRY
    from .forcefields import registered_inorganic_ffs

    rows = []
    for mid, spec in MATERIAL_REGISTRY.items():
        quals = [q for q in QUALIFIED_COMBINATIONS if q.material == mid]
        rows.append({
            "material": mid,
            "A": True,
            "B": spec.capability("geometry_ready"),
            "C": spec.capability("surface_ready"),
            "D": bool(registered_inorganic_ffs(mid)),
            "E": bool(quals),
            "F": bool(quals) and all(q.evidence for q in quals),
            "md_ready_combinations": [q.id for q in quals],
        })
    return rows


def qualification_markdown() -> str:
    """The qualification matrix as a Markdown table."""
    head = ("| material | (A) registry | (B) geometry | (C) surface | "
            "(D) inorganic FF | (E) cross terms | (F) LAMMPS smoke | "
            "md_ready combinations |")
    sep = "|" + "---|" * 8
    lines = [head, sep]
    for r in qualification_matrix():
        mark = {True: "yes", False: "no"}
        lines.append(
            f"| `{r['material']}` | " + " | ".join(
                mark[bool(r[k])] for k in "ABCDEF")
            + " | " + (", ".join(f"`{c}`" for c in
                                 r["md_ready_combinations"]) or "-")
            + " |"
        )
    return "\n".join(lines)
