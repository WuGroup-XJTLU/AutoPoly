#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inorganic force-field adapters for alumina, carbon, gold and silver.

All values are taken by reference from tables.py.  Every set below is
nonbonded-only (rigid/frozen particles); none is a validated flexible
model.  Titania has deliberately no adapter: no TiO2-polymer interaction
model is bundled, so TiO2 particles stay geometry-only (MD export is
refused with an actionable error).

Created on 2026-11-13
@author: zwu
"""
from .registry import InorganicForceField, RoleParameters, register_inorganic_ff
from .silica import CYGAN_2004
from .tables import (
    CLAYFF_ALUMINA,
    INTERFACE_METALS,
    OH_BOND_LENGTH,
    OPLSAA_AROMATIC_CARBON,
)

HEINZ_2008 = (
    "Heinz, H.; Vaia, R. A.; Farmer, B. L.; Naik, R. R. J. Phys. Chem. C "
    "2008, 112, 17281-17290 (12-6 LJ parameters of fcc metals; "
    "doi:10.1021/jp801931d)"
)
OPLS_AROMATIC = (
    "Jorgensen, W. L.; Severance, D. L. J. Am. Chem. Soc. 1990, 112, 4768 "
    "(aromatic CA/HA); Jorgensen, W. L.; Maxwell, D. S.; Tirado-Rives, J. "
    "J. Am. Chem. Soc. 1996, 118, 11225 (OPLS-AA)"
)


def _roles(table):
    return {r: RoleParameters.from_table(e) for r, e in table.items()}


CLAYFF_ALUMINA_FF = InorganicForceField(
    name="clayff_alumina",
    version="2004",
    label="CLAYFF alumina subset (ao/ob/oh/ho/o*/h*)",
    material_ids=("al2o3",),
    roles=_roles(CLAYFF_ALUMINA),
    combining_rule="arithmetic",
    motions=("frozen", "rigid"),
    cap_bond_length=OH_BOND_LENGTH["clayff"],
    aliases=("clayff",),
    citations=(CYGAN_2004,),
    notes="Nonbonded-only subset; CLAYFF charges are 0.525 x formal "
          "charges, so formally neutral terminations are neutral.",
)

OPLSAA_AROMATIC_CARBON_FF = InorganicForceField(
    name="oplsaa_aromatic_carbon",
    version="1996",
    label="OPLS-AA aromatic carbon (CA/HA) for graphene / CNT",
    material_ids=("c",),
    roles=_roles(OPLSAA_AROMATIC_CARBON),
    combining_rule="host",
    motions=("frozen", "rigid"),
    cap_bond_length=1.08,
    aliases=("oplsaa", "opls_aromatic"),
    citations=(OPLS_AROMATIC,),
    notes="Interior sp2 C neutral, edge C-H -0.115/+0.115 (PAH "
          "convention); qualified with OPLS-AA hosts only.",
)

INTERFACE_METAL_FFS = {}
for _el, _entry in INTERFACE_METALS.items():
    _ff = InorganicForceField(
        name=f"interface_metal_{_el.lower()}",
        version="2008",
        label=f"INTERFACE FF 12-6 LJ fcc {_el} (Heinz et al. 2008)",
        material_ids=(_el.lower(),),
        roles={_el: RoleParameters.from_table(_entry)},
        combining_rule="host",
        motions=("frozen", "rigid"),
        aliases=("interface", "heinz"),
        citations=(HEINZ_2008,),
        notes="Effective interfacial LJ model (neutral, no image charges); "
              "NOT an EAM/many-body potential and not mixed with one.",
    )
    register_inorganic_ff(_ff)
    INTERFACE_METAL_FFS[_el] = _ff

register_inorganic_ff(CLAYFF_ALUMINA_FF)
register_inorganic_ff(OPLSAA_AROMATIC_CARBON_FF)
