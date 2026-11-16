#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Silica (SiO2) inorganic force-field adapters.

Wraps the built-in silica role tables (INTERFACE FF v1.5 and CLAYFF, see
tables.py) once as registered, versioned adapters.  The numerical values
are taken by reference from tables.py -- never duplicated here.

Both sets are nonbonded-only (LJ + charges): permitted motions are
"frozen" and "rigid".  A flexible silica model needs an independently
validated bonded parameter set and is rejected.

Created on 2026-11-13
@author: zwu
"""
from .registry import InorganicForceField, RoleParameters, register_inorganic_ff
from .tables import CLAYFF, INTERFACE_FF, OH_BOND_LENGTH

EMAMI_2014 = (
    "Emami, F. S.; Puddu, V.; Berry, R. J.; Varshney, V.; Patwardhan, S. V.; "
    "Perry, C. C.; Heinz, H. Chem. Mater. 2014, 26, 2647-2658 "
    "(INTERFACE FF v1.5 silica; doi:10.1021/cm500365c)"
)
CYGAN_2004 = (
    "Cygan, R. T.; Liang, J.-J.; Kalinichev, A. G. J. Phys. Chem. B 2004, "
    "108, 1255-1266 (CLAYFF; doi:10.1021/jp0363287)"
)

def _with_graft_anchor(table):
    """
    Role table + "OG" (grafted silanol O): silanol-O LJ, charge = silanol
    group charge q(OH) + q(HO).  The grafting typer adds the charge of the
    organic H removed by the Si-O-C condensation, so the total charge of
    particle + grafts is conserved exactly.
    """
    roles = {r: RoleParameters.from_table(e) for r, e in table.items()}
    oh, ho = roles["OH"], roles["HO"]
    roles["OG"] = RoleParameters(f"{oh.type_name}g", oh.mass,
                                 oh.charge + ho.charge, oh.epsilon, oh.sigma)
    return roles


INTERFACE_SILICA = InorganicForceField(
    name="interface_silica",
    version="1.5",
    label="INTERFACE FF v1.5 silica (12-6 LJ, CHARMM/AMBER/OPLS-AA form)",
    material_ids=("sio2",),
    roles=_with_graft_anchor(INTERFACE_FF),
    # Heinz et al.: the 12-6 parameters are used with the combination rule
    # of the host force field (CHARMM/AMBER arithmetic, OPLS-AA geometric).
    combining_rule="host",
    motions=("frozen", "rigid"),
    cap_bond_length=OH_BOND_LENGTH["interface"],
    aliases=("interface",),
    citations=(EMAMI_2014,),
    notes="Nonbonded-only (rigid/frozen silica). Si 4-coordinated surfaces "
          "with Q3/Q2 silanols; ionized silanols not built.",
)

CLAYFF_SILICA = InorganicForceField(
    name="clayff_silica",
    version="2004",
    label="CLAYFF silica subset (st/ob/oh/ho)",
    material_ids=("sio2",),
    roles=_with_graft_anchor(CLAYFF),
    # CLAYFF was parameterized with the arithmetic-mean (Lorentz-Berthelot)
    # combination rule.
    combining_rule="arithmetic",
    motions=("frozen", "rigid"),
    cap_bond_length=OH_BOND_LENGTH["clayff"],
    aliases=("clayff",),
    citations=(CYGAN_2004,),
    notes="Nonbonded-only subset (no CLAYFF O-H bond/Si-O-H angle terms).",
)

register_inorganic_ff(INTERFACE_SILICA)
register_inorganic_ff(CLAYFF_SILICA)

#: Legacy SubstrateSpec.slab_ff names -> adapter.
SLAB_FF_ADAPTERS = {
    "interface": INTERFACE_SILICA,
    "clayff": CLAYFF_SILICA,
}
