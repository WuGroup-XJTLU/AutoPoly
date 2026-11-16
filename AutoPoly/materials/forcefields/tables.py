#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical parameter tables for the built-in inorganic force fields.

Single source of truth for every inorganic parameter value shipped with
AutoPoly.  ``AutoPoly.surfaces.ff_tables`` re-exports the silica tables
(INTERFACE_FF, CLAYFF, SLAB_FF_REGISTRY, OH_BOND_LENGTH, SLAB_ROLES) for
backward compatibility; the force-field adapters in
``AutoPoly.materials.forcefields`` wrap the same dictionaries.  Never copy
these numbers elsewhere.

Silica tables are keyed by *role*:

- "Si": tetrahedral silicon (4 O neighbors)
- "OB": bridging oxygen (2 Si neighbors)
- "OH": hydroxyl (silanol) oxygen (1 Si + 1 H neighbor)
- "HO": hydroxyl hydrogen

Each entry gives the moltemplate @atom type name, mass (amu), charge (e),
and 12-6 Lennard-Jones parameters in LAMMPS ``lj/cut/coul/long`` form
(epsilon in kcal/mol, sigma in Angstrom).

Since the built slabs/particles are intended to be held rigid/frozen in
MD, only nonbonded parameters are tabulated.  Published bonded terms
(quoted in the references below) can be added by the user via
``slab_types``/custom .lt files if a flexible model is wanted.

References
----------
INTERFACE FF v1.5 silica parameters (types SC4/OC23/OC24/HOY):
    Emami, F. S.; Puddu, V.; Berry, R. J.; Varshney, V.; Patwardhan, S. V.;
    Perry, C. C.; Heinz, H. "Force Field and a Surface Model Database for
    Silica to Simulate Interfacial Properties in Atomic Resolution",
    Chem. Mater. 2014, 26, 2647-2658.  DOI: 10.1021/cm500365c.
    Parameter file: charmm27_interface_v1_5.prm (H. Heinz lab distribution).

CLAYFF (types st/ob/oh/ho):
    Cygan, R. T.; Liang, J.-J.; Kalinichev, A. G. "Molecular Models of
    Hydroxide, Oxyhydroxide, and Clay Phases and the Development of a
    General Force Field", J. Phys. Chem. B 2004, 108, 1255-1266.
    DOI: 10.1021/jp0363287.

Created on 2026-08-12 (as AutoPoly/surfaces/ff_tables.py)
Moved on 2026-11-13
@author: zwu
"""

#: 2**(1/6): converts a well-minimum distance Rmin to LJ sigma.
_TWO_POW_1_6 = 2.0 ** (1.0 / 6.0)


def _entry(type_name, mass, charge, epsilon, rmin):
    """Build a table entry; sigma derived from the well minimum Rmin."""
    return {
        "type": type_name,
        "mass": mass,
        "charge": charge,
        "epsilon": epsilon,
        "sigma": rmin / _TWO_POW_1_6 if rmin > 0.0 else 0.0,
    }


#: INTERFACE FF v1.5, silica set (CHARMM form: epsilon, Rmin/2 in .prm;
#: Rmin = 2 * Rmin/2).  Charges: neutral slab requires every Si to stay
#: 4-coordinated (the quartz builder guarantees this).
INTERFACE_FF = {
    "Si": _entry("i15_sc4",  28.0855, +1.100, 0.093, 2.0 * 2.075),
    "OB": _entry("i15_oc23", 15.9994, -0.550, 0.054, 2.0 * 1.735),
    "OH": _entry("i15_oc24", 15.9994, -0.675, 0.122, 2.0 * 1.735),
    "HO": _entry("i15_hoy",   1.0080, +0.400, 0.015, 2.0 * 0.5425),
}

#: CLAYFF species parameters, Cygan et al. 2004, Table 1:
#: E = D0[(R0/r)^12 - 2(R0/r)^6] with D0 in kcal/mol and R0 the
#: well-minimum distance in A; epsilon = D0, sigma = R0/2^(1/6)
#: (st: sigma 3.3020 A, ob/oh: 3.1655 A, ao: 4.2712 A).
#: species -> (charge e, D0 kcal/mol, R0 A).
CLAYFF_SPECIES = {
    "st": (+2.100, 1.8405e-6, 3.7064),   # tetrahedral silicon
    "ao": (+1.575, 1.3298e-6, 4.7943),   # octahedral aluminum
    "ob": (-1.050, 0.1554, 3.5532),      # bridging oxygen
    "oh": (-0.950, 0.1554, 3.5532),      # hydroxyl oxygen
    "ho": (+0.425, 0.0, 0.0),            # hydroxyl hydrogen
    "o*": (-0.820, 0.1554, 3.5532),      # water oxygen (SPC)
    "h*": (+0.410, 0.0, 0.0),            # water hydrogen (SPC)
}


def _clayff(species, type_name, mass):
    q, d0, r0 = CLAYFF_SPECIES[species]
    return _entry(type_name, mass, q, d0, r0)


#: CLAYFF silica subset (st/ob/oh/ho).
#: Fixed 2026-11-13: earlier versions passed sigma (3.30203 / 3.16554 A) as
#: R0, so the slab sigmas were 2^(1/6) too small (2.94177 / 2.82018 A).
CLAYFF = {
    "Si": _clayff("st", "cff_st", 28.0855),
    "OB": _clayff("ob", "cff_ob", 15.9994),
    "OH": _clayff("oh", "cff_oh", 15.9994),
    "HO": _clayff("ho", "cff_ho", 1.0080),
}

#: CLAYFF alumina role table (alpha-Al2O3 clusters).  Roles from the
#: oxide chemistry adapter: "Al" (octahedral Al), "OB" (O without H),
#: "OH"/"HO" (hydroxyl), "OW"/"HW" (aqua ligand: O carrying two H).  The
#: CLAYFF charges of these species are 0.525 x their formal charges
#: (Al +3, O -2, OH -1, OH2 0), so formal neutrality == CLAYFF neutrality.
CLAYFF_ALUMINA = {
    "Al": _clayff("ao", "cff_ao", 26.9815),
    "OB": _clayff("ob", "cff_ob", 15.9994),
    "OH": _clayff("oh", "cff_oh", 15.9994),
    "HO": _clayff("ho", "cff_ho", 1.0080),
    "OW": _clayff("o*", "cff_ow", 15.9994),
    "HW": _clayff("h*", "cff_hw", 1.0080),
}

#: INTERFACE FF 12-6 Lennard-Jones parameters of fcc metals (Heinz, Vaia,
#: Farmer & Naik, J. Phys. Chem. C 2008, 112, 17281, Table 1: well depth
#: epsilon (kcal/mol) and well-minimum distance r0 (A); sigma =
#: r0/2^(1/6)).  Neutral atoms, no bonded terms.
INTERFACE_METALS = {
    "Au": _entry("ifm_au", 196.96657, 0.0, 5.29, 2.951),
    "Ag": _entry("ifm_ag", 107.8682, 0.0, 4.56, 2.955),
}

#: OPLS-AA aromatic carbon (benzene types CA/HA; Jorgensen & Severance,
#: J. Am. Chem. Soc. 1990, 112, 4768; OPLS-AA, Jorgensen, Maxwell &
#: Tirado-Rives, J. Am. Chem. Soc. 1996, 118, 11225): sigma/epsilon of
#: CA 3.55 A / 0.070 kcal/mol, HA 2.42 A / 0.030 kcal/mol; edge C-H
#: charges -0.115/+0.115, interior sp2 carbon neutral (graphene / CNT /
#: polycyclic aromatic convention).  Values given as sigma (rmin = sigma *
#: 2^(1/6)).
OPLSAA_AROMATIC_CARBON = {
    "CA": _entry("oac_ca", 12.011, 0.0, 0.070, 3.55 * _TWO_POW_1_6),
    "CE": _entry("oac_ce", 12.011, -0.115, 0.070, 3.55 * _TWO_POW_1_6),
    "HE": _entry("oac_he", 1.008, +0.115, 0.030, 2.42 * _TWO_POW_1_6),
}

#: Registry of built-in slab force fields.
SLAB_FF_REGISTRY = {
    "interface": INTERFACE_FF,
    "clayff": CLAYFF,
}

#: Equilibrium O-H bond length used when capping dangling surface oxygens
#: (INTERFACE FF: OC23-HOY 0.945 A; CLAYFF commonly uses ~1.0 A).
OH_BOND_LENGTH = {
    "interface": 0.945,
    "clayff": 1.0,
}

#: All roles a slab force field must cover.
SLAB_ROLES = ("Si", "OB", "OH", "HO")
