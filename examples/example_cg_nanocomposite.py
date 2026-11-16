#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: coarse-grained nanocomposite with the bead-spring backend.

Two rigid FCC-bead fillers (radius 4 sigma) in a Kremer-Grest melt: the
fillers are placed first, chains grow around them by a periodic SAW, and
in.polymer holds the fillers during minimization, then integrates each
filler as one rigid body.  The polymer-filler attraction is set
explicitly (epsilon 1.5, cutoff 2.5 sigma) on top of the repulsive WCA
melt.  The SAW packs below melt density; the run script contains a
commented NPT compression stage.

Run:
    python example_cg_nanocomposite.py
"""
import numpy as np

from AutoPoly import System
from AutoPoly.models import architectures as arch
from AutoPoly.models.bead_spring import BeadType
from AutoPoly.models.bead_spring_system import BeadSpringSystem
from AutoPoly.nanoparticles import CGFiller

np.random.seed(42)
system = System(out="cg_nanocomposite_out")
bss = BeadSpringSystem(
    "kg_filled", system,
    bead_types=[BeadType("A"), BeadType("F")],
    bond_style="fene", pair_style="wca",
    density=0.3,            # chain beads per sigma^3 outside the fillers
)
bss.add_species(arch.linear([("A", 30)]), n_chains=60)
bss.add_filler(CGFiller(radius=4.0, count=2, bead_type="F",
                        structure="fcc", motion="rigid"))
bss.set_pair_coeff("A", "F", epsilon=1.5, cutoff=2.5)
bss.generate(backend="direct")      # or "moltemplate"
print(bss.get_system_info()["fillers"])
