#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: PMMA / silica nanocomposite (qualified combination
sio2-interface-hydroxylated, OPLS-AA matrix, rigid particle).

Fillers are placed first; the PMMA chains are then placed around them
(pre-grown conformations, regrown inside the box when they do not fit),
with periodic minimum-image clearance checks.  Explicit PMMA-silica pair
coefficients are appended to system.in.settings; in.run minimizes with
the particle held, then runs one rigid body per particle + matrix NVT;
the commented compression stage targets 1.18 g/cm^3.

Run:
    python example_pmma_silica_nanocomposite.py
"""
from AutoPoly import Polymer, System, generate
from AutoPoly.nanoparticles import Nanoparticle
from AutoPoly.packing import NanocompositePacking

PMMA = (["CC(C)(C(=O)OC)[*]"] + ["[*]CC(C)(C(=O)OC)[*]"] * 8
        + ["[*]CC(C)(C(=O)OC)"])              # DOP 10, H/CH3-capped ends

system = System(out="pmma_silica_out")
pmma = Polymer(chain_num=16, sequence=PMMA, tacticity="atactic")
silica = Nanoparticle(material="SiO2", structure="alpha_quartz", radius=9.0,
                      count=1, motion="rigid", name="silica")

result = generate(
    system, "pmma_silica", [pmma, silica],
    force_field="oplsaa",
    packing=NanocompositePacking(
        initial_density=0.35,           # g/cm^3 for box sizing
        target_polymer_density=1.18,    # compression target in in.run
        filler_clearance=2.5,           # matrix atom - filler atom (A)
        seed=7,
    ),
    rng_seed=7,
)
print("loading:", result.report["loading"]["filler_weight_fraction"],
      "(weight fraction)")
print("min distances (A):", result.report["min_distance"])
print("Done -- see pmma_silica_out/pmma_silica/ff_compatibility.md")
