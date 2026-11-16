#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: PMMA with a hydroxylated alpha-alumina particle (qualified
combination al2o3-clayff-hydroxylated, CLAYFF alumina + OPLS-AA).

The corundum cluster keeps Al 6-coordinated; under-coordinated O are
protonated in bond-valence deficit order until the particle is neutral
(CLAYFF ao/ob/oh/ho roles).  CLAYFF was fitted with the arithmetic
(Lorentz-Berthelot) rule, so the explicit alumina-PMMA cross terms use
arithmetic sigma even though the OPLS-AA host mixes geometrically.

Run:
    python example_alumina_pmma.py
"""
from AutoPoly import Polymer, System, generate
from AutoPoly.nanoparticles import Nanoparticle
from AutoPoly.packing import NanocompositePacking

PMMA = (["CC(C)(C(=O)OC)[*]"] + ["[*]CC(C)(C(=O)OC)[*]"] * 6
        + ["[*]CC(C)(C(=O)OC)"])

system = System(out="alumina_pmma_out")
alumina = Nanoparticle(material="Al2O3", radius=8.0, count=1,
                       motion="rigid",           # or "frozen"
                       surface="hydroxylated",   # or "stoichiometric" (dry)
                       name="alumina")
pmma = Polymer(chain_num=12, sequence=PMMA, tacticity="atactic")

generate(system, "pmma_alumina", [pmma, alumina], force_field="oplsaa",
         packing=NanocompositePacking(seed=5, target_polymer_density=1.18),
         rng_seed=5)
print("Done -- see alumina_pmma_out/pmma_alumina/ff_compatibility.md")
