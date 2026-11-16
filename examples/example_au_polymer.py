#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: polyethylene with a gold nanoparticle (qualified combination
au-interface-lj: Heinz et al. 2008 fcc-metal 12-6 LJ, OPLS-AA or GAFF2
matrix).  Faceted clusters are available with shape="cuboctahedron",
"octahedron" or "wulff" (requires ASE).

The 12-6 interface model reproduces surface and interface energies; it is
not a many-body (EAM) metal potential, which AutoPoly never mixes with
organic LJ.  Silver ("Ag") works the same way.

Run:
    python example_au_polymer.py
"""
from AutoPoly import Polymer, System, generate
from AutoPoly.nanoparticles import Nanoparticle
from AutoPoly.packing import NanocompositePacking

PE = ["CC[*]"] + ["[*]CC[*]"] * 8 + ["[*]CC"]

system = System(out="au_polymer_out")
gold = Nanoparticle(material="Au", radius=8.0, count=1, motion="rigid",
                    name="gold")
pe = Polymer(chain_num=20, sequence=PE, tacticity="atactic")
generate(system, "pe_gold", [pe, gold], force_field="oplsaa",
         packing=NanocompositePacking(seed=3), rng_seed=3)
print("Done -- see au_polymer_out/pe_gold/in.run")
