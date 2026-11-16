#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: polyethylene with a graphene flake and a (6,6) carbon nanotube
(qualified combination c-oplsaa-graphene, OPLS-AA aromatic CA/HA types
for the H-passivated sp2 network).

Run:
    python example_graphene_polymer.py
"""
from AutoPoly import Polymer, System, generate
from AutoPoly.nanoparticles import Nanoparticle
from AutoPoly.packing import NanocompositePacking

PE = ["CC[*]"] + ["[*]CC[*]"] * 8 + ["[*]CC"]

system = System(out="graphene_polymer_out")
flake = Nanoparticle(material="C", shape="sheet", length=22.0, width=20.0,
                     count=1, motion="rigid", name="graphene")
pe = Polymer(chain_num=20, sequence=PE, tacticity="atactic")
generate(system, "pe_graphene", [pe, flake], force_field="oplsaa",
         packing=NanocompositePacking(seed=3), rng_seed=3)

cnt = Nanoparticle(material="C", structure="cnt", shape="tube", n=6, m=6,
                   length=20.0, count=1, motion="rigid", name="cnt66")
generate(system, "pe_cnt", [Polymer(chain_num=20, sequence=PE), cnt],
         force_field="oplsaa", packing=NanocompositePacking(seed=3),
         rng_seed=3)
print("Done -- see graphene_polymer_out/pe_graphene and pe_cnt")
