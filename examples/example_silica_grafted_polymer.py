#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: polyethylene-grafted silica particle in a polyethylene melt
(qualified combination sio2-interface-grafted, OPLS-AA).

Graft sites are silanol oxygens chosen by farthest-point sampling at the
requested density; each Si-OH + H-C(head) pair condenses to a silyl ether
Si-O-C.  The grafts grow monomer by monomer from their anchors around
the core and become part of the particle class (GNP_<name>); linkage
bond/angle/dihedral terms come from the bundled oplsaa.lt by OPLS bonded
class.  The rigid body in in.run is the core only; the grafted chains are
flexible.

Run:
    python example_silica_grafted_polymer.py
"""
import json

from AutoPoly import Polymer, System, generate
from AutoPoly.nanoparticles import Grafting, Nanoparticle
from AutoPoly.packing import NanocompositePacking

GRAFT = ["CC[*]"] + ["[*]CC[*]"] * 4 + ["[*]CC"]      # PE6 graft
MATRIX = ["CC[*]"] + ["[*]CC[*]"] * 8 + ["[*]CC"]     # PE10 matrix

system = System(out="grafted_silica_out")
brush = Grafting(polymer=Polymer(chain_num=1, sequence=GRAFT),
                 density=0.5,      # chains/nm^2 of accessible area
                 seed=2)
particle = Nanoparticle(material="SiO2", radius=9.0, count=1, motion="rigid",
                        name="gsilica", grafts=brush)
matrix = Polymer(chain_num=20, sequence=MATRIX, tacticity="atactic")

generate(system, "pe_grafted_silica", [matrix, particle],
         force_field="oplsaa",
         packing=NanocompositePacking(seed=3), rng_seed=3)

units = json.load(open("grafted_silica_out/pe_grafted_silica/build/oplsaa/"
                       "units.json"))
filler = next(u for u in units["units"] if u["role"] == "filler")
print("grafts:", json.dumps(filler["material"]["grafts"], indent=1)[:600])
