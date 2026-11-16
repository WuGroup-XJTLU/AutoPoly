#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: TiO2 is geometry-ready but NOT md_ready.

AutoPoly builds hydroxylated anatase/rutile clusters (Ti 6-coordinated,
bond-valence protonation to neutrality) and writes them for inspection,
but no validated TiO2-polymer interaction model is bundled, so MD export
is refused with an explanation.  Supplying your own validated parameters
is possible with allow_unqualified=True plus a registered inorganic
force-field adapter (see docs/guides/nanocomposites.md).

Run:
    python example_titania_polymer.py
"""
from pathlib import Path

from AutoPoly import Polymer, System, generate
from AutoPoly.core.exceptions import AutoPolyError
from AutoPoly.materials import get_material
from AutoPoly.nanoparticles import Nanoparticle

titania = get_material("TiO2")
print("Capabilities:", titania.capability_report())

np_ = Nanoparticle(material="TiO2", structure="anatase", radius=8.0)
s = np_.summary()
print(f"anatase cluster {s['formula']}: {s['n_atoms']} atoms, "
      f"{s['accessible_area_nm2']:.1f} nm^2")
Path("titania_out").mkdir(exist_ok=True)
np_.write_structure("titania_out/anatase_np.pdb")

system = System(out="titania_out")
pe = Polymer(chain_num=10, sequence=["CC[*]"] + ["[*]CC[*]"] * 8 + ["[*]CC"])
try:
    generate(system, "pe_titania", [pe, np_], force_field="oplsaa")
except AutoPolyError as err:
    print("MD export refused (expected):", err)
