#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: a hydroxylated silica nanoparticle on its own.

Builds a beta-cristobalite-derived SiO2 sphere (Si-centered carve, Q0/Q1
removal, silanol capping), prints its geometry/surface summary and the
material capability report, writes the structure (JSON/PDB/XYZ), and
prepares a standalone moltemplate/LAMMPS system of two rigid particles
(INTERFACE FF v1.5 silica, no polymer).

Run:
    python example_silica_nanoparticle.py
"""
import json

from AutoPoly import System, generate
from AutoPoly.materials import get_material
from AutoPoly.nanoparticles import Nanoparticle, SilicaSurface

silica = get_material("SiO2")
print("Capabilities:", silica.capability_report())

np_ = Nanoparticle(
    material="SiO2",
    structure="beta_cristobalite",   # or "alpha_quartz"
    shape="sphere", radius=10.0,     # core cut radius (A)
    surface=SilicaSurface(oh_density=4.6),  # target; achieved value reported
    force_field="interface",         # INTERFACE FF v1.5 ("clayff" also built in)
    count=2, motion="rigid",
)

summary = np_.summary()
print(json.dumps({k: summary[k] for k in
                  ("formula", "n_atoms", "mass_amu", "accessible_area_nm2")},
                 indent=1))
meta = summary["metadata"]
print(f"silanols: {meta['n_silanol']}, achieved OH density "
      f"{meta['achieved_oh_density']:.2f}/nm^2 (target 4.6), "
      f"Q distribution {meta['q_distribution']}")

system = System(out="silica_nanoparticle_out")
np_.write_structure("silica_nanoparticle_out/silica_np.pdb")   # CONECT bonds
np_.write_structure("silica_nanoparticle_out/silica_np.json")  # AtomicStructure

# Standalone particles (no matrix): system.data + in.run with one rigid
# body per particle; box from box_size (A).
generate(system, "silica_np", [np_], force_field="oplsaa", box_size=60.0,
         rng_seed=42)

print("Done -- see silica_nanoparticle_out/silica_np/ "
      "(ff_compatibility.md, in.run, system.data)")
