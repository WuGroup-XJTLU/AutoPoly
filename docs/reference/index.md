# API Reference

AutoPoly's public API, generated from the source docstrings.

## Core classes

| Class | Module | Purpose |
|---|---|---|
| [`System`](system.md) | `AutoPoly.core.system` | Output directory and path management |
| [`Polymer`](polymer.md) | `AutoPoly.models.polymer` | Polymer chains from complement SMILES sequences |
| [`Molecule`](molecule.md) | `AutoPoly.models.molecule` | Small molecules (solvents, additives) |
| [`generate`](generate.md) | `AutoPoly.pipeline.workflow` | One-shot pipeline: geometry → typing → packing → LAMMPS files |

## Coarse-grained

| Class | Module | Purpose |
|---|---|---|
| [`BeadSpringPolymer`](bead-spring.md) | `AutoPoly.models.bead_spring` | Direct LAMMPS data files for bead-spring models |
| [`BeadType` / `AngleType`](bead-spring.md) | `AutoPoly.models.bead_spring` | Bead species and angle parameters |
| [`MCConfig` / `SAWConfig`](bead-spring.md) | `AutoPoly.models.bead_spring` | Generation and equilibration tuning |

## Nanocomposites

| Class | Module | Purpose |
|---|---|---|
| [`Nanoparticle`](nanoparticles.md) | `AutoPoly.nanoparticles` | Finite inorganic particle (material, phase, shape, surface, motion, inorganic FF) |
| [`Grafting`](nanoparticles.md) | `AutoPoly.nanoparticles.grafting` | Covalently grafted chains on silica |
| [`CGFiller`](nanoparticles.md) | `AutoPoly.nanoparticles.cg` | Coarse-grained filler for `BeadSpringSystem` |
| [`get_material` / `MaterialSpec`](nanoparticles.md) | `AutoPoly.materials` | Material registry and capability states |
| [`qualification_matrix`](nanoparticles.md) | `AutoPoly.materials.qualification` | md_ready combinations and their evidence |
| [`NanocompositePacking`](nanoparticles.md) | `AutoPoly.packing` | Fillers-first, obstacle-aware packing configuration |
| [`run_smoke`](nanoparticles.md) | `AutoPoly.pipeline.smoke` | Short LAMMPS health check of a generated project |

## Building blocks

| Module | Purpose |
|---|---|
| [`AutoPoly.monomers.monomer_generator`](monomer-generator.md) | SMILES → moltemplate `.lt` monomer templates |
| [`AutoPoly.mc`](mc.md) | Monte Carlo placement: collision detection, SAW chain growth |
| [`AutoPoly.core.exceptions`](exceptions.md) | Exception hierarchy |

!!! note "RDKit-dependent modules"
    `Polymer`, `Molecule`, `generate`, `MonomerGenerator`, and `mc` require RDKit. `System` and `BeadSpringPolymer` work without it.
