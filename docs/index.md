# AutoPoly

**Automated Polymer Generation and Simulation Package** — turn SMILES strings into complete, ready-to-run LAMMPS input files.

AutoPoly generates polymer structures and prepares them for molecular dynamics simulations with [LAMMPS](https://www.lammps.org/). Build everything from simple homopolymers to complex block copolymers with explicit sequence control, then let AutoPoly handle monomer templates, force field parameters, chain placement, and data file generation.

## Key Features

- **6 Force Fields** — OPLS-AA, LOPLS, GAFF, GAFF2, DREIDING, COMPASS
- **Block Copolymers** — explicit sequence control for any block arrangement
- **Complement SMILES** — unique format for precise positional control of every monomer
- **Small Molecules** — built-in support for solvents and additives
- **Ring & Linear** — both topologies supported
- **SAW Placement** — Monte Carlo self-avoiding walk for realistic initial configurations
- **Coarse-Grained Models** — bead-spring polymers written directly as LAMMPS data files
- **Nanocomposites** — silica, alumina, carbon, Au/Ag particles (and CG fillers) in a polymer matrix, with qualified force-field combinations and explicit cross terms
- **Three-Stage Pipeline** — geometry → typing → packing as independently usable stages

## Your First Polymer in 3 Steps

```python
from AutoPoly import System, Polymer, generate

# Step 1: Create system
system = System(out="my_polymer")

# Step 2: Define polymer
polymer = Polymer(
    chain_num=10,                    # 10 chains
    sequence=["CC[*]"] + ["[*]CC[*]"] * 48 + ["[*]CC"],  # PE, DOP=50
    topology="linear",
    tacticity="atactic"
)

# Step 3: Generate LAMMPS files
generate(system, "polyethylene", [polymer], force_field="oplsaa")
```

**Output:** ready-to-run LAMMPS files (`system.data`, `system.in.init`, `system.in.settings`, `system.in.charges`) in `my_polymer/polyethylene/`.

## Where to Go Next

<div class="grid cards" markdown>

- :material-download: **[Installation](getting-started/installation.md)** — get AutoPoly and its dependencies installed
- :material-rocket-launch: **[Quickstart](getting-started/quickstart.md)** — the 3-step workflow end to end
- :material-dna: **[Complement SMILES](guides/complement-smiles.md)** — AutoPoly's unique monomer format, explained
- :material-school: **[Tutorials](tutorials/index.md)** — 12 runnable examples, from PMMA to bead-spring melts
- :material-tune: **[Force Fields](guides/force-fields.md)** — pick the right one of the six
- :material-api: **[API Reference](reference/index.md)** — every class and function, from the source

</div>

!!! note "Atom typing"
    The current atom typing system (for all force fields) relies on manually built SMARTS patterns. A data-driven approach such as BESMARTS could generate more robust patterns, and we are exploring this for a future atom typing system.

## Citation

If you use AutoPoly in your research, please cite:

```bibtex
@software{autopoly,
  title={AutoPoly: Automated Polymer Generation for Molecular Simulation},
  author={Wu, Zhenghao},
  url={https://github.com/WuGroup-XJTLU/AutoPoly}
}
```

## License

AutoPoly is released under the MIT License — see [license.md](https://github.com/WuGroup-XJTLU/AutoPoly/blob/v1.0/license.md) in the repository.
