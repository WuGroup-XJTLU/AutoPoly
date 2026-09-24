# Tutorials

Thirteen runnable scripts live in the [`examples/`](https://github.com/WuGroup-XJTLU/AutoPoly/tree/v1.0/examples) directory of the repository. Each one writes LAMMPS input files (`system.data`, `system.in.init`, `system.in.settings`, `system.in.charges`) into a new output directory.

## Running them

```bash
# 1. Install AutoPoly (once, from the repo root)
pip install -e .

# 2. Run any example from the examples directory
cd examples
python example_pmma_linear.py
```

Output directories are git-ignored — just re-run a script to regenerate them.

## The examples

### Polymers (atomistic)

| Tutorial | System | Force field | Demonstrates |
|---|---|---|---|
| [First Polymer: PMMA](pmma-linear.md) | PMMA | OPLS-AA | **Start here.** Complete beginner workflow, complement SMILES step by step |
| [Block Copolymer](block-copolymer.md) | PE-PS-PE ABA triblock | OPLS-AA | Explicit per-position monomer sequences |
| [Condensation Polymer (PLA)](pla-condensation.md) | PLA | GAFF | Step-growth polymers, ester backbone |
| `example_gasteiger_charges.py` | PMMA | GAFF | Automatic Gasteiger charge assignment |
| `example_commodity_polymers_10.py` | 10 commodity polymers | OPLS-AA | Batch generation, CLI selection |
| [Force Field Comparison](force-field-comparison.md) | PEO | all 6 | The six built-in force fields side by side (OpenFF: see `example_peo_openff.py`) |
| [MC Placement Methods](mc-placement.md) | PEO | OPLS-AA | grid vs MC random vs MC chain growth |

### Molecules and mixtures

| Tutorial | System | Force field | Demonstrates |
|---|---|---|---|
| [Small Molecules & Mixtures](molecules.md) | water, ethanol, benzene | GAFF | The `Molecule` class, mixtures, polymer + solvent |
| [Polymer Solution](peo-solution.md) | PEO + 200 water | GAFF | Explicit-solvent polymer solution |
| `example_d4ppd.py` | D4PPD antioxidant | GAFF2 | Larger organic molecule, extended atom types |

### Coarse-grained

| Tutorial | Demonstrates |
|---|---|
| [Bead-Spring Models](bead-spring.md) | Homopolymer, diblock (FENE), ring with angles, MC equilibration, SAW generation — direct LAMMPS data files, no moltemplate |
| `example_bead_spring_side_groups.py` | Comb with side groups, graft copolymer with oligomeric side chains, comb from an explicit `MonomerTemplate` (backbone + side-group beads), branch-point angles — via the standard moltemplate backend, plus the direct-writer alternative |

## Before you start

The examples assume you know the [3-step workflow](../guides/workflow.md) and the basics of [complement SMILES](../guides/complement-smiles.md). If those are new, read them first — or just start with [PMMA](pmma-linear.md), which explains as it goes.
