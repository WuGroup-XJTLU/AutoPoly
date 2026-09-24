# AutoPoly Examples

Tutorial examples for generating polymer and molecular structures with AutoPoly — from SMILES to complete LAMMPS input files.

## Running the Examples

```bash
# 1. Install AutoPoly (once, from the repo root)
pip install -e .

# 2. Run any example from this directory
cd examples
python example_pmma_linear.py
```

Each example writes LAMMPS input files (`system.data`, `system.in.init`,
`system.in.settings`, `system.in.charges`) into a new output directory.
Output directories are git-ignored — just re-run a script to regenerate them.

## Available Examples

### Polymers (atomistic)

| Script | System | Force field | Demonstrates |
|---|---|---|---|
| `example_pmma_linear.py` | PMMA | OPLS-AA | **Start here.** Complete beginner workflow, complement SMILES explained step by step |
| `example_block_copolymer.py` | PE-PS-PE ABA triblock | OPLS-AA | Explicit per-position monomer sequences |
| `example_pla_condensation.py` | PLA | GAFF | Condensation (step-growth) polymers, ester backbone |
| `example_gasteiger_charges.py` | PMMA | GAFF | Automatic Gasteiger charge assignment |
| `example_commodity_polymers_10.py` | PE, PP, PS, PVC, PVAc, PMMA, PAN, PB, PI, PEO | OPLS-AA | Batch generation of 10 polymers, CLI selection |
| `example_peo_all_forcefields.py` | PEO | all 6 | Force field comparison (oplsaa, lopls, gaff, gaff2, dreiding, compass) |
| `example_peo_openff.py` | PEO | OpenFF (Sage) | Optional backend: SMIRNOFF typing + NAGL GNN charges (needs openff-toolkit) |
| `example_peo_mc_placement.py` | PEO | OPLS-AA | Placement methods: grid vs MC random vs MC chain growth |
| `example_film_on_substrate.py` | PE film on ethanol slab | GAFF | **Surfaces:** physical substrate (`SubstrateSpec`), `box_dims`, carve subtract (`Cylinder`) |
| `example_film_on_quartz.py` | PE film on alpha-quartz(0001) | GAFF | **Built-in silica:** `builder="alpha_quartz"`, hydroxylated Q2 slab, INTERFACE FF |
| `example_film_on_cristobalite.py` | PE film on beta-cristobalite(111) | GAFF | **Built-in silica:** `builder="beta_cristobalite"`, Q3 silanols, CLAYFF option |

### Pipeline API

| Script | Demonstrates |
|---|---|
| `example_three_stage_pipeline.py` | Stage-level API: `GeometryBuilder` → `UnitTyper` → `BoxPacker`; one geometry typed under multiple force fields |

### Reactive MD

| Script | System | Demonstrates |
|---|---|---|
| `example_reactor_polyester.py` | EG + adipic acid melt | `Reactor`: reaction detection, `fix bond/react` templates, `in.bond_react` |

### Molecules and mixtures

| Script | System | Force field | Demonstrates |
|---|---|---|---|
| `example_benzene_system.py` | 100 benzene | GAFF | Simplest `Molecule` example |
| `example_molecules.py` | water, water+ethanol, PE+water | GAFF | Single molecules, mixtures, polymer+solvent |
| `example_peo_solution.py` | PEO + 200 water | GAFF | Explicit-solvent polymer solution |
| `example_d4ppd.py` | D4PPD antioxidant | GAFF2 | Larger organic molecule, extended atom types |

### Coarse-grained (bead-spring)

| Script | Demonstrates |
|---|---|
| `example_bead_spring.py` | Homopolymer, diblock (FENE), ring with angle potentials, MC equilibration, SAW generation — direct LAMMPS data files, no moltemplate |
| `example_bead_spring_side_groups.py` | **Graph architectures:** comb with single-bead side groups, graft copolymer with oligomeric side chains, `MonomerTemplate` multi-bead monomer, moltemplate vs direct backends |

## Understanding Complement SMILES (pSMILES)

AutoPoly defines monomers with `[*]` wildcards marking the atoms that
connect to neighbors during polymerization. The wildcard count and
position determine the monomer's role in the chain:

| Role | Wildcards | PMMA example |
|---|---|---|
| First (chain start) | 1, right | `"CC(C)(C(=O)OC)[*]"` |
| Middle (interior) | 2 | `"[*]CC([*])(C)C(=O)OC"` |
| Last (chain end) | 1, left | `"[*]CC(C)(C(=O)OC)"` |

A chain is an explicit sequence — first + (DOP−2)×middle + last:

```python
DOP = 10
sequence = [PMMA_FIRST] + [PMMA_MIDDLE] * (DOP - 2) + [PMMA_LAST]

polymer = Polymer(
    chain_num=4,           # chains in the box
    sequence=sequence,     # DOP derived from sequence length
    topology="linear",     # or "ring"
    tacticity="atactic",   # or "isotactic" / "syndiotactic"
)
```

Block copolymers are just mixed sequences — see `example_block_copolymer.py`.

## Polymerization Mechanisms

### Vinyl addition (chain-growth)

C=C double bond opens to form C-C single bonds. All-carbon backbone, no
byproducts. Examples: PE, PP, PS, PMMA.

```
n CH2=C(CH3)COOCH3  ->  [-CH2-C(CH3)(COOCH3)-]n
```

### Condensation (step-growth)

Functional groups react with elimination of small molecules. Heteroatom
backbone. Examples: PLA, polyesters, polyamides.

```
n HO-CH(CH3)-COOH  ->  [-O-CH(CH3)-CO-]n + n H2O
```

See `example_pla_condensation.py`.

## Common Complement SMILES

| Polymer | First | Middle | Last |
|---|---|---|---|
| PE | `CC[*]` | `[*]CC[*]` | `[*]CC` |
| PP | `CC(C)[*]` | `[*]CC([*])(C)` | `[*]CC(C)` |
| PS | `CC(c1ccccc1)[*]` | `[*]CC([*])c1ccccc1` | `[*]CC(c1ccccc1)` |
| PMMA | `CC(C)(C(=O)OC)[*]` | `[*]CC([*])(C)C(=O)OC` | `[*]CC(C)(C(=O)OC)` |
| PEO | `CCO[*]` | `[*]CCO[*]` | `[*]CCO` |
| PLA | `OC(C)C(=O)[*]` | `[*]OC(C)C(=O)[*]` | `[*]OC(C)C(=O)O` |

## Force Field Selection

| Force field | Best for | Notes |
|---|---|---|
| `oplsaa` | Vinyl polymers (PE, PP, PS, PMMA) | Well-parameterized hydrocarbons |
| `gaff` / `gaff2` | Polyesters, small molecules, diverse organics | Extensive functional-group coverage |
| `lopls` | Long hydrocarbon chains | Optimized for alkanes |
| `dreiding` | Generic/organic | Requires external charges |
| `compass` | Class II systems | Requires LAMMPS CLASS2 package |
| `openff` | SMIRNOFF (Sage) typing, novel chemistries | Optional: `conda install -c conda-forge openff-toolkit openff-nagl`; NAGL GNN charges by default (see `example_peo_openff.py`) |

Gasteiger charges are assigned automatically (see
`example_gasteiger_charges.py`). For production runs, replace them with
AM1-BCC (Antechamber) or RESP charges in `system.in.charges`.

## Best Practices

1. **Start small** — 2 chains, DOP 10 — then scale up.
2. **Validate against experiment** before production:
   - PMMA: density ~1.18 g/cm³, Tg ~378 K
   - PLA: density ~1.24–1.26 g/cm³, Tg ~330 K
   - PS: density ~1.04–1.06 g/cm³, Tg ~373 K
3. **Equilibrate in stages**: minimize → NVT heating → NPT compression →
   NPT production. Boxes are built at low density to allow overlap-free
   placement.
4. **Check charges** in `system.in.charges` before long runs.

## Troubleshooting

**"No module named 'AutoPoly'"**
Install from the repo root: `pip install -e .`

**"Monomer not found"**
Check that your complement SMILES is valid and has the right number of
`[*]` wildcards for the position (first/middle/last).

**Simulation explodes in LAMMPS**
Ensure proper equilibration (minimize → NVT → NPT), use a 0.5–1.0 fs
timestep for atomistic systems, verify charges are assigned, and check
for bad contacts in `system.data`.

**Moltemplate execution failed**
Inspect the `.lt` files in the output's `moltemplate/input/` directory
and check the terminal output for the specific syntax error.

## Contributing an Example

1. Follow the naming convention: `example_<description>.py`
2. Import AutoPoly directly (no `sys.path` hacks) — installation via
   `pip install -e .` is a documented prerequisite
3. Include a module docstring: what it demonstrates, the complement
   SMILES used, and the expected output location
4. Use argparse (not interactive `input()`) if the example has options
5. Keep default sizes small enough to run in a few minutes
6. Update the table in this README

## Resources

- LAMMPS: https://docs.lammps.org/
- Moltemplate: https://moltemplate.org/
- RDKit (SMILES): https://www.rdkit.org/
