# AutoPoly: Automated Polymer Generation for Molecular Simulation

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![LAMMPS](https://img.shields.io/badge/LAMMPS-compatible-orange.svg)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://wugroup-xjtlu.github.io/AutoPoly/)

AutoPoly generates polymer structures and prepares them for molecular dynamics simulations with LAMMPS. Build everything from simple homopolymers to complex block copolymers with explicit sequence control.

**Key Features:**
- **6 Force Fields** - OPLS-AA, LOPLS, GAFF, GAFF2, DREIDING, COMPASS
- **Block Copolymers** - Explicit sequence control for any block arrangement
- **Complement SMILES** - Unique format for precise positional control
- **Small Molecules** - Built-in support for solvents and additives
- **Substrates & Films** - Polymer films on physical slabs — in-pipeline, external, or built-in crystalline silica — with lithography-style carve subtract
- **Reactive MD (Reactor)** - AutoREACTER-style `fix bond/react` templates so monomers polymerize during MD
- **CG Bead-Spring Models** - Graph-based architectures: linear, ring, star, comb, graft, tadpole, dendrimer, custom — plus multi-species mixtures
- **Ring & Linear** - Both topologies supported
- **SAW Placement** - Monte Carlo self-avoiding walk for realistic initial configurations
- **Automatic Setup** - Generates complete LAMMPS input files

Note: The current atom typing system (for all force fields) relies on manually built SMARTS patterns. A data-driven approach such as BESMARTS could generate more robust SMARTS patterns, and we are exploring this for a better atom typing system.

## Quick Start

### Installation

```bash
git clone https://github.com/WuGroup-XJTLU/AutoPoly.git
cd AutoPoly
pip install -e .
```

### Your First Polymer (3 Steps)

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

**Output:** Ready-to-run LAMMPS files in `my_polymer/` directory!

## The Three-Stage Pipeline

Under the hood, `generate` composes three independent stages, each
usable on its own:

```
models ──▶ GeometryBuilder ──▶ geometry/geometry.json   (FF-agnostic)
                                   │  + force_field
                                   ▼
                              UnitTyper ──▶ build/<ff>/  (typed .lt + units.json)
                                   │
                                   ▼
              BoxPacker + packing strategies ──▶ moltemplate/ ──▶ system.data
```

1. **GeometryBuilder** — builds chain graphs, conformers, and per-chain
   placements with *no force field involved*. The result is a single
   inspectable `geometry.json`.
2. **UnitTyper** — assigns atom types/charges on the full chain graph and
   joins them onto the stored geometry via atom-map numbers. The same
   geometry can be typed under multiple force fields (`build/oplsaa/`,
   `build/gaff2/`, ...) without rebuilding chains.
3. **BoxPacker** — packs the typed units into the simulation box with a
   pluggable placement strategy (`mc_random`, `grid`, or your own), then
   runs moltemplate.

Stage-level usage (see `examples/example_three_stage_pipeline.py`):

```python
from AutoPoly import (System, Polymer, GeometryBuilder, GeometryConfig,
                      UnitTyper, BoxPacker)

system = System(out="peo_run")
polymer = Polymer(chain_num=5, sequence=["CCO[*]"] + ["[*]CCO[*]"] * 8 + ["[*]CCO"])

# Build geometry ONCE
geom = GeometryBuilder(system, "peo", GeometryConfig(rng_seed=42)).build([polymer])

# Type it under as many force fields as you like
UnitTyper(geom.dir, "oplsaa").type()          # build/oplsaa/
UnitTyper(geom.dir, "gaff2").type()           # build/gaff2/ — same geometry

# Pack the typed units into a box and run moltemplate
BoxPacker(system, "peo", strategy="mc_random", rng_seed=42).pack("peo_run/peo/build/gaff2")
```

Custom packing strategies can be registered at runtime:

```python
from AutoPoly import PlacementStrategy, register_strategy

class MyStrategy(PlacementStrategy):
    name = "my_strategy"
    def place(self, ctx):   # ctx.units: UnitLibrary manifest
        ...                 # return PlacementResult(records, box_bounds)

register_strategy("my_strategy", MyStrategy)
```

## Core Concepts

### The 3-Step Workflow

```
System → Polymer/Molecule → generate → LAMMPS Files
```

1. **System** - Defines output directory
2. **Polymer/Molecule** - Defines what to build
3. **generate** - Generates files with chosen force field

### Complement SMILES (Unique to AutoPoly)

AutoPoly uses **complement SMILES** to control each monomer's position:

- **First position:** `"CC[*]"` (1 wildcard, right)
- **Middle positions:** `"[*]CC[*]"` (2 wildcards, both sides)
- **Last position:** `"[*]CC"` (1 wildcard, left)

This enables precise block copolymer design:

```python
# ABA triblock: PE(2)-PS(3)-PE(2)
sequence = [
    "CC[*]",                    # PE first
    "[*]CC[*]",                 # PE middle
    "[*]CC([*])c1ccccc1",       # PS middle
    "[*]CC([*])c1ccccc1",       # PS middle
    "[*]CC([*])c1ccccc1",       # PS middle
    "[*]CC[*]",                 # PE middle
    "[*]CC"                     # PE last
]
```

**Why it matters:** Standard SMILES can't distinguish first/middle/last positions. Complement SMILES gives you control over every monomer.

[Deep dive: Complement SMILES Guide →](https://wugroup-xjtlu.github.io/AutoPoly/guides/complement-smiles/)

### Polymer vs Molecule

| Feature | Polymer | Molecule |
|---------|---------|----------|
| **Use for** | Polymers, chains | Solvents, small molecules |
| **SMILES** | Complement SMILES with `[*]` | Regular SMILES, no `[*]` |
| **Parameters** | `chain_num`, `sequence`, `topology` | `Count`, `Smiles`, `Name` |
| **Example** | `Polymer(chain_num=10, sequence=["CC[*]"]+["[*]CC[*]"]*48+["[*]CC"])` | `Molecule(Count=100, Smiles="O", Name="water")` |

## Examples

### Block Copolymers (Complement SMILES)

```python
# PE-PS-PE triblock (10-20-10)
sequence = (
    ["CC[*]"] + ["[*]CC[*]"] * 9 +                  # PE block (10)
    ["[*]CC([*])c1ccccc1"] * 20 +                   # PS block (20)
    ["[*]CC[*]"] * 9 + ["[*]CC"]                    # PE block (10)
)

polymer = Polymer(
    chain_num=5,
    sequence=sequence,  # DOP = 40
    topology="linear",
    tacticity="atactic"
)
```

[Full example: examples/example_block_copolymer.py →](examples/example_block_copolymer.py)

### Small Molecules (Solvent)

```python
from AutoPoly import Molecule

# Water molecules
water = Molecule(
    Count=100,
    Smiles="O",     # Regular SMILES (no wildcards)
    Name="water"
)

# Ethanol molecules
ethanol = Molecule(
    Count=20,
    Smiles="CCO",
    Name="ethanol"
)

# Generate system
generate(system, "solvent_mixture", [water, ethanol], force_field="gaff")
```

[Full example: examples/example_molecules.py →](examples/example_molecules.py)

### Mixed System (Polymer + Solvent)

```python
# Polymer in water
polymer = Polymer(
    chain_num=5,
    sequence=["CC[*]"] + ["[*]CC[*]"] * 48 + ["[*]CC"]
)
water = Molecule(Count=100, Smiles="O", Name="water")

generate(system, "polymer_solution", [polymer, water], force_field="gaff")
```

### Ring Polymer

```python
# Ring: use only middle variants (all 2 wildcards)
polymer = Polymer(
    chain_num=5,
    sequence=["[*]CC[*]"] * 30,  # All middle
    topology="ring"              # Specify ring
)
```

### Monte Carlo Placement with Self-Avoiding Walk

AutoPoly uses a Monte Carlo (MC) self-avoiding walk (SAW) algorithm to generate realistic initial polymer configurations. Instead of placing chains on a grid, the SAW method grows each chain monomer-by-monomer with collision detection, producing coiled conformations that better approximate equilibrium structures.

```python
from AutoPoly import generate, GeometryConfig

generate(
    system,
    "polymer_mc",
    [polymer],
    force_field="oplsaa",
    strategy="mc_random",                 # Monte Carlo placement (default)
    mc_max_attempts=10000,                # Max placement attempts
    monomer_density=0.085,                # Target density (monomers/Å³)
    geometry_config=GeometryConfig(
        use_mc_chain_growth=True,         # SAW chain growth (default)
        mc_bond_angle_min=50.0,           # Min deflection angle (degrees)
        mc_bond_angle_max=90.0,           # Max deflection angle (degrees)
    ),
)
```

**Placement strategies:**
- `"mc_random"` (default) — Monte Carlo with SAW chain growth
- `"grid"` — Deterministic grid placement
- `"on_substrate"` — Film on a physical substrate slab (auto-selected when `substrate=` is given)

Box sizing uses SAW scaling (`N^0.6 × bond_length`) rather than fully-extended chain length, producing compact, realistic simulation boxes.

### Film on a Substrate

Pass a `SubstrateSpec` to build a polymer/molecule **film on top of a physical
slab** (fully periodic, two-interface slab model). The slab can be built
in-pipeline from a `Molecule`/`Polymer` (ordered `"grid"` or amorphous `"mc"`
packing), or imported as a pre-built external surface:

```python
from AutoPoly import System, Polymer, Molecule, generate
from AutoPoly.packing import SubstrateSpec, CutAbove, Cylinder

film = Polymer(chain_num=10, sequence=["CC[*]"] + ["[*]CC[*]"] * 18 + ["[*]CC"])

substrate = SubstrateSpec(
    model=Molecule(Count=64, Smiles="CCO", Name="etoh_sub"),
    thickness=10.0,   # slab z-extent (Å)
    packing="grid",   # ordered slab; "mc" = amorphous
    gap=3.0,          # empty space between slab top and film (Å)
)

generate(
    system, "pe_film", [film],
    force_field="gaff",
    substrate=substrate,              # auto-selects strategy="on_substrate"
    box_dims=(50.0, 50.0, 50.0),      # (lx, ly, lz); any entry may be None = auto
)
```

For a crystalline surface built elsewhere, point at its moltemplate class
instead: `SubstrateSpec(lt_file="au111.lt", class_name="Au111", thickness=12.0)`.

**Built-in silica substrates:** pass `builder=` to generate a hydroxylated
crystalline SiO2 slab at pack time (`AutoPoly.surfaces`) — no external
surface files needed:

```python
substrate = SubstrateSpec(
    builder="alpha_quartz",       # or "beta_cristobalite"
    thickness=12.0,               # slab envelope (Å), incl. hydroxyl coatings
    gap=3.0,
    slab_ff="interface",          # INTERFACE FF v1.5; "clayff" also built in
)

generate(system, "pe_on_quartz", [film], force_field="gaff",
         substrate=substrate, box_dims=(54.0, 51.0, 57.0))
```

- `builder="alpha_quartz"` — alpha-quartz(0001), geminal Q2 silanols
  (~9.6/nm² intrinsic); `builder="beta_cristobalite"` —
  beta-cristobalite(111), isolated Q3 silanols (~4.5/nm², the Zhuravlev
  density of amorphous silica). The cleavage plane is chosen automatically
  and both faces are hydroxylated (`hydroxylate_bottom=False` to leave the
  bottom bare).
- The lateral box snaps to integer surface cells (pass explicit lateral
  `box_dims`); the slab is self-typed with LJ + charges only (rigid-slab
  model — freeze or `fix rigid` the slab atoms in MD).
- Slab force fields: `slab_ff="interface"` (INTERFACE FF v1.5) or
  `"clayff"` built in, per-role overrides via
  `slab_types`/`slab_charges`/`slab_lj`, and a full `"custom"` mode.

[Examples: example_film_on_quartz.py →](examples/example_film_on_quartz.py) · [example_film_on_cristobalite.py →](examples/example_film_on_cristobalite.py)

**Subtract (carve):** remove whole instances after placement — chains and
molecules are removed intact, so **no covalent bonds are ever cut**:

```python
generate(
    system, "pe_film_patterned", [film],
    substrate=substrate,
    box_dims=(50.0, 50.0, 50.0),
    subtract=[
        Cylinder(axis="z", center=(0, 0), radius=8.0),  # hole through the film
        # CutAbove(z=25.0),                             # trim film to a thickness
        # CutAbove(z=-10.0, apply_to="substrate"),      # carve the slab instead
    ],
)
```

Regions (`CutAbove`, `CutBelow`, `Cylinder`, `BoxRegion`) apply to the film by
default (`apply_to="substrate"` or `"all"` to target the slab); subtract also
works with plain `mc_random` melts — carve a `Cylinder` through a melt box and
you have a nanopore.

[Full example: examples/example_film_on_substrate.py →](examples/example_film_on_substrate.py) · [Guide: Substrates & Films →](https://wugroup-xjtlu.github.io/AutoPoly/guides/substrates/)

### Reactive MD (Reactor)

The **Reactor** prepares LAMMPS `fix bond/react` inputs so monomers packed
into a box **react during MD** (step-growth polymerization). Build a monomer
melt, then let the reactor detect the polymerization and write the reaction
templates + input script:

```python
from AutoPoly import System, Molecule, generate, Reactor

system = System(out="reactor_out")
eg     = Molecule(Count=20, Smiles="OCCO", Name="eg")
adipic = Molecule(Count=20, Smiles="O=C(O)CCCCC(=O)O", Name="adipic")

generate(system, "melt", [eg, adipic], force_field="gaff2")

reactor = Reactor(system.get_folder_path() + "/melt", monomers=[eg, adipic])
reactor.detect_reactions()          # -> polyesterification (diol + diacid)
result = reactor.build()            # writes reactor/ + in.bond_react
```

This detects the diol + diacid polyesterification, builds the pre/post
reaction templates and map file (typed with the same force field), writes
supplementary parameters for the types the reaction creates (ester linkage,
water byproduct), and emits a ready-to-run script:

```bash
cd reactor_out/melt && lmp -in in.bond_react
```

Supported reaction families include polyesterification, polyamidation,
polyurethane formation, and polyanhydride/polythioester condensation, with a
custom reaction/functional-group library API. Inspired by
[AutoREACTER](https://github.com/NanoCIPHER-Lab/AutoREACTER).

[Full example: examples/example_reactor_polyester.py →](examples/example_reactor_polyester.py) · [Guide: Reactive MD →](https://wugroup-xjtlu.github.io/AutoPoly/guides/reactor/)

### Coarse-Grained Bead-Spring Models

For coarse-grained work, `BeadSpringPolymer` builds bead-spring chains on an
explicit **graph architecture** (`AutoPoly.models.architectures`), so
arbitrary topologies are supported uniformly — no rdkit required:

```python
from AutoPoly import System, BeadSpringPolymer, BeadType
from AutoPoly.models import architectures as arch

system = System(out="cg_comb")
bead_A = BeadType(name="A", mass=1.0, epsilon=1.0, sigma=1.0)  # backbone
bead_B = BeadType(name="B", mass=1.0, epsilon=1.0, sigma=1.0)  # side group

comb = arch.comb(
    backbone=[("A", 40)],  # 40 backbone beads
    side="B",              # one side-group bead per graft point
    every=4,               # graft every 4 backbone beads
)

polymer = BeadSpringPolymer(
    name="comb", system=system, n_chains=10,
    bead_types=[bead_A, bead_B],
    architecture=comb,
    bond_style="fene", pair_style="wca",
    use_angles=True, density=0.4,
)
polymer.generate()  # moltemplate backend (default) -> system.data + system.in.*
```

- **Architecture factories:** `linear`, `ring`, `star` (incl. miktoarm),
  `comb`, `graft` (explicit graft points / side groups), `tadpole`,
  `dendrimer`, and `custom` (explicit bead + bond lists). Legacy
  `sequence=`/`topology=` inputs remain fully backward compatible.
- **Multi-bead monomers:** `MonomerTemplate` +
  `BeadArchitecture.from_monomers(...)` define monomers with backbone +
  side-group beads and named connection points (`head`/`tail`/`side`).
- **Copolymer sequences:** `block_sequence`, `alternating_sequence`,
  `random_sequence` (seeded, weighted), `gradient_sequence`.
- **Branch-aware angles:** triplets centered on branch points get their own
  canonical angle types (`include_branch_angles=False` to disable).
- **Mixtures:** `BeadSpringSystem` packs multiple species
  (`add_species(architecture, n_chains)` — e.g. rings + linear + combs) into
  one box and one data file with a shared bead-type table.
- **Backends:** `generate()` dispatches to the **moltemplate** backend by
  default (standard AutoPoly output layout); use `generate(backend="direct")`
  for the lightweight direct writer (e.g. very large melts).
- **Growth & equilibration:** graph-based SAW (DFS spanning-tree growth with
  cycle-closing constraints; `SAWConfig.system_retries` retries crowded
  systems) and branched MC moves (tree-pivot, segment crankshaft).

[Examples: example_bead_spring.py →](examples/example_bead_spring.py) · [example_bead_spring_side_groups.py →](examples/example_bead_spring_side_groups.py)

### More Examples

The [examples directory](examples/) contains 18 runnable scripts covering:

- **Beginner tutorial** — PMMA step by step ([example_pmma_linear.py](examples/example_pmma_linear.py))
- **Film on substrate** — PE film on a slab + carve subtract ([example_film_on_substrate.py](examples/example_film_on_substrate.py))
- **Built-in silica substrates** — films on alpha-quartz(0001) and beta-cristobalite(111) ([example_film_on_quartz.py](examples/example_film_on_quartz.py), [example_film_on_cristobalite.py](examples/example_film_on_cristobalite.py))
- **Reactive MD** — monomer melt → `fix bond/react` polyesterification ([example_reactor_polyester.py](examples/example_reactor_polyester.py))
- **Three-stage pipeline** — GeometryBuilder → UnitTyper → BoxPacker, one geometry typed under multiple force fields ([example_three_stage_pipeline.py](examples/example_three_stage_pipeline.py))
- **Condensation polymers** — PLA with GAFF ([example_pla_condensation.py](examples/example_pla_condensation.py))
- **Polymer solutions** — PEO in explicit water ([example_peo_solution.py](examples/example_peo_solution.py))
- **Batch generation** — 10 commodity polymers ([example_commodity_polymers_10.py](examples/example_commodity_polymers_10.py))
- **Force field comparison** — all 6 force fields on PEO ([example_peo_all_forcefields.py](examples/example_peo_all_forcefields.py))
- **Placement methods** — grid vs MC random vs MC chain growth ([example_peo_mc_placement.py](examples/example_peo_mc_placement.py))
- **Bead-spring models** — coarse-grained homo/block/ring polymers ([example_bead_spring.py](examples/example_bead_spring.py))
- **Bead-spring architectures** — comb/graft with side groups, MonomerTemplate, moltemplate backend ([example_bead_spring_side_groups.py](examples/example_bead_spring_side_groups.py))

[See the full list with descriptions →](examples/README.md)

## API Quick Reference

### System

```python
System(out="folder_name")
```

Creates output directory for simulation files.

### Polymer

```python
Polymer(
    chain_num=10,               # Number of chains
    sequence=["[*]CC[*]"] * 50, # Explicit sequence (DOP=50)
    topology="linear",          # or "ring"
    tacticity="atactic"         # or "isotactic", "syndiotactic"
)
```

**Key points:**
- DOP is automatic from `len(sequence)`
- Use complement SMILES: first=`"CC[*]"`, middle=`"[*]CC[*]"`, last=`"[*]CC"`
- [Complete API →](https://wugroup-xjtlu.github.io/AutoPoly/reference/polymer/)

### Molecule

```python
Molecule(
    Count=100,       # Number of molecules
    Smiles="O",      # Regular SMILES (no wildcards)
    Name="water"     # Identifier
)
```

**Key point:** Use regular SMILES without `[*]` wildcards.

[Complete API →](https://wugroup-xjtlu.github.io/AutoPoly/reference/molecule/)

### generate

```python
generate(
    system,
    "project",
    [polymer1, polymer2, molecule1],  # Mix polymers and molecules
    force_field="oplsaa"  # See force fields below
)
```

**Supported force fields:**

| Force Field | Value | Best For |
|------------|-------|----------|
| OPLS-AA | `"oplsaa"` | General organic polymers |
| LOPLS | `"lopls"` | Liquid-phase, better densities |
| GAFF | `"gaff"` | Small molecules, drug-like |
| GAFF2 | `"gaff2"` | Updated GAFF |
| DREIDING | `"dreiding"` | Generic, metals, inorganics |
| COMPASS | `"compass"` | Commercial polymers |

[Force field selection guide →](https://wugroup-xjtlu.github.io/AutoPoly/guides/force-fields/)

[Complete API →](https://wugroup-xjtlu.github.io/AutoPoly/reference/)

## Force Field Selection

Quick guide:

- **Organic polymers** → OPLS-AA or LOPLS
- **Small molecules/solvents** → GAFF or GAFF2
- **Accurate densities** → LOPLS or COMPASS
- **Exploratory/generic** → DREIDING
- **Commercial polymers** → COMPASS

**Note:** Gasteiger charges are assigned automatically for GAFF/GAFF2. For production runs, replace them with AM1-BCC or RESP charges in `system.in.charges`.

[Detailed comparison →](https://wugroup-xjtlu.github.io/AutoPoly/guides/force-fields/)

## Troubleshooting

### Common Issues

**API errors:**
- `TypeError: 'ChainNum'` → Use `chain_num` (v1.0 uses snake_case)
- `TypeError: 'DOP'` → Removed, DOP = `len(sequence)` automatically

**Sequence errors:**
- `ValidationError: sequence cannot be empty` → Provide at least one monomer
- `ValidationError: Sequence length exceeds maximum` → Max DOP is 10000

**SMILES errors:**
- `ValidationError: Invalid SMILES` → Check wildcard count (first=1, middle=2, last=1)
- Ring polymers → Use only middle variants: `["[*]CC[*]"] * 50`

**Moltemplate:**
- `Moltemplate not found` → Install: `pip install moltemplate`

[Full troubleshooting guide →](https://wugroup-xjtlu.github.io/AutoPoly/guides/troubleshooting/)

## Output Structure

Each run writes into `<System out>/<name>/`:

```
my_polymer/polyethylene/
├── geometry/              # Stage 1: geometry.json (FF-agnostic coordinates)
├── build/oplsaa/          # Stage 2: typed .lt files + units.json manifest
├── moltemplate/           # Stage 3: intermediate files (.lt inputs, moltemplate output)
├── system.data            # LAMMPS data file (topology & coordinates)
├── system.in.init         # Units, atom/bond/angle styles
├── system.in.settings     # Force field parameters
└── system.in.charges      # Atomic charges
```

Run with LAMMPS by including the pieces from your own input script:
```bash
lmp -in your_run.in   # with: include system.in.init / system.in.settings
```

## More Information

**Documentation:**
- 📖 [Complete API Reference](https://wugroup-xjtlu.github.io/AutoPoly/reference/) - All classes and methods
- 🧬 [Complement SMILES Guide](https://wugroup-xjtlu.github.io/AutoPoly/guides/complement-smiles/) - Deep dive on SMILES format
- ⚙️ [Force Field Guide](https://wugroup-xjtlu.github.io/AutoPoly/guides/force-fields/) - Detailed comparison of all 6 force fields
- 🐛 [Troubleshooting](https://wugroup-xjtlu.github.io/AutoPoly/guides/troubleshooting/) - Solutions to common issues
- 📝 [Examples Directory](examples/) - 18 working examples

**Quick links:**
- [Installation details](https://wugroup-xjtlu.github.io/AutoPoly/guides/troubleshooting/#installation-issues)
- [Ring vs linear polymers](https://wugroup-xjtlu.github.io/AutoPoly/guides/complement-smiles/#ring-vs-linear-polymers)
- [Common monomer SMILES](https://wugroup-xjtlu.github.io/AutoPoly/guides/complement-smiles/#common-monomers-reference)

## Citation

If you use AutoPoly in your research, please cite:

```bibtex
@software{autopoly,
  title={AutoPoly: Automated Polymer Generation for Molecular Simulation},
  author={Wu, Zhenghao},
  url={https://github.com/WuGroup-XJTLU/AutoPoly}
}
```

## Acknowledgements

The **Reactor** module (reactive MD via `fix bond/react`) is adapted from
[AutoREACTER](https://github.com/NanoCIPHER-Lab/AutoREACTER) (NanoCIPHER
Lab, MIT license): the reaction-detection logic, functional-group SMARTS
library, reaction library, and template-walking algorithm are
reimplementations of AutoREACTER's `ReactionDetector`, `PrepareReactions`,
and `walker` modules, integrated into AutoPoly's own typing and packing
pipeline. If you use the Reactor in your work, please also cite AutoREACTER.

## License

MIT License - see [license.md](license.md) for details.
