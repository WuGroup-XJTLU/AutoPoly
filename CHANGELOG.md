# Changelog

All notable changes to AutoPoly are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Polymer nanocomposites** (`AutoPoly.nanoparticles`, `AutoPoly.materials`)
  — a `Nanoparticle` passed to `generate()` next to `Polymer`/`Molecule`
  models builds finite inorganic particles in a polymer matrix:
  - *Materials foundation* (`AutoPoly.materials`): `AtomicStructure`
    (elements, positions, bonds, periodic images, cell, roles, surface
    sites, provenance), periodic topology helpers, crystal tables
    (alpha-quartz, beta-cristobalite, corundum, rutile, anatase, fcc Au/Ag,
    graphene; generated tables reproducible with
    `tools/generate_crystal_tables.py`), a material registry with six IDs
    (`sio2`, `al2o3`, `tio2`, `c`, `au`, `ag`), per-material chemistry
    adapters (silica hydroxylation/condensation, oxide bond-valence
    protonation, carbon edge passivation, fcc metal clusters) and a shared
    moltemplate writer (`AutoPoly.exporters.moltemplate`).
  - *Shapes*: crystal-carved spheres (oxides, metals), ASE metal
    polyhedra (cuboctahedron/octahedron/Wulff), graphene sheets, (n, m)
    nanotubes, and imported clusters (`Nanoparticle.from_file`: JSON, PDB
    with CONECT, XYZ with bond perception), validated against the
    material's valence rules.
  - *Inorganic force-field adapters* (registered, versioned): INTERFACE
    FF v1.5 silica, CLAYFF silica and alumina, OPLS-AA aromatic carbon,
    Heinz 2008 12-6 Au/Ag; user adapters via `register_inorganic_ff`.
  - *Explicit cross interactions*: one `pair_coeff i j` per
    organic-inorganic (and inorganic-inorganic) type pair appended to
    `system.in.settings`, with the combining rule each parameter set was
    fitted with and the host's pair sub-style; `ff_compatibility.json` /
    `.md` report per build.
  - *Qualification matrix* (`AutoPoly.materials.qualification`): MD
    export only for exact qualified combinations (material, phase,
    geometry, surface, inorganic FF, polymer FF, motion, linker), each
    backed by a LAMMPS smoke test; `md_ready` is derived per material,
    never declared. TiO2 is geometry-only. `allow_unqualified=True`
    records user-qualified builds.
  - *`nanocomposite_mc` packing* (`NanocompositePacking`): fillers first
    with periodic atom-level gaps, then the matrix (rigid pre-grown
    conformations, regrown in the box around obstacles when they do not
    fit), conservative voxel occupancy plus an exact minimum-image
    verification; never falls back to overlapping positions. The report
    records minimum distances, box sizing and loading.
  - *Grafted chains* (`Grafting`): silyl-ether grafting onto silica
    silanols at a target density, anchored growth around the core,
    charge-conserving grafted-O type, linkage bonded terms read from the
    bundled `oplsaa.lt` by OPLS bonded class.
  - *Run template*: `in.run` with fillers held during minimization, one
    rigid body per particle (`fix rigid/nvt molecule`) or frozen
    particles, matrix NVT with a matrix-only temperature, optional
    compression stage; `system.in.groups` with per-instance, per-species
    and per-motion groups. `AutoPoly.pipeline.smoke.run_smoke()` runs a
    short LAMMPS health check of any project.
  - Docs: `guides/nanocomposites.md`, reference page; examples
    `example_silica_nanoparticle.py`, `example_pmma_silica_nanocomposite.py`,
    `example_silica_grafted_polymer.py`, `example_alumina_pmma.py`,
    `example_titania_polymer.py`, `example_graphene_polymer.py`,
    `example_au_polymer.py`.
- **Coarse-grained fillers for the bead-spring backend** —
  `BeadSpringSystem.add_filler(CGFiller(...))`: rigid or frozen spheres
  of LJ beads (solid FCC or surface shell) placed before the chains, which
  grow around them by SAW with periodic overlap checks; both backends
  write one molecule per filler and the run script integrates rigid
  fillers with `fix rigid/nvt molecule`. `set_pair_coeff()` overrides a
  bead-pair interaction (optionally with its own cutoff), e.g. an
  attractive polymer-filler pair in a WCA melt. `saw_generate_graphs`
  gains an optional `obstacles=` argument (default unchanged). Example:
  `example_cg_nanocomposite.py`.
- **Golden equivalence tests** for the built-in silica substrates
  (`tests/golden/`): geometry, topology, typed coefficients, generated
  files and LAMMPS energies are compared with the pre-refactoring
  reference.

- **LAMMPS run script for substrate systems** — `generate()` with
  `substrate=` now writes a ready-to-run `in.run` (frozen slab via
  `fix setforce`, film NVT 300 K with a film-only temperature compute,
  so the logged temperature is not biased by the frozen slab).
- **Early film-density warning** — the `on_substrate` strategy warns at
  placement time when the film units' bounding-sphere volume exceeds
  30% of the film region (whole-chain MC placement is then likely to
  fail).
- **Regression tests** for the issues exposed by running built
  film-on-substrate systems through LAMMPS.

### Changed

- **Built-in silica slabs go through the materials foundation** —
  `SubstrateSpec(builder=...)` slabs are recorded as stage-1 inorganic
  entries in `geometry.json` and typed in stage 2 by the registered
  silica adapters (`AutoPoly.surfaces` is now a compatibility facade over
  `AutoPoly.materials`). Calls, geometry, topology and charges are
  unchanged (golden tests); see the CLAYFF fix below for the one
  intentional parameter change.
- **Manifest formats v2** — `units.json` (format 2) adds the `inorganic`
  kind, `filler`/`graft` roles, motion, material provenance and particle
  envelopes; `geometry.json` (format 2) adds an `inorganic` section and is
  still written as version 1 when there is none. Version-1 files are
  read unchanged.
- **Strategy checks** — `mc_random`, `grid` and `on_substrate` refuse
  inorganic filler units with a pointer to `nanocomposite_mc`; passing a
  `Nanoparticle` together with an incompatible `strategy=` or with
  `substrate=` raises instead of silently ignoring one of them.
  `UnitLibrary.total_particle_count` counts organic units only.

### Fixed

- **CLAYFF sigma** — CLAYFF publishes the well-minimum distance R0;
  the built-in table converted R0 to sigma twice, so CLAYFF Si/O sigmas
  were 2^(1/6) too small (e.g. Si 2.942 instead of 3.302 A, O 2.820
  instead of 3.166 A). Slabs built with `slab_ff="clayff"` now carry the
  published values; only these sigmas and the resulting LJ energies
  change (checked by the golden tests). INTERFACE FF is unaffected.
- **Bead-spring moltemplate backend: `in.polymer` did not run** — it read
  `system.data` before `include system.in.init`, so LAMMPS aborted with
  "No bonds allowed with this atom style"; the init file is now included
  first.
- **Chain regrowth junction geometry** — the open valence of a template
  connection atom now uses the vacant tetrahedral/trigonal site (minus
  the sum of its bond vectors), falling back to the clearest direction
  only for flattened fragment conformers; the clearance-only criterion
  gave ~95 degree backbone angles for PMMA and made in-box regrowth fail
  in most attempts.
- **Unclear cross-rule wording** in `ff_compatibility.md`.
- **Spurious intra-slab nonbonded interactions** — built substrate
  slabs now emit their Si-O/O-H bond topology with zero-force-coefficient
  bond terms, so `special_bonds` excludes intra-slab 1-2/1-3/1-4
  nonbonded interactions (bonded O-H pairs at 0.945 A previously
  contributed ~1e7 kcal/mol of spurious LJ/Coulomb energy). The
  `pair_coeff`/`bond_coeff` lines automatically carry the sub-style
  required by the film force field's hybrid pair/bond styles.
- **LAMMPS crash on unused hybrid styles** — `system.in.init` is
  patched after moltemplate: hybrid bond/angle/dihedral/improper styles
  with zero records in `system.data` are replaced with their `none`
  form (e.g. `improper_style hybrid cvff` for polyethylene).
- **Silent instance stacking in placement fallbacks** — both the
  `mc_random` and `on_substrate` fallback grids now raise
  `ValidationError` when full instead of placing instances on identical
  coordinates (produced atom clashes in the data file). Note: several
  pre-existing tests had unknowingly relied on the silent stacking.
- **`mc_random` ignored `box_dims`** — explicit per-axis boxes are now
  honored (per-axis auto-size for unspecified axes), as documented on
  `BoxSpec`.

- **Built-in crystalline silica substrates** — `SubstrateSpec` gains a
  third source, `builder=`: a hydroxylated crystalline SiO2 slab is
  generated at pack time (`AutoPoly.surfaces`, following the mBuild
  carve-and-functionalize recipe adapted to crystalline bulk), written
  as a self-contained moltemplate `.lt` class and instantiated once like
  an external slab. Two surfaces are available:
  `builder="alpha_quartz"` — alpha-quartz(0001), geminal Q2 silanols
  (~9.6/nm^2 intrinsic); and `builder="beta_cristobalite"` —
  beta-cristobalite(111), isolated Q3 silanols (~4.5/nm^2, matching the
  Zhuravlev density of amorphous silica). The cleavage plane is chosen
  automatically (fewest broken bonds, full Si coordination); both faces
  are hydroxylated; the lateral box snaps to integer surface cells. The
  slab is self-typed with LJ + charges only (rigid-slab model), with
  built-in INTERFACE FF v1.5 (Emami et al. 2014) and CLAYFF (Cygan et
  al. 2004) tables selected via `slab_ff=`, plus per-role overrides
  (`slab_types`/`slab_charges`/`slab_lj`) and a full `"custom"` mode.
  See the substrates guide, `examples/example_film_on_quartz.py` and
  `examples/example_film_on_cristobalite.py`.

- **Bead-spring architectures (graph core)** — bead-spring chains are now
  built on an explicit graph representation (`BeadArchitecture`: nodes =
  beads, edges = bonds) instead of an implicit linear path, so arbitrary
  topologies are supported uniformly:
  - New module `AutoPoly.models.architectures` with factories `linear`,
    `ring`, `star` (incl. miktoarm/asymmetric arms), `comb`, `graft`
    (explicit graft points / side groups), `tadpole` (ring + tail),
    `dendrimer`, and `custom` (explicit bead + bond lists).
  - `MonomerTemplate` + `BeadArchitecture.from_monomers(...)` for
    explicitly defined multi-bead monomers (backbone bead, side-group
    bead) with named connection points (`head`/`tail`/`side`).
  - Copolymer sequence generators: `block_sequence`,
    `alternating_sequence`, `random_sequence` (seeded, weighted),
    `gradient_sequence`; usable in any sequence slot.
  - `BeadSpringPolymer(..., architecture=...)` accepts any architecture;
    legacy `sequence=`/`topology=` remain fully backward compatible.
  - Branch-aware angles: triplets centered on branch points are included
    by default with their own canonical triplet types (configurable via
    `AngleType`); disable with `include_branch_angles=False`.
  - Graph-based SAW generation (`saw_grow_graph`, `saw_generate_graphs`):
    DFS spanning-tree growth with cycle-closing constraints.
  - Branched MC equilibration: tree-pivot moves across bridge edges
    (`mc_tree_pivot_move`) and segment crankshaft
    (`mc_segment_crankshaft_move`); reptation is linear-only.
- **Bead-spring mixtures** — new `BeadSpringSystem` packs multiple
  species (`(architecture, n_chains)` pairs — e.g. rings + linear +
  combs) into one box and one LAMMPS data file with a shared bead-type
  table.
- **Moltemplate is now the default backend for bead-spring models** —
  new unified `generate()` entry point on `BeadSpringPolymer` and
  `BeadSpringSystem` dispatches to the moltemplate backend by default
  (constructor `backend="moltemplate"`; override per call with
  `generate(backend="direct")` for the lightweight direct writer, e.g.
  very large melts). `generate_moltemplate()` and `generate_data_file()`
  remain available for explicit control.
- **SAW whole-system retries** — `SAWConfig.system_retries` (default 3)
  retries the whole multi-chain SAW generation when a single crowded
  chain fails, instead of immediately falling back to geometric
  placement.
- **Example: bead-spring with side groups** —
  `examples/example_bead_spring_side_groups.py` demonstrates a comb with
  single-bead side groups + branch angles, a graft copolymer with
  oligomeric side chains, a comb built from an explicit
  `MonomerTemplate`, and the moltemplate backend.
- **Moltemplate backend for bead-spring models** — `generate_moltemplate()`
  on both `BeadSpringPolymer` and `BeadSpringSystem` emits
  `bead_spring.lt` (CG force field), `bead_<Type>.lt` monomer objects,
  `chains.lt` (one object per chain at generated coordinates, explicit
  bond list, typed angle list), and `system.lt`, then optionally runs the
  bundled moltemplate to produce `system.data` + `system.in.init/settings`.
- **Reactor** (`AutoPoly.reactor`) — AutoREACTER-style reactive-MD
  preparation for LAMMPS `fix bond/react`:
  - `Reactor(project_dir, monomers=...)` scans the packed monomers for
    reactive functional groups, enumerates step-growth polymerization
    reactions from a built-in reaction-SMARTS library
    (polyesterification, polyamidation, polyurethane formation,
    polyanhydride/polythioester condensation), and builds the full
    `fix bond/react` input set via `reactor.build()`.
  - Writes per-reaction pre/post molecule templates and the superimpose
    map file (including a `_with_delete_ids.map` variant for
    condensation byproducts such as water), typed with the system's force
    field. Numeric types reuse the ids already present in `system.data`
    (by-example matching); genuinely new interactions (cross-monomer bonds,
    rehybridized atoms, byproducts) get fresh ids with coefficients taken
    from the force-field tables into `reactor/system.in.settings.reactor`.
  - Writes a ready-to-run `in.bond_react` reaction-stage script that
    includes `system.in.init/.settings/.charges`, reads `system.data` with
    the extra per-atom topology slots, and thermostats the non-reacting
    group (`statted_grp_REACT`).
  - Stage-level helpers are public: `detect_monomer_roles`,
    `detect_reactions`, `prepare_reactions`, plus `ForceFieldTables` /
    `SystemData` parsers and the `TemplateBuilder`. Custom functional-group
    and reaction libraries are supported.
  - Exported at the top level as `Reactor` / `ReactorResult`; see
    `examples/example_reactor_polyester.py` and `docs/guides/reactor.md`.
- **Physical substrates** — build a polymer/molecule film on top of a
  substrate slab via `generate(..., substrate=SubstrateSpec(...))`:
  - `SubstrateSpec(model=...)` packs a `Molecule`/`Polymer` into a slab of
    given `thickness` at the bottom of the box (`packing="grid"` for an
    ordered slab, `"mc"` for amorphous), with a `gap` to the film region.
    The substrate is typed in-pipeline alongside the film models;
    `UnitSpec.role` ("film"/"substrate") records the partition.
  - `SubstrateSpec(lt_file=..., class_name=...)` instantiates a pre-built
    external surface (e.g. an Au(111) or SiO2 slab) once, centered
    laterally at the slab mid-plane.
  - `count="auto"` derives a `Molecule` substrate's instance count from
    slab volume × `density` (requires explicit lateral `box_dims`).
- **New packing strategy `on_substrate`** (auto-selected when `substrate`
  is passed): z-layered assembly — slab at the bottom, film MC-placed
  above `slab_top + gap`, shared collision detector so film and slab
  never interpenetrate. Box sides are set per axis with
  `box_dims=(lx, ly, lz)` (any element `None` = auto;
  `lz = thickness + gap + film thickness at monomer_density + vacuum`).
- **Subtract (carve regions)** — whole-instance removal after placement,
  no covalent bonds cut: `CutAbove`, `CutBelow`, `Cylinder`, `BoxRegion`
  from `AutoPoly.packing`, passed as `generate(..., subtract=[...])`.
  Each region has `apply_to` ("film"/"substrate"/"all") and an optional
  `conservative=True` sphere-inflated boundary. Supported by the
  `mc_random` and `on_substrate` strategies (e.g. carve a `Cylinder`
  through a bulk melt to build a nanopore); removals are logged.
- `BoxSpec.box_dims` for rectangular (non-cubic) boxes.

## [2.0.0] - 2026-07-30

AutoPoly 2.0 is a ground-up rearchitecture of the generation pipeline. It is a
**breaking release**: the `Polymerization` class, the agent/CLI surface, and
loose monomer inputs are gone. See the migration guide below.

### Breaking changes

| 1.x | 2.0 | Notes |
|---|---|---|
| `Polymerization(name=..., system=..., model=..., force_field=...)` | `generate(system, name, models, force_field=...)` | One-shot function composing the three pipeline stages; returns a `PlacementResult` |
| `Polymerization(..., run=False)` then manual method calls | `GeometryBuilder` → `UnitTyper` → `BoxPacker` | The stages are now public, independently callable APIs |
| `Polymerization(placement_method="mc_random"/"grid")` | `generate(..., strategy="mc_random"/"grid")` | Strategies are pluggable via `register_strategy()` |
| `Polymerization(mc_monomer_density=...)` | `generate(..., monomer_density=...)` | Renamed |
| `Polymerization(mc_bond_angle_min/max=..., mc_intrachain_exclude_neighbors=...)` | `generate(..., geometry_config=GeometryConfig(...))` | Chain-growth parameters live in `GeometryConfig` |
| `Polymer(chain_num=..., sequence=["PE", "PS"], ...)` with monomer **names** | `Polymer(chain_num=..., sequence=["CC[*]", "[*]C=C"], ...)` with **pSMILES** | Sequences must be complement SMILES with explicit wildcards |
| Implicit DOP / sequence cycling | `DOP = len(sequence)`, no cycling | The sequence you pass is the chain you get |
| `AutoPoly.agent`, `AutoPoly.cli`, `AutoPoly.tools` (LangChain agent & CLI) | removed | Library-only package; drive it from Python |
| `from AutoPoly.polymer import Polymer` (flat layout) | `from AutoPoly import Polymer` (unchanged) or `AutoPoly.models.polymer` | Package reorganized into `core/`, `models/`, `monomers/`, `forcefields/`, `pipeline/`, `packing/`, `mc/` subpackages |

### Stricter pSMILES validation

`Polymer` now validates wildcard counts per position at construction time:

- **First monomer:** exactly 1 wildcard (right connection), e.g. `"CC[*]"`
- **Middle monomers:** exactly 2 wildcards, e.g. `"[*]CC[*]"`
- **Last monomer:** exactly 1 wildcard (left connection), e.g. `"[*]CC"`
- **Single-monomer sequence (DOP=1):** 0 wildcards, e.g. `"CC"`

Monomer names such as `"PE"` or `"PS"` are no longer accepted and raise
`ValidationError` with a message identifying the offending position.

### Added

- **Three-stage pipeline**, each stage usable standalone:
  - `GeometryBuilder` (stage 1): force-field-agnostic chain growth → `geometry/geometry.json`
  - `UnitTyper` (stage 2): atom typing → `build/<ff>/*.lt` + `units.json`; type one geometry for multiple force fields without regrowing chains
  - `BoxPacker` (stage 3): packing + moltemplate → `system.data`, `system.in.*`
- `generate()` one-shot convenience function (`AutoPoly.pipeline.workflow`)
- `UnitLibrary` / `UnitSpec` manifest contract between stages 2 and 3
- Pluggable packing strategies: `PlacementStrategy`, `register_strategy()`, `get_strategy()`; built-ins `mc_random` and `grid`
- `GeometryConfig` for MC chain-growth parameters
- GAFF `gaff.lt` now covers the `nu`/`nv` amine-to-aromatic nitrogen types
  (parameters aliased from `nh`), closing a runtime gap where rdlt could
  assign types that moltemplate could not parameterize

### Removed

- `AutoPoly.agent` (LangChain agent), `AutoPoly.cli`, `AutoPoly.tools`
- `Polymerization` class (superseded by `generate()` and the stage APIs)
- Implicit DOP parameter and monomer cycling in `Polymer`

### Migration

**1.x:**

```python
from AutoPoly import System, Polymer, Polymerization

system = System(out="my_polymer")
poly = Polymer(chain_num=4, sequence=["PE"] * 50)
Polymerization(name="pe", system=system, model=[poly], force_field="oplsaa")
```

**2.0:**

```python
from AutoPoly import System, Polymer, generate

system = System(out="my_polymer")
sequence = ["CC[*]"] + ["[*]CC[*]"] * 48 + ["[*]CC"]  # DOP = 50
poly = Polymer(chain_num=4, sequence=sequence)
generate(system, "pe", [poly], force_field="oplsaa")
```

For multi-force-field work, build the geometry once and type it per force field:

```python
from AutoPoly import GeometryBuilder, UnitTyper, BoxPacker

GeometryBuilder(system, "pe").build([poly])  # writes <out>/pe/geometry/
for ff in ("oplsaa", "gaff", "gaff2"):
    units = UnitTyper("<out>/pe/geometry", force_field=ff).type()
    BoxPacker(system, "pe").pack(units)
```

See the [quickstart](docs/getting-started/quickstart.md) and
[workflow guide](docs/guides/workflow.md) for full examples.

## [1.0.0]

Initial stable release: `Polymerization` workflow, OPLS-AA/GAFF/GAFF2/l-OPLS
typing, MC chain growth and placement, bead-spring models, LangChain agent
and CLI.
