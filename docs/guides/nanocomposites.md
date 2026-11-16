# Nanocomposites (Nanoparticles & Fillers)

AutoPoly builds **polymer nanocomposites**: a polymer (or molecule) matrix
with finite inorganic particles — silica, alumina, titania, carbon
(graphene, nanotubes), gold and silver — placed in one fully periodic box.
A `Nanoparticle` is passed to `generate()` next to your `Polymer` /
`Molecule` models; nothing else in the workflow changes.

```python
from AutoPoly import System, Polymer, generate
from AutoPoly.nanoparticles import Nanoparticle
from AutoPoly.packing import NanocompositePacking

PS = ["CC(c1ccccc1)[*]"] + ["[*]CC(c1ccccc1)[*]"] * 6 + ["[*]CC(c1ccccc1)"]

system = System(out="ps_silica")
ps = Polymer(chain_num=16, sequence=PS, tacticity="atactic")
silica = Nanoparticle(material="SiO2", structure="beta_cristobalite",
                      radius=9.0, count=2, motion="rigid")

result = generate(system, "ps_silica", [ps, silica], force_field="oplsaa",
                  packing=NanocompositePacking(seed=3), rng_seed=3)
print(result.report["loading"])
```

The project directory then holds `system.data`, `system.in.*` (with the
explicit polymer–particle pair coefficients appended to
`system.in.settings`), `system.in.groups`, a starting `in.run`, and a
force-field compatibility report (`ff_compatibility.json` / `.md`).

!!! warning "What `md_ready` means"
    AutoPoly only exports an MD system for an **exact combination** of
    material, phase, shape, surface, inorganic force field, polymer force
    field and motion that has been qualified (see
    [Qualification matrix](#qualification-matrix)). Building a geometry
    never implies that a validated interaction model exists for it.

## Materials

Six material IDs are registered (`AutoPoly.materials.get_material`;
formulas and aliases such as `"silica"`, `"alumina"`, `"graphene"`,
`"gold"` resolve too):

| ID | Phases / structures | Shapes | Surfaces | Inorganic FF (default first) | md_ready |
|---|---|---|---|---|---|
| `sio2` | `beta_cristobalite` (default), `alpha_quartz`, `amorphous` (imported only) | sphere, imported cluster, slab (`SubstrateSpec`) | `hydroxylated` (Q3/Q2 silanols, optional condensation) | `interface_silica` (INTERFACE FF v1.5), `clayff_silica` | yes |
| `al2o3` | `alpha_corundum` | sphere | `hydroxylated`, `stoichiometric` (dry) | `clayff_alumina` | yes |
| `tio2` | `anatase` (default), `rutile` | sphere | `hydroxylated` | — | **no** (geometry only) |
| `c` | `graphene`, `cnt`, `carbon_black` (imported only) | sheet, tube | `h_passivated` (edge C–H) | `oplsaa_aromatic_carbon` | yes |
| `au` | `fcc` | sphere, cuboctahedron, octahedron, wulff | `bare` | `interface_metal_au` (Heinz 2008) | yes |
| `ag` | `fcc` | as Au | `bare` | `interface_metal_ag` (Heinz 2008) | yes |

Each material reports its capability state — `md_ready` is *derived*
from the qualification matrix, never declared:

```python
from AutoPoly.materials import get_material
get_material("TiO2").capability_report()
# {'registered': True, 'geometry_ready': True, 'surface_ready': True, 'md_ready': False}
```

## Building particles

```python
from AutoPoly.nanoparticles import Nanoparticle, SilicaSurface, Sphere, Sheet, Tube, Polyhedron

Nanoparticle(material="SiO2", radius=12.0,                       # sphere (A)
             surface=SilicaSurface(oh_density=4.6), force_field="interface")
Nanoparticle(material="Al2O3", radius=8.0, surface="stoichiometric")
Nanoparticle(material="C", shape="sheet", length=30.0, width=25.0, layers=1)
Nanoparticle(material="C", structure="cnt", shape="tube", n=10, m=10, length=30.0)
Nanoparticle(material="Au", shape="cuboctahedron", radius=10.0)  # needs ASE
```

| Argument | Meaning |
|---|---|
| `material` | Material ID / formula / alias, or a `Material` object |
| `structure` (`phase`) | Phase; default per material |
| `shape` + size | `"sphere"` (`radius`, optional `center_offset`), `"sheet"` (`length`, `width`, `layers`), `"tube"` (`n`, `m`, `length`), `"cuboctahedron"`/`"octahedron"`/`"wulff"` (`radius`), or a `Sphere`/`Sheet`/`Tube`/`Polyhedron` object |
| `surface` | Termination ID or `Surface` object; default per material |
| `force_field` (`atom_types`) | Inorganic adapter name or alias (`"interface"`, `"clayff"`) |
| `count` | Identical instances to place |
| `motion` | `"rigid"` (one LAMMPS rigid body per instance, default) or `"frozen"`; `"flexible"` is refused (no bundled set has validated bonded terms) |
| `name` | Component ID (unique per system) |
| `grafts` | Covalently grafted chains ([Grafting](#grafted-chains)) |
| `allow_unqualified` | Export a combination outside the qualification matrix (recorded as user-qualified) |

How the geometries are made:

- **Oxides** are carved from the bulk crystal around a cation: every
  framework cation within `radius` is kept with its full oxygen shell,
  under-coordinated fragments are removed, and dangling oxygens are
  terminated. Silica keeps Si 4-coordinated and caps dangling O with H
  (Si–O–H 116°, H placed in the clearest direction); a target
  `oh_density` is approached by Si–O–Si condensation where geometrically
  valid. Alumina and titania protonate under-coordinated O in
  bond-valence-deficit order until the particle is neutral under formal
  charges; the dry alumina cut removes the least-coordinated O instead.
- **Carbon**: AB-stacked graphene flakes and (n, m) single-wall tubes;
  edge carbons with two C neighbors get one H, dangling carbons are
  removed.
- **Metals**: fcc spheres or ASE polyhedra; no bonds.

`Nanoparticle.summary()` reports the formula, mass, envelope, core
volume, accessible area and the surface statistics (silanol count,
achieved OH density, Q distribution, …); `write_structure("np.pdb")`
writes PDB (with CONECT), XYZ or the native JSON.

!!! note "Silanol density of crystal-cut silica"
    Crystal-derived spheres carry their intrinsic silanol density (about
    5–6/nm² for cristobalite, 7–8/nm² for quartz). Condensation lowers it
    only where two dangling oxygens are close enough to bridge; the
    achieved value is always reported, and a warning is logged when the
    target is not reached. For low-OH amorphous silica, import a relaxed
    structure.

### Imported clusters

```python
np_ = Nanoparticle.from_file("silica_amorphous.pdb", material="SiO2",
                             count=2, motion="rigid")
np_ = Nanoparticle.from_file("cluster.xyz", material="Al2O3",
                             perceive_bonds=True)   # XYZ has no bonds
```

JSON (as written by `write_structure`), PDB with CONECT records and XYZ
are read. The structure is validated against the material's valence
rules; roles — and therefore charges and LJ parameters — always come
from the material's chemistry adapter and the inorganic force field,
never from the file.

## Inorganic force fields

Adapters are registered, versioned parameter sets
(`AutoPoly.materials.forcefields`). All bundled sets are nonbonded
(12-6 LJ + charges), which is why particles are rigid or frozen; intra-
particle pairs are excluded via zero-force bonds and
`neigh_modify exclude molecule/intra`.

| Adapter | Material | Combining rule for cross terms | Reference |
|---|---|---|---|
| `interface_silica` (`"interface"`) | SiO2 | host rule | Emami et al., *Chem. Mater.* 2014 (INTERFACE FF v1.5) |
| `clayff_silica` (`"clayff"`) | SiO2 | arithmetic | Cygan et al., *J. Phys. Chem. B* 2004 |
| `clayff_alumina` | Al2O3 | arithmetic | Cygan et al. 2004 (ao/ob/oh/ho) |
| `oplsaa_aromatic_carbon` | C | host rule | OPLS-AA CA/HA |
| `interface_metal_au` / `interface_metal_ag` | Au / Ag | host rule | Heinz et al., *J. Phys. Chem. C* 2008 |

Choose an adapter per component (`force_field=` on the particle, or
`generate(..., inorganic_force_fields={"<component id>": "clayff"})`).
The Heinz metal model is an effective interfacial 12-6 LJ potential;
many-body EAM potentials are never mixed with organic LJ.

### Explicit cross interactions

AutoPoly does not rely on `pair_modify mix` for polymer–particle pairs.
After moltemplate runs, one explicit `pair_coeff i j` line per
(organic type, inorganic type) pair — and per pair of inorganic types —
is appended to `system.in.settings`, computed with the rule the
inorganic parameter set was fitted with (arithmetic for CLAYFF; the host
rule for INTERFACE FF, OPLS-AA carbon and the metal model). The host
style is respected (e.g. `lj/charmm/coul/long` sub-styles for OPLS-AA).
`ff_compatibility.md` lists the host styles, every component with its
qualification, the cross rule and the number of explicit pairs.

Supported hosts: OPLS-AA, GAFF and GAFF2 (12-6 LJ). Class II hosts
(PCFF/COMPASS, 9-6) and non-LJ hosts are refused; LOPLS is not
qualified.

## Qualification matrix

An MD export is allowed only when the exact combination is listed (each
entry is backed by a LAMMPS smoke test,
`tests/test_nanocomposite_lammps_smoke.py`):

| ID | Phases | Geometry | Surface | Inorganic FF | Polymer FFs | Motion | Cross rule | Grafting |
|---|---|---|---|---|---|---|---|---|
| `sio2-interface-hydroxylated` | cristobalite, quartz, amorphous | cluster, imported | hydroxylated | interface_silica | oplsaa, gaff, gaff2 | rigid, frozen | host | – |
| `sio2-interface-grafted` | cristobalite, quartz | cluster | hydroxylated | interface_silica | oplsaa | rigid, frozen | host | silyl_ether |
| `sio2-clayff-hydroxylated` | cristobalite, quartz, amorphous | cluster, imported | hydroxylated | clayff_silica | oplsaa | rigid | arithmetic | – |
| `al2o3-clayff-hydroxylated` | corundum | cluster | hydroxylated | clayff_alumina | oplsaa | rigid, frozen | arithmetic | – |
| `al2o3-clayff-stoichiometric` | corundum | cluster | stoichiometric | clayff_alumina | oplsaa | rigid | arithmetic | – |
| `c-oplsaa-graphene` | graphene, cnt | sheet, rod | h_passivated | oplsaa_aromatic_carbon | oplsaa | rigid, frozen | host | – |
| `au-interface-lj` | fcc | cluster | bare | interface_metal_au | oplsaa, gaff2 | rigid | host | – |
| `ag-interface-lj` | fcc | cluster | bare | interface_metal_ag | oplsaa | rigid | host | – |

The live matrix is available as data and as Markdown:

```python
from AutoPoly.materials.qualification import qualification_matrix, qualification_markdown
print(qualification_markdown())
```

Stages: (A) registry, (B) geometry, (C) surface, (D) inorganic FF,
(E) explicit cross terms, (F) LAMMPS smoke. A combination outside the
matrix stops with an explanation; `allow_unqualified=True` lets you
proceed with your own validated parameters and is recorded as
`"user-qualified"` in the report.

### Your own parameters

Register an adapter covering the roles the material's chemistry adapter
assigns (e.g. TiO2: `Ti`, `OB`, `OH`, `OW`, `HO`, `HW`):

```python
from AutoPoly.materials.forcefields import register_inorganic_ff
from AutoPoly.materials.forcefields.registry import InorganicForceField, RoleParameters

register_inorganic_ff(InorganicForceField(
    name="my_tio2", version="1", label="TiO2 (my validated set)",
    material_ids=("tio2",), combining_rule="arithmetic",
    roles={"Ti": RoleParameters("ti", 47.867, q_ti, eps_ti, sig_ti), ...},
    citations=("...",),
))
Nanoparticle(material="TiO2", radius=8.0, force_field="my_tio2",
             allow_unqualified=True)
```

Particles must be neutral: a net charge is refused (counter-ions are not
supported).

## Packing

Passing a `Nanoparticle` selects the `"nanocomposite_mc"` strategy
(configure it with `packing=NanocompositePacking(...)`):

1. **Box** — from `box_size` / `box_dims`, or sized so the matrix has
   `initial_density` (g/cm³, default 0.35) in the volume outside the
   particle cores. Pre-grown chains cannot be packed at melt density;
   build dilute and compress in MD (`target_polymer_density` adds a
   compression stage to `in.run`).
2. **Fillers first** — random positions and orientations with atom-level
   surface gaps ≥ `min_gap` (3 Å) under periodic boundaries, periodic
   self-images included.
3. **Matrix** — each chain is tried as a rigid pre-grown conformation;
   when it does not fit (or self-overlaps), it is **regrown inside the
   box** monomer by monomer around the obstacles (open-valence junctions,
   sampled torsions, 1-4 checks, backtracking). Matrix atoms keep
   `filler_clearance` (2.5 Å) from particle atoms and
   `polymer_clearance` / `hydrogen_clearance` (1.7 / 1.2 Å) from other
   instances.
4. **Exact verification** — all minimum distances are recomputed with
   the minimum image; nothing is ever placed on top of something else
   (`packing_failure="raise"`: an infeasible build stops with a
   diagnosis and suggestions instead of overlapping).

`result.report` records the minimum distances, the box, the loading
(nominal filler volume and weight fraction, matrix mass, initial and
target densities), the regrown chains and an intra-chain contact check.

Substrate slabs and nanoparticles cannot be combined in one build yet.

## Grafted chains

Silica particles can carry covalently grafted chains (OPLS-AA host):

```python
from AutoPoly.nanoparticles import Grafting, Nanoparticle

brush = Grafting(polymer=Polymer(chain_num=1, sequence=["CC[*]"] + ["[*]CC[*]"] * 4 + ["[*]CC"]),
                 density=0.5,          # chains/nm^2 (or count=N)
                 seed=2)
np_ = Nanoparticle(material="SiO2", radius=9.0, grafts=brush, name="gsilica")
```

- **Sites**: silanol oxygens spread by farthest-point sampling (at most
  one per Si), screened for room to grow; crowded sites are swapped.
- **Chemistry** (`linker="silyl_ether"`): Si–OH + H–C → Si–O–C. The
  grafted O keeps the silanol-O LJ parameters and carries the silanol
  group charge plus the charge of the removed organic H, so the total
  charge is conserved exactly.
- **Conformations**: each graft grows from its anchor around the core
  and earlier grafts; the grafts belong to the particle class
  (`GNP_<name>`), not to the free matrix.
- **Bonded linkage terms** (O–C bond, Si–O–C and O–C–X angles, O–C–C–X
  dihedrals) are read from the bundled `oplsaa.lt` by OPLS bonded class,
  never invented; Si–O–C–X torsions are not defined for silica-bound
  ethers in OPLS-AA and are omitted.
- In MD the **core** is the rigid body; the grafted chains are flexible.

## Running the system

`in.run` is a starting point, not an equilibration protocol:

```lammps
include         system.in.init
read_data       system.data
include         system.in.settings
include         system.in.groups        # per instance / species / motion, fillers, matrix
neigh_modify    exclude molecule/intra fillers
fix             hold fillers setforce 0.0 0.0 0.0   # relax the matrix around held fillers
minimize        1.0e-4 1.0e-6 2000 20000
unfix           hold
fix             rigid rigid_fillers rigid/nvt molecule temp 300.0 300.0 100.0
fix             nvt matrix nvt temp 300.0 300.0 100.0
compute         tmatrix matrix temp     # matrix-only temperature in the log
...
```

Frozen particles use `fix setforce` instead of the rigid fix. Uncharged
systems get an explicit `kspace_modify gewald`. `rigid/nvt` is used
rather than `rigid/nvt/small`, which aborts together with
`neigh_modify exclude` in LAMMPS 29Aug2024.

A quick health check of any generated project:

```python
from AutoPoly.pipeline.smoke import run_smoke
report = run_smoke("ps_silica/ps_silica", steps=500, expected_rigid_bodies=2)
assert report.ok, report.errors
```

## Stage by stage

Particles flow through the same three stages as polymers:

- **Stage 1** (`GeometryBuilder`): `geometry.json` (format v2) gains an
  `inorganic` section with each particle's `AtomicStructure`.
- **Stage 2** (`UnitTyper`): the adapter types the structure into
  `NP_<id>.lt` (plus `GNP_<id>.lt` for grafted particles); `units.json`
  (format v2) records kind `inorganic`, role `filler`, motion, material
  provenance (phase, surface, adapter version, qualification, net
  charge) and the particle envelope. One geometry can be typed with
  several inorganic force fields.
- **Stage 3** (`BoxPacker(..., packing=NanocompositePacking())`):
  placement, moltemplate, cross terms, groups, `in.run`, report.

Version-1 `geometry.json` / `units.json` files from earlier releases are
still read.

## Coarse-grained fillers (bead-spring)

For large scales, the bead-spring backend supports **CG fillers**:
rigid (or frozen) spheres of LJ beads in reduced units.

```python
import numpy as np
from AutoPoly.models import architectures as arch
from AutoPoly.models.bead_spring import BeadType
from AutoPoly.models.bead_spring_system import BeadSpringSystem
from AutoPoly.nanoparticles import CGFiller

np.random.seed(42)
bss = BeadSpringSystem("kg_filled", system,
                       bead_types=[BeadType("A"), BeadType("F")],
                       bond_style="fene", pair_style="wca", density=0.3)
bss.add_species(arch.linear([("A", 30)]), n_chains=60)
bss.add_filler(CGFiller(radius=4.0, count=2, bead_type="F",
                        structure="fcc",       # or "shell"
                        spacing=1.0, motion="rigid"))
bss.set_pair_coeff("A", "F", epsilon=1.5, cutoff=2.5)   # attractive pair in a WCA melt
bss.generate(backend="direct")                        # or "moltemplate"
```

- Fillers are placed first (envelope gaps ≥ `bss.filler_min_gap`, 1σ);
  chains then grow by SAW around them. With fillers every overlap check
  — fillers and chain beads — uses periodic minimum images, so the
  start has no clashes across the box boundary.
- Each filler is one molecule of its own bead type (which no chain may
  use); `density` is the chain-bead density outside the fillers.
- `in.polymer` holds the fillers during minimization, integrates rigid
  fillers with `fix rigid/nvt molecule` (frozen ones are kept at rest),
  skips intra-filler pairs and logs a matrix-only temperature
  (`c_tmatrix`). A commented NPT block compresses toward melt density:
  the periodic SAW cannot pack at 0.85 σ⁻³ directly.
- Only `generation_method="saw"` is supported with fillers.
- CG fillers are generic LJ particles: mapping them onto a real
  material is your modelling choice, outside the atomistic qualification
  matrix.

## Limitations

- TiO2 is geometry-only (no validated TiO2–polymer model is bundled).
- Carbon is qualified with OPLS-AA hosts only; grafting only for silica
  with OPLS-AA (`silyl_ether`).
- Bundled inorganic models are nonbonded: no flexible particles.
- Charged particles / counter-ions, nanoparticles together with a
  substrate slab, and LOPLS / Class II hosts are not supported.
- Long, bulky chains (e.g. PMMA) are slow to regrow in crowded boxes;
  build dilute (`initial_density` 0.25–0.35) and compress in MD.

## See also

- Examples: `example_silica_nanoparticle.py`,
  `example_pmma_silica_nanocomposite.py`,
  `example_silica_grafted_polymer.py`, `example_alumina_pmma.py`,
  `example_titania_polymer.py`, `example_graphene_polymer.py`,
  `example_au_polymer.py`, `example_cg_nanocomposite.py`
- [Substrates & Films](substrates.md) — planar slabs (shares the silica
  chemistry and force-field tables)
- [Nanoparticles API](../reference/nanoparticles.md)
