# Substrates & Films

AutoPoly can build a polymer or small-molecule **film on top of a physical
substrate slab** — the standard fully periodic, two-interface slab model used
for surface, coating, and interfacial simulations:

```
zhi ┌───────────────────────────┐
    │    vacuum (optional)      │  spec.vacuum, 0 by default
    ├── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
    │    film region            │  your models, MC-placed with
    │                           │  collision detection
    ├── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤  film_zmin = slab_top + gap
    │    gap                    │
    ├───────────────────────────┤  slab_top = zlo + thickness
    │    substrate slab         │  model-built (grid or mc) or
    │                           │  one external .lt instance
zlo └───────────────────────────┘
```

Passing `substrate=` to `generate()` auto-selects the `on_substrate` packing
strategy — no other changes to your workflow.

## Quick start

```python
from AutoPoly import System, Polymer, Molecule, generate
from AutoPoly.packing import SubstrateSpec

film = Polymer(chain_num=10, sequence=["CC[*]"] + ["[*]CC[*]"] * 18 + ["[*]CC"])

substrate = SubstrateSpec(
    model=Molecule(Count=64, Smiles="CCO", Name="etoh_sub"),
    thickness=10.0,   # slab z-extent (Å), must be > 0
    packing="grid",   # "grid" = ordered slab, "mc" = amorphous
    gap=3.0,          # empty space between slab top and film (Å)
)

generate(
    system, "pe_film", [film],
    force_field="gaff",
    substrate=substrate,
    box_dims=(50.0, 50.0, 50.0),
)
```

## Choosing the substrate source

Three sources are available: an in-pipeline `model`, a built-in
crystalline `builder` (alpha-quartz silica), or an external pre-built
`.lt` surface.

### In-pipeline model: `SubstrateSpec(model=...)`

Any `Molecule` or `Polymer` works. It flows through the same three pipeline
stages as the film (same force field), and its units are tagged
`role="substrate"` in `units.json`. The slab is packed two ways:

| `packing` | Result |
|---|---|
| `"grid"` (default) | Ordered layers — instances on a 3D grid spanning the slab. Spacing is `1.8 ×` the unit's bounding radius; the run errors if your count exceeds grid capacity (grow `box_dims`/`thickness` or lower the count). |
| `"mc"` | Amorphous — MC placement with collision detection inside the slab region. Errors if the slab is too dense to place (grow the slab or use `"grid"`). |

The instance count comes from the model (`Count` / `chain_num`), an explicit
`count=N` override, or `count="auto"`:

```python
SubstrateSpec(
    model=Molecule(Count=1, Smiles="CCO", Name="etoh_sub"),  # Count ignored
    count="auto", density=0.01,   # particles/Å³
    thickness=10.0,
)
# -> count = density × lx × ly × thickness  (requires explicit box_dims lateral)
```

!!! note
    `density` here is *particles* per Å³ (whole molecules), not the
    `monomer_density` used for film sizing. For a small-molecule liquid,
    ~0.01/Å³ is a typical magnitude — 0.085 would be atomic density.

### Built-in crystalline silica: `SubstrateSpec(builder=...)`

Two hydroxylated crystalline SiO2 slabs can be generated at pack time —
no external files needed:

| `builder` | Surface | Termination | Silanol density | Surface cell (A) |
|---|---|---|---|---|
| `"alpha_quartz"` | alpha-quartz (0001) | geminal Q2 | ~9.6/nm² (intrinsic) | 4.9019 × 8.4903 |
| `"beta_cristobalite"` | beta-cristobalite (111) | isolated Q3 | ~4.5/nm² (matches Zhuravlev) | 10.1258 × 17.5383 |

For a target silanol density near the experimental value for amorphous
silica (4.6/nm², Zhuravlev 2000), use `"beta_cristobalite"`. Example
with alpha-quartz:

```python
substrate = SubstrateSpec(
    builder="alpha_quartz",
    thickness=12.0,       # slab envelope (A), incl. hydroxyl coatings
    gap=3.0,
    slab_ff="interface",  # "interface" (INTERFACE FF v1.5) or "clayff"
)

generate(
    system, "pe_on_quartz", [film],
    force_field="gaff",
    substrate=substrate,
    box_dims=(30.0, 34.0, 40.0),   # lateral dims are required
)
```

The slab is built from published crystal structures (alpha-quartz:
P3121, COD 1526860; beta-cristobalite: Fd-3m average structure, COD
1010944 with oxygens displaced off the Si-Si axis to physical 1.61 A
bonds and 148.7° Si-O-Si angles), cleaved along the natural cleavage
plane (chosen automatically as the cut that breaks the fewest bonds
while keeping every Si tetrahedrally coordinated), and hydroxylated on
**both** faces. `oh_density` accepts a target density and the builder
attempts to reach it by forming Si-O-Si bridges between dangling
oxygens (mbuild-style); note that on both supported faces the surface
silicons are too far apart for bridging, so the surface stays at its
intrinsic density — a warning is logged when the target cannot be met.
Set `hydroxylate_bottom=False` to leave the bottom face bare.

Practical consequences of the crystalline slab:

- **Lateral `box_dims` are required** (lz may still be `None`), and the
  box is *snapped* to integer surface cells (see the table above) so the
  slab is seamlessly periodic. The snap is logged.
- `thickness` is an **envelope**: the Si core occupies roughly
  `thickness - 5.4 A`; the physical slab (with –OH caps) always fits
  inside `[zlo, zlo + thickness]`.
- The slab is a single self-typed `.lt` class (not SMILES-typed through
  the pipeline) with LJ + charges plus a **zero-force-constant bond
  topology** (`bond_coeff ... 0.0`): the bonds add no forces, but they
  make `special_bonds` exclude intra-slab 1-2/1-3/1-4 nonbonded terms —
  without them, bonded O-H pairs (0.945 A) would contribute enormous
  spurious LJ/Coulomb energies. The slab is meant to be held
  rigid/frozen in MD (`fix rigid` or freeze the slab atoms, e.g.
  `fix freeze slab setforce 0.0 0.0 0.0`). The `pair_coeff`/`bond_coeff`
  lines automatically carry the sub-style required by the film force
  field (hybrid vs plain styles).
- Slab charges sum to zero; mixing with the film force field follows
  LAMMPS mixing rules (`pair_modify mix ...`).

The film keeps its own force field; the slab's is chosen independently:

| `slab_ff` | Types (roles Si/OB/OH/HO) | Charges (Si, OB, OH, HO) | Reference |
|---|---|---|---|
| `"interface"` (default) | `i15_sc4`, `i15_oc23`, `i15_oc24`, `i15_hoy` | +1.10, −0.55, −0.675, +0.40 | Emami et al., *Chem. Mater.* 2014 (INTERFACE FF v1.5) |
| `"clayff"` | `cff_st`, `cff_ob`, `cff_oh`, `cff_ho` | +2.10, −1.05, −0.95, +0.425 | Cygan et al., *J. Phys. Chem. B* 2004 (CLAYFF) |
| `"custom"` | user-provided | user-provided | — |

!!! note "CLAYFF sigma corrected"
    Cygan et al. tabulate the well-minimum distance R0; the LJ sigma is
    R0 / 2^(1/6) (st 3.3020 A, ob/oh 3.1655 A). Earlier releases applied
    this conversion twice, so CLAYFF slab sigmas were 2^(1/6) too small.
    Slabs built with `slab_ff="clayff"` now carry the published values;
    coordinates, topology, charges and the INTERFACE FF tables are
    unchanged (see the CHANGELOG).

With `"custom"` (or to override individual entries of a built-in table),
pass per-role maps covering the roles `"Si"` (tetrahedral Si), `"OB"`
(bridging O), `"OH"` (silanol O), `"HO"` (silanol H):

```python
SubstrateSpec(
    builder="alpha_quartz", thickness=12.0,
    slab_ff="custom",
    slab_types={"Si": "si_q", "OB": "ob_q", "OH": "oh_q", "HO": "ho_q"},
    slab_charges={"Si": 1.5, "OB": -0.75, "OH": -0.85, "HO": 0.35},
    slab_lj={"Si": (0.093, 3.697), "OB": (0.054, 3.091),
             "OH": (0.122, 3.091), "HO": (0.015, 0.967)},  # (eps kcal/mol, sigma A)
)
```

!!! warning "Mixing rules"
    INTERFACE FF and CLAYFF both assume arithmetic-sigma/geometric-epsilon
    mixing (`pair_modify mix arithmetic`), which also matches GAFF/AMBER
    conventions. OPLS films expect geometric mixing. Check your
    `system.in.settings` and set the mixing rule appropriate for your
    combination. (Nanoparticle systems do not depend on this: they get
    explicit polymer–particle `pair_coeff` lines, see
    [Nanocomposites](nanocomposites.md#explicit-cross-interactions).)

Internally the builder slabs share the materials foundation with the
nanoparticles (`AutoPoly.materials`: crystal tables, silica chemistry,
force-field adapters and the moltemplate writer). The slab is recorded
as a stage-1 inorganic entry in `geometry.json` and typed in stage 2
like any other unit; the `SubstrateSpec` calls and the generated
geometry, topology and charges are unchanged.

### External pre-built surface: `SubstrateSpec(lt_file=..., class_name=...)`

For crystalline surfaces (Au(111), SiO2, ...) built outside AutoPoly:

```python
SubstrateSpec(
    lt_file="surfaces/au111.lt", class_name="Au111",
    thickness=12.0, gap=3.0,
)
```

The file is copied into the moltemplate directory, imported in `system.lt`,
and instantiated **once**, centered laterally at the slab mid-plane. The
slab's atoms are not collision-checked individually — instead the film is
kept out geometrically (every film bounding-sphere center stays above
`slab_top + gap`). Make sure your external slab's coordinates are centered
on its own origin so the `.move()` placement lands it correctly, and that it
imports whatever force-field file it needs (it is *not* part of AutoPoly's
force-field subsetting).

## Box geometry: `box_dims`

Substrate systems use rectangular boxes. `box_dims=(lx, ly, lz)` takes
precedence per axis over the cubic `box_size`; any element may be `None`:

- **Lateral (`lx`, `ly`)** — auto: the standard melt box estimate.
- **`lz`** — auto: `thickness + gap + film_thickness + vacuum`, where
  `film_thickness = film_particles / (monomer_density × lx × ly)`.

Validation catches impossible stacks early: `lz` smaller than
`thickness + gap`, or a film region thinner than the largest film chain's
bounding-sphere diameter, both raise `ValidationError` with the numbers.

With the default `vacuum=0` the film fills the rest of the box, giving the
usual fully periodic slab model (film touches the slab's periodic image —
that's the second interface). Set `vacuum > 0` if you plan to run with
`boundary p p f`.

## Running the system in LAMMPS

When `substrate=` is used, `generate()` also writes a ready-to-run
**`in.run`** into the project directory: frozen slab
(`fix ... setforce 0`), film NVT at 300 K with a film-only temperature
compute (`compute tfilm film temp` — the default whole-system
temperature would be biased low by the frozen slab). Review and edit it
for production runs.

Note: AutoPoly also patches `system.in.init` after moltemplate — hybrid
bond/angle/dihedral/improper styles with zero records in `system.data`
(e.g. `improper_style hybrid cvff` for an improper-free polyethylene
film) are replaced with their `none` form, since LAMMPS aborts on
unused hybrid sub-styles.

## Subtract: carving after placement

`subtract=` removes **whole instances** (chains/molecules) whose centers fall
inside a carve region. Because moltemplate instantiates whole `.lt` classes,
removal never cuts a covalent bond — the carved boundary is fuzzy at the
scale of one bounding radius (tighten it with `conservative=True`, which
also removes instances merely *touching* the region):

```python
from AutoPoly.packing import CutAbove, CutBelow, Cylinder, BoxRegion

generate(
    system, "pe_film_patterned", [film],
    substrate=substrate,
    box_dims=(50.0, 50.0, 50.0),
    subtract=[
        Cylinder(axis="z", center=(0, 0), radius=8.0),       # hole
        BoxRegion(x=(-25, -5), y=None, z=None),              # trench
        CutAbove(z=25.0),                                    # thickness trim
    ],
)
```

| Region | Carves |
|---|---|
| `CutAbove(z)` / `CutBelow(z)` | everything above/below a plane |
| `Cylinder(axis, center, radius)` | infinite cylinder (`axis` = "x"/"y"/"z") |
| `BoxRegion(x=, y=, z=)` | axis-aligned box; `None` = full span on that axis |

Every region takes `apply_to` — `"film"` (default), `"substrate"`, or
`"all"` — so a film pattern never touches the slab unless you ask. Regions
apply in order; every removal is logged.

Subtract is strategy-agnostic: it also works with plain `mc_random` melts
(`generate(..., subtract=[Cylinder(...)])` → nanopore). The `grid` strategy
rejects it.

## What subtract does *not* do

Whole-instance granularity means no dangling bonds, no charge drift, and a
topology that always matches the `.lt` definitions — at the price of a
carved surface that is smooth only at the chain-radius scale. If your model
needs atomically sharp interfaces (severed bonds, capped fragments), that is
a different, post-processing operation on `system.data`; it is not part of
this feature.

## See also

- [example_film_on_substrate.py](https://github.com/WuGroup-XJTLU/AutoPoly/blob/v2.0/examples/example_film_on_substrate.py) — runnable film + patterned variant
- [MC Placement & Chain Growth](mc-placement.md) — the placement engine the film uses
- [generate API](../reference/generate.md) — `substrate`, `subtract`, `box_dims` parameters
- [Nanocomposites](nanocomposites.md) — finite particles (silica, alumina, carbon, Au/Ag) in a matrix
