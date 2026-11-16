#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Material registry: the single source of truth for inorganic materials.

Every inorganic component (built-in substrate slab or nanoparticle)
resolves its material through this registry.  A MaterialSpec records
identity and provenance -- phases/structure sources, shape builders,
surface-termination models, the inorganic force-field adapters defined
for it -- and *where each builder lives* (dotted "module:function"
references resolved lazily), so the pipeline never branches on material
names.

Registration is not a promise of simulation readiness.  Capability is
tracked per stage:

- ``registered``:     metadata/selection exists,
- ``geometry_ready``: a built-in (or import) path produces structures that
                      pass topology checks and can be exported,
- ``surface_ready``:  a surface-termination model with chemical
                      validation (valence, stoichiometry, charge) exists,
- ``md_ready``:       at least one exact combination (material + phase +
                      surface + inorganic FF + polymer FF + motion) is
                      listed in the qualification matrix
                      (AutoPoly.materials.qualification) with passing
                      smoke-test evidence.

Five material families are registered as six IDs: ``sio2``, ``al2o3``,
``tio2``, ``c``, ``au``, ``ag``.

Created on 2026-11-13
@author: zwu
"""
import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..core.exceptions import ValidationError

CAPABILITY_STAGES = ("registered", "geometry_ready", "surface_ready",
                     "md_ready")


@dataclass(frozen=True)
class PhaseSpec:
    """
    One phase / structure source of a material.

    Attributes:
        name: Stable phase ID (e.g. "alpha_quartz").
        description: Human-readable description.
        space_group: Space group (crystalline phases) or "" (amorphous).
        crystal: Key into AutoPoly.materials.crystals.CRYSTALS (bulk cell
                 used by the built-in shape builders), or None when the
                 phase is only available as an imported structure.
        shapes: Shapes with a built-in builder for this phase.
        aliases: Alternative names.
        citation: Structure reference.
    """
    name: str
    description: str
    space_group: str = ""
    crystal: Optional[str] = None
    shapes: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    citation: str = ""


@dataclass(frozen=True)
class SurfaceStateSpec:
    """
    One surface-termination model.

    Attributes:
        name: Stable ID ("hydroxylated", "stoichiometric", "h_passivated",
              "bare", "periodic").
        description: What the termination does.
        site_types: Surface-site labels it produces.
        validated: True when chemical validation (valence, stoichiometry,
                   charge) is implemented and tested.
    """
    name: str
    description: str
    site_types: Tuple[str, ...] = ()
    validated: bool = True


@dataclass
class MaterialSpec:
    """
    Registry entry of one inorganic material (see module docstring).

    Attributes:
        id: Stable material ID ("sio2", ...).
        name: Display name.
        formula: Chemical formula.
        family: "oxide", "carbon", or "metal".
        elements: Elements a structure of this material may contain
                  (including termination atoms).
        aliases: Accepted alternative names (case-insensitive).
        phases: phase ID -> PhaseSpec.
        default_phase: Phase used when none is given.
        surface_states: termination ID -> SurfaceStateSpec.
        default_surface: Termination used when none is given.
        chemistry: Module implementing the chemistry adapter
                   (``assign_roles``/``validate``/``bond_cutoffs``).
        builders: shape -> "module:function" geometry builder.
        force_fields: Inorganic FF adapter names defined for the material.
        default_force_field: Adapter used when none is requested.
        radius_range: Allowed built-in particle radius range (A).
        capabilities: Declared stage -> bool for registered /
                      geometry_ready / surface_ready (cross-checked by
                      tests); md_ready is derived from the qualification
                      matrix.
        citations / notes: Provenance.
    """
    id: str
    name: str
    formula: str
    family: str
    elements: Tuple[str, ...]
    aliases: Tuple[str, ...] = ()
    phases: Dict[str, PhaseSpec] = field(default_factory=dict)
    default_phase: Optional[str] = None
    surface_states: Dict[str, SurfaceStateSpec] = field(default_factory=dict)
    default_surface: Optional[str] = None
    chemistry: Optional[str] = None
    builders: Dict[str, str] = field(default_factory=dict)
    force_fields: Tuple[str, ...] = ()
    default_force_field: Optional[str] = None
    radius_range: Tuple[float, float] = (5.0, 60.0)
    capabilities: Dict[str, bool] = field(default_factory=dict)
    citations: Tuple[str, ...] = ()
    notes: str = ""

    # ------------------------------------------------------------------
    def resolve_phase(self, phase: Optional[str]) -> PhaseSpec:
        """Phase by ID or alias (None = default phase)."""
        if phase is None:
            if self.default_phase is None:
                raise ValidationError(
                    f"Material '{self.id}' has no default phase; choose "
                    f"one of {sorted(self.phases)}"
                )
            return self.phases[self.default_phase]
        key = phase.lower()
        for p in self.phases.values():
            if key == p.name or key in (a.lower() for a in p.aliases):
                return p
        raise ValidationError(
            f"Unknown phase/structure '{phase}' for material '{self.id}'; "
            f"available: {sorted(self.phases)}"
        )

    def resolve_surface(self, surface: Optional[str]) -> SurfaceStateSpec:
        """Surface termination by ID (None = default)."""
        name = surface or self.default_surface
        if name is None or name not in self.surface_states:
            raise ValidationError(
                f"Unknown surface state '{surface}' for material "
                f"'{self.id}'; available: {sorted(self.surface_states)}"
            )
        return self.surface_states[name]

    def builder(self, shape: str) -> Callable:
        """Geometry builder callable for a shape."""
        if shape not in self.builders:
            raise ValidationError(
                f"Material '{self.id}' has no built-in '{shape}' builder; "
                f"available shapes: {sorted(self.builders)} (or import a "
                "structure with Nanoparticle.from_file)"
            )
        return resolve_callable(self.builders[shape])

    def chemistry_module(self):
        """The chemistry adapter module (None if not implemented)."""
        if self.chemistry is None:
            return None
        return importlib.import_module(self.chemistry)

    def capability(self, stage: str) -> bool:
        """
        Capability of one stage.  ``md_ready`` is derived from the
        qualification matrix (never declared); the other stages are
        declared here and cross-checked by the test suite.
        """
        if stage not in CAPABILITY_STAGES:
            raise ValidationError(f"Unknown capability stage '{stage}'")
        if stage == "md_ready":
            from .qualification import material_md_ready
            return material_md_ready(self.id)
        return bool(self.capabilities.get(stage, stage == "registered"))

    def capability_report(self) -> Dict[str, bool]:
        return {stage: self.capability(stage) for stage in CAPABILITY_STAGES}


def resolve_callable(ref: str) -> Callable:
    """Import a "package.module:function" reference."""
    module, _, name = ref.partition(":")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as e:
        raise ValidationError(f"Cannot resolve builder '{ref}': {e}") from e


MATERIAL_REGISTRY: Dict[str, MaterialSpec] = {}


def register_material(spec: MaterialSpec) -> None:
    """Register a material (IDs and aliases must be unique)."""
    if spec.id in MATERIAL_REGISTRY:
        raise ValidationError(f"Material '{spec.id}' is already registered")
    taken = {a.lower() for m in MATERIAL_REGISTRY.values()
             for a in (m.id, m.formula, *m.aliases)}
    for alias in (spec.id, spec.formula, *spec.aliases):
        if alias.lower() in taken:
            raise ValidationError(
                f"Material alias '{alias}' of '{spec.id}' is already used"
            )
    MATERIAL_REGISTRY[spec.id] = spec


def get_material(name: str) -> MaterialSpec:
    """
    Resolve a material by ID, formula or alias (case-insensitive):
    ``get_material("SiO2") is get_material("silica")``.
    """
    key = str(name).strip().lower()
    for spec in MATERIAL_REGISTRY.values():
        if key in {spec.id.lower(), spec.formula.lower(),
                   *(a.lower() for a in spec.aliases)}:
            return spec
    raise ValidationError(
        f"Unknown material '{name}'. Registered materials: "
        f"{sorted(MATERIAL_REGISTRY)}"
    )


def registered_materials() -> List[str]:
    return list(MATERIAL_REGISTRY)


# ----------------------------------------------------------------------
# Built-in materials: five families, six IDs
# ----------------------------------------------------------------------
_NP = "AutoPoly.nanoparticles.geometry"

register_material(MaterialSpec(
    id="sio2",
    name="Silica",
    formula="SiO2",
    family="oxide",
    elements=("Si", "O", "H"),
    aliases=("silica", "silicon dioxide"),
    phases={
        "alpha_quartz": PhaseSpec(
            "alpha_quartz", "alpha-quartz (P3121), orthohexagonal cell",
            space_group="P3121", crystal="alpha_quartz",
            shapes=("slab", "sphere"), aliases=("quartz",),
            citation="Tucker, Dove & Keen, Mineral. Mag. 2001, 65, 489 "
                     "(COD 1526860)"),
        "beta_cristobalite": PhaseSpec(
            "beta_cristobalite",
            "beta-cristobalite (Fd-3m average structure), (111) cell",
            space_group="Fd-3m", crystal="beta_cristobalite",
            shapes=("slab", "sphere"), aliases=("cristobalite",),
            citation="COD 1010944"),
        "amorphous": PhaseSpec(
            "amorphous",
            "amorphous silica (imported relaxed structures only)",
            crystal=None, shapes=(), aliases=("a-sio2", "glass")),
    },
    default_phase="beta_cristobalite",
    surface_states={
        "hydroxylated": SurfaceStateSpec(
            "hydroxylated",
            "Si kept 4-coordinated; dangling O capped with H (Q3/Q2 "
            "silanols); optional Si-O-Si condensation toward a target "
            "silanol density",
            site_types=("silanol",)),
    },
    default_surface="hydroxylated",
    chemistry="AutoPoly.materials.chemistry.silica",
    builders={"sphere": f"{_NP}:build_silica_sphere"},
    force_fields=("interface_silica", "clayff_silica"),
    default_force_field="interface_silica",
    radius_range=(6.0, 60.0),
    capabilities={"registered": True, "geometry_ready": True,
                  "surface_ready": True},
    notes="P0-P2 reference material: slabs (SubstrateSpec builders), "
          "crystal-derived spheres and imported clusters.",
))

register_material(MaterialSpec(
    id="al2o3",
    name="Alumina",
    formula="Al2O3",
    family="oxide",
    elements=("Al", "O", "H"),
    aliases=("alumina", "corundum", "aluminium oxide", "aluminum oxide"),
    phases={
        "alpha_corundum": PhaseSpec(
            "alpha_corundum", "alpha-Al2O3 corundum (R-3c), hexagonal cell",
            space_group="R-3c", crystal="alpha_corundum",
            shapes=("sphere",), aliases=("corundum", "alpha")),
    },
    default_phase="alpha_corundum",
    surface_states={
        "hydroxylated": SurfaceStateSpec(
            "hydroxylated",
            "Al kept 6-coordinated; under-coordinated O protonated in "
            "bond-valence deficit order until the cluster is neutral "
            "under Al(+3)/O(-2)/H(+1) (CLAYFF-consistent: ao/ob/oh/ho)",
            site_types=("mu1_oh", "mu2_oh", "mu3_oh")),
        "stoichiometric": SurfaceStateSpec(
            "stoichiometric",
            "dry Al2O3 stoichiometric cut (least-coordinated O removed); "
            "under-coordinated surface Al",
            site_types=("al_cus",)),
    },
    default_surface="hydroxylated",
    chemistry="AutoPoly.materials.chemistry.oxides",
    builders={"sphere": f"{_NP}:build_oxide_sphere"},
    force_fields=("clayff_alumina",),
    default_force_field="clayff_alumina",
    radius_range=(6.0, 50.0),
    capabilities={"registered": True, "geometry_ready": True,
                  "surface_ready": True},
))

register_material(MaterialSpec(
    id="tio2",
    name="Titania",
    formula="TiO2",
    family="oxide",
    elements=("Ti", "O", "H"),
    aliases=("titania", "titanium dioxide"),
    phases={
        "rutile": PhaseSpec(
            "rutile", "rutile TiO2 (P4_2/mnm)", space_group="P4_2/mnm",
            crystal="rutile", shapes=("sphere",)),
        "anatase": PhaseSpec(
            "anatase", "anatase TiO2 (I4_1/amd)", space_group="I4_1/amd",
            crystal="anatase", shapes=("sphere",)),
    },
    default_phase="anatase",
    surface_states={
        "hydroxylated": SurfaceStateSpec(
            "hydroxylated",
            "Ti kept 6-coordinated; under-coordinated O protonated "
            "(up to 2 H) in bond-valence deficit order to formal-charge "
            "neutrality (Ti+4/O-2/H+1); facet-agnostic",
            site_types=("ti_oh", "ti_oh2", "bridging_oh")),
    },
    default_surface="hydroxylated",
    chemistry="AutoPoly.materials.chemistry.oxides",
    builders={"sphere": f"{_NP}:build_oxide_sphere"},
    force_fields=(),
    default_force_field=None,
    radius_range=(6.0, 50.0),
    capabilities={"registered": True, "geometry_ready": True,
                  "surface_ready": True},
    notes="No validated TiO2-polymer interaction model is bundled: MD "
          "export is refused. Facet-specific terminations and a "
          "documented TiO2 FF (e.g. Matsui-Akaogi + validated organic "
          "cross terms) are required before md_ready.",
))

register_material(MaterialSpec(
    id="c",
    name="Carbon",
    formula="C",
    family="carbon",
    elements=("C", "H"),
    aliases=("carbon", "graphene", "graphite", "cnt", "carbon nanotube"),
    phases={
        "graphene": PhaseSpec(
            "graphene", "single graphene sheet (a = 2.46 A)",
            crystal="graphene", shapes=("sheet",)),
        "cnt": PhaseSpec(
            "cnt", "single-wall carbon nanotube (n, m)",
            crystal="graphene", shapes=("tube",),
            aliases=("nanotube", "swcnt")),
        "carbon_black": PhaseSpec(
            "carbon_black",
            "carbon black (imported atomistic/CG structures only)",
            crystal=None, shapes=()),
    },
    default_phase="graphene",
    surface_states={
        "h_passivated": SurfaceStateSpec(
            "h_passivated",
            "sp2 network; edge C with 2 C neighbors capped with H, "
            "dangling C (1 neighbor) removed", site_types=("edge_ch",)),
    },
    default_surface="h_passivated",
    chemistry="AutoPoly.materials.chemistry.carbon",
    builders={"sheet": f"{_NP}:build_graphene_sheet",
              "tube": f"{_NP}:build_nanotube"},
    force_fields=("oplsaa_aromatic_carbon",),
    default_force_field="oplsaa_aromatic_carbon",
    radius_range=(4.0, 80.0),
    capabilities={"registered": True, "geometry_ready": True,
                  "surface_ready": True},
))

for _id, _name, _a in (("au", "Gold", 4.0782), ("ag", "Silver", 4.0853)):
    register_material(MaterialSpec(
        id=_id,
        name=_name,
        formula=_id.capitalize(),
        family="metal",
        elements=(_id.capitalize(),),
        aliases=(_name.lower(),),
        phases={
            "fcc": PhaseSpec(
                "fcc", f"fcc {_name.lower()} (Fm-3m, a = {_a} A)",
                space_group="Fm-3m", crystal=f"fcc_{_id}",
                shapes=("sphere", "cuboctahedron", "octahedron", "wulff")),
        },
        default_phase="fcc",
        surface_states={
            "bare": SurfaceStateSpec(
                "bare", "clean metallic surface (no ligands)",
                site_types=("metal_surface",)),
        },
        default_surface="bare",
        chemistry="AutoPoly.materials.chemistry.metals",
        builders={"sphere": f"{_NP}:build_metal_cluster",
                  "cuboctahedron": f"{_NP}:build_metal_cluster",
                  "octahedron": f"{_NP}:build_metal_cluster",
                  "wulff": f"{_NP}:build_metal_cluster"},
        force_fields=(f"interface_metal_{_id}",),
        default_force_field=f"interface_metal_{_id}",
        radius_range=(4.0, 60.0),
        capabilities={"registered": True, "geometry_ready": True,
                      "surface_ready": True},
        notes="Effective 12-6 LJ interfacial model (Heinz et al. 2008); "
              "many-body EAM potentials are NOT mixed with organic LJ.",
    ))
