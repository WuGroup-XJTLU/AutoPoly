#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base classes for AutoPoly box-packing strategies.

A placement strategy takes a PackingContext (typed units + box spec) and
returns a PlacementResult: one lt instantiation command per placed instance
plus the box bounds actually used. Strategies read unit metadata only —
they never parse .lt files.

Created on 2026-07-30
@author: zwu
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Tuple, Union

from ..core.exceptions import ValidationError
from ..pipeline.units import UnitLibrary

if TYPE_CHECKING:
    from .regions import CarveRegion

# ((xmin, xmax), (ymin, ymax), (zmin, zmax))
BoxBounds = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]


def symmetric_bounds(box_size: float) -> BoxBounds:
    """Build box bounds centered on the origin from a cubic box size."""
    half = box_size / 2.0
    return ((-half, half), (-half, half), (-half, half))


def bounds_size(bounds: BoxBounds) -> float:
    """Side length of cubic bounds."""
    return bounds[0][1] - bounds[0][0]


@dataclass
class BoxSpec:
    """
    Box specification handed to a strategy.

    Attributes:
        requested_box_size: Explicit cubic box side in Angstrom, or None
                            to let the strategy auto-size the box.
        box_dims: Optional per-axis box sides (lx, ly, lz) in Angstrom.
                  Any element may be None (strategy auto-sizes that axis).
                  Takes precedence over requested_box_size per axis.
    """
    requested_box_size: Optional[float] = None
    box_dims: Tuple[Optional[float], Optional[float], Optional[float]] = (
        None, None, None,
    )


#: Valid slab packing modes for a model-built substrate.
SUBSTRATE_PACKING_MODES = ("grid", "mc")

#: Sentinel for deriving the substrate instance count from slab volume.
AUTO_COUNT = "auto"


@dataclass
class SubstrateSpec:
    """
    Physical substrate specification for the "on_substrate" strategy.

    The substrate is a slab of lateral size (lx, ly) sitting at the bottom
    of the box (z in [zlo, zlo + thickness]); film units are placed above
    it with a `gap` of empty space in between.

    Exactly one source must be given:

    - model: a Molecule or Polymer model, typed in-pipeline alongside the
             film models. Its units carry role="substrate" and are packed
             into the slab region ("grid" = ordered layers, "mc" =
             amorphous MC placement).
    - lt_file + class_name: a pre-built surface class (e.g. an Au(111) or
             SiO2 slab built outside AutoPoly). The file is copied into the
             moltemplate directory, imported in system.lt, and instantiated
             once, centered laterally at zlo + thickness/2.
    - builder: a built-in surface builder ("alpha_quartz" or
             "beta_cristobalite"). A hydroxylated crystalline SiO2 slab
             is generated at pack time (AutoPoly.surfaces builders),
             written as a self-contained .lt, and instantiated once like
             an external slab. Requires explicit lateral box_dims; the
             box sides are snapped to integer surface cells.

    Attributes:
        model: Molecule/Polymer model for an in-pipeline substrate.
        lt_file: Path to an external pre-built substrate .lt file.
        class_name: Moltemplate class name defined by lt_file.
        thickness: Slab z-extent in Angstrom (must be > 0).
        packing: Slab packing mode for model substrates: "grid" or "mc".
        gap: Empty space in Angstrom between the slab top and the film.
        count: None (use the model's own Count/chain_num), a positive int
               override, or AUTO_COUNT to derive the instance count from
               slab volume x density (Molecule models only; requires
               explicit lateral box_dims at generate() time).
        density: Particle density (particles/A^3) used with AUTO_COUNT.
        vacuum: Extra empty space in Angstrom above the film (0 = fully
                periodic slab model with the film filling the rest of z).
        builder: Built-in slab builder name ("alpha_quartz") or None.
        oh_density: Target top-surface silanol density (sites/nm^2) for
                builder slabs (default 4.6, after Zhuravlev).
        slab_ff: Slab force field: "interface" (INTERFACE FF v1.5, Emami
                et al. 2014), "clayff" (Cygan et al. 2004), or "custom"
                (requires slab_types + slab_charges + slab_lj covering
                the roles "Si", "OB", "OH", "HO").
        slab_types: Optional per-role moltemplate @atom type-name override.
        slab_charges: Optional per-role charge (e) override.
        slab_lj: Optional per-role (epsilon kcal/mol, sigma A) override.
        hydroxylate_bottom: Cap bottom-face dangling oxygens with H
                (default True; both faces are exposed when vacuum=0).
        slab_seed: RNG seed for the silanol bridge pairing.
    """
    model: Optional[object] = None
    lt_file: Optional[str] = None
    class_name: Optional[str] = None
    thickness: float = 10.0
    packing: str = "grid"
    gap: float = 3.0
    count: Union[int, str, None] = None
    density: float = 0.085
    vacuum: float = 0.0
    builder: Optional[str] = None
    oh_density: float = 4.6
    slab_ff: str = "interface"
    slab_types: Optional[dict] = None
    slab_charges: Optional[dict] = None
    slab_lj: Optional[dict] = None
    hydroxylate_bottom: bool = True
    slab_seed: int = 42

    def __post_init__(self) -> None:
        n_sources = sum(
            s is not None for s in (self.model, self.lt_file, self.builder)
        )
        if n_sources != 1:
            raise ValidationError(
                "SubstrateSpec requires exactly one source: 'model' "
                "(in-pipeline Molecule/Polymer), 'lt_file' (external "
                "slab), or 'builder' (built-in surface builder)"
            )
        if self.builder is not None:
            from ..surfaces import SLAB_BUILDERS
            if self.builder not in SLAB_BUILDERS:
                raise ValidationError(
                    f"Unknown substrate builder '{self.builder}'; "
                    f"available: {sorted(SLAB_BUILDERS)}"
                )
            if self.slab_ff not in ("interface", "clayff", "custom"):
                raise ValidationError(
                    f"Unknown slab_ff '{self.slab_ff}'; choose "
                    "'interface', 'clayff', or 'custom'"
                )
            if self.slab_ff == "custom":
                from ..surfaces.ff_tables import SLAB_ROLES
                for name, table in (("slab_types", self.slab_types),
                                    ("slab_charges", self.slab_charges),
                                    ("slab_lj", self.slab_lj)):
                    if table is None or any(
                            r not in table for r in SLAB_ROLES):
                        raise ValidationError(
                            f"slab_ff='custom' requires {name} covering "
                            f"all roles {SLAB_ROLES}"
                        )
            if self.oh_density < 0:
                raise ValidationError(
                    f"oh_density must be >= 0, got {self.oh_density}"
                )
        if self.lt_file is not None and not self.class_name:
            raise ValidationError(
                "SubstrateSpec with lt_file requires 'class_name' "
                "(the moltemplate class defined by that file)"
            )
        if self.thickness <= 0:
            raise ValidationError(
                f"SubstrateSpec thickness must be > 0, got {self.thickness}"
            )
        if self.gap < 0:
            raise ValidationError(
                f"SubstrateSpec gap must be >= 0, got {self.gap}"
            )
        if self.vacuum < 0:
            raise ValidationError(
                f"SubstrateSpec vacuum must be >= 0, got {self.vacuum}"
            )
        if self.packing not in SUBSTRATE_PACKING_MODES:
            raise ValidationError(
                f"SubstrateSpec packing must be one of "
                f"{list(SUBSTRATE_PACKING_MODES)}, got '{self.packing}'"
            )
        if self.count is not None:
            if isinstance(self.count, str):
                if self.count != AUTO_COUNT:
                    raise ValidationError(
                        f"SubstrateSpec count must be None, a positive int, "
                        f"or '{AUTO_COUNT}', got '{self.count}'"
                    )
            elif not (isinstance(self.count, int) and self.count > 0):
                raise ValidationError(
                    f"SubstrateSpec count must be a positive int, "
                    f"got {self.count}"
                )
        if self.density <= 0:
            raise ValidationError(
                f"SubstrateSpec density must be > 0, got {self.density}"
            )

    @property
    def is_external(self) -> bool:
        """True for a pre-built external .lt slab (no in-pipeline model)."""
        return self.lt_file is not None

    @property
    def is_builder(self) -> bool:
        """True for a built-in surface builder slab (e.g. alpha_quartz)."""
        return self.builder is not None


@dataclass
class PackingContext:
    """
    Everything a placement strategy needs.

    Attributes:
        units: Typed unit manifest from stage 2.
        box: Box specification (explicit size or auto).
        mc_max_attempts: Maximum random placement attempts per instance.
        rng_seed: Optional seed for reproducible stochastic strategies.
        offset: Nominal monomer-monomer spacing in Angstrom (grid layout,
                chain-length box estimates).
        monomer_density: Target monomer density (monomers/A^3) for box sizing.
        substrate: Optional SubstrateSpec (required by the "on_substrate"
                   strategy; ignored otherwise).
        subtract: Optional list of CarveRegion specs applied to placed
                  instances before lt command emission (whole-instance
                  removal; supported by "mc_random" and "on_substrate").
        nanocomposite: Optional NanocompositePacking configuration
                  ("nanocomposite_mc" strategy).
    """
    units: UnitLibrary
    box: BoxSpec = field(default_factory=BoxSpec)
    mc_max_attempts: int = 10000
    rng_seed: Optional[int] = None
    offset: float = 4.0
    monomer_density: float = 0.085
    substrate: Optional[SubstrateSpec] = None
    subtract: Optional[List["CarveRegion"]] = None
    nanocomposite: Optional[object] = None


@dataclass
class PlacementRecord:
    """
    One placed instance.

    Attributes:
        unit_id: ID of the UnitSpec this instance belongs to.
        instance_name: Moltemplate instance name ("polymer_1", "molecule_3").
        lt_command: Full moltemplate instantiation command line.
    """
    unit_id: str
    instance_name: str
    lt_command: str


@dataclass
class PlacementResult:
    """
    Result of a placement strategy run.

    Attributes:
        records: Placement commands in system.lt emission order.
        box_bounds: Box bounds actually used (after auto-sizing).
        report: Optional strategy report (achieved clearances, loading,
                box sizing); written to the force-field/placement report.
        extra_lt_files: Optional {file name: .lt text} written into the
                moltemplate directory and imported by system.lt (e.g.
                per-instance chain classes regrown in place).
    """
    records: List[PlacementRecord]
    box_bounds: BoxBounds
    report: Optional[dict] = None
    extra_lt_files: Dict[str, str] = field(default_factory=dict)


def reject_inorganic_units(ctx: "PackingContext", strategy: str,
                           allow_substrate_slab: bool = False) -> None:
    """
    Refuse inorganic units a strategy cannot place (never treat a filler
    or slab as a generic molecule).

    Args:
        allow_substrate_slab: Accept the stage-2 typed built-in substrate
            slab unit (on_substrate places it as the substrate instance).
    """
    bad = [
        u.id for u in ctx.units.units
        if getattr(u, "kind", None) == "inorganic"
        and not (allow_substrate_slab and u.role == "substrate")
    ]
    if bad:
        raise ValidationError(
            f"Strategy '{strategy}' cannot place inorganic unit(s) {bad}: "
            "use strategy='nanocomposite_mc' for nanoparticle fillers, or "
            "SubstrateSpec(builder=...) with 'on_substrate' for built-in "
            "slabs"
        )


class PlacementStrategy(ABC):
    """
    Abstract base class for box-packing strategies.

    Subclasses set the class attribute `name` and implement `place()`.
    Register custom strategies at runtime with
    ``AutoPoly.packing.register_strategy(name, cls)``.

    Reserved built-in names for future sub-projects: ``grafted_surface``,
    ``nanopore``.
    """

    name: ClassVar[str] = ""

    @abstractmethod
    def place(self, ctx: PackingContext) -> PlacementResult:
        """
        Compute placements for all units in the context.

        Args:
            ctx: Packing context with units, box spec, and MC config.

        Returns:
            PlacementResult with per-instance lt commands and box bounds.
        """
        raise NotImplementedError
