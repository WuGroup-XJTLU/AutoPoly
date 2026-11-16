#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Substrate packing strategy: a polymer/molecule film on a physical slab.

Box layout (z-layered assembly, laterally centered on the origin):

    zhi ┌───────────────────────────┐
        │    vacuum (spec.vacuum)   │  0 by default (fully periodic slab)
        ├── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
        │    film region            │  film units, MC-placed with
        │                           │  collision detection
        ├── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤  film_zmin = slab_top + gap
        │    gap                    │
        ├───────────────────────────┤  slab_top = zlo + thickness
        │    substrate slab         │  model-built (grid or mc packing)
        │                           │  or one external .lt instance
    zlo └───────────────────────────┘

Box sides come from BoxSpec: explicit per-axis box_dims win, then the
cubic requested_box_size, then auto-sizing (lateral from the standard
melt estimate, lz = thickness + gap + film thickness at monomer_density
+ vacuum).

Collision handling: model-built substrate instances are registered as
spheres in the shared CollisionDetector before film placement, so film
chains cannot interpenetrate the slab. For an external .lt slab the
exclusion is geometric: film sphere centers stay at least one bounding
radius above film_zmin (the placer's margin), i.e. the slab is treated
as a hard wall at slab_top + gap.

Created on 2026-07-30
@author: zwu
"""
import numpy as np

from ..core.exceptions import ValidationError
from ..core.system import logger
from ..pipeline.units import UNIT_KIND_POLYMER, UNIT_ROLE_SUBSTRATE
from ..mc import CollisionDetector, MolecularPlacementMC
from .base import (
    PackingContext,
    PlacementRecord,
    PlacementResult,
    PlacementStrategy,
    SubstrateSpec,
    reject_inorganic_units,
)
from .random_mc import compute_auto_box_size, emit_lt_command, unit_role
from .regions import PlacedItem, apply_carve_regions

#: Sphere overlap tolerance for ordered grid slabs (bounding radii are
#: conservative; 10% overlap of spheres is normally clash-free atomistically).
GRID_SPACING_FACTOR = 1.8
#: Collision-ID offset for substrate spheres (avoids placer ID ranges).
SUBSTRATE_ID_OFFSET = 300000


class OnSubstrateStrategy(PlacementStrategy):
    """Film-on-substrate placement with optional carve subtract."""

    name = "on_substrate"

    def place(self, ctx: PackingContext) -> PlacementResult:
        spec = ctx.substrate
        if spec is None:
            raise ValidationError(
                "Strategy 'on_substrate' requires a SubstrateSpec "
                "(pass substrate=... to generate()/BoxPacker)"
            )
        reject_inorganic_units(ctx, self.name,
                               allow_substrate_slab=spec.is_builder)
        if ctx.rng_seed is not None:
            np.random.seed(ctx.rng_seed)

        lx, ly, lz = self._resolve_box_dims(ctx, spec)
        zlo, zhi = -lz / 2.0, lz / 2.0
        slab_top = zlo + spec.thickness
        film_zmin = slab_top + spec.gap
        self._validate_film_room(ctx, zhi - film_zmin, lz)
        self._warn_if_film_dense(ctx, (lx, ly, zhi - film_zmin))

        box_bounds = ((-lx / 2, lx / 2), (-ly / 2, ly / 2), (zlo, zhi))
        cell_size = max(5.0, min(lx, ly, lz) / 20)
        detector = CollisionDetector(box_bounds, cell_size)

        items = []
        if spec.is_external or spec.is_builder:
            items.append(self._external_slab_item(spec, zlo))
        else:
            self._place_model_substrate(ctx, spec, detector, box_bounds, items)

        film_bounds = (box_bounds[0], box_bounds[1], (film_zmin, zhi))
        placer = MolecularPlacementMC(
            film_bounds, detector, ctx.mc_max_attempts
        )
        self._place_film(ctx, placer, film_bounds, items)

        kept = apply_carve_regions(items, ctx.subtract)
        records = [
            PlacementRecord(
                item.unit_id, item.instance_name,
                emit_lt_command(placer, item),
            )
            for item in kept
        ]

        logger.info(
            f"Substrate placement complete: box ({lx:.2f} x {ly:.2f} x "
            f"{lz:.2f}) A, slab z in [{zlo:.2f}, {slab_top:.2f}], "
            f"film z in [{film_zmin:.2f}, {zhi:.2f}], "
            f"{len(records)} instances"
        )
        return PlacementResult(records=records, box_bounds=box_bounds)

    # ------------------------------------------------------------------
    # Box geometry
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_box_dims(ctx: PackingContext, spec: SubstrateSpec):
        """
        Resolve (lx, ly, lz): explicit box_dims per axis, then cubic
        requested_box_size, then auto-sizing. Auto lz stacks slab + gap +
        film at monomer_density + vacuum. Builder slabs snap the lateral
        box to the slab's exact dimensions (seamless periodic slab).
        """
        dims = ctx.box.box_dims or (None, None, None)
        cubic = ctx.box.requested_box_size

        if spec.is_builder:
            built = getattr(spec, "_built_slab", None)
            if built is None:
                raise ValidationError(
                    "Builder substrate has not been materialized — the "
                    "slab is generated by BoxPacker before the strategy "
                    "runs (use generate() or BoxPacker.pack())"
                )
            if dims[0] is not None and abs(dims[0] - built.lx) > 0.01:
                logger.info(
                    f"Box lx snapped {dims[0]:.3f} -> {built.lx:.3f} A "
                    "(integer quartz surface cells)"
                )
            if dims[1] is not None and abs(dims[1] - built.ly) > 0.01:
                logger.info(
                    f"Box ly snapped {dims[1]:.3f} -> {built.ly:.3f} A "
                    "(integer quartz surface cells)"
                )
            dims = (built.lx, built.ly, dims[2])

        auto_lateral = None

        def lateral(value):
            nonlocal auto_lateral
            if value is not None:
                return float(value)
            if cubic is not None:
                return float(cubic)
            if auto_lateral is None:
                auto_lateral = compute_auto_box_size(ctx)
            return auto_lateral

        lx = lateral(dims[0])
        ly = lateral(dims[1])

        if dims[2] is not None:
            lz = float(dims[2])
        elif cubic is not None:
            lz = float(cubic)
        else:
            film_particles = sum(
                (u.n_monomers or 1) * u.count
                if u.kind == UNIT_KIND_POLYMER else u.count
                for u in ctx.units.units
                if unit_role(u) != UNIT_ROLE_SUBSTRATE
            )
            t_film = film_particles / (ctx.monomer_density * lx * ly)
            lz = spec.thickness + spec.gap + t_film + spec.vacuum

        if lz < spec.thickness + spec.gap:
            raise ValidationError(
                f"Box lz ({lz:.2f} A) is smaller than substrate thickness "
                f"+ gap ({spec.thickness + spec.gap:.2f} A); increase "
                f"box_dims[2] or reduce the slab"
            )
        return lx, ly, lz

    @staticmethod
    def _validate_film_room(ctx: PackingContext,
                            film_thickness: float, lz: float) -> None:
        """Film region must fit the largest film bounding sphere."""
        film_radii = [
            u.radius for u in ctx.units.units
            if unit_role(u) != UNIT_ROLE_SUBSTRATE
        ]
        if not film_radii:
            return
        needed = 2.0 * max(film_radii)
        if film_thickness < needed:
            raise ValidationError(
                f"Film region ({film_thickness:.2f} A) is thinner than the "
                f"largest film unit diameter ({needed:.2f} A); increase "
                f"box_dims[2] (currently lz={lz:.2f} A) or lower the "
                f"substrate thickness/gap"
            )

    @staticmethod
    def _warn_if_film_dense(ctx: PackingContext, dims) -> None:
        """
        Warn early when the film units' bounding-sphere volume is a
        large fraction of the film region: MC placement of whole chains
        is likely to fail (the fallback grid then errors out).
        """
        lx, ly, film_thickness = dims
        film_vol = lx * ly * film_thickness
        sphere_vol = 0.0
        for u in ctx.units.units:
            if unit_role(u) == UNIT_ROLE_SUBSTRATE:
                continue
            n = u.count if u.kind != UNIT_KIND_POLYMER else (
                u.count or 1
            )
            sphere_vol += n * (4.0 / 3.0) * np.pi * u.radius ** 3
        if film_vol > 0 and sphere_vol / film_vol > 0.3:
            logger.warning(
                f"Film bounding-sphere volume ({sphere_vol:.0f} A^3) is "
                f"{100 * sphere_vol / film_vol:.0f}% of the film region "
                f"({film_vol:.0f} A^3); whole-chain MC placement is "
                "likely to fail. Consider a larger box, fewer/shorter "
                "chains, or coiled chains (use_mc_chain_growth=True)."
            )

    # ------------------------------------------------------------------
    # Substrate placement
    # ------------------------------------------------------------------
    @staticmethod
    def _external_slab_item(spec: SubstrateSpec, zlo: float) -> PlacedItem:
        """Single external/built .lt slab instance, centered laterally."""
        if spec.is_builder:
            built = getattr(spec, "_built_slab", None)
            if built is None:
                raise ValidationError(
                    "Builder substrate has not been materialized — the "
                    "slab is generated by BoxPacker before the strategy "
                    "runs (use generate() or BoxPacker.pack())"
                )
            class_name = built.class_name
        else:
            class_name = spec.class_name
        z_center = zlo + spec.thickness / 2.0
        command = (
            f"substrate = new {class_name}"
            f".move(0.0,0.0,{z_center:.4f})"
        )
        return PlacedItem(
            "external", "substrate", UNIT_ROLE_SUBSTRATE,
            (0.0, 0.0, z_center), 0.0,
            ("command", command),
        )

    def _place_model_substrate(
        self,
        ctx: PackingContext,
        spec: SubstrateSpec,
        detector: CollisionDetector,
        box_bounds,
        items: list,
    ) -> None:
        substrate_units = [
            u for u in ctx.units.units
            if unit_role(u) == UNIT_ROLE_SUBSTRATE
            and getattr(u, "kind", None) != "inorganic"
        ]
        if not substrate_units:
            raise ValidationError(
                "SubstrateSpec uses a model substrate, but the unit "
                "manifest contains no role='substrate' units — was the "
                "substrate model passed to the geometry/typing stages?"
            )

        (xlo, xhi), (ylo, yhi), (zlo, _) = box_bounds
        slab_bounds = ((xlo, xhi), (ylo, yhi), (zlo, zlo + spec.thickness))

        if spec.packing == "mc":
            self._place_substrate_mc(ctx, slab_bounds, detector,
                                     substrate_units, items)
        else:
            self._place_substrate_grid(ctx, slab_bounds, detector,
                                       substrate_units, items)

    def _place_substrate_mc(self, ctx, slab_bounds, detector,
                            substrate_units, items) -> None:
        """Amorphous slab: MC placement restricted to the slab region."""
        placer = MolecularPlacementMC(
            slab_bounds, detector, ctx.mc_max_attempts
        )
        index = 0
        for unit in substrate_units:
            for _ in range(unit.count):
                index += 1
                instance_name = f"substrate_{index}"
                if unit.kind == UNIT_KIND_POLYMER:
                    placement = placer.place_polymer(
                        poly_name=unit.class_name, radius=unit.radius
                    )
                    payload = ("polymer", placement) if placement else None
                else:
                    placement = placer.place_molecule(
                        molecule_name=unit.class_name,
                        radius=unit.radius,
                        instance_name=instance_name,
                    )
                    payload = ("molecule", placement) if placement else None
                if placement is None:
                    raise ValidationError(
                        f"Failed to place substrate instance {index} "
                        f"({unit.id}) in the slab region — slab too dense "
                        f"for MC packing. Increase thickness/box_dims, "
                        f"lower the count, or use packing='grid'."
                    )
                items.append(PlacedItem(
                    unit.id, instance_name, UNIT_ROLE_SUBSTRATE,
                    placement.position, placement.radius, payload,
                ))
        logger.info(f"Substrate slab: {index} instances MC-placed")

    def _place_substrate_grid(self, ctx, slab_bounds, detector,
                              substrate_units, items) -> None:
        """Ordered slab: instances on a 3D grid inside the slab region."""
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = slab_bounds
        lx, ly = xhi - xlo, yhi - ylo
        thickness = zhi - zlo
        index = 0
        for unit in substrate_units:
            spacing = max(GRID_SPACING_FACTOR * unit.radius, 1.0)
            nx = max(1, int(lx / spacing))
            ny = max(1, int(ly / spacing))
            nz = max(1, int(thickness / spacing))
            if unit.count > nx * ny * nz:
                raise ValidationError(
                    f"Substrate unit '{unit.id}' count ({unit.count}) "
                    f"exceeds grid capacity ({nx}x{ny}x{nz} = "
                    f"{nx * ny * nz}) for spacing {spacing:.2f} A; "
                    f"increase box_dims/thickness or lower the count"
                )
            x0 = xlo + (lx - (nx - 1) * spacing) / 2.0
            y0 = ylo + (ly - (ny - 1) * spacing) / 2.0
            z0 = zlo + (thickness - (nz - 1) * spacing) / 2.0
            for n in range(unit.count):
                index += 1
                ix = n % nx
                iy = (n // nx) % ny
                iz = n // (nx * ny)
                pos = (x0 + ix * spacing, y0 + iy * spacing,
                       z0 + iz * spacing)
                instance_name = f"substrate_{index}"
                command = (
                    f"{instance_name} = new {unit.class_name}"
                    f".move({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f})"
                )
                detector.add_monomer(
                    SUBSTRATE_ID_OFFSET + index,
                    np.array(pos), unit.radius,
                )
                items.append(PlacedItem(
                    unit.id, instance_name, UNIT_ROLE_SUBSTRATE,
                    pos, unit.radius, ("command", command),
                ))
        logger.info(f"Substrate slab: {index} instances grid-packed")

    # ------------------------------------------------------------------
    # Film placement
    # ------------------------------------------------------------------
    def _place_film(self, ctx, placer, film_bounds, items) -> None:
        """MC-place film units above slab_top + gap (grid fallback)."""
        (xlo, xhi), (ylo, yhi), (film_zmin, _) = film_bounds
        polymer_index = 0
        molecule_index = 0
        for unit in ctx.units.units:
            if unit_role(unit) == UNIT_ROLE_SUBSTRATE:
                continue
            if unit.kind == UNIT_KIND_POLYMER:
                polymer_index += 1
                instance_name = f"polymer_{polymer_index}"
                placement = placer.place_polymer(
                    poly_name=unit.class_name, radius=unit.radius
                )
                if placement is not None:
                    items.append(PlacedItem(
                        unit.id, instance_name, "film",
                        placement.position, placement.radius,
                        ("polymer", placement),
                    ))
                else:
                    items.append(self._film_fallback(
                        unit, instance_name, polymer_index - 1,
                        xlo, xhi, ylo, yhi, film_zmin,
                    ))
            else:
                for _ in range(unit.count):
                    molecule_index += 1
                    instance_name = f"molecule_{molecule_index}"
                    placement = placer.place_molecule(
                        molecule_name=unit.class_name,
                        radius=unit.radius,
                        instance_name=instance_name,
                    )
                    if placement is not None:
                        items.append(PlacedItem(
                            unit.id, instance_name, "film",
                            placement.position, placement.radius,
                            ("molecule", placement),
                        ))
                    else:
                        items.append(self._film_fallback(
                            unit, instance_name, molecule_index - 1,
                            xlo, xhi, ylo, yhi, film_zmin,
                        ))

    @staticmethod
    def _film_fallback(unit, instance_name, index, xlo, xhi, ylo, yhi,
                       film_zmin) -> PlacedItem:
        """
        Deterministic lateral grid position at the film bottom.

        The grid guarantees non-overlapping bounding spheres; if it
        cannot hold one more instance this is a hard error (silently
        stacking instances would produce atom clashes in the data file).
        """
        radius = unit.radius
        spacing = radius * 2.5
        nx = max(1, int((xhi - xlo - 2 * radius) / spacing))
        ny = max(1, int((yhi - ylo - 2 * radius) / spacing))
        if index >= nx * ny:
            raise ValidationError(
                f"Failed to place film instance {instance_name} "
                f"({unit.id}) and the fallback grid is full "
                f"({nx}x{ny} positions for radius {radius:.2f} A). "
                "Use a larger box, fewer film units, or enable MC chain "
                "growth (GeometryConfig(use_mc_chain_growth=True)) so "
                "chains are coiled and have smaller bounding spheres."
            )
        logger.warning(
            f"Failed to place film instance {instance_name} ({unit.id}), "
            f"using fallback grid position {index + 1}/{nx * ny}"
        )
        ix = index % nx
        iy = (index // nx) % ny
        pos = (xlo + radius + ix * spacing,
               ylo + radius + iy * spacing,
               film_zmin + radius)
        command = (
            f"{instance_name} = new {unit.class_name}"
            f".move({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f})"
        )
        return PlacedItem(
            unit.id, instance_name, "film", pos, radius,
            ("command", command),
        )
