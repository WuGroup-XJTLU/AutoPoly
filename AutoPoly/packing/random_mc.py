#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo random packing strategy.

Polymers and molecules are placed at random positions/orientations with
collision detection (MolecularPlacementMC). The box is auto-sized as the
maximum of three estimates:

- density-based: total monomers at the target monomer density
- chain-length: longest chain's SAW end-to-end distance x safety factor
- packing-volume: sum of unit collision volumes at 30% packing fraction

If an instance cannot be placed after mc_max_attempts tries, it falls back
to a deterministic grid position with a warning (same policy as the legacy
workflow).

Extracted from the former WorkflowManager.make_system_lt_mc (workflow.py).

Created on 2026-07-30
@author: zwu
"""
import numpy as np

from ..core.system import logger
from ..core.exceptions import ValidationError
from ..pipeline.units import UNIT_KIND_POLYMER
from ..mc import CollisionDetector, MolecularPlacementMC, calculate_box_size
from .base import (
    PackingContext,
    PlacementRecord,
    PlacementResult,
    PlacementStrategy,
    bounds_size,
    reject_inorganic_units,
    symmetric_bounds,
)
from .regions import PlacedItem, apply_carve_regions

PACKING_FRACTION = 0.3
CHAIN_LENGTH_SAFETY_FACTOR = 3.0
GRID_FALLBACK_FACTOR = 2.5


def unit_role(unit) -> str:
    """Unit role ("film" default; manifests predate the role field)."""
    return getattr(unit, "role", "film") or "film"


def emit_lt_command(placer, item: PlacedItem) -> str:
    """Materialize the moltemplate command for a kept PlacedItem."""
    kind, payload = item.payload
    if kind == "command":
        return payload
    if kind == "polymer":
        return placer.generate_polymer_lt_commands([payload])[0]
    return placer.generate_molecule_lt_commands([payload])[0]


def compute_auto_box_size(ctx: PackingContext) -> float:
    """
    Auto-size the simulation box as max(density, chain-length, packing-volume).

    Args:
        ctx: Packing context with the unit manifest.

    Returns:
        Cubic box side length in Angstrom.
    """
    units = ctx.units

    # Density-based estimate
    density_box = calculate_box_size(
        units.total_particle_count(), ctx.monomer_density
    )

    # Chain-length estimate: ensure the longest chain fits inside the box.
    # For a SAW, end-to-end ~ n^0.6 * bond length; apply a safety factor.
    max_dop = units.max_chain_length()
    chain_length_box = max_dop**0.6 * ctx.offset * CHAIN_LENGTH_SAFETY_FACTOR

    # Packing-volume estimate: enough room for non-overlapping placement
    # (inorganic substrate slabs are sized by the box, not packed)
    total_collision_volume = sum(
        unit.count * (4.0 / 3.0) * np.pi * unit.radius**3
        for unit in units.units
        if not (getattr(unit, "kind", None) == "inorganic"
                and unit.role == "substrate")
    )
    packing_box = (
        (total_collision_volume / PACKING_FRACTION) ** (1.0 / 3.0)
        if total_collision_volume > 0 else 0.0
    )

    return max(density_box, chain_length_box, packing_box)


class RandomMCStrategy(PlacementStrategy):
    """Monte Carlo random placement with collision detection."""

    name = "mc_random"

    def place(self, ctx: PackingContext) -> PlacementResult:
        reject_inorganic_units(ctx, self.name)
        if ctx.rng_seed is not None:
            np.random.seed(ctx.rng_seed)

        dims = ctx.box.box_dims or (None, None, None)
        if any(d is not None for d in dims):
            # explicit per-axis box: auto-size only the unspecified axes
            auto = compute_auto_box_size(ctx)
            lx, ly, lz = (
                float(d) if d is not None else auto for d in dims
            )
            box_bounds = ((-lx / 2, lx / 2), (-ly / 2, ly / 2),
                          (-lz / 2, lz / 2))
            half_box = min(lx, ly, lz) / 2.0
            box_size = max(lx, ly, lz)
        else:
            box_size = (ctx.box.requested_box_size
                        or compute_auto_box_size(ctx))
            box_bounds = symmetric_bounds(box_size)
            half_box = box_size / 2.0

        cell_size = max(5.0, box_size / 20)
        collision_detector = CollisionDetector(box_bounds, cell_size)
        placer = MolecularPlacementMC(
            box_bounds, collision_detector, ctx.mc_max_attempts
        )

        items = []
        polymer_index = 0
        molecule_index = 0

        for unit in ctx.units.units:
            if unit.kind == UNIT_KIND_POLYMER:
                polymer_index += 1
                items.append(self._place_polymer(
                    placer, unit, polymer_index, half_box
                ))
            else:
                for _ in range(unit.count):
                    molecule_index += 1
                    items.append(self._place_molecule(
                        placer, unit, molecule_index, half_box
                    ))

        kept = apply_carve_regions(items, ctx.subtract)
        records = [
            PlacementRecord(
                item.unit_id, item.instance_name,
                emit_lt_command(placer, item),
            )
            for item in kept
        ]

        stats = placer.get_placement_stats()
        logger.info(
            f"MC placement complete: {stats['polymers']} polymers, "
            f"{stats['molecules']} molecules placed, "
            f"box size {bounds_size(box_bounds):.2f} A"
        )
        return PlacementResult(records=records, box_bounds=box_bounds)

    def _place_polymer(
        self, placer, unit, polymer_index: int, half_box: float
    ) -> PlacedItem:
        """Place one polymer chain; grid fallback on failure."""
        instance_name = f"polymer_{polymer_index}"
        placement = placer.place_polymer(
            poly_name=unit.class_name, radius=unit.radius
        )
        if placement is not None:
            return PlacedItem(
                unit.id, instance_name, unit_role(unit),
                placement.position, placement.radius,
                ("polymer", placement),
            )

        logger.warning(
            f"Failed to place polymer {polymer_index} ({unit.id}), "
            "using fallback grid position"
        )
        return self._fallback_item(
            unit, instance_name, polymer_index - 1, half_box
        )

    def _place_molecule(
        self, placer, unit, molecule_index: int, half_box: float
    ) -> PlacedItem:
        """Place one molecule instance; grid fallback on failure."""
        instance_name = f"molecule_{molecule_index}"
        placement = placer.place_molecule(
            molecule_name=unit.class_name,
            radius=unit.radius,
            instance_name=instance_name,
        )
        if placement is not None:
            return PlacedItem(
                unit.id, instance_name, unit_role(unit),
                placement.position, placement.radius,
                ("molecule", placement),
            )

        logger.warning(
            f"Failed to place molecule {molecule_index} ({unit.id}), "
            "using fallback grid position"
        )
        return self._fallback_item(
            unit, instance_name, molecule_index - 1, half_box
        )

    @staticmethod
    def _fallback_item(
        unit, instance_name: str, index: int, half_box: float
    ) -> PlacedItem:
        """
        Deterministic 3D grid position used when MC placement fails.

        The grid guarantees non-overlapping bounding spheres; if it
        cannot hold one more instance this is a hard error (silently
        wrapping the index would stack instances on identical positions
        and produce atom clashes in the data file).
        """
        radius = unit.radius
        spacing = radius * GRID_FALLBACK_FACTOR
        grid_per_dim = max(1, int((2 * half_box - 2 * radius) / spacing))
        if index >= grid_per_dim ** 3:
            raise ValidationError(
                f"Failed to place instance {instance_name} ({unit.id}) "
                f"and the fallback grid is full ({grid_per_dim}^3 = "
                f"{grid_per_dim ** 3} positions for radius "
                f"{radius:.2f} A). Use a larger box or fewer units."
            )
        ix = index % grid_per_dim
        iy = (index // grid_per_dim) % grid_per_dim
        iz = (index // (grid_per_dim * grid_per_dim)) % grid_per_dim
        pos_x = -half_box + radius + ix * spacing
        pos_y = -half_box + radius + iy * spacing
        pos_z = -half_box + radius + iz * spacing
        command = (
            f"{instance_name} = new {unit.class_name}"
            f".move({pos_x:.4f},{pos_y:.4f},{pos_z:.4f})"
        )
        return PlacedItem(
            unit.id, instance_name, unit_role(unit),
            (pos_x, pos_y, pos_z), radius,
            ("command", command),
        )
