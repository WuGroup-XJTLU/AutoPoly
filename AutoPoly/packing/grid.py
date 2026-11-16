#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic grid packing strategy.

Units are placed on a square 2D grid in the z=0 plane with per-unit spacing:
- molecules: fixed 5.0 A spacing
- ring polymers: 2.5x the ring radius
- linear polymers: offset * (n_monomers + 2)

The box is sized from the largest grid extent plus 20% padding, unless an
explicit box size was requested.

Extracted from the former WorkflowManager.make_system_lt (workflow.py).

Created on 2026-07-30
@author: zwu
"""
import numpy as np

from ..core.exceptions import ValidationError
from ..core.system import logger
from ..pipeline.units import UNIT_KIND_POLYMER
from .base import (
    BoxSpec,
    PackingContext,
    PlacementRecord,
    PlacementResult,
    PlacementStrategy,
    reject_inorganic_units,
    symmetric_bounds,
)

MOLECULE_SPACING = 5.0
RING_RADIUS_FACTOR = 2.5
BOX_PADDING_FACTOR = 1.2


class GridStrategy(PlacementStrategy):
    """Deterministic planar grid placement."""

    name = "grid"

    def place(self, ctx: PackingContext) -> PlacementResult:
        reject_inorganic_units(ctx, self.name)
        if ctx.substrate is not None:
            raise ValidationError(
                "Strategy 'grid' does not support substrates; "
                "use strategy 'on_substrate'"
            )
        if ctx.subtract:
            raise ValidationError(
                "Strategy 'grid' does not support subtract regions; "
                "use strategy 'mc_random' or 'on_substrate'"
            )

        records = []
        polymer_index = 0
        molecule_index = 0
        max_extent = 0.0

        for unit in ctx.units.units:
            spacing = self._unit_spacing(unit, ctx.offset)
            n_instances = unit.count
            grid_size = int(np.ceil(np.sqrt(n_instances)))
            max_extent = max(max_extent, grid_size * spacing)

            for i in range(n_instances):
                grid_x = i % grid_size
                grid_y = i // grid_size
                pos_x = grid_x * spacing
                pos_y = grid_y * spacing
                pos_z = 0.0

                if unit.kind == UNIT_KIND_POLYMER:
                    polymer_index += 1
                    instance_name = f"polymer_{polymer_index}"
                else:
                    molecule_index += 1
                    instance_name = f"molecule_{molecule_index}"

                command = (
                    f"{instance_name} = new {unit.class_name}"
                    f".move({pos_x:.4f},{pos_y:.4f},{pos_z:.4f})"
                )
                records.append(PlacementRecord(unit.id, instance_name, command))

        box_size = ctx.box.requested_box_size or (max_extent * BOX_PADDING_FACTOR)
        box_bounds = symmetric_bounds(box_size)

        logger.info(
            f"Grid placement complete: {polymer_index} polymers, "
            f"{molecule_index} molecules, box size {box_size:.2f} A"
        )
        return PlacementResult(records=records, box_bounds=box_bounds)

    @staticmethod
    def _unit_spacing(unit, offset: float) -> float:
        """Per-unit grid spacing in Angstrom."""
        if unit.kind != UNIT_KIND_POLYMER:
            return MOLECULE_SPACING
        if unit.topology == "ring":
            return unit.radius * RING_RADIUS_FACTOR
        return offset * ((unit.n_monomers or 1) + 2)
