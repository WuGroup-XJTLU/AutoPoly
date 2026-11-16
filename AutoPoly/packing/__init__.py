#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pluggable box-packing strategies for AutoPoly.

This package contains the stage-3 placement strategies that arrange typed
units (from a UnitLibrary manifest) inside the simulation box:

- base.py:        BoxSpec, SubstrateSpec, PackingContext, PlacementResult,
                  PlacementStrategy ABC
- registry.py:    strategy registry (built-ins + user-defined at runtime)
- grid.py:        deterministic planar grid placement
- random_mc.py:   Monte Carlo random placement with collision detection
- on_substrate.py: film-on-substrate placement (physical slab + film)
- regions.py:     carve regions (CutAbove/CutBelow/Cylinder/BoxRegion) for
                  whole-instance subtract after placement
- nanocomposite.py: filler-first, obstacle-aware atom-level packing of
                  nanoparticles + polymer matrix ("nanocomposite_mc")
- collision_geometry.py: periodic voxel collision grid, PBC contact checks

Created on 2026-07-30
@author: zwu
"""
from .base import (
    AUTO_COUNT,
    SUBSTRATE_PACKING_MODES,
    BoxSpec,
    PackingContext,
    PlacementRecord,
    PlacementResult,
    PlacementStrategy,
    SubstrateSpec,
)
from .registry import (
    get_strategy,
    register_strategy,
    registered_strategies,
    RESERVED_STRATEGY_NAMES,
)
from .regions import (
    APPLY_TO_CHOICES,
    BoxRegion,
    CarveRegion,
    CutAbove,
    CutBelow,
    Cylinder,
    PlacedItem,
    apply_carve_regions,
)
from .grid import GridStrategy
from .random_mc import RandomMCStrategy
from .on_substrate import OnSubstrateStrategy
from .nanocomposite import (
    NanocompositeMCStrategy,
    NanocompositePacking,
    PackingError,
)

# Register built-in strategies.
register_strategy(GridStrategy.name, GridStrategy)
register_strategy(RandomMCStrategy.name, RandomMCStrategy)
register_strategy(OnSubstrateStrategy.name, OnSubstrateStrategy)
register_strategy(NanocompositeMCStrategy.name, NanocompositeMCStrategy)

__all__ = [
    "AUTO_COUNT",
    "SUBSTRATE_PACKING_MODES",
    "BoxSpec",
    "PackingContext",
    "PlacementRecord",
    "PlacementResult",
    "PlacementStrategy",
    "SubstrateSpec",
    "APPLY_TO_CHOICES",
    "BoxRegion",
    "CarveRegion",
    "CutAbove",
    "CutBelow",
    "Cylinder",
    "PlacedItem",
    "apply_carve_regions",
    "GridStrategy",
    "RandomMCStrategy",
    "OnSubstrateStrategy",
    "NanocompositeMCStrategy",
    "NanocompositePacking",
    "PackingError",
    "get_strategy",
    "register_strategy",
    "registered_strategies",
    "RESERVED_STRATEGY_NAMES",
]
