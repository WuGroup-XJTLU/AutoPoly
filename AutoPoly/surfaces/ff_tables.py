#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Force-field parameter tables for built-in substrate surface builders.

Compatibility module: the tables now live in
``AutoPoly.materials.forcefields.tables`` (single source of truth shared
with the nanoparticle builders) and are re-exported here unchanged.  See
that module for the role conventions, units and references
(INTERFACE FF v1.5, Emami et al. 2014; CLAYFF, Cygan et al. 2004).

Created on 2026-08-12
@author: zwu
"""
from ..materials.forcefields.tables import (  # noqa: F401
    _TWO_POW_1_6,
    _entry,
    CLAYFF,
    INTERFACE_FF,
    OH_BOND_LENGTH,
    SLAB_FF_REGISTRY,
    SLAB_ROLES,
)

__all__ = [
    "CLAYFF",
    "INTERFACE_FF",
    "OH_BOND_LENGTH",
    "SLAB_FF_REGISTRY",
    "SLAB_ROLES",
]
