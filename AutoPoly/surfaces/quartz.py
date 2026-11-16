#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crystalline alpha-quartz (0001) substrate slab builder.

The orthorhombic surface cell (a x a*sqrt(3) x c) is derived from the
alpha-quartz structure of Tucker, Dove & Keen (Mineralogical Magazine
2001, 65, 489; COD 1526860), space group P3121 (No. 152):
a = 4.9019 A, c = 5.3988 A.  Each Si is tetrahedrally coordinated
(Si-O = 1.607-1.616 A) and every oxygen bridges two silicons.

The (0001) cleavage leaves two dangling oxygens on each surface silicon
(geminal Q2 termination, ~9.6 silanols/nm^2); the surface silicons are
~4.9 A apart, so Si-O-Si bridging cannot reduce this density — the
builder warns when a lower oh_density target cannot be met.

Created on 2026-08-12
@author: zwu
"""
from ..materials.crystals import (  # noqa: F401  (public constants)
    ORTHO_CELL_FRACTIONAL,
    QUARTZ_A,
    QUARTZ_B,
    QUARTZ_C,
)
from .base import SlabBuilder, SurfaceSlab

#: Backwards-compatible alias.
QuartzSlab = SurfaceSlab


class QuartzBuilder(SlabBuilder):
    """Hydroxylated alpha-quartz (0001) slab; see SlabBuilder for args."""

    CELL_FRACTIONAL = ORTHO_CELL_FRACTIONAL
    CELL_A = QUARTZ_A
    CELL_B = QUARTZ_B
    CELL_C = QUARTZ_C
    BUILDER_NAME = "alpha_quartz"
    ATOM_PREFIX = "qz"
    DEFAULT_CLASS_NAME = "QuartzSlab"
    DEFAULT_LT_FILENAME = "alpha_quartz_slab.lt"
