#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crystalline beta-cristobalite (111) substrate slab builder.

beta-cristobalite (cubic, Fd-3m, a = 7.16 A; ideal average structure,
COD 1010944): Si on diamond sites, oxygens displaced 0.435 A off the
Si-Si axis (the 96g-style disorder of the real material), giving Si-O =
1.61 A and Si-O-Si = 148.7 deg at the correct density.  Because the
displacement pattern breaks the face-centering, the surface cell is the
simple-cubic (111) cell a*sqrt(2) x a*sqrt(6) x a*sqrt(3) holding
144 atoms (48 Si + 96 O).

The (111) cleavage runs between the diamond-net bilayers and leaves one
dangling oxygen on each surface silicon (isolated Q3 silanols,
~4.5 sites/nm^2) — close to the experimental silanol density of
amorphous silica (Zhuravlev 2000).  The automatic cleavage search in
SlabBuilder selects the bilayer-boundary cut.

Created on 2026-08-12
@author: zwu
"""
from ..materials.crystals import (  # noqa: F401  (public constants)
    CRISTOBALITE_111_A,
    CRISTOBALITE_111_B,
    CRISTOBALITE_111_C,
    CRISTOBALITE_111_FRACTIONAL,
    CRISTOBALITE_A,
)
from .base import SlabBuilder, SurfaceSlab

#: Backwards-compatible alias.
CristobaliteSlab = SurfaceSlab


class CristobaliteBuilder(SlabBuilder):
    """Hydroxylated beta-cristobalite (111) slab; see SlabBuilder."""

    CELL_FRACTIONAL = CRISTOBALITE_111_FRACTIONAL
    CELL_A = CRISTOBALITE_111_A
    CELL_B = CRISTOBALITE_111_B
    CELL_C = CRISTOBALITE_111_C
    BUILDER_NAME = "beta_cristobalite"
    ATOM_PREFIX = "cb"
    DEFAULT_CLASS_NAME = "CristobaliteSlab"
    DEFAULT_LT_FILENAME = "beta_cristobalite_slab.lt"
