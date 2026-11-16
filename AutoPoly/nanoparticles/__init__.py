# -*- coding: utf-8 -*-
"""
Nanoparticles and nanocomposite fillers.

Public API:

- Nanoparticle: a finite inorganic component (material, phase, shape,
  surface, count, motion, inorganic force field) passed to generate()
  next to Polymer/Molecule models; Nanoparticle.from_file() imports a
  validated cluster.
- Shapes: Sphere, Sheet, Tube, Polyhedron.
- Surfaces: Surface, SilicaSurface.
- CGFiller: coarse-grained LJ-bead filler for the bead-spring backend
  (BeadSpringSystem.add_filler).

Geometry is built in stage 1 from the material registry
(AutoPoly.materials), typed in stage 2 by the registered inorganic
force-field adapter (nanoparticles.typing), and placed in stage 3 by the
"nanocomposite_mc" packing strategy.
"""
from .cg import CGFiller
from .grafting import Grafting
from .model import Nanoparticle
from .shapes import (
    Polyhedron,
    Shape,
    Sheet,
    SilicaSurface,
    Sphere,
    Surface,
    Tube,
)

__all__ = [
    "CGFiller",
    "Grafting",
    "Nanoparticle",
    "Polyhedron",
    "Shape",
    "Sheet",
    "SilicaSurface",
    "Sphere",
    "Surface",
    "Tube",
]
