#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shape and surface specifications for nanoparticles.

Shapes define *which atoms belong to the object* (carved from the bulk
crystal of the selected phase); surfaces define *how the exposed atoms are
terminated* (delegated to the material's chemistry adapter).  Neither says
anything about interactions.

Created on 2026-11-13
@author: zwu
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..core.exceptions import ValidationError


@dataclass(frozen=True)
class Shape:
    """Base class of shape specifications."""
    name: str = ""
    geometry_type: str = "cluster"

    def to_dict(self) -> Dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items()}
        return out


@dataclass(frozen=True)
class Sphere(Shape):
    """
    Spherical particle.

    Args:
        radius: Core cut radius (A): every framework cation (Si, Al, Ti,
            metal atom) within this distance of the center is kept.
        center_offset: Optional shift of the particle center inside the
            crystal (A); varies which surface sites are exposed.
    """
    radius: float = 10.0
    center_offset: Optional[Tuple[float, float, float]] = None
    name: str = "sphere"
    geometry_type: str = "cluster"

    def __post_init__(self) -> None:
        if not self.radius > 0:
            raise ValidationError(f"Sphere radius must be > 0, got {self.radius}")


@dataclass(frozen=True)
class Polyhedron(Shape):
    """
    Faceted fcc metal cluster (optional ASE builders).

    Args:
        kind: "cuboctahedron", "octahedron" or "wulff".
        size: Approximate radius (A) used to choose the shell count / layer
            numbers.
        surface_energies: Wulff construction only: {(h, k, l): energy}
            (relative values); defaults to (111) < (100) ratios for fcc.
    """
    kind: str = "cuboctahedron"
    size: float = 10.0
    surface_energies: Optional[Dict[Tuple[int, int, int], float]] = None
    name: str = "polyhedron"
    geometry_type: str = "cluster"

    def __post_init__(self) -> None:
        if self.kind not in ("cuboctahedron", "octahedron", "wulff"):
            raise ValidationError(
                f"Polyhedron kind must be cuboctahedron/octahedron/wulff, "
                f"got '{self.kind}'"
            )
        if not self.size > 0:
            raise ValidationError(f"Polyhedron size must be > 0")


@dataclass(frozen=True)
class Sheet(Shape):
    """
    Finite sheet (graphene flake) in the xy plane.

    Args:
        length: Extent along x (A, zigzag direction).
        width: Extent along y (A, armchair direction).
        layers: Number of AB-stacked layers (3.35 A apart).
    """
    length: float = 30.0
    width: float = 30.0
    layers: int = 1
    name: str = "sheet"
    geometry_type: str = "sheet"

    def __post_init__(self) -> None:
        if not (self.length > 0 and self.width > 0):
            raise ValidationError("Sheet length and width must be > 0")
        if self.layers < 1:
            raise ValidationError("Sheet layers must be >= 1")


@dataclass(frozen=True)
class Tube(Shape):
    """
    Single-wall carbon nanotube along z.

    Args:
        n: First chiral index (n > 0).
        m: Second chiral index (0 <= m <= n).
        length: Tube length (A).
    """
    n: int = 10
    m: int = 10
    length: float = 30.0
    name: str = "tube"
    geometry_type: str = "rod"

    def __post_init__(self) -> None:
        if not (self.n > 0 and 0 <= self.m <= self.n):
            raise ValidationError(
                f"Tube chiral indices need n > 0 and 0 <= m <= n, got "
                f"({self.n}, {self.m})"
            )
        if not self.length > 0:
            raise ValidationError("Tube length must be > 0")


@dataclass(frozen=True)
class Imported(Shape):
    """Shape of an imported structure (no carving)."""
    source: str = ""
    name: str = "imported"
    geometry_type: str = "imported"


def make_shape(shape, **kwargs) -> Shape:
    """Shape object from a name (+ size kwargs) or a Shape instance."""
    if isinstance(shape, Shape):
        if kwargs:
            raise ValidationError(
                f"Size arguments {sorted(kwargs)} given together with a "
                f"{type(shape).__name__} object"
            )
        return shape
    name = str(shape).lower()
    if name == "sphere":
        if "radius" not in kwargs:
            raise ValidationError("shape='sphere' requires radius=...")
        return Sphere(radius=float(kwargs.pop("radius")),
                      center_offset=kwargs.pop("center_offset", None),
                      **_no_more(kwargs, name))
    if name in ("cuboctahedron", "octahedron", "wulff"):
        size = kwargs.pop("size", kwargs.pop("radius", None))
        if size is None:
            raise ValidationError(f"shape='{name}' requires radius=...")
        return Polyhedron(kind=name, size=float(size),
                          surface_energies=kwargs.pop("surface_energies", None),
                          **_no_more(kwargs, name))
    if name == "sheet":
        return Sheet(length=float(kwargs.pop("length", 30.0)),
                     width=float(kwargs.pop("width", 30.0)),
                     layers=int(kwargs.pop("layers", 1)),
                     **_no_more(kwargs, name))
    if name in ("tube", "cnt", "rod"):
        return Tube(n=int(kwargs.pop("n", 10)), m=int(kwargs.pop("m", 10)),
                    length=float(kwargs.pop("length", 30.0)),
                    **_no_more(kwargs, name))
    raise ValidationError(
        f"Unknown nanoparticle shape '{shape}'; use 'sphere', 'sheet', "
        "'tube', 'cuboctahedron', 'octahedron', 'wulff', or a Shape object "
        "(imported structures: Nanoparticle.from_file)"
    )


def _no_more(kwargs, name):
    if kwargs:
        raise ValidationError(
            f"Unexpected argument(s) {sorted(kwargs)} for shape '{name}'"
        )
    return {}


# ----------------------------------------------------------------------
# Surfaces
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Surface:
    """
    Surface-termination request.

    Args:
        state: Termination ID registered for the material ("hydroxylated",
            "stoichiometric", "h_passivated", "bare"); None = material
            default.
        params: Termination parameters (material specific).
    """
    state: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"state": self.state, "params": dict(self.params)}


def SilicaSurface(oh_density: Optional[float] = None,
                  state: str = "hydroxylated") -> Surface:
    """
    Silica termination: Q3/Q2 silanols on 4-coordinated Si.

    Args:
        oh_density: Target silanol density (sites/nm^2).  Applied by Si-O-Si
            condensation where geometrically possible; the achieved density
            is reported (crystal-derived spheres are intrinsically at ~5-6
            (cristobalite) or ~7-8 (quartz) /nm^2).  For imported
            structures the value is the expected density, checked within
            tolerance.
    """
    if oh_density is not None and oh_density < 0:
        raise ValidationError(f"oh_density must be >= 0, got {oh_density}")
    return Surface(state=state, params={"oh_density": oh_density})


def make_surface(surface) -> Surface:
    if surface is None:
        return Surface()
    if isinstance(surface, Surface):
        return surface
    if isinstance(surface, str):
        return Surface(state=surface)
    if isinstance(surface, dict):
        return Surface(state=surface.get("state"),
                       params=dict(surface.get("params", {})))
    raise ValidationError(f"Invalid surface specification {surface!r}")
