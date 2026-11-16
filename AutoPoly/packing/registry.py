#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registry for AutoPoly box-packing strategies.

Built-in strategies ("grid", "mc_random") are registered when the
AutoPoly.packing package is imported. User-defined strategies are importable
PlacementStrategy subclasses registered at runtime:

    >>> from AutoPoly.packing import PlacementStrategy, register_strategy
    >>> class MyStrategy(PlacementStrategy):
    ...     name = "my_strategy"
    ...     def place(self, ctx):
    ...         ...
    >>> register_strategy("my_strategy", MyStrategy)

Created on 2026-07-30
@author: zwu
"""
from typing import Dict, List, Type

from ..core.exceptions import ValidationError
from .base import PlacementStrategy

#: Names reserved for future built-in strategies (grafting, confinement,
#: filled films on substrates).  Registering these is rejected so user
#: code cannot shadow planned built-ins.
RESERVED_STRATEGY_NAMES = ("grafted_surface", "nanopore",
                           "nanocomposite_on_substrate")

_STRATEGY_REGISTRY: Dict[str, Type[PlacementStrategy]] = {}


def register_strategy(name: str, cls: Type[PlacementStrategy]) -> None:
    """
    Register a placement strategy class under `name`.

    Args:
        name: Strategy name used by BoxPacker(strategy=name).
        cls: PlacementStrategy subclass (not an instance).

    Raises:
        ValidationError: If the name is reserved, the class is not a
                         PlacementStrategy subclass, or its `name` attribute
                         mismatches.
    """
    if name in RESERVED_STRATEGY_NAMES:
        raise ValidationError(
            f"Strategy name '{name}' is reserved for a future built-in "
            f"strategy. Reserved names: {list(RESERVED_STRATEGY_NAMES)}"
        )
    if not (isinstance(cls, type) and issubclass(cls, PlacementStrategy)):
        raise ValidationError(
            f"Strategy class for '{name}' must be a PlacementStrategy subclass, "
            f"got {cls!r}"
        )
    if not cls.name:
        raise ValidationError(
            f"Strategy class {cls.__name__} must define a non-empty 'name' "
            "class attribute"
        )
    _STRATEGY_REGISTRY[name] = cls


def get_strategy(name: str) -> PlacementStrategy:
    """
    Instantiate the strategy registered under `name`.

    Raises:
        ValidationError: If no strategy is registered under that name.
    """
    try:
        cls = _STRATEGY_REGISTRY[name]
    except KeyError:
        raise ValidationError(
            f"Unknown packing strategy '{name}'. "
            f"Registered strategies: {registered_strategies()}"
        ) from None
    return cls()


def registered_strategies() -> List[str]:
    """Return the sorted list of registered strategy names."""
    return sorted(_STRATEGY_REGISTRY)
