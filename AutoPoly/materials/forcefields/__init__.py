# -*- coding: utf-8 -*-
"""
Inorganic force-field adapters (versioned parameter sets).

Importing this package registers the built-in adapters.
"""
from .registry import (
    COMBINING_RULES,
    MOTIONS,
    InorganicForceField,
    ParameterAssignment,
    RoleParameters,
    get_inorganic_ff,
    register_inorganic_ff,
    registered_inorganic_ffs,
)
from . import silica  # noqa: F401  (registers interface_silica, clayff_silica)
from . import others  # noqa: F401  (alumina, carbon, Au/Ag adapters)

__all__ = [
    "COMBINING_RULES",
    "MOTIONS",
    "InorganicForceField",
    "ParameterAssignment",
    "RoleParameters",
    "get_inorganic_ff",
    "register_inorganic_ff",
    "registered_inorganic_ffs",
]
