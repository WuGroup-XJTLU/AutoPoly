# -*- coding: utf-8 -*-
"""
AutoPoly Reactor — LAMMPS fix bond/react preparation

Adds AutoREACTER-style reactive-MD setup to AutoPoly: functional-group
detection on the packed monomers, reaction enumeration from a SMARTS
library, pre/post molecule template + map file construction, and a
ready-to-run ``in.bond_react`` script.

Public API:

- :class:`Reactor` — orchestrator bound to a generated project directory.
- :class:`ReactorResult` — build output manifest.
- :func:`detect_monomer_roles`, :func:`detect_reactions`,
  :func:`prepare_reactions` — stage-level helpers for custom workflows.

Portions of this package (``functional_groups``, ``reaction_library``,
``detector``, ``mapper``, and ``reactor`` modules) contain code copied or
adapted from the AutoREACTER project
(https://github.com/NanoCIPHER-Lab/AutoREACTER), Copyright (c) 2026 Janitha
Mahanthe, used under the MIT License. The AutoREACTER copyright notice and
full license text are reproduced in the ``LICENSE`` file in this directory
and in the header comments of the affected modules.

Created on 2026-08-04
@author: zwu
"""

from .reactor import Reactor, ReactorResult
from .functional_groups import (
    FUNCTIONAL_GROUPS,
    FunctionalGroupInfo,
    MonomerRole,
    detect_functional_groups,
    detect_monomer_roles,
)
from .reaction_library import REACTIONS, get_reaction_library
from .detector import ReactionInstance, detect_reactions
from .mapper import ReactionMetadata, ReactionMappingError, prepare_reactions
from .templates import TemplateBuilder, TemplateSet, TypeAssignment
from .fftables import ForceFieldTables, load_force_field_tables
from .system_data import SystemData, parse_in_init
from .lammps import write_bond_react_script, write_settings_reactor

__all__ = [
    "Reactor",
    "ReactorResult",
    "FUNCTIONAL_GROUPS",
    "FunctionalGroupInfo",
    "MonomerRole",
    "detect_functional_groups",
    "detect_monomer_roles",
    "REACTIONS",
    "get_reaction_library",
    "ReactionInstance",
    "detect_reactions",
    "ReactionMetadata",
    "ReactionMappingError",
    "prepare_reactions",
    "TemplateBuilder",
    "TemplateSet",
    "TypeAssignment",
    "ForceFieldTables",
    "load_force_field_tables",
    "SystemData",
    "parse_in_init",
    "write_bond_react_script",
    "write_settings_reactor",
]
