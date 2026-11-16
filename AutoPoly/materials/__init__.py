# -*- coding: utf-8 -*-
"""
Shared inorganic-material foundation (substrates and nanoparticles).

- structure.py:     AtomicStructure (force-field-agnostic geometry/topology)
- topology.py:      neighbor search, bond perception, PBC helpers
- crystals.py:      bulk crystal sources (orthorhombic cells)
- registry.py:      MaterialSpec registry (five families, six IDs)
- material.py:      Material (user-facing material selection)
- chemistry/:       material-specific roles, termination and validation
- forcefields/:     versioned inorganic parameter adapters + organic
                    compatibility checks / explicit cross terms
- qualification.py: md_ready qualification matrix
"""
from .structure import AtomicStructure, GEOMETRY_TYPES, element_mass
from . import forcefields  # noqa: F401  (registers built-in adapters)
from .registry import (
    CAPABILITY_STAGES,
    MATERIAL_REGISTRY,
    MaterialSpec,
    PhaseSpec,
    SurfaceStateSpec,
    get_material,
    register_material,
    registered_materials,
)
from .material import Material
from .qualification import (
    QUALIFIED_COMBINATIONS,
    QualifiedCombination,
    find_qualification,
    qualification_markdown,
    qualification_matrix,
    register_qualification,
    require_md_ready,
)

__all__ = [
    "AtomicStructure",
    "CAPABILITY_STAGES",
    "GEOMETRY_TYPES",
    "MATERIAL_REGISTRY",
    "Material",
    "MaterialSpec",
    "PhaseSpec",
    "QUALIFIED_COMBINATIONS",
    "QualifiedCombination",
    "SurfaceStateSpec",
    "element_mass",
    "find_qualification",
    "get_material",
    "qualification_markdown",
    "qualification_matrix",
    "register_material",
    "register_qualification",
    "registered_materials",
    "require_md_ready",
]
