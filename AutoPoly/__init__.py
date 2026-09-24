# -*- coding: utf-8 -*-
"""
AutoPoly: Automated Polymer Generation and Simulation Package

This package provides tools for generating polymer structures and preparing them
for molecular dynamics simulations using LAMMPS and Moltemplate.

Main Components:
- System: Utility class for managing file paths and system operations
- Polymer: Class for defining polymer properties and sequences
- Molecule: Class for defining small molecule structures (water, benzene, etc.)
- generate: One-shot convenience function composing the three pipeline stages
- BeadSpringPolymer: Simplified bead-spring polymer model generator
- mc: Monte Carlo placement module for chain growth and molecular placement

Three-Stage Pipeline (Geometry -> Typing -> Packing):
- GeometryBuilder: Stage 1, force-field-agnostic coordinates -> geometry.json
- UnitTyper: Stage 2, force-field assignment -> build/<ff>/ + units.json
- BoxPacker: Stage 3, box packing + moltemplate -> system.data
- UnitLibrary: units.json manifest contract between stages 2 and 3
- PlacementStrategy / register_strategy: pluggable packing strategies

External Dependencies:
- Moltemplate: For generating LAMMPS data files from molecular templates
- LAMMPS: Molecular dynamics simulation engine
- OPLS-AA: Force field parameters for atomistic simulations

Created on Fri Dec 21 12:19:08 2018
@author: zwu
"""

__author__ = "Zhenghao Wu"
__license__ = "BSD License"
__version__ = "2.0.0"
__description__ = "Automated Polymer Generation and Simulation Package"

# Explicit exports
__all__ = [
    "System",
    "Polymer",
    "Molecule",
    "generate",
    "BeadSpringPolymer",
    "BeadSpringSystem",
    "BeadType",
    "AngleType",
    "MCConfig",
    "SAWConfig",
    "BeadArchitecture",
    "MonomerTemplate",
    "architectures",
    "block_sequence",
    "alternating_sequence",
    "random_sequence",
    "gradient_sequence",
    "MonomerGenerator",
    "GeometryBuilder",
    "GeometryConfig",
    "UnitTyper",
    "BoxPacker",
    "UnitLibrary",
    "UnitSpec",
    "PlacementStrategy",
    "SubstrateSpec",
    "CutAbove",
    "CutBelow",
    "Cylinder",
    "BoxRegion",
    "register_strategy",
    "get_strategy",
    "Reactor",
    "ReactorResult",
    "OpenFFTyper",
    "mc",
    "__version__",
    "__author__",
]

# Always available imports (no rdkit dependency)
from .core.system import System
from .models.bead_spring import BeadSpringPolymer, BeadType, AngleType, MCConfig, SAWConfig
from .models.bead_spring_system import BeadSpringSystem
from .models.architectures import (
    BeadArchitecture,
    MonomerTemplate,
    block_sequence,
    alternating_sequence,
    random_sequence,
    gradient_sequence,
)
from .models import architectures

# Optional imports that require rdkit
try:
    from .models.polymer import Polymer
    from .models.molecule import Molecule
    from .monomers.monomer_generator import MonomerGenerator
    from .pipeline import (
        GeometryBuilder,
        GeometryConfig,
        UnitTyper,
        BoxPacker,
        UnitLibrary,
        UnitSpec,
        generate,
    )
    from .packing import (
        PlacementStrategy,
        SubstrateSpec,
        CutAbove,
        CutBelow,
        Cylinder,
        BoxRegion,
        register_strategy,
        get_strategy,
    )
    from .reactor import Reactor, ReactorResult
    from . import mc
    # OpenFFTyper imports the openff packages lazily (only when used), so the
    # class itself is always importable; parameterization without the optional
    # openff-toolkit installed raises GenerationError with install hints.
    from .forcefields.openff_typing import OpenFFTyper
except ImportError as e:
    import warnings
    warnings.warn(
        f"Some AutoPoly components require rdkit: {e}. "
        "Install rdkit for full functionality. "
        "BeadSpringPolymer and System are still available."
    )
    Polymer = None
    Molecule = None
    MonomerGenerator = None
    generate = None
    Reactor = None
    ReactorResult = None
    OpenFFTyper = None
    mc = None
