#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration Module for AutoPoly Package

This module contains configuration settings for the AutoPoly package including
logging configuration and output path settings.

The configuration includes:
- Output path settings for generated files
- Logging level configurations
- File output settings
- Resource limits to prevent DoS attacks

Configuration can be modified by changing the values in the LOG dictionary
or by setting the OUT_PATH variable.
"""
import logging
import os

# Output path configuration
OUT_PATH = os.path.join(os.path.expanduser("~"))

# Logging configuration dictionary
LOG = {
    'ROOT_LEVEL': logging.INFO,      # Root logger level
    'CONSOLE_LEVEL': logging.INFO,   # Console output level
    'FILE_LEVEL': logging.INFO,      # File output level
    'TO_FILE': False                 # Whether to log to file (True/False)
}

# Resource limits to prevent DoS attacks
# These limits prevent resource exhaustion from excessively large values
MAX_DOP = 10000  # Maximum degree of polymerization
MAX_SEQUENCE_LENGTH = 10000  # Maximum sequence length
MAX_UNIQUE_MONOMERS = 100  # Maximum unique monomer types

# Metadata for agent API discovery
FORCE_FIELD_DESCRIPTIONS = {
    "oplsaa": "OPLS-AA. Best for organic molecules and proteins.",
    "lopls": "L-OPLS. Better for long alkyl chains.",
    "gaff": "General AMBER Force Field. Broad organic coverage.",
    "gaff2": "GAFF2. Updated GAFF with improved parameters.",
    "dreiding": "DREIDING. Generic, element-based. Good for unusual chemistry.",
    "compass": "COMPASS. Optimized for condensed-phase properties.",
    "openff": "OpenFF SMIRNOFF (Sage). Direct chemical perception; NAGL GNN "
              "charges by default. Optional: requires openff-toolkit.",
}

# Central registry of per-force-field file/class mappings.
# Single source of truth for the lt parameter file, moltemplate class to
# inherit, and the RDlt .fdefn SMARTS file used for atom typing.
FORCE_FIELD_REGISTRY = {
    "oplsaa": {
        "lt_file": "oplsaa.lt",
        "inherits": "OPLSAA",
        "fdefn": "opls_lt_2024.fdefn",  # OPLS-AA 2024 numbering (moltemplate 2.22.5)
    },
    "lopls": {
        "lt_file": "loplsaa.lt",
        "inherits": "OPLSAA",  # LOPLS extends OPLSAA
        "fdefn": "lopls_lt.fdefn",
    },
    "gaff": {
        "lt_file": "gaff.lt",
        "inherits": "GAFF",
        "fdefn": "gaff_lt.fdefn",
    },
    "gaff2": {
        "lt_file": "gaff2.lt",
        "inherits": "GAFF2",
        "fdefn": "gaff_lt.fdefn",
    },
    "dreiding": {
        "lt_file": "dreiding.lt",
        "inherits": "DREIDING",
        "fdefn": "dreiding_lt.fdefn",
    },
    "compass": {
        "lt_file": "compass_published.lt",
        "inherits": "COMPASS",
        "fdefn": "compass_lt.fdefn",
    },
    # OpenFF is different in kind: no static .lt ships in extern/ and no
    # .fdefn SMARTS table is used. Typing runs through the openff-toolkit
    # (label_molecules) in pipeline/typing.py, which *generates* a per-system
    # openff.lt into the build directory (see forcefields/openff_typing.py).
    "openff": {
        "lt_file": "openff.lt",
        "inherits": "OPENFF",
        "fdefn": None,
    },
}

VALID_FORCE_FIELDS = list(FORCE_FIELD_REGISTRY.keys())

TOPOLOGY_DESCRIPTIONS = {
    "linear": "Linear chain. Distinct first/last monomers.",
    "ring": "Ring (cyclic). All monomers are middle-type, chain closes on itself.",
}

TACTICITY_DESCRIPTIONS = {
    "atactic": "Random stereochemistry at each chiral center.",
    "isotactic": "Same stereochemistry at all chiral centers.",
    "syndiotactic": "Alternating stereochemistry at chiral centers.",
}

EXAMPLE_CONFIGS = {
    "polyethylene_10mer": {
        "type": "atomistic",
        "name": "pe_system",
        "force_field": "oplsaa",
        "polymers": [{
            "chain_num": 2,
            "sequence": ["CC[*]"] + ["[*]CC[*]"] * 8 + ["[*]CC"],
            "topology": "linear",
            "tacticity": "atactic",
        }],
    },
    "diblock_copolymer": {
        "type": "atomistic",
        "name": "diblock",
        "force_field": "gaff2",
        "polymers": [{
            "chain_num": 5,
            "sequence": ["CC[*]"] + ["[*]CC[*]"] * 4
                        + ["[*]C(=O)O[*]"] * 4 + ["[*]C(=O)O"],
            "topology": "linear",
        }],
    },
    "bead_spring_melt": {
        "type": "bead_spring",
        "name": "melt",
        "n_chains": 100,
        "bead_types": [{"name": "A"}],
        "sequence": [["A", 50]],
        "bond_style": "fene",
        "pair_style": "wca",
        "density": 0.85,
    },
    "ring_polymer": {
        "type": "bead_spring",
        "name": "ring",
        "n_chains": 50,
        "bead_types": [{"name": "A"}],
        "sequence": [["A", 30]],
        "topology": "ring",
        "bond_style": "fene",
        "pair_style": "wca",
        "generation_method": "saw",
    },
}