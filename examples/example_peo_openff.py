#!/usr/bin/env python3
"""
OpenFF (SMIRNOFF) Force Field Example

Builds a small PEO melt typed with the OpenFF Sage force field, with partial
charges from the NAGL graph neural network (AM1-BCC quality).

OpenFF is an OPTIONAL backend and is not installed with AutoPoly by default:

    conda install -c conda-forge openff-toolkit openff-nagl   # recommended
    # (openff-nagl needs Python <= 3.12 because of its DGL dependency;
    #  on Python 3.13 use charge_method="gasteiger" with openff-toolkit only)

What happens differently from the type-based force fields (oplsaa, gaff, ...):
the OpenFF toolkit assigns parameters per molecule (direct chemical
perception), so the build directory contains a generated per-system
``openff.lt`` parameter file and every bonded term is written explicitly in
the monomer/chain .lt files (no "By Type" inference).

Requires: pip install -e .  (from the AutoPoly repo root)
"""

from AutoPoly import System, Polymer, generate

# PEO configuration (same complement SMILES as the other examples)
DOP = 10
SEQUENCE = ["CCO[*]"] + ["[*]CCO[*]"] * (DOP - 2) + ["[*]CCO"]


def main():
    system = System(out="peo_openff_out")

    polymer = Polymer(
        chain_num=10,
        sequence=SEQUENCE,
        topology="linear",
        tacticity="atactic",
    )

    generate(
        system,
        "peo_openff",
        [polymer],
        force_field="openff",
        # Optional: override the SMIRNOFF force field or the charge backend
        # (defaults: newest bundled Sage + NAGL GNN charges):
        # typer_options={
        #     "offxml": "openff-2.2.1.offxml",
        #     "charge_method": "nagl",        # "am1bcc" | "gasteiger"
        #     "nagl_model": "openff-gnn-am1bcc-1.0.0.pt",
        # },
        box_size=45.0,
        rng_seed=42,
    )
    print("Done — see peo_openff_out/peo_openff/ (system.data, system.in.*)")


if __name__ == "__main__":
    main()
