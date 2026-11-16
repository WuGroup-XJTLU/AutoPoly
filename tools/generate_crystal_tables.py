#!/usr/bin/env python3
"""
Development helper: (re)generate the orthorhombic bulk cells embedded in
AutoPoly/materials/crystals.py from space-group descriptions (requires ASE;
AutoPoly itself does not).  Prints Python tables and a coordination check.

    python tools/generate_crystal_tables.py
"""
import numpy as np
from ase.spacegroup import crystal
from ase.build import make_supercell
from ase.neighborlist import neighbor_list

SOURCES = {
    # name: (symbols, basis, spacegroup, cellpar, setting, ortho P matrix, cutoff, citation)
    "alpha_corundum": (["Al", "O"], [(0, 0, 0.35216), (0.30624, 0, 0.25)], 167,
                       [4.7602, 4.7602, 12.9933, 90, 90, 120], 1,
                       [[1, 0, 0], [1, 2, 0], [0, 0, 1]], 2.1,
                       "Lewis, Schwarzenbach & Flack, Acta Cryst. A 1982, 38, 733"),
    "rutile": (["Ti", "O"], [(0, 0, 0), (0.30483, 0.30483, 0)], 136,
               [4.5937, 4.5937, 2.9587, 90, 90, 90], 1, None, 2.2,
               "Burdett et al., J. Am. Chem. Soc. 1987, 109, 3639"),
    "anatase": (["Ti", "O"], [(0, 0, 0), (0, 0, 0.20806)], 141,
                [3.7845, 3.7845, 9.5143, 90, 90, 90], 1, None, 2.2,
                "Burdett et al., J. Am. Chem. Soc. 1987, 109, 3639"),
}

for name, (sym, basis, sg, cellpar, setting, P, cut, cite) in SOURCES.items():
    at = crystal(sym, basis=basis, spacegroup=sg, cellpar=cellpar, setting=setting)
    if P is not None:
        at = make_supercell(at, P)
    cell = at.cell.array
    assert np.allclose(cell, np.diag(np.diag(cell)), atol=1e-8), (name, cell)
    i, j, d = neighbor_list("ijd", at, cut)
    cn = np.bincount(i, minlength=len(at))
    print(f"# {name}: {len(at)} atoms, cell {np.round(np.diag(cell), 6)}; {cite}")
    for el in sorted(set(at.get_chemical_symbols())):
        idx = [k for k, s in enumerate(at.get_chemical_symbols()) if s == el]
        print(f"#   {el}: CN {sorted(set(cn[idx]))}, d range "
              f"{d[np.isin(i, idx)].min():.4f}-{d[np.isin(i, idx)].max():.4f}")
    frac = at.get_scaled_positions(wrap=True)
    print(f"{name.upper()}_CELL = ({', '.join(f'{x:.6f}' for x in np.diag(cell))})")
    print(f"{name.upper()}_FRACTIONAL = (")
    for s, f in sorted(zip(at.get_chemical_symbols(), frac), key=lambda t: (t[0] != sym[0], t[1][2], t[1][1], t[1][0])):
        print(f'    ("{s}", ({f[0]:.8f}, {f[1]:.8f}, {f[2]:.8f})),')
    print(")\n")
