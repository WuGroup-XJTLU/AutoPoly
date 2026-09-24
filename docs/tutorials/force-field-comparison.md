# Force Field Comparison

The same polymer — poly(ethylene oxide), 10 chains of 10 monomers — built with **all six built-in force fields** in one script. This is the fastest way to see how force field choice changes the generated files.

| Force field | Notes |
|---|---|
| `oplsaa` | Standard for vinyl polymers |
| `lopls` | Liquid-optimized OPLS |
| `gaff` | General AMBER force field |
| `gaff2` | Updated GAFF |
| `dreiding` | Generic; requires external charges for production |
| `compass` | Class II; requires LAMMPS built with the CLASS2 package |
| `openff` | SMIRNOFF (Sage) + NAGL GNN charges; optional install (`conda install -c conda-forge openff-toolkit openff-nagl`), see `example_peo_openff.py` |

You will learn:

- How the `force_field` string changes atom typing and the `system.in.settings` output
- Which force fields need follow-up charge work (see [Force Fields](../guides/force-fields.md))

## The script

```python
--8<-- "examples/example_peo_all_forcefields.py"
```

## Run it

```bash
cd examples
python example_peo_all_forcefields.py
```

## What to compare

After the run, diff the outputs across the six directories:

- **`system.in.settings`** — different `pair_coeff`/`bond_coeff` sets per force field
- **`system.in.init`** — Class II force fields (COMPASS) select different angle/dihedral styles
- **`system.in.charges`** — OPLS-AA charges come from the parameter set; GAFF/GAFF2 use Gasteiger charges
