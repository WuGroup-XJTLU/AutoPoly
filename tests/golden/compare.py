#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physical-equivalence comparison of LAMMPS/moltemplate artifacts.

Used by the legacy-substrate golden tests: files are compared by their
parsed physical content (box, masses, atoms with charges/coordinates,
bonded topology, pair/bond/angle/dihedral coefficients) with a numeric
tolerance instead of byte equality, so harmless formatting changes do not
break the gate but any physical change does.
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DATA_SECTIONS = ("Masses", "Atoms", "Bonds", "Angles", "Dihedrals",
                 "Impropers")


def _floats(tokens):
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except ValueError:
            out.append(t)
    return out


def parse_data(path: Path) -> Dict[str, Any]:
    """Parse a LAMMPS data file into header counts, box and sections."""
    lines = Path(path).read_text().splitlines()
    out: Dict[str, Any] = {"counts": {}, "box": {}, "sections": {}}
    section = None
    for raw in lines[1:]:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 1 and parts[0] in DATA_SECTIONS:
            section = parts[0]
            out["sections"][section] = []
            continue
        if section is None:
            if parts[-1] in ("xhi", "yhi", "zhi"):
                out["box"][parts[-1][0]] = (float(parts[0]), float(parts[1]))
            elif parts[0].isdigit():
                out["counts"][" ".join(parts[1:])] = int(parts[0])
            continue
        out["sections"][section].append(_floats(parts))
    return out


def parse_coeffs(path: Path) -> Dict[str, List[list]]:
    """Coefficient lines of a system.in.settings, keyed by command."""
    out: Dict[str, List[list]] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0].endswith("_coeff"):
            out.setdefault(parts[0], []).append(_floats(parts[1:]))
    for key in out:
        out[key].sort(key=lambda r: [str(x) for x in r[:2]])
    return out


def _close(a, b, tol):
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b


def compare_rows(name: str, rows_a, rows_b, tol: float = 1e-6) -> List[str]:
    """Differences between two lists of parsed rows (same order)."""
    diffs = []
    if len(rows_a) != len(rows_b):
        return [f"{name}: {len(rows_a)} rows vs {len(rows_b)}"]
    for k, (ra, rb) in enumerate(zip(rows_a, rows_b)):
        if len(ra) != len(rb) or not all(
                _close(x, y, tol) for x, y in zip(ra, rb)):
            diffs.append(f"{name} row {k}: {ra} != {rb}")
            if len(diffs) > 5:
                diffs.append(f"{name}: ...")
                break
    return diffs


def compare_data(a: Path, b: Path, tol: float = 1e-6) -> List[str]:
    """Physical differences between two data files ([] if equivalent)."""
    pa, pb = parse_data(a), parse_data(b)
    diffs = []
    if pa["counts"] != pb["counts"]:
        diffs.append(f"header counts {pa['counts']} != {pb['counts']}")
    for axis in "xyz":
        ba, bb = pa["box"].get(axis), pb["box"].get(axis)
        if ba is None or bb is None or not all(
                _close(x, y, tol) for x, y in zip(ba, bb)):
            diffs.append(f"box {axis}: {ba} != {bb}")
    for sec in DATA_SECTIONS:
        diffs += compare_rows(sec, pa["sections"].get(sec, []),
                              pb["sections"].get(sec, []), tol)
    return diffs


def compare_coeffs(a: Path, b: Path, tol: float = 1e-6,
                   scale: Optional[Dict[Tuple[str, str], float]] = None
                   ) -> List[str]:
    """
    Differences between the coefficient lines of two settings files.

    scale: optional {(command, first-type-token): factor} applied to the
    last numeric column of the *expected* file ``a`` before comparing
    (documented intentional parameter changes, e.g. a sigma fix).
    """
    ca, cb = parse_coeffs(a), parse_coeffs(b)
    diffs = []
    if set(ca) != set(cb):
        diffs.append(f"coeff commands {sorted(ca)} != {sorted(cb)}")
    for key in sorted(set(ca) & set(cb)):
        rows_a = [list(r) for r in ca[key]]
        if scale:
            for r in rows_a:
                factor = scale.get((key, str(int(r[0])) if isinstance(
                    r[0], float) else str(r[0])))
                if factor is not None:
                    r[-1] = r[-1] * factor
        diffs += compare_rows(key, rows_a, cb[key], tol)
    return diffs


def normalized_text(path: Path) -> str:
    """File text with whitespace runs collapsed (for scripts)."""
    return "\n".join(
        re.sub(r"\s+", " ", line.strip())
        for line in Path(path).read_text().splitlines() if line.strip()
    )
