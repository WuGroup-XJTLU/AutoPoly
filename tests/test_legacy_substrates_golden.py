#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0-R regression gate: built-in silica substrates stay physically
equivalent to the pre-refactoring reference (tests/golden/).

- Every slab builder configuration in tests/golden/make_golden.py must
  reproduce its recorded .lt file (byte for byte) and SurfaceSlab
  attributes.
- The two recorded film-on-substrate pipelines must reproduce the same
  system.data (atoms, charges, coordinates, topology, box), the same
  pair/bond/angle/dihedral coefficients, the same in.init / in.run, and --
  when a LAMMPS binary is on PATH -- the same ``run 0`` energies.

Intentional output changes are listed in INTENTIONAL_CHANGES with the
CHANGELOG entry that documents them.
"""
import json
import re
import sys
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "golden"
sys.path.insert(0, str(GOLDEN))

from make_golden import (  # noqa: E402
    DATA,
    PIPELINE_CASES,
    PIPELINE_FILES,
    SLAB_CASES,
    lammps_binary,
    run_lammps_energies,
    run_pipeline,
)
from compare import compare_coeffs, compare_data, normalized_text  # noqa: E402

from AutoPoly.surfaces import SLAB_BUILDERS  # noqa: E402

#: Documented intentional deviations from the recorded reference.
#: case -> {"lt_sigma_scale": {type: factor}, "settings_scale":
#: {(command, type-id): factor}, "energies_unchanged": [thermo keys that
#: must still match] (others are allowed to change)}
_TWO_POW_1_6 = 2.0 ** (1.0 / 6.0)
INTENTIONAL_CHANGES = {
    # CHANGELOG "Fixed": CLAYFF sigma was 2^(1/6) too small (sigma passed
    # as R0).  Only the CLAYFF Si/O pair_coeff sigmas and the LJ energy
    # change; coordinates, topology, charges, Coulomb and bonded terms
    # are unchanged.
    "quartz_clayff": {
        "lt_sigma_scale": {"cff_st": _TWO_POW_1_6, "cff_ob": _TWO_POW_1_6,
                           "cff_oh": _TWO_POW_1_6},
    },
    "pe_on_cristobalite_oplsaa_clayff": {
        # types 5/6/7 = cff_ob/cff_oh/cff_st (golden Masses section)
        "settings_scale": {("pair_coeff", "5"): _TWO_POW_1_6,
                           ("pair_coeff", "6"): _TWO_POW_1_6,
                           ("pair_coeff", "7"): _TWO_POW_1_6},
        "energies_unchanged": ["ecoul", "elong", "ebond", "eangle",
                               "edihed", "eimp"],
    },
}


def _slab_kwargs(kwargs):
    out = {}
    for k, v in kwargs.items():
        if k == "slab_lj":
            v = {r: tuple(x) for r, x in v.items()}
        out[k] = v
    return out


@pytest.mark.parametrize("name", sorted(SLAB_CASES))
def test_slab_matches_golden(name):
    builder, kwargs = SLAB_CASES[name]
    slab = SLAB_BUILDERS[builder](**_slab_kwargs(kwargs)).build()
    meta = json.loads((DATA / "slabs" / f"{name}.json").read_text())

    assert slab.class_name == meta["class_name"]
    assert slab.lt_filename == meta["lt_filename"]
    assert slab.lx == pytest.approx(meta["lx"], abs=1e-9)
    assert slab.ly == pytest.approx(meta["ly"], abs=1e-9)
    assert slab.z_extent == pytest.approx(meta["z_extent"], abs=1e-9)
    assert len(slab.atoms) == meta["n_atoms"]
    assert len(slab.bonds or []) == meta["n_bonds"]
    assert slab.n_silanol_top == meta["n_silanol_top"]
    assert slab.n_silanol_bottom == meta["n_silanol_bottom"]
    assert {r: sum(a.role == r for a in slab.atoms)
            for r in ("Si", "OB", "OH", "HO")} == meta["roles"]
    assert sum(a.charge for a in slab.atoms) == pytest.approx(
        meta["net_charge"], abs=1e-9)

    expected = (DATA / "slabs" / f"{name}.lt").read_text()
    change = INTENTIONAL_CHANGES.get(name, {})
    if "lt_sigma_scale" not in change:
        assert slab.lt_text() == expected
    else:
        _assert_lt_equal_with_sigma_scale(
            expected, slab.lt_text(), change["lt_sigma_scale"])


def _assert_lt_equal_with_sigma_scale(expected, actual, scale):
    """Line-by-line equality except documented pair_coeff sigma changes."""
    exp_lines = expected.splitlines()
    act_lines = actual.splitlines()
    assert len(exp_lines) == len(act_lines)
    for e, a in zip(exp_lines, act_lines):
        ep, ap = e.split(), a.split()
        if ep[:1] == ["pair_coeff"] and ep[1].split(":")[1] in scale:
            factor = scale[ep[1].split(":")[1]]
            assert ep[:-1] == ap[:-1]
            assert float(ap[-1]) == pytest.approx(
                float(ep[-1]) * factor, rel=1e-5)
        else:
            assert e == a


def test_slab_structure_is_ff_agnostic_and_round_trips():
    """The SurfaceSlab carries its AtomicStructure + ParameterAssignment."""
    from AutoPoly.materials import AtomicStructure

    builder, kwargs = SLAB_CASES["quartz_interface"]
    slab = SLAB_BUILDERS[builder](**kwargs).build()
    s = slab.structure
    assert s is not None and s.geometry_type == "slab"
    assert s.pbc == (True, True, False)
    # no force-field data in the structure
    assert not any(k in s.to_dict() for k in ("charges", "types", "lj"))
    back = AtomicStructure.from_dict(json.loads(json.dumps(s.to_dict())))
    assert back.bonds == s.bonds
    assert (back.images == s.images).all()
    # periodic bonds crossing the lateral edges carry image shifts and
    # have physical lengths
    assert (s.images != 0).any()
    assert s.bond_lengths().max() < 1.9
    assert slab.assignment.net_charge == pytest.approx(0.0, abs=1e-9)


@pytest.mark.integration
@pytest.mark.parametrize("name", sorted(PIPELINE_CASES))
def test_pipeline_matches_golden(name, tmp_path):
    import rdkit

    project = run_pipeline(PIPELINE_CASES[name], tmp_path / "out", name)
    gold = DATA / "pipelines" / name
    change = INTENTIONAL_CHANGES.get(name, {})
    golden_e = json.loads((gold / "energies.json").read_text())
    # The film's monomer conformers come from RDKit embedding and change
    # between RDKit releases: with a different RDKit than the one the
    # reference was recorded with, film coordinates (and therefore the
    # LAMMPS energies) are not compared; the slab coordinates, every type,
    # charge and topology row, and the coefficient files still are.
    same_rdkit = golden_e.get("rdkit") == rdkit.__version__
    slab_mols = {int(m) for m in re.findall(
        r"group\s+slab\s+molecule\s+([\d ]+)", (gold / "in.run").read_text())
        for m in m.split()}
    assert slab_mols, "golden in.run defines no slab molecule group"

    diffs = compare_data(gold / "system.data", project / "system.data",
                         tol=1e-6,
                         coordinates_only_for=None if same_rdkit else slab_mols)
    assert not diffs, "\n".join(diffs)
    # scaled golden values carry the 6-significant-digit rounding of the
    # recorded file, hence the looser tolerance for those cases
    diffs = compare_coeffs(gold / "system.in.settings",
                           project / "system.in.settings",
                           tol=2e-5 if change.get("settings_scale") else 1e-6,
                           scale=change.get("settings_scale"))
    assert not diffs, "\n".join(diffs)
    for fname in ("system.in.init", "in.run"):
        assert normalized_text(gold / fname) == \
            normalized_text(project / fname), fname
    assert set(PIPELINE_FILES) <= {p.name for p in project.iterdir()}

    lmp = lammps_binary()
    if lmp is None or not golden_e["lammps"] or not same_rdkit:
        return
    energies = run_lammps_energies(project, lmp)
    keys = change.get("energies_unchanged", list(golden_e["run0"]))
    for key in keys:
        ref = golden_e["run0"][key]
        assert energies[key] == pytest.approx(ref, rel=1e-6, abs=1e-6), key
    if "energies_unchanged" in change:
        # the intentional parameter change must actually show up
        assert energies["evdwl"] != pytest.approx(
            golden_e["run0"]["evdwl"], rel=1e-6)
