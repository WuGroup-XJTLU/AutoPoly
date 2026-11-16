#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The material qualification matrix is consistent and backed by evidence.

Every md_ready combination must cite smoke-test cases that exist and that
exercise exactly that combination; materials without a combination are
not md_ready.
"""
import re

import pytest

from AutoPoly.materials import (
    QUALIFIED_COMBINATIONS,
    find_qualification,
    get_material,
    qualification_markdown,
    qualification_matrix,
    require_md_ready,
)
from AutoPoly.core.exceptions import ValidationError
from AutoPoly.materials.forcefields import get_inorganic_ff

from tests import test_nanocomposite_lammps_smoke as smoke


def _case_combination(case):
    seq, chains, ff, spec, _ = smoke.CASES[case]
    if spec == "imported":
        return ("sio2", "amorphous", "imported", "hydroxylated",
                "interface_silica", ff, "rigid", None)
    from AutoPoly.nanoparticles import Nanoparticle
    np_ = Nanoparticle(**spec)
    adapter = get_inorganic_ff(np_.force_field, np_.material.id).name
    linker = np_.grafts.linker if np_.grafts is not None else None
    return (np_.material.id, np_.phase, np_.geometry_type, np_.surface_state,
            adapter, ff, np_.motion, linker)


def test_ids_unique():
    ids = [q.id for q in QUALIFIED_COMBINATIONS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("q", QUALIFIED_COMBINATIONS, ids=lambda q: q.id)
def test_evidence_exists_and_matches(q):
    assert q.evidence, f"{q.id} has no evidence"
    for ev in q.evidence:
        m = re.match(r"tests/test_nanocomposite_lammps_smoke\.py::"
                     r"test_lammps_smoke\[(\w+)\]$", ev)
        assert m, ev
        case = m.group(1)
        assert case in smoke.CASES, f"{q.id}: missing smoke case {case}"
        assert smoke.CASES[case][4] == q.id
        combo = _case_combination(case)
        assert q.covers(*combo), (q.id, combo)
    # every adapter/material named is registered
    assert get_inorganic_ff(q.inorganic_ff, q.material).name == q.inorganic_ff
    spec = get_material(q.material)
    assert q.surface in spec.surface_states
    for ph in q.phases:
        assert ph in spec.phases


def test_every_smoke_case_backs_a_qualification():
    cited = {re.search(r"\[(\w+)\]", ev).group(1)
             for q in QUALIFIED_COMBINATIONS for ev in q.evidence}
    assert cited == set(smoke.CASES)


def test_matrix_rows():
    rows = {r["material"]: r for r in qualification_matrix()}
    assert set(rows) >= {"sio2", "al2o3", "tio2", "c", "au", "ag"}
    assert rows["tio2"]["D"] is False and rows["tio2"]["F"] is False
    assert rows["tio2"]["md_ready_combinations"] == []
    for mid in ("sio2", "al2o3", "c", "au", "ag"):
        assert all(rows[mid][k] for k in "ABCDEF"), mid
    md = qualification_markdown()
    for mid in rows:
        assert f"`{mid}`" in md


def test_lookup_semantics():
    q = find_qualification("sio2", "alpha_quartz", "cluster", "hydroxylated",
                           "interface_silica", "gaff2", "rigid")
    assert q is not None and q.id == "sio2-interface-hydroxylated"
    assert find_qualification("ag", "fcc", "cluster", "bare",
                              "interface_metal_ag", "gaff2", "rigid") is None
    with pytest.raises(ValidationError, match="ag-interface-lj"):
        require_md_ready("ag", "fcc", "cluster", "bare",
                         "interface_metal_ag", "gaff2", "rigid")
    with pytest.raises(ValidationError, match="No combination"):
        require_md_ready("tio2", "anatase", "cluster", "hydroxylated",
                         "x", "oplsaa", "rigid")
