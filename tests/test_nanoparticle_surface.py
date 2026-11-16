#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Curved-surface termination: coordination, OH density, charge, sites."""
import numpy as np
import pytest

from AutoPoly.materials.chemistry import oxides, silica
from AutoPoly.materials.crystals import CRYSTALS
from AutoPoly.materials.forcefields import get_inorganic_ff
from AutoPoly.nanoparticles import Nanoparticle, SilicaSurface


@pytest.mark.parametrize("crystal", ["beta_cristobalite", "alpha_quartz"])
@pytest.mark.parametrize("radius", [8.0, 12.0])
def test_silica_termination_valence(crystal, radius):
    s = silica.build_silica_cluster(CRYSTALS[crystal], radius)
    report = silica.validate_silica(s)
    nb = s.neighbors()
    for i, e in enumerate(s.elements):
        ne = sorted(s.elements[j] for j in nb[i])
        if e == "Si":
            assert ne == ["O"] * 4
        elif e == "H":
            assert ne == ["O"]
        else:
            assert ne in (["Si", "Si"], ["H", "Si"])
    assert set(report["q_distribution"]) <= {"Q2", "Q3", "Q4"}
    # intrinsic silanol density of crystal-derived spheres
    assert 4.0 < s.metadata["intrinsic_oh_density"] < 9.5
    assert s.metadata["achieved_oh_density"] == pytest.approx(
        s.metadata["intrinsic_oh_density"], rel=1e-9)


def test_silanol_geometry():
    s = silica.build_silica_cluster(CRYSTALS["beta_cristobalite"], 10.0)
    nb = s.neighbors()
    for o, role in enumerate(s.roles):
        if role != "OH":
            continue
        si = next(j for j in nb[o] if s.elements[j] == "Si")
        h = next(j for j in nb[o] if s.elements[j] == "H")
        v1 = s.positions[si] - s.positions[o]
        v2 = s.positions[h] - s.positions[o]
        ang = np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1)
                                   / np.linalg.norm(v2)))
        assert ang == pytest.approx(silica.SI_O_H_ANGLE, abs=0.5)
        assert np.linalg.norm(v2) == pytest.approx(0.945, abs=1e-6)


@pytest.mark.parametrize("ff", ["interface", "clayff"])
def test_silica_particle_neutral_under_both_tables(ff):
    s = silica.build_silica_cluster(CRYSTALS["beta_cristobalite"], 10.0)
    a = get_inorganic_ff(ff, "sio2").assign(s.roles)
    assert a.net_charge == pytest.approx(0.0, abs=1e-9)


def test_oh_density_target_reported_not_forced(caplog):
    np_ = Nanoparticle(material="SiO2", radius=10.0,
                       surface=SilicaSurface(oh_density=2.0))
    s = np_.build_structure()
    m = s.metadata
    assert m["target_oh_density"] == 2.0
    assert m["achieved_oh_density"] >= m["target_oh_density"]
    assert "condensation reached" in caplog.text or m["n_condensed"] > 0


def test_condensation_on_close_silanols():
    """Two dangling O on nearby, non-adjacent Si condense to Si-O-Si."""
    pos = np.array([[0, 0, 0], [3.0, 0, 0],
                    [-0.4, 1.5, 0], [3.4, 1.5, 0],
                    [0, -1.0, 1.2], [0, -1.0, -1.2], [-1.3, -0.5, 0],
                    [3.0, -1.0, 1.2], [3.0, -1.0, -1.2], [4.3, -0.5, 0]],
                   dtype=float)
    species = np.array(["Si", "Si"] + ["O"] * 8)
    bonds = [(0, 2), (1, 3), (0, 4), (0, 5), (0, 6), (1, 7), (1, 8), (1, 9)]
    sp, p, b, dangling, n, dmax = silica._condense_silanols(
        species, pos, bonds, [2, 3], target=0, seed=0)
    assert n == 1 and len(dangling) == 0
    o_bridge = [i for i in range(len(sp)) if sp[i] == "O"
                and sum(1 for x, y in b if i in (x, y)) == 2]
    assert len(o_bridge) == 1
    assert dmax <= 1.65


@pytest.mark.parametrize("material, crystal", [
    ("al2o3", "alpha_corundum"), ("tio2", "anatase"), ("tio2", "rutile")])
def test_oxide_bond_valence_protonation(material, crystal):
    s = oxides.build_oxide_cluster(material, CRYSTALS[crystal], 9.0)
    rules = oxides.RULES[material]
    nb = s.neighbors()
    for i, e in enumerate(s.elements):
        if e == rules.cation:
            assert sum(s.elements[j] == "O" for j in nb[i]) == rules.cation_cn
        if e == "O":
            n_h = sum(s.elements[j] == "H" for j in nb[i])
            assert n_h <= 2
    assert oxides.validate(s)["formal_charge"] == 0
    assert s.metadata["facet_specific"] is False


def test_alumina_clayff_neutral_and_roles():
    s = oxides.build_oxide_cluster("al2o3", CRYSTALS["alpha_corundum"], 9.0)
    a = get_inorganic_ff("clayff_alumina").assign(s.roles)
    assert a.net_charge == pytest.approx(0.0, abs=1e-9)
    assert set(s.roles) <= {"Al", "OB", "OH", "HO", "OW", "HW"}


def test_surface_flags_and_normals():
    s = Nanoparticle(material="SiO2", radius=10.0).build_structure()
    surf = np.array(s.surface)
    assert surf.any() and not surf.all()
    n = s.normals[surf]
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    assert np.all(np.einsum("ij,ij->i", n, s.positions[surf]) > 0)
