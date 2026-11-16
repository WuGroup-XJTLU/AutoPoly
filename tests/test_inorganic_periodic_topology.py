#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neighbor search / bond perception with and without periodicity."""
import numpy as np
import pytest

from AutoPoly.core.exceptions import ValidationError
from AutoPoly.materials import topology


def _brute_force(pos, cutoff, lengths, pbc):
    out = set()
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[j] - pos[i]
            for a in range(3):
                if pbc[a]:
                    d[a] -= lengths[a] * np.round(d[a] / lengths[a])
            if np.linalg.norm(d) < cutoff:
                out.add((i, j))
    return out


@pytest.mark.parametrize("pbc", [(False, False, False), (True, True, False),
                                 (True, True, True)])
@pytest.mark.parametrize("n", [50, 2500])
def test_neighbor_pairs_match_brute_force(pbc, n):
    rng = np.random.default_rng(1)
    lengths = np.array([20.0, 22.0, 24.0])
    pos = rng.uniform(0, 1, (n, 3)) * lengths
    cell = np.diag(lengths)
    i, j, img, d = topology.neighbor_pairs(pos, 2.5, cell, pbc)
    got = set(zip(i.tolist(), j.tolist()))
    if n <= 200:
        assert got == _brute_force(pos, 2.5, lengths, pbc)
    # distances and image shifts are consistent
    vec = pos[j] - pos[i] + img * lengths
    np.testing.assert_allclose(np.linalg.norm(vec, axis=1), d, atol=1e-9)
    assert np.all(d < 2.5)
    for a in range(3):
        if not pbc[a]:
            assert not img[:, a].any()


def test_cell_list_fallback_matches():
    """The scipy-free cell-list path finds the same pairs."""
    rng = np.random.default_rng(3)
    lengths = np.array([30.0, 30.0, 30.0])
    pos = rng.uniform(0, 30, (3000, 3))
    pbc = (True, True, False)
    ref = topology._cell_list_pairs(pos, 2.0, lengths, pbc)
    i, j, _, _ = topology.neighbor_pairs(pos, 2.0, np.diag(lengths), pbc)
    ref_set = set()
    for a, b in zip(*ref):
        d = pos[b] - pos[a]
        d[:2] -= 30.0 * np.round(d[:2] / 30.0)
        if np.linalg.norm(d) < 2.0:
            ref_set.add((int(min(a, b)), int(max(a, b))))
    assert set(zip(i.tolist(), j.tolist())) == ref_set


def test_cutoff_must_be_below_half_box():
    with pytest.raises(ValidationError, match="half the periodic box"):
        topology.neighbor_pairs(np.zeros((3, 3)), 6.0, np.diag([10, 10, 10]),
                                (True, False, False))


def test_triclinic_periodic_cell_rejected():
    cell = np.array([[10, 0, 0], [2, 10, 0], [0, 0, 10]], dtype=float)
    with pytest.raises(ValidationError, match="orthorhombic"):
        topology.neighbor_pairs(np.zeros((3, 3)), 2.0, cell,
                                (True, True, True))


def test_perceive_bonds_element_pairs_and_images():
    # Si at the -x edge bonded to O across the boundary; H only to O
    pos = np.array([[0.3, 5, 5], [9.0, 5, 5], [8.1, 5, 5], [5, 5, 5]])
    el = ["Si", "O", "H", "Si"]
    bonds, images = topology.perceive_bonds(
        el, pos, {("Si", "O"): 1.9, ("O", "H"): 1.2},
        cell=np.diag([10.0, 10.0, 10.0]), pbc=(True, True, True))
    assert bonds == [(0, 1), (1, 2)]
    assert images.tolist() == [[-1, 0, 0], [0, 0, 0]]
    inferred = topology.infer_images(pos, bonds, np.diag([10.0] * 3),
                                     (True, True, True))
    assert inferred.tolist() == images.tolist()


def test_wrap_and_minimum_image():
    cell = np.diag([10.0, 10.0, 10.0])
    pos = np.array([[-1.0, 11.0, 3.0]])
    w = topology.wrap_positions(pos, cell, (True, True, False))
    np.testing.assert_allclose(w, [[9.0, 1.0, 3.0]])
    mi = topology.minimum_image(np.array([[9.0, -6.0, 9.0]]), cell,
                                (True, True, False))
    np.testing.assert_allclose(mi, [[-1.0, 4.0, 9.0]])


def test_components_ordered_by_size():
    labels = topology.connected_components(6, [(4, 5), (0, 1), (1, 2)])
    assert labels == [0, 0, 0, 2, 1, 1]
    assert topology.largest_component(6, [(4, 5), (0, 1), (1, 2)]) == [0, 1, 2]


def test_silica_slab_bonds_cross_lateral_edges():
    """Lateral periodic slab: periodic bonds have images and true lengths."""
    from AutoPoly.surfaces import CristobaliteBuilder

    slab = CristobaliteBuilder(lx_target=21.0, ly_target=18.0,
                               thickness=13.0).build()
    s = slab.structure
    assert (s.images[:, :2] != 0).any()
    assert not s.images[:, 2].any()
    lengths = s.bond_lengths()
    si_o = [L for (i, j), L in zip(s.bonds, lengths)
            if {s.elements[i], s.elements[j]} == {"Si", "O"}]
    assert 1.55 < min(si_o) and max(si_o) < 1.7
    # Every bond is recovered by independent perception
    bonds, _ = topology.perceive_bonds(
        s.elements, s.positions, {("Si", "O"): 1.9, ("O", "H"): 1.2},
        s.cell, s.pbc)
    assert {tuple(sorted(b)) for b in bonds} == \
        {tuple(sorted(b)) for b in s.bonds}
