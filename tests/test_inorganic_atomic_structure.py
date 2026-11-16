#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the force-field-agnostic AtomicStructure."""
import json

import numpy as np
import pytest

from AutoPoly.core.exceptions import ValidationError
from AutoPoly.materials import AtomicStructure


def _water_like():
    return AtomicStructure(
        elements=["O", "H", "H"],
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        bonds=[(0, 1), (0, 2)],
        roles=["OW", "HW", "HW"],
        geometry_type="cluster",
        descriptors={"envelope": {"shape": "sphere", "radius": 1.0}},
        provenance={"source": "test"},
        metadata={"material_id": "test"},
    )


class TestConstruction:
    def test_basic_properties(self):
        s = _water_like()
        assert s.n_atoms == 3 and s.n_bonds == 2
        assert s.formula() == "OH2"
        assert s.element_counts() == {"O": 1, "H": 2}
        assert s.coordination().tolist() == [2, 1, 1]
        assert s.images.shape == (2, 3) and not s.images.any()
        assert s.total_mass() == pytest.approx(15.9994 + 2 * 1.008)

    def test_no_forcefield_fields(self):
        d = _water_like().to_dict()
        for key in ("charges", "charge", "epsilon", "sigma", "types"):
            assert key not in d

    def test_validation_errors(self):
        with pytest.raises(ValidationError, match="outside"):
            AtomicStructure(["O", "H"], [[0, 0, 0], [1, 0, 0]], bonds=[(0, 5)])
        with pytest.raises(ValidationError, match="itself"):
            AtomicStructure(["O"], [[0, 0, 0]], bonds=[(0, 0)])
        with pytest.raises(ValidationError, match="duplicate"):
            AtomicStructure(["O", "H"], [[0, 0, 0], [1, 0, 0]],
                            bonds=[(0, 1), (1, 0)])
        with pytest.raises(ValidationError, match="require a cell"):
            AtomicStructure(["O"], [[0, 0, 0]], pbc=(True, False, False))
        with pytest.raises(ValidationError, match="non-zero bond images"):
            AtomicStructure(["O", "H"], [[0, 0, 0], [1, 0, 0]],
                            bonds=[(0, 1)], images=[[1, 0, 0]])
        with pytest.raises(ValidationError, match="geometry_type"):
            AtomicStructure(["O"], [[0, 0, 0]], geometry_type="blob")
        with pytest.raises(ValidationError, match="'roles'"):
            AtomicStructure(["O", "H"], [[0, 0, 0], [1, 0, 0]], roles=["O"])

    def test_image_on_nonperiodic_axis_rejected(self):
        with pytest.raises(ValidationError, match="non-periodic axis"):
            AtomicStructure(["O", "O"], [[0, 0, 0], [9, 0, 0]],
                            bonds=[(0, 1)], images=[[0, 0, 1]],
                            cell=np.diag([10, 10, 10]),
                            pbc=(True, True, False))


class TestRoundTrip:
    def test_json_round_trip_preserves_everything(self):
        s = _water_like()
        s.surface = [True, True, True]
        s.normals = np.array([[1, 0, 0]] * 3, dtype=float)
        s.annotate_components()
        back = AtomicStructure.from_dict(json.loads(json.dumps(s.to_dict())))
        assert back.elements == s.elements
        np.testing.assert_array_equal(back.positions, s.positions)
        assert back.bonds == s.bonds
        assert back.roles == s.roles
        assert back.surface == s.surface
        np.testing.assert_array_equal(back.normals, s.normals)
        assert back.component_ids == [0, 0, 0]
        assert back.descriptors == s.descriptors
        assert back.provenance == s.provenance

    def test_periodic_round_trip_keeps_images(self):
        cell = np.diag([10.0, 10.0, 30.0])
        s = AtomicStructure(
            ["Si", "O"], [[0.2, 5, 5], [9.6, 5, 5]], bonds=[(0, 1)],
            images=[[-1, 0, 0]], cell=cell, pbc=(True, True, False),
            geometry_type="slab",
        )
        assert s.bond_lengths()[0] == pytest.approx(0.6)
        back = AtomicStructure.from_dict(json.loads(json.dumps(s.to_dict())))
        assert back.images.tolist() == [[-1, 0, 0]]
        assert back.bond_lengths()[0] == pytest.approx(0.6)


class TestEditing:
    def test_subset_keeps_internal_bonds_only(self):
        s = _water_like()
        sub = s.subset([0, 1])
        assert sub.elements == ["O", "H"]
        assert sub.bonds == [(0, 1)]
        assert sub.roles == ["OW", "HW"]

    def test_rotate_and_translate(self):
        s = _water_like()
        r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        lengths = s.bond_lengths().copy()
        s.rotate(r)
        s.translate([1, 2, 3])
        np.testing.assert_allclose(s.bond_lengths(), lengths)
        np.testing.assert_allclose(s.positions[0], [1, 2, 3])

    def test_components(self):
        s = AtomicStructure(["O", "H", "O"], [[0, 0, 0], [1, 0, 0], [5, 0, 0]],
                            bonds=[(0, 1)])
        assert s.connected_components() == [0, 0, 1]
