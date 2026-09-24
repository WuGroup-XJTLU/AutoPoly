# -*- coding: utf-8 -*-
"""Tests for the generate() convenience function."""

from unittest.mock import MagicMock, patch

import pytest

from AutoPoly.core.exceptions import WorkflowError
from AutoPoly.pipeline.workflow import generate


@pytest.fixture
def mock_system():
    system = MagicMock()
    system.get_folder_path.return_value = "/tmp/autopoly_test"
    return system


@pytest.fixture
def stages():
    """Mock the three pipeline stages in the workflow module."""
    with patch("AutoPoly.pipeline.workflow.GeometryBuilder") as geometry_cls, \
         patch("AutoPoly.pipeline.workflow.UnitTyper") as typer_cls, \
         patch("AutoPoly.pipeline.workflow.BoxPacker") as packer_cls:
        geometry = geometry_cls.return_value
        geometry_result = geometry.build.return_value
        units = typer_cls.return_value.type.return_value
        pack_result = packer_cls.return_value.pack.return_value
        yield {
            "geometry_cls": geometry_cls,
            "typer_cls": typer_cls,
            "packer_cls": packer_cls,
            "geometry": geometry,
            "geometry_result": geometry_result,
            "units": units,
            "pack_result": pack_result,
        }


class TestGenerateWiring:
    """Test that generate() wires the three stages together correctly."""

    def test_runs_stages_in_order(self, mock_system, stages):
        models = [MagicMock()]
        result = generate(mock_system, "peo", models, force_field="oplsaa")

        # Stage 1: geometry built from the models (no substrate models)
        stages["geometry_cls"].assert_called_once()
        stages["geometry"].build.assert_called_once_with(
            models, substrate_models=[]
        )

        # Stage 2: typing runs on the stage-1 geometry dir with the force field
        stages["typer_cls"].assert_called_once_with(
            stages["geometry_result"].dir, "oplsaa", typer_options=None
        )

        # Stage 3: packing consumes the stage-2 units
        stages["packer_cls"].return_value.pack.assert_called_once_with(stages["units"])

        # The packing result is returned to the caller
        assert result is stages["pack_result"]

    def test_passes_parameters_to_packer(self, mock_system, stages):
        generate(
            mock_system,
            "peo",
            [MagicMock()],
            force_field="gaff2",
            strategy="grid",
            box_size=50.0,
            mc_max_attempts=500,
            monomer_density=0.05,
            rng_seed=42,
            run_moltemplate=False,
        )

        stages["packer_cls"].assert_called_once_with(
            mock_system,
            "peo",
            strategy="grid",
            box_size=50.0,
            mc_max_attempts=500,
            monomer_density=0.05,
            rng_seed=42,
            run_moltemplate=False,
            substrate=None,
            subtract=None,
            box_dims=None,
        )

    def test_default_parameters(self, mock_system, stages):
        generate(mock_system, "peo", [MagicMock()])

        stages["packer_cls"].assert_called_once_with(
            mock_system,
            "peo",
            strategy="mc_random",
            box_size=None,
            mc_max_attempts=10000,
            monomer_density=0.085,
            rng_seed=None,
            run_moltemplate=True,
            substrate=None,
            subtract=None,
            box_dims=None,
        )

    def test_geometry_config_forwarded(self, mock_system, stages):
        config = MagicMock()
        generate(mock_system, "peo", [MagicMock()], geometry_config=config)

        stages["geometry_cls"].assert_called_once_with(
            mock_system, "peo", config=config
        )


class TestGenerateErrors:
    """Test error handling."""

    def test_stage_failure_wrapped_in_workflow_error(self, mock_system, stages):
        stages["geometry"].build.side_effect = ValueError("bad SMILES")

        with pytest.raises(WorkflowError, match="bad SMILES"):
            generate(mock_system, "peo", [MagicMock()])

    def test_invalid_force_field_raises(self, mock_system, stages):
        from AutoPoly.core.exceptions import ValidationError
        stages["typer_cls"].side_effect = ValidationError("Invalid force field 'bogus'")

        with pytest.raises(WorkflowError, match="Invalid force field"):
            generate(mock_system, "peo", [MagicMock()], force_field="bogus")
