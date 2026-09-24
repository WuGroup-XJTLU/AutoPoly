#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow Convenience Function for AutoPoly Package

This module provides :func:`generate`, a thin one-shot convenience function
that composes the three pipeline stages:

    GeometryBuilder  (stage 1: FF-agnostic coordinates -> geometry/)
    UnitTyper        (stage 2: force-field typing      -> build/<ff>/)
    BoxPacker        (stage 3: box packing + moltemplate -> moltemplate/)

All stage logic lives in the stage modules (geometry.py, typing.py,
packer.py, packing/). Use the stage classes directly when you need control
between stages (e.g. typing one geometry under multiple force fields); use
generate() when a single force field one-shot run is all you need.

Created on 2026-01-06
@author: zwu
"""
from typing import List, Optional

from ..core.system import logger
from ..core.exceptions import ValidationError, WorkflowError
from ..models.molecule import Molecule
from ..packing import AUTO_COUNT, PlacementResult, SubstrateSpec
from .geometry import GeometryBuilder, GeometryConfig, is_molecule_model
from .typing import UnitTyper
from .packer import BoxPacker

__all__ = ["generate"]


def generate(
    system: object,
    name: str,
    models: List[object],
    force_field: str = "oplsaa",
    *,
    strategy: str = "mc_random",
    geometry_config: Optional[GeometryConfig] = None,
    box_size: Optional[float] = None,
    mc_max_attempts: int = 10000,
    monomer_density: float = 0.085,
    rng_seed: Optional[int] = None,
    run_moltemplate: bool = True,
    substrate: Optional[SubstrateSpec] = None,
    subtract: Optional[list] = None,
    box_dims: Optional[tuple] = None,
    typer_options: Optional[dict] = None,
) -> PlacementResult:
    """
    Generate a complete LAMMPS system in one call (three-stage pipeline).

    Runs GeometryBuilder -> UnitTyper -> BoxPacker, producing:

    - ``<out>/<name>/geometry/geometry.json``
    - ``<out>/<name>/build/<ff>/*.lt`` + ``units.json``
    - ``<out>/<name>/moltemplate/system.lt`` -> moltemplate -> ``system.data`` etc.

    Args:
        system: System object providing get_folder_path().
        name: Project name; outputs go to ``<out>/<name>/``.
        models: List of Polymer and/or Molecule objects (the "film" models
            when a substrate is used).
        force_field: Force field name (see FORCE_FIELD_REGISTRY).
            Defaults to "oplsaa".
        strategy: Box packing strategy ("mc_random", "grid",
            "on_substrate", ...). Defaults to "mc_random". When
            ``substrate`` is given and strategy is left at its default,
            "on_substrate" is selected automatically.
        geometry_config: Optional GeometryConfig for stage 1 (MC chain
            growth parameters). Defaults to GeometryConfig().
        box_size: Optional explicit box edge length (Angstrom) for stage 3.
        mc_max_attempts: Maximum placement attempts for MC placement.
            Defaults to 10000.
        monomer_density: Target monomer density (monomers/A^3) for MC box
            sizing. Defaults to 0.085.
        rng_seed: Optional RNG seed for reproducible placement.
        run_moltemplate: Run moltemplate at the end of stage 3.
            Defaults to True.
        substrate: Optional SubstrateSpec describing a physical substrate
            slab; film models are packed on top of it. Selects the
            "on_substrate" strategy automatically.
        subtract: Optional list of CarveRegion specs (CutAbove, Cylinder,
            ...) removing whole instances after placement. Supported by
            "mc_random" and "on_substrate".
        box_dims: Optional per-axis box sides (lx, ly, lz) in Angstrom;
            any element may be None (auto-sized). Overrides box_size per
            axis where given.
        typer_options: Optional force-field-specific typing options, forwarded
            to the stage-2 typer. Currently only used for
            force_field="openff" (OpenFFTyper kwargs: ``offxml``,
            ``charge_method`` ("nagl"/"am1bcc"/"gasteiger"), ``nagl_model``,
            ``neutralize``).

    Returns:
        PlacementResult from the packing stage.

    Raises:
        WorkflowError: If any critical step fails or required files are missing
        ValidationError: On contradictory strategy/substrate/subtract
                         combinations or unresolvable substrate counts.

    Example:
        >>> from AutoPoly import System, Polymer, generate
        >>> system = System(out="pmma_out")
        >>> polymer = Polymer(chain_num=4, sequence=sequence,
        ...                   topology="linear", tacticity="atactic")
        >>> generate(system, "pmma", [polymer], force_field="oplsaa")

        >>> # Film on a substrate slab:
        >>> from AutoPoly.packing import SubstrateSpec
        >>> spec = SubstrateSpec(model=Molecule(Count=200, Smiles="O=[Si]=O",
        ...                                   Name="sio2"),
        ...                      thickness=15.0, packing="grid", gap=3.0)
        >>> generate(system, "pmma_on_sio2", [polymer],
        ...          substrate=spec, box_dims=(60.0, 60.0, None))
    """
    try:
        logger.info("Starting LAMMPS data file generation using Moltemplate")

        strategy = _resolve_strategy(strategy, substrate, subtract)
        substrate_models = _resolve_substrate_models(substrate, box_dims)

        # Stage 1: force-field-agnostic geometry
        geometry_result = GeometryBuilder(
            system, name, config=geometry_config
        ).build(models, substrate_models=substrate_models)

        # Stage 2: force-field typing on the stored geometry
        units = UnitTyper(
            geometry_result.dir, force_field, typer_options=typer_options
        ).type()

        # Stage 3: box packing + moltemplate
        result = BoxPacker(
            system,
            name,
            strategy=strategy,
            box_size=box_size,
            mc_max_attempts=mc_max_attempts,
            monomer_density=monomer_density,
            rng_seed=rng_seed,
            run_moltemplate=run_moltemplate,
            substrate=substrate,
            subtract=subtract,
            box_dims=box_dims,
        ).pack(units)

        logger.info("Successfully completed polymer generation")
        return result

    except Exception as e:
        raise WorkflowError(
            f"Error in generate: {str(e)}"
        ) from e


def _resolve_strategy(
    strategy: str,
    substrate: Optional[SubstrateSpec],
    subtract: Optional[list],
) -> str:
    """
    Pick the packing strategy from the substrate/subtract arguments.

    A substrate auto-selects "on_substrate" when strategy is left at the
    "mc_random" default; combining a substrate with any other explicit
    strategy is contradictory and rejected. Subtract regions require a
    strategy that implements whole-instance removal.
    """
    if substrate is not None:
        if strategy == "mc_random":
            logger.info("Substrate given: selecting strategy 'on_substrate'")
            return "on_substrate"
        if strategy != "on_substrate":
            raise ValidationError(
                f"substrate=... conflicts with strategy='{strategy}'; "
                "use strategy='on_substrate' or leave the strategy default"
            )
    if subtract and strategy == "grid":
        raise ValidationError(
            "subtract=... is not supported by strategy 'grid'; "
            "use 'mc_random' or 'on_substrate'"
        )
    return strategy


def _resolve_substrate_models(
    substrate: Optional[SubstrateSpec],
    box_dims: Optional[tuple],
) -> List[object]:
    """
    Resolve the substrate model list for the geometry stage.

    Handles count overrides: a positive int replaces the model's
    Count/chain_num; AUTO_COUNT derives the instance count from slab
    volume x density (Molecule substrates only, explicit lateral box_dims
    required). External .lt substrates contribute no models.
    """
    if substrate is None or substrate.is_external or substrate.is_builder:
        return []

    model = substrate.model
    count = substrate.count
    if count is None:
        return [model]

    if count == AUTO_COUNT:
        if not is_molecule_model(model):
            raise ValidationError(
                "SubstrateSpec count='auto' is only supported for Molecule "
                "substrates; give a Polymer substrate an explicit chain_num"
            )
        dims = box_dims or (None, None, None)
        lx, ly = dims[0], dims[1]
        if lx is None or ly is None:
            raise ValidationError(
                "SubstrateSpec count='auto' requires explicit lateral "
                "box_dims=(lx, ly, ...) so the slab area is known"
            )
        count = int(substrate.density * lx * ly * substrate.thickness)
        if count < 1:
            raise ValidationError(
                f"Auto substrate count resolved to {count} "
                f"(density={substrate.density}, area={lx}x{ly}, "
                f"thickness={substrate.thickness}); increase the slab "
                f"dimensions or density"
            )
        logger.info(f"Auto substrate count: {count} instances")

    if is_molecule_model(model):
        return [Molecule(Count=int(count), Smiles=model.Smiles,
                         Name=model.molecule_name)]

    # Polymer substrate: override chain_num, keep the chemistry
    overridden = object.__new__(type(model))
    overridden.__dict__.update(model.__dict__)
    overridden.chain_num = int(count)
    overridden.set_Sequence()
    return [overridden]
