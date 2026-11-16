#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Built-in nanoparticle shape builders (registered per material).

Each builder has the signature

    builder(material, phase, shape, surface, surface_state, seed,
            cap_length) -> AtomicStructure

and delegates every chemical decision (which atoms survive at the
surface, termination, validation) to the material's chemistry adapter.
The MaterialSpec.builders table of AutoPoly.materials.registry maps
shapes to these functions; nothing here branches on material names.

Created on 2026-11-13
@author: zwu
"""
from ..core.exceptions import ValidationError
from ..materials.crystals import get_crystal


def _crystal_of(phase):
    if phase.crystal is None:
        raise ValidationError(
            f"Phase '{phase.name}' has no built-in crystal source "
            "(import a structure with Nanoparticle.from_file)"
        )
    return get_crystal(phase.crystal)


def build_silica_sphere(material, phase, shape, surface, surface_state,
                        seed, cap_length):
    """Hydroxylated silica sphere carved from quartz / cristobalite."""
    from ..materials.chemistry import silica

    if surface_state != "hydroxylated":
        raise ValidationError(
            f"Silica spheres support surface 'hydroxylated', got "
            f"'{surface_state}'"
        )
    return silica.build_silica_cluster(
        _crystal_of(phase),
        radius=shape.radius,
        oh_density=surface.params.get("oh_density"),
        oh_length=cap_length or silica.DEFAULT_OH_LENGTH,
        seed=seed,
        center_offset=shape.center_offset,
        phase=phase.name,
    )


def build_oxide_sphere(material, phase, shape, surface, surface_state,
                       seed, cap_length):
    """Metal-oxide sphere (alumina, titania) with bond-valence termination."""
    from ..materials.chemistry import oxides

    return oxides.build_oxide_cluster(
        material.id,
        _crystal_of(phase),
        radius=shape.radius,
        termination=surface_state,
        oh_length=cap_length or oxides.DEFAULT_OH_LENGTH,
        center_offset=shape.center_offset,
        phase=phase.name,
    )


def build_graphene_sheet(material, phase, shape, surface, surface_state,
                         seed, cap_length):
    """H-passivated graphene flake (optionally AB-stacked layers)."""
    from ..materials.chemistry import carbon

    return carbon.build_graphene_flake(
        length=shape.length, width=shape.width, layers=shape.layers,
        ch_length=cap_length or carbon.CH_LENGTH,
    )


def build_nanotube(material, phase, shape, surface, surface_state,
                   seed, cap_length):
    """Open-ended, H-passivated single-wall carbon nanotube."""
    from ..materials.chemistry import carbon

    return carbon.build_nanotube(
        n=shape.n, m=shape.m, length=shape.length,
        ch_length=cap_length or carbon.CH_LENGTH,
    )


def build_metal_cluster(material, phase, shape, surface, surface_state,
                        seed, cap_length):
    """fcc metal cluster: sphere cut, or ASE polyhedra (optional)."""
    from ..materials.chemistry import metals

    element = material.formula
    if shape.name == "sphere":
        return metals.build_fcc_sphere(element, _crystal_of(phase),
                                       radius=shape.radius,
                                       center_offset=shape.center_offset)
    return metals.build_fcc_polyhedron(element, kind=shape.kind,
                                       size=shape.size,
                                       surface_energies=shape.surface_energies)
