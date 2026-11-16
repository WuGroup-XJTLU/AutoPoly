#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Material: a user-facing material selection resolved through the registry.

    >>> from AutoPoly.materials import Material
    >>> silica = Material(name="SiO2", crystal="alpha_quartz",
    ...                   force_field="interface")
    >>> silica.id, silica.phase, silica.force_field_adapter.name
    ('sio2', 'alpha_quartz', 'interface_silica')

The same Material can describe nanoparticles (Nanoparticle(material=...))
and is reported for substrates; it records identity and the requested
parameter set, never a guarantee that the combination is MD-ready (see
AutoPoly.materials.qualification).

Created on 2026-11-13
@author: zwu
"""
from dataclasses import dataclass
from typing import Optional

from .registry import MaterialSpec, get_material


@dataclass(frozen=True)
class Material:
    """
    Args:
        name: Material ID, formula or alias ("SiO2", "silica", "Au", ...).
        crystal: Phase / structure ID (e.g. "alpha_quartz"); None = the
            material's default phase.  ``phase`` is an alias.
        force_field: Inorganic force-field adapter name or alias (e.g.
            "interface", "clayff_alumina"); None = material default.
        surface: Default surface-termination ID.
    """
    name: str
    crystal: Optional[str] = None
    force_field: Optional[str] = None
    surface: Optional[str] = None
    phase: Optional[str] = None

    @property
    def spec(self) -> MaterialSpec:
        return get_material(self.name)

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def resolved_phase(self) -> Optional[str]:
        requested = self.crystal or self.phase
        spec = self.spec
        if requested is None and spec.default_phase is None:
            return None
        return spec.resolve_phase(requested).name

    @property
    def force_field_adapter(self):
        """The resolved inorganic adapter (raises if none is defined)."""
        from ..core.exceptions import ValidationError
        from .forcefields import get_inorganic_ff

        name = self.force_field or self.spec.default_force_field
        if name is None:
            raise ValidationError(
                f"Material '{self.id}' has no validated inorganic force "
                "field; it is geometry-only"
            )
        return get_inorganic_ff(name, self.id)

    def describe(self) -> dict:
        spec = self.spec
        return {
            "id": spec.id,
            "formula": spec.formula,
            "phase": self.resolved_phase,
            "force_field": self.force_field or spec.default_force_field,
            "surface": self.surface or spec.default_surface,
            "capabilities": spec.capability_report(),
        }
