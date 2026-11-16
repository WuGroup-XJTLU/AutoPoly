#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nanocomposite packing: fillers first, then an obstacle-aware matrix.

Strategy "nanocomposite_mc" (stage 3; reads only units.json + the per-unit
class-local coordinate files written by stage 2):

1. Box: explicit box_dims / box_size win; otherwise a cubic box sized for
   the matrix at ``initial_density`` plus the nominal filler core volume:
   V_box = M_matrix / rho_init + sum(V_core).  (Atomistic estimate only:
   build at modest density, compress/equilibrate in MD.)
2. Fillers: every instance of every filler unit gets a random position and
   orientation under full 3D periodic boundaries; its atoms must keep
   ``min_gap`` from every atom of every other filler *including periodic
   images*.  A particle must also be smaller than the box (no self-image
   contact).
3. Matrix: every polymer chain / molecule instance gets a random rigid-body
   placement (orientation + an anchor atom dropped into a free voxel); all
   its atoms must keep ``filler_clearance`` from filler atoms and
   ``polymer_clearance`` from atoms of other matrix instances (minimum
   image).  Coils interpenetrate freely as long as atoms do not clash --
   whole-chain bounding spheres are never used.  When rigid placement of
   a pre-grown chain fails ("auto" mode, default) -- or always ("regrow")
   -- the chain is regrown monomer by monomer inside the box around the
   obstacles (packing/chain_regrowth.py) and written as a per-instance
   class with the new monomer transforms (typing untouched).
4. No silent fallback: an instance that cannot be placed raises a
   PackingError with the free-volume diagnostics and concrete suggestions.
5. An exact minimum-image neighbor search over all atoms verifies every
   clearance before the lt commands are emitted; the achieved minimum
   distances, nominal loading and box are returned in the result report.

Created on 2026-11-13
@author: zwu
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.exceptions import ValidationError
from ..core.system import logger
from ..pipeline.unit_coords import matrix_to_axis_angle, read_xyz
from .base import (
    PackingContext,
    PlacementRecord,
    PlacementResult,
    PlacementStrategy,
)
from .collision_geometry import (
    PeriodicVoxelGrid,
    min_image_contacts,
    random_rotation,
)

#: amu / (g/cm^3) -> A^3
AMU_PER_GCM3_TO_A3 = 1.0 / 0.602214076
#: Intra-chain contacts (beyond 1-4 / 1-4) closer than these (A) mark a
#: chain conformation as self-overlapping (regrown in "auto" mode).
INTRA_CHECK_CUTOFF = 1.2
INTRA_CHECK_CUTOFF_14 = 1.3


class PackingError(ValidationError):
    """Raised when the nanocomposite packing cannot place an instance."""


@dataclass
class NanocompositePacking:
    """
    Configuration of the "nanocomposite_mc" strategy.

    Attributes:
        initial_density: Matrix mass density (g/cm^3) used to size the box
            when no explicit box is given.  Building pre-grown chains at
            the final melt density is rarely possible; the generated run
            script compresses toward target_polymer_density.
        target_polymer_density: Final matrix density (g/cm^3) recorded in
            the report and used by the compression stage of in.run;
            None = no compression stage.
        min_gap: Minimum atom-atom distance between two fillers (A),
            periodic images included (surface-to-surface gap).
        filler_clearance: Minimum matrix-atom to filler-atom distance (A).
        polymer_clearance: Minimum distance (A) between atoms of different
            matrix instances (heavy atoms).
        hydrogen_clearance: Same, for contacts with matrix hydrogen atoms
            (A); smaller because H atoms are small.
        max_initial_overlap: Allowed penetration below the clearances (A);
            0.0 = strict.  The exact final check uses clearance - this.
        max_attempts: Placement attempts per instance before PackingError
            ("rigid" mode).
        matrix_placement: "auto" (rigid placement, regrow the chain in
            place when that fails), "rigid" (pre-grown conformations
            only), or "regrow" (always regrow polymer chains in the box).
        rigid_attempts: "auto" mode: rigid attempts before regrowing.
        regrow_starts: Independent in-box regrowth attempts per chain.
        intra_clearance: Minimum distance between atoms of the same chain
            more than three bonds apart (regrown chains) (A).
        intra14_clearance: Same for 1-4 pairs across a junction (A).
        voxel_size: Collision-grid resolution (A).
        seed: RNG seed (None = nondeterministic; the BoxPacker rng_seed is
            used when unset).
        packing_failure: Only "raise" (never fall back to overlapping
            positions).
    """
    initial_density: float = 0.35
    target_polymer_density: Optional[float] = None
    min_gap: float = 3.0
    filler_clearance: float = 2.5
    polymer_clearance: float = 1.7
    hydrogen_clearance: float = 1.2
    max_initial_overlap: float = 0.0
    max_attempts: int = 4000
    matrix_placement: str = "auto"
    rigid_attempts: int = 300
    regrow_starts: int = 40
    intra_clearance: float = 1.3
    intra14_clearance: float = 1.4
    voxel_size: float = 0.25
    seed: Optional[int] = None
    packing_failure: str = "raise"

    def __post_init__(self) -> None:
        if self.packing_failure != "raise":
            raise ValidationError(
                "NanocompositePacking.packing_failure must be 'raise' (silent "
                "fallback to overlapping positions is not supported)"
            )
        for name in ("initial_density", "min_gap", "filler_clearance",
                     "polymer_clearance", "hydrogen_clearance", "voxel_size"):
            if not getattr(self, name) > 0:
                raise ValidationError(f"NanocompositePacking.{name} must be > 0")
        if self.max_initial_overlap < 0:
            raise ValidationError("max_initial_overlap must be >= 0")
        if self.target_polymer_density is not None:
            if self.target_polymer_density <= 0:
                raise ValidationError("target_polymer_density must be > 0")
            if self.initial_density > self.target_polymer_density:
                raise ValidationError(
                    f"initial_density ({self.initial_density}) exceeds "
                    f"target_polymer_density ({self.target_polymer_density})"
                )
        if self.max_attempts < 1:
            raise ValidationError("max_attempts must be >= 1")
        if self.matrix_placement not in ("auto", "rigid", "regrow"):
            raise ValidationError(
                "matrix_placement must be 'auto', 'rigid' or 'regrow', got "
                f"'{self.matrix_placement}'"
            )
        if not (self.intra_clearance > 0 and self.intra14_clearance > 0):
            raise ValidationError("intra clearances must be > 0")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class _Body:
    unit_id: str
    instance: str
    class_name: str
    local: np.ndarray            # (n, 3) class-local coordinates
    elements: List[str]
    is_filler: bool
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    check_self_image: bool = False
    chain: Optional[object] = None          # ChainTemplate (polymers)
    lt_path: Optional[Path] = None          # typed class file (polymers)
    override: Optional[np.ndarray] = None   # regrown world coordinates
    regrown: Optional[list] = None          # regrown monomer transforms
    intra_clash: int = 0                    # stage-1 intra-chain contacts
    n_core: Optional[int] = None            # grafted fillers: core atoms

    @property
    def core_mask(self) -> np.ndarray:
        """True for filler core atoms (grafted chains are False)."""
        m = np.zeros(len(self.elements), dtype=bool)
        if self.is_filler:
            m[:self.n_core if self.n_core is not None else len(m)] = True
        return m

    def world(self) -> np.ndarray:
        if self.override is not None:
            return self.override
        return self.local @ self.rotation.T + self.translation

    @property
    def regrown_class(self) -> str:
        return f"{self.class_name}__{self.instance}"


class NanocompositeMCStrategy(PlacementStrategy):
    """Filler-first, obstacle-aware atom-level random packing."""

    name = "nanocomposite_mc"

    def place(self, ctx: PackingContext) -> PlacementResult:
        cfg = ctx.nanocomposite or NanocompositePacking()
        if ctx.substrate is not None:
            raise ValidationError(
                "Strategy 'nanocomposite_mc' does not combine fillers with a "
                "substrate yet ('nanocomposite_on_substrate' is a later "
                "milestone): it would have to respect both the slab region "
                "and the filler obstacles"
            )
        if ctx.subtract:
            raise ValidationError(
                "subtract=... is not supported by 'nanocomposite_mc'"
            )
        units = ctx.units
        bad = [u.id for u in units.units if getattr(u, "kind", None) ==
               "inorganic" and u.role != "filler"]
        if bad:
            raise ValidationError(
                f"'nanocomposite_mc' places filler units only; got inorganic "
                f"non-filler unit(s) {bad}"
            )
        seed = cfg.seed if cfg.seed is not None else ctx.rng_seed
        rng = np.random.default_rng(seed)
        build_dir = Path(units.source_dir) if units.source_dir else None

        fillers, matrix = self._bodies(units, build_dir)
        lo, lengths, sizing = self._box(ctx, cfg, units)
        self._check_box(fillers, matrix, lengths, cfg)

        # --- 2. fillers -------------------------------------------------
        gap_grid = PeriodicVoxelGrid(lo, lengths, cfg.voxel_size)
        soft_grid = None
        if any(not b.core_mask.all() for b in fillers):
            # grafted shells: shell atoms of different fillers keep the
            # polymer clearance (cores keep min_gap via gap_grid)
            soft_grid = PeriodicVoxelGrid(lo, lengths, cfg.voxel_size)
        matrix_grid = PeriodicVoxelGrid(lo, lengths, cfg.voxel_size)
        for k, body in enumerate(fillers):
            self._place_filler(body, k, fillers, gap_grid, lo, lengths,
                               cfg, rng, soft_grid)
            world = body.world()
            core = body.core_mask
            is_h = np.array([e == "H" for e in body.elements])
            matrix_grid.mark(world[core], cfg.filler_clearance)
            matrix_grid.mark(world[~core & ~is_h], cfg.polymer_clearance)
            matrix_grid.mark(world[~core & is_h], cfg.hydrogen_clearance)
        # --- 3. matrix --------------------------------------------------
        for k, body in enumerate(matrix):
            self._place_matrix(body, k, matrix, fillers, matrix_grid, lo,
                               lengths, cfg, rng)
            world = body.world()
            is_h = np.array([e == "H" for e in body.elements])
            matrix_grid.mark(world[~is_h], cfg.polymer_clearance)
            matrix_grid.mark(world[is_h], cfg.hydrogen_clearance)

        bounds = tuple((float(lo[a]), float(lo[a] + lengths[a]))
                       for a in range(3))
        report = self._verify(fillers, matrix, lengths, cfg)
        report.update(sizing)
        report["config"] = cfg.to_dict()
        report["regrown_chains"] = [b.instance for b in matrix if b.regrown]
        report["intrachain"] = self._verify_intrachain(matrix, lengths, cfg)
        records = [self._record(b) for b in fillers + matrix]
        extra = {}
        for b in matrix:
            if b.regrown:
                from .chain_regrowth import regrown_class_text
                extra[f"{b.regrown_class}.lt"] = regrown_class_text(
                    b.lt_path, b.regrown_class, b.regrown)
        logger.info(
            f"Nanocomposite placement complete: {len(fillers)} filler and "
            f"{len(matrix)} matrix instances in a "
            f"{lengths[0]:.2f} x {lengths[1]:.2f} x {lengths[2]:.2f} A box; "
            f"min filler-filler {report['min_distance']['filler_filler']}, "
            f"filler-matrix {report['min_distance']['filler_matrix']}, "
            f"matrix-matrix {report['min_distance']['matrix_matrix']} A"
        )
        return PlacementResult(records=records, box_bounds=bounds,
                               report=report, extra_lt_files=extra)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    @staticmethod
    def _bodies(units, build_dir) -> Tuple[List[_Body], List[_Body]]:
        if build_dir is None:
            raise ValidationError(
                "'nanocomposite_mc' needs the build directory of the units "
                "(per-unit coordinate files)"
            )
        fillers: List[_Body] = []
        matrix: List[_Body] = []
        poly_k = mol_k = 0
        for u in units.units:
            if u.role == "graft":
                continue   # instantiated by its filler's composite class
            if not u.coords_file:
                raise ValidationError(
                    f"Unit '{u.id}' has no coords_file (pre-v2 build): re-run "
                    "stage 2 (UnitTyper) to write unit coordinates"
                )
            el, xyz = read_xyz(build_dir / u.coords_file)
            if u.kind == "inorganic":
                n_core = (u.envelope or {}).get("n_core_atoms")
                for k in range(u.count):
                    fillers.append(_Body(u.id, f"{u.id}_{k + 1}",
                                         u.class_name, xyz, el, True,
                                         n_core=n_core))
            elif u.kind == "polymer":
                chain = None
                clash = 0
                if u.chain_file:
                    from .chain_regrowth import (
                        ChainTemplate,
                        intrachain_contacts,
                    )
                    chain = ChainTemplate.load(build_dir / u.chain_file)
                    # pre-grown (stage-1) conformations may contain
                    # atom-level self-overlaps; such chains are regrown
                    clash, _ = intrachain_contacts(
                        xyz, chain.chain_bonds(), INTRA_CHECK_CUTOFF,
                        cutoff14=INTRA_CHECK_CUTOFF_14)
                for _ in range(u.count):
                    poly_k += 1
                    matrix.append(_Body(u.id, f"polymer_{poly_k}",
                                        u.class_name, xyz, el, False,
                                        chain=chain,
                                        lt_path=build_dir / u.lt_file,
                                        intra_clash=clash))
            else:
                for _ in range(u.count):
                    mol_k += 1
                    matrix.append(_Body(u.id, f"molecule_{mol_k}",
                                        u.class_name, xyz, el, False))
        if not fillers:
            raise ValidationError(
                "'nanocomposite_mc' found no filler units; pass a "
                "Nanoparticle to generate() or use 'mc_random'"
            )
        # place large instances first (better acceptance)
        matrix.sort(key=lambda b: -len(b.elements))
        return fillers, matrix

    @staticmethod
    def _box(ctx, cfg, units):
        from ..materials.structure import element_mass

        matrix_mass = 0.0
        for u in units.units:
            if u.kind != "inorganic" and u.role != "graft":
                m = u.mass
                if m is None:
                    raise ValidationError(f"Unit '{u.id}' has no mass")
                matrix_mass += m * u.count
        filler_core = sum((u.envelope or {}).get("core_volume") or 0.0
                          for u in units.filler_units for _ in range(u.count))
        filler_mass = sum((u.mass or 0.0) * u.count
                          for u in units.filler_units)
        graft_mass = sum(((u.material or {}).get("grafts") or {}).get(
            "graft_mass", 0.0) * u.count for u in units.filler_units)
        dims = ctx.box.box_dims or (None, None, None)
        cubic = ctx.box.requested_box_size
        if any(d is not None for d in dims) or cubic is not None:
            auto = None
            lengths = []
            for d in dims:
                if d is not None:
                    lengths.append(float(d))
                elif cubic is not None:
                    lengths.append(float(cubic))
                else:
                    if auto is None:
                        auto = NanocompositeMCStrategy._auto_side(
                            matrix_mass, filler_core, cfg)
                    lengths.append(auto)
            how = "explicit"
        else:
            side = NanocompositeMCStrategy._auto_side(matrix_mass,
                                                      filler_core, cfg)
            lengths = [side, side, side]
            how = f"initial_density {cfg.initial_density} g/cm^3"
        lengths = np.array(lengths, dtype=float)
        volume = float(np.prod(lengths))
        matrix_volume = volume - filler_core
        sizing = {
            "box": {"lengths": lengths.tolist(), "volume_A3": volume,
                    "sizing": how},
            "loading": {
                "n_fillers": {u.id: u.count for u in units.filler_units},
                "filler_core_volume_A3": filler_core,
                "nominal_filler_volume_fraction": filler_core / volume,
                "filler_mass_amu": filler_mass,
                "graft_shell_mass_amu": graft_mass,
                "core_mass_amu": filler_mass - graft_mass,
                "matrix_mass_amu": matrix_mass,
                "filler_weight_fraction": filler_mass /
                (filler_mass + matrix_mass) if filler_mass + matrix_mass
                else 0.0,
                "initial_matrix_density_gcm3": (
                    matrix_mass / (matrix_volume / AMU_PER_GCM3_TO_A3)
                    if matrix_volume > 0 and matrix_mass > 0 else 0.0),
                "target_polymer_density_gcm3": cfg.target_polymer_density,
                "note": "nominal geometric loading (filler core volume / "
                        "box); weight fraction includes surface groups; "
                        "measure the apparent loading after equilibration",
            },
        }
        _ = element_mass  # (masses come from units.json)
        lo = -0.5 * lengths
        return lo, lengths, sizing

    @staticmethod
    def _auto_side(matrix_mass, filler_core, cfg) -> float:
        v_matrix = matrix_mass / cfg.initial_density * AMU_PER_GCM3_TO_A3
        return float((v_matrix + filler_core) ** (1.0 / 3.0))

    @staticmethod
    def _check_box(fillers, matrix, lengths, cfg) -> None:
        """
        Flag bodies that may touch their own periodic image.

        A body whose diameter bound 2*max|r - centroid| plus its clearance
        is below the smallest box side can never meet its own image (fast
        path).  Larger bodies (long chains in small boxes) get an exact
        per-trial self-image contact test; fillers must always fit
        (a nanoparticle wider than the box is not a dispersed filler).
        """
        for body in fillers + matrix:
            c = body.local.mean(axis=0)
            diam = 2.0 * float(np.linalg.norm(body.local - c, axis=1).max())
            clearance = cfg.min_gap if body.is_filler else cfg.polymer_clearance
            body.check_self_image = diam + clearance >= lengths.min()
            # the smallest extent of the body (e.g. sheet thickness) must
            # fit in any orientation search: compare the thinnest
            # principal extent with the box
            c_loc = body.local - c
            thin = float(np.ptp(c_loc @ np.linalg.svd(
                c_loc, full_matrices=False)[2].T, axis=0).min())
            if body.is_filler and thin + clearance >= lengths.min():
                raise PackingError(
                    f"filler '{body.unit_id}' (thinnest extent {thin:.1f} A) "
                    f"cannot fit in the box (min side {lengths.min():.1f} A) "
                    f"with min_gap {cfg.min_gap} A to its own periodic image; "
                    "enlarge the box (lower initial_density / explicit "
                    "box_dims) or use smaller particles"
                )
            if diam >= 2.0 * lengths.min():
                raise PackingError(
                    f"matrix unit '{body.unit_id}' (diameter {diam:.1f} A) "
                    f"spans more than twice the box (min side "
                    f"{lengths.min():.1f} A); enlarge the box or use coiled "
                    "chains (GeometryConfig(use_mc_chain_growth=True))"
                )

    @staticmethod
    def _self_image_clash(world: np.ndarray, lengths: np.ndarray,
                          clearance: float) -> bool:
        """True if two atoms of one body meet through a periodic image."""
        from ..materials.topology import neighbor_pairs

        cut = min(clearance, 0.499 * float(lengths.min()))
        i, j, img, d = neighbor_pairs(world - world.min(axis=0), cut,
                                      np.diag(lengths), (True, True, True))
        return bool(np.any(np.abs(img).sum(axis=1) > 0))

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------
    @staticmethod
    def _place_filler(body, k, fillers, gap_grid, lo, lengths, cfg, rng,
                      soft_grid=None):
        tol = cfg.max_initial_overlap
        core = body.core_mask
        for attempt in range(cfg.max_attempts):
            body.rotation = random_rotation(rng)
            center = lo + rng.random(3) * lengths
            body.translation = center
            world = body.world()
            if gap_grid.hits(world[core]):
                continue
            if soft_grid is not None and soft_grid.hits(world):
                continue
            if body.check_self_image and \
                    NanocompositeMCStrategy._self_image_clash(
                        world[core], lengths, cfg.min_gap):
                continue
            gap_grid.mark(world[core], max(cfg.min_gap - tol, 1e-3))
            if soft_grid is not None:
                soft_grid.mark(world, cfg.polymer_clearance)
                # shell atoms also stay clear of other cores
                gap_grid.mark(world[~core], cfg.filler_clearance)
            return
        raise PackingError(
            f"Could not place filler instance {body.instance} "
            f"({k + 1}/{len(fillers)}) after {cfg.max_attempts} attempts "
            f"with min_gap {cfg.min_gap} A: free volume "
            f"{100 * gap_grid.free_fraction():.1f}% of the box. Reduce the "
            "filler count/size, enlarge the box, or lower min_gap."
        )

    @staticmethod
    def _place_matrix(body, k, matrix, fillers, grid, lo, lengths, cfg, rng):
        mode = cfg.matrix_placement
        regrowable = body.chain is not None and body.chain.topology != "ring"
        best_hits = None
        if mode == "auto" and regrowable and body.intra_clash:
            mode = "regrow"   # never keep a self-overlapping conformation
        if mode == "rigid" or not regrowable or mode == "auto":
            attempts = (cfg.rigid_attempts if mode == "auto" and regrowable
                        else cfg.max_attempts)
            ok, best_hits = NanocompositeMCStrategy._rigid_trials(
                body, grid, lengths, cfg, rng, attempts)
            if ok:
                return
        if regrowable and mode in ("auto", "regrow"):
            from .chain_regrowth import ObstacleAwareGrower
            grower = ObstacleAwareGrower(
                grid, lengths, intra=cfg.intra_clearance,
                intra14=cfg.intra14_clearance)
            for _ in range(cfg.regrow_starts):
                transforms = grower.grow(body.chain, rng)
                if transforms is not None:
                    body.regrown = transforms
                    body.override = np.vstack([
                        body.chain.local(m) @ R.T + t
                        for m, (R, t) in enumerate(transforms)])
                    return
        raise PackingError(
            f"Could not place matrix instance {body.instance} "
            f"({body.unit_id}, {len(body.elements)} atoms; {k}/{len(matrix)} "
            f"matrix instances placed; mode '{mode}'): "
            f"{100 * grid.free_fraction():.1f}% of the box is free at "
            f"clearances filler {cfg.filler_clearance} / polymer "
            f"{cfg.polymer_clearance} A (best rigid trial still had "
            f"{best_hits} blocked atoms"
            + (f"; {cfg.regrow_starts} in-box regrowth attempts failed"
               if regrowable and mode != "rigid" else "")
            + "). Suggested sequence: lower initial_density (build dilute, "
              "compress in MD), use MC chain growth "
              "(GeometryConfig(use_mc_chain_growth=True)) for compact coils, "
              "reduce polymer_clearance, or increase regrow_starts / "
              "max_attempts."
        )

    @staticmethod
    def _rigid_trials(body, grid, lengths, cfg, rng, attempts):
        """Random rigid-body trials; (accepted, fewest blocked atoms)."""
        centroid = body.local.mean(axis=0)
        anchor = int(np.argmin(np.linalg.norm(body.local - centroid, axis=1)))
        best_hits = None
        for _ in range(attempts):
            body.rotation = random_rotation(rng)
            p = grid.random_free_point(rng)
            if p is None:
                break
            body.translation = p - body.rotation @ body.local[anchor]
            world = body.world()
            n = grid.n_hits(world)
            if n == 0 and body.check_self_image and \
                    NanocompositeMCStrategy._self_image_clash(
                        world, lengths, cfg.polymer_clearance):
                n = 1
            if n == 0:
                return True, 0
            if best_hits is None or n < best_hits:
                best_hits = n
        return False, best_hits

    # ------------------------------------------------------------------
    # Verification + output
    # ------------------------------------------------------------------
    @staticmethod
    def _verify(fillers, matrix, lengths, cfg) -> Dict[str, object]:
        bodies = fillers + matrix
        pos = np.vstack([b.world() for b in bodies])
        owner = np.concatenate([np.full(len(b.elements), k)
                                for k, b in enumerate(bodies)])
        is_filler = np.concatenate([b.core_mask for b in bodies])
        tol = cfg.max_initial_overlap + 1e-6
        cutoff = max(cfg.min_gap, cfg.filler_clearance,
                     cfg.polymer_clearance) + 2.0
        cutoff = min(cutoff, 0.499 * float(lengths.min()))
        i, j, d = min_image_contacts(pos, owner, lengths, cutoff)
        mins = {"filler_filler": None, "filler_matrix": None,
                "matrix_matrix": None}
        limits = {"filler_filler": cfg.min_gap,
                  "filler_matrix": cfg.filler_clearance,
                  "matrix_matrix": cfg.polymer_clearance}
        is_h = np.concatenate([[e == "H" for e in b.elements] for b in bodies])
        violations = {}
        for a, b, dd in zip(i, j, d):
            fa, fb = is_filler[a], is_filler[b]
            key = ("filler_filler" if fa and fb else
                   "matrix_matrix" if not (fa or fb) else "filler_matrix")
            if mins[key] is None or dd < mins[key]:
                mins[key] = float(dd)
            limit = limits[key]
            if key == "matrix_matrix" and (is_h[a] or is_h[b]):
                limit = cfg.hydrogen_clearance
            if dd < limit - tol:
                violations[key] = min(float(dd), violations.get(key, 1e9))
        if violations:
            raise PackingError(
                f"Placement verification failed (minimum-image contacts "
                f"below clearance): {violations}"
            )
        if mins["filler_matrix"] is None and is_filler.any() and \
                (~is_filler).any():
            # no contact within the cutoff: report the exact nearest
            # filler-matrix distance instead of None
            mins["filler_matrix"] = NanocompositeMCStrategy._nearest(
                pos, owner, is_filler, lengths)
        limits = dict(limits, matrix_hydrogen=cfg.hydrogen_clearance)
        return {
            "min_distance": {k: (None if v is None else round(v, 3))
                             for k, v in mins.items()},
            "clearances": limits,
            "pbc_checked": True,
            "n_atoms": int(len(pos)),
        }

    @staticmethod
    def _nearest(pos: np.ndarray, owner: np.ndarray, is_filler: np.ndarray,
                 lengths: np.ndarray) -> Optional[float]:
        """Minimum-image nearest distance between filler-core atoms and
        non-core atoms of other bodies (orthorhombic box); None without
        scipy."""
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return None
        # wrapped into [0, L) as cKDTree(boxsize=...) requires
        wrapped = np.mod(pos - pos.min(axis=0), lengths)
        wrapped = np.minimum(wrapped, np.nextafter(lengths, 0.0))
        best = None
        for k in np.unique(owner[is_filler]):
            core = is_filler & (owner == k)
            other = ~is_filler & (owner != k)
            if not other.any():
                continue
            d, _ = cKDTree(wrapped[other], boxsize=lengths).query(
                wrapped[core], k=1)
            best = float(d.min()) if best is None else min(best, float(d.min()))
        return best

    @staticmethod
    def _verify_intrachain(matrix, lengths, cfg) -> Dict[str, object]:
        """Atom-level self-overlap check of every placed chain (beyond 1-4)."""
        from .chain_regrowth import intrachain_contacts

        worst, n_bad, kept = None, 0, []
        for b in matrix:
            if b.chain is None:
                continue
            n, dmin = intrachain_contacts(b.world(), b.chain.chain_bonds(),
                                          INTRA_CHECK_CUTOFF, lengths,
                                          cutoff14=INTRA_CHECK_CUTOFF_14)
            if n:
                n_bad += 1
                kept.append(b.instance)
                worst = dmin if worst is None else min(worst, dmin)
        if n_bad and cfg.matrix_placement != "rigid":
            raise PackingError(
                f"{n_bad} placed chain(s) still self-overlap (closest "
                f"intra-chain contact {worst:.2f} A beyond 1-4): {kept[:5]}"
            )
        return {"cutoff": INTRA_CHECK_CUTOFF,
                "cutoff_14": INTRA_CHECK_CUTOFF_14,
                "overlapping_chains": n_bad,
                "min_distance": worst,
                "note": "pre-grown conformations kept as-is in 'rigid' mode"
                if n_bad else "no intra-chain contacts beyond 1-4"}

    @staticmethod
    def _record(body: _Body) -> PlacementRecord:
        if body.regrown:
            # regrown chains carry absolute (box-frame) monomer transforms
            return PlacementRecord(
                body.unit_id, body.instance,
                f"{body.instance} = new {body.regrown_class}")
        angle, ax, ay, az = matrix_to_axis_angle(body.rotation)
        x, y, z = body.translation
        if abs(angle) < 1e-9:
            cmd = (f"{body.instance} = new {body.class_name}"
                   f".move({x:.4f},{y:.4f},{z:.4f})")
        else:
            cmd = (f"{body.instance} = new {body.class_name}"
                   f".rot({angle:.6f},{ax:.6f},{ay:.6f},{az:.6f})"
                   f".move({x:.4f},{y:.4f},{z:.4f})")
        return PlacementRecord(body.unit_id, body.instance, cmd)
