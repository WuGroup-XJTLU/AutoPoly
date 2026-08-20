# -*- coding: utf-8 -*-
"""
Ladder schedules for growth-assisted equilibration.

Implements the "grow-relax-rewire" protocol (see equilibration_method.md):

1. Start from short oligomers at the target density (overlap-free start
   is easy at short N).
2. Growth levels: join equal-length chains end-to-end (P_n + P_n ->
   P_2n), doubling the contour length per level. Joins run at a softened
   core (generalized-LJ exponent n_pair < 12) where junction acceptance
   is affordable.
3. Annealing: local MC mixed with directed segment-exchange (heat-bath)
   topology swaps. Swaps fire almost exclusively in the soft-core
   window (n_pair <= ~5); the schedule spends the swap budget there.
4. Softness ramp: n_pair = 2 -> 3 -> ... -> 12 (standard KG WCA) with
   k_FENE(n_pair) re-solved at each level to hold the mechanical-
   equilibrium bond length at 0.9609 sigma (Dietz & Hoy 2022, Eqs. 8-9).
5. Final relaxation at n_pair = 12 followed by certification
   (``certify`` module).

The growth stage (joins) is heuristic by design; the equilibrium
guarantee comes from the fixed-N annealing + certification, not from
the growth dynamics.

Created on 2026-08-18
@author: zwu
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from typing import Any, Dict, List, Optional

from ..core.system import logger
from .runner import MCParams, MCRunner
from .state import MeltState

#: Mechanical-equilibrium KG bond length (Kremer-Grest convention).
ELL0 = 0.960897

#: Softness ramp (Dietz & Hoy 2022 set, skipping the n=6 singularity).
DEFAULT_RAMP = [2.0, 3.0, 4.0, 5.0, 5.75, 6.5, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]


def k_fene_for_mie(n: float, r0max: float = 1.5, ell0: float = ELL0) -> float:
    """
    FENE spring constant giving mechanical-equilibrium bond length
    ``ell0`` for the generalized-LJ pair potential with repulsive
    exponent ``n`` (m = 6). Solves d[U_pair + U_FENE]/dr = 0 at ell0.
    Returns 30.0 at n = 12 (standard KG).
    """
    if abs(n - 6.0) < 1e-9:
        raise ValueError("n=6 is the singular point of the (n,6) form")
    c1 = 6.0 * 2.0 ** (n / 6.0) / (n - 6.0)
    c2 = 2.0 * n / (n - 6.0)
    srn = (1.0 / ell0) ** n
    sr6 = (1.0 / ell0) ** 6
    dU_pair = (-n * c1 * srn + 6.0 * c2 * sr6) / ell0
    return -dU_pair * (r0max**2 - ell0**2) / (ell0 * r0max**2)


def params_at_softness(n_pair: float, angle_k: float = 0.0) -> MCParams:
    """KG parameters (FENE + WCA-style generalized LJ) at core softness n_pair."""
    return MCParams(
        pair_wca=True,
        exclude_bonded=False,
        mie_n=n_pair,
        bond_model="fene",
        bond_k=k_fene_for_mie(n_pair),
        fene_r0max=1.5,
        angle_k=angle_k,
    )


@dataclass
class LadderConfig:
    """Configuration for one growth + annealing ladder run."""

    # Softness ramp (pair exponents); swaps concentrate at the soft end.
    ramp: List[float] = field(default_factory=lambda: list(DEFAULT_RAMP))
    # Steps per (level, ramp-point): local+swap MC steps.
    mix_steps: int = 20_000
    # Phase-2 budget allocation (acceptance is 10-13% at the soft end,
    # ~1.5-3% at n=12, so the topology annealing budget belongs there):
    #   n_pair <= soft_until          -> soft_mix_steps
    #   n_pair in mid_hold_ns         -> mid_hold_steps with heavier swaps
    #   otherwise                     -> mix_steps
    soft_until: float = 5.0
    # None = uniform mix_steps everywhere (original behavior)
    soft_mix_steps: Optional[int] = None
    mid_hold_ns: tuple = (5.75, 6.5, 7.0)
    mid_hold_steps: Optional[int] = None
    mid_hold_swap_weight: float = 0.6
    # Anneal steps after each growth level (None = mix_steps)
    growth_anneal_steps: Optional[int] = None
    # Join search radius and per-level attempt budget multiplier.
    join_radius: float = 1.3
    join_rounds: int = 200
    # Swap search radius (junction length ceiling, Dietz-Hoy use 1.3).
    swap_r_max: float = 1.3
    # DBH acceptance convention (bond+angle ΔU) is the default; set True
    # for the strict full-ΔU convention (much lower acceptance).
    swap_full_delta: bool = False
    # Move mix during annealing blocks.
    anneal_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "displacement": 0.25,
            "pivot": 0.12,
            "crankshaft": 0.08,
            "reptation": 0.05,
            "translation": 0.05,
            "rotation": 0.05,
            "segment_exchange": 0.40,
        }
    )
    # Move mix while joins are in progress (no swaps needed there).
    join_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "displacement": 0.4,
            "pivot": 0.2,
            "crankshaft": 0.15,
            "reptation": 0.05,
            "translation": 0.1,
            "rotation": 0.1,
        }
    )


class GrowthAnnealer:
    """
    Drives the molecular-weight annealing ladder on a :class:`MeltState`.

    The starting state must be monodisperse with chain length N0 and a
    chain count divisible by 2^k for the k levels to the target.
    """

    def __init__(
        self,
        state: MeltState,
        config: Optional[LadderConfig] = None,
        angle_k: float = 0.0,
        seed: int = 12345,
    ) -> None:
        state.assert_monodisperse()
        self.state = state
        self.config = config or LadderConfig()
        self.angle_k = angle_k
        self.seed = seed
        self.runner: Optional[MCRunner] = None
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Engine management
    # ------------------------------------------------------------------ #
    def _make_runner(self, n_pair: float, weights: Dict[str, float]) -> MCRunner:
        params = params_at_softness(n_pair, self.angle_k)
        params.move_weights = dict(weights)
        runner = MCRunner(self.state, params, seed=self.seed)
        runner._engine = None  # rebuild with extra kwargs below
        self.seed += 1
        self._build(runner, params, weights)
        return runner

    def _build(self, runner: MCRunner, params: MCParams, weights: Dict[str, float]) -> None:
        import autopoly_mc

        p = params
        positions, chains, box = self.state.to_engine_args()
        runner._engine = autopoly_mc.McEngine(
            positions,
            chains,
            box,
            seed=self.seed,
            pair_epsilon=p.pair_epsilon,
            pair_sigma=p.pair_sigma,
            pair_cutoff=p.pair_cutoff,
            pair_shifted=p.pair_shifted,
            pair_wca=p.pair_wca,
            mie_n=p.mie_n,
            exclude_bonded=p.exclude_bonded,
            bond_model=p.bond_model,
            bond_k=p.bond_k,
            bond_r0=p.bond_r0,
            fene_r0max=p.fene_r0max,
            angle_k=p.angle_k,
            temperature=p.temperature,
            max_displacement=p.max_displacement,
            max_angle=p.max_angle,
            swap_r_max=self.config.swap_r_max,
            swap_full_delta=self.config.swap_full_delta,
            move_weights=weights,
        )

    def _sync(self) -> None:
        assert self.runner is not None
        self.state.update_from_engine(self.runner._engine)

    # ------------------------------------------------------------------ #
    # Growth
    # ------------------------------------------------------------------ #
    def _join_level(self, level_len: int, n_pair: float) -> Dict[str, Any]:
        """Join all chains of contour length `level_len` pairwise."""
        rounds = 0
        engine = self.runner._engine
        while True:
            n_level = sum(1 for l in engine.chain_lengths() if l == level_len)
            if n_level < 2:
                break
            progressed = False
            for _ in range(50):
                if sum(1 for l in engine.chain_lengths() if l == level_len) < 2:
                    break
                if engine.try_join(self.config.join_radius, level_len):
                    progressed = True
            engine.run(self.config.mix_steps // 4)
            rounds += 1
            if not progressed:
                if rounds > self.config.join_rounds:
                    n_left = self._force_join_stragglers(level_len)
                    logger.info(
                        f"level {level_len}: forced {n_left} straggler joins "
                        f"(heuristic, relaxed afterwards)"
                    )
                    self._rebuild_current()
                    engine = self.runner._engine
                    if n_left == 0:
                        break
        self._sync()
        self.state.assert_monodisperse()
        return {"level_from": level_len, "level_to": 2 * level_len,
                "n_chains": self.state.n_chains, "join_rounds": rounds}

    def _rebuild_current(self) -> None:
        """Rebuild the engine after direct state manipulation."""
        n_pair = self.runner.params.mie_n
        weights = dict(self.runner.params.move_weights)
        self.runner = self._make_runner(n_pair, weights)

    def _force_join_stragglers(self, level_len: int) -> int:
        """
        Heuristic last resort for pairs that never meet by diffusion:
        rigidly translate one chain so an eligible end sits at the
        equilibrium bond length from a partner's end, then merge. Only
        valid at softened core (overlaps created are cheap); the caller
        relaxes immediately afterwards. Returns the number of forced
        joins performed (0 if none needed).
        """
        self._sync()
        idx_by_len = [
            i for i, c in enumerate(self.state.chains) if len(c) == level_len
        ]
        if len(idx_by_len) < 2:
            return 0
        pos = self.state.positions
        box = self.state.box_size
        n_forced = 0
        # pair them off in order
        while len(idx_by_len) >= 2:
            a = idx_by_len.pop(0)
            b = idx_by_len.pop(0)
            ca = self.state.chains[a]
            cb = self.state.chains[b]
            ea = ca[-1]  # join at A's last, B's first
            eb = cb[0]
            pa = pos[ea].copy()
            d0 = pos[ca[-2]] - pa if len(ca) > 1 else np.zeros(3)
            d0 -= box * np.round(d0 / box)
            nrm = np.linalg.norm(d0)
            dirv = d0 / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])
            target = (pa + ELL0 * dirv) % box
            shift = target - pos[eb]
            shift -= box * np.round(shift / box)
            for bead in cb:
                pos[bead] = (pos[bead] + shift) % box
            self.state.chains[a] = ca + cb
            del self.state.chains[b]
            idx_by_len = [i if i < b else i - 1 for i in idx_by_len]
            n_forced += 1
        self.state.positions = pos
        self.state.validate()
        return n_forced

    # ------------------------------------------------------------------ #
    # Annealing
    # ------------------------------------------------------------------ #
    def _budget_for(self, n_pair: float) -> tuple:
        """(steps, move_weights) for a ramp point."""
        cfg = self.config
        if cfg.soft_mix_steps is not None and n_pair <= cfg.soft_until:
            return cfg.soft_mix_steps, dict(cfg.anneal_weights)
        if cfg.mid_hold_steps is not None and any(
            abs(n_pair - m) < 1e-9 for m in cfg.mid_hold_ns
        ):
            w = dict(cfg.anneal_weights)
            w.pop("segment_exchange", None)
            total = sum(w.values())
            w = {k: v / total * (1.0 - cfg.mid_hold_swap_weight)
                 for k, v in w.items()}
            w["segment_exchange"] = cfg.mid_hold_swap_weight
            return cfg.mid_hold_steps, w
        return cfg.mix_steps, dict(cfg.anneal_weights)

    def _anneal_level(self, n_pair: float) -> Dict[str, Any]:
        steps, weights = self._budget_for(n_pair)
        self.runner = self._make_runner(n_pair, weights)
        self.runner.run(steps)
        self._sync()
        rates = self.runner._engine.acceptance_rates()
        offset = self.runner.bookkeeping_offset()
        return {"n_pair": n_pair, "steps": steps,
                "acceptance": rates, "energy_offset": offset}

    # ------------------------------------------------------------------ #
    # Full ladder
    # ------------------------------------------------------------------ #
    def grow_to(
        self,
        target_len: int,
        ramp_start: float = 3.0,
        final_relax_steps: Optional[int] = None,
    ) -> MeltState:
        """
        Grow from the current (monodisperse) chain length to
        ``target_len`` by repeated doubling, then ramp the core softness
        to n_pair = 12 and relax. Returns the state (also in-place).
        """
        n0 = self.state.chain_lengths[0]
        if target_len % n0 != 0 or (target_len // n0) & (target_len // n0 - 1):
            raise ValueError(
                f"target_len {target_len} must be a power-of-two multiple "
                f"of the current chain length {n0}"
            )

        # --- growth levels at soft core ---
        level = n0
        while level < target_len:
            self.runner = self._make_runner(ramp_start, self.config.join_weights)
            rec = self._join_level(level, ramp_start)
            logger.info(f"joined {rec['level_from']} -> {rec['level_to']}: "
                        f"{rec['n_chains']} chains after {rec['join_rounds']} rounds")
            self.history.append({"stage": "join", **rec})
            # anneal at the new level (soft)
            self.runner = self._make_runner(ramp_start, self.config.anneal_weights)
            gsteps = self.config.growth_anneal_steps or self.config.mix_steps
            self.runner.run(gsteps)
            self._sync()
            out = {"n_pair": ramp_start, "steps": gsteps,
                   "acceptance": self.runner._engine.acceptance_rates(),
                   "energy_offset": self.runner.bookkeeping_offset()}
            self.history.append({"stage": "anneal", "level": 2 * level, **out})
            level *= 2

        # --- softness ramp to production hardness ---
        ramp = [n for n in self.config.ramp if n > ramp_start]
        for n_pair in ramp:
            out = self._anneal_level(n_pair)
            self.history.append({"stage": "ramp", **out})
            logger.info(f"ramp n={n_pair}: swap acc "
                        f"{out['acceptance'].get('segment_exchange', 0.0):.4f}")

        # --- final relaxation at n_pair = 12 (swaps included: they
        # still fire at a few percent at full hardness and keep the
        # topology mixing) ---
        steps = final_relax_steps or 2 * self.config.mix_steps
        self.runner = self._make_runner(12.0, self.config.anneal_weights)
        self.runner.run(steps)
        self._sync()
        self.state.validate()
        return self.state
