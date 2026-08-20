# -*- coding: utf-8 -*-
"""
Connectivity-annealing driver for atomistic/UA melts (Phase 3 protocol).

Protocol (validated on the C52 x 8 PE melt, 2026-08-20):
  1. soften LJ + dihedrals to the anneal point (0.5, 0.5),
  2. interleave local MC with multi-try (R=16) CBMC k-mer double-bridge
     attempts drawn from the reach enumerator (system-wide uniform),
  3. harden stepwise back to the production potential with local MC
     relaxation on each rung.

Design rationale (measured): pure-production all-atom CBMC acceptance
is ~e^-300 (24 regrown atoms x few-kT dilution each); on the TraPPE-UA
view at 50% softening with R=16 MTM it is ~4% per attempt (~14 s),
which is the workable regime — mirroring the KG Phase-2 finding that
connectivity moves belong at the soft end of the ramp.

Created on 2026-08-20
@author: zwu
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AnnealConfig:
    """Schedule for one connectivity-annealing run."""

    kbt: float = 0.894                    # kcal/mol (450 K)
    # anneal (soft) point and ramp back to production
    anneal_lj: float = 0.50
    anneal_dih: float = 0.50
    harden_steps: Tuple[Tuple[float, float], ...] = ((0.75, 0.75), (1.0, 1.0))
    # CBMC move parameters (validated on C52 x 8)
    r_reach: float = 4.8
    k_regrow: int = 3
    n_trials: int = 30
    n_psi: int = 144
    n_mtm: int = 16
    # work split
    local_between_attempts: int = 4_000   # local MC steps between attempts
    soften_equil_steps: int = 200_000     # local MC after entering soft level
    harden_equil_steps: int = 100_000     # local MC per hardening rung
    n_attempts: int = 500                 # CBMC attempts at the soft level
    r_reach_soft: Optional[float] = None  # override reach at soft level
    seed: int = 12345
    log_every: int = 25


@dataclass
class AnnealStats:
    attempts: int = 0
    accepts: int = 0
    no_cand_rounds: int = 0
    log_ratios: List[float] = field(default_factory=list)
    wall_time_s: float = 0.0

    @property
    def acceptance(self) -> float:
        return self.accepts / max(self.attempts, 1)


def anneal_connectivity(melt, cfg: AnnealConfig) -> Tuple[AnnealStats, Dict[str, Any]]:
    """Run the soften -> CBMC-anneal -> harden protocol on `melt`
    (an AtomisticMelt; UA view for the UA-level anneal). Positions and
    topology are modified in place. Returns (stats, level_records).
    """
    import numpy as _np

    rng = _np.random.default_rng(cfg.seed)
    n_chains = len(melt.backbone_chains)
    eps0 = _np.asarray(melt._engine.pair_eps())
    dih0 = _np.asarray(melt._engine.dih_k())
    stats = AnnealStats()
    records: List[Dict[str, Any]] = []
    t_start = time.time()

    def set_level(lj: float, dih: float) -> None:
        melt._engine.set_soft_params((eps0 * lj).tolist(), (dih0 * dih).tolist())

    # ---- soften + equilibrate ----
    set_level(cfg.anneal_lj, cfg.anneal_dih)
    melt.run(cfg.soften_equil_steps)
    records.append({"stage": "soften", "lj": cfg.anneal_lj, "dih": cfg.anneal_dih,
                    "energy": melt.recompute_energy()})

    # ---- anneal ----
    reach = cfg.r_reach_soft or cfg.r_reach
    t0 = time.time()
    while stats.attempts < cfg.n_attempts:
        melt.run(cfg.local_between_attempts)
        cands = [
            c
            for a in range(n_chains)
            for c in melt._engine.cbmc_candidates(a, reach, cfg.k_regrow)
        ]
        if not cands:
            stats.no_cand_rounds += 1
            continue
        a, b, s, flip = cands[int(rng.integers(len(cands)))]
        ok, lr = melt._engine.cbmc_bridge_attempt(
            a, b, s, flip, cfg.kbt, reach, cfg.n_trials, cfg.n_psi,
            cfg.k_regrow, cfg.n_mtm,
        )
        stats.attempts += 1
        stats.log_ratios.append(lr)
        stats.accepts += int(ok)
        if stats.attempts % cfg.log_every == 0:
            print(f"  anneal {stats.attempts}/{cfg.n_attempts}: "
                  f"acc {stats.accepts} ({100.0*stats.acceptance:.1f}%), "
                  f"no-cand {stats.no_cand_rounds}, "
                  f"{(time.time()-t0)/stats.attempts:.1f}s/att", flush=True)
    stats.wall_time_s = time.time() - t0
    records.append({"stage": "anneal", "accepts": stats.accepts,
                    "attempts": stats.attempts,
                    "acceptance": stats.acceptance,
                    "energy": melt.recompute_energy()})

    # ---- harden back to production ----
    for lj, dih in cfg.harden_steps:
        set_level(lj, dih)
        melt.run(cfg.harden_equil_steps)
        records.append({"stage": "harden", "lj": lj, "dih": dih,
                        "energy": melt.recompute_energy()})

    stats.wall_time_s = time.time() - t_start
    return stats, {"records": records}
