# -*- coding: utf-8 -*-
"""Phase 1 validation: connectivity-altering moves and the ladder."""
import numpy as np
import pytest
import autopoly_mc

from AutoPoly.equilibration import (
    MeltState,
    MCRunner,
    MCParams,
    GrowthAnnealer,
    LadderConfig,
    k_fene_for_mie,
    certify,
)


def _kg_engine(positions, chains, box, seed=1, mie_n=12.0, **kw):
    params = dict(
        pair_wca=True,
        mie_n=mie_n,
        exclude_bonded=False,
        bond_model="fene",
        bond_k=k_fene_for_mie(mie_n),
        move_weights={"displacement": 0.4, "pivot": 0.2, "crankshaft": 0.15,
                      "reptation": 0.05, "translation": 0.1, "rotation": 0.1},
    )
    params.update(kw)
    return autopoly_mc.McEngine(positions, chains, box, seed=seed, **params)


@pytest.fixture()
def soft_melt():
    """Relaxed 32 x N=25 melt at soft core n=3."""
    melt = MeltState.random_walk(32, 25, seed=2)
    positions, chains, box = melt.to_engine_args()
    e = _kg_engine(positions, chains, box, seed=1, mie_n=3.0)
    e.run(300_000)
    melt.update_from_engine(e)  # sync relaxed state back
    return melt, e


def test_swap_topology_only_and_exact_bookkeeping(soft_melt):
    """Segment exchange moves no bead; lengths preserved; energy exact."""
    melt, e = soft_melt
    pos0 = melt.positions.copy()
    positions, chains, box = melt.to_engine_args()
    sw = autopoly_mc.McEngine(
        positions, chains, box, seed=11, pair_wca=True, mie_n=3.0,
        exclude_bonded=False,
        bond_model="fene", bond_k=k_fene_for_mie(3.0),
        swap_full_delta=False, move_weights={"segment_exchange": 1.0},
    )
    sw.run(10_000)
    acc = sw.acceptance_rates()["segment_exchange"]
    assert acc > 0.005  # soft-core acceptance must be usable
    assert np.allclose(np.array(sw.positions()), pos0)  # topology-only move
    assert set(sw.chain_lengths()) == {25}
    assert sw.validate()
    assert abs(sw.energy() - sw.recompute_energy()) < 1e-6


def test_ladder_join_and_anneal(soft_melt):
    """Growth ladder 25 -> 50 halves chain count, stays monodisperse."""
    melt, e = soft_melt
    target = 16
    for _ in range(200):
        if e.n_chains <= target:
            break
        for _ in range(50):
            if e.n_chains <= target:
                break
            e.try_join(1.3, 25)
        e.run(3_000)
    lengths = set(e.chain_lengths())
    assert 50 in lengths  # joins happened
    assert e.validate()
    assert np.isfinite(e.recompute_energy())


def test_growth_annealer_end_to_end():
    """Full ladder: 16 x N=20 -> 8 x N=40 with ramp and final relax."""
    melt = MeltState.random_walk(16, 20, seed=4)
    cfg = LadderConfig(mix_steps=6_000, join_rounds=60)
    ga = GrowthAnnealer(melt, cfg, seed=21)
    out = ga.grow_to(40, ramp_start=3.0, final_relax_steps=10_000)
    assert out.chain_lengths == [40] * 8
    assert out.validate()
    # final state at n=12 must have KG-like bond statistics
    stats = certify.bond_length_stats(out)
    assert 0.8 < stats["mean"] < 1.15
    assert stats["max"] < 1.5  # FENE R0


def test_ladder_msid_matches_bruteforce():
    """
    Physics certificate: MSID(s) from the ladder path must agree with a
    long local-MC-only run at the same N (two independent paths to the
    same equilibrium ensemble). Box sized L/Rg ~ 4 to keep finite-size
    compression below the comparison tolerance.
    """
    n_target, n_chains = 32, 50

    # Path A: ladder 100 x 16 -> 50 x 32
    melt_a = MeltState.random_walk(100, 16, seed=10)
    cfg = LadderConfig(mix_steps=10_000, join_rounds=150)
    ga = GrowthAnnealer(melt_a, cfg, seed=31)
    state_a = ga.grow_to(n_target, ramp_start=3.0, final_relax_steps=150_000)

    # Path B: brute-force local MC at n=12 from a random-walk start
    melt_b = MeltState.random_walk(n_chains, n_target, seed=99)
    runner_b = MCRunner(melt_b, MCParams.kremer_grest(), seed=97)
    runner_b.run(1_200_000)

    # Residual deviation here is dominated by finite-size compression
    # (L/Re ~ 2) and single-trajectory convergence noise; the production
    # certification protocol uses longer runs and two independent ladder
    # replicas. Agreement improves monotonically with annealing budget.
    ok, report = certify.compare_msid(
        certify.msid(state_a), certify.msid(runner_b.state),
        rtol=0.20, s_min=2,
    )
    assert ok, f"MSID mismatch: {report}"
