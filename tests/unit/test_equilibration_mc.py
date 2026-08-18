# -*- coding: utf-8 -*-
"""Phase 0 validation: Rust MC kernel vs the numpy reference path.

Statistical parity between two independent correct samplers is judged on
thermodynamic observables (mean energy, MSID), not trajectory equality.
"""
import numpy as np
import pytest

from AutoPoly.equilibration import MeltState, MCRunner, MCParams


@pytest.fixture()
def melt():
    # 4 chains x N=50 at rho=0.85, relaxed overlap-free start
    return MeltState.random_walk(n_chains=4, n_per_chain=50, seed=0)


@pytest.fixture()
def relaxed_melt(melt):
    runner = MCRunner(
        melt,
        MCParams(move_weights={"displacement": 1.0}),
        seed=42,
    )
    runner.run(40_000)
    return runner.state


def test_state_invariants(melt):
    assert melt.validate()
    assert melt.n_beads == 200
    assert melt.monodisperse()
    assert melt.chain_lengths == [50] * 4


def test_serialization_roundtrip(melt, tmp_path):
    melt.save(tmp_path / "melt", meta={"purpose": "test"})
    loaded = MeltState.load(tmp_path / "melt")
    assert loaded.n_beads == melt.n_beads
    assert loaded.chains == melt.chains
    np.testing.assert_allclose(loaded.positions, melt.positions)
    assert loaded.box_size == melt.box_size


def test_energy_bookkeeping_production_clean(relaxed_melt):
    """Incremental vs exact energy must agree on production runs."""
    runner = MCRunner(relaxed_melt, MCParams(), seed=7)
    runner.run(50_000)
    assert abs(runner.bookkeeping_offset()) < 1e-6
    assert runner.validate()


def test_accepts_physical_moves(relaxed_melt):
    runner = MCRunner(relaxed_melt, MCParams(), seed=11)
    rates = runner.run(30_000)
    # displacement and crankshaft must have nonzero acceptance at melt density
    assert rates["displacement"] > 0.01
    assert rates["crankshaft"] > 0.01


def test_rust_matches_numpy_statistics(relaxed_melt):
    """
    Rust kernel vs numpy mc_equilibrate: same equilibrium statistics.

    Both sample the same ensemble from the same relaxed start; compare
    mean energy per bead and MSID(1) over production samples.
    """
    from AutoPoly.models.bead_spring import mc_equilibrate

    state = relaxed_melt
    n = state.n_beads

    # Snapshot the contiguous-start state BEFORE the Rust run (reptation
    # reorders chains, breaking the slice representation numpy needs).
    bonds = state.bonds()
    chain_indices = [(ch[0], ch[-1] + 1) for ch in state.chains]
    start_positions = [state.positions[i].copy() for i in range(n)]

    # Rust: 8 x 15k steps, sample energy after each block
    runner = MCRunner(state, MCParams(), seed=101)
    rust_e = []
    for _ in range(8):
        runner.run(15_000)
        rust_e.append(runner.recompute_energy() / n)
    rust_msid1 = dict(runner.msid(1))[1]

    # numpy reference: same model (harmonic k=100 r0=1, LJ unshifted 2.5)
    np_positions = start_positions
    np_e = []
    for _ in range(8):
        np_positions, _ = mc_equilibrate(
            np_positions,
            bonds,
            chain_indices=chain_indices,
            n_steps=15_000,
            temperature=1.0,
            box_size=state.box_size,
            lj_sigma=1.0,
            lj_epsilon=1.0,
            lj_cutoff=2.5,
            bond_k=100.0,
            bond_r0=1.0,
        )
        from AutoPoly.models.bead_spring import compute_total_energy

        np_e.append(
            compute_total_energy(
                np_positions, bonds, 1.0, 1.0, 2.5, 100.0, 1.0, state.box_size
            )
            / n
        )

    rust_mean, np_mean = np.mean(rust_e), np.mean(np_e)
    # Block means are correlated samples; compare via standard error of
    # the block means with a floor reflecting the correlated statistics.
    sem = np.sqrt(np.var(rust_e, ddof=1) / len(rust_e)
                  + np.var(np_e, ddof=1) / len(np_e))
    tol = max(0.2, 4.0 * sem)
    assert abs(rust_mean - np_mean) < tol, (
        f"energy/bead mismatch: rust={rust_mean:.3f} numpy={np_mean:.3f} "
        f"(sem={sem:.3f})"
    )
    # MSID(1) ~ mean bond length^2 ~ 1 for harmonic r0=1
    assert 0.8 < rust_msid1 < 1.4


def test_kremer_grest_flavor_runs():
    """FENE+WCA must relax a cold start while keeping bonds r < R0.

    (Harmonic bonds cannot be used for cold-start relaxation: hard-core
    repulsion blasts overlapped beads apart and over-stretches them.
    FENE's r >= R0 rejection is precisely what prevents this.)
    """
    melt = MeltState.random_walk(n_chains=4, n_per_chain=50, seed=0)
    runner = MCRunner(melt, MCParams.kremer_grest(), seed=5)
    runner.run(60_000)
    assert runner.validate()
    import math

    assert math.isfinite(runner.recompute_energy())
    pos = runner.state.positions
    for chain in runner.state.chains:
        for a, b in zip(chain, chain[1:]):
            d = pos[b] - pos[a]
            d -= runner.state.box_size * np.round(d / runner.state.box_size)
            assert np.linalg.norm(d) < 1.5  # FENE R0
