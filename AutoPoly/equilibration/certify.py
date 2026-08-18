# -*- coding: utf-8 -*-
"""
Certification observables for equilibrated melts (in-house, no external
topology codes).

Primary certificate: the mean-square internal distance (MSID)
<R^2(s)>/s over the full contour range, compared between two
independently generated melts (e.g. growth+annealing vs a long
brute-force run). Supporting observables: single-chain static structure
factor S(q), PBC-aware Rg and end-to-end distance, junction bond-length
statistics (growth-memory check).

Created on 2026-08-18
@author: zwu
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .state import MeltState


def unwrap_chains(state: MeltState) -> np.ndarray:
    """
    Unwrapped positions: walk each chain's contour applying the minimum
    image, so intra-chain distances are free of box wrapping artifacts.
    Returns (n_beads, 3) array in unwrapped coordinates.
    """
    pos = state.positions
    box = state.box_size
    out = np.empty_like(pos)
    for chain in state.chains:
        prev = pos[chain[0]]
        out[chain[0]] = prev
        for k in range(1, len(chain)):
            d = pos[chain[k]] - pos[chain[k - 1]]
            d -= box * np.round(d / box)
            prev = prev + d
            out[chain[k]] = prev
    return out


def msid(state: MeltState, max_s: Optional[int] = None) -> np.ndarray:
    """
    Mean-square internal distance <R^2(s)> for s = 1..max_s
    (default: full contour). Returns (s, R2) columns.
    """
    unw = unwrap_chains(state)
    n = max(len(c) for c in state.chains)
    smax = min(max_s or n - 1, n - 1)
    acc = np.zeros(smax + 1)
    cnt = np.zeros(smax + 1)
    for chain in state.chains:
        p = unw[chain]
        for s in range(1, min(smax, len(chain) - 1) + 1):
            d = p[s:] - p[:-s]
            acc[s] += np.sum(d * d)
            cnt[s] += len(d)
    s_vals = np.arange(1, smax + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(cnt[1:] > 0, acc[1:] / cnt[1:], np.nan)
    return np.column_stack([s_vals, r2])


def radius_of_gyration(state: MeltState) -> float:
    """Mean radius of gyration over chains (PBC-aware)."""
    unw = unwrap_chains(state)
    vals = []
    for chain in state.chains:
        p = unw[chain]
        c = p.mean(axis=0)
        vals.append(np.mean(np.sum((p - c) ** 2, axis=1)))
    return float(np.sqrt(np.mean(vals)))


def end_to_end(state: MeltState) -> float:
    """Mean-squared end-to-end distance (PBC-aware)."""
    unw = unwrap_chains(state)
    vals = []
    for chain in state.chains:
        d = unw[chain[-1]] - unw[chain[0]]
        vals.append(np.sum(d * d))
    return float(np.mean(vals))


def structure_factor(
    state: MeltState,
    q_values: np.ndarray,
    n_directions: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """
    Single-chain static structure factor S(q), averaged over chains and
    `n_directions` isotropic q directions per magnitude.
    """
    unw = unwrap_chains(state)
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_directions, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    out = np.empty(len(q_values))
    for iq, q in enumerate(q_values):
        s_acc = 0.0
        n_acc = 0
        for chain in state.chains:
            p = unw[chain]
            n = len(p)
            for d in dirs:
                phase = (p @ d) * q
                f = np.sum(np.cos(phase)) ** 2 + np.sum(np.sin(phase)) ** 2
                s_acc += f / n
                n_acc += 1
        out[iq] = s_acc / n_acc
    return out


def bond_length_stats(state: MeltState) -> Dict[str, float]:
    """Bond length statistics (junction-memory diagnostic)."""
    pos = state.positions
    box = state.box_size
    lens = []
    for chain in state.chains:
        for a, b in zip(chain, chain[1:]):
            d = pos[b] - pos[a]
            d -= box * np.round(d / box)
            lens.append(float(np.linalg.norm(d)))
    arr = np.asarray(lens)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p99": float(np.percentile(arr, 99)),
    }


def compare_msid(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    rtol: float = 0.05,
    s_min: int = 1,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare two MSID curves (from independent equilibration paths).
    Returns (agree, report) where agreement requires every s >= s_min to
    match within `rtol` relative deviation.
    """
    sa, ra = curve_a[:, 0], curve_a[:, 1]
    sb, rb = curve_b[:, 0], curve_b[:, 1]
    common = np.intersect1d(sa, sb)
    common = common[common >= s_min]
    dev = []
    for s in common:
        va = ra[np.searchsorted(sa, s)]
        vb = rb[np.searchsorted(sb, s)]
        dev.append(abs(va - vb) / max(abs(vb), 1e-12))
    dev = np.asarray(dev)
    worst_s = common[int(np.argmax(dev))] if len(dev) else None
    report = {
        "max_rel_dev": float(dev.max()) if len(dev) else float("nan"),
        "worst_s": int(worst_s) if worst_s is not None else None,
        "n_points": int(len(dev)),
        "rtol": rtol,
    }
    return bool(len(dev) and dev.max() < rtol), report
