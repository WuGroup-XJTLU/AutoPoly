#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Periodic collision representations for atom-level packing.

PeriodicVoxelGrid is a boolean occupancy grid over an orthorhombic
periodic box.  ``mark(points, r)`` blocks every voxel whose center lies
within ``r + delta`` of a point (delta = voxel half-diagonal); ``hits``
tests the voxels containing query points.  The test is conservative: a
query atom closer than ``r`` to a marked atom is always detected (no false
negatives); atoms up to ``r + 2*delta`` away may be rejected.  Clearances
are therefore guaranteed by construction and are verified exactly
afterwards (min_image_contacts).

This is the narrow phase of the nanocomposite strategy: it handles any
filler shape (spheres, sheets, tubes, imported clusters) and interpenetrating
polymer coils, which whole-chain bounding spheres cannot.

Created on 2026-11-13
@author: zwu
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


class PeriodicVoxelGrid:
    """
    Boolean occupancy grid with periodic wrapping on all three axes.

    Args:
        lo: Box lower corner (3,).
        lengths: Box side lengths (3,).
        spacing: Target voxel size (A); actual per-axis size is L / n.
    """

    def __init__(self, lo: Sequence[float], lengths: Sequence[float],
                 spacing: float = 0.35) -> None:
        self.lo = np.asarray(lo, dtype=float)
        self.lengths = np.asarray(lengths, dtype=float)
        self.shape = np.maximum(1, np.ceil(self.lengths / spacing)).astype(int)
        self.h = self.lengths / self.shape
        self.delta = 0.5 * float(np.linalg.norm(self.h))
        self.grid = np.zeros(tuple(self.shape), dtype=bool)
        self._stencils: Dict[float, np.ndarray] = {}

    # ------------------------------------------------------------------
    def voxel_index(self, points: np.ndarray) -> np.ndarray:
        """(N, 3) integer voxel indices of (wrapped) points."""
        frac = np.mod(np.asarray(points, dtype=float) - self.lo,
                      self.lengths) / self.h
        idx = np.floor(frac).astype(int)
        return np.mod(idx, self.shape)

    def _stencil(self, radius: float) -> np.ndarray:
        key = round(float(radius), 4)
        st = self._stencils.get(key)
        if st is None:
            reach = radius + self.delta
            n = np.ceil(reach / self.h).astype(int) + 1
            ax = [np.arange(-k, k + 1) for k in n]
            g = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
            # voxel centers relative to the point's voxel center: the point
            # lies anywhere inside its voxel, so use reach + delta
            d = np.linalg.norm(g * self.h, axis=1)
            st = g[d <= reach + self.delta]
            self._stencils[key] = st
        return st

    def mark(self, points: np.ndarray, radius: float) -> None:
        """Block all voxels within radius (+ voxel slack) of the points."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(pts) == 0:
            return
        base = self.voxel_index(pts)
        # exact test of voxel centers against the point positions
        st = self._stencil(radius)
        chunk = max(1, 2_000_000 // max(len(st), 1))
        for s in range(0, len(pts), chunk):
            b = base[s:s + chunk]
            p = pts[s:s + chunk]
            cells = (b[:, None, :] + st[None, :, :])
            centers = self.lo + (cells + 0.5) * self.h
            dvec = centers - p[:, None, :]
            dvec -= self.lengths * np.round(dvec / self.lengths)
            near = np.einsum("ijk,ijk->ij", dvec, dvec) <= \
                (radius + self.delta) ** 2
            sel = np.mod(cells[near], self.shape)
            self.grid[sel[:, 0], sel[:, 1], sel[:, 2]] = True

    def copy(self) -> "PeriodicVoxelGrid":
        """Independent copy (same geometry, copied occupancy)."""
        out = PeriodicVoxelGrid.__new__(PeriodicVoxelGrid)
        out.lo, out.lengths = self.lo.copy(), self.lengths.copy()
        out.shape, out.h, out.delta = self.shape.copy(), self.h.copy(), \
            self.delta
        out.grid = self.grid.copy()
        out._stencils = self._stencils
        return out

    def hits(self, points: np.ndarray) -> bool:
        idx = self.voxel_index(points)
        return bool(self.grid[idx[:, 0], idx[:, 1], idx[:, 2]].any())

    def n_hits(self, points: np.ndarray) -> int:
        idx = self.voxel_index(points)
        return int(self.grid[idx[:, 0], idx[:, 1], idx[:, 2]].sum())

    def free_fraction(self) -> float:
        return 1.0 - float(self.grid.mean())

    def local_free_fraction(self, point: np.ndarray, radius: float) -> float:
        """Fraction of free voxels in a cube of half-width radius."""
        idx = self.voxel_index(np.asarray(point)[None, :])[0]
        n = np.maximum(1, np.round(radius / self.h).astype(int))
        sl = [np.arange(i - k, i + k + 1) % s
              for i, k, s in zip(idx, n, self.shape)]
        window = self.grid[np.ix_(sl[0], sl[1], sl[2])]
        return 1.0 - float(window.mean())

    def open_free_point(self, rng: np.random.Generator, radius: float = 4.0,
                        candidates: int = 24) -> Optional[np.ndarray]:
        """Free point with the most open neighborhood among candidates."""
        best, best_f = None, -1.0
        for _ in range(candidates):
            p = self.random_free_point(rng)
            if p is None:
                break
            f = self.local_free_fraction(p, radius)
            if f > best_f:
                best, best_f = p, f
        return best

    def random_free_point(self, rng: np.random.Generator,
                          tries: int = 64) -> Optional[np.ndarray]:
        """Uniform random point inside a free voxel (None if none found)."""
        for _ in range(tries):
            p = self.lo + rng.random(3) * self.lengths
            idx = self.voxel_index(p[None, :])[0]
            if not self.grid[tuple(idx)]:
                return p
        free = np.argwhere(~self.grid)
        if len(free) == 0:
            return None
        k = free[rng.integers(len(free))]
        return self.lo + (k + rng.random(3)) * self.h


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed rotation matrix (quaternion sampling)."""
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def min_image_contacts(
    positions: np.ndarray,
    owner: np.ndarray,
    lengths: Sequence[float],
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inter-instance atom pairs closer than ``cutoff`` (full 3D PBC).

    Args:
        positions: (N, 3) atom positions.
        owner: (N,) instance index per atom (pairs within one instance are
               ignored).
        lengths: Box lengths.

    Returns:
        (i, j, d) of the offending pairs.
    """
    from ..materials.topology import neighbor_pairs

    lengths = np.asarray(lengths, dtype=float)
    i, j, _, d = neighbor_pairs(
        np.asarray(positions) - np.asarray(positions).min(axis=0),
        cutoff, np.diag(lengths), (True, True, True))
    mask = owner[i] != owner[j]
    return i[mask], j[mask], d[mask]
