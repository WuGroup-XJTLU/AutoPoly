#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bond-graph and neighbor-search utilities for inorganic structures.

Shared by the substrate slab builders and the nanoparticle builders:

- minimum_image / wrap_positions: orthorhombic periodic boundary helpers
  (per-axis PBC flags; non-periodic axes are left untouched),
- neighbor_pairs: all atom pairs closer than a cutoff, finite or periodic
  (scipy cKDTree when available, otherwise a numpy cell list),
- perceive_bonds: element-pair distance-cutoff bond perception that
  returns periodic image shifts,
- infer_images: recompute bond image shifts from minimum-image geometry,
- connected_components / largest_component / coordination.

Only orthorhombic periodic cells are supported (every built-in slab and
crystal source is orthorhombic); a periodic triclinic cell raises.

Created on 2026-11-13
@author: zwu
"""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.exceptions import ValidationError

PBC = Tuple[bool, bool, bool]


# ----------------------------------------------------------------------
# Periodic boundary helpers
# ----------------------------------------------------------------------
def cell_lengths(cell: Optional[np.ndarray], pbc: PBC) -> np.ndarray:
    """
    Orthorhombic box lengths per axis (inf for non-periodic axes).

    Raises:
        ValidationError: If a periodic cell is not orthorhombic.
    """
    lengths = np.full(3, np.inf)
    if cell is None or not any(pbc):
        return lengths
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    off_diag = cell - np.diag(np.diag(cell))
    if np.any(np.abs(off_diag) > 1e-8):
        raise ValidationError(
            "Only orthorhombic periodic cells are supported by the "
            "inorganic topology utilities"
        )
    for axis in range(3):
        if pbc[axis]:
            lengths[axis] = float(cell[axis, axis])
    return lengths


def minimum_image(vectors: np.ndarray, cell: Optional[np.ndarray],
                  pbc: PBC) -> np.ndarray:
    """Apply the orthorhombic minimum-image convention on periodic axes."""
    vec = np.array(vectors, dtype=float, copy=True)
    lengths = cell_lengths(cell, pbc)
    for axis in range(3):
        if pbc[axis]:
            vec[..., axis] -= lengths[axis] * np.round(
                vec[..., axis] / lengths[axis]
            )
    return vec


def wrap_positions(positions: np.ndarray, cell: Optional[np.ndarray],
                   pbc: PBC, origin: Sequence[float] = (0.0, 0.0, 0.0)
                   ) -> np.ndarray:
    """Wrap coordinates into [origin, origin + L) on periodic axes."""
    pos = np.array(positions, dtype=float, copy=True)
    lengths = cell_lengths(cell, pbc)
    for axis in range(3):
        if pbc[axis]:
            o = origin[axis]
            pos[:, axis] = o + np.mod(pos[:, axis] - o, lengths[axis])
    return pos


# ----------------------------------------------------------------------
# Neighbor search
# ----------------------------------------------------------------------
def neighbor_pairs(
    positions: np.ndarray,
    cutoff: float,
    cell: Optional[np.ndarray] = None,
    pbc: PBC = (False, False, False),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    All unordered atom pairs (i < j) closer than ``cutoff``.

    Periodic axes use the minimum image, which requires
    ``cutoff < L/2`` on every periodic axis.

    Returns:
        (i, j, images, distances): index arrays, (K, 3) integer image
        shifts such that ``r_j + images . L - r_i`` is the pair vector,
        and the pair distances.  Pairs are sorted by (i, j).
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = len(pos)
    empty = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3), int),
             np.zeros(0))
    if n < 2 or cutoff <= 0:
        return empty
    lengths = cell_lengths(cell, pbc)
    for axis in range(3):
        if pbc[axis] and cutoff >= lengths[axis] / 2.0:
            raise ValidationError(
                f"Neighbor cutoff {cutoff} A is not smaller than half the "
                f"periodic box length {lengths[axis]:.3f} A along "
                f"{'xyz'[axis]}"
            )

    ii, jj = _candidate_pairs(pos, cutoff, lengths, pbc)
    if len(ii) == 0:
        return empty
    vec = pos[jj] - pos[ii]
    shift = np.zeros((len(ii), 3))
    for axis in range(3):
        if pbc[axis]:
            shift[:, axis] = -np.round(vec[:, axis] / lengths[axis])
            vec[:, axis] += shift[:, axis] * lengths[axis]
    dist = np.linalg.norm(vec, axis=1)
    keep = dist < cutoff
    ii, jj, shift, dist = ii[keep], jj[keep], shift[keep], dist[keep]
    order = np.lexsort((jj, ii))
    return (ii[order], jj[order], shift[order].astype(int), dist[order])


def _candidate_pairs(pos, cutoff, lengths, pbc):
    """Candidate (i < j) pairs within cutoff (superset allowed)."""
    try:  # pragma: no cover - exercised when scipy is installed
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover
        cKDTree = None

    if cKDTree is not None:
        shifted = pos.copy()
        boxsize = np.empty(3)
        for axis in range(3):
            if pbc[axis]:
                shifted[:, axis] = np.mod(shifted[:, axis], lengths[axis])
                boxsize[axis] = lengths[axis]
            else:
                lo = shifted[:, axis].min()
                shifted[:, axis] -= lo
                # large enough that no artificial wrap can occur
                boxsize[axis] = shifted[:, axis].max() * 3.0 + 10.0 * cutoff
        tree = cKDTree(shifted, boxsize=boxsize)
        pairs = tree.query_pairs(cutoff * (1.0 + 1e-9), output_type="ndarray")
        if len(pairs) == 0:
            return np.zeros(0, int), np.zeros(0, int)
        i = np.minimum(pairs[:, 0], pairs[:, 1])
        j = np.maximum(pairs[:, 0], pairs[:, 1])
        return i, j
    return _cell_list_pairs(pos, cutoff, lengths, pbc)


def _cell_list_pairs(pos, cutoff, lengths, pbc):
    """Numpy cell-list candidate pair search (scipy-free fallback)."""
    n = len(pos)
    if n <= 2000:
        i, j = np.triu_indices(n, k=1)
        vec = pos[j] - pos[i]
        for axis in range(3):
            if pbc[axis]:
                vec[:, axis] -= lengths[axis] * np.round(
                    vec[:, axis] / lengths[axis])
        keep = np.einsum("ij,ij->i", vec, vec) < cutoff * cutoff
        return i[keep], j[keep]

    p = pos.copy()
    ncell = np.empty(3, dtype=int)
    size = np.empty(3)
    lo = np.empty(3)
    for axis in range(3):
        if pbc[axis]:
            p[:, axis] = np.mod(p[:, axis], lengths[axis])
            lo[axis] = 0.0
            ncell[axis] = max(1, int(lengths[axis] // cutoff))
            size[axis] = lengths[axis] / ncell[axis]
        else:
            lo[axis] = p[:, axis].min()
            span = p[:, axis].max() - lo[axis]
            ncell[axis] = max(1, int(span // cutoff) + 1)
            size[axis] = max(cutoff, span / ncell[axis] + 1e-9)
    idx = np.floor((p - lo) / size).astype(int)
    idx = np.minimum(idx, ncell - 1)
    lin = (idx[:, 0] * ncell[1] + idx[:, 1]) * ncell[2] + idx[:, 2]
    order = np.argsort(lin, kind="stable")
    lin_sorted = lin[order]
    cells, starts = np.unique(lin_sorted, return_index=True)
    ends = np.append(starts[1:], len(order))
    members = {int(c): order[s:e] for c, s, e in zip(cells, starts, ends)}

    out_i, out_j = [], []
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1)
               for c in (-1, 0, 1)]
    for c, atoms in members.items():
        cz = c % ncell[2]
        cy = (c // ncell[2]) % ncell[1]
        cx = c // (ncell[1] * ncell[2])
        neigh = set()
        for dx, dy, dz in offsets:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            ok = True
            for axis, v in enumerate((nx, ny, nz)):
                if not (0 <= v < ncell[axis]):
                    if pbc[axis]:
                        continue
                    ok = False
            if not ok:
                continue
            nx, ny, nz = nx % ncell[0], ny % ncell[1], nz % ncell[2]
            neigh.add(int((nx * ncell[1] + ny) * ncell[2] + nz))
        others = [members[m] for m in neigh if m in members]
        if not others:
            continue
        cand = np.concatenate(others)
        a = np.repeat(atoms, len(cand))
        b = np.tile(cand, len(atoms))
        mask = a < b
        a, b = a[mask], b[mask]
        vec = pos[b] - pos[a]
        for axis in range(3):
            if pbc[axis]:
                vec[:, axis] -= lengths[axis] * np.round(
                    vec[:, axis] / lengths[axis])
        keep = np.einsum("ij,ij->i", vec, vec) < cutoff * cutoff
        out_i.append(a[keep])
        out_j.append(b[keep])
    if not out_i:
        return np.zeros(0, int), np.zeros(0, int)
    i = np.concatenate(out_i)
    j = np.concatenate(out_j)
    pairs = np.unique(np.stack([i, j], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


# ----------------------------------------------------------------------
# Bond perception
# ----------------------------------------------------------------------
def perceive_bonds(
    elements: Sequence[str],
    positions: np.ndarray,
    cutoffs: Dict[Tuple[str, str], float],
    cell: Optional[np.ndarray] = None,
    pbc: PBC = (False, False, False),
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Distance-cutoff bond perception for the element pairs in ``cutoffs``.

    Args:
        elements: Element symbol per atom.
        positions: (N, 3) coordinates.
        cutoffs: {(elem_a, elem_b): max bond length (A)}; order of the
                 pair does not matter.  Pairs not listed are never bonded.
        cell, pbc: Optional orthorhombic periodicity.

    Returns:
        (bonds, images): (i, j) pairs with i < j sorted by (i, j), and the
        (M, 3) image shifts of j.
    """
    table = {}
    for (a, b), r in cutoffs.items():
        table[(a, b)] = float(r)
        table[(b, a)] = float(r)
    if not table:
        return [], np.zeros((0, 3), dtype=int)
    rmax = max(table.values())
    i, j, img, dist = neighbor_pairs(positions, rmax, cell, pbc)
    bonds, images = [], []
    for a, b, s, d in zip(i, j, img, dist):
        r = table.get((elements[a], elements[b]))
        if r is not None and d < r:
            bonds.append((int(a), int(b)))
            images.append(s)
    return bonds, np.array(images, dtype=int).reshape(-1, 3)


def infer_images(positions: np.ndarray, bonds: Sequence[Tuple[int, int]],
                 cell: Optional[np.ndarray], pbc: PBC) -> np.ndarray:
    """Bond image shifts from the minimum-image convention."""
    if not bonds:
        return np.zeros((0, 3), dtype=int)
    b = np.asarray(bonds, dtype=int)
    vec = positions[b[:, 1]] - positions[b[:, 0]]
    images = np.zeros((len(b), 3), dtype=int)
    lengths = cell_lengths(cell, pbc)
    for axis in range(3):
        if pbc[axis]:
            images[:, axis] = -np.round(vec[:, axis] / lengths[axis]).astype(int)
    return images


# ----------------------------------------------------------------------
# Graph helpers
# ----------------------------------------------------------------------
def coordination(n: int, bonds: Iterable[Tuple[int, int]]) -> np.ndarray:
    """Per-atom bonded coordination number."""
    coord = np.zeros(n, dtype=int)
    for i, j in bonds:
        coord[i] += 1
        coord[j] += 1
    return coord


def adjacency(n: int, bonds: Iterable[Tuple[int, int]]) -> List[List[int]]:
    """Per-atom neighbor lists (bond order preserved)."""
    nbrs: List[List[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        nbrs[i].append(j)
        nbrs[j].append(i)
    return nbrs


def connected_components(n: int, bonds: Iterable[Tuple[int, int]]) -> List[int]:
    """
    Component label per atom; components ordered by decreasing size, ties
    by smallest member index (isolated atoms are their own components).
    """
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in bonds:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = [find(i) for i in range(n)]
    groups: Dict[int, List[int]] = {}
    for i, r in enumerate(roots):
        groups.setdefault(r, []).append(i)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    label = [0] * n
    for k, members in enumerate(ordered):
        for i in members:
            label[i] = k
    return label


def largest_component(n: int, bonds: Iterable[Tuple[int, int]]) -> List[int]:
    """Sorted atom indices of the largest connected component."""
    labels = connected_components(n, list(bonds))
    return [i for i, lab in enumerate(labels) if lab == 0]


def fibonacci_sphere(n: int = 600) -> np.ndarray:
    """(n, 3) near-uniform unit vectors (Fibonacci lattice)."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    theta = np.pi * (1.0 + 5 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


#: Shared direction set for clearance searches (open valences, caps).
UNIT_SPHERE = fibonacci_sphere()


def clearest_direction(positions: np.ndarray, atom: int, length: float,
                       exclude: Sequence[int] = (),
                       avoid: Sequence[np.ndarray] = (),
                       reach: float = 5.0) -> np.ndarray:
    """
    Unit direction from ``atom`` whose site at ``length`` has the largest
    clearance from all other atoms within ``reach`` (minus ``exclude``)
    and from the extra points ``avoid``.
    """
    pos = np.asarray(positions, dtype=float)
    sites = pos[atom] + length * UNIT_SPHERE
    mask = np.ones(len(pos), dtype=bool)
    mask[atom] = False
    for k in exclude:
        mask[k] = False
    others = pos[mask]
    others = others[np.linalg.norm(others - pos[atom], axis=1) < reach]
    if len(avoid):
        others = np.vstack([others, np.asarray(avoid).reshape(-1, 3)]) \
            if len(others) else np.asarray(avoid).reshape(-1, 3)
    if len(others) == 0:
        return UNIT_SPHERE[0].copy()
    clearance = np.linalg.norm(sites[:, None, :] - others[None, :, :],
                               axis=2).min(axis=1)
    best = clearance.max()
    near = clearance >= best - 0.02
    u = UNIT_SPHERE[near].sum(axis=0)
    n = np.linalg.norm(u)
    return u / n if n > 1e-9 else UNIT_SPHERE[int(np.argmax(clearance))]
