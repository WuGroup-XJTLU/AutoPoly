#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Obstacle-aware in-box chain growth for the nanocomposite strategy.

Pre-grown chains placed as rigid bodies saturate at low matrix densities.
When rigid placement fails, the chain is regrown monomer by monomer
*inside the box* around the already placed fillers and chains:

- each monomer is attached with chemically sensible junction geometry:
  the open valence of the previous monomer's right connection atom and of
  this monomer's left connection atom (the direction of the bond removed
  at the "[*]" connection, estimated as minus the sum of the existing
  bond unit vectors -- the vacant tetrahedral position for sp3 atoms) are
  aligned anti-parallel along the junction bond (stored junction length);
  the torsion about the junction bond is sampled uniformly and the new
  monomer may tilt by up to ``max_tilt`` degrees from the ideal alignment
  (bulky junctions in distorted fragment conformers otherwise dead-end),
- every trial monomer is tested at the *atom* level: against the shared
  occupancy grid (fillers and other chains, minimum image), against this
  chain's own earlier monomers (minimum image, clearance ``intra``), and
  against the previous monomer: 1-2/1-3 pairs across the junction are
  skipped, 1-4 pairs must exceed ``intra14``, all others ``intra``,
- dead ends trigger a bounded backtrack.

The grown conformation is written as a per-instance copy of the typed
chain class with only the monomer transforms replaced, so stage-2 typing
(atom types, charges, bonds) is untouched.

Created on 2026-11-13
@author: zwu
"""
import json
import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.exceptions import ValidationError
from ..pipeline.unit_coords import matrix_to_axis_angle, rotation_matrix
from .collision_geometry import PeriodicVoxelGrid, random_rotation

_LINE = re.compile(r"^(\s*monomer\[(\d+)\]\s*=\s*new\s+)(\S+?)"
                   r"(\.rot\([^)]*\))?(\.move\([^)]*\))?\s*$")


class ChainTemplate:
    """Per-monomer local geometry of one chain unit (stage-2 sidecar)."""

    def __init__(self, data: Dict) -> None:
        self.unit = data["unit"]
        self.topology = data.get("topology", "linear")
        self.monomers = data["monomers"]
        self.templates = data["templates"]
        self.junctions = [float(x) for x in data["junction_lengths"]]
        self._local = {}
        for name, t in self.templates.items():
            self._local[name] = np.array(t["coords"], dtype=float)

    @staticmethod
    def load(path: Path) -> "ChainTemplate":
        return ChainTemplate(json.loads(Path(path).read_text()))

    def local(self, k: int) -> np.ndarray:
        return self._local[self.monomers[k]]

    def info(self, k: int) -> Dict:
        return self.templates[self.monomers[k]]

    @property
    def n(self) -> int:
        return len(self.monomers)

    def offsets(self) -> List[int]:
        """Index of each monomer's first atom in the chain atom order."""
        out, k = [], 0
        for name in self.monomers:
            out.append(k)
            k += len(self.templates[name]["elements"])
        return out

    def chain_bonds(self) -> List[Tuple[int, int]]:
        """All bonds of the chain (intra-monomer + junctions), chain order."""
        offs = self.offsets()
        bonds = []
        for k, name in enumerate(self.monomers):
            o = offs[k]
            bonds += [(o + i, o + j) for i, j in self.templates[name]["bonds"]]
            if k + 1 < self.n:
                right = self.templates[name]["right"]
                left = self.templates[self.monomers[k + 1]]["left"]
                if right is not None and left is not None:
                    bonds.append((o + right, offs[k + 1] + left))
        if self.topology == "ring" and self.n > 2:
            right = self.templates[self.monomers[-1]]["right"]
            left = self.templates[self.monomers[0]]["left"]
            if right is not None and left is not None:
                bonds.append((offs[-1] + right, left))
        return bonds


def intrachain_contacts(coords: np.ndarray, bonds: Sequence[Tuple[int, int]],
                        cutoff: float, lengths: Optional[np.ndarray] = None,
                        cutoff14: Optional[float] = None) -> Tuple[int, float]:
    """
    (count, minimum distance) of intra-molecular atom pairs that are too
    close: 1-2 and 1-3 pairs are skipped, 1-4 pairs are flagged below
    ``cutoff14`` (default: ``cutoff``), all others below ``cutoff``.
    ``lengths``: optional periodic box (minimum image).
    """
    from ..materials.topology import neighbor_pairs

    c14 = cutoff if cutoff14 is None else cutoff14
    reach = max(cutoff, c14)
    n = len(coords)
    adj: List[List[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    if lengths is not None:
        i, j, _, d = neighbor_pairs(coords - coords.min(axis=0), reach,
                                    np.diag(lengths), (True, True, True))
    else:
        i, j, _, d = neighbor_pairs(coords, reach)
    count, dmin = 0, np.inf
    cache: Dict[int, Dict[int, int]] = {}
    for a, b, dd in zip(i, j, d):
        if a not in cache:
            seen = {a: 0}
            q = deque([a])
            while q:
                x = q.popleft()
                if seen[x] >= 3:
                    continue
                for y in adj[x]:
                    if y not in seen:
                        seen[y] = seen[x] + 1
                        q.append(y)
            cache[a] = seen
        g = cache[a].get(b, 4)
        if g <= 2:
            continue
        if dd < (c14 if g == 3 else cutoff):
            count += 1
            dmin = min(dmin, float(dd))
    return count, dmin


def _junction_distances(prev: Dict, cur: Dict) -> np.ndarray:
    """
    (n_prev, n_cur) graph distances (capped at 4) between atoms of two
    consecutive monomers joined by the junction prev.right -- cur.left.
    """
    n1, n2 = len(prev["elements"]), len(cur["elements"])
    adj: List[List[int]] = [[] for _ in range(n1 + n2)]
    for i, j in prev["bonds"]:
        adj[i].append(j)
        adj[j].append(i)
    for i, j in cur["bonds"]:
        adj[n1 + i].append(n1 + j)
        adj[n1 + j].append(n1 + i)
    a, b = prev["right"], n1 + cur["left"]
    adj[a].append(b)
    adj[b].append(a)
    out = np.full((n1, n2), 4, dtype=int)
    for start in range(n1):
        dist = {start: 0}
        q = deque([start])
        while q:
            x = q.popleft()
            if dist[x] >= 3:
                continue
            for y in adj[x]:
                if y not in dist:
                    dist[y] = dist[x] + 1
                    q.append(y)
        for y, dd in dist.items():
            if y >= n1:
                out[start, y - n1] = dd
    return out


from ..materials.topology import UNIT_SPHERE as _SPHERE  # noqa: E402


def open_valence(info: Dict, atom: int, coords: np.ndarray,
                 bond_length: float = 1.54) -> np.ndarray:
    """
    Unit direction of the missing bond of ``atom`` (template frame).

    A pyramidal sp3 center (three bonds) or a trigonal sp2 center (two
    bonds) has a well-defined vacant site: minus the sum of its unit bond
    vectors, which keeps the junction bond angles near tetrahedral /
    trigonal.  When that sum is degenerate (a flattened sp3 fragment
    conformer, where it would point into neighboring hydrogens, or a
    linear center) the direction whose bond-length site has the largest
    clearance from the monomer's other atoms is used instead.  (A pure
    clearance criterion is biased away from bulky substituents: for the
    PMMA backbone CH2 it gave ~95 degree C-C-C junction angles and
    crowded i/i+2 monomers.)
    """
    others = np.delete(coords, atom, axis=0)
    if len(others) == 0:
        return np.array([1.0, 0.0, 0.0])
    nbrs = [j if i == atom else i for i, j in info.get("bonds", ())
            if atom in (i, j)]
    if len(nbrs) in (2, 3):
        b = coords[nbrs] - coords[atom]
        b /= np.linalg.norm(b, axis=1)[:, None]
        s = b.sum(axis=0)
        if np.linalg.norm(s) >= 0.5:
            return -s / np.linalg.norm(s)
    sites = coords[atom] + bond_length * _SPHERE
    d = np.linalg.norm(sites[:, None, :] - others[None, :, :], axis=2)
    clearance = d.min(axis=1)
    best = np.argmax(clearance)
    # refine: average the near-optimal directions (smooth, deterministic)
    near = clearance >= clearance[best] - 0.02
    u = _SPHERE[near].sum(axis=0)
    return u / np.linalg.norm(u)


def _cone(prev_dir, angle, dihedral):
    if abs(prev_dir[2]) < 0.9:
        u = np.cross(prev_dir, [0.0, 0.0, 1.0])
    else:
        u = np.cross(prev_dir, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(prev_dir, u)
    d = (np.cos(angle) * prev_dir
         + np.sin(angle) * (np.cos(dihedral) * u + np.sin(dihedral) * v))
    return d / np.linalg.norm(d)


def _align(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector v1 onto unit vector v2."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    c = float(np.dot(v1, v2))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        axis = np.cross(v1, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v1, [0.0, 1.0, 0.0])
        return rotation_matrix(180.0, axis)
    axis = np.cross(v1, v2)
    return rotation_matrix(np.degrees(np.arccos(c)), axis)


class ObstacleAwareGrower:
    """
    Grow one chain inside the box (see module docstring).

    Args:
        grid: Shared occupancy grid (fillers + placed matrix instances).
        lengths: Box lengths (periodic).
        intra: Clearance between atoms of the same chain (A).
        bond_angle_min / bond_angle_max: Cone angle range (deg) between
            consecutive backbone directions.
        trials: Trial orientations per monomer.
        max_backtracks: Total backtrack budget per chain.
    """

    def __init__(self, grid: PeriodicVoxelGrid, lengths: Sequence[float],
                 intra: float = 1.3, intra14: float = 1.4, trials: int = 100,
                 max_backtracks: int = 200, max_tilt: float = 35.0,
                 **_ignored) -> None:
        self.max_tilt = np.radians(max_tilt)
        self.grid = grid
        self.lengths = np.asarray(lengths, dtype=float)
        self.intra = float(intra)
        self.intra14 = float(intra14)
        self.trials = trials
        self.max_backtracks = max_backtracks
        self._valence: Dict[Tuple[str, str], np.ndarray] = {}

    def _open(self, tpl: ChainTemplate, k: int, side: str) -> np.ndarray:
        name = tpl.monomers[k]
        key = (name, side)
        if key not in self._valence:
            info = tpl.info(k)
            junction = tpl.junctions[k - 1] if side == "left" else \
                tpl.junctions[min(k, len(tpl.junctions) - 1)]
            self._valence[key] = open_valence(info, info[side], tpl.local(k),
                                              bond_length=junction)
        return self._valence[key]

    def _clash_self(self, cand: np.ndarray, placed: List[np.ndarray],
                    k: int, jdist: Optional[np.ndarray]) -> bool:
        """Atom-level clash of monomer k with this chain's monomers < k."""
        r2 = self.intra ** 2
        if k >= 2:
            older = np.vstack(placed[:k - 1])
            d = cand[:, None, :] - older[None, :, :]
            d -= self.lengths * np.round(d / self.lengths)
            if np.any(np.einsum("ijk,ijk->ij", d, d) < r2):
                return True
        if k >= 1:
            d = cand[:, None, :] - placed[k - 1][None, :, :]
            d -= self.lengths * np.round(d / self.lengths)
            dd = np.einsum("ijk,ijk->ij", d, d)
            if jdist is not None:
                g = jdist.T            # (n_cur, n_prev)
                limit = np.where(g <= 2, -1.0,
                                 np.where(g == 3, self.intra14 ** 2, r2))
            else:
                limit = r2
            if np.any(dd < limit):
                return True
        return False

    def grow(self, tpl: ChainTemplate, rng: np.random.Generator,
             start: Optional[np.ndarray] = None,
             anchor: Optional[Dict] = None,
             ) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Grow the chain; returns per-monomer (R, t) such that world =
        R @ local + t, or None when the budget is exhausted.

        anchor: grafting -- {"position": anchor atom (surface O),
            "direction": unit vector of the vacated O-H bond, "length":
            O-C bond length, "head": head-atom index in monomer 0,
            "cap": index of the head H removed at grafting}.  Monomer 0 is
            then attached so the head atom sits on the anchor's open
            valence and the head's vacated C-H direction points back at
            the anchor; the cap H is ignored in every clash test.
        """
        if tpl.topology == "ring":
            return None
        self._anchor = anchor
        n = tpl.n
        excl = [None] + [_junction_distances(tpl.info(k - 1), tpl.info(k))
                         for k in range(1, n)]
        transforms: List[Tuple[np.ndarray, np.ndarray]] = []
        placed: List[np.ndarray] = []
        backtracks = 0
        k = 0
        if start is None:
            start = self.grid.open_free_point(rng)
        while k < n:
            ok = self._place(tpl, k, transforms, placed, excl[k], rng, start)
            if ok:
                k += 1
                continue
            if backtracks >= self.max_backtracks or k == 0:
                return None
            back = min(k, int(rng.integers(1, 4)))
            for _ in range(back):
                transforms.pop()
                placed.pop()
            k -= back
            backtracks += 1
        return transforms

    def _place(self, tpl, k, transforms, placed, excl, rng, start) -> bool:
        """Batch-evaluate ``trials`` placements of monomer k; keep one."""
        local = tpl.local(k)
        info = tpl.info(k)
        T = self.trials
        anchor = getattr(self, "_anchor", None)
        if k == 0 and anchor is not None:
            head, cap = anchor["head"], anchor["cap"]
            d_anchor = np.asarray(anchor["direction"], dtype=float)
            d_anchor = d_anchor / np.linalg.norm(d_anchor)
            ov = local[cap] - local[head]
            ov /= np.linalg.norm(ov)
            R0 = _align(ov, -d_anchor)
            Rs = _axis_rotations(d_anchor,
                                 rng.uniform(0.0, 2.0 * np.pi, T)) @ R0
            tilt = min(self.max_tilt, np.radians(anchor.get("max_tilt", 15.0)))
            if tilt > 0:
                Rs = _tilt_rotations(d_anchor, tilt, T, rng) @ Rs
            target = np.asarray(anchor["position"], dtype=float) + \
                anchor["length"] * d_anchor
            ts = target - np.einsum("tij,j->ti", Rs, local[head])
        elif k == 0:
            Rs = _random_rotations(rng, T)
            if start is not None:
                points = np.repeat(np.asarray(start)[None, :], T, axis=0)
                points[1:] = [self.grid.random_free_point(rng) for _ in
                              range(T - 1)]
            else:
                points = np.array([self.grid.random_free_point(rng)
                                   for _ in range(T)])
            ts = points - np.einsum("tij,j->ti", Rs, local.mean(axis=0))
        else:
            pR, pt = transforms[-1]
            prev = tpl.info(k - 1)
            right = pR @ tpl.local(k - 1)[prev["right"]] + pt
            d_prev = pR @ self._open(tpl, k - 1, "right")
            d_cur = self._open(tpl, k, "left")
            # left atom along the previous open valence; this monomer's
            # open valence points back along the junction; torsion about
            # the junction bond sampled uniformly
            R0 = _align(d_cur, -d_prev)
            spins = _axis_rotations(d_prev,
                                    rng.uniform(0.0, 2.0 * np.pi, T))
            Rs = spins @ R0
            if self.max_tilt > 0:
                # tilt about random axes perpendicular to the junction
                # bond, pivoting on the new left atom (applied below)
                tilts = _tilt_rotations(d_prev, self.max_tilt, T, rng)
                Rs = tilts @ Rs
            target = right + tpl.junctions[k - 1] * d_prev
            ts = target - np.einsum("tij,j->ti", Rs, local[info["left"]])
        cands = np.einsum("nj,tij->tni", local, Rs) + ts[:, None, :]
        active = np.ones(cands.shape[1], dtype=bool)
        if k == 0 and anchor is not None:
            # the cap H is removed at grafting: NaN coordinates take part
            # in no distance test
            active[anchor["cap"]] = False
            cands[:, anchor["cap"], :] = np.nan
        idx = self.grid.voxel_index(cands[:, active].reshape(-1, 3))
        blocked = self.grid.grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        ok = ~blocked.reshape(T, -1).any(axis=1)
        if anchor is not None:
            # atoms the grid does not see (the own anchor O / Si are
            # excluded from it): 1-3 / 1-4 distances across the new bond
            # for monomer 0, plain clearance for the rest of the chain
            others = active.copy()
            if k == 0:
                others[anchor["head"]] = False
                partners = anchor.get("partners", ())
            else:
                partners = anchor.get("partners_far", ())
            for point, dmin in partners:
                d = np.linalg.norm(cands[:, others] - np.asarray(point),
                                   axis=2)
                ok &= (d >= dmin).all(axis=1)
        free = np.where(ok)[0]
        if len(free) == 0:
            return False
        ok = ~self._clash_batch(cands[free], placed, k, excl)
        if not ok.any():
            return False
        t = free[int(np.argmax(ok))]
        transforms.append((Rs[t], ts[t]))
        placed.append(cands[t])
        return True

    def _clash_batch(self, cands: np.ndarray, placed: List[np.ndarray],
                     k: int, jdist: Optional[np.ndarray]) -> np.ndarray:
        """(F,) clash flags of F candidate placements of monomer k."""
        r2 = self.intra ** 2
        clash = np.zeros(len(cands), dtype=bool)
        L = self.lengths
        if k >= 2:
            older = np.vstack(placed[:k - 1])
            # cheap prefilter: older atoms near the candidates' bounding box
            lo = cands.reshape(-1, 3).min(axis=0) - self.intra
            hi = cands.reshape(-1, 3).max(axis=0) + self.intra
            if not np.any((hi - lo) >= L / 2):
                rel = older - lo
                rel -= L * np.floor(rel / L)
                near = np.all(rel <= (hi - lo), axis=1)
                older = older[near]
            if len(older):
                d = cands[:, :, None, :] - older[None, None, :, :]
                d -= L * np.round(d / L)
                dd = np.einsum("fnmk,fnmk->fnm", d, d)
                clash |= (dd < r2).any(axis=(1, 2))
        if k >= 1:
            d = cands[:, :, None, :] - placed[k - 1][None, None, :, :]
            d -= L * np.round(d / L)
            dd = np.einsum("fnmk,fnmk->fnm", d, d)
            if jdist is not None:
                g = jdist.T            # (n_cur, n_prev)
                limit = np.where(g <= 2, -1.0,
                                 np.where(g == 3, self.intra14 ** 2, r2))
            else:
                limit = np.full(dd.shape[1:], r2)
            clash |= (dd < limit[None, :, :]).any(axis=(1, 2))
        return clash


def _axis_rotations(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """(T, 3, 3) rotations by ``angles`` (rad) about a unit axis."""
    u = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]],
                  [-u[1], u[0], 0.0]])
    c, s_ = np.cos(angles)[:, None, None], np.sin(angles)[:, None, None]
    return c * np.eye(3) + s_ * K + (1.0 - c) * np.outer(u, u)


def _tilt_rotations(axis: np.ndarray, max_angle: float, n: int,
                    rng: np.random.Generator) -> np.ndarray:
    """(n, 3, 3) rotations by angles in [0, max_angle] (uniform in solid
    angle) about random axes perpendicular to ``axis``; the first is the
    identity (ideal alignment always tried)."""
    u = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else \
        np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    theta = np.arccos(rng.uniform(np.cos(max_angle), 1.0, n))
    theta[0] = 0.0
    ax = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2   # (n, 3)
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -ax[:, 2], ax[:, 1]
    K[:, 1, 0], K[:, 1, 2] = ax[:, 2], -ax[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -ax[:, 1], ax[:, 0]
    c = np.cos(theta)[:, None, None]
    s_ = np.sin(theta)[:, None, None]
    outer = ax[:, :, None] * ax[:, None, :]
    return c * np.eye(3) + s_ * K + (1.0 - c) * outer


def _random_rotations(rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, 3, 3) uniformly distributed rotations (quaternion sampling)."""
    u1, u2, u3 = rng.random((3, n))
    x = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    y = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    z = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    w = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                  2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                  2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                  1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=1)


def regrown_class_text(src_lt: Path, new_class: str,
                       transforms: List[Tuple[np.ndarray, np.ndarray]]) -> str:
    """
    Copy of a typed chain class (poly_N.lt) under a new class name with the
    monomer transforms replaced by ``transforms`` (absolute box frame).
    """
    text = Path(src_lt).read_text().splitlines()
    old_class = Path(src_lt).stem
    out = []
    seen = set()
    for line in text:
        m = _LINE.match(line)
        if m:
            k = int(m.group(2))
            if k >= len(transforms):
                raise ValidationError(
                    f"{src_lt.name}: monomer[{k}] beyond grown chain length"
                )
            R, t = transforms[k]
            angle, ax, ay, az = matrix_to_axis_angle(R)
            line = (f"{m.group(1)}{m.group(3)}"
                    f".rot({angle:.6f},{ax:.6f},{ay:.6f},{az:.6f})"
                    f".move({t[0]:.4f},{t[1]:.4f},{t[2]:.4f})")
            seen.add(k)
        elif re.match(rf"^{re.escape(old_class)}\s+inherits\s", line) or \
                re.match(rf"^{re.escape(old_class)}\s*\{{", line):
            line = line.replace(old_class, new_class, 1)
        elif line.strip().startswith("}") and old_class in line:
            line = line.replace(old_class, new_class)
        out.append(line)
    if len(seen) != len(transforms):
        raise ValidationError(
            f"{Path(src_lt).name}: found {len(seen)} monomer lines for "
            f"{len(transforms)} grown monomers"
        )
    return "\n".join(out) + "\n"
