#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class-local atom coordinates of packable units.

Stage 2 writes one XYZ file per unit (``<unit>.xyz`` in the build dir,
referenced by UnitSpec.coords_file) with the coordinates moltemplate will
produce for one instance of the unit's class *before* the placement
transform.  Atom-level packing strategies (nanocomposite_mc) read these
files instead of parsing .lt contents.

Transform semantics follow moltemplate: ``.rot(theta, ax, ay, az)``
rotates about the origin by theta degrees (right-hand rule, Rodrigues
formula as in ttree_matrix_stack.RotMatAXYZ), then ``.move(x, y, z)``
translates.

Created on 2026-11-13
@author: zwu
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def rotation_matrix(angle_deg: float, axis: Sequence[float]) -> np.ndarray:
    """Rotation matrix of moltemplate's .rot(angle_deg, ax, ay, az)."""
    ax = np.asarray(axis, dtype=float)
    r = float(np.linalg.norm(ax))
    if r == 0.0:
        return np.eye(3)
    x, y, z = ax / r
    a = np.radians(angle_deg)
    c, s, t = np.cos(a), np.sin(a), 1.0 - np.cos(a)
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def matrix_to_axis_angle(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    (angle_deg, ax, ay, az) of a proper rotation matrix (robust for all
    angles including 180 degrees, via the unit quaternion).
    """
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    angle = 2.0 * np.degrees(np.arccos(np.clip(q[0], -1.0, 1.0)))
    v = q[1:]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return (0.0, 1.0, 0.0, 0.0)
    v = v / n
    return (float(angle), float(v[0]), float(v[1]), float(v[2]))


def placement_matrix(rotation: Optional[Dict[str, Any]]) -> np.ndarray:
    if rotation is None:
        return np.eye(3)
    return rotation_matrix(rotation["angle_deg"], rotation["axis"])


def chain_coordinates(
    placements: List[Dict[str, Any]],
    variants: Dict[str, Any],
) -> Tuple[List[str], np.ndarray]:
    """
    World coordinates of a chain from its stored monomer placements
    (geometry.json), in the order moltemplate writes the atoms.
    """
    elements: List[str] = []
    coords = []
    for p in placements:
        atoms = variants[p["variant"]]["atoms"]
        local = np.array([a["coords"] for a in atoms], dtype=float)
        R = placement_matrix(p.get("rotation"))
        t = np.zeros(3) if p.get("translation") is None else \
            np.asarray(p["translation"], dtype=float)
        coords.append(local @ R.T + t)
        elements.extend(a["element"] for a in atoms)
    return elements, np.vstack(coords) if coords else np.zeros((0, 3))


def write_xyz(path: Path, elements: Sequence[str], coords: np.ndarray,
              comment: str = "") -> Path:
    """Write a standard XYZ file."""
    path = Path(path)
    lines = [str(len(elements)), comment.replace("\n", " ")]
    for e, (x, y, z) in zip(elements, np.asarray(coords, dtype=float)):
        lines.append(f"{e} {x:.6f} {y:.6f} {z:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return path


def read_xyz(path: Path) -> Tuple[List[str], np.ndarray]:
    """Read a standard XYZ file (first frame)."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    elements, coords = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        elements.append(parts[0])
        coords.append([float(v) for v in parts[1:4]])
    return elements, np.array(coords, dtype=float).reshape(-1, 3)


def chain_template_data(unit_id: str, chain: Dict[str, Any],
                        variants: Dict[str, Any]) -> Dict[str, Any]:
    """
    Per-monomer local geometry of one chain (variant sequence, local
    coordinates, intra-monomer bonds, connection atoms, junction lengths)
    -- the data behind packing.chain_regrowth.ChainTemplate.
    """
    placements = chain["placements"]
    templates: Dict[str, Any] = {}
    world_conn = []
    for p in placements:
        name = p["variant"]
        v = variants[name]
        conn = v.get("connection_atoms") or {}
        templates[name] = {
            "elements": [a["element"] for a in v["atoms"]],
            "coords": [a["coords"] for a in v["atoms"]],
            "bonds": [list(b) for b in v["bonds"]],
            "left": conn.get("left"),
            "right": conn.get("right"),
        }
        R = placement_matrix(p.get("rotation"))
        t = np.zeros(3) if p.get("translation") is None else \
            np.asarray(p["translation"], dtype=float)
        local = np.array([a["coords"] for a in v["atoms"]], dtype=float)
        world = local @ R.T + t
        world_conn.append((
            None if conn.get("left") is None else world[conn["left"]],
            None if conn.get("right") is None else world[conn["right"]],
        ))
    junctions = []
    for k in range(len(placements) - 1):
        right, left = world_conn[k][1], world_conn[k + 1][0]
        d = (float(np.linalg.norm(left - right))
             if right is not None and left is not None else 1.54)
        # deterministic fallback placements may leave odd junction
        # distances: regrowth uses a C-C-like length then
        junctions.append(d if 1.2 <= d <= 1.9 else 1.54)
    return {
        "unit": unit_id,
        "topology": chain["topology"],
        "monomers": [p["variant"] for p in placements],
        "templates": templates,
        "junction_lengths": junctions,
    }


def placement_from_transform(variant: str, R: np.ndarray,
                             t: np.ndarray) -> Dict[str, Any]:
    """geometry.json placement dict of a monomer transform world = R x + t."""
    angle, ax, ay, az = matrix_to_axis_angle(R)
    rotation = None if abs(angle) < 1e-12 else \
        {"angle_deg": angle, "axis": [ax, ay, az]}
    return {"variant": variant, "rotation": rotation,
            "translation": [float(x) for x in t]}
