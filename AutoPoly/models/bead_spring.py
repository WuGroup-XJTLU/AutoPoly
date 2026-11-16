"""
Bead-Spring Polymer Model Module

This module provides the BeadSpringPolymer class for generating coarse-grained
bead-spring polymer models for molecular dynamics simulations in LAMMPS.

Features:
- Multiple bead types (block copolymers)
- Per-triplet angle stiffness
- FENE bond support
- Linear and ring topologies
- Monte Carlo pre-equilibration for initial configurations
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import math
import numpy as np

from ..core.logger import setup_logger
from ..mc.collision import CollisionDetector, ObstacleCollisionDetector
from .architectures import BeadArchitecture, linear as _linear_arch, ring as _ring_arch

logger = setup_logger()


# Default bead density for box sizing (beads per sigma^3)
DEFAULT_BEAD_DENSITY = 0.85


@dataclass
class BeadType:
    """LJ parameters for a bead type."""
    name: str
    mass: float = 1.0
    epsilon: float = 1.0
    sigma: float = 1.0


@dataclass
class AngleType:
    """Angle parameters for a bead triplet."""
    triplet: Tuple[str, str, str]
    k: float = 10.0
    theta0: float = 180.0


@dataclass
class SAWConfig:
    """Configuration for Self-Avoiding Random Walk generation."""
    collision_sigma: float = 1.0          # Bead diameter for collision
    collision_tolerance: float = 0.1      # Overlap tolerance
    n_trials: int = 50                    # Trial positions per bead
    max_backtrack_depth: int = 10         # Max beads to remove when stuck
    max_total_backtracks: int = 1000      # Total backtrack budget per chain
    bond_angle_min: float = 60.0          # Min bond angle (degrees)
    bond_angle_max: float = 180.0         # Max bond angle
    ring_closure_trials: int = 100        # Extra trials for ring closure
    ring_closure_tolerance: float = 0.2   # Distance tolerance for ring closure
    system_retries: int = 3               # Whole-system retries when a chain fails


@dataclass
class MCConfig:
    """Monte Carlo equilibration configuration."""
    # Box sizing
    density: float = 0.85           # Bead density (beads per sigma^3)

    # MC move parameters
    max_displacement: float = 0.5   # Max single bead displacement
    max_angle: float = 0.3          # Max rotation angle (radians)
    temperature: float = 1.0        # Reduced temperature for Metropolis

    # Energy parameters
    lj_epsilon: float = 1.0         # LJ energy
    lj_sigma: float = 1.0           # LJ length
    lj_cutoff: float = 2.5          # LJ cutoff in sigma units
    bond_k: float = 100.0           # Bond spring constant
    bond_tolerance: float = 0.3     # Acceptable bond stretch

    # Equilibration settings
    n_steps: int = 10000            # Number of MC steps
    move_weights: Optional[Dict[str, float]] = None  # Move type probabilities


# =============================================================================
# Utility Functions for MC Equilibration
# =============================================================================

def calculate_box_size(n_beads: int, density: float = DEFAULT_BEAD_DENSITY) -> float:
    """
    Calculate cubic box size from number of beads and target density.

    Parameters:
        n_beads: Total number of beads in system.
        density: Target bead density (beads per sigma^3).

    Returns:
        Side length of cubic box.

    Formula: V = n_beads / density, box_size = V^(1/3)
    """
    volume = n_beads / density
    box_size = volume ** (1/3)
    return box_size


def compute_lj_energy(
    positions: Union[List[np.ndarray], np.ndarray],
    sigma: float = 1.0,
    epsilon: float = 1.0,
    cutoff: float = 2.5,
    exclude_bonded: Optional[List[Tuple[int, int]]] = None,
    box_size: Optional[float] = None,
) -> float:
    """
    Compute total LJ energy for non-bonded pairs.

    Parameters:
        positions: List of position arrays or (N, 3) array for each bead.
        sigma: LJ sigma parameter.
        epsilon: LJ epsilon parameter.
        cutoff: Cutoff distance in units of sigma.
        exclude_bonded: List of bonded pairs (i, j) to exclude from LJ calculation.
        box_size: Box size for periodic boundary conditions (None = no PBC).

    Returns:
        Total LJ energy.
    """
    n_beads = len(positions)
    cutoff_dist = cutoff * sigma
    cutoff_dist_sq = cutoff_dist ** 2
    energy = 0.0

    # Build set of excluded pairs for O(1) lookup
    excluded = set()
    if exclude_bonded:
        for i, j in exclude_bonded:
            excluded.add((min(i, j), max(i, j)))

    for i in range(n_beads):
        for j in range(i + 1, n_beads):
            if (i, j) in excluded:
                continue

            r_vec = positions[j] - positions[i]

            # Apply minimum image convention for PBC
            if box_size is not None:
                r_vec = r_vec - box_size * np.round(r_vec / box_size)

            r_sq = np.dot(r_vec, r_vec)

            if r_sq < cutoff_dist_sq and r_sq > 1e-10:
                r2_inv = (sigma * sigma) / r_sq
                r6_inv = r2_inv ** 3
                r12_inv = r6_inv ** 2
                energy += 4.0 * epsilon * (r12_inv - r6_inv)

    return energy


def compute_lj_energy_vectorized(
    positions: np.ndarray,
    excluded_mask: np.ndarray,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    cutoff: float = 2.5,
    box_size: Optional[float] = None,
) -> float:
    """
    Vectorized LJ energy using numpy broadcasting.

    Parameters:
        positions: (N, 3) array of bead positions.
        excluded_mask: (N, N) boolean array where True means pair is excluded.
        sigma: LJ sigma parameter.
        epsilon: LJ epsilon parameter.
        cutoff: Cutoff distance in units of sigma.
        box_size: Box size for periodic boundary conditions.

    Returns:
        Total LJ energy.
    """
    cutoff_dist_sq = (cutoff * sigma) ** 2

    # Pairwise differences: (N, N, 3)
    diff = positions[:, None, :] - positions[None, :, :]

    # Apply PBC
    if box_size is not None:
        diff = diff - box_size * np.round(diff / box_size)

    # Squared distances: (N, N)
    r_sq = np.sum(diff ** 2, axis=2)

    # Mask for valid pairs (not excluded, within cutoff, not self)
    # Only consider upper triangle to avoid double counting
    upper_tri = np.triu(np.ones_like(r_sq, dtype=bool), k=1)
    mask = upper_tri & (r_sq < cutoff_dist_sq) & (r_sq > 1e-10) & ~excluded_mask

    # LJ calculation only for valid pairs
    r_sq_valid = np.where(mask, r_sq, 1.0)  # Avoid division by zero
    r2_inv = (sigma ** 2) / r_sq_valid
    r6_inv = r2_inv ** 3
    r12_inv = r6_inv ** 2

    energy = 4.0 * epsilon * np.sum(np.where(mask, r12_inv - r6_inv, 0.0))
    return energy


def compute_local_lj_energy(
    positions: np.ndarray,
    bead_idx: int,
    excluded_set: set,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    cutoff: float = 2.5,
    box_size: Optional[float] = None,
) -> float:
    """
    Compute LJ energy contribution from one bead to all others - O(N).

    This is the key optimization: instead of recomputing O(N²) energy,
    we only compute the O(N) interactions involving the moved bead.

    Parameters:
        positions: (N, 3) array of bead positions.
        bead_idx: Index of the bead to compute interactions for.
        excluded_set: Set of bead indices that are bonded to bead_idx.
        sigma: LJ sigma parameter.
        epsilon: LJ epsilon parameter.
        cutoff: Cutoff distance in units of sigma.
        box_size: Box size for periodic boundary conditions.

    Returns:
        LJ energy contribution from bead_idx to all other beads.
    """
    cutoff_dist_sq = (cutoff * sigma) ** 2
    pos_i = positions[bead_idx]
    n_beads = len(positions)

    # Vectorized distance to all other beads
    diff = positions - pos_i  # (N, 3)

    if box_size is not None:
        diff = diff - box_size * np.round(diff / box_size)

    r_sq = np.sum(diff ** 2, axis=1)  # (N,)

    # Build mask: valid pairs (not excluded, within cutoff, not self)
    mask = (r_sq < cutoff_dist_sq) & (r_sq > 1e-10)
    mask[bead_idx] = False  # Exclude self

    # Exclude bonded neighbors
    for j in excluded_set:
        mask[j] = False

    # LJ only for valid pairs
    r_sq_valid = np.where(mask, r_sq, 1.0)  # Avoid division by zero
    r2_inv = (sigma ** 2) / r_sq_valid
    r6_inv = r2_inv ** 3
    r12_inv = r6_inv ** 2

    energy = 4.0 * epsilon * np.sum(np.where(mask, r12_inv - r6_inv, 0.0))
    return energy


def compute_local_bond_energy(
    positions: np.ndarray,
    bead_idx: int,
    bead_bonds: List[int],
    k_bond: float = 100.0,
    r0: float = 1.0,
    box_size: Optional[float] = None,
) -> float:
    """
    Compute bond energy for bonds involving a specific bead - O(degree).

    Parameters:
        positions: (N, 3) array of bead positions.
        bead_idx: Index of the bead.
        bead_bonds: List of bead indices that are bonded to bead_idx.
        k_bond: Bond spring constant.
        r0: Equilibrium bond length.
        box_size: Box size for periodic boundary conditions.

    Returns:
        Bond energy contribution from bead_idx.
    """
    if not bead_bonds:
        return 0.0

    pos_i = positions[bead_idx]
    energy = 0.0

    for j in bead_bonds:
        r_vec = positions[j] - pos_i

        if box_size is not None:
            r_vec = r_vec - box_size * np.round(r_vec / box_size)

        r = np.linalg.norm(r_vec)
        dr = r - r0
        energy += 0.5 * k_bond * dr * dr

    return energy


def compute_bond_energy(
    positions: Union[List[np.ndarray], np.ndarray],
    bonds: List[Tuple[int, int]],
    k_bond: float = 100.0,
    r0: float = 1.0,
    box_size: Optional[float] = None,
) -> float:
    """
    Compute harmonic bond energy.

    Parameters:
        positions: List of position arrays or (N, 3) array for each bead.
        bonds: List of (atom1_idx, atom2_idx) tuples (0-indexed).
        k_bond: Bond spring constant.
        r0: Equilibrium bond length.
        box_size: Box size for periodic boundary conditions.

    Returns:
        Total bond energy.
    """
    energy = 0.0

    for i, j in bonds:
        r_vec = positions[j] - positions[i]

        # Apply minimum image convention for PBC
        if box_size is not None:
            r_vec = r_vec - box_size * np.round(r_vec / box_size)

        r = np.linalg.norm(r_vec)
        dr = r - r0
        energy += 0.5 * k_bond * dr * dr

    return energy


def compute_bond_energy_vectorized(
    positions: np.ndarray,
    bonds: np.ndarray,
    k_bond: float = 100.0,
    r0: float = 1.0,
    box_size: Optional[float] = None,
) -> float:
    """
    Vectorized bond energy calculation.

    Parameters:
        positions: (N, 3) array of bead positions.
        bonds: (M, 2) array of bond pairs.
        k_bond: Bond spring constant.
        r0: Equilibrium bond length.
        box_size: Box size for periodic boundary conditions.

    Returns:
        Total bond energy.
    """
    if len(bonds) == 0:
        return 0.0

    pos_i = positions[bonds[:, 0]]
    pos_j = positions[bonds[:, 1]]

    diff = pos_j - pos_i

    if box_size is not None:
        diff = diff - box_size * np.round(diff / box_size)

    r = np.linalg.norm(diff, axis=1)
    dr = r - r0
    return 0.5 * k_bond * np.sum(dr ** 2)


def compute_total_energy(
    positions: Union[List[np.ndarray], np.ndarray],
    bonds: List[Tuple[int, int]],
    lj_sigma: float = 1.0,
    lj_epsilon: float = 1.0,
    lj_cutoff: float = 2.5,
    bond_k: float = 100.0,
    bond_r0: float = 1.0,
    box_size: Optional[float] = None,
) -> float:
    """
    Compute total system energy for Metropolis criterion.

    Parameters:
        positions: List of position arrays or (N, 3) array for each bead.
        bonds: List of (atom1_idx, atom2_idx) tuples.
        lj_sigma: LJ sigma parameter.
        lj_epsilon: LJ epsilon parameter.
        lj_cutoff: LJ cutoff in sigma units.
        bond_k: Bond spring constant.
        bond_r0: Equilibrium bond length.
        box_size: Box size for PBC.

    Returns:
        Total energy (LJ + bond).
    """
    lj_energy = compute_lj_energy(
        positions, lj_sigma, lj_epsilon, lj_cutoff,
        exclude_bonded=bonds, box_size=box_size
    )
    bond_energy = compute_bond_energy(
        positions, bonds, bond_k, bond_r0, box_size
    )
    return lj_energy + bond_energy


def build_exclusion_structures(
    n_beads: int,
    bonds: List[Tuple[int, int]],
) -> Tuple[np.ndarray, List[set], List[List[int]]]:
    """
    Build exclusion data structures for efficient energy calculations.

    Parameters:
        n_beads: Total number of beads.
        bonds: List of (atom1_idx, atom2_idx) tuples.

    Returns:
        Tuple of:
        - excluded_mask: (N, N) boolean array for vectorized LJ
        - excluded_neighbors: List of sets for local LJ energy
        - bond_neighbors: List of lists for local bond energy
    """
    # Build excluded pair mask (N, N) for vectorized calculation
    excluded_mask = np.zeros((n_beads, n_beads), dtype=bool)
    for i, j in bonds:
        excluded_mask[i, j] = True
        excluded_mask[j, i] = True

    # Build per-bead exclusion sets for local LJ energy
    excluded_neighbors = [set() for _ in range(n_beads)]
    for i, j in bonds:
        excluded_neighbors[i].add(j)
        excluded_neighbors[j].add(i)

    # Build per-bead bond lists for local bond energy
    bond_neighbors = [[] for _ in range(n_beads)]
    for i, j in bonds:
        bond_neighbors[i].append(j)
        bond_neighbors[j].append(i)

    return excluded_mask, excluded_neighbors, bond_neighbors


def metropolis_accept(delta_E: float, temperature: float = 1.0) -> bool:
    """
    Metropolis acceptance criterion.

    Accept if delta_E <= 0, else accept with probability exp(-delta_E/T).

    Parameters:
        delta_E: Energy change.
        temperature: Reduced temperature.

    Returns:
        True if move should be accepted.
    """
    if delta_E <= 0:
        return True
    return np.random.random() < np.exp(-delta_E / temperature)


def _apply_pbc(position: np.ndarray, box_size: float) -> np.ndarray:
    """Apply periodic boundary conditions to wrap position into box."""
    return position - box_size * np.floor(position / box_size + 0.5)


def _rotation_matrix_around_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Generate rotation matrix for rotation around an arbitrary axis.

    Uses Rodrigues' rotation formula.

    Parameters:
        axis: Unit vector defining rotation axis.
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def _random_rotation_matrix(max_angle: float) -> np.ndarray:
    """
    Generate a random rotation matrix with angle up to max_angle.

    Parameters:
        max_angle: Maximum rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    # Random axis (uniform on sphere)
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    axis = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    # Random angle
    angle = np.random.uniform(-max_angle, max_angle)

    return _rotation_matrix_around_axis(axis, angle)


# =============================================================================
# MC Move Functions
# =============================================================================

def mc_single_bead_displacement(
    positions: Union[List[np.ndarray], np.ndarray],
    bead_idx: int,
    max_disp: float = 0.5,
    box_size: Optional[float] = None,
) -> Tuple[Union[List[np.ndarray], np.ndarray], bool]:
    """
    Displace a single bead by random vector.

    Parameters:
        positions: List of position arrays or (N, 3) array.
        bead_idx: Index of bead to displace.
        max_disp: Maximum displacement in each direction.
        box_size: Box size for PBC.

    Returns:
        Tuple of (new_positions, is_valid).
    """
    if isinstance(positions, np.ndarray):
        new_positions = positions.copy()
        delta = np.random.uniform(-max_disp, max_disp, 3)
        new_positions[bead_idx] = new_positions[bead_idx] + delta
        if box_size is not None:
            new_positions[bead_idx] = _apply_pbc(new_positions[bead_idx], box_size)
        return new_positions, True
    else:
        new_positions = [p.copy() for p in positions]
        delta = np.random.uniform(-max_disp, max_disp, 3)
        new_positions[bead_idx] = new_positions[bead_idx] + delta
        if box_size is not None:
            new_positions[bead_idx] = _apply_pbc(new_positions[bead_idx], box_size)
        return new_positions, True


def mc_crankshaft_move(
    positions: Union[List[np.ndarray], np.ndarray],
    chain_start: int,
    chain_end: int,
    max_angle: float = 0.3,
    box_size: Optional[float] = None,
) -> Tuple[Union[List[np.ndarray], np.ndarray], bool]:
    """
    Rotate beads between i and j around the i-j axis.

    Keeps beads i and j fixed; rotates all beads strictly between them.

    Parameters:
        positions: List of position arrays or (N, 3) array.
        chain_start: Start index of the chain.
        chain_end: End index of the chain (exclusive).
        max_angle: Maximum rotation angle in radians.
        box_size: Box size for PBC.

    Returns:
        Tuple of (new_positions, is_valid).
    """
    chain_length = chain_end - chain_start

    # Need at least 4 beads for crankshaft (2 endpoints + at least 2 in between)
    if chain_length < 4:
        return positions, False

    # Select two beads i, j with at least one bead between them
    i_local = np.random.randint(0, chain_length - 3)
    j_local = np.random.randint(i_local + 3, chain_length)

    i = chain_start + i_local
    j = chain_start + j_local

    is_numpy = isinstance(positions, np.ndarray)
    if is_numpy:
        new_positions = positions.copy()
        pos_i = positions[i]
        pos_j = positions[j]
    else:
        new_positions = [p.copy() for p in positions]
        pos_i = positions[i]
        pos_j = positions[j]

    # Rotation axis
    axis = pos_j - pos_i
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-10:
        return positions, False

    # Random rotation angle
    angle = np.random.uniform(-max_angle, max_angle)
    R = _rotation_matrix_around_axis(axis, angle)

    # Rotate beads between i and j (exclusive of endpoints)
    for k in range(i + 1, j):
        rel_pos = new_positions[k] - pos_i
        new_rel_pos = R @ rel_pos
        new_positions[k] = pos_i + new_rel_pos

        if box_size is not None:
            new_positions[k] = _apply_pbc(new_positions[k], box_size)

    return new_positions, True


def mc_pivot_move(
    positions: Union[List[np.ndarray], np.ndarray],
    chain_start: int,
    chain_end: int,
    max_angle: float = 0.3,
    box_size: Optional[float] = None,
) -> Tuple[Union[List[np.ndarray], np.ndarray], bool]:
    """
    Rotate arm (pivot_idx+1 to chain_end) around pivot bead.

    Parameters:
        positions: List of position arrays or (N, 3) array.
        chain_start: Start index of the chain.
        chain_end: End index of the chain (exclusive).
        max_angle: Maximum rotation angle in radians.
        box_size: Box size for PBC.

    Returns:
        Tuple of (new_positions, is_valid).
    """
    chain_length = chain_end - chain_start

    # Need at least 2 beads
    if chain_length < 2:
        return positions, False

    # Select pivot (not the last bead)
    pivot_local = np.random.randint(0, chain_length - 1)
    pivot_idx = chain_start + pivot_local

    is_numpy = isinstance(positions, np.ndarray)
    if is_numpy:
        new_positions = positions.copy()
        pivot_pos = positions[pivot_idx].copy()
    else:
        new_positions = [p.copy() for p in positions]
        pivot_pos = positions[pivot_idx]

    # Random rotation
    R = _random_rotation_matrix(max_angle)

    # Rotate beads after pivot
    for k in range(pivot_idx + 1, chain_end):
        rel_pos = new_positions[k] - pivot_pos
        new_rel_pos = R @ rel_pos
        new_positions[k] = pivot_pos + new_rel_pos

        if box_size is not None:
            new_positions[k] = _apply_pbc(new_positions[k], box_size)

    return new_positions, True


def mc_reptation_move(
    positions: Union[List[np.ndarray], np.ndarray],
    chain_start: int,
    chain_end: int,
    bond_length: float = 1.0,
    box_size: Optional[float] = None,
) -> Tuple[Union[List[np.ndarray], np.ndarray], bool]:
    """
    Remove bead from one end (tail) and attach at other end (head).

    Slithering snake move that maintains chain connectivity.

    Parameters:
        positions: List of position arrays or (N, 3) array.
        chain_start: Start index of the chain.
        chain_end: End index of the chain (exclusive).
        bond_length: Bond length for new position.
        box_size: Box size for PBC.

    Returns:
        Tuple of (new_positions, is_valid).
    """
    chain_length = chain_end - chain_start

    # Need at least 2 beads
    if chain_length < 2:
        return positions, False

    is_numpy = isinstance(positions, np.ndarray)
    if is_numpy:
        new_positions = positions.copy()
    else:
        new_positions = [p.copy() for p in positions]

    # Randomly choose direction
    forward = np.random.random() < 0.5

    if forward:
        # Move bead from tail (chain_start) to head (chain_end-1)
        # Shift all beads toward chain_start
        head_pos = positions[chain_end - 1].copy() if is_numpy else positions[chain_end - 1]

        for k in range(chain_start, chain_end - 1):
            new_positions[k] = positions[k + 1].copy()

        # Generate new position at head
        random_dir = np.random.randn(3)
        random_dir = random_dir / np.linalg.norm(random_dir)
        new_positions[chain_end - 1] = head_pos + bond_length * random_dir

    else:
        # Move bead from head (chain_end-1) to tail (chain_start)
        # Shift all beads toward chain_end
        tail_pos = positions[chain_start].copy() if is_numpy else positions[chain_start]

        for k in range(chain_end - 1, chain_start, -1):
            new_positions[k] = positions[k - 1].copy()

        # Generate new position at tail
        random_dir = np.random.randn(3)
        random_dir = random_dir / np.linalg.norm(random_dir)
        new_positions[chain_start] = tail_pos + bond_length * random_dir

    # Apply PBC
    if box_size is not None:
        for k in range(chain_start, chain_end):
            new_positions[k] = _apply_pbc(new_positions[k], box_size)

    return new_positions, True


def mc_tree_pivot_move(
    positions: Union[List[np.ndarray], np.ndarray],
    pivot_idx: int,
    subtree_indices: List[int],
    max_angle: float = 0.3,
    box_size: Optional[float] = None,
) -> Tuple[Union[List[np.ndarray], np.ndarray], bool]:
    """
    Rotate a subtree of beads around a random axis through the pivot bead.

    This is the graph generalization of the pivot move for branched
    architectures: cutting edge (pivot, q) splits the molecule; the subtree
    on q's side (``subtree_indices``, must include q) is rotated rigidly.
    Because the rotation axis passes through the pivot bead, the (pivot, q)
    bond length and all internal subtree bonds are preserved exactly.

    Parameters:
        positions: List of position arrays or (N, 3) array.
        pivot_idx: Index of the pivot bead (rotation axis passes through it).
        subtree_indices: Indices of beads to rotate (the component beyond
            the cut edge, including q).
        max_angle: Maximum rotation angle (radians).
        box_size: Box size for PBC wrapping.

    Returns:
        Tuple of (new_positions, is_valid).
    """
    if not subtree_indices:
        return positions, False

    if isinstance(positions, list):
        new_positions = [p.copy() for p in positions]
    else:
        new_positions = positions.copy()

    pivot_pos = np.array(positions[pivot_idx], dtype=float)
    angle = np.random.uniform(-max_angle, max_angle)
    R = _random_rotation_matrix(angle)

    for idx in subtree_indices:
        rel = np.array(positions[idx], dtype=float) - pivot_pos
        new_positions[idx] = pivot_pos + R @ rel
        if box_size is not None:
            new_positions[idx] = _apply_pbc(new_positions[idx], box_size)

    return new_positions, True


def mc_segment_crankshaft_move(
    positions: Union[List[np.ndarray], np.ndarray],
    segment: List[int],
    max_angle: float = 0.3,
    box_size: Optional[float] = None,
) -> Tuple[Union[List[np.ndarray], np.ndarray], bool]:
    """
    Crankshaft move along an explicit bead path (segment).

    Graph generalization of :func:`mc_crankshaft_move` for branched
    architectures: two beads i, j are chosen along the segment path (with at
    least two path beads between them) and the path beads strictly between
    them are rotated around the i-j axis. The segment must be a genuine bond
    path so all rotated bonds are preserved.

    Parameters:
        positions: List of position arrays or (N, 3) array.
        segment: Bead indices forming a connected path (length >= 4).
        max_angle: Maximum rotation angle in radians.
        box_size: Box size for PBC.

    Returns:
        Tuple of (new_positions, is_valid).
    """
    if len(segment) < 4:
        return positions, False

    a = np.random.randint(0, len(segment) - 3)
    b = np.random.randint(a + 3, len(segment))

    i = segment[a]
    j = segment[b]

    is_numpy = isinstance(positions, np.ndarray)
    if is_numpy:
        new_positions = positions.copy()
    else:
        new_positions = [p.copy() for p in positions]

    pos_i = np.array(positions[i], dtype=float)
    pos_j = np.array(positions[j], dtype=float)

    axis = pos_j - pos_i
    if np.linalg.norm(axis) < 1e-10:
        return positions, False

    angle = np.random.uniform(-max_angle, max_angle)
    R = _rotation_matrix_around_axis(axis, angle)

    for k in range(a + 1, b):
        idx = segment[k]
        rel_pos = np.array(positions[idx], dtype=float) - pos_i
        new_positions[idx] = pos_i + R @ rel_pos
        if box_size is not None:
            new_positions[idx] = _apply_pbc(new_positions[idx], box_size)

    return new_positions, True


def _subtree_excluding(
    adjacency: Dict[int, List[int]],
    exclude: int,
    start: int,
) -> List[int]:
    """Beads reachable from ``start`` without visiting ``exclude``."""
    visited = {exclude, start}
    queue = deque([start])
    while queue:
        x = queue.popleft()
        for y in adjacency[x]:
            if y not in visited:
                visited.add(y)
                queue.append(y)
    visited.discard(exclude)
    return sorted(visited)


def _find_bridge_edges(
    adjacency: Dict[int, List[int]],
    edges: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Find bridge edges: edges whose removal disconnects the graph.

    Tree-pivot moves may only rotate across bridges; rotating across a
    cycle edge would break the other bond(s) of the cycle.
    """
    bridges = []
    for u, v in edges:
        # BFS from u without using edge (u, v)
        visited = {u}
        queue = deque([u])
        while queue:
            x = queue.popleft()
            for y in adjacency[x]:
                if (x == u and y == v) or (x == v and y == u):
                    continue
                if y not in visited:
                    visited.add(y)
                    queue.append(y)
        if v not in visited:
            bridges.append((u, v))
    return bridges


def build_chain_graph(
    architecture: BeadArchitecture,
    start: int,
) -> Dict[str, any]:
    """
    Build per-chain graph metadata for branched MC moves.

    Parameters:
        architecture: The chain's BeadArchitecture.
        start: Global index of the chain's first bead (bead i of the
            architecture maps to global index ``start + i``).

    Returns:
        Dict with keys:
            - "branched": True if any bead has degree > 2 or the graph is
              not a simple path
            - "adjacency": global-index adjacency dict
            - "edges": global-index bond list
            - "pivot_edges": bridge edges valid for tree-pivot moves
            - "segments": linear bead paths valid for segment crankshaft
    """
    n = architecture.n_beads
    adjacency: Dict[int, List[int]] = {start + i: [] for i in range(n)}
    edges = [(start + i, start + j) for i, j in architecture.bonds]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    degrees = architecture.degrees()
    # A simple path: exactly 2 ends (deg 1), all others deg 2, n-1 bonds
    is_simple_path = (
        architecture.n_bonds == n - 1
        and sum(1 for d in degrees if d == 1) == 2
        and all(d <= 2 for d in degrees)
    )
    branched = not is_simple_path

    return {
        "branched": branched,
        "adjacency": adjacency,
        "edges": edges,
        "pivot_edges": _find_bridge_edges(adjacency, edges) if branched else [],
        "segments": [
            [start + i for i in seg] for seg in architecture.linear_segments()
        ] if branched else [],
    }


# =============================================================================
# Multi-Chain Functions
# =============================================================================

def place_chains_in_box(
    chain_positions: List[List[np.ndarray]],
    box_size: float,
    min_separation: float = 2.0,
    max_attempts: int = 1000,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Place multiple chains randomly in periodic box.

    Parameters:
        chain_positions: List of chain positions (each chain is list of np.array).
        box_size: Cubic box side length.
        min_separation: Minimum distance between chain COMs.
        max_attempts: Maximum placement attempts per chain.

    Returns:
        Tuple of (all_positions, chain_indices) where chain_indices is
        list of (start, end) indices for each chain.
    """
    all_positions = []
    chain_indices = []
    placed_coms = []

    for chain_idx, chain in enumerate(chain_positions):
        # Compute original COM
        orig_com = np.mean(chain, axis=0)

        # Center chain at origin
        centered_chain = [p - orig_com for p in chain]

        placed = False
        for attempt in range(max_attempts):
            # Random position in box
            new_com = np.random.uniform(-box_size/2, box_size/2, 3)

            # Check separation from existing chains
            too_close = False
            for existing_com in placed_coms:
                dist_vec = new_com - existing_com
                dist_vec = dist_vec - box_size * np.round(dist_vec / box_size)
                if np.linalg.norm(dist_vec) < min_separation:
                    too_close = True
                    break

            if too_close:
                continue

            # Random rotation
            R = _random_rotation_matrix(np.pi)

            # Transform chain
            start_idx = len(all_positions)
            for p in centered_chain:
                rotated = R @ p
                new_pos = rotated + new_com
                new_pos = _apply_pbc(new_pos, box_size)
                all_positions.append(new_pos)

            end_idx = len(all_positions)
            chain_indices.append((start_idx, end_idx))
            placed_coms.append(new_com)
            placed = True
            break

        if not placed:
            logger.warning(
                f"Could not place chain {chain_idx} with min_separation={min_separation}. "
                "Placing without separation constraint."
            )
            new_com = np.random.uniform(-box_size/2, box_size/2, 3)
            R = _random_rotation_matrix(np.pi)

            start_idx = len(all_positions)
            for p in centered_chain:
                rotated = R @ p
                new_pos = rotated + new_com
                new_pos = _apply_pbc(new_pos, box_size)
                all_positions.append(new_pos)

            end_idx = len(all_positions)
            chain_indices.append((start_idx, end_idx))
            placed_coms.append(new_com)

    return all_positions, chain_indices


def mc_chain_translation(
    positions: Union[List[np.ndarray], np.ndarray],
    chain_indices: List[Tuple[int, int]],
    chain_idx: int,
    max_disp: float,
    box_size: float,
) -> Tuple[Union[List[np.ndarray], np.ndarray], bool]:
    """
    Translate entire chain by random displacement with PBC.

    Parameters:
        positions: List of position arrays or (N, 3) array.
        chain_indices: List of (start, end) indices for each chain.
        chain_idx: Index of chain to translate.
        max_disp: Maximum displacement.
        box_size: Box size for PBC.

    Returns:
        Tuple of (new_positions, is_valid).
    """
    start, end = chain_indices[chain_idx]
    is_numpy = isinstance(positions, np.ndarray)

    if is_numpy:
        new_positions = positions.copy()
    else:
        new_positions = [p.copy() for p in positions]

    delta = np.random.uniform(-max_disp, max_disp, 3)

    for k in range(start, end):
        new_positions[k] = new_positions[k] + delta
        new_positions[k] = _apply_pbc(new_positions[k], box_size)

    return new_positions, True


def mc_chain_rotation(
    positions: Union[List[np.ndarray], np.ndarray],
    chain_indices: List[Tuple[int, int]],
    chain_idx: int,
    max_angle: float,
    box_size: Optional[float] = None,
) -> Tuple[Union[List[np.ndarray], np.ndarray], bool]:
    """
    Rotate entire chain around its center of mass.

    Parameters:
        positions: List of position arrays or (N, 3) array.
        chain_indices: List of (start, end) indices for each chain.
        chain_idx: Index of chain to rotate.
        max_angle: Maximum rotation angle.
        box_size: Box size for PBC.

    Returns:
        Tuple of (new_positions, is_valid).
    """
    start, end = chain_indices[chain_idx]
    is_numpy = isinstance(positions, np.ndarray)

    if is_numpy:
        new_positions = positions.copy()
        chain_pos = positions[start:end]
    else:
        new_positions = [p.copy() for p in positions]
        chain_pos = [positions[k] for k in range(start, end)]

    com = np.mean(chain_pos, axis=0)

    # Random rotation
    R = _random_rotation_matrix(max_angle)

    # Rotate around COM
    for k in range(start, end):
        rel_pos = new_positions[k] - com
        new_rel_pos = R @ rel_pos
        new_positions[k] = com + new_rel_pos

        if box_size is not None:
            new_positions[k] = _apply_pbc(new_positions[k], box_size)

    return new_positions, True


# =============================================================================
# Self-Avoiding Random Walk (SAW) Functions
# =============================================================================

def _generate_uniform_sphere_points(n_points: int) -> np.ndarray:
    """
    Generate uniformly distributed points on unit sphere.

    Uses the golden spiral method for deterministic uniform distribution.

    Parameters:
        n_points: Number of points to generate.

    Returns:
        Array of shape (n_points, 3) with unit vectors.
    """
    indices = np.arange(n_points, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    # Y coordinates evenly spaced from -1 to 1
    y = 1 - (indices / (n_points - 1)) * 2 if n_points > 1 else np.array([0.0])
    radius_at_y = np.sqrt(1 - y * y)

    theta = phi * indices
    x = np.cos(theta) * radius_at_y
    z = np.sin(theta) * radius_at_y

    return np.column_stack([x, y, z])


def _generate_trial_positions(
    center: np.ndarray,
    bond_length: float,
    n_trials: int,
    prev_direction: Optional[np.ndarray] = None,
    angle_min: float = 60.0,
    angle_max: float = 180.0,
) -> np.ndarray:
    """
    Generate candidate positions on sphere with optional angle constraints.

    Parameters:
        center: Center point (previous bead position).
        bond_length: Distance from center for trial positions.
        n_trials: Number of trial positions to generate.
        prev_direction: Unit vector of previous bond direction (for angle filter).
        angle_min: Minimum allowed bond angle in degrees.
        angle_max: Maximum allowed bond angle in degrees.

    Returns:
        Array of shape (M, 3) with valid trial positions (M <= n_trials).
    """
    # Generate points on unit sphere
    directions = _generate_uniform_sphere_points(n_trials)

    # Scale and translate to bond length from center
    positions = center + bond_length * directions

    # Apply angle filter if we have a previous direction
    if prev_direction is not None and np.linalg.norm(prev_direction) > 1e-10:
        prev_dir_unit = prev_direction / np.linalg.norm(prev_direction)

        # Compute angle between new direction and NEGATIVE of previous direction
        # (so that 180 degrees means straight continuation)
        cos_angles = np.dot(directions, -prev_dir_unit)

        # Convert angle limits to cosines (note: cos is decreasing)
        cos_min = np.cos(np.radians(angle_max))  # Max angle -> min cos
        cos_max = np.cos(np.radians(angle_min))  # Min angle -> max cos

        # Filter positions by angle
        mask = (cos_angles >= cos_min) & (cos_angles <= cos_max)
        positions = positions[mask]

    return positions


def saw_grow_chain(
    n_beads: int,
    bond_length: float,
    start_position: np.ndarray,
    collision_detector: CollisionDetector,
    config: SAWConfig,
    topology: str = "linear",
    chain_id_offset: int = 0,
) -> Tuple[Optional[List[np.ndarray]], int]:
    """
    Grow a single polymer chain using Self-Avoiding Random Walk with backtracking.

    Parameters:
        n_beads: Number of beads in the chain.
        bond_length: Distance between consecutive beads.
        start_position: Position of the first bead.
        collision_detector: CollisionDetector for checking overlaps.
        config: SAWConfig with algorithm parameters.
        topology: "linear" or "ring".
        chain_id_offset: Offset for bead IDs in collision detector.

    Returns:
        Tuple of (positions, backtracks_used) where positions is None if failed.
    """
    if n_beads < 1:
        return [], 0

    positions = [start_position.copy()]
    backtrack_count = 0
    backtrack_depth = 1  # Current backtrack depth (exponential increase)

    # Add first bead to collision detector
    collision_detector.add_monomer(
        chain_id_offset,
        start_position,
        config.collision_sigma / 2
    )

    bead_idx = 1
    while bead_idx < n_beads:
        # Get previous direction for angle constraint
        if bead_idx >= 2:
            prev_direction = positions[bead_idx - 1] - positions[bead_idx - 2]
        else:
            prev_direction = None

        # For ring closure: last bead needs to connect back to first
        is_closing_ring = (topology == "ring" and bead_idx == n_beads - 1)

        # Generate trial positions
        n_trials = config.ring_closure_trials if is_closing_ring else config.n_trials
        trial_positions = _generate_trial_positions(
            positions[bead_idx - 1],
            bond_length,
            n_trials,
            prev_direction,
            config.bond_angle_min,
            config.bond_angle_max,
        )

        # Shuffle trials for randomness
        if len(trial_positions) > 0:
            np.random.shuffle(trial_positions)

        # Find valid position
        valid_position = None
        exclude_set = {chain_id_offset + bead_idx - 1}  # Exclude bonded neighbor

        for trial_pos in trial_positions:
            # Check collision with existing beads
            has_collision = collision_detector.check_collision(
                trial_pos,
                config.collision_sigma / 2,
                exclude_ids=exclude_set,
                tolerance=config.collision_tolerance,
            )

            if has_collision:
                continue

            # For ring closure: check distance to first bead
            if is_closing_ring:
                dist_to_first = np.linalg.norm(trial_pos - positions[0])
                if abs(dist_to_first - bond_length) > config.ring_closure_tolerance:
                    continue

            valid_position = trial_pos
            break

        if valid_position is not None:
            # Accept position
            positions.append(valid_position.copy())
            collision_detector.add_monomer(
                chain_id_offset + bead_idx,
                valid_position,
                config.collision_sigma / 2
            )
            bead_idx += 1
            backtrack_depth = 1  # Reset backtrack depth on success
        else:
            # Backtrack
            if backtrack_count >= config.max_total_backtracks:
                # Failed - remove all beads we added from detector
                for i in range(len(positions)):
                    collision_detector.remove_monomer(chain_id_offset + i)
                return None, backtrack_count

            # Determine how many beads to remove
            n_remove = min(backtrack_depth, bead_idx - 1, config.max_backtrack_depth)
            if n_remove == 0:
                # Can't backtrack further - failed
                for i in range(len(positions)):
                    collision_detector.remove_monomer(chain_id_offset + i)
                return None, backtrack_count

            # Remove beads from end
            for _ in range(n_remove):
                bead_idx -= 1
                collision_detector.remove_monomer(chain_id_offset + bead_idx)
                positions.pop()

            backtrack_count += 1
            backtrack_depth = min(backtrack_depth * 2, config.max_backtrack_depth)

    return positions, backtrack_count


def saw_generate_multi_chain(
    n_chains: int,
    n_beads_per_chain: int,
    bond_length: float,
    box_size: float,
    config: SAWConfig,
    topology: str = "linear",
    max_start_attempts: int = 100,
) -> Tuple[Optional[List[np.ndarray]], Optional[List[Tuple[int, int]]], Dict[str, any]]:
    """
    Generate multiple polymer chains using SAW.

    Parameters:
        n_chains: Number of chains to generate.
        n_beads_per_chain: Beads per chain.
        bond_length: Bond length between consecutive beads.
        box_size: Cubic box side length.
        config: SAWConfig with algorithm parameters.
        topology: "linear" or "ring".
        max_start_attempts: Max attempts to find valid starting position per chain.

    Returns:
        Tuple of (all_positions, chain_indices, stats) or (None, None, stats) if failed.
    """
    half_box = box_size / 2
    box_bounds = ((-half_box, half_box), (-half_box, half_box), (-half_box, half_box))

    # Cell size should be at least collision diameter
    cell_size = max(config.collision_sigma * 2, bond_length * 2)
    collision_detector = CollisionDetector(box_bounds, cell_size)

    all_positions = []
    chain_indices = []
    total_backtracks = 0
    chains_completed = 0

    for chain_idx in range(n_chains):
        chain_offset = len(all_positions)

        # Find valid starting position
        start_found = False
        for attempt in range(max_start_attempts):
            # Random position within box (with margin)
            margin = config.collision_sigma * 2
            start_pos = np.random.uniform(
                -half_box + margin,
                half_box - margin,
                3
            )

            # Check if position is collision-free
            if not collision_detector.check_collision(
                start_pos,
                config.collision_sigma / 2,
                tolerance=config.collision_tolerance
            ):
                start_found = True
                break

        if not start_found:
            logger.warning(
                f"SAW: Could not find valid start position for chain {chain_idx}"
            )
            return None, None, {
                "success": False,
                "chains_completed": chains_completed,
                "total_backtracks": total_backtracks,
                "failure_reason": "start_position",
            }

        # Grow chain
        chain_positions, backtracks = saw_grow_chain(
            n_beads_per_chain,
            bond_length,
            start_pos,
            collision_detector,
            config,
            topology,
            chain_offset,
        )

        total_backtracks += backtracks

        if chain_positions is None:
            logger.warning(
                f"SAW: Failed to grow chain {chain_idx} after {backtracks} backtracks"
            )
            return None, None, {
                "success": False,
                "chains_completed": chains_completed,
                "total_backtracks": total_backtracks,
                "failure_reason": "chain_growth",
            }

        # Store chain
        all_positions.extend(chain_positions)
        chain_indices.append((chain_offset, chain_offset + len(chain_positions)))
        chains_completed += 1

    return all_positions, chain_indices, {
        "success": True,
        "chains_completed": chains_completed,
        "total_backtracks": total_backtracks,
    }


# =============================================================================
# Graph-based SAW Generation (arbitrary architectures)
# =============================================================================


def saw_grow_graph(
    architecture: BeadArchitecture,
    bond_length: float,
    start_position: np.ndarray,
    collision_detector: CollisionDetector,
    config: SAWConfig,
    chain_id_offset: int = 0,
) -> Tuple[Optional[List[np.ndarray]], int]:
    """
    Grow a single chain of arbitrary architecture using SAW with backtracking.

    Beads are placed along the spanning-tree growth order of the architecture
    (BFS from bead 0): each bead is placed at ``bond_length`` from its
    already-placed parent. Cycle-closing edges (e.g. the ring closure) are
    enforced as distance constraints when the second endpoint is placed.

    Parameters:
        architecture: BeadArchitecture defining bead connectivity.
        bond_length: Distance between bonded beads.
        start_position: Position of the root bead (bead 0).
        collision_detector: CollisionDetector for checking overlaps.
        config: SAWConfig with algorithm parameters.
        chain_id_offset: Offset for bead IDs in collision detector.

    Returns:
        Tuple of (positions, backtracks_used); positions is None if failed.
        ``positions[i]`` corresponds to bead ``i`` of the architecture.
    """
    n_beads = architecture.n_beads
    if n_beads < 1:
        return [], 0

    order, parents, closing_edges = architecture.growth_order(root=0)

    # Closing constraints per bead: list of already-placed neighbors that a
    # bead must land within ring_closure_tolerance of bond_length from.
    closing_neighbors: Dict[int, List[int]] = {}
    for u, v in closing_edges:
        closing_neighbors.setdefault(u, []).append(v)
        closing_neighbors.setdefault(v, []).append(u)

    positions: Dict[int, np.ndarray] = {0: start_position.copy()}
    placed: List[int] = [0]  # placement stack in growth order
    backtrack_count = 0
    backtrack_depth = 1

    collision_detector.add_monomer(
        chain_id_offset, start_position, config.collision_sigma / 2
    )

    step = 1  # next index into order
    while step < n_beads:
        bead = order[step]
        parent = parents[bead]
        assert parent is not None
        grandparent = parents[parent]
        prev_direction = (
            positions[parent] - positions[grandparent]
            if grandparent is not None else None
        )

        # Beads with closing constraints get extra trials (like ring closure)
        pending_closure = [
            nbr for nbr in closing_neighbors.get(bead, []) if nbr in positions
        ]
        n_trials = config.ring_closure_trials if pending_closure else config.n_trials

        trial_positions = _generate_trial_positions(
            positions[parent],
            bond_length,
            n_trials,
            prev_direction,
            config.bond_angle_min,
            config.bond_angle_max,
        )
        if len(trial_positions) > 0:
            np.random.shuffle(trial_positions)

        valid_position = None
        exclude_set = {chain_id_offset + parent}
        for trial_pos in trial_positions:
            if collision_detector.check_collision(
                trial_pos,
                config.collision_sigma / 2,
                exclude_ids=exclude_set,
                tolerance=config.collision_tolerance,
            ):
                continue
            # Enforce closing-edge distance constraints
            closure_ok = True
            for nbr in pending_closure:
                dist = np.linalg.norm(trial_pos - positions[nbr])
                if abs(dist - bond_length) > config.ring_closure_tolerance:
                    closure_ok = False
                    break
            if closure_ok:
                valid_position = trial_pos
                break

        if valid_position is not None:
            positions[bead] = valid_position.copy()
            placed.append(bead)
            collision_detector.add_monomer(
                chain_id_offset + bead, valid_position, config.collision_sigma / 2
            )
            step += 1
            backtrack_depth = 1
        else:
            # Backtrack: remove the most recently placed beads (growth order)
            if backtrack_count >= config.max_total_backtracks or len(placed) <= 1:
                for b in placed:
                    collision_detector.remove_monomer(chain_id_offset + b)
                return None, backtrack_count

            n_remove = min(backtrack_depth, len(placed) - 1,
                           config.max_backtrack_depth)
            for _ in range(n_remove):
                b = placed.pop()
                del positions[b]
                collision_detector.remove_monomer(chain_id_offset + b)
            # Resume from the first unplaced bead in growth order
            placed_set = set(placed)
            step = next(
                i for i, b in enumerate(order) if b not in placed_set
            )
            backtrack_count += 1
            backtrack_depth = min(backtrack_depth * 2, config.max_backtrack_depth)

    return [positions[i] for i in range(n_beads)], backtrack_count


def saw_generate_graphs(
    architectures: List[BeadArchitecture],
    bond_length: float,
    box_size: float,
    config: SAWConfig,
    max_start_attempts: int = 100,
    obstacles: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[Optional[List[np.ndarray]], Optional[List[Tuple[int, int]]], Dict[str, any]]:
    """
    Generate multiple chains of (possibly different) architectures using SAW.

    This is the graph-based generalization of :func:`saw_generate_multi_chain`
    and also the driver for mixtures: each chain may have its own
    architecture.

    Parameters:
        architectures: One BeadArchitecture per chain to generate.
        bond_length: Bond length between bonded beads.
        box_size: Cubic box side length.
        config: SAWConfig with algorithm parameters.
        max_start_attempts: Max attempts to find valid starting position.
        obstacles: Optional (centers (k, 3), exclusion radii (k,)) of fixed
            spheres (coarse-grained fillers) no bead may enter.  With
            obstacles, all overlap checks (obstacles and beads) use the
            periodic minimum image of the box.  None (default) leaves the
            generator unchanged.

    Returns:
        Tuple of (all_positions, chain_indices, stats) or
        (None, None, stats) if failed.
    """
    half_box = box_size / 2
    box_bounds = ((-half_box, half_box), (-half_box, half_box), (-half_box, half_box))

    cell_size = max(config.collision_sigma * 2, bond_length * 2)
    if obstacles is None:
        collision_detector = CollisionDetector(box_bounds, cell_size)
    else:
        collision_detector = ObstacleCollisionDetector(
            box_bounds, cell_size, obstacles[0], obstacles[1], period=box_size
        )

    all_positions: List[np.ndarray] = []
    chain_indices: List[Tuple[int, int]] = []
    total_backtracks = 0
    chains_completed = 0

    for chain_idx, architecture in enumerate(architectures):
        chain_offset = len(all_positions)

        # Find valid starting position
        start_found = False
        for attempt in range(max_start_attempts):
            margin = config.collision_sigma * 2
            start_pos = np.random.uniform(-half_box + margin, half_box - margin, 3)
            if not collision_detector.check_collision(
                start_pos,
                config.collision_sigma / 2,
                tolerance=config.collision_tolerance,
            ):
                start_found = True
                break

        if not start_found:
            logger.warning(
                f"SAW: Could not find valid start position for chain {chain_idx}"
            )
            return None, None, {
                "success": False,
                "chains_completed": chains_completed,
                "total_backtracks": total_backtracks,
                "failure_reason": "start_position",
            }

        chain_positions, backtracks = saw_grow_graph(
            architecture,
            bond_length,
            start_pos,
            collision_detector,
            config,
            chain_offset,
        )

        total_backtracks += backtracks

        if chain_positions is None:
            logger.warning(
                f"SAW: Failed to grow chain {chain_idx} after {backtracks} backtracks"
            )
            return None, None, {
                "success": False,
                "chains_completed": chains_completed,
                "total_backtracks": total_backtracks,
                "failure_reason": "chain_growth",
            }

        all_positions.extend(chain_positions)
        chain_indices.append((chain_offset, chain_offset + len(chain_positions)))
        chains_completed += 1

    return all_positions, chain_indices, {
        "success": True,
        "chains_completed": chains_completed,
        "total_backtracks": total_backtracks,
    }


# =============================================================================
# Main MC Equilibration Routine
# =============================================================================

def mc_equilibrate(
    positions: Union[List[np.ndarray], np.ndarray],
    bonds: List[Tuple[int, int]],
    chain_indices: Optional[List[Tuple[int, int]]] = None,
    chain_graphs: Optional[List[Dict[str, any]]] = None,
    n_steps: int = 10000,
    temperature: float = 1.0,
    move_weights: Optional[Dict[str, float]] = None,
    box_size: Optional[float] = None,
    lj_sigma: float = 1.0,
    lj_epsilon: float = 1.0,
    lj_cutoff: float = 2.5,
    bond_k: float = 100.0,
    bond_r0: float = 1.0,
    max_displacement: float = 0.5,
    max_angle: float = 0.3,
    verbose: bool = False,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """
    Pre-equilibrate polymer configuration using MC moves.

    This implementation uses local energy updates for single-bead moves,
    reducing complexity from O(N²) to O(N) per step for the most common
    move type. For multi-bead moves, full energy recalculation is used.

    Parameters:
        positions: List of np.array or (N, 3) array, atom positions.
        bonds: List of (atom1_idx, atom2_idx) tuples (0-indexed).
        chain_indices: List of (start, end) tuples for each chain.
            If None, treats entire system as one chain.
        chain_graphs: Optional per-chain graph metadata for branched
            architectures (see :func:`build_chain_graph`). For chains marked
            "branched", pivot moves become tree-pivots across bridge edges,
            crankshaft moves are restricted to linear segments, and
            reptation is disabled. Linear chains (or chain_graphs=None)
            use the legacy path-based moves.
        n_steps: Number of MC steps.
        temperature: Reduced temperature.
        move_weights: Dict of move type probabilities.
        box_size: Apply PBC if specified.
        lj_sigma: LJ sigma parameter.
        lj_epsilon: LJ epsilon parameter.
        lj_cutoff: LJ cutoff in sigma units.
        bond_k: Bond spring constant.
        bond_r0: Equilibrium bond length.
        max_displacement: Max displacement for single bead moves.
        max_angle: Max rotation angle for pivot/crankshaft.
        verbose: Print progress.

    Returns:
        Tuple of (equilibrated_positions, acceptance_stats).
    """
    n_beads = len(positions)

    # Default chain indices
    if chain_indices is None:
        chain_indices = [(0, n_beads)]

    is_multi_chain = len(chain_indices) > 1

    # Default move weights - favor displacement moves for efficiency
    if move_weights is None:
        if is_multi_chain:
            move_weights = {
                "displacement": 0.4,  # Increased - most efficient move
                "crankshaft": 0.1,
                "pivot": 0.15,
                "reptation": 0.1,
                "chain_translation": 0.15,
                "chain_rotation": 0.1,
            }
        else:
            move_weights = {
                "displacement": 0.5,  # Increased - most efficient move
                "crankshaft": 0.15,
                "pivot": 0.2,
                "reptation": 0.15,
            }

    # Reptation is a linear-chain move; drop it when all chains are branched
    if chain_graphs is not None and chain_graphs:
        if all(g.get("branched", False) for g in chain_graphs):
            move_weights = {k: v for k, v in move_weights.items()
                            if k != "reptation"}

    # Normalize weights
    total_weight = sum(move_weights.values())
    move_probs = {k: v / total_weight for k, v in move_weights.items()}

    # Build move list and cumulative probabilities
    moves = list(move_probs.keys())
    cum_probs = np.cumsum([move_probs[m] for m in moves])

    # Statistics
    move_attempts = {m: 0 for m in moves}
    move_accepts = {m: 0 for m in moves}

    # Convert to numpy array for efficient operations
    if isinstance(positions, list):
        current_positions = np.array([p.copy() for p in positions])
    else:
        current_positions = positions.copy()

    # Build exclusion structures once (O(N + M) where M = num bonds)
    excluded_mask, excluded_neighbors, bond_neighbors = build_exclusion_structures(
        n_beads, bonds
    )

    # Convert bonds to numpy array for vectorized bond energy
    bonds_array = np.array(bonds) if bonds else np.zeros((0, 2), dtype=int)

    # Compute initial total energy
    current_energy = compute_lj_energy_vectorized(
        current_positions, excluded_mask, lj_sigma, lj_epsilon, lj_cutoff, box_size
    ) + compute_bond_energy_vectorized(
        current_positions, bonds_array, bond_k, bond_r0, box_size
    )

    # Progress reporting interval
    report_interval = max(1, n_steps // 10)

    # MC loop
    for step in range(n_steps):
        # Select move type using searchsorted for efficiency
        r = np.random.random()
        move_idx = np.searchsorted(cum_probs, r)
        move_type = moves[move_idx]

        move_attempts[move_type] += 1

        # Select a chain for moves that operate on chains
        chain_idx = np.random.randint(len(chain_indices))
        start, end = chain_indices[chain_idx]

        # Perform move and compute energy change
        accepted = False

        if move_type == "displacement":
            # Single bead displacement - use LOCAL energy update (O(N) instead of O(N²))
            bead_idx = np.random.randint(start, end)

            # Store old position
            old_pos = current_positions[bead_idx].copy()

            # Compute old local energy (LJ + bonds involving this bead)
            old_lj = compute_local_lj_energy(
                current_positions, bead_idx, excluded_neighbors[bead_idx],
                lj_sigma, lj_epsilon, lj_cutoff, box_size
            )
            old_bond = compute_local_bond_energy(
                current_positions, bead_idx, bond_neighbors[bead_idx],
                bond_k, bond_r0, box_size
            )

            # Apply displacement in-place
            delta = np.random.uniform(-max_displacement, max_displacement, 3)
            current_positions[bead_idx] = old_pos + delta
            if box_size is not None:
                current_positions[bead_idx] = _apply_pbc(current_positions[bead_idx], box_size)

            # Compute new local energy
            new_lj = compute_local_lj_energy(
                current_positions, bead_idx, excluded_neighbors[bead_idx],
                lj_sigma, lj_epsilon, lj_cutoff, box_size
            )
            new_bond = compute_local_bond_energy(
                current_positions, bead_idx, bond_neighbors[bead_idx],
                bond_k, bond_r0, box_size
            )

            delta_E = (new_lj + new_bond) - (old_lj + old_bond)

            if metropolis_accept(delta_E, temperature):
                current_energy += delta_E
                accepted = True
            else:
                # Reject: restore old position
                current_positions[bead_idx] = old_pos

        else:
            # Multi-bead moves - use full energy recalculation
            new_positions = current_positions
            is_valid = False

            graph = chain_graphs[chain_idx] if chain_graphs is not None else None
            branched = graph is not None and graph.get("branched", False)

            if move_type == "crankshaft":
                if branched:
                    segments = graph.get("segments", [])
                    if segments:
                        segment = segments[np.random.randint(len(segments))]
                        new_positions, is_valid = mc_segment_crankshaft_move(
                            current_positions, segment, max_angle, box_size
                        )
                else:
                    new_positions, is_valid = mc_crankshaft_move(
                        current_positions, start, end, max_angle, box_size
                    )

            elif move_type == "pivot":
                if branched:
                    pivot_edges = graph.get("pivot_edges", [])
                    if pivot_edges:
                        u, v = pivot_edges[np.random.randint(len(pivot_edges))]
                        adjacency = graph["adjacency"]
                        subtree_v = _subtree_excluding(adjacency, u, v)
                        subtree_u = _subtree_excluding(adjacency, v, u)
                        # Rotate the smaller side for efficiency
                        if len(subtree_v) <= len(subtree_u):
                            pivot_idx, subtree = u, subtree_v
                        else:
                            pivot_idx, subtree = v, subtree_u
                        new_positions, is_valid = mc_tree_pivot_move(
                            current_positions, pivot_idx, subtree,
                            max_angle, box_size
                        )
                else:
                    new_positions, is_valid = mc_pivot_move(
                        current_positions, start, end, max_angle, box_size
                    )

            elif move_type == "reptation":
                if not branched:
                    new_positions, is_valid = mc_reptation_move(
                        current_positions, start, end, bond_r0, box_size
                    )

            elif move_type == "chain_translation":
                if box_size is not None:
                    new_positions, is_valid = mc_chain_translation(
                        current_positions, chain_indices, chain_idx,
                        max_displacement, box_size
                    )

            elif move_type == "chain_rotation":
                new_positions, is_valid = mc_chain_rotation(
                    current_positions, chain_indices, chain_idx,
                    max_angle, box_size
                )

            if is_valid:
                # Compute new total energy using vectorized functions
                new_energy = compute_lj_energy_vectorized(
                    new_positions, excluded_mask, lj_sigma, lj_epsilon, lj_cutoff, box_size
                ) + compute_bond_energy_vectorized(
                    new_positions, bonds_array, bond_k, bond_r0, box_size
                )

                delta_E = new_energy - current_energy
                if metropolis_accept(delta_E, temperature):
                    current_positions = new_positions
                    current_energy = new_energy
                    accepted = True

        if accepted:
            move_accepts[move_type] += 1

        # Progress report
        if verbose and (step + 1) % report_interval == 0:
            logger.info(
                f"MC step {step + 1}/{n_steps}, Energy: {current_energy:.4f}"
            )

    # Compute acceptance rates
    acceptance_stats = {}
    for m in moves:
        if move_attempts[m] > 0:
            acceptance_stats[m] = move_accepts[m] / move_attempts[m]
        else:
            acceptance_stats[m] = 0.0

    if verbose:
        logger.info(f"Final energy: {current_energy:.4f}")
        logger.info(f"Acceptance rates: {acceptance_stats}")

    # Convert back to list for backward compatibility
    result_positions = [current_positions[i].copy() for i in range(n_beads)]
    return result_positions, acceptance_stats


# =============================================================================
# Shared helpers (used by BeadSpringPolymer and BeadSpringSystem)
# =============================================================================


def canonical_bead_triplet(t1: str, t2: str, t3: str) -> Tuple[str, str, str]:
    """Return canonical triplet (smaller endpoint first alphabetically)."""
    if t1 <= t3:
        return (t1, t2, t3)
    return (t3, t2, t1)


def lb_pair_coeffs(
    bead_types: List["BeadType"],
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """
    Lorentz-Berthelot mixing for all bead type pairs.

    Returns:
        Dict mapping (type_i, type_j) (1-based) to (epsilon_ij, sigma_ij).
    """
    coeffs = {}
    n_types = len(bead_types)
    for i in range(n_types):
        for j in range(i, n_types):
            bt_i = bead_types[i]
            bt_j = bead_types[j]
            sigma_ij = (bt_i.sigma + bt_j.sigma) / 2
            epsilon_ij = math.sqrt(bt_i.epsilon * bt_j.epsilon)
            coeffs[(i + 1, j + 1)] = (epsilon_ij, sigma_ij)
    return coeffs


def resolve_angle_params(
    triplet: Tuple[str, str, str],
    angle_types: List["AngleType"],
    default_k: float,
    default_theta0: float,
) -> Tuple[float, float]:
    """Return (k, theta0) for a canonical triplet, using defaults if unset."""
    for angle_type in angle_types:
        if canonical_bead_triplet(*angle_type.triplet) == triplet:
            return (angle_type.k, angle_type.theta0)
    return (default_k, default_theta0)


def build_angle_type_map(
    chain_bead_types: List[List[str]],
    chain_triplets: List[List[Tuple[int, int, int]]],
) -> Dict[Tuple[str, str, str], int]:
    """
    Map canonical triplets to LAMMPS angle type IDs across all chains.

    Args:
        chain_bead_types: Bead type names per chain.
        chain_triplets: Local (i, j, k) triplets per chain.

    Returns:
        Dict mapping canonical triplet to 1-based angle type ID.
    """
    unique_triplets = set()
    for bead_types_seq, triplets in zip(chain_bead_types, chain_triplets):
        for i, j, k in triplets:
            unique_triplets.add(
                canonical_bead_triplet(
                    bead_types_seq[i], bead_types_seq[j], bead_types_seq[k]
                )
            )
    sorted_triplets = sorted(unique_triplets)
    return {triplet: i + 1 for i, triplet in enumerate(sorted_triplets)}


def write_lammps_input_script(
    path: str,
    bead_types: List["BeadType"],
    pair_coeffs: Dict[Tuple[int, int], Tuple[float, float]],
    bond_style: str,
    k_bond: float,
    bond_length: float,
    fene_r0: float,
    pair_style: str,
    use_angles: bool,
    angle_type_map: Dict[Tuple[str, str, str], int],
    angle_types: List["AngleType"],
    default_k_angle: float,
    default_theta0: float,
    fillers: Optional[Dict[str, List[str]]] = None,
) -> None:
    """
    Write the LAMMPS input script (in.polymer) for a bead-spring system.

    Shared by BeadSpringPolymer (single species) and BeadSpringSystem
    (mixtures).

    Note: For Kremer-Grest polymers using FENE bonds, the WCA cutoff
    (2^(1/6) * sigma ≈ 1.12246) should be used instead of full LJ cutoff
    (pair_style="wca"). The equilibrium bond length of ~0.97 emerges from
    the balance of FENE + WCA potentials - there is no explicit r0 in FENE.

    ``fillers`` (motion -> filler atom-type ids) adds the CG-filler groups,
    the held-filler minimization and rigid/frozen filler integration (see
    AutoPoly.nanoparticles.cg); a pair coefficient may carry a third value,
    its own cutoff.
    """
    if fillers:
        from ..nanoparticles import cg
    max_sigma = max(bt.sigma for bt in bead_types)

    if pair_style == 'wca':
        cutoff = (2 ** (1/6)) * max_sigma
        pair_modify = "pair_modify     shift yes\n"
    else:
        cutoff = 2.5 * max_sigma
        pair_modify = ""

    with open(f"{path}/in.polymer", 'w') as f:
        f.write("# LAMMPS input script for bead-spring polymer\n")
        f.write("# Auto-generated by AutoPoly BeadSpringPolymer\n\n")

        f.write("units           lj\n")
        f.write("atom_style      molecular\n")
        f.write("boundary        p p p\n\n")

        f.write("read_data       polymer.data\n\n")

        f.write(f"pair_style      lj/cut {cutoff:.5f}\n")
        for (i, j), coeff in sorted(pair_coeffs.items()):
            eps, sig = coeff[:2]
            rc = f" {coeff[2]:.5f}" if len(coeff) > 2 else ""
            f.write(f"pair_coeff      {i} {j} {eps:.4f} {sig:.4f}{rc}\n")
        if pair_modify:
            f.write(pair_modify)
        f.write("\n")

        if bond_style == "fene":
            eps = bead_types[0].epsilon
            sig = bead_types[0].sigma
            f.write("bond_style      fene\n")
            # FENE bond: K, R0, epsilon, sigma
            # Note: No r0 parameter - equilibrium emerges from FENE+WCA
            f.write(f"bond_coeff      1 {k_bond:.1f} {fene_r0:.4f} {eps:.4f} {sig:.4f}\n")
            f.write("special_bonds   fene\n\n")
        else:
            f.write("bond_style      harmonic\n")
            f.write(f"bond_coeff      1 {k_bond:.1f} {bond_length:.4f}\n\n")

        if use_angles:
            f.write("angle_style     harmonic\n")
            for triplet, type_id in sorted(angle_type_map.items(), key=lambda x: x[1]):
                k, theta0 = resolve_angle_params(
                    triplet, angle_types, default_k_angle, default_theta0
                )
                triplet_str = "-".join(triplet)
                f.write(f"angle_coeff     {type_id} {k:.4f} {theta0:.1f}  # {triplet_str}\n")
            f.write("\n")

        # Larger skin (2.0) for FENE to prevent lost atoms
        f.write("neighbor        2.0 bin\n")
        f.write("neigh_modify    every 2 delay 4 check yes\n\n")

        if fillers:
            for line in cg.group_lines(fillers):
                f.write(line + "\n")
            f.write(cg.SETUP + "\n")
            f.write(cg.THERMO)
        else:
            f.write("thermo_style    custom step temp pe ke etotal press vol density\n")
        f.write("thermo          1000\n\n")

        f.write("dump            1 all custom 1000 dump.lammpstrj id type mol x y z\n")
        f.write("dump_modify     1 sort id\n\n")

        f.write("# Energy minimization\n")
        if fillers:
            f.write(cg.HOLD)
        f.write("minimize        1.0e-4 1.0e-6 1000 10000\n")
        if fillers:
            f.write(cg.UNHOLD)
        f.write("write_restart   min.restart\n")
        f.write("write_data      min.data\n")
        f.write("reset_timestep  0\n\n")

        # Timestep: 0.001 is standard for Kremer-Grest with FENE bonds
        f.write("# Production MD\n")
        f.write("timestep        0.001\n")
        # Tdamp = 0.1 (100x timestep) for proper temperature control
        if fillers:
            f.write(cg.production_lines(fillers))
        else:
            f.write("fix             1 all nvt temp 1.0 1.0 0.1 tchain 3\n")
        f.write("run             100000\n")
        f.write("write_restart   prod.restart\n")
        f.write("write_data      prod.data\n")
        if fillers:
            f.write(cg.compression_hint(fillers))


class BeadSpringPolymer:
    """
    Bead-spring polymer model generator for LAMMPS simulations.

    Supports multiple bead types for block copolymers, per-triplet angle
    stiffness, and both harmonic and FENE bond styles.
    """

    VALID_TOPOLOGIES = ["linear", "ring"]
    VALID_BOND_STYLES = ["harmonic", "fene"]
    VALID_PAIR_STYLES = ["lj", "wca"]
    VALID_GENERATION_METHODS = ["geometric", "saw", "mc"]
    VALID_BACKENDS = ["moltemplate", "direct"]

    def __init__(
        self,
        name: str,
        system: object,
        n_chains: int,
        bead_types: List[BeadType],
        sequence: Optional[Union[List[str], List[Tuple[str, int]], str]] = None,
        topology: str = "linear",
        # Architecture (alternative to sequence/topology)
        architecture: Optional[BeadArchitecture] = None,
        # Bond parameters
        bond_length: float = 1.0,
        bond_style: str = "harmonic",
        k_bond: float = 30.0,
        fene_r0: float = 1.5,
        # Pair style parameters
        pair_style: str = "lj",  # "lj" (full LJ 2.5) or "wca" (WCA 2^(1/6))
        # Angle parameters
        use_angles: bool = False,
        default_k_angle: float = 10.0,
        default_theta0: float = 180.0,
        angle_types: Optional[List[AngleType]] = None,
        include_branch_angles: bool = True,
        # Box sizing
        density: Optional[float] = None,
        box_size: Optional[float] = None,
        # Generation method
        generation_method: str = "saw",  # "geometric", "saw", or "mc"
        saw_config: Optional[SAWConfig] = None,
        # Output backend
        backend: str = "moltemplate",  # "moltemplate" (default) or "direct"
        # MC equilibration (legacy, use generation_method="mc" instead)
        equilibrate: bool = False,
        mc_config: Optional[MCConfig] = None,
    ) -> None:
        """
        Initialize bead-spring polymer generator.

        Args:
            name: Name for output files.
            system: System object containing path information.
            n_chains: Number of polymer chains.
            bead_types: List of BeadType objects defining bead parameters.
            sequence: Chain structure as block pattern, explicit list, or string.
                Mutually exclusive with ``architecture``.
            topology: "linear" or "ring" (only used with ``sequence``).
            architecture: BeadArchitecture describing an arbitrary chain
                graph (star, comb, graft, tadpole, dendrimer, custom...).
                Mutually exclusive with ``sequence``/``topology``. Build one
                with the factories in ``AutoPoly.models.architectures``
                (``linear``, ``ring``, ``star``, ``comb``, ``graft``,
                ``tadpole``, ``dendrimer``, ``custom``).
            bond_length: Equilibrium bond length.
            bond_style: "harmonic" or "fene".
            k_bond: Bond force constant.
            fene_r0: FENE maximum extension (only used if bond_style="fene").
            pair_style: Pair interaction style - "lj" for full LJ (cutoff 2.5) or
                "wca" for WCA purely repulsive (cutoff 2^(1/6)*sigma ≈ 1.12246).
                For Kremer-Grest melts at P=0, use "wca" (recommended).
            use_angles: Whether to include angle potentials.
            default_k_angle: Default angle force constant for unspecified triplets.
            default_theta0: Default equilibrium angle for unspecified triplets.
            angle_types: List of AngleType objects for specific triplet parameters.
            include_branch_angles: Whether to include angle triplets centered
                on branch points (beads with degree > 2, e.g. comb graft
                points and star centers). Their stiffness is configured like
                any other triplet via ``angle_types``/``default_k_angle``.
                Only relevant for branched architectures.
            density: Target bead density (beads/sigma^3). If set, box size is calculated.
            box_size: Explicit box size. Overrides density if both are set.
            generation_method: Method for generating initial configurations:
                - "geometric": Simple geometric placement (fast, may have overlaps)
                - "saw": Self-Avoiding Random Walk (fast, overlap-free)
                - "mc": Monte Carlo equilibration (slow, equilibrated)
            saw_config: SAW configuration. Uses defaults if None.
            backend: Output backend used by :meth:`generate`:
                - "moltemplate" (default): emit .lt files and run the bundled
                  moltemplate -> moltemplate/system.data + system.in.*
                  (standard AutoPoly output layout; mixable with other
                  moltemplate objects).
                - "direct": write polymer.data + in.polymer directly
                  (lightweight; better for very large melts).
            equilibrate: Whether to run MC equilibration (legacy, use generation_method="mc").
            mc_config: Monte Carlo configuration. Uses defaults if None.

        Raises:
            ValueError: If parameters are invalid.

        Note on FENE Bonds:
            The FENE potential has NO explicit r0 (equilibrium bond length) parameter.
            The equilibrium distance of ~0.97 for Kremer-Grest polymers EMERGES from
            the balance of FENE (attractive, wants r→0) + WCA (repulsive, prevents r<~1.0).
            LAMMPS FENE syntax: bond_coeff * K R0 epsilon sigma
        """
        if architecture is None and topology not in self.VALID_TOPOLOGIES:
            raise ValueError(f"Topology must be one of: {self.VALID_TOPOLOGIES}")
        if architecture is not None and (sequence is not None or topology != "linear"):
            raise ValueError(
                "Pass either 'architecture' or 'sequence'/'topology', not both"
            )
        if architecture is None and sequence is None:
            raise ValueError("Either 'sequence' or 'architecture' is required")
        if bond_style not in self.VALID_BOND_STYLES:
            raise ValueError(f"Bond style must be one of: {self.VALID_BOND_STYLES}")
        if generation_method not in self.VALID_GENERATION_METHODS:
            raise ValueError(f"Generation method must be one of: {self.VALID_GENERATION_METHODS}")
        if pair_style not in self.VALID_PAIR_STYLES:
            raise ValueError(f"Pair style must be one of: {self.VALID_PAIR_STYLES}")
        if backend not in self.VALID_BACKENDS:
            raise ValueError(f"Backend must be one of: {self.VALID_BACKENDS}")
        if not bead_types:
            raise ValueError("At least one bead type is required")

        self.name = name
        self.system = system
        self.path = f"{self.system.get_folder_path()}/{self.name}" if system else f"./{name}"
        self.n_chains = n_chains

        # Store bead types
        self.bead_types = bead_types
        self._bead_type_map: Dict[str, BeadType] = {bt.name: bt for bt in bead_types}
        self._bead_type_id: Dict[str, int] = {bt.name: i + 1 for i, bt in enumerate(bead_types)}

        # Build the chain graph (the single internal representation)
        if architecture is not None:
            architecture.validate(
                known_bead_types=[bt.name for bt in bead_types]
            )
            self._architecture = architecture
            self.topology = architecture.name
            self._sequence = architecture.bead_types
        else:
            self.topology = topology
            # Parse and validate sequence
            self._sequence = self._parse_sequence(sequence)
            self._validate()
            self._architecture = (
                _ring_arch(self._sequence) if topology == "ring"
                else _linear_arch(self._sequence)
            )
        if architecture is not None:
            self._validate()

        # Bond parameters
        self.bond_length = bond_length
        self.bond_style = bond_style
        self.k_bond = k_bond
        self.fene_r0 = fene_r0

        # Pair style parameter
        self._pair_style = pair_style

        # Angle parameters
        self.use_angles = use_angles
        self.default_k_angle = default_k_angle
        self.default_theta0 = default_theta0
        self._angle_types = angle_types or []
        self.include_branch_angles = include_branch_angles

        # Graph-derived bond/angle structure (local bead indices per chain)
        self._bonds_local: List[Tuple[int, int]] = list(self._architecture.bonds)
        self._angle_triplets_local: List[Tuple[int, int, int]] = (
            self._architecture.angle_triplets(
                include_branch=self.include_branch_angles
            )
        )

        # Box sizing parameters
        self.density = density
        self._explicit_box_size = box_size

        # Generation method parameters
        self._generation_method = generation_method
        self._saw_config = saw_config

        # Output backend
        self._backend = backend

        # MC equilibration parameters (legacy support)
        self._equilibrate = equilibrate
        self._mc_config = mc_config

        # If equilibrate=True is set, use mc generation method
        if equilibrate and generation_method == "geometric":
            self._generation_method = "mc"

        # Internal state for positions (populated during generate_data_file)
        self._positions: Optional[List[np.ndarray]] = None
        self._chain_indices: Optional[List[Tuple[int, int]]] = None
        self._bonds: Optional[List[Tuple[int, int]]] = None

        # Build internal maps
        self._pair_coeffs = self._build_pair_coeffs()
        self._angle_type_map: Dict[Tuple[str, str, str], int] = {}
        if self.use_angles:
            self._angle_type_map = self._build_angle_type_map()

        # Create output directory
        Path(self.path).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized bead-spring polymer: {n_chains} chains, "
            f"{len(self._sequence)} beads each, {self.topology} architecture, "
            f"{len(bead_types)} bead type(s)"
        )

    @property
    def n_beads(self) -> int:
        """Number of beads per chain (derived from sequence)."""
        return len(self._sequence)

    def _parse_sequence(self, seq: Union[List[str], List[Tuple[str, int]], str]) -> List[str]:
        """
        Convert sequence input to explicit bead type list.

        Args:
            seq: Sequence as block pattern, explicit list, or string.

        Returns:
            List of bead type names.
        """
        if isinstance(seq, str):
            # String format: "AABB" -> ["A", "A", "B", "B"]
            return list(seq)

        result = []
        for item in seq:
            if isinstance(item, tuple):
                # Block format: ("A", 20) -> ["A"] * 20
                bead_name, count = item
                result.extend([bead_name] * count)
            else:
                # Explicit list format
                result.append(item)
        return result

    def _validate(self) -> None:
        """Validate bead types and sequence consistency."""
        if not self._sequence:
            raise ValueError("Sequence cannot be empty")

        for bead_name in self._sequence:
            if bead_name not in self._bead_type_map:
                raise ValueError(
                    f"Unknown bead type '{bead_name}' in sequence. "
                    f"Available types: {list(self._bead_type_map.keys())}"
                )

    def _get_canonical_triplet(self, t1: str, t2: str, t3: str) -> Tuple[str, str, str]:
        """
        Return canonical triplet (smaller endpoint first alphabetically).

        Args:
            t1, t2, t3: Bead type names.

        Returns:
            Canonical triplet tuple.
        """
        return canonical_bead_triplet(t1, t2, t3)

    def _build_pair_coeffs(self) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """
        Calculate Lorentz-Berthelot mixing for all bead type pairs.

        Returns:
            Dict mapping (type_i, type_j) to (epsilon_ij, sigma_ij).
        """
        return lb_pair_coeffs(self.bead_types)

    def _build_angle_type_map(self) -> Dict[Tuple[str, str, str], int]:
        """
        Map canonical triplets to LAMMPS type IDs.

        Returns:
            Dict mapping canonical triplet to angle type ID.
        """
        # Collect all unique canonical triplets from the chain graph.
        # For linear/ring architectures this reproduces the legacy
        # path-based enumeration (including ring wrap-arounds) exactly;
        # branched architectures additionally contribute their triplets
        # (branch-point triplets only if include_branch_angles is True).
        unique_triplets = set()

        for i, j, k in self._angle_triplets_local:
            triplet = self._get_canonical_triplet(
                self._sequence[i],
                self._sequence[j],
                self._sequence[k]
            )
            unique_triplets.add(triplet)

        # Sort triplets for consistent ordering
        sorted_triplets = sorted(unique_triplets)

        return {triplet: i + 1 for i, triplet in enumerate(sorted_triplets)}

    def _get_angle_params(self, triplet: Tuple[str, str, str]) -> Tuple[float, float]:
        """
        Return (k, theta0) for a triplet, using default if not specified.

        Args:
            triplet: Canonical triplet tuple.

        Returns:
            Tuple of (k, theta0).
        """
        return resolve_angle_params(
            triplet, self._angle_types,
            self.default_k_angle, self.default_theta0,
        )

    def _calculate_box_size(self) -> float:
        """
        Calculate box size based on density or explicit setting.

        Returns:
            Box side length.
        """
        total_beads = self.n_chains * self.n_beads

        if self._explicit_box_size is not None:
            return self._explicit_box_size
        elif self.density is not None:
            return calculate_box_size(total_beads, self.density)
        else:
            # Default behavior (original logic)
            if self.topology == "ring":
                radius = self.bond_length * self.n_beads / (2 * np.pi)
                spacing = radius * 3
                n_per_dim = int(np.ceil(np.cbrt(self.n_chains)))
                return max(n_per_dim * spacing * 2, 50.0)
            else:
                return max(self.n_beads * self.bond_length * 2, 50.0)

    def _compute_total_energy(self) -> float:
        """
        Compute total system energy (LJ + bonds).

        Returns:
            Total energy.
        """
        if self._positions is None or self._bonds is None:
            raise ValueError("Positions and bonds must be initialized first")

        # Get average LJ parameters from bead types
        avg_sigma = np.mean([bt.sigma for bt in self.bead_types])
        avg_epsilon = np.mean([bt.epsilon for bt in self.bead_types])

        box_size = self._calculate_box_size()

        return compute_total_energy(
            self._positions,
            self._bonds,
            lj_sigma=avg_sigma,
            lj_epsilon=avg_epsilon,
            lj_cutoff=2.5,
            bond_k=self.k_bond,
            bond_r0=self.bond_length,
            box_size=box_size,
        )

    def _generate_initial_positions(self) -> None:
        """Generate initial chain positions and bonds."""
        n_beads = self.n_beads
        box_size = self._calculate_box_size()

        # Generate positions for each chain
        chain_positions = []

        for chain in range(self.n_chains):
            chain_pos = []
            if self.topology == "ring" and not self._architecture.is_branched:
                radius = self.bond_length * n_beads / (2 * np.pi)
                for bead in range(n_beads):
                    angle = 2 * np.pi * bead / n_beads
                    pos = np.array([
                        radius * np.cos(angle),
                        radius * np.sin(angle),
                        0.0
                    ])
                    chain_pos.append(pos)
            elif self._architecture.is_branched or self._architecture.is_cyclic:
                # Generic graph: grow along the spanning tree with random
                # directions (may contain overlaps; SAW/MC can relax it)
                order, parents, _ = self._architecture.growth_order(root=0)
                pos_map = {0: np.zeros(3)}
                for bead in order[1:]:
                    parent = parents[bead]
                    direction = np.random.normal(size=3)
                    direction /= np.linalg.norm(direction)
                    pos_map[bead] = pos_map[parent] + direction * self.bond_length
                chain_pos = [pos_map[i] for i in range(n_beads)]
            else:
                # Linear chain
                for bead in range(n_beads):
                    pos = np.array([
                        bead * self.bond_length,
                        0.0,
                        0.0
                    ])
                    chain_pos.append(pos)

            chain_positions.append(chain_pos)

        # Place chains in box
        if self.n_chains > 1 and (self.density is not None or self._explicit_box_size is not None):
            # Use random placement with separation check
            min_sep = max(2.0 * self.bead_types[0].sigma, self.bond_length * 2)
            self._positions, self._chain_indices = place_chains_in_box(
                chain_positions, box_size, min_separation=min_sep
            )
        else:
            # Simple grid/linear placement
            self._positions = []
            self._chain_indices = []

            for chain_idx, chain_pos in enumerate(chain_positions):
                start_idx = len(self._positions)

                if self.topology == "ring":
                    n_per_dim = int(np.ceil(np.cbrt(self.n_chains)))
                    radius = self.bond_length * n_beads / (2 * np.pi)
                    spacing = radius * 3

                    ix = chain_idx % n_per_dim
                    iy = (chain_idx // n_per_dim) % n_per_dim
                    iz = chain_idx // (n_per_dim * n_per_dim)

                    center = np.array([
                        (ix - n_per_dim / 2 + 0.5) * spacing,
                        (iy - n_per_dim / 2 + 0.5) * spacing,
                        (iz - n_per_dim / 2 + 0.5) * spacing
                    ])

                    R = _random_rotation_matrix(np.pi)

                    for pos in chain_pos:
                        rotated = R @ pos
                        self._positions.append(rotated + center)
                else:
                    # Simple offset for linear chains
                    offset = np.array([0.0, chain_idx * self.bond_length * 2, 0.0])
                    for pos in chain_pos:
                        self._positions.append(pos + offset)

                end_idx = len(self._positions)
                self._chain_indices.append((start_idx, end_idx))

        # Generate bonds from the chain graph (0-indexed for internal use)
        self._bonds = []
        for start, end in self._chain_indices:
            for i, j in self._bonds_local:
                self._bonds.append((start + i, start + j))

    def equilibrate(self, mc_config: Optional[MCConfig] = None) -> None:
        """
        Pre-equilibrate the polymer configuration using Monte Carlo moves.

        Updates internal positions in-place.

        Args:
            mc_config: Monte Carlo configuration. Uses instance config or defaults.
        """
        config = mc_config or self._mc_config or MCConfig()

        # Initialize positions if not done
        if self._positions is None:
            self._generate_initial_positions()

        box_size = self._calculate_box_size()

        # Get average LJ parameters
        avg_sigma = np.mean([bt.sigma for bt in self.bead_types])
        avg_epsilon = np.mean([bt.epsilon for bt in self.bead_types])

        logger.info(
            f"Starting MC equilibration: {config.n_steps} steps, "
            f"T={config.temperature}, box_size={box_size:.3f}"
        )

        # Per-chain graph metadata enables branched MC moves (tree-pivot,
        # segment crankshaft); linear chains use the legacy moves.
        chain_graphs = [
            build_chain_graph(self._architecture, start)
            for start, _ in self._chain_indices
        ]

        self._positions, acceptance_stats = mc_equilibrate(
            positions=self._positions,
            bonds=self._bonds,
            chain_indices=self._chain_indices,
            chain_graphs=chain_graphs,
            n_steps=config.n_steps,
            temperature=config.temperature,
            move_weights=config.move_weights,
            box_size=box_size,
            lj_sigma=config.lj_sigma if config.lj_sigma else avg_sigma,
            lj_epsilon=config.lj_epsilon if config.lj_epsilon else avg_epsilon,
            lj_cutoff=config.lj_cutoff,
            bond_k=config.bond_k,
            bond_r0=self.bond_length,
            max_displacement=config.max_displacement,
            max_angle=config.max_angle,
            verbose=True,
        )

        logger.info(f"MC equilibration complete. Acceptance rates: {acceptance_stats}")

    def saw_generate(self, saw_config: Optional[SAWConfig] = None) -> bool:
        """
        Generate configuration using Self-Avoiding Random Walk.

        This is a fast alternative to MC equilibration that generates
        overlap-free configurations directly.

        Args:
            saw_config: SAW configuration. Uses instance config or defaults.

        Returns:
            True if successful, False if SAW failed.
        """
        config = saw_config or self._saw_config or SAWConfig()

        # Set collision sigma based on bead type if not specified
        if config.collision_sigma == 1.0 and self.bead_types:
            config.collision_sigma = max(bt.sigma for bt in self.bead_types)

        box_size = self._calculate_box_size()

        logger.info(
            f"Starting SAW generation: {self.n_chains} chains, "
            f"{self.n_beads} beads each, box_size={box_size:.3f}"
        )

        # SAW success is stochastic per attempt (a single crowded chain
        # fails the whole system), so retry the whole system a few times
        # before giving up.
        stats = {"success": False}
        positions, chain_indices = None, None
        for attempt in range(max(1, config.system_retries)):
            if self.topology in self.VALID_TOPOLOGIES:
                # Legacy path for linear/ring (identical connectivity)
                positions, chain_indices, stats = saw_generate_multi_chain(
                    n_chains=self.n_chains,
                    n_beads_per_chain=self.n_beads,
                    bond_length=self.bond_length,
                    box_size=box_size,
                    config=config,
                    topology=self.topology,
                )
            else:
                # Graph path for arbitrary architectures (star, comb, ...)
                positions, chain_indices, stats = saw_generate_graphs(
                    architectures=[self._architecture] * self.n_chains,
                    bond_length=self.bond_length,
                    box_size=box_size,
                    config=config,
                )
            if stats["success"]:
                break
            logger.info(
                f"SAW attempt {attempt + 1}/{config.system_retries} failed "
                f"({stats['failure_reason']}), retrying with fresh state"
            )

        if not stats["success"]:
            logger.warning(
                f"SAW generation failed: {stats['failure_reason']}. "
                f"Chains completed: {stats['chains_completed']}, "
                f"Backtracks: {stats['total_backtracks']}"
            )
            return False

        self._positions = positions
        self._chain_indices = chain_indices

        # Generate bonds from the chain graph (0-indexed)
        self._bonds = []
        for start, end in self._chain_indices:
            for i, j in self._bonds_local:
                self._bonds.append((start + i, start + j))

        logger.info(
            f"SAW generation complete. Backtracks: {stats['total_backtracks']}"
        )
        return True

    def generate(
        self,
        backend: Optional[str] = None,
        run_moltemplate: bool = True,
    ) -> None:
        """
        Generate LAMMPS input files using the configured output backend.

        This is the standard entry point for producing simulation files.

        Args:
            backend: "moltemplate" (default; .lt files + moltemplate ->
                moltemplate/system.data + system.in.*) or "direct"
                (polymer.data + in.polymer written directly). Defaults to
                the constructor's ``backend`` parameter.
            run_moltemplate: Only for the moltemplate backend: run the
                bundled moltemplate after writing .lt files.
        """
        backend = backend or self._backend
        if backend not in self.VALID_BACKENDS:
            raise ValueError(f"Backend must be one of: {self.VALID_BACKENDS}")
        if backend == "moltemplate":
            self.generate_moltemplate(run_moltemplate=run_moltemplate)
        else:
            self.generate_data_file()

    def generate_data_file(self) -> None:
        """Generate LAMMPS data file for bead-spring polymer (direct backend)."""
        n_beads = self.n_beads
        # Bond/angle counts come from the chain graph edge/triplet lists
        # (a tree has N-1 bonds, a ring N, etc.)
        n_bonds_per_chain = len(self._bonds_local)
        n_angles_per_chain = 0
        if self.use_angles:
            n_angles_per_chain = len(self._angle_triplets_local)

        total_atoms = self.n_chains * n_beads
        total_bonds = self.n_chains * n_bonds_per_chain
        total_angles = self.n_chains * n_angles_per_chain if self.use_angles else 0

        n_atom_types = len(self.bead_types)
        n_angle_types = len(self._angle_type_map) if self.use_angles else 0

        # Calculate box size using the new method
        box_size = self._calculate_box_size()

        # Generate initial positions based on generation method
        if self._positions is None:
            if self._generation_method == "saw":
                success = self.saw_generate(self._saw_config)
                if not success:
                    logger.warning("SAW failed, falling back to geometric placement")
                    self._generate_initial_positions()
            else:
                self._generate_initial_positions()

        # Run MC equilibration if requested (for "mc" method or legacy equilibrate flag)
        if self._generation_method == "mc" or self._equilibrate:
            self.equilibrate(self._mc_config)

        # Positions are always available at this point (both SAW and
        # geometric placement populate self._positions)
        if self._positions is None:
            self._generate_initial_positions()

        with open(f"{self.path}/polymer.data", 'w') as f:
            # Header
            f.write("LAMMPS Bead-Spring Polymer Data File\n\n")
            f.write(f"{total_atoms} atoms\n")
            f.write(f"{total_bonds} bonds\n")
            if self.use_angles:
                f.write(f"{total_angles} angles\n")
            f.write("\n")
            f.write(f"{n_atom_types} atom types\n")
            f.write("1 bond types\n")
            if self.use_angles:
                f.write(f"{n_angle_types} angle types\n")
            f.write("\n")

            # Box dimensions
            f.write(f"{-box_size/2:.1f} {box_size/2:.1f} xlo xhi\n")
            f.write(f"{-box_size/2:.1f} {box_size/2:.1f} ylo yhi\n")
            f.write(f"{-box_size/2:.1f} {box_size/2:.1f} zlo zhi\n\n")

            # Masses
            f.write("Masses\n\n")
            for bt in self.bead_types:
                type_id = self._bead_type_id[bt.name]
                f.write(f"{type_id} {bt.mass:.3f}  # {bt.name}\n")
            f.write("\n")

            # Atoms section: atom-ID molecule-ID atom-type x y z
            f.write("Atoms  # molecular\n\n")
            atom_id = 1

            for chain_idx, (start, end) in enumerate(self._chain_indices):
                for local_bead, global_idx in enumerate(range(start, end)):
                    pos = self._positions[global_idx]
                    bead_name = self._sequence[local_bead]
                    type_id = self._bead_type_id[bead_name]
                    f.write(f"{atom_id} {chain_idx + 1} {type_id} {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}\n")
                    atom_id += 1

            # Bonds (from the chain graph edge list)
            f.write("\nBonds\n\n")
            bond_id = 1
            for chain in range(self.n_chains):
                start_id = chain * n_beads + 1
                for i_local, j_local in self._bonds_local:
                    f.write(f"{bond_id} 1 {start_id + i_local} {start_id + j_local}\n")
                    bond_id += 1

            # Angles (from the chain graph triplet list)
            if self.use_angles:
                f.write("\nAngles\n\n")
                angle_id = 1
                for chain in range(self.n_chains):
                    start_id = chain * n_beads + 1

                    for i_local, j_local, k_local in self._angle_triplets_local:
                        triplet = self._get_canonical_triplet(
                            self._sequence[i_local],
                            self._sequence[j_local],
                            self._sequence[k_local]
                        )
                        angle_type_id = self._angle_type_map[triplet]
                        f.write(f"{angle_id} {angle_type_id} {start_id + i_local} {start_id + j_local} {start_id + k_local}\n")
                        angle_id += 1

        # Generate LAMMPS input script
        self._generate_input_script()
        logger.info(f"Generated bead-spring polymer files in {self.path}")

    def generate_moltemplate(self, run_moltemplate: bool = True) -> Path:
        """
        Generate moltemplate .lt files (and optionally run moltemplate) for
        this bead-spring polymer.

        Mirrors the atomistic pipeline layout under
        ``<path>/moltemplate/``: bead_spring.lt (CG force field),
        bead_<Type>.lt monomer objects, chains.lt (one object per chain
        with generated coordinates, explicit bond list, typed angle list),
        and system.lt. Works for any architecture, since bonds and angles
        come from the chain graph.

        Args:
            run_moltemplate: Run the bundled moltemplate after writing
                files (produces system.data, system.in.init/settings).

        Returns:
            Path to the moltemplate directory.
        """
        from .bead_spring_lt import generate_moltemplate_files

        # Ensure positions exist (same logic as generate_data_file)
        if self._positions is None:
            if self._generation_method == "saw":
                success = self.saw_generate(self._saw_config)
                if not success:
                    logger.warning("SAW failed, falling back to geometric placement")
                    self._generate_initial_positions()
            else:
                self._generate_initial_positions()

        if self._generation_method == "mc" or self._equilibrate:
            self.equilibrate(self._mc_config)

        box_size = self._calculate_box_size()

        return generate_moltemplate_files(
            path=self.path,
            bead_types=self.bead_types,
            chain_architectures=[self._architecture] * self.n_chains,
            positions=self._positions,
            chain_indices=self._chain_indices,
            box_size=box_size,
            pair_coeffs=self._pair_coeffs,
            bond_style=self.bond_style,
            k_bond=self.k_bond,
            bond_length=self.bond_length,
            fene_r0=self.fene_r0,
            pair_style=self._pair_style,
            use_angles=self.use_angles,
            angle_type_map=self._angle_type_map if self.use_angles else {},
            resolve_angle_params=self._get_angle_params,
            include_branch_angles=self.include_branch_angles,
            canonical_triplet=canonical_bead_triplet,
            run_moltemplate=run_moltemplate,
        )

    def _generate_input_script(self) -> None:
        """Generate LAMMPS input script for the bead-spring polymer.

        Note: For Kremer-Grest polymers using FENE bonds, the WCA cutoff
        (2^(1/6) * sigma ≈ 1.12246) should be used instead of full LJ cutoff.
        This is controlled by setting pair_style='wca' in the class constructor.
        The equilibrium bond length of ~0.97 emerges from the balance of
        FENE + WCA potentials - there is no explicit r0 parameter in FENE.
        """
        write_lammps_input_script(
            path=self.path,
            bead_types=self.bead_types,
            pair_coeffs=self._pair_coeffs,
            bond_style=self.bond_style,
            k_bond=self.k_bond,
            bond_length=self.bond_length,
            fene_r0=self.fene_r0,
            pair_style=self._pair_style,
            use_angles=self.use_angles,
            angle_type_map=self._angle_type_map,
            angle_types=self._angle_types,
            default_k_angle=self.default_k_angle,
            default_theta0=self.default_theta0,
        )

    def get_system_info(self) -> dict:
        """
        Get comprehensive information about the bead-spring polymer system.

        Returns:
            Dictionary containing system properties.
        """
        n_beads = self.n_beads
        n_bonds_per_chain = len(self._bonds_local)
        n_angles_per_chain = 0
        if self.use_angles:
            n_angles_per_chain = len(self._angle_triplets_local)

        total_atoms = self.n_chains * n_beads
        total_bonds = self.n_chains * n_bonds_per_chain
        total_angles = self.n_chains * n_angles_per_chain if self.use_angles else 0

        return {
            'name': self.name,
            'n_chains': self.n_chains,
            'n_beads_per_chain': n_beads,
            'topology': self.topology,
            'architecture': self._architecture.name,
            'is_branched': self._architecture.is_branched,
            'total_atoms': total_atoms,
            'total_bonds': total_bonds,
            'total_angles': total_angles,
            'bond_style': self.bond_style,
            'bond_length': self.bond_length,
            'k_bond': self.k_bond,
            'use_angles': self.use_angles,
            'bead_types': [bt.name for bt in self.bead_types],
            'sequence': self._sequence,
            'output_path': self.path,
        }

    @classmethod
    def kremer_grest(
        cls,
        name: str,
        system: object,
        n_chains: int,
        n_beads: int,
        topology: str = "linear",
        density: float = 0.74,
        generation_method: str = "saw",
    ) -> "BeadSpringPolymer":
        """
        Create a Kremer-Grest bead-spring polymer with standard parameters.

        The Kremer-Grest model is a standard coarse-grained polymer model used
        for universal polymer behavior studies. It uses:
        - FENE bonds (K=30, R0=1.5)
        - WCA purely repulsive interactions (cutoff = 2^(1/6)*sigma ≈ 1.12246)
        - Standard LJ parameters (epsilon=1.0, sigma=1.0, mass=1.0)

        The equilibrium bond length of ~0.97 EMERGES from the balance of
        FENE (attractive) + WCA (repulsive) - there is no explicit r0 parameter.

        Args:
            name: Name for output files.
            system: System object containing path information.
            n_chains: Number of polymer chains.
            n_beads: Number of beads per chain.
            topology: "linear" or "ring" topology.
            density: Target bead density (beads/sigma^3). Default 0.74 for
                initial placement (will compress to ~0.85-0.90 during NPT).
            generation_method: Method for generating initial configurations
                ("geometric", "saw", or "mc").

        Returns:
            BeadSpringPolymer instance configured for Kremer-Grest model.

        Example:
            >>> from AutoPoly import System
            >>> from AutoPoly import BeadSpringPolymer
            >>> system = System(out="kg_simulation")
            >>> kg_polymer = BeadSpringPolymer.kremer_grest(
            ...     name="kg_melt",
            ...     system=system,
            ...     n_chains=100,
            ...     n_beads=10,
            ...     topology="linear"
            ... )
            >>> kg_polymer.generate_data_file()

        References:
            Kremer, K., & Grest, G. S. (1990). Dynamics of entangled linear
            polymer melts: A molecular-dynamics simulation. J. Chem. Phys.
            92(8), 5057-5086.
        """
        # Standard Kremer-Grest bead type
        kg_bead_type = BeadType(name="KG", mass=1.0, epsilon=1.0, sigma=1.0)

        # Uniform sequence of KG beads
        sequence = [("KG", n_beads)]

        return cls(
            name=name,
            system=system,
            n_chains=n_chains,
            bead_types=[kg_bead_type],
            sequence=sequence,
            topology=topology,
            bond_length=0.97,  # Starting guess, equilibrium emerges from FENE+WCA
            bond_style="fene",
            k_bond=30.0,
            fene_r0=1.5,
            pair_style="wca",  # WCA purely repulsive for KG melts
            density=density,
            generation_method=generation_method,
        )
