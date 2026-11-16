#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collision Detection Module for Monte Carlo Placement

This module provides efficient 3D collision detection using monomer-level spheres
and cell-linked list data structure for O(N) expected complexity.

Key Features:
- Hard-sphere collision model at monomer level
- Cell-linked list for spatial hashing
- Efficient add/remove operations
- Configurable tolerance for overlap detection

Created on 2026-01-29
@author: zwu
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict


@dataclass
class MonomerSphere:
    """
    Represents a monomer as a single sphere for collision detection.

    Attributes:
        center: 3D coordinates of the sphere center (monomer geometric center)
        radius: Collision radius of the monomer
        monomer_id: Unique identifier for this monomer
    """
    center: np.ndarray
    radius: float
    monomer_id: int

    def __post_init__(self):
        """Ensure center is a numpy array."""
        if not isinstance(self.center, np.ndarray):
            self.center = np.array(self.center, dtype=np.float64)


class CollisionDetector:
    """
    Efficient collision detection using cell-linked list spatial hashing.

    This class implements a 3D grid-based spatial hashing scheme for fast
    collision queries. Each cell in the grid contains references to spheres
    whose centers fall within that cell.

    The algorithm achieves O(N) expected time for both insertion and collision
    queries when the cell size is properly tuned to the sphere sizes.

    Attributes:
        box_bounds: Tuple of ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        cell_size: Size of each cell in the spatial grid
        cells: Dictionary mapping cell indices to sets of monomer IDs
        spheres: Dictionary mapping monomer IDs to MonomerSphere objects

    Example:
        >>> detector = CollisionDetector(
        ...     box_bounds=((-50, 50), (-50, 50), (-50, 50)),
        ...     cell_size=5.0
        ... )
        >>> detector.add_monomer(0, np.array([0, 0, 0]), 2.0)
        >>> detector.check_collision(np.array([1, 0, 0]), 2.0)
        True
    """

    def __init__(
        self,
        box_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        cell_size: float = 5.0
    ):
        """
        Initialize the collision detector.

        Args:
            box_bounds: Simulation box boundaries as ((xmin, xmax), (ymin, ymax), (zmin, zmax))
            cell_size: Size of each cell in the spatial grid. Should be at least
                      as large as the largest expected sphere diameter for optimal
                      performance.
        """
        self.box_bounds = box_bounds
        self.cell_size = cell_size

        # Calculate grid dimensions
        self._xmin, self._xmax = box_bounds[0]
        self._ymin, self._ymax = box_bounds[1]
        self._zmin, self._zmax = box_bounds[2]

        self._nx = max(1, int(np.ceil((self._xmax - self._xmin) / cell_size)))
        self._ny = max(1, int(np.ceil((self._ymax - self._ymin) / cell_size)))
        self._nz = max(1, int(np.ceil((self._zmax - self._zmin) / cell_size)))

        # Cell storage: maps cell index tuple to set of monomer IDs
        self.cells: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)

        # Sphere storage: maps monomer ID to MonomerSphere
        self.spheres: Dict[int, MonomerSphere] = {}

    def _get_cell_index(self, position: np.ndarray) -> Tuple[int, int, int]:
        """
        Convert a 3D position to cell indices.

        Args:
            position: 3D coordinates

        Returns:
            Tuple of (ix, iy, iz) cell indices
        """
        ix = int((position[0] - self._xmin) / self.cell_size)
        iy = int((position[1] - self._ymin) / self.cell_size)
        iz = int((position[2] - self._zmin) / self.cell_size)

        # Clamp to valid range
        ix = max(0, min(ix, self._nx - 1))
        iy = max(0, min(iy, self._ny - 1))
        iz = max(0, min(iz, self._nz - 1))

        return (ix, iy, iz)

    def _get_neighbor_cells(
        self,
        cell_index: Tuple[int, int, int],
        search_radius: int = 1
    ) -> List[Tuple[int, int, int]]:
        """
        Get indices of neighboring cells within search radius.

        Args:
            cell_index: Center cell index
            search_radius: Number of cells to search in each direction

        Returns:
            List of valid neighboring cell indices
        """
        ix, iy, iz = cell_index
        neighbors = []

        for di in range(-search_radius, search_radius + 1):
            for dj in range(-search_radius, search_radius + 1):
                for dk in range(-search_radius, search_radius + 1):
                    ni = ix + di
                    nj = iy + dj
                    nk = iz + dk

                    if 0 <= ni < self._nx and 0 <= nj < self._ny and 0 <= nk < self._nz:
                        neighbors.append((ni, nj, nk))

        return neighbors

    @staticmethod
    def estimate_radius(
        left_conn: np.ndarray,
        right_conn: np.ndarray,
        buffer: float = 1.5
    ) -> float:
        """
        Estimate collision radius for a monomer from connection atom positions.

        The radius is estimated as half the distance between connection atoms
        plus a buffer to account for atoms extending beyond the backbone.

        Args:
            left_conn: Coordinates of left connection atom
            right_conn: Coordinates of right connection atom
            buffer: Additional padding in Angstroms (default: 1.5)

        Returns:
            Estimated collision radius in Angstroms
        """
        conn_distance = np.linalg.norm(right_conn - left_conn)
        return conn_distance / 2 + buffer

    @staticmethod
    def estimate_radius_from_coords(
        atom_coords: np.ndarray,
        buffer: float = 0.5
    ) -> float:
        """
        Estimate collision radius from atom coordinates using bounding sphere.

        Computes the center of mass and finds the maximum distance from center
        to any atom, then adds a buffer.

        Args:
            atom_coords: Array of shape (N, 3) containing atom coordinates
            buffer: Additional padding in Angstroms (default: 0.5)

        Returns:
            Estimated collision radius in Angstroms
        """
        if len(atom_coords) == 0:
            return buffer

        center = np.mean(atom_coords, axis=0)
        distances = np.linalg.norm(atom_coords - center, axis=1)
        max_distance = np.max(distances) if len(distances) > 0 else 0

        return max_distance + buffer

    def add_monomer(
        self,
        mon_id: int,
        center: np.ndarray,
        radius: float
    ) -> None:
        """
        Add a monomer sphere to the collision detector.

        Args:
            mon_id: Unique identifier for the monomer
            center: 3D coordinates of the sphere center
            radius: Collision radius of the sphere
        """
        sphere = MonomerSphere(center=center.copy(), radius=radius, monomer_id=mon_id)
        self.spheres[mon_id] = sphere

        # Add to cell
        cell_idx = self._get_cell_index(center)
        self.cells[cell_idx].add(mon_id)

    def remove_monomer(self, mon_id: int) -> None:
        """
        Remove a monomer sphere from the collision detector.

        Args:
            mon_id: Unique identifier of the monomer to remove
        """
        if mon_id not in self.spheres:
            return

        sphere = self.spheres[mon_id]
        cell_idx = self._get_cell_index(sphere.center)

        # Remove from cell
        if cell_idx in self.cells:
            self.cells[cell_idx].discard(mon_id)
            if not self.cells[cell_idx]:
                del self.cells[cell_idx]

        # Remove from spheres
        del self.spheres[mon_id]

    def check_collision(
        self,
        center: np.ndarray,
        radius: float,
        exclude_ids: Optional[Set[int]] = None,
        tolerance: float = 0.1
    ) -> bool:
        """
        Check if a sphere at the given position would collide with existing spheres.

        Args:
            center: 3D coordinates of the test sphere center
            radius: Radius of the test sphere
            exclude_ids: Set of monomer IDs to exclude from collision check
            tolerance: Overlap tolerance in Angstroms. Negative values allow slight
                      penetration before reporting collision. (default: 0.1)

        Returns:
            True if collision detected, False otherwise
        """
        if exclude_ids is None:
            exclude_ids = set()

        # Calculate search radius in cells
        max_search_distance = radius + max(
            s.radius for s in self.spheres.values()
        ) if self.spheres else radius
        search_cells = max(1, int(np.ceil(max_search_distance / self.cell_size)))

        # Get cell and neighbors
        cell_idx = self._get_cell_index(center)
        neighbor_cells = self._get_neighbor_cells(cell_idx, search_cells)

        # Check all spheres in neighboring cells
        for ncell in neighbor_cells:
            if ncell not in self.cells:
                continue

            for mon_id in self.cells[ncell]:
                if mon_id in exclude_ids:
                    continue

                sphere = self.spheres[mon_id]
                distance = np.linalg.norm(center - sphere.center)
                min_distance = radius + sphere.radius - tolerance

                if distance < min_distance:
                    return True

        return False

    def check_collision_detailed(
        self,
        center: np.ndarray,
        radius: float,
        exclude_ids: Optional[Set[int]] = None,
        tolerance: float = 0.1
    ) -> Tuple[bool, Optional[int], Optional[float]]:
        """
        Check collision with detailed information about the colliding sphere.

        Args:
            center: 3D coordinates of the test sphere center
            radius: Radius of the test sphere
            exclude_ids: Set of monomer IDs to exclude from collision check
            tolerance: Overlap tolerance in Angstroms

        Returns:
            Tuple of (collision_detected, colliding_monomer_id, overlap_distance)
        """
        if exclude_ids is None:
            exclude_ids = set()

        max_search_distance = radius + max(
            s.radius for s in self.spheres.values()
        ) if self.spheres else radius
        search_cells = max(1, int(np.ceil(max_search_distance / self.cell_size)))

        cell_idx = self._get_cell_index(center)
        neighbor_cells = self._get_neighbor_cells(cell_idx, search_cells)

        for ncell in neighbor_cells:
            if ncell not in self.cells:
                continue

            for mon_id in self.cells[ncell]:
                if mon_id in exclude_ids:
                    continue

                sphere = self.spheres[mon_id]
                distance = np.linalg.norm(center - sphere.center)
                min_distance = radius + sphere.radius - tolerance

                if distance < min_distance:
                    overlap = min_distance - distance
                    return True, mon_id, overlap

        return False, None, None

    def check_bounds(self, center: np.ndarray, radius: float) -> bool:
        """
        Check if a sphere is within the simulation box bounds.

        Args:
            center: 3D coordinates of the sphere center
            radius: Radius of the sphere

        Returns:
            True if the sphere is completely within bounds, False otherwise
        """
        return bool(
            center[0] - radius >= self._xmin and center[0] + radius <= self._xmax and
            center[1] - radius >= self._ymin and center[1] + radius <= self._ymax and
            center[2] - radius >= self._zmin and center[2] + radius <= self._zmax
        )

    def get_sphere_count(self) -> int:
        """Return the total number of spheres in the detector."""
        return len(self.spheres)

    def clear(self) -> None:
        """Remove all spheres from the detector."""
        self.cells.clear()
        self.spheres.clear()


class ObstacleCollisionDetector(CollisionDetector):
    """
    CollisionDetector that also rejects positions inside fixed spherical
    obstacles (e.g. coarse-grained fillers), with periodic checks.

    Obstacles are kept out of the cell list (their radius would widen
    every neighbor search) and checked directly.  With ``period`` set, all
    checks -- obstacles and the spheres added during growth -- use the
    minimum-image convention of the cubic periodic box
    [-period/2, period/2)^3, so chains that leave the box cannot overlap
    the periodic images of fillers or of other chains.

    Args:
        box_bounds: As for CollisionDetector.
        cell_size: As for CollisionDetector.
        centers: (k, 3) obstacle centers.
        radii: (k,) exclusion radii: a test sphere of radius r collides
            when its center is closer than radii[i] + r - tolerance.
        period: Periodic box side (None: open box, obstacles only).
    """

    def __init__(self, box_bounds: Tuple[Tuple[float, float], ...],
                 cell_size: float, centers: np.ndarray, radii: np.ndarray,
                 period: Optional[float] = None):
        super().__init__(box_bounds, cell_size)
        self.obstacle_centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        self.obstacle_radii = np.asarray(radii, dtype=np.float64).reshape(-1)
        self.period = period
        if period:
            # whole cells tile the period exactly
            self._nx = self._ny = self._nz = max(1, int(period // cell_size))
            self._pcell = period / self._nx

    def _min_image(self, d: np.ndarray) -> np.ndarray:
        if self.period:
            d = d - self.period * np.round(d / self.period)
        return d

    def _get_cell_index(self, position: np.ndarray) -> Tuple[int, int, int]:
        if not self.period:
            return super()._get_cell_index(position)
        w = np.mod(np.asarray(position, dtype=np.float64) + self.period / 2,
                   self.period)
        idx = np.minimum((w // self._pcell).astype(int), self._nx - 1)
        return (int(idx[0]), int(idx[1]), int(idx[2]))

    def hits_obstacle(self, center: np.ndarray, radius: float,
                      tolerance: float = 0.1) -> bool:
        """True if a sphere at ``center`` overlaps any obstacle."""
        if len(self.obstacle_radii) == 0:
            return False
        d = self._min_image(self.obstacle_centers - np.asarray(center, dtype=np.float64))
        r2 = np.einsum("ij,ij->i", d, d)
        lim = self.obstacle_radii + radius - tolerance
        return bool(np.any(r2 < lim * lim))

    def check_collision(self, center, radius, exclude_ids=None,
                        tolerance: float = 0.1) -> bool:
        if self.hits_obstacle(center, radius, tolerance):
            return True
        if not self.period:
            return super().check_collision(center, radius, exclude_ids, tolerance)
        if not self.spheres:
            return False
        exclude_ids = exclude_ids or set()
        reach = radius + max(s.radius for s in self.spheres.values())
        n = min(self._nx // 2, max(1, int(np.ceil(reach / self._pcell))))
        ix, iy, iz = self._get_cell_index(center)
        cells = {((ix + i) % self._nx, (iy + j) % self._ny, (iz + k) % self._nz)
                 for i in range(-n, n + 1) for j in range(-n, n + 1)
                 for k in range(-n, n + 1)}
        for cell in cells:
            for mon_id in self.cells.get(cell, ()):
                if mon_id in exclude_ids:
                    continue
                sphere = self.spheres[mon_id]
                d = self._min_image(center - sphere.center)
                if np.linalg.norm(d) < radius + sphere.radius - tolerance:
                    return True
        return False


def calculate_box_size(
    total_monomers: int,
    monomer_density: float = 0.1
) -> float:
    """
    Calculate cubic box size from monomer count and target density.

    A low default density (0.1 monomers/A^3) ensures sufficient space
    between monomers for successful MC placement.

    Args:
        total_monomers: Total number of monomers in the system
        monomer_density: Target number density (monomers/A^3), default 0.1

    Returns:
        Box side length (Angstroms) for a cubic box

    Example:
        >>> calculate_box_size(100, monomer_density=0.1)
        21.544...  # 100 monomers at density 0.1 -> volume = 1000 A^3 -> box ~ 10 A
    """
    if monomer_density <= 0:
        raise ValueError("monomer_density must be positive")
    if total_monomers <= 0:
        raise ValueError("total_monomers must be positive")

    volume = total_monomers / monomer_density
    box_size = volume ** (1/3)
    return box_size
