"""
Bead-Spring System (Mixtures)

This module provides BeadSpringSystem: a multi-species bead-spring melt
builder. Each species is a (BeadArchitecture, n_chains) pair — linear
chains, rings, stars, combs, or any custom graph can be mixed freely in
one simulation box with one shared LAMMPS data file.

Example — a ring/linear blend::

    from AutoPoly import System
    from AutoPoly.models.bead_spring_system import BeadSpringSystem
    from AutoPoly.models import architectures as arch
    from AutoPoly.models.bead_spring import BeadType

    system = System(out="ring_linear_blend")
    bss = BeadSpringSystem(
        name="blend",
        system=system,
        bead_types=[BeadType("A")],
        bond_style="fene",
        pair_style="wca",
        density=0.85,
    )
    bss.add_species(arch.ring([("A", 50)]), n_chains=20)
    bss.add_species(arch.linear([("A", 50)]), n_chains=20)
    bss.generate_data_file()

Coarse-grained fillers (AutoPoly.nanoparticles.CGFiller) are added with
:meth:`BeadSpringSystem.add_filler`: they are placed first, chains grow
around them by SAW, and the run script integrates them as rigid bodies
(or keeps them frozen).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.logger import setup_logger
from .architectures import BeadArchitecture
from .bead_spring import (
    AngleType,
    BeadType,
    MCConfig,
    SAWConfig,
    build_angle_type_map,
    build_chain_graph,
    calculate_box_size,
    lb_pair_coeffs,
    mc_equilibrate,
    saw_generate_graphs,
    write_lammps_input_script,
    DEFAULT_BEAD_DENSITY,
)

logger = setup_logger()


class BeadSpringSystem:
    """
    Multi-species bead-spring system (mixture) generator.

    All species share one bead-type table, one box, and one LAMMPS data
    file. Chains are placed collision-free with the graph-based SAW
    generator; branched species are equilibrated with the branched MC
    moves (tree-pivot, segment crankshaft).

    Args:
        name: Name for output files.
        system: System object containing path information.
        bead_types: List of BeadType objects covering ALL bead types used
            by any species.
        bond_length: Equilibrium bond length.
        bond_style: "harmonic" or "fene".
        k_bond: Bond force constant.
        fene_r0: FENE maximum extension (only used if bond_style="fene").
        pair_style: "lj" (full LJ, cutoff 2.5) or "wca" (repulsive).
        use_angles: Whether to include angle potentials.
        default_k_angle: Default angle force constant.
        default_theta0: Default equilibrium angle.
        angle_types: Per-triplet AngleType parameters.
        include_branch_angles: Whether to include angle triplets centered
            on branch points.
        density: Target bead density (beads/sigma^3) for box sizing (with
            CG fillers: chain-bead density of the volume outside the
            filler envelopes).
        box_size: Explicit box size (overrides density).
        generation_method: "geometric", "saw" (default), or "mc".
        saw_config: SAW configuration.
        mc_config: MC equilibration configuration.
        backend: Output backend used by :meth:`generate`: "moltemplate"
            (default) or "direct".

    Raises:
        ValueError: If parameters are invalid.
    """

    VALID_BOND_STYLES = ["harmonic", "fene"]
    VALID_PAIR_STYLES = ["lj", "wca"]
    VALID_GENERATION_METHODS = ["geometric", "saw", "mc"]
    VALID_BACKENDS = ["moltemplate", "direct"]

    def __init__(
        self,
        name: str,
        system: object,
        bead_types: List[BeadType],
        bond_length: float = 1.0,
        bond_style: str = "harmonic",
        k_bond: float = 30.0,
        fene_r0: float = 1.5,
        pair_style: str = "lj",
        use_angles: bool = False,
        default_k_angle: float = 10.0,
        default_theta0: float = 180.0,
        angle_types: Optional[List[AngleType]] = None,
        include_branch_angles: bool = True,
        density: Optional[float] = None,
        box_size: Optional[float] = None,
        generation_method: str = "saw",
        saw_config: Optional[SAWConfig] = None,
        mc_config: Optional[MCConfig] = None,
        backend: str = "moltemplate",
    ) -> None:
        if backend not in self.VALID_BACKENDS:
            raise ValueError(f"Backend must be one of: {self.VALID_BACKENDS}")
        if bond_style not in self.VALID_BOND_STYLES:
            raise ValueError(f"Bond style must be one of: {self.VALID_BOND_STYLES}")
        if pair_style not in self.VALID_PAIR_STYLES:
            raise ValueError(f"Pair style must be one of: {self.VALID_PAIR_STYLES}")
        if generation_method not in self.VALID_GENERATION_METHODS:
            raise ValueError(
                f"Generation method must be one of: {self.VALID_GENERATION_METHODS}"
            )
        if not bead_types:
            raise ValueError("At least one bead type is required")

        self.name = name
        self.system = system
        self.path = f"{self.system.get_folder_path()}/{self.name}" if system else f"./{name}"

        self.bead_types = bead_types
        self._bead_type_map: Dict[str, BeadType] = {bt.name: bt for bt in bead_types}
        self._bead_type_id: Dict[str, int] = {bt.name: i + 1 for i, bt in enumerate(bead_types)}

        # Bond/pair/angle parameters
        self.bond_length = bond_length
        self.bond_style = bond_style
        self.k_bond = k_bond
        self.fene_r0 = fene_r0
        self._pair_style = pair_style
        self.use_angles = use_angles
        self.default_k_angle = default_k_angle
        self.default_theta0 = default_theta0
        self._angle_types = angle_types or []
        self.include_branch_angles = include_branch_angles

        # Box sizing
        self.density = density
        self._explicit_box_size = box_size

        # Generation
        self._generation_method = generation_method
        self._saw_config = saw_config
        self._mc_config = mc_config

        # Output backend ("moltemplate" default, or "direct")
        self._backend = backend

        # Species: list of (architecture, n_chains, species_name)
        self._species: List[Tuple[BeadArchitecture, int, str]] = []

        # Internal state (populated during generation)
        self._positions: Optional[List[np.ndarray]] = None
        self._chain_indices: Optional[List[Tuple[int, int]]] = None
        self._chain_architectures: Optional[List[BeadArchitecture]] = None
        self._bonds: Optional[List[Tuple[int, int]]] = None

        self._pair_coeffs = lb_pair_coeffs(bead_types)

        # Coarse-grained fillers (AutoPoly.nanoparticles.cg)
        self._fillers: List = []
        self._placed_fillers: Optional[List] = None
        self.filler_min_gap = 1.0

        Path(self.path).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Species management
    # ------------------------------------------------------------------ #

    def add_species(
        self,
        architecture: BeadArchitecture,
        n_chains: int,
        name: Optional[str] = None,
    ) -> None:
        """
        Add a chain species to the mixture.

        Args:
            architecture: BeadArchitecture of the chains.
            n_chains: Number of chains of this species.
            name: Optional species label (defaults to the architecture name).

        Raises:
            ValueError: If n_chains < 1 or the architecture is invalid.
        """
        if n_chains < 1:
            raise ValueError(f"n_chains must be >= 1, got {n_chains}")
        architecture.validate(
            known_bead_types=[bt.name for bt in self.bead_types]
        )
        species_name = name or architecture.name
        self._species.append((architecture, n_chains, species_name))
        # Invalidate generated state
        self._placed_fillers = None
        self._positions = None
        self._chain_indices = None
        self._chain_architectures = None
        self._bonds = None
        logger.info(
            f"Added species '{species_name}': {n_chains} chains, "
            f"{architecture.n_beads} beads each"
        )

    def add_filler(self, filler) -> None:
        """
        Add a coarse-grained filler species (AutoPoly.nanoparticles.CGFiller).

        Fillers are placed before the chains (random positions and
        orientations, envelope gaps >= ``filler_min_gap``); chains then
        grow around them by SAW, so only generation_method="saw" is
        supported.  Filler beads are written as one molecule per particle.

        Raises:
            ValueError: Unknown filler bead type.
        """
        from ..nanoparticles.cg import CGFiller
        if not isinstance(filler, CGFiller):
            raise ValueError("add_filler() expects an AutoPoly.nanoparticles.CGFiller")
        if filler.bead_type not in self._bead_type_map:
            raise ValueError(
                f"CG filler bead type '{filler.bead_type}' is not among the "
                f"bead types {list(self._bead_type_map)}"
            )
        self._fillers.append(filler)
        self._placed_fillers = None
        self._positions = None
        self._chain_indices = None
        self._chain_architectures = None
        self._bonds = None
        logger.info(
            f"Added CG filler '{filler.name}': {filler.count} x R="
            f"{filler.radius} ({filler.structure}, {filler.n_beads} beads, "
            f"{filler.motion})"
        )

    def set_pair_coeff(self, type_a: str, type_b: str, epsilon: float,
                       sigma: Optional[float] = None,
                       cutoff: Optional[float] = None) -> None:
        """
        Override the Lorentz-Berthelot pair coefficients of one bead-type
        pair (e.g. the polymer-filler attraction), optionally with its own
        cutoff (e.g. an attractive 2.5 sigma pair in a WCA melt).
        """
        for t in (type_a, type_b):
            if t not in self._bead_type_id:
                raise ValueError(f"Unknown bead type '{t}'")
        i, j = sorted((self._bead_type_id[type_a], self._bead_type_id[type_b]))
        sig = sigma if sigma is not None else self._pair_coeffs[(i, j)][1]
        self._pair_coeffs[(i, j)] = (
            (epsilon, sig) if cutoff is None else (epsilon, sig, cutoff)
        )

    @property
    def n_fillers(self) -> int:
        """Number of CG filler particles."""
        return sum(f.count for f in self._fillers)

    def _check_fillers(self) -> None:
        """Filler bead types are exclusive and have one motion each."""
        from ..nanoparticles.cg import motion_groups
        chain_types = {t for arch, _, _ in self._species for t in arch.bead_types}
        for f in self._fillers:
            if f.bead_type in chain_types:
                raise ValueError(
                    f"CG filler bead type '{f.bead_type}' is also used by a "
                    "chain species; fillers need their own bead type"
                )
        if self._generation_method != "saw":
            raise ValueError(
                "CG fillers need generation_method='saw' (chains are grown "
                "around the placed fillers)"
            )
        motion_groups(self._fillers)

    def _filler_groups(self, numeric: bool) -> Dict[str, List[str]]:
        """motion -> filler type tokens (numeric ids or type names)."""
        from ..nanoparticles.cg import motion_groups
        groups = motion_groups(self._fillers)
        if not numeric:
            return groups
        return {m: [str(self._bead_type_id[t]) for t in ts]
                for m, ts in groups.items()}

    @property
    def n_species(self) -> int:
        return len(self._species)

    @property
    def n_chains(self) -> int:
        return sum(n for _, n, _ in self._species)

    @property
    def total_beads(self) -> int:
        return sum(arch.n_beads * n for arch, n, _ in self._species)

    def _expanded_chain_architectures(self) -> List[BeadArchitecture]:
        """One architecture entry per chain, in species order."""
        chains = []
        for arch, n, _ in self._species:
            chains.extend([arch] * n)
        return chains

    # ------------------------------------------------------------------ #
    # Box sizing
    # ------------------------------------------------------------------ #

    def _calculate_box_size(self) -> float:
        """Box size from explicit value, density, or default density."""
        if self._explicit_box_size is not None:
            return self._explicit_box_size
        density = self.density if self.density is not None else DEFAULT_BEAD_DENSITY
        if self._fillers:
            from ..nanoparticles.cg import filler_volume
            sigma = {bt.name: bt.sigma for bt in self.bead_types}
            volume = self.total_beads / density + filler_volume(self._fillers, sigma)
            return volume ** (1 / 3)
        return calculate_box_size(self.total_beads, density)

    # ------------------------------------------------------------------ #
    # Configuration generation
    # ------------------------------------------------------------------ #

    def _require_species(self) -> None:
        if not self._species:
            raise ValueError("No species added. Call add_species() first.")

    def _generate_geometric_positions(self) -> None:
        """Grow each chain along its spanning tree with random directions."""
        if self._fillers:
            raise ValueError(
                "Geometric placement ignores CG fillers; use generation_method='saw'"
            )
        chains = self._expanded_chain_architectures()
        self._positions = []
        self._chain_indices = []
        self._chain_architectures = chains
        for chain_idx, arch in enumerate(chains):
            start_idx = len(self._positions)
            order, parents, _ = arch.growth_order(root=0)
            offset = np.array([0.0, chain_idx * self.bond_length * 2, 0.0])
            pos_map = {0: offset.copy()}
            for bead in order[1:]:
                parent = parents[bead]
                direction = np.random.normal(size=3)
                direction /= np.linalg.norm(direction)
                pos_map[bead] = pos_map[parent] + direction * self.bond_length
            for i in range(arch.n_beads):
                self._positions.append(pos_map[i])
            self._chain_indices.append((start_idx, len(self._positions)))
        self._generate_bonds()

    def _generate_bonds(self) -> None:
        """Global bond list from per-chain architectures (0-indexed)."""
        self._bonds = []
        assert self._chain_indices is not None
        assert self._chain_architectures is not None
        for (start, _), arch in zip(self._chain_indices, self._chain_architectures):
            for i, j in arch.bonds:
                self._bonds.append((start + i, start + j))

    def saw_generate(self, saw_config: Optional[SAWConfig] = None) -> bool:
        """
        Generate the mixture configuration using graph-based SAW.

        Returns:
            True if successful, False otherwise.
        """
        self._require_species()
        config = saw_config or self._saw_config or SAWConfig()
        if self._fillers:
            self._check_fillers()
        filler_types = {f.bead_type for f in self._fillers}
        if config.collision_sigma == 1.0 and self.bead_types:
            config.collision_sigma = max(
                bt.sigma for bt in self.bead_types if bt.name not in filler_types
            )

        box_size = self._calculate_box_size()
        chains = self._expanded_chain_architectures()
        if self._fillers:
            from ..nanoparticles.cg import place_fillers
            sigma = {bt.name: bt.sigma for bt in self.bead_types}

        logger.info(
            f"Starting SAW generation: {len(chains)} chains "
            f"({self.n_species} species), box_size={box_size:.3f}"
        )

        # Whole-system retries (see BeadSpringPolymer.saw_generate)
        stats = {"success": False}
        positions, chain_indices = None, None
        for attempt in range(max(1, config.system_retries)):
            obstacles = None
            if self._fillers:
                self._placed_fillers = place_fillers(
                    self._fillers, box_size, sigma, self.filler_min_gap
                )
                obstacles = (
                    np.array([p.center for p in self._placed_fillers]),
                    np.array([p.envelope for p in self._placed_fillers]),
                )
            positions, chain_indices, stats = saw_generate_graphs(
                architectures=chains,
                bond_length=self.bond_length,
                box_size=box_size,
                config=config,
                obstacles=obstacles,
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
            if self._fillers:
                # the geometric fallback would ignore the fillers
                raise RuntimeError(
                    "SAW generation around the CG fillers failed "
                    f"({stats['failure_reason']}): lower the density or the "
                    "filler loading, or enlarge the box"
                )
            return False

        self._positions = positions
        self._chain_indices = chain_indices
        self._chain_architectures = chains
        self._generate_bonds()

        logger.info(
            f"SAW generation complete. Backtracks: {stats['total_backtracks']}"
        )
        return True

    def equilibrate(self, mc_config: Optional[MCConfig] = None) -> None:
        """
        Pre-equilibrate the mixture with Monte Carlo moves.

        Branched chains use tree-pivot and segment-crankshaft moves;
        linear chains additionally use reptation.
        """
        self._require_species()
        if self._fillers:
            raise ValueError(
                "MC equilibration does not see CG fillers; use "
                "generation_method='saw'"
            )
        config = mc_config or self._mc_config or MCConfig()

        if self._positions is None:
            if self._generation_method == "saw":
                if not self.saw_generate(self._saw_config):
                    self._generate_geometric_positions()
            else:
                self._generate_geometric_positions()

        box_size = self._calculate_box_size()
        avg_sigma = np.mean([bt.sigma for bt in self.bead_types])
        avg_epsilon = np.mean([bt.epsilon for bt in self.bead_types])

        chain_graphs = [
            build_chain_graph(arch, start)
            for arch, (start, _) in zip(
                self._chain_architectures, self._chain_indices
            )
        ]

        logger.info(
            f"Starting MC equilibration: {config.n_steps} steps, "
            f"T={config.temperature}, box_size={box_size:.3f}"
        )

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

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def _chain_triplets(self) -> List[List[Tuple[int, int, int]]]:
        """Local angle triplets per chain (empty lists if not use_angles)."""
        assert self._chain_architectures is not None
        if not self.use_angles:
            return [[] for _ in self._chain_architectures]
        return [
            arch.angle_triplets(include_branch=self.include_branch_angles)
            for arch in self._chain_architectures
        ]

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
        """Generate the LAMMPS data file (and input script) for the mixture
        (direct backend)."""
        self._require_species()

        # Generate positions
        if self._positions is None:
            if self._generation_method == "saw":
                if not self.saw_generate(self._saw_config):
                    logger.warning("SAW failed, falling back to geometric placement")
                    self._generate_geometric_positions()
            elif self._generation_method == "mc":
                self._generate_geometric_positions()
                self.equilibrate(self._mc_config)
            else:
                self._generate_geometric_positions()

        assert self._chain_architectures is not None
        assert self._chain_indices is not None

        chain_triplets = self._chain_triplets()
        chain_bead_types = [arch.bead_types for arch in self._chain_architectures]

        placed = self._placed_fillers or []
        total_atoms = sum(arch.n_beads for arch in self._chain_architectures)
        total_atoms += sum(len(p.positions) for p in placed)
        total_bonds = sum(arch.n_bonds for arch in self._chain_architectures)
        total_angles = sum(len(t) for t in chain_triplets)

        n_atom_types = len(self.bead_types)
        angle_type_map = (
            build_angle_type_map(chain_bead_types, chain_triplets)
            if self.use_angles else {}
        )
        n_angle_types = len(angle_type_map)

        box_size = self._calculate_box_size()

        with open(f"{self.path}/polymer.data", 'w') as f:
            # Header
            f.write("LAMMPS Bead-Spring Polymer Data File (mixture)\n\n")
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

            # Box dimensions (exact with CG fillers: the filler-aware SAW
            # checks overlaps with the periodic image of this box)
            lo = f"{-box_size/2:.6f}" if placed else f"{-box_size/2:.1f}"
            hi = f"{box_size/2:.6f}" if placed else f"{box_size/2:.1f}"
            f.write(f"{lo} {hi} xlo xhi\n")
            f.write(f"{lo} {hi} ylo yhi\n")
            f.write(f"{lo} {hi} zlo zhi\n\n")

            # Masses
            f.write("Masses\n\n")
            for bt in self.bead_types:
                type_id = self._bead_type_id[bt.name]
                f.write(f"{type_id} {bt.mass:.3f}  # {bt.name}\n")
            f.write("\n")

            # Atoms section: atom-ID molecule-ID atom-type x y z
            f.write("Atoms  # molecular\n\n")
            atom_id = 1
            for chain_idx, ((start, end), arch) in enumerate(
                zip(self._chain_indices, self._chain_architectures)
            ):
                for local_bead, global_idx in enumerate(range(start, end)):
                    pos = self._positions[global_idx]
                    bead_name = arch.bead_types[local_bead]
                    type_id = self._bead_type_id[bead_name]
                    f.write(
                        f"{atom_id} {chain_idx + 1} {type_id} "
                        f"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}\n"
                    )
                    atom_id += 1
            # CG fillers: one molecule per particle, after the chains
            for k, p in enumerate(placed):
                mol_id = len(self._chain_architectures) + k + 1
                type_id = self._bead_type_id[p.filler.bead_type]
                for pos in p.positions:
                    f.write(
                        f"{atom_id} {mol_id} {type_id} "
                        f"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}\n"
                    )
                    atom_id += 1

            # Bonds
            f.write("\nBonds\n\n")
            bond_id = 1
            atom_offset = 0
            for arch in self._chain_architectures:
                for i, j in arch.bonds:
                    f.write(
                        f"{bond_id} 1 {atom_offset + i + 1} {atom_offset + j + 1}\n"
                    )
                    bond_id += 1
                atom_offset += arch.n_beads

            # Angles
            if self.use_angles:
                from .bead_spring import canonical_bead_triplet
                f.write("\nAngles\n\n")
                angle_id = 1
                atom_offset = 0
                for arch, triplets in zip(self._chain_architectures, chain_triplets):
                    for i, j, k in triplets:
                        triplet = canonical_bead_triplet(
                            arch.bead_types[i],
                            arch.bead_types[j],
                            arch.bead_types[k],
                        )
                        angle_type_id = angle_type_map[triplet]
                        f.write(
                            f"{angle_id} {angle_type_id} "
                            f"{atom_offset + i + 1} {atom_offset + j + 1} "
                            f"{atom_offset + k + 1}\n"
                        )
                        angle_id += 1
                    atom_offset += arch.n_beads

        # LAMMPS input script
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
            angle_type_map=angle_type_map,
            angle_types=self._angle_types,
            default_k_angle=self.default_k_angle,
            default_theta0=self.default_theta0,
            fillers=self._filler_groups(numeric=True) if placed else None,
        )
        logger.info(f"Generated bead-spring mixture files in {self.path}")

    def generate_moltemplate(self, run_moltemplate: bool = True) -> Path:
        """
        Generate moltemplate .lt files (and optionally run moltemplate) for
        this mixture.

        Each chain becomes its own moltemplate object with generated
        coordinates; species share the bead-type table, so mixed systems
        (rings + stars + combs + ...) land in one system.data.

        Args:
            run_moltemplate: Run the bundled moltemplate after writing
                files (produces system.data, system.in.init/settings).

        Returns:
            Path to the moltemplate directory.
        """
        from .bead_spring import canonical_bead_triplet, resolve_angle_params
        from .bead_spring_lt import generate_moltemplate_files

        self._require_species()

        # Ensure positions exist (same logic as generate_data_file)
        if self._positions is None:
            if self._generation_method == "saw":
                if not self.saw_generate(self._saw_config):
                    logger.warning("SAW failed, falling back to geometric placement")
                    self._generate_geometric_positions()
            elif self._generation_method == "mc":
                self._generate_geometric_positions()
                self.equilibrate(self._mc_config)
            else:
                self._generate_geometric_positions()

        assert self._chain_architectures is not None
        assert self._chain_indices is not None

        chain_triplets = self._chain_triplets()
        chain_bead_types = [arch.bead_types for arch in self._chain_architectures]
        angle_type_map = (
            build_angle_type_map(chain_bead_types, chain_triplets)
            if self.use_angles else {}
        )

        box_size = self._calculate_box_size()

        return generate_moltemplate_files(
            path=self.path,
            bead_types=self.bead_types,
            chain_architectures=self._chain_architectures,
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
            angle_type_map=angle_type_map,
            resolve_angle_params=lambda t: resolve_angle_params(
                t, self._angle_types, self.default_k_angle, self.default_theta0
            ),
            include_branch_angles=self.include_branch_angles,
            canonical_triplet=canonical_bead_triplet,
            run_moltemplate=run_moltemplate,
            fillers=((self._placed_fillers, self._filler_groups(numeric=False))
                     if self._placed_fillers else None),
        )

    def get_system_info(self) -> dict:
        """
        Get comprehensive information about the bead-spring mixture.

        Returns:
            Dictionary containing system properties.
        """
        chain_triplets = (
            [
                arch.angle_triplets(include_branch=self.include_branch_angles)
                for arch, n, _ in self._species for _ in range(n)
            ]
            if self.use_angles else []
        )
        info = {
            'name': self.name,
            'n_species': self.n_species,
            'n_chains': self.n_chains,
            'species': [
                {
                    'name': sname,
                    'architecture': arch.name,
                    'n_chains': n,
                    'n_beads_per_chain': arch.n_beads,
                    'n_bonds_per_chain': arch.n_bonds,
                    'is_branched': arch.is_branched,
                }
                for arch, n, sname in self._species
            ],
            'total_atoms': self.total_beads,
            'total_bonds': sum(arch.n_bonds * n for arch, n, _ in self._species),
            'total_angles': sum(len(t) for t in chain_triplets),
            'bond_style': self.bond_style,
            'bond_length': self.bond_length,
            'k_bond': self.k_bond,
            'use_angles': self.use_angles,
            'bead_types': [bt.name for bt in self.bead_types],
            'output_path': self.path,
        }
        if self._fillers:
            from ..nanoparticles.cg import summary
            if self._placed_fillers:
                info['fillers'] = summary(
                    self._placed_fillers, self._calculate_box_size()
                )
            else:
                info['fillers'] = {
                    'species': {f.name: {'count': f.count, 'radius': f.radius}
                                for f in self._fillers},
                    'n_particles': self.n_fillers,
                }
        return info
