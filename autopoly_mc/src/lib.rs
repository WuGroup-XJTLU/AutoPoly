//! autopoly_mc — Rust Monte Carlo kernel for AutoPoly melt equilibration.
//!
//! Phase 0: Kremer-Grest bead-spring melts, connectivity-preserving moves
//! with local ΔU, seeded RNG, structural invariant checks. Connectivity-
//! altering moves (join / segment exchange) arrive in Phase 1.

mod atomistic;
mod atomistic_mc;
mod atomistic_md;
mod bridge;
mod cbmc;
mod ncmc;
mod energy;
mod engine;
mod moves;
mod state;

use engine::Engine as RustEngine;
use energy::{BondModel, PairParams};
use moves::{MoveKind, MoveParams};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use state::MeltState;

fn to_py_err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

fn parse_move_weights(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(MoveKind, f64)>> {
    let dict: &Bound<'_, PyDict> = obj.downcast()?;
    let mut out = Vec::new();
    for (k, v) in dict.iter() {
        let name: String = k.extract()?;
        let weight: f64 = v.extract()?;
        let kind = match name.as_str() {
            "displacement" => MoveKind::Displacement,
            "pivot" => MoveKind::Pivot,
            "crankshaft" => MoveKind::Crankshaft,
            "reptation" => MoveKind::Reptation,
            "translation" | "chain_translation" => MoveKind::Translation,
            "rotation" | "chain_rotation" => MoveKind::Rotation,
            "segment_exchange" => MoveKind::SegmentExchange,
            "join" => MoveKind::Join,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown move kind: {other}"
                )))
            }
        };
        out.push((kind, weight));
    }
    Ok(out)
}

/// Monte Carlo engine bound to a melt state.
#[pyclass(name = "McEngine")]
struct PyEngine {
    engine: RustEngine,
    rng: ChaCha8Rng,
    seed: u64,
}

#[pymethods]
impl PyEngine {
    /// Create an engine.
    ///
    /// Args:
    ///     positions: list of [x, y, z] (wrapped into the box).
    ///     chains: list of ordered bead-id lists (the contour).
    ///     box_size: cubic periodic box edge.
    ///     seed: RNG seed (ChaCha8).
    ///     pair_*: LJ parameters; pair_wca selects the WCA variant
    ///         (cutoff 2^(1/6)*sigma, shifted).
    ///     bond_model: "harmonic" (bond_k, bond_r0) or "fene"
    ///         (bond_k, fene_r0max).
    ///     angle_k: stiffness of the k*(1-cos(theta)) angle potential
    ///         (0 disables it).
    ///     move_weights: {name: weight}; names: displacement, pivot,
    ///         crankshaft, reptation, translation, rotation.
    #[new]
    #[pyo3(signature = (
        positions,
        chains,
        box_size,
        seed = 12345,
        pair_epsilon = 1.0,
        pair_sigma = 1.0,
        pair_cutoff = 2.5,
        pair_shifted = false,
        pair_wca = false,
        mie_n = 12.0,
        exclude_bonded = true,
        bond_model = "harmonic",
        bond_k = 100.0,
        bond_r0 = 1.0,
        fene_r0max = 1.5,
        angle_k = 0.0,
        temperature = 1.0,
        max_displacement = 0.5,
        max_angle = 0.3,
        swap_r_max = 1.3,
        swap_full_delta = true,
        move_weights = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        positions: Vec<[f64; 3]>,
        chains: Vec<Vec<usize>>,
        box_size: f64,
        seed: u64,
        pair_epsilon: f64,
        pair_sigma: f64,
        pair_cutoff: f64,
        pair_shifted: bool,
        pair_wca: bool,
        mie_n: f64,
        exclude_bonded: bool,
        bond_model: &str,
        bond_k: f64,
        bond_r0: f64,
        fene_r0max: f64,
        angle_k: f64,
        temperature: f64,
        max_displacement: f64,
        max_angle: f64,
        swap_r_max: f64,
        swap_full_delta: bool,
        move_weights: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let pair = PairParams {
            epsilon: pair_epsilon,
            sigma: pair_sigma,
            cutoff: pair_cutoff,
            shifted: pair_shifted,
            wca: pair_wca,
            mie_n,
            exclude_bonded,
        };
        let bond = match bond_model {
            "harmonic" => BondModel::Harmonic { k: bond_k, r0: bond_r0 },
            "fene" => BondModel::Fene {
                k: bond_k,
                r0max: fene_r0max,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "bond_model must be 'harmonic' or 'fene', got {other:?}"
                )))
            }
        };
        let weights = match move_weights {
            Some(obj) => parse_move_weights(obj)?,
            None => vec![
                (MoveKind::Displacement, 0.4),
                (MoveKind::Crankshaft, 0.1),
                (MoveKind::Pivot, 0.15),
                (MoveKind::Reptation, 0.1),
                (MoveKind::Translation, 0.15),
                (MoveKind::Rotation, 0.1),
            ],
        };
        let state = MeltState::new(positions, chains, box_size).map_err(to_py_err)?;
        let mut engine = RustEngine::new(
            state,
            pair,
            bond,
            angle_k,
            temperature,
            MoveParams {
                max_displacement,
                max_angle,
            },
            weights,
        )
        .map_err(to_py_err)?;
        engine.swap_r_max = swap_r_max;
        engine.swap_full_delta = swap_full_delta;
        Ok(PyEngine {
            engine,
            rng: ChaCha8Rng::seed_from_u64(seed),
            seed,
        })
    }

    /// Run `n_steps` MC steps.
    fn run(&mut self, n_steps: usize) {
        self.engine.run(n_steps, &mut self.rng);
    }

    /// Attempt one proximity-directed end-to-end join of two equal-length
    /// chains (growth ladder). When `level_len` is given, only chains of
    /// that contour length are eligible. Returns True if accepted.
    #[pyo3(signature = (max_r, level_len = None))]
    fn try_join(&mut self, max_r: f64, level_len: Option<usize>) -> bool {
        self.engine.try_join(max_r, level_len, &mut self.rng)
    }

    /// Per-chain contour lengths.
    fn chain_lengths(&self) -> Vec<usize> {
        self.engine.state.chains.iter().map(|c| c.len()).collect()
    }



    /// Current positions as (N, 3) nested lists.
    fn positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out: Vec<[f64; 3]> = self.engine.state.pos.clone();
        PyList::new(py, out)
    }

    /// Per-chain ordered bead ids.
    fn chains(&self) -> Vec<Vec<usize>> {
        self.engine.state.chains.clone()
    }

    /// Total energy (incrementally maintained, compensated per-bead
    /// accumulator). Compare against `recompute_energy` to audit
    /// bookkeeping; note the comparison is limited by float64
    /// cancellation (~1e-9 relative) whenever hard overlaps exist.
    fn energy(&self) -> f64 {
        (self.engine.energy_per_bead + self.engine.energy_comp())
            * self.engine.state.n_beads().max(1) as f64
    }

    /// Recompute the total energy from scratch (consistency check).
    fn recompute_energy(&self) -> f64 {
        energy::total_energy(
            &self.engine.state,
            &self.engine.cells,
            &self.engine.pair,
            &self.engine.bond,
            self.engine.angle_k,
        )
    }

    /// Per-move acceptance rates as {name: rate}.
    fn acceptance_rates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (kind, rate) in self.engine.acceptance_rates() {
            dict.set_item(kind.name(), rate)?;
        }
        Ok(dict)
    }

    /// Per-move attempt counts as {name: count}.
    fn attempt_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (kind, &n) in &self.engine.attempts {
            dict.set_item(kind.name(), n)?;
        }
        Ok(dict)
    }

    /// Validate structural invariants; returns True or raises ValueError.
    fn validate(&self) -> PyResult<bool> {
        self.engine.state.validate().map_err(to_py_err)?;
        Ok(true)
    }

    /// Mean-square internal distance <R^2(s)> over all chains and contour
    /// separations s = 1..max_s (None => full range), minimum-image.
    #[pyo3(signature = (max_s = None))]
    fn msid(&self, max_s: Option<usize>) -> Vec<(usize, f64)> {
        let state = &self.engine.state;
        let n_chain = state.chains.first().map(|c| c.len()).unwrap_or(0);
        let smax = max_s.unwrap_or(n_chain.saturating_sub(1)).min(n_chain.saturating_sub(1));
        let mut acc = vec![0.0f64; smax + 1];
        let mut cnt = vec![0u64; smax + 1];
        for chain in &state.chains {
            let n = chain.len();
            for s in 1..=smax.min(n - 1) {
                for i in 0..n - s {
                    let d = state.disp(chain[i], chain[i + s]);
                    acc[s] += d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    cnt[s] += 1;
                }
            }
        }
        (1..=smax)
            .filter(|&s| cnt[s] > 0)
            .map(|s| (s, acc[s] / cnt[s] as f64))
            .collect()
    }






    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    #[getter]
    fn n_beads(&self) -> usize {
        self.engine.state.n_beads()
    }

    #[getter]
    fn n_chains(&self) -> usize {
        self.engine.state.n_chains()
    }
}

/// Cubic box edge for `n_beads` at bead number `density`.
#[pyfunction]
fn box_size_for_density(n_beads: usize, density: f64) -> f64 {
    (n_beads as f64 / density).cbrt()
}

/// Atomistic melt MC engine (OPLS-AA/GAFF subset from AutoPoly outputs).
#[pyclass(name = "AtomisticMC")]
struct PyAtomisticMC {
    mc: atomistic_mc::AtomisticMC,
    rng: ChaCha8Rng,
    seed: u64,
    md: Option<atomistic_md::MdState>,
    masses_by_type: Vec<f64>,
}

impl PyAtomisticMC {
    fn ensure_md(&mut self) {
        if self.md.is_none() {
            let mut md = atomistic_md::MdState::new(
                &self.mc.engine.state,
                &self.masses_by_type,
            );
            atomistic_md::compute_forces(
                &self.mc.engine.state,
                &self.mc.engine.params,
                None,
                &mut md.forces,
            );
            self.md = Some(md);
        }
    }
}

#[pymethods]
impl PyAtomisticMC {
    /// Build from explicit arrays (see AutoPoly.equilibration.atomistic_loader).
    ///
    /// `temperature` is kBT in kcal/mol (units real; e.g. 450 K = 0.894).
    #[new]
    #[pyo3(signature = (
        positions,
        box_size,
        types,
        charges,
        mols,
        bonds,
        angles,
        dihedrals,
        chains,
        pair_eps,
        pair_sig,
        bond_k,
        bond_r0,
        angle_k,
        angle_t0,
        dih_k,
        masses_by_type,
        lj_cut = 11.0,
        coul_cut = 11.0,
        scale14_lj = 0.5,
        scale14_coul = 0.5,
        temperature = 1.0,
        seed = 12345,
        max_displacement = 0.06,
        max_torsion = 0.35,
        w_displacement = 0.7,
        w_torsion = 0.3,
        bridge_every = 0,
        bridge_r_max = 1.9,
        bridge_jb_min = 1.30,
        bridge_jb_max = 1.75,
        bridge_ja_min = 1.65,
        bridge_ja_max = 2.35,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        positions: Vec<[f64; 3]>,
        box_size: f64,
        types: Vec<usize>,
        charges: Vec<f64>,
        mols: Vec<usize>,
        bonds: Vec<(usize, usize, usize)>,
        angles: Vec<(usize, usize, usize, usize)>,
        dihedrals: Vec<(usize, usize, usize, usize, usize)>,
        chains: Vec<Vec<usize>>,
        pair_eps: Vec<f64>,
        pair_sig: Vec<f64>,
        bond_k: Vec<f64>,
        bond_r0: Vec<f64>,
        angle_k: Vec<f64>,
        angle_t0: Vec<f64>,
        dih_k: Vec<[f64; 4]>,
        masses_by_type: Vec<f64>,
        lj_cut: f64,
        coul_cut: f64,
        scale14_lj: f64,
        scale14_coul: f64,
        temperature: f64,
        seed: u64,
        max_displacement: f64,
        max_torsion: f64,
        w_displacement: f64,
        w_torsion: f64,
        bridge_every: usize,
        bridge_r_max: f64,
        bridge_jb_min: f64,
        bridge_jb_max: f64,
        bridge_ja_min: f64,
        bridge_ja_max: f64,
    ) -> PyResult<Self> {
        let state = atomistic::AtomisticState::new(
            positions, box_size, types, charges, mols, bonds, angles, dihedrals, chains,
        );
        let params = atomistic::AtomisticParams {
            pair_eps,
            pair_sig,
            bond_k,
            bond_r0,
            angle_k,
            angle_t0,
            dih_k,
            lj_cut,
            coul_cut,
            scale14_lj,
            scale14_coul,
        };
        let engine = atomistic::AtomisticEngine::new(state, params, temperature);
        let mut mc = atomistic_mc::AtomisticMC::new(
            engine,
            atomistic_mc::AMcParams {
                max_displacement,
                max_torsion,
                w_displacement,
                w_torsion,
                bridge_every,
            },
        );
        mc.bridge_gate.r_max = bridge_r_max;
        mc.bridge_gate.jb_min = bridge_jb_min;
        mc.bridge_gate.jb_max = bridge_jb_max;
        mc.bridge_gate.ja_min = bridge_ja_min;
        mc.bridge_gate.ja_max = bridge_ja_max;
        Ok(PyAtomisticMC {
            mc,
            rng: ChaCha8Rng::seed_from_u64(seed),
            seed,
            md: None,
            masses_by_type,
        })
    }

    /// Total energy recomputed from scratch (kcal/mol).
    fn recompute_energy(&self) -> f64 {
        self.mc.engine.total_energy()
    }

    /// Replace the LJ epsilon and OPLS dihedral coefficient tables
    /// (atomistic core-softening ramp). The caller passes complete
    /// 1-indexed tables computed from the pristine force field; the
    /// tracked energy is refreshed against the new parameters.
    fn set_soft_params(&mut self, pair_eps: Vec<f64>, dih_k: Vec<[f64; 4]>) {
        self.mc.engine.params.pair_eps = pair_eps;
        self.mc.engine.params.dih_k = dih_k;
        let e = self.mc.engine.total_energy();
        self.mc.energy = atomistic_mc::Acc::new(e);
        self.md = None;
    }

    /// Current LJ epsilon table (1-indexed).
    fn pair_eps(&self) -> Vec<f64> {
        self.mc.engine.params.pair_eps.clone()
    }

    /// Current dihedral coefficient table (1-indexed).
    fn dih_k(&self) -> Vec<[f64; 4]> {
        self.mc.engine.params.dih_k.clone()
    }

    /// Incrementally tracked energy (kcal/mol).
    fn energy(&self) -> f64 {
        self.mc.energy.total()
    }

    /// Run n MC steps (atom displacements + torsion turns, plus bridge
    /// attempts every `bridge_every` steps if enabled).
    fn run(&mut self, n_steps: usize) {
        self.mc.run(n_steps, &mut self.rng);
    }

    /// One double-bridge attempt; returns True if a swap was applied.
    fn step_bridge(&mut self) -> bool {
        self.mc.step_bridge(&mut self.rng)
    }

    /// Initialize Maxwell-Boltzmann velocities at kBT (kcal/mol) and
    /// return the kinetic energy.
    fn md_init_velocities(&mut self, kbt: f64) -> f64 {
        self.ensure_md();
        atomistic_md::init_velocities(self.md.as_mut().unwrap(), kbt, &mut self.rng);
        self.md.as_ref().unwrap().kinetic_energy()
    }

    /// Run `n_steps` NVE velocity-Verlet steps with timestep `dt_fs`.
    /// Returns (potential_energy, kinetic_energy).
    fn md_run_nve(&mut self, n_steps: usize, dt_fs: f64) -> (f64, f64) {
        self.ensure_md();
        let par = self.mc.engine.params.clone();
        let cutoff = par.lj_cut.max(par.coul_cut);
        let md = self.md.as_mut().unwrap();
        let state = &mut self.mc.engine.state;
        let mut cells = atomistic_mc::ACellList::build(state, cutoff);
        for _ in 0..n_steps {
            atomistic_md::verlet_step(state, &par, md, Some(&cells), dt_fs);
            cells = atomistic_mc::ACellList::build(state, cutoff);
        }
        let ke = md.kinetic_energy();
        (self.mc.engine.total_energy(), ke)
    }

    /// Current velocities.
    fn md_velocities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out: Vec<[f64; 3]> = match &self.md {
            Some(md) => md.vel.clone(),
            None => Vec::new(),
        };
        PyList::new(py, out)
    }

    /// Loose-gate bridge candidates for chain `a` as (a, b, s, flip)
    /// tuples (HMC-heal driver).
    fn enumerate_bridge_candidates(&mut self, a: usize) -> Vec<(usize, usize, usize, bool)> {
        self.mc
            .enumerate_loose(a)
            .into_iter()
            .map(|p| (p.a, p.b, p.s, p.flip))
            .collect()
    }

    /// Number of backbone chains.
    fn n_chains(&self) -> usize {
        self.mc.engine.state.chains.len()
    }

    /// Apply a bridge proposal; returns True on success.
    fn apply_bridge_proposal(&mut self, a: usize, b: usize, s: usize, flip: bool) -> bool {
        self.mc.apply_bridge_pub(&bridge::BridgeProposal { a, b, s, flip })
    }

    /// Bond list as (i, j, type) tuples.
    fn bonds(&self) -> Vec<(usize, usize, usize)> {
        self.mc.engine.state.bonds.clone()
    }

    /// Angle list as (i, j, k, type) tuples.
    fn angles(&self) -> Vec<(usize, usize, usize, usize)> {
        self.mc.engine.state.angles.clone()
    }

    /// Dihedral list as (i, j, k, l, type) tuples.
    fn dihedrals(&self) -> Vec<(usize, usize, usize, usize, usize)> {
        self.mc.engine.state.dihedrals.clone()
    }

    /// Replace all positions (e.g. with an MD-healed configuration).
    fn set_positions(&mut self, positions: Vec<[f64; 3]>) {
        self.mc.set_positions(positions);
    }

    /// Diagnostic: energy of the λ-switched system at current positions
    /// for the given bridge proposal (decomposition sanity check).
    fn ncmc_probe_energy(&self, a: usize, b: usize, s: usize, flip: bool, lambda: f64) -> (f64, f64, f64) {
        let old_state = self.mc.engine.state.clone();
        let mut new_state = old_state.clone();
        let pr = bridge::BridgeProposal { a, b, s, flip };
        let _ = bridge::apply_bridge(&mut new_state, &self.mc.bridge_tables, &pr);
        let terms = ncmc::NcmcTerms::build(&old_state, &new_state);
        let par = &self.mc.engine.params;
        let pos: Vec<[f64; 3]> = old_state.pos.clone();
        let e_lam = terms.energy_at_lambda(&pos, old_state.box_size, par, lambda);
        let e_old = self.mc.engine.total_energy();
        let eng_new = atomistic::AtomisticEngine::new(new_state, par.clone(), 1.0);
        (e_lam, e_old, eng_new.total_energy())
    }

    /// CBMC-reachable bridge candidates for chain `a` as (a, b, s, flip)
    /// tuples, using the k-mer regrowth reach criterion (r_reach Å).
    #[pyo3(signature = (a, r_reach, k_regrow = 3))]
    fn cbmc_candidates(&self, a: usize, r_reach: f64, k_regrow: usize) -> Vec<(usize, usize, usize, bool)> {
        let cell_cutoff = self.mc.engine.params.lj_cut.max(self.mc.engine.params.coul_cut);
        let cells = atomistic_mc::ACellList::build(&self.mc.engine.state, cell_cutoff);
        cbmc::enumerate_cbmc_candidates(&self.mc.engine.state, &cells, a, r_reach, k_regrow)
            .into_iter()
            .map(|p| (p.a, p.b, p.s, p.flip))
            .collect()
    }

    /// One CBMC k-mer-regrowth double-bridge attempt.
    /// Returns (accepted, log_accept_ratio).
    #[pyo3(signature = (a, b, s, flip, kbt, r_reach = 4.5, n_trials = 30, n_psi = 144, k_regrow = 3, n_mtm = 1))]
    fn cbmc_bridge_attempt(
        &mut self,
        a: usize,
        b: usize,
        s: usize,
        flip: bool,
        kbt: f64,
        r_reach: f64,
        n_trials: usize,
        n_psi: usize,
        k_regrow: usize,
        n_mtm: usize,
    ) -> (bool, f64) {
        let cfg = cbmc::CbmcConfig {
            r_reach,
            n_trials,
            n_psi,
            n_regrow: k_regrow,
            ..cbmc::CbmcConfig::default()
        };
        let cell_cutoff = self.mc.engine.params.lj_cut.max(self.mc.engine.params.coul_cut);
        let cells = atomistic_mc::ACellList::build(&self.mc.engine.state, cell_cutoff);
        let n_fwd_total: usize = (0..self.mc.engine.state.chains.len())
            .map(|c| {
                cbmc::enumerate_cbmc_candidates(&self.mc.engine.state, &cells, c, cfg.r_reach, k_regrow)
                    .len()
            })
            .sum();
        let pr = bridge::BridgeProposal { a, b, s, flip };
        let par = self.mc.engine.params.clone();
        let out = cbmc::cbmc_double_bridge_mtm(
            &mut self.mc.engine.state,
            &self.mc.bridge_tables,
            &par,
            &pr,
            &cfg,
            kbt,
            n_fwd_total,
            n_mtm,
            &mut self.rng,
        );
        if out.accepted {
            self.mc.set_positions(self.mc.engine.state.pos.clone());
            self.mc.unit_atoms = atomistic_mc::build_unit_atoms(&self.mc.engine.state);
            self.md = None;
        }
        (out.accepted, out.log_accept_ratio)
    }

    /// One NCMC double-bridge attempt for a candidate on chain `a`.
    /// Returns (accepted, work, accept_prob).
    #[pyo3(signature = (a, b, s, flip, kbt, n_switch = 20, steps_per_switch = 15, dt_fs = 0.25))]
    fn ncmc_bridge_attempt(
        &mut self,
        a: usize,
        b: usize,
        s: usize,
        flip: bool,
        kbt: f64,
        n_switch: usize,
        steps_per_switch: usize,
        dt_fs: f64,
    ) -> (bool, f64, f64) {
        // build the new topology on a clone of the state
        let old_state = self.mc.engine.state.clone();
        let mut new_state = old_state.clone();
        let pr = bridge::BridgeProposal { a, b, s, flip };
        if bridge::apply_bridge(&mut new_state, &self.mc.bridge_tables, &pr).is_err() {
            return (false, f64::INFINITY, 0.0);
        }
        let par = self.mc.engine.params.clone();
        let (result, pos, _vel, prob) = ncmc::ncmc_attempt(
            &old_state,
            &new_state,
            &par,
            &self.masses_by_type,
            kbt,
            n_switch,
            steps_per_switch,
            dt_fs,
            &mut self.rng,
        );
        if result.accepted {
            // adopt new topology and healed positions
            self.mc.engine.state = new_state;
            self.mc.set_positions(pos);
            self.mc.unit_atoms = atomistic_mc::build_unit_atoms(&self.mc.engine.state);
            self.md = None; // forces/velocities stale; rebuilt lazily
        }
        (result.accepted, result.work, prob)
    }

    fn positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out: Vec<[f64; 3]> = self.mc.engine.state.pos.clone();
        PyList::new(py, out)
    }

    fn chains(&self) -> Vec<Vec<usize>> {
        self.mc.engine.state.chains.clone()
    }

    fn acceptance_rates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (k, &att) in &self.mc.attempts {
            let acc = self.mc.accepts.get(k).copied().unwrap_or(0);
            dict.set_item(k, if att > 0 { acc as f64 / att as f64 } else { 0.0 })?;
        }
        Ok(dict)
    }

    /// MSID over backbone chains: list of (s, R^2(s)).
    #[pyo3(signature = (max_s = None))]
    fn msid(&self, max_s: Option<usize>) -> Vec<(usize, f64)> {
        let st = &self.mc.engine.state;
        let n_chain = st.chains.first().map(|c| c.len()).unwrap_or(0);
        let smax = max_s
            .unwrap_or(n_chain.saturating_sub(1))
            .min(n_chain.saturating_sub(1));
        let mut acc = vec![0.0f64; smax + 1];
        let mut cnt = vec![0u64; smax + 1];
        for chain in &st.chains {
            let n = chain.len();
            for s in 1..=smax.min(n - 1) {
                for i in 0..n - s {
                    let d = st.disp(chain[i], chain[i + s]);
                    acc[s] += d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    cnt[s] += 1;
                }
            }
        }
        (1..=smax)
            .filter(|&s| cnt[s] > 0)
            .map(|s| (s, acc[s] / cnt[s] as f64))
            .collect()
    }
}

#[pymodule]
fn autopoly_mc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyAtomisticMC>()?;
    m.add_function(wrap_pyfunction!(box_size_for_density, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
