//! autopoly_mc — Rust Monte Carlo kernel for AutoPoly melt equilibration.
//!
//! Phase 0: Kremer-Grest bead-spring melts, connectivity-preserving moves
//! with local ΔU, seeded RNG, structural invariant checks. Connectivity-
//! altering moves (join / segment exchange) arrive in Phase 1.

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
        bond_model = "harmonic",
        bond_k = 100.0,
        bond_r0 = 1.0,
        fene_r0max = 1.5,
        angle_k = 0.0,
        temperature = 1.0,
        max_displacement = 0.5,
        max_angle = 0.3,
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
        bond_model: &str,
        bond_k: f64,
        bond_r0: f64,
        fene_r0max: f64,
        angle_k: f64,
        temperature: f64,
        max_displacement: f64,
        max_angle: f64,
        move_weights: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let pair = PairParams {
            epsilon: pair_epsilon,
            sigma: pair_sigma,
            cutoff: pair_cutoff,
            shifted: pair_shifted,
            wca: pair_wca,
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
        let engine = RustEngine::new(
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

#[pymodule]
fn autopoly_mc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_function(wrap_pyfunction!(box_size_for_density, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
