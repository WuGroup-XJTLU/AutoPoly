//! MC engine: move scheduling, local ΔU evaluation, Metropolis test,
//! commit/rollback, and per-move acceptance statistics.

use crate::energy::{local_energy, BondModel, CellList, PairParams};
use crate::moves::{propose, MoveKind, MoveParams};
use crate::state::MeltState;
use rand::Rng;
use std::collections::HashMap;

pub struct Engine {
    pub state: MeltState,
    pub cells: CellList,
    pub pair: PairParams,
    pub bond: BondModel,
    pub angle_k: f64,
    pub temperature: f64,
    pub move_params: MoveParams,
    /// (kind, relative weight); normalized on the fly.
    pub move_weights: Vec<(MoveKind, f64)>,
    cum_weights: Vec<f64>,
    pub attempts: HashMap<MoveKind, u64>,
    pub accepts: HashMap<MoveKind, u64>,
    /// Total energy per bead (internal representation: keeps the
    /// accumulator at O(1) so local updates retain full precision).
    pub energy_per_bead: f64,
    /// Kahan compensation for `energy_per_bead`.
    energy_comp: f64,
    pub debug_watch: bool,
    pub debug_events: Vec<String>,
    last_drift: f64,
}

impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        state: MeltState,
        pair: PairParams,
        bond: BondModel,
        angle_k: f64,
        temperature: f64,
        move_params: MoveParams,
        move_weights: Vec<(MoveKind, f64)>,
    ) -> Result<Self, String> {
        if temperature <= 0.0 {
            return Err(format!("temperature must be positive, got {temperature}"));
        }
        let usable: Vec<(MoveKind, f64)> = move_weights
            .into_iter()
            .filter(|(_, w)| *w > 0.0)
            .collect();
        if usable.is_empty() {
            return Err("at least one move must have positive weight".to_string());
        }
        let total: f64 = usable.iter().map(|(_, w)| w).sum();
        let mut cum = Vec::with_capacity(usable.len());
        let mut acc = 0.0;
        for (_, w) in &usable {
            acc += w / total;
            cum.push(acc);
        }
        let cells = CellList::build(&state, pair.r_cut());
        let energy = crate::energy::total_energy(&state, &cells, &pair, &bond, angle_k)
            / state.n_beads().max(1) as f64;
        Ok(Engine {
            state,
            cells,
            pair,
            bond,
            angle_k,
            temperature,
            move_params,
            move_weights: usable,
            cum_weights: cum,
            attempts: HashMap::new(),
            accepts: HashMap::new(),
            energy_per_bead: energy,
            energy_comp: 0.0,
            debug_watch: false,
            debug_events: Vec::new(),
            last_drift: 0.0,
        })
    }

    #[inline]
    fn pick_move<R: Rng>(&self, rng: &mut R) -> MoveKind {
        let r: f64 = rng.random();
        let idx = self
            .cum_weights
            .partition_point(|&cw| cw < r)
            .min(self.move_weights.len() - 1);
        self.move_weights[idx].0
    }

    pub fn run<R: Rng>(&mut self, n_steps: usize, rng: &mut R) {
        let n_chains = self.state.n_chains();
        for _ in 0..n_steps {
            let kind = self.pick_move(rng);
            *self.attempts.entry(kind).or_insert(0) += 1;
            let c = rng.random_range(0..n_chains);

            let Some(prop) = propose(kind, &self.state, c, &self.move_params, rng) else {
                continue;
            };

            let e_old = local_energy(
                &self.state,
                &self.cells,
                &self.pair,
                &self.bond,
                self.angle_k,
                &prop.moved,
            );

            // Apply trial geometry.
            for &(b, np) in &prop.new_pos {
                self.state.pos[b] = np;
                self.cells.update(b, np);
            }

            let e_new = local_energy(
                &self.state,
                &self.cells,
                &self.pair,
                &self.bond,
                self.angle_k,
                &prop.moved,
            );

            let delta = e_new - e_old;
            let accept =
                delta <= 0.0 || rng.random::<f64>() < (-delta / self.temperature).exp();

            if accept {
                // Keep the accumulator at the per-bead scale and use
                // Neumaier compensated summation. A single huge local
                // delta (hard overlap during relaxation) would otherwise
                // leave a permanent ~1e4-ulp offset; the compensation
                // term recovers the lost low-order bits on later steps.
                let d = delta / self.state.n_beads().max(1) as f64;
                let t = self.energy_per_bead + d;
                if self.energy_per_bead.abs() >= d.abs() {
                    self.energy_comp += (self.energy_per_bead - t) + d;
                } else {
                    self.energy_comp += (d - t) + self.energy_per_bead;
                }
                self.energy_per_bead = t;
                *self.accepts.entry(kind).or_insert(0) += 1;
                if let Some(rc) = prop.reverse_chain {
                    self.state.chains[rc].reverse();
                    self.state.reindex_chain(rc);
                }
            } else {
                // Roll back to the pre-move geometry.
                for &(b, op) in &prop.old_pos {
                    self.state.pos[b] = op;
                    self.cells.update(b, op);
                }
            }

            // Debug: watch for NaN absorption or incremental drift.
            if self.debug_watch {
                if !delta.is_finite() {
                    self.debug_events
                        .push(format!("non-finite delta for {kind:?} on chain {c}"));
                } else {
                    let rec = crate::energy::total_energy(
                        &self.state,
                        &self.cells,
                        &self.pair,
                        &self.bond,
                        self.angle_k,
                    ) / self.state.n_beads().max(1) as f64;
                    let drift = (self.energy_per_bead + self.energy_comp) - rec;
                    if drift.abs() > self.last_drift.abs() + 1e-9 * (1.0 + rec.abs()) {
                        self.debug_events.push(format!(
                            "drift jump {last:+.6e} -> {drift:+.6e} after accept={accept} \
                             kind={kind:?} chain={c} moved={moved:?} delta={delta:+.6e}",
                            last = self.last_drift,
                            moved = prop.moved,
                        ));
                        self.last_drift = drift;
                    }
                }
            }
        }
    }

    pub fn energy_comp(&self) -> f64 {
        self.energy_comp
    }

    pub fn acceptance_rates(&self) -> HashMap<MoveKind, f64> {
        let mut out = HashMap::new();
        for (kind, &att) in &self.attempts {
            let acc = self.accepts.get(kind).copied().unwrap_or(0);
            out.insert(*kind, if att > 0 { acc as f64 / att as f64 } else { 0.0 });
        }
        out
    }
}
