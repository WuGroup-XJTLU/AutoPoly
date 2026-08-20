//! Monte Carlo on the atomistic engine: atom displacement and torsion
//! turns (rigid rotation of one chain side about a backbone bond,
//! preserving all bonds/angles), with cell-list-accelerated local ΔU.

use crate::atomistic::{AtomisticEngine, AtomisticState};
use crate::bridge::{self, BridgeGate, TypeTables};
use rand::Rng;
use std::collections::HashMap;

/// Simple cell list for the atomistic pair cutoff.
pub struct ACellList {
    n: usize,
    w: f64,
    cells: Vec<Vec<usize>>,
    cell_of: Vec<usize>,
    brute: bool,
    n_atoms: usize,
}

impl ACellList {
    pub fn build(state: &AtomisticState, cutoff: f64) -> Self {
        Self::build_from(&state.pos, state.box_size, cutoff)
    }

    /// Build from an explicit position buffer (e.g. a working copy mid-move).
    pub fn build_from(pos: &[[f64; 3]], box_size: f64, cutoff: f64) -> Self {
        let l = box_size;
        let nc = (l / cutoff).floor() as usize;
        let n = nc.max(1);
        let brute = n < 3;
        let w = l / n as f64;
        let total = n * n * n;
        let mut cl = ACellList {
            n,
            w,
            cells: vec![Vec::new(); total],
            cell_of: vec![0; pos.len()],
            brute,
            n_atoms: pos.len(),
        };
        for i in 0..pos.len() {
            let idx = cl.index(pos[i]);
            cl.cells[idx].push(i);
            cl.cell_of[i] = idx;
        }
        cl
    }

    #[inline]
    fn index(&self, p: [f64; 3]) -> usize {
        let mut c = [0usize; 3];
        for a in 0..3 {
            let mut i = (p[a] / self.w).floor() as isize;
            i = i.clamp(0, self.n as isize - 1);
            c[a] = i as usize;
        }
        (c[2] * self.n + c[1]) * self.n + c[0]
    }

    pub fn update(&mut self, atom: usize, pos: [f64; 3]) {
        if self.brute {
            return;
        }
        let idx = self.index(pos);
        let old = self.cell_of[atom];
        if idx != old {
            if let Some(p) = self.cells[old].iter().position(|&x| x == atom) {
                self.cells[old].swap_remove(p);
            }
            self.cells[idx].push(atom);
            self.cell_of[atom] = idx;
        }
    }

    pub fn candidates_into(&self, p: [f64; 3], out: &mut Vec<usize>) {
        if self.brute {
            out.extend(0..self.n_atoms);
            return;
        }
        let mut c = [0usize; 3];
        for a in 0..3 {
            let mut i = (p[a] / self.w).floor() as isize;
            i = i.clamp(0, self.n as isize - 1);
            c[a] = i as usize;
        }
        for dz in -1isize..=1 {
            let cz = (c[2] as isize + dz).rem_euclid(self.n as isize) as usize;
            for dy in -1isize..=1 {
                let cy = (c[1] as isize + dy).rem_euclid(self.n as isize) as usize;
                for dx in -1isize..=1 {
                    let cx = (c[0] as isize + dx).rem_euclid(self.n as isize) as usize;
                    out.extend_from_slice(&self.cells[(cz * self.n + cy) * self.n + cx]);
                }
            }
        }
    }
}

/// Neumaier-compensated energy accumulator.
pub struct Acc {
    pub value: f64,
    comp: f64,
}

impl Acc {
    pub fn new(v: f64) -> Self {
        Acc { value: v, comp: 0.0 }
    }
    pub fn add(&mut self, d: f64) {
        let t = self.value + d;
        if self.value.abs() >= d.abs() {
            self.comp += (self.value - t) + d;
        } else {
            self.comp += (d - t) + self.value;
        }
        self.value = t;
    }
    pub fn total(&self) -> f64 {
        self.value + self.comp
    }
}

#[derive(Clone, Copy)]
pub struct AMcParams {
    pub max_displacement: f64, // Angstrom
    pub max_torsion: f64,      // radians
    pub w_displacement: f64,
    pub w_torsion: f64,
    /// Attempt a double-bridge every `bridge_every` steps (0 = off).
    pub bridge_every: usize,
}

impl Default for AMcParams {
    fn default() -> Self {
        AMcParams {
            max_displacement: 0.06,
            max_torsion: 0.35,
            w_displacement: 0.7,
            w_torsion: 0.3,
            bridge_every: 0,
        }
    }
}

pub struct AtomisticMC {
    pub engine: AtomisticEngine,
    pub cells: ACellList,
    pub params: AMcParams,
    pub energy: Acc,
    pub attempts: HashMap<String, u64>,
    pub accepts: HashMap<String, u64>,
    /// per chain: backbone position -> all member atom indices
    /// (backbone atom + its non-backbone substituents)
    pub unit_atoms: Vec<Vec<Vec<usize>>>,
    pub bridge_tables: TypeTables,
    pub bridge_gate: BridgeGate,
}

impl AtomisticMC {
    pub fn new(engine: AtomisticEngine, params: AMcParams) -> Self {
        let cutoff = engine.params.lj_cut.max(engine.params.coul_cut);
        let cells = ACellList::build(&engine.state, cutoff);
        let e0 = engine.total_energy();
        let unit_atoms = build_unit_atoms(&engine.state);
        let bridge_tables = TypeTables::from_state(&engine.state);
        AtomisticMC {
            engine,
            cells,
            params,
            energy: Acc::new(e0),
            attempts: HashMap::new(),
            accepts: HashMap::new(),
            unit_atoms,
            bridge_tables,
            bridge_gate: BridgeGate::default(),
        }
    }

    /// Local energy over moved atoms using the cell list for pairs.
    fn local_energy(&self, moved: &[usize]) -> f64 {
        let st = &self.engine.state;
        let mut e = 0.0;
        let mut moved_sorted = moved.to_vec();
        moved_sorted.sort_unstable();
        moved_sorted.dedup();
        let in_moved = |x: usize| moved_sorted.binary_search(&x).is_ok();

        let mut cand: Vec<usize> = Vec::with_capacity(128);
        for &i in &moved_sorted {
            cand.clear();
            self.cells.candidates_into(st.pos[i], &mut cand);
            for &j in &cand {
                if j == i || (in_moved(j) && j < i) {
                    continue;
                }
                e += self.engine.pair_e_pub(i, j);
            }
        }
        // bonded terms via the engine's term lists
        e += self.engine.local_bonded_energy(&moved_sorted);
        e
    }

    pub fn run<R: Rng>(&mut self, n_steps: usize, rng: &mut R) {
        let n_atoms = self.engine.state.pos.len();
        let n_chains = self.engine.state.chains.len();
        let bridge_every = self.params.bridge_every;
        for step in 0..n_steps {
            if bridge_every > 0 && step % bridge_every == 0 {
                self.step_bridge(rng);
            }
            let r: f64 = rng.random();
            if r < self.params.w_displacement
                || n_chains == 0
            {
                *self.attempts.entry("displacement".into()).or_insert(0) += 1;
                let i = rng.random_range(0..n_atoms);
                let old = self.engine.state.pos[i];
                let d = [
                    rng.random_range(-self.params.max_displacement..self.params.max_displacement),
                    rng.random_range(-self.params.max_displacement..self.params.max_displacement),
                    rng.random_range(-self.params.max_displacement..self.params.max_displacement),
                ];
                let e_old = self.local_energy(&[i]);
                self.engine.state.pos[i] = self.engine.state.wrap([
                    old[0] + d[0],
                    old[1] + d[1],
                    old[2] + d[2],
                ]);
                self.cells.update(i, self.engine.state.pos[i]);
                let e_new = self.local_energy(&[i]);
                let delta = e_new - e_old;
                if delta <= 0.0
                    || rng.random::<f64>() < (-delta / self.engine.temperature_kbt()).exp()
                {
                    self.energy.add(delta);
                    *self.accepts.entry("displacement".into()).or_insert(0) += 1;
                } else {
                    self.engine.state.pos[i] = old;
                    self.cells.update(i, old);
                }
            } else {
                *self.attempts.entry("torsion".into()).or_insert(0) += 1;
                self.torsion_turn(rng);
            }
        }
    }

    /// Loose-gate candidate enumeration for the HMC-heal driver.
    pub fn enumerate_loose(&mut self, a: usize) -> Vec<bridge::BridgeProposal> {
        let cell = |p: [f64; 3]| -> Vec<usize> {
            let mut v = Vec::new();
            self.cells.candidates_into(p, &mut v);
            v
        };
        bridge::enumerate_bridges_loose(&self.engine, &self.bridge_gate, a, &cell)
    }

    /// Apply a bridge proposal (HMC heal driver adopts it after
    /// acceptance). Returns false if type resolution failed.
    pub fn apply_bridge_pub(&mut self, pr: &bridge::BridgeProposal) -> bool {
        match bridge::apply_bridge(&mut self.engine.state, &self.bridge_tables, pr) {
            Ok(()) => {
                self.unit_atoms = build_unit_atoms(&self.engine.state);
                true
            }
            Err(_) => false,
        }
    }

    /// Replace all positions (healed state from the MD proxy); rebuilds
    /// the cell list.
    pub fn set_positions(&mut self, positions: Vec<[f64; 3]>) {
        for (i, p) in positions.into_iter().enumerate() {
            self.engine.state.pos[i] = p;
        }
        let cutoff = self.engine.params.lj_cut.max(self.engine.params.coul_cut);
        self.cells = ACellList::build(&self.engine.state, cutoff);
        let e = self.engine.total_energy();
        self.energy = Acc::new(e);
    }

    /// One heat-bath double-bridge step: enumerate feasible candidates
    /// for a random chain, choose among {null, candidates} with
    /// p ~ exp(-dU/kBT). Positions never move (topology-only), so the
    /// cell list stays valid; the per-atom term lists are rebuilt.
    pub fn step_bridge<R: Rng>(&mut self, rng: &mut R) -> bool {
        *self.attempts.entry("bridge".into()).or_insert(0) += 1;
        let n_chains = self.engine.state.chains.len();
        if n_chains < 2 {
            return false;
        }
        let a = rng.random_range(0..n_chains);
        let cell = |p: [f64; 3]| -> Vec<usize> {
            let mut v = Vec::new();
            self.cells.candidates_into(p, &mut v);
            v
        };
        let cands = bridge::enumerate_bridges(
            &self.engine,
            &self.bridge_tables,
            &self.bridge_gate,
            a,
            &cell,
            rng,
        );
        if cands.is_empty() {
            return false;
        }
        let kbt = self.engine.temperature_kbt();
        let dmin = cands.iter().map(|(_, d)| *d).fold(0.0f64, f64::min);
        let w: Vec<f64> = cands
            .iter()
            .map(|(_, d)| (-(d - dmin) / kbt).exp())
            .collect();
        let w_null = (dmin / kbt).exp();
        let w_total: f64 = w.iter().sum::<f64>() + w_null;
        let mut r = rng.random::<f64>() * w_total;
        let mut chosen: Option<usize> = None;
        for (i, wi) in w.iter().enumerate() {
            r -= wi;
            if r <= 0.0 {
                chosen = Some(i);
                break;
            }
        }
        let Some(k) = chosen else {
            return false;
        };
        let (pr, delta) = cands[k].clone();
        match bridge::apply_bridge(
            &mut self.engine.state,
            &self.bridge_tables,
            &pr,
        ) {
            Ok(()) => {
                self.energy.add(delta);
                *self.accepts.entry("bridge".into()).or_insert(0) += 1;
                // membership cache for torsion turns changed
                self.unit_atoms = build_unit_atoms(&self.engine.state);
                true
            }
            Err(_) => false,
        }
    }

    /// Rigidly rotate one side of a random backbone bond about the bond
    /// axis by a small random angle. Bonds/angles are preserved exactly.
    fn torsion_turn<R: Rng>(&mut self, rng: &mut R) {
        let n_chains = self.engine.state.chains.len();
        let c = rng.random_range(0..n_chains);
        let n_bb = self.engine.state.chains[c].len();
        if n_bb < 3 {
            return;
        }
        let k = rng.random_range(0..n_bb - 1);
        // rotate the smaller side
        let left_size = k + 1;
        let right_size = n_bb - k - 1;
        let rotate_tail = right_size <= left_size;
        let (axis_a, axis_b) = if rotate_tail {
            (self.engine.state.chains[c][k], self.engine.state.chains[c][k + 1])
        } else {
            (self.engine.state.chains[c][k + 1], self.engine.state.chains[c][k])
        };
        let positions: Vec<usize> = if rotate_tail {
            (k + 1..n_bb).collect()
        } else {
            (0..=k).collect()
        };
        let mut moved: Vec<usize> = Vec::new();
        for p in &positions {
            moved.extend_from_slice(&self.unit_atoms[c][*p]);
        }
        moved.sort_unstable();
        moved.dedup();

        let pa = self.engine.state.pos[axis_a];
        let axis_d = self.engine.state.disp(axis_a, axis_b);
        let alen = crate::atomistic::norm_pub(axis_d);
        if alen < 1e-9 {
            return;
        }
        let u = [axis_d[0] / alen, axis_d[1] / alen, axis_d[2] / alen];
        let theta = rng.random_range(-self.params.max_torsion..self.params.max_torsion);

        let e_old = self.local_energy(&moved);
        let mut old_pos: Vec<(usize, [f64; 3])> = Vec::with_capacity(moved.len());
        let (ct, stheta) = (theta.cos(), theta.sin());
        for &b in &moved {
            let pb = self.engine.state.pos[b];
            old_pos.push((b, pb));
            let rel = self.engine.state.disp(axis_a, b);
            let rot = crate::atomistic::rotate_pub(rel, u, ct, stheta);
            let np = self
                .engine
                .state
                .wrap([pa[0] + rot[0], pa[1] + rot[1], pa[2] + rot[2]]);
            self.engine.state.pos[b] = np;
            self.cells.update(b, np);
        }
        let e_new = self.local_energy(&moved);
        let delta = e_new - e_old;
        if delta <= 0.0
            || rng.random::<f64>() < (-delta / self.engine.temperature_kbt()).exp()
        {
            self.energy.add(delta);
            *self.accepts.entry("torsion".into()).or_insert(0) += 1;
        } else {
            for (b, op) in old_pos {
                self.engine.state.pos[b] = op;
                self.cells.update(b, op);
            }
        }
    }
}

/// Per chain, per backbone position: member atoms (backbone atom plus
/// bonded substituents that are not themselves backbone members).
pub fn build_unit_atoms(state: &AtomisticState) -> Vec<Vec<Vec<usize>>> {
    let mut backbone_set: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    for chain in &state.chains {
        for &b in chain {
            backbone_set.insert(b);
        }
    }
    let mut out = Vec::with_capacity(state.chains.len());
    for chain in &state.chains {
        let mut per_chain = Vec::with_capacity(chain.len());
        for &bb in chain {
            let mut members = vec![bb];
            for &(i, j, _) in &state.bonds {
                if i == bb && !backbone_set.contains(&j) {
                    members.push(j);
                } else if j == bb && !backbone_set.contains(&i) {
                    members.push(i);
                }
            }
            per_chain.push(members);
        }
        out.push(per_chain);
    }
    out
}
