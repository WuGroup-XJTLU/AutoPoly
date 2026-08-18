//! MC engine: move scheduling, local ΔU evaluation, Metropolis test,
//! commit/rollback, and per-move acceptance statistics.

use crate::energy::{local_energy, BondModel, CellList, PairParams};
use crate::moves::{propose, MoveKind, MoveParams};
use crate::state::MeltState;

/// Proposal for an equal-length segment exchange between two chains:
/// cut both chains at contour position `s` and swap the tails,
/// optionally reversing each tail. For fixed (a, b, s, flips) the move
/// is an involution, so proposal probabilities are symmetric and the
/// Metropolis test needs only ΔU.
#[derive(Clone)]
pub struct SwapProposal {
    pub a: usize,
    pub b: usize,
    pub s: usize,
    pub flip_p: bool,
    pub flip_q: bool,
}
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
    /// Search radius for directed connectivity moves (segment exchange).
    /// Must be < FENE R0 for junctions to be physical; ~1.3 sigma works.
    pub swap_r_max: f64,
    /// If false, the swap ΔU follows the DBH convention (bond + angle
    /// terms only; pair terms neglected — Dietz & Hoy 2022, Fig. 1).
    /// If true, the exact ΔU including 1-2 exclusion flips is used.
    pub swap_full_delta: bool,
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
            swap_r_max: 1.3,
            swap_full_delta: true,
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

    /// Compensated (Neumaier) accumulation of an energy change.
    #[inline]
    fn acc_energy(&mut self, delta: f64) {
        let d = delta / self.state.n_beads().max(1) as f64;
        let t = self.energy_per_bead + d;
        if self.energy_per_bead.abs() >= d.abs() {
            self.energy_comp += (self.energy_per_bead - t) + d;
        } else {
            self.energy_comp += (d - t) + self.energy_per_bead;
        }
        self.energy_per_bead = t;
    }

    /// Beads within contour window [s-W, s+W) of a chain (clipped).
    /// Covers every bond / angle / exclusion term that a topology change
    /// at contour position `s` can touch (with margin).
    fn junction_window(&self, c: usize, s: usize) -> Vec<usize> {
        const W: usize = 3;
        let chain = &self.state.chains[c];
        let n = chain.len();
        let lo = s.saturating_sub(W);
        let hi = (s + W).min(n);
        chain[lo..hi].to_vec()
    }

    // ------------------------------------------------------------------
    // Segment exchange (connectivity-altering, monodisperse-preserving)
    // ------------------------------------------------------------------

    /// Proposal for an equal-length segment exchange between two chains:
    /// cut both at contour position `s` and swap the tails, optionally
    /// reversing each tail. The move is an involution for fixed
    /// (a, b, s, flips), so proposal probabilities are symmetric and the
    /// Metropolis test needs only ΔU.
    /// Enumerate every valid directed swap for chain `a` (all cut
    /// positions, both flip orientations), with each candidate's exact
    /// ΔU computed analytically (no state mutation). Orientations are
    /// restricted to flip_p == flip_q so each move is an involution.
    fn enumerate_swaps(&self, a: usize, r_max: f64) -> Vec<(SwapProposal, f64, f64)> {
        let n = self.state.chains[a].len();
        if n < 2 {
            return Vec::new();
        }
        let rc2 = r_max * r_max;
        let mut out = Vec::new();
        let mut cand: Vec<usize> = Vec::with_capacity(64);
        for s in 1..n {
            let a1 = self.state.chains[a][s - 1];
            cand.clear();
            self.cells.candidates_into(self.state.pos[a1], &mut cand);
            for &j in &cand {
                let cj = self.state.chain_of[j];
                if cj == a || cj == usize::MAX || self.state.chains[cj].len() != n {
                    continue;
                }
                let ij = self.state.idx_in_chain[j];
                let d = self.state.disp(a1, j);
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 >= rc2 {
                    continue;
                }
                let b1 = self.state.chains[cj][s - 1];
                if ij == s {
                    let a2 = self.state.chains[a][s];
                    let d2 = self.state.disp(b1, a2);
                    if d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2] < rc2 {
                        let pr = SwapProposal {
                            a,
                            b: cj,
                            s,
                            flip_p: false,
                            flip_q: false,
                        };
                        let (df, dd) = self.swap_delta(&pr);
                        out.push((pr, df, dd));
                    }
                } else if ij == n - 1 {
                    let a_end = *self.state.chains[a].last().unwrap();
                    let d2 = self.state.disp(b1, a_end);
                    if d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2] < rc2 {
                        let pr = SwapProposal {
                            a,
                            b: cj,
                            s,
                            flip_p: true,
                            flip_q: true,
                        };
                        let (df, dd) = self.swap_delta(&pr);
                        out.push((pr, df, dd));
                    }
                }
            }
        }
        out
    }

    /// Exact ΔU of a segment exchange, computed from the current state
    /// without mutating anything: junction bond swaps, angle-triplet
    /// replacements, and the four 1-2 exclusion flips.
    /// Returns (full ΔU, DBH ΔU). The full ΔU includes the 1-2
    /// exclusion-flip pair terms and is used for energy bookkeeping;
    /// the DBH ΔU (bond + angle only, Dietz & Hoy 2022 Fig. 1) drives
    /// the heat-bath weights when `swap_full_delta` is false.
    fn swap_delta(&self, pr: &SwapProposal) -> (f64, f64) {
        let n = self.state.chains[pr.a].len();
        let s = pr.s;
        let ca = &self.state.chains[pr.a];
        let cb = &self.state.chains[pr.b];
        let a1 = ca[s - 1];
        let a2 = ca[s];
        let b1 = cb[s - 1];
        let b2 = cb[s];
        // New junction partners under the chosen orientation.
        let (x, y) = if pr.flip_p {
            (cb[n - 1], ca[n - 1])
        } else {
            (b2, a2)
        };
        let mut de = 0.0;

        // Bonds: remove (a1,a2),(b1,b2); add (a1,x),(b1,y).
        de += self.bond_e(a1, x) + self.bond_e(b1, y)
            - self.bond_e(a1, a2) - self.bond_e(b1, b2);

        // 1-2 exclusion flips: old bonded pairs enter LJ, new leave it.
        // (Only models that exclude bonded pairs have these terms; under
        // the KG convention all pairs interact regardless, and the swap
        // leaves every pair energy unchanged.)
        let de_pair = if self.pair.exclude_bonded {
            self.lj_e(a1, a2) + self.lj_e(b1, b2)
                - self.lj_e(a1, x) - self.lj_e(b1, y)
        } else {
            0.0
        };
        let de_bond_only = de;
        de += de_pair;

        // Angles: triplets broken at the two cuts leave; triplets formed
        // across the new junctions enter.
        if self.angle_k != 0.0 {
            if s >= 2 {
                de -= self.angle_e(ca[s - 2], a1, a2);
                de -= self.angle_e(cb[s - 2], b1, b2);
            }
            if s + 1 < n {
                de -= self.angle_e(a1, a2, ca[s + 1]);
                de -= self.angle_e(b1, b2, cb[s + 1]);
            }
            // new junction triplets; x/y successors depend on orientation
            let succ_x = if pr.flip_p {
                if s + 1 < n { Some(cb[n - 2]) } else { None }
            } else if s + 1 < n {
                Some(cb[s + 1])
            } else {
                None
            };
            let succ_y = if pr.flip_p {
                if s + 1 < n { Some(ca[n - 2]) } else { None }
            } else if s + 1 < n {
                Some(ca[s + 1])
            } else {
                None
            };
            if s >= 2 {
                de += self.angle_e(ca[s - 2], a1, x);
                de += self.angle_e(cb[s - 2], b1, y);
            }
            if let Some(sx) = succ_x {
                de += self.angle_e(a1, x, sx);
            }
            if let Some(sy) = succ_y {
                de += self.angle_e(b1, y, sy);
            }
            (de, de - de_pair) // DBH = bonds + angles
        } else {
            (de, de_bond_only)
        }
    }

    #[inline]
    fn bond_e(&self, i: usize, j: usize) -> f64 {
        let d = self.state.disp(i, j);
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        crate::energy::bond_energy(r, &self.bond)
    }

    #[inline]
    fn lj_e(&self, i: usize, j: usize) -> f64 {
        let d = self.state.disp(i, j);
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        crate::energy::pair_energy(r2, &self.pair)
    }

    #[inline]
    fn angle_e(&self, a: usize, b: usize, c: usize) -> f64 {
        let b1 = self.state.disp(b, a);
        let b2 = self.state.disp(b, c);
        let n1 = (b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2]).sqrt();
        let n2 = (b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2]).sqrt();
        if n1 < 1e-12 || n2 < 1e-12 {
            return 0.0;
        }
        let dot = b1[0] * b2[0] + b1[1] * b2[1] + b1[2] * b2[2];
        let cos_theta = (dot / (n1 * n2)).clamp(-1.0, 1.0);
        self.angle_k * (1.0 - cos_theta)
    }

    /// One directed segment-exchange step as a heat-bath (Glauber)
    /// update: enumerate all valid swaps for a random chain, then choose
    /// among {no-op, swaps} with probability ∝ exp(-ΔU/T). Heat-bath
    /// over an involutive move set satisfies detailed balance directly;
    /// no separate Metropolis test is needed.
    fn step_segment_exchange<R: Rng>(&mut self, rng: &mut R) {
        let kind = MoveKind::SegmentExchange;
        *self.attempts.entry(kind).or_insert(0) += 1;
        let m = self.state.n_chains();
        if m < 2 {
            return;
        }
        let a = rng.random_range(0..m);
        let cands = self.enumerate_swaps(a, self.swap_r_max);
        if cands.is_empty() {
            return;
        }
        // Weights shifted by min ΔU (the null option has ΔU = 0).
        let wd = |c: &(SwapProposal, f64, f64)| if self.swap_full_delta { c.1 } else { c.2 };
        let dmin = cands.iter().map(|c| wd(c)).fold(0.0f64, f64::min);
        let w: Vec<f64> = cands
            .iter()
            .map(|c| (-(wd(c) - dmin) / self.temperature).exp())
            .collect();
        let w_null = (dmin / self.temperature).exp();
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
            return; // null option selected: rejected attempt
        };
        let (pr, d_full, _) = cands[k].clone();
        self.apply_segment_exchange(&pr);
        self.acc_energy(d_full);
        *self.accepts.entry(kind).or_insert(0) += 1;
    }

    /// Apply (or revert — it is an involution) a segment exchange.
    /// Returns the affected bead set (valid for both topologies).
    fn apply_segment_exchange(&mut self, pr: &SwapProposal) -> Vec<usize> {
        let s = pr.s;
        let mut affected = self.swap_affected(pr);

        let a_pre = self.state.chains[pr.a][..s].to_vec();
        let q = self.state.chains[pr.a][s..].to_vec(); // A tail
        let b_pre = self.state.chains[pr.b][..s].to_vec();
        let p = self.state.chains[pr.b][s..].to_vec(); // B tail

        let mut p2 = p.clone();
        if pr.flip_p {
            p2.reverse();
        }
        let mut q2 = q.clone();
        if pr.flip_q {
            q2.reverse();
        }

        let a1 = *a_pre.last().unwrap();
        let b1 = *b_pre.last().unwrap();
        self.state.remove_bond(a1, q[0]);
        self.state.remove_bond(b1, p[0]);
        self.state.add_bond(a1, p2[0]);
        self.state.add_bond(b1, q2[0]);

        let mut new_a = a_pre;
        new_a.extend_from_slice(&p2);
        let mut new_b = b_pre;
        new_b.extend_from_slice(&q2);
        for (i, &bead) in new_a.iter().enumerate() {
            self.state.chain_of[bead] = pr.a;
            self.state.idx_in_chain[bead] = i;
        }
        for (i, &bead) in new_b.iter().enumerate() {
            self.state.chain_of[bead] = pr.b;
            self.state.idx_in_chain[bead] = i;
        }
        self.state.chains[pr.a] = new_a;
        self.state.chains[pr.b] = new_b;

        affected.sort_unstable();
        affected.dedup();
        affected
    }

    /// Affected bead set for a segment exchange: the junction windows of
    /// both chains plus both chain ends. The ends matter because a
    /// flipped tail brings its far-end beads next to the junction, and
    /// their angle terms enter the new-topology evaluation.
    fn swap_affected(&self, pr: &SwapProposal) -> Vec<usize> {
        let mut v = self.junction_window(pr.a, pr.s);
        v.extend(self.junction_window(pr.b, pr.s));
        let na = self.state.chains[pr.a].len();
        let nb = self.state.chains[pr.b].len();
        v.extend(self.junction_window(pr.a, na));
        v.extend(self.junction_window(pr.b, nb));
        v.sort_unstable();
        v.dedup();
        v
    }

    // ------------------------------------------------------------------
    // Join (growth ladder; heuristic stage — see equilibration_method.md)
    // ------------------------------------------------------------------

    /// Attempt one proximity-directed end-to-end join of two equal-length
    /// chains: pick a random chain end, find the nearest end bead of a
    /// different equal-length chain within `max_r` (cell-list query), and
    /// join them with a Metropolis test on ΔU. The proximity direction
    /// breaks strict detailed balance; the equilibrium guarantee comes
    /// from the fixed-N annealing stage, not from this move.
    pub fn try_join<R: Rng>(
        &mut self,
        max_r: f64,
        level_len: Option<usize>,
        rng: &mut R,
    ) -> bool {
        let kind = MoveKind::Join;
        *self.attempts.entry(kind).or_insert(0) += 1;
        let m = self.state.n_chains();
        if m < 2 {
            return false;
        }
        // Ladder discipline: when level_len is set, only chains of that
        // contour length are eligible to join (binary ladder).
        let a = match level_len {
            Some(l) => {
                let eligible: Vec<usize> = (0..m)
                    .filter(|&c| self.state.chains[c].len() == l)
                    .collect();
                if eligible.len() < 2 {
                    return false;
                }
                eligible[rng.random_range(0..eligible.len())]
            }
            None => rng.random_range(0..m),
        };
        let na = self.state.chains[a].len();
        let ea_is_first = rng.random::<bool>();
        let ea = if ea_is_first {
            self.state.chains[a][0]
        } else {
            *self.state.chains[a].last().unwrap()
        };

        // Nearest eligible end bead of another equal-length chain.
        let mut cand: Vec<usize> = Vec::with_capacity(64);
        self.cells
            .candidates_into(self.state.pos[ea], &mut cand);
        let rc2 = max_r * max_r;
        let mut best: Option<(usize, usize, f64)> = None;
        for j in cand {
            let cj = self.state.chain_of[j];
            if cj == a || self.state.chains[cj].len() != na {
                continue;
            }
            let ij = self.state.idx_in_chain[j];
            if ij != 0 && ij + 1 != na {
                continue; // ends only
            }
            let d = self.state.disp(ea, j);
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 < rc2 && best.is_none_or(|(_, _, br2)| r2 < br2) {
                best = Some((cj, j, r2));
            }
        }
        let Some((c, ej, _)) = best else {
            return false;
        };

        // Affected beads: end neighborhoods of both chains (covers the
        // seam triplets in the new topology too).
        let mut affected: Vec<usize> = Vec::with_capacity(12);
        for &b in self.state.chains[a].iter().take(3) {
            affected.push(b);
        }
        for &b in self.state.chains[a].iter().rev().take(3) {
            affected.push(b);
        }
        for &b in self.state.chains[c].iter().take(3) {
            affected.push(b);
        }
        for &b in self.state.chains[c].iter().rev().take(3) {
            affected.push(b);
        }
        affected.sort_unstable();
        affected.dedup();

        let e_old = local_energy(
            &self.state,
            &self.cells,
            &self.pair,
            &self.bond,
            self.angle_k,
            &affected,
        );

        // Save originals for the revert path.
        let orig_a = self.state.chains[a].clone();
        let orig_c = self.state.chains[c].clone();
        let (lo, hi) = if a < c { (a, c) } else { (c, a) };

        // Orient: a_part ends at ea, c_part starts at ej.
        let mut a_part = orig_a.clone();
        if ea_is_first {
            a_part.reverse();
        }
        let mut c_part = orig_c.clone();
        if self.state.idx_in_chain[ej] + 1 == na {
            c_part.reverse();
        }
        let mut new_chain = a_part;
        new_chain.extend_from_slice(&c_part);

        self.state.add_bond(ea, ej);
        self.state.chains[lo] = new_chain;
        self.state.chains.remove(hi);
        self.state.rebuild_index();

        let e_new = local_energy(
            &self.state,
            &self.cells,
            &self.pair,
            &self.bond,
            self.angle_k,
            &affected,
        );

        let delta = e_new - e_old;
        if delta <= 0.0 || rng.random::<f64>() < (-delta / self.temperature).exp() {
            self.acc_energy(delta);
            *self.accepts.entry(kind).or_insert(0) += 1;
            true
        } else {
            // Revert: restore both original chains.
            self.state.remove_bond(ea, ej);
            self.state.chains[lo] = if a < c { orig_a.clone() } else { orig_c.clone() };
            self.state
                .chains
                .insert(hi, if a < c { orig_c } else { orig_a });
            self.state.rebuild_index();
            false
        }
    }

    pub fn run<R: Rng>(&mut self, n_steps: usize, rng: &mut R) {
        let n_chains = self.state.n_chains();
        for _ in 0..n_steps {
            let kind = self.pick_move(rng);
            if kind == MoveKind::SegmentExchange {
                self.step_segment_exchange(rng);
                continue;
            }
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
                // Per-bead Neumaier-compensated accumulation: keeps the
                // incremental energy exact through the cold-start
                // hard-overlap regime.
                self.acc_energy(delta);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn two_chain_state() -> MeltState {
        let mut pos = Vec::new();
        for i in 0..6 {
            pos.push([i as f64, 0.0, 0.0]);
        }
        for i in 0..6 {
            pos.push([i as f64, 20.0, 20.0]);
        }
        MeltState::new(pos, vec![vec![0, 1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10, 11]], 40.0)
            .unwrap()
    }

    #[test]
    fn swap_once_cross_swaps_tails() {
        let st = two_chain_state();
        let mut eng = Engine::new(
            st,
            PairParams::default(),
            BondModel::default(),
            0.0,
            1.0,
            MoveParams::default(),
            vec![(MoveKind::SegmentExchange, 1.0)],
        )
        .unwrap();
        let pr = SwapProposal { a: 0, b: 1, s: 4, flip_p: false, flip_q: false };
        eng.apply_segment_exchange(&pr);
        assert_eq!(eng.state.chains[0], vec![0, 1, 2, 3, 10, 11]);
        assert_eq!(eng.state.chains[1], vec![6, 7, 8, 9, 4, 5]);
        assert!(eng.state.validate().is_ok());
    }

    #[test]
    fn swap_energy_roundtrip() {
        // dense-ish pair of chains so bonds/LJ contribute
        let mut pos = Vec::new();
        for i in 0..10 {
            pos.push([0.9 * i as f64, 0.0, 0.0]);
        }
        for i in 0..10 {
            pos.push([0.9 * i as f64, 1.1, 0.5]);
        }
        let st = MeltState::new(
            pos,
            vec![(0..10).collect(), (10..20).collect()],
            30.0,
        )
        .unwrap();
        let mut eng = Engine::new(
            st,
            PairParams::default(),
            BondModel::Harmonic { k: 100.0, r0: 1.0 },
            1.5,
            1.0,
            MoveParams::default(),
            vec![(MoveKind::SegmentExchange, 1.0)],
        )
        .unwrap();
        let e0 = crate::energy::total_energy(
            &eng.state, &eng.cells, &eng.pair, &eng.bond, eng.angle_k,
        );
        for s_cut in [1, 3, 5, 8] {
            for (fp, fq) in [(false, false), (true, false), (false, true), (true, true)] {
                let pr = SwapProposal { a: 0, b: 1, s: s_cut, flip_p: fp, flip_q: fq };
                eng.apply_segment_exchange(&pr);
                let pr_rev = SwapProposal { flip_p: fq, flip_q: fp, ..pr.clone() };
                eng.apply_segment_exchange(&pr_rev);
                let e1 = crate::energy::total_energy(
                    &eng.state, &eng.cells, &eng.pair, &eng.bond, eng.angle_k,
                );
                assert!(
                    (e0 - e1).abs() < 1e-10,
                    "energy drift after roundtrip: {e0} -> {e1} (s={s_cut}, flips={fp},{fq})"
                );
                assert!(eng.state.validate().is_ok());
            }
        }
    }

    #[test]
    fn swap_delta_matches_apply_path() {
        // dense-ish chains so bonds/angles/LJ all contribute
        let mut pos = Vec::new();
        for i in 0..10 {
            pos.push([0.95 * i as f64, 0.0, 0.0]);
        }
        for i in 0..10 {
            pos.push([0.95 * i as f64, 1.05, 0.4]);
        }
        let st = MeltState::new(
            pos,
            vec![(0..10).collect(), (10..20).collect()],
            30.0,
        )
        .unwrap();
        let mut eng = Engine::new(
            st,
            PairParams::default(),
            BondModel::Harmonic { k: 100.0, r0: 1.0 },
            1.5,
            1.0,
            MoveParams::default(),
            vec![(MoveKind::SegmentExchange, 1.0)],
        )
        .unwrap();
        for s_cut in [1, 2, 5, 8, 9] {
            for (fp, fq) in [(false, false), (true, true)] {
                let pr = SwapProposal { a: 0, b: 1, s: s_cut, flip_p: fp, flip_q: fq };
                let (d_analytic, _) = eng.swap_delta(&pr);
                let affected = eng.swap_affected(&pr);
                let e_old = local_energy(
                    &eng.state, &eng.cells, &eng.pair, &eng.bond, eng.angle_k, &affected,
                );
                eng.apply_segment_exchange(&pr);
                let e_new = local_energy(
                    &eng.state, &eng.cells, &eng.pair, &eng.bond, eng.angle_k, &affected,
                );
                let d_apply = e_new - e_old;
                assert!(
                    (d_analytic - d_apply).abs() < 1e-9,
                    "s={s_cut} flips=({fp},{fq}): analytic={d_analytic} apply={d_apply}"
                );
                // revert
                let pr_rev = SwapProposal { flip_p: fq, flip_q: fp, ..pr.clone() };
                eng.apply_segment_exchange(&pr_rev);
            }
        }
    }

    #[test]
    fn swap_twice_is_identity() {
        for (flip_p, flip_q) in [(false, false), (true, false), (false, true), (true, true)] {
            let st = two_chain_state();
            let mut eng = Engine::new(
                st,
                PairParams::default(),
                BondModel::default(),
                0.0,
                1.0,
                MoveParams::default(),
                vec![(MoveKind::SegmentExchange, 1.0)],
            )
            .unwrap();
            let before = eng.state.chains.clone();
            let pr = SwapProposal { a: 0, b: 1, s: 4, flip_p, flip_q };
            eng.apply_segment_exchange(&pr);
            // the reverse swap uses exchanged flips (engine revert path)
            let pr_rev = SwapProposal {
                flip_p: pr.flip_q,
                flip_q: pr.flip_p,
                ..pr.clone()
            };
            eng.apply_segment_exchange(&pr_rev);
            assert_eq!(eng.state.chains, before, "flips=({flip_p},{flip_q})");
            assert!(eng.state.validate().is_ok());
        }
    }
}
