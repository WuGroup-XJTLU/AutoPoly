//! Nonequilibrium Candidate Monte Carlo (NCMC) double-bridging.
//!
//! The Hamiltonian is switched gradually between the two topologies:
//!
//!   U(x, λ) = U_common(x) + (1-λ) U_oldonly(x) + λ U_newonly(x)
//!
//! with pair classifications (1-2/1-3 excluded, 1-4 scaled) interpolated
//! in λ. Velocity Verlet propagates between switches; acceptance uses
//! the accumulated nonequilibrium work W = Σ [U(x,λ_{k+1}) − U(x,λ_k)].

use crate::atomistic::{AtomisticParams, AtomisticState, COULOMB_REAL};
use crate::atomistic_md::{dihedral_force_contrib, FORCE_TO_ACC, KIN_KCAL_PER};
use rand::Rng;
use std::collections::{HashMap, HashSet};

type BondT = (usize, usize, usize);
type AngleT = (usize, usize, usize, usize);
type DihT = (usize, usize, usize, usize, usize);

fn split3(a: &[BondT], b: &[BondT]) -> (Vec<BondT>, Vec<BondT>, Vec<BondT>) {
    let sa: HashSet<BondT> = a.iter().copied().collect();
    let sb: HashSet<BondT> = b.iter().copied().collect();
    (
        sa.intersection(&sb).copied().collect(),
        sa.difference(&sb).copied().collect(),
        sb.difference(&sa).copied().collect(),
    )
}
fn split4(a: &[AngleT], b: &[AngleT]) -> (Vec<AngleT>, Vec<AngleT>, Vec<AngleT>) {
    let sa: HashSet<AngleT> = a.iter().copied().collect();
    let sb: HashSet<AngleT> = b.iter().copied().collect();
    (
        sa.intersection(&sb).copied().collect(),
        sa.difference(&sb).copied().collect(),
        sb.difference(&sa).copied().collect(),
    )
}
fn split5(a: &[DihT], b: &[DihT]) -> (Vec<DihT>, Vec<DihT>, Vec<DihT>) {
    let sa: HashSet<DihT> = a.iter().copied().collect();
    let sb: HashSet<DihT> = b.iter().copied().collect();
    (
        sa.intersection(&sb).copied().collect(),
        sa.difference(&sb).copied().collect(),
        sb.difference(&sa).copied().collect(),
    )
}

/// λ-decomposed term sets for one bridge proposal.
pub struct NcmcTerms {
    types: Vec<usize>,
    charges: Vec<f64>,
    common_bonds: Vec<BondT>,
    old_bonds: Vec<BondT>,
    new_bonds: Vec<BondT>,
    common_angles: Vec<AngleT>,
    old_angles: Vec<AngleT>,
    new_angles: Vec<AngleT>,
    common_dihs: Vec<DihT>,
    old_dihs: Vec<DihT>,
    new_dihs: Vec<DihT>,
    old_excl: Vec<HashSet<usize>>,
    new_excl: Vec<HashSet<usize>>,
    old_sc14: HashSet<(usize, usize)>,
    new_sc14: HashSet<(usize, usize)>,
}

impl NcmcTerms {
    pub fn build(old: &AtomisticState, new: &AtomisticState) -> Self {
        let (cb, ob, nb) = split3(&old.bonds, &new.bonds);
        let (ca, oa, na) = split4(&old.angles, &new.angles);
        let (cd, od, nd) = split5(&old.dihedrals, &new.dihedrals);
        NcmcTerms {
            types: old.types.clone(),
            charges: old.charges.clone(),
            common_bonds: cb,
            old_bonds: ob,
            new_bonds: nb,
            common_angles: ca,
            old_angles: oa,
            new_angles: na,
            common_dihs: cd,
            old_dihs: od,
            new_dihs: nd,
            old_excl: old.excl.clone(),
            new_excl: new.excl.clone(),
            old_sc14: old.scaled14.clone(),
            new_sc14: new.scaled14.clone(),
        }
    }

    #[inline]
    fn pair_factors(&self, i: usize, j: usize) -> (f64, f64, f64, f64) {
        let key = (i.min(j), i.max(j));
        let (ol, oc) = if self.old_excl[i].contains(&j) {
            (0.0, 0.0)
        } else if self.old_sc14.contains(&key) {
            (0.5, 0.5)
        } else {
            (1.0, 1.0)
        };
        let (nl, nc) = if self.new_excl[i].contains(&j) {
            (0.0, 0.0)
        } else if self.new_sc14.contains(&key) {
            (0.5, 0.5)
        } else {
            (1.0, 1.0)
        };
        (ol, oc, nl, nc)
    }

    #[inline]
    fn pair_active(&self, i: usize, j: usize) -> bool {
        !(self.old_excl[i].contains(&j) && self.new_excl[i].contains(&j))
    }

    // ------------------------------------------------------------------
    // λ-switched energy
    // ------------------------------------------------------------------
    pub fn energy_at_lambda(
        &self,
        pos: &[[f64; 3]],
        box_size: f64,
        par: &AtomisticParams,
        lambda: f64,
    ) -> f64 {
        let l_old = 1.0 - lambda;
        let l_new = lambda;
        let disp = |i: usize, j: usize| -> [f64; 3] {
            let (a, b) = (pos[i], pos[j]);
            let mut d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            for c in 0..3 {
                d[c] -= box_size * (d[c] / box_size).round();
            }
            d
        };
        let nrm = |d: [f64; 3]| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let mut e = 0.0;

        for &(i, j, t) in &self.common_bonds {
            let r = nrm(disp(i, j));
            let dr = r - par.bond_r0[t];
            e += par.bond_k[t] * dr * dr;
        }
        for (w, list) in [(l_old, &self.old_bonds), (l_new, &self.new_bonds)] {
            for &(i, j, t) in list {
                let r = nrm(disp(i, j));
                let dr = r - par.bond_r0[t];
                e += w * par.bond_k[t] * dr * dr;
            }
        }

        let angle_e = |i: usize, j: usize, k: usize, t: usize| -> f64 {
            let b1 = disp(j, i);
            let b2 = disp(j, k);
            let n1 = nrm(b1);
            let n2 = nrm(b2);
            if n1 < 1e-12 || n2 < 1e-12 {
                return 0.0;
            }
            let c = ((b1[0] * b2[0] + b1[1] * b2[1] + b1[2] * b2[2]) / (n1 * n2))
                .clamp(-1.0, 1.0);
            let th = c.acos();
            let dth = th - par.angle_t0[t];
            par.angle_k[t] * dth * dth
        };
        for &(i, j, k, t) in &self.common_angles {
            e += angle_e(i, j, k, t);
        }
        for (w, list) in [(l_old, &self.old_angles), (l_new, &self.new_angles)] {
            for &(i, j, k, t) in list {
                e += w * angle_e(i, j, k, t);
            }
        }

        let dih_e = |i: usize, j: usize, k: usize, l: usize, t: usize| -> f64 {
            let b12 = disp(j, i);
            let b23 = disp(k, j);
            let b34 = disp(l, k);
            let n1 = cross(b12, b23);
            let n2 = cross(b23, b34);
            let n1n = nrm(n1);
            let n2n = nrm(n2);
            if n1n < 1e-12 || n2n < 1e-12 {
                return 0.0;
            }
            let b23n = nrm(b23);
            let b23u = [b23[0] / b23n, b23[1] / b23n, b23[2] / b23n];
            let x = dot(n1, n2) / (n1n * n2n);
            let y = dot(cross(n1, n2), b23u) / (n1n * n2n);
            let phi = y.atan2(x);
            let ks = par.dih_k[t];
            0.5 * ks[0] * (1.0 + phi.cos())
                + 0.5 * ks[1] * (1.0 - (2.0 * phi).cos())
                + 0.5 * ks[2] * (1.0 + (3.0 * phi).cos())
                + 0.5 * ks[3] * (1.0 - (4.0 * phi).cos())
        };
        for &(i, j, k, l, t) in &self.common_dihs {
            e += dih_e(i, j, k, l, t);
        }
        for (w, list) in [(l_old, &self.old_dihs), (l_new, &self.new_dihs)] {
            for &(i, j, k, l, t) in list {
                e += w * dih_e(i, j, k, l, t);
            }
        }

        // pairs
        let n = pos.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if !self.pair_active(i, j) {
                    continue;
                }
                let (ol, oc, nl, nc) = self.pair_factors(i, j);
                let w_lj = l_old * ol + l_new * nl;
                let w_coul = l_old * oc + l_new * nc;
                if w_lj == 0.0 && w_coul == 0.0 {
                    continue;
                }
                let d = disp(i, j);
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let (ti, tj) = (self.types[i], self.types[j]);
                let sig = (par.pair_sig[ti] * par.pair_sig[tj]).sqrt();
                let eps = (par.pair_eps[ti] * par.pair_eps[tj]).sqrt();
                if w_lj != 0.0 && r2 < par.lj_cut * par.lj_cut {
                    let sr2 = sig * sig / r2;
                    let sr6 = sr2 * sr2 * sr2;
                    e += w_lj * 4.0 * eps * (sr6 * sr6 - sr6);
                }
                if w_coul != 0.0 && r2 < par.coul_cut * par.coul_cut {
                    e += w_coul * COULOMB_REAL * self.charges[i] * self.charges[j] / r2.sqrt();
                }
            }
        }
        e
    }

    // ------------------------------------------------------------------
    // λ-switched forces
    // ------------------------------------------------------------------
    pub fn forces_at_lambda(
        &self,
        pos: &[[f64; 3]],
        box_size: f64,
        par: &AtomisticParams,
        lambda: f64,
        out: &mut [[f64; 3]],
    ) {
        for f in out.iter_mut() {
            *f = [0.0; 3];
        }
        let l_old = 1.0 - lambda;
        let l_new = lambda;
        let disp = |i: usize, j: usize| -> [f64; 3] {
            let (a, b) = (pos[i], pos[j]);
            let mut d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            for c in 0..3 {
                d[c] -= box_size * (d[c] / box_size).round();
            }
            d
        };
        let nrm = |d: [f64; 3]| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();

        // bonds
        let bond_f = |i: usize, j: usize, t: usize, w: f64, out: &mut [[f64; 3]]| {
            let d = disp(i, j);
            let r = nrm(d);
            if r < 1e-12 {
                return;
            }
            let fs = w * 2.0 * par.bond_k[t] * (r - par.bond_r0[t]) / r;
            let f = [d[0] * fs, d[1] * fs, d[2] * fs];
            out[i][0] += f[0];
            out[i][1] += f[1];
            out[i][2] += f[2];
            out[j][0] -= f[0];
            out[j][1] -= f[1];
            out[j][2] -= f[2];
        };
        for &(i, j, t) in &self.common_bonds {
            bond_f(i, j, t, 1.0, out);
        }
        for &(i, j, t) in &self.old_bonds {
            bond_f(i, j, t, l_old, out);
        }
        for &(i, j, t) in &self.new_bonds {
            bond_f(i, j, t, l_new, out);
        }

        // angles
        let angle_f = |i: usize, j: usize, k: usize, t: usize, w: f64, out: &mut [[f64; 3]]| {
            let b1 = disp(j, i);
            let b2 = disp(j, k);
            let n1 = nrm(b1);
            let n2 = nrm(b2);
            if n1 < 1e-12 || n2 < 1e-12 {
                return;
            }
            let c = (dot(b1, b2) / (n1 * n2)).clamp(-1.0, 1.0);
            let th = c.acos();
            let dth = th - par.angle_t0[t];
            let sin_th = (1.0 - c * c).sqrt().max(1e-9);
            let de_dth = w * 2.0 * par.angle_k[t] * dth;
            let dth_dc = -1.0 / sin_th;
            let dc_di = sub(
                [b2[0] / (n1 * n2), b2[1] / (n1 * n2), b2[2] / (n1 * n2)],
                [b1[0] * c / (n1 * n1), b1[1] * c / (n1 * n1), b1[2] * c / (n1 * n1)],
            );
            let dc_dk = sub(
                [b1[0] / (n1 * n2), b1[1] / (n1 * n2), b1[2] / (n1 * n2)],
                [b2[0] * c / (n2 * n2), b2[1] * c / (n2 * n2), b2[2] * c / (n2 * n2)],
            );
            let dc_dj = [-dc_di[0] - dc_dk[0], -dc_di[1] - dc_dk[1], -dc_di[2] - dc_dk[2]];
            let s = -de_dth * dth_dc;
            for c in 0..3 {
                out[i][c] += dc_di[c] * s;
                out[j][c] += dc_dj[c] * s;
                out[k][c] += dc_dk[c] * s;
            }
        };
        for &(i, j, k, t) in &self.common_angles {
            angle_f(i, j, k, t, 1.0, out);
        }
        for &(i, j, k, t) in &self.old_angles {
            angle_f(i, j, k, t, l_old, out);
        }
        for &(i, j, k, t) in &self.new_angles {
            angle_f(i, j, k, t, l_new, out);
        }

        // dihedrals (shared FD-of-phi helper)
        for &(i, j, k, l, t) in &self.common_dihs {
            let f = dihedral_force_contrib(pos, box_size, i, j, k, l, par.dih_k[t], 1.0);
            for (a, fa) in f {
                out[a] = add(out[a], fa);
            }
        }
        for &(i, j, k, l, t) in &self.old_dihs {
            let f = dihedral_force_contrib(pos, box_size, i, j, k, l, par.dih_k[t], l_old);
            for (a, fa) in f {
                out[a] = add(out[a], fa);
            }
        }
        for &(i, j, k, l, t) in &self.new_dihs {
            let f = dihedral_force_contrib(pos, box_size, i, j, k, l, par.dih_k[t], l_new);
            for (a, fa) in f {
                out[a] = add(out[a], fa);
            }
        }

        // pairs
        let n = pos.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if !self.pair_active(i, j) {
                    continue;
                }
                let (ol, oc, nl, nc) = self.pair_factors(i, j);
                let w_lj = l_old * ol + l_new * nl;
                let w_coul = l_old * oc + l_new * nc;
                if w_lj == 0.0 && w_coul == 0.0 {
                    continue;
                }
                let d = disp(i, j);
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let (ti, tj) = (self.types[i], self.types[j]);
                let sig = (par.pair_sig[ti] * par.pair_sig[tj]).sqrt();
                let eps = (par.pair_eps[ti] * par.pair_eps[tj]).sqrt();
                let mut fscal = 0.0;
                if w_lj != 0.0 && r2 < par.lj_cut * par.lj_cut {
                    let sr2 = sig * sig / r2;
                    let sr6 = sr2 * sr2 * sr2;
                    fscal += w_lj * 24.0 * eps * (2.0 * sr6 * sr6 - sr6) / r2;
                }
                if w_coul != 0.0 && r2 < par.coul_cut * par.coul_cut {
                    fscal += w_coul * COULOMB_REAL * self.charges[i] * self.charges[j]
                        / (r2 * r2.sqrt());
                }
                if fscal != 0.0 {
                    let f = [-d[0] * fscal, -d[1] * fscal, -d[2] * fscal];
                    out[i][0] += f[0];
                    out[i][1] += f[1];
                    out[i][2] += f[2];
                    out[j][0] -= f[0];
                    out[j][1] -= f[1];
                    out[j][2] -= f[2];
                }
            }
        }
    }
}

#[inline]
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ----------------------------------------------------------------------
// NCMC driver
// ----------------------------------------------------------------------

pub struct NcmcResult {
    pub accepted: bool,
    pub work: f64,
    pub n_fwd: usize,
    pub n_rev: usize,
}

/// One NCMC bridge attempt. `positions` and `velocities` are updated in
/// place (on acceptance: healed new-topology state; on rejection: the
/// old state restored from `old_state_backup`).
#[allow(clippy::too_many_arguments)]
pub fn ncmc_attempt<R: Rng>(
    old_state: &AtomisticState,
    new_state: &AtomisticState,
    par: &AtomisticParams,
    masses: &[f64],
    kbt: f64,
    n_switch: usize,
    steps_per_switch: usize,
    dt: f64,
    rng: &mut R,
) -> (NcmcResult, Vec<[f64; 3]>, Vec<[f64; 3]>, f64) {
    let terms = NcmcTerms::build(old_state, new_state);
    let box_size = old_state.box_size;
    let n = old_state.pos.len();

    // refresh momenta
    let mut pos = old_state.pos.clone();
    let mut vel: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let sigma = (kbt / (masses[old_state.types[i]] * KIN_KCAL_PER)).sqrt();
            let g = |rng: &mut R| -> f64 {
                let u1: f64 = rng.random::<f64>().max(1e-300);
                let u2: f64 = rng.random::<f64>();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            };
            [sigma * g(rng), sigma * g(rng), sigma * g(rng)]
        })
        .collect();
    // zero total momentum
    let mut p = [0.0; 3];
    for v in &vel {
        p = add(p, *v);
    }
    for v in vel.iter_mut() {
        *v = sub(*v, [p[0] / n as f64, p[1] / n as f64, p[2] / n as f64]);
    }

    let mut work = 0.0;
    let mut forces = vec![[0.0; 3]; n];
    let mut lambda_prev = 0.0;
    let wrap = |p: [f64; 3]| -> [f64; 3] {
        [
            p[0] - box_size * (p[0] / box_size).floor(),
            p[1] - box_size * (p[1] / box_size).floor(),
            p[2] - box_size * (p[2] / box_size).floor(),
        ]
    };

    for k in 1..=n_switch {
        let lambda = k as f64 / n_switch as f64;
        // work accumulated at fixed positions
        work += terms.energy_at_lambda(&pos, box_size, par, lambda)
            - terms.energy_at_lambda(&pos, box_size, par, lambda_prev);
        lambda_prev = lambda;
        // propagate at fixed lambda
        terms.forces_at_lambda(&pos, box_size, par, lambda, &mut forces);
        for _ in 0..steps_per_switch {
            for i in 0..n {
                let m = masses[old_state.types[i]];
                let acc = [
                    forces[i][0] * FORCE_TO_ACC / m,
                    forces[i][1] * FORCE_TO_ACC / m,
                    forces[i][2] * FORCE_TO_ACC / m,
                ];
                vel[i] = [
                    vel[i][0] + 0.5 * dt * acc[0],
                    vel[i][1] + 0.5 * dt * acc[1],
                    vel[i][2] + 0.5 * dt * acc[2],
                ];
                pos[i] = wrap([
                    pos[i][0] + vel[i][0] * dt,
                    pos[i][1] + vel[i][1] * dt,
                    pos[i][2] + vel[i][2] * dt,
                ]);
            }
            terms.forces_at_lambda(&pos, box_size, par, lambda, &mut forces);
            for i in 0..n {
                let m = masses[old_state.types[i]];
                vel[i] = [
                    vel[i][0] + 0.5 * dt * forces[i][0] * FORCE_TO_ACC / m,
                    vel[i][1] + 0.5 * dt * forces[i][1] * FORCE_TO_ACC / m,
                    vel[i][2] + 0.5 * dt * forces[i][2] * FORCE_TO_ACC / m,
                ];
            }
        }
    }

    // NCMC acceptance: P = min(1, exp(-W / kBT))
    let accept_prob = if work <= 0.0 { 1.0 } else { (-work / kbt).exp() };
    let accepted = work <= 0.0 || rng.random::<f64>() < accept_prob;
    (
        NcmcResult {
            accepted,
            work,
            n_fwd: 0,
            n_rev: 0,
        },
        pos,
        vel,
        accept_prob,
    )
}
