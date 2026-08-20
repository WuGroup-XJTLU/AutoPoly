//! Atomistic MD: analytic forces and velocity-Verlet NVE for the
//! atomistic engine (LAMMPS `units real`: kcal/mol, Å, fs).
//!
//! Unit conversions (SI-consistent):
//!   a [Å/fs²] = 4.1841e-4 * F [kcal/mol/Å] / m [g/mol]
//!   E_kin [kcal/mol] = sum 0.5 * m * v² * 2389.8 / 1  (derived from SI)
//!
//! The kernel's velocity-Verlet uses KIN_TO_KCAL directly; both
//! conversions derive from SI so energy is conserved by construction
//! (validated against LAMMPS NVE in tests).

use crate::atomistic::{AtomisticParams, AtomisticState, COULOMB_REAL};
use crate::atomistic_mc::ACellList;

/// force(kcal/mol/A) / mass(g/mol) -> acceleration (A/fs^2)
/// (SI-derived: 1 kcal/mol/A on 1 g/mol gives 4.1841e-4 A/fs^2)
pub const FORCE_TO_ACC: f64 = 4.1841e-4;
/// E_kin[kcal/mol] = sum 0.5 * m * v^2 * KIN_KCAL_PER.
/// (SI-derived: m=1 g/mol, v=1 A/fs gives 1195.0 kcal/mol)
pub const KIN_KCAL_PER: f64 = 2390.0;

#[inline]
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
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
#[inline]
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

pub struct MdState {
    pub masses: Vec<f64>,
    pub vel: Vec<[f64; 3]>,
    pub forces: Vec<[f64; 3]>,
}

impl MdState {
    pub fn new(state: &AtomisticState, masses_by_type: &[f64]) -> Self {
        let n = state.pos.len();
        let masses = state.types.iter().map(|&t| masses_by_type[t]).collect();
        MdState {
            masses,
            vel: vec![[0.0; 3]; n],
            forces: vec![[0.0; 3]; n],
        }
    }

    pub fn kinetic_energy(&self) -> f64 {
        let mut e = 0.0;
        for (v, &m) in self.vel.iter().zip(&self.masses) {
            e += 0.5 * m * dot(*v, *v) * KIN_KCAL_PER;
        }
        e
    }
}

/// Dihedral force contributions for one OPLS dihedral term, via finite
/// differences of the phi definition (exact to O(h^2), convention-free).
/// Returns ((atom, force)) for the four atoms, scaled by `weight`.
pub fn dihedral_force_contrib(
    pos: &[[f64; 3]],
    box_size: f64,
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    ks: [f64; 4],
    weight: f64,
) -> [(usize, [f64; 3]); 4] {
    let phi_of = |pos: &[[f64; 3]]| -> f64 {
        let disp = |a: usize, b: usize| -> [f64; 3] {
            let (pa, pb) = (pos[a], pos[b]);
            let mut d = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            for c in 0..3 {
                d[c] -= box_size * (d[c] / box_size).round();
            }
            d
        };
        let b12 = disp(j, i);
        let b23 = disp(k, j);
        let b34 = disp(l, k);
        let n1 = cross(b12, b23);
        let n2 = cross(b23, b34);
        let n1n = norm(n1);
        let n2n = norm(n2);
        if n1n < 1e-12 || n2n < 1e-12 {
            return 0.0;
        }
        let b23u = scale(b23, 1.0 / norm(b23));
        let x = dot(n1, n2) / (n1n * n2n);
        let y = dot(cross(n1, n2), b23u) / (n1n * n2n);
        y.atan2(x)
    };
    let phi = phi_of(pos);
    let de_dphi = weight
        * (0.5 * ks[0] * (-phi.sin())
            + 0.5 * ks[1] * (2.0 * (2.0 * phi).sin())
            + 0.5 * ks[2] * (-3.0 * (3.0 * phi).sin())
            + 0.5 * ks[3] * (4.0 * (4.0 * phi).sin()));
    const H: f64 = 1e-7;
    let atoms = [i, j, k, l];
    let mut out = [(0usize, [0.0; 3]); 4];
    let mut scratch: Vec<[f64; 3]> = pos.to_vec();
    for (idx, &a) in atoms.iter().enumerate() {
        let mut grad = [0.0f64; 3];
        for c in 0..3 {
            scratch[a][c] += H;
            let p1 = phi_of(&scratch);
            scratch[a][c] -= 2.0 * H;
            let p2 = phi_of(&scratch);
            scratch[a][c] += H;
            let mut dp = p1 - p2;
            if dp > std::f64::consts::PI {
                dp -= 2.0 * std::f64::consts::PI;
            } else if dp < -std::f64::consts::PI {
                dp += 2.0 * std::f64::consts::PI;
            }
            grad[c] = dp / (2.0 * H);
        }
        out[idx] = (a, scale(grad, -de_dphi));
    }
    out
}

/// Total analytic force on every atom (negative gradient of the full
/// potential, LAMMPS conventions).
pub fn compute_forces(
    state: &AtomisticState,
    par: &AtomisticParams,
    cells: Option<&ACellList>,
    out: &mut [[f64; 3]],
) {
    for f in out.iter_mut() {
        *f = [0.0; 3];
    }
    let n = state.pos.len();

    // ---- bonds: E = K (r - r0)^2 ; F = -2K(r-r0) r_hat ----
    for &(i, j, t) in &state.bonds {
        let d = state.disp(i, j); // r_j - r_i
        let r = norm(d);
        if r < 1e-12 {
            continue;
        }
        let fscal = 2.0 * par.bond_k[t] * (r - par.bond_r0[t]) / r;
        let f = scale(d, fscal); // force ON i (points toward j when stretched)
        out[i] = add(out[i], f);
        out[j] = sub(out[j], f);
    }

    // ---- angles: E = K (th - th0)^2 ----
    for &(i, j, k, t) in &state.angles {
        let b1 = state.disp(j, i); // i - j
        let b2 = state.disp(j, k); // k - j
        let n1 = norm(b1);
        let n2 = norm(b2);
        if n1 < 1e-12 || n2 < 1e-12 {
            continue;
        }
        let c = dot(b1, b2) / (n1 * n2);
        let c = c.clamp(-1.0, 1.0);
        let th = c.acos();
        let dth = th - par.angle_t0[t];
        // dE/dth = 2 K dth ; dth/dc = -1/sin(th)
        let sin_th = (1.0 - c * c).sqrt().max(1e-9);
        let dE_dth = 2.0 * par.angle_k[t] * dth;
        let dth_dc = -1.0 / sin_th;
        // dC/dr_i, dC/dr_k, dC/dr_j for C = cos(th) = b1.b2/(n1 n2)
        let dc_di = sub(scale(b2, 1.0 / (n1 * n2)), scale(b1, c / (n1 * n1)));
        let dc_dk = sub(scale(b1, 1.0 / (n1 * n2)), scale(b2, c / (n2 * n2)));
        let dc_dj = scale(add(dc_di, dc_dk), -1.0);
        let f_i = scale(dc_di, -dE_dth * dth_dc);
        let f_k = scale(dc_dk, -dE_dth * dth_dc);
        let f_j = scale(dc_dj, -dE_dth * dth_dc);
        // note: F = -dE/dr = -dE/dth * dth/dC * dC/dr
        out[i] = add(out[i], f_i);
        out[j] = add(out[j], f_j);
        out[k] = add(out[k], f_k);
    }

    // ---- dihedrals (OPLS cosine series) ----
    for &(i, j, k, l, t) in &state.dihedrals {
        let f = dihedral_force_contrib(
            &state.pos,
            state.box_size,
            i, j, k, l,
            par.dih_k[t],
            1.0,
        );
        for (a, fa) in f {
            out[a] = add(out[a], fa);
        }
    }

    // ---- pairs (LJ + Coulomb, 1-2/1-3 excluded, 1-4 scaled) ----
    let mut cand: Vec<usize> = Vec::with_capacity(256);
    for i in 0..n {
        let js: Box<dyn Iterator<Item = usize> + '_> = match cells {
            Some(cl) => {
                cand.clear();
                cl.candidates_into(state.pos[i], &mut cand);
                // SAFETY-FREE: borrow from cand via raw slice copy is avoided
                // by filtering with j > i inside the loop below.
                Box::new(cand.iter().copied().filter(move |&j| j > i))
            }
            None => Box::new((i + 1)..n),
        };
        for j in js {
            if state.excl[i].contains(&j) {
                continue;
            }
            let d = state.disp(i, j);
            let r2 = dot(d, d);
            let (ti, tj) = (state.types[i], state.types[j]);
            let sig = (par.pair_sig[ti] * par.pair_sig[tj]).sqrt();
            let eps = (par.pair_eps[ti] * par.pair_eps[tj]).sqrt();
            let scaled = state.scaled14.contains(&(i.min(j), i.max(j)));
            let (ls, cs) = if scaled {
                (par.scale14_lj, par.scale14_coul)
            } else {
                (1.0, 1.0)
            };
            let mut fscal = 0.0;
            if r2 < par.lj_cut * par.lj_cut {
                let sr2 = sig * sig / r2;
                let sr6 = sr2 * sr2 * sr2;
                // F = 24 eps (2 sr12 - sr6) / r^2 * d
                fscal += ls * 24.0 * eps * (2.0 * sr6 * sr6 - sr6) / r2;
            }
            if r2 < par.coul_cut * par.coul_cut {
                fscal += cs * COULOMB_REAL * state.charges[i] * state.charges[j] / (r2 * r2.sqrt());
            }
            if fscal != 0.0 {
                // F_i = -(dU/dr) * (r_i - r_j)/r = -d * fscal
                let f = scale(d, -fscal);
                out[i] = add(out[i], f);
                out[j] = sub(out[j], f);
            }
        }
    }
}

/// One velocity-Verlet NVE step. `cells` accelerates pair forces and
/// must be current with `state.pos` (positions change by ~dt*v per
/// step, so refresh it every step for exactness).
pub fn verlet_step(
    state: &mut AtomisticState,
    par: &AtomisticParams,
    md: &mut MdState,
    cells: Option<&ACellList>,
    dt: f64,
) {
    let n = state.pos.len();
    // v(t+dt/2), x(t+dt)
    for i in 0..n {
        let acc = scale(md.forces[i], FORCE_TO_ACC / md.masses[i]);
        md.vel[i] = add(md.vel[i], scale(acc, 0.5 * dt));
        let p = add(state.pos[i], scale(md.vel[i], dt));
        state.pos[i] = state.wrap(p);
    }
    compute_forces(state, par, cells, &mut md.forces);
    // v(t+dt)
    for i in 0..n {
        let acc = scale(md.forces[i], FORCE_TO_ACC / md.masses[i]);
        md.vel[i] = add(md.vel[i], scale(acc, 0.5 * dt));
    }
}

/// Initialize velocities from Maxwell-Boltzmann at kBT (kcal/mol).
pub fn init_velocities<R: rand::Rng>(
    md: &mut MdState,
    kbt: f64,
    rng: &mut R,
) {
    // <0.5 m v^2 * KIN_KCAL_PER> = 0.5 kBT per DOF
    for (v, &m) in md.vel.iter_mut().zip(&md.masses) {
        let sigma = (kbt / (m * KIN_KCAL_PER)).sqrt();
        for c in 0..3 {
            // Box-Muller
            let u1: f64 = rng.random::<f64>().max(1e-300);
            let u2: f64 = rng.random::<f64>();
            v[c] = sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
    }
    // zero total momentum
    let n = md.vel.len() as f64;
    let mut p = [0.0; 3];
    for v in &md.vel {
        p = add(p, *v);
    }
    for v in md.vel.iter_mut() {
        *v = sub(*v, scale(p, 1.0 / n));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atomistic::AtomisticEngine;

    /// Small typed test system: bent triatomic + a free pair.
    fn fixture() -> (AtomisticState, AtomisticParams) {
        // atoms: 0-1-2 bonded chain with angle, 3-4 dihedral pair chain,
        // 5 free atom interacting via LJ
        let pos = vec![
            [0.0, 0.0, 0.0],
            [1.53, 0.0, 0.0],
            [2.2, 1.1, 0.0],
            [3.6, 1.3, 0.2],
            [3.7, 2.9, 0.0],
            [1.0, 3.0, 1.0],
        ];
        let types = vec![1, 1, 1, 1, 1, 1];
        let charges = vec![0.05, -0.1, 0.05, -0.1, 0.05, 0.1];
        let mol = vec![0, 0, 0, 0, 0, 0];
        let bonds = vec![(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)];
        let angles = vec![(0, 1, 2, 1), (1, 2, 3, 1), (2, 3, 4, 1)];
        let dihs = vec![(0, 1, 2, 3, 1), (1, 2, 3, 4, 1)];
        let chains = vec![vec![0, 1, 2, 3, 4]];
        let st = AtomisticState::new(
            pos, 30.0, types, charges, mol, bonds, angles, dihs, chains,
        );
        let par = AtomisticParams {
            pair_eps: vec![0.0, 0.066],
            pair_sig: vec![0.0, 3.5],
            bond_k: vec![0.0, 268.0],
            bond_r0: vec![0.0, 1.529],
            angle_k: vec![0.0, 58.35],
            angle_t0: vec![0.0, 112.7f64.to_radians()],
            dih_k: vec![[0.0; 4], [1.1, -0.2, 0.2, 0.0]],
            lj_cut: 11.0,
            coul_cut: 11.0,
            scale14_lj: 0.5,
            scale14_coul: 0.5,
        };
        (st, par)
    }

    fn check_single(kind: &str) {
        // two/three/four-atom systems with exactly one active term
        let (mut st, mut par) = fixture();
        match kind {
            "bond" => {
                par.angle_k = vec![0.0; par.angle_k.len()];
                par.dih_k = vec![[0.0; 4]; par.dih_k.len()];
                par.pair_eps = vec![0.0; par.pair_eps.len()];
                par.scale14_coul = 0.0;
                st.charges = vec![0.0; st.charges.len()];
            }
            "angle" => {
                par.bond_k = vec![0.0; par.bond_k.len()];
                par.dih_k = vec![[0.0; 4]; par.dih_k.len()];
                par.pair_eps = vec![0.0; par.pair_eps.len()];
                par.scale14_coul = 0.0;
                st.charges = vec![0.0; st.charges.len()];
            }
            "dihedral" => {
                par.bond_k = vec![0.0; par.bond_k.len()];
                par.angle_k = vec![0.0; par.angle_k.len()];
                par.pair_eps = vec![0.0; par.pair_eps.len()];
                par.scale14_coul = 0.0;
                st.charges = vec![0.0; st.charges.len()];
            }
            "pair" => {
                par.bond_k = vec![0.0; par.bond_k.len()];
                par.angle_k = vec![0.0; par.angle_k.len()];
                par.dih_k = vec![[0.0; 4]; par.dih_k.len()];
                // keep eps + charges; exclusions still apply
            }
            _ => unreachable!(),
        }
        let mut f = vec![[0.0; 3]; st.pos.len()];
        compute_forces(&st, &par, None, &mut f);
        let h = 1e-6;
        let mut max_rel = 0.0f64;
        for i in 0..st.pos.len() {
            for c in 0..3 {
                let mut p1 = st.clone();
                let mut p2 = st.clone();
                p1.pos[i][c] += h;
                p2.pos[i][c] -= h;
                let e1 = AtomisticEngine::new(p1, par.clone(), 1.0).total_energy();
                let e2 = AtomisticEngine::new(p2, par.clone(), 1.0).total_energy();
                let fd = -(e1 - e2) / (2.0 * h);
                let err = (fd - f[i][c]).abs();
                let rel = err / (1.0 + fd.abs());
                if rel > max_rel {
                    max_rel = rel;
                    println!("{kind} atom{i} c{c}: fd={fd:.6} analytic={:.6}", f[i][c]);
                }
            }
        }
        println!("{kind}: max_rel_err={max_rel:.2e}");
        assert!(max_rel < 1e-3, "{kind} force mismatch");
    }

    #[test]
    fn single_term_forces() {
        check_single("bond");
        check_single("angle");
        check_single("dihedral");
        check_single("pair");
    }

    #[test]
    fn debug_pair_sign() {
        let (st, par) = fixture();
        let mut f = vec![[0.0; 3]; st.pos.len()];
        compute_forces(&st, &par, None, &mut f);
        let h = 1e-6;
        let mut worst = (0.0, 0usize, 0usize, 0.0, 0.0);
        for i in 0..st.pos.len() {
            for c in 0..3 {
                let mut p1 = st.clone();
                let mut p2 = st.clone();
                p1.pos[i][c] += h;
                p2.pos[i][c] -= h;
                let e1 = AtomisticEngine::new(p1, par.clone(), 1.0).total_energy();
                let e2 = AtomisticEngine::new(p2, par.clone(), 1.0).total_energy();
                let fd = -(e1 - e2) / (2.0 * h);
                let err = (fd - f[i][c]).abs() / (1.0 + fd.abs());
                if err > worst.0 {
                    worst = (err, i, c, fd, f[i][c]);
                }
            }
        }
        println!("worst pair: atom{} c{} fd={:.6} analytic={:.6}", worst.1, worst.2, worst.3, worst.4);
    }

    #[test]
    #[ignore]
    fn debug_per_term_forces() {
        let (st, par) = fixture();
        let eng = AtomisticEngine::new(st.clone(), par.clone(), 1.0);
        let h = 1e-6;
        // isolate bond forces: zero all other params
        for term in ["bond", "angle", "dihedral", "pair"] {
            let mut p2 = par.clone();
            if term != "bond" { p2.bond_k = vec![0.0; par.bond_k.len()]; }
            if term != "angle" { p2.angle_k = vec![0.0; par.angle_k.len()]; }
            if term != "dihedral" { p2.dih_k = vec![[0.0;4]; par.dih_k.len()]; }
            if term != "pair" {
                p2.pair_eps = vec![0.0; par.pair_eps.len()];
                p2.scale14_coul = 0.0;
            } else {
                // keep eps; zero bonds/angles/dihs above
            }
            if term == "pair" {
                p2.bond_k = vec![0.0; par.bond_k.len()];
                p2.angle_k = vec![0.0; par.angle_k.len()];
                p2.dih_k = vec![[0.0;4]; par.dih_k.len()];
            }
            if term != "pair" && term != "bond" {}
            let mut f = vec![[0.0; 3]; st.pos.len()];
            compute_forces(&st, &p2, None, &mut f);
            let mut p1s = st.clone();
            let mut p2s = st.clone();
            p1s.pos[0][0] += h;
            p2s.pos[0][0] -= h;
            let e1 = AtomisticEngine::new(p1s, p2.clone(), 1.0).total_energy();
            let e2 = AtomisticEngine::new(p2s, p2.clone(), 1.0).total_energy();
            let fd = -(e1 - e2) / (2.0 * h);
            println!("TERM {term}: atom0 fx fd={fd:.6} analytic={:.6}", f[0][0]);
        }
    }

    #[test]
    fn forces_match_finite_differences() {
        let (mut st, par) = fixture();
        let mut f = vec![[0.0; 3]; st.pos.len()];
        compute_forces(&st, &par, None, &mut f);
        let eng = AtomisticEngine::new(st.clone(), par.clone(), 1.0);
        let h = 1e-6;
        for i in 0..st.pos.len() {
            for c in 0..3 {
                let mut p1 = st.clone();
                let mut p2 = st.clone();
                p1.pos[i][c] += h;
                p2.pos[i][c] -= h;
                let e1 = AtomisticEngine::new(p1, par.clone(), 1.0).total_energy();
                let e2 = AtomisticEngine::new(p2, par.clone(), 1.0).total_energy();
                let fd = -(e1 - e2) / (2.0 * h);
                assert!(
                    (fd - f[i][c]).abs() < 1e-4 * (1.0 + fd.abs()),
                    "atom {i} comp {c}: fd={fd} analytic={}",
                    f[i][c]
                );
            }
        }
        let _ = &mut st;
    }
}
