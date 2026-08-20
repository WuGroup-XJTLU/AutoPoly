//! Atomistic force-field engine (OPLS-AA/GAFF subset emitted by AutoPoly).
//!
//! Conventions follow LAMMPS `units real`:
//! * bonds:     E = K (r - r0)^2            (harmonic, no 1/2)
//! * angles:    E = K (theta - theta0)^2    (harmonic, radians)
//! * dihedrals: E = sum_n 0.5 K_n (1 + (-1)^{n+1} cos(n phi))  (opls)
//! * pairs:     LJ (geometric mixing) + Coulomb real-space cutoff,
//!              1-2 and 1-3 excluded, 1-4 scaled by (scale14_lj/coul)
//! * Coulomb constant: 332.06371 kcal A / (mol e^2)

use std::collections::{HashMap, HashSet};

pub const COULOMB_REAL: f64 = 332.06371; // kcal*A/(mol*e^2), units real

#[derive(Clone)]
pub struct AtomisticParams {
    pub pair_eps: Vec<f64>, // 1-indexed by atom type
    pub pair_sig: Vec<f64>,
    pub bond_k: Vec<f64>,   // 1-indexed by bond type
    pub bond_r0: Vec<f64>,
    pub angle_k: Vec<f64>,  // 1-indexed by angle type
    pub angle_t0: Vec<f64>, // radians
    pub dih_k: Vec<[f64; 4]>, // 1-indexed by dihedral type, OPLS K1..K4
    pub lj_cut: f64,
    pub coul_cut: f64,
    pub scale14_lj: f64,
    pub scale14_coul: f64,
}

#[derive(Clone)]
pub struct AtomisticState {
    pub pos: Vec<[f64; 3]>,
    pub box_size: f64,
    pub types: Vec<usize>,
    pub charges: Vec<f64>,
    pub mol: Vec<usize>,
    pub bonds: Vec<(usize, usize, usize)>,
    pub angles: Vec<(usize, usize, usize, usize)>,
    pub dihedrals: Vec<(usize, usize, usize, usize, usize)>,
    /// Per-atom bonded-term index lists (for local energy gathering).
    pub bonds_of: Vec<Vec<usize>>,
    pub angles_of: Vec<Vec<usize>>,
    pub dihedrals_of: Vec<Vec<usize>>,
    /// Pair classification: 1-2 and 1-3 excluded, 1-4 scaled.
    pub excl: Vec<HashSet<usize>>,
    pub scaled14: HashSet<(usize, usize)>,
    /// Backbone chains (ordered heavy-atom contour per molecule).
    pub chains: Vec<Vec<usize>>,
}

impl AtomisticState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pos: Vec<[f64; 3]>,
        box_size: f64,
        types: Vec<usize>,
        charges: Vec<f64>,
        mol: Vec<usize>,
        bonds: Vec<(usize, usize, usize)>,
        angles: Vec<(usize, usize, usize, usize)>,
        dihedrals: Vec<(usize, usize, usize, usize, usize)>,
        chains: Vec<Vec<usize>>,
    ) -> Self {
        let n = pos.len();
        let mut bonds_of = vec![Vec::new(); n];
        let mut angles_of = vec![Vec::new(); n];
        let mut dihedrals_of = vec![Vec::new(); n];
        let mut excl: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        let mut scaled14: HashSet<(usize, usize)> = HashSet::new();

        for (t, &(i, j, _)) in bonds.iter().enumerate() {
            bonds_of[i].push(t);
            bonds_of[j].push(t);
            excl[i].insert(j);
            excl[j].insert(i);
        }
        for (t, &(i, j, k, _)) in angles.iter().enumerate() {
            angles_of[i].push(t);
            angles_of[j].push(t);
            angles_of[k].push(t);
            // 1-3 pair: angle endpoints
            excl[i].insert(k);
            excl[k].insert(i);
        }
        for (t, &(i, j, k, l, _)) in dihedrals.iter().enumerate() {
            dihedrals_of[i].push(t);
            dihedrals_of[j].push(t);
            dihedrals_of[k].push(t);
            dihedrals_of[l].push(t);
            // 1-4 pair: dihedral endpoints
            scaled14.insert((i.min(l), i.max(l)));
        }
        // a pair that is both 1-3 and 1-4 (cyclic) stays excluded (excl wins)
        AtomisticState {
            pos,
            box_size,
            types,
            charges,
            mol,
            bonds,
            angles,
            dihedrals,
            bonds_of,
            angles_of,
            dihedrals_of,
            excl,
            scaled14,
            chains,
        }
    }

    #[inline]
    pub fn disp(&self, i: usize, j: usize) -> [f64; 3] {
        let l = self.box_size;
        let (a, b) = (self.pos[i], self.pos[j]);
        let mut d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        for c in 0..3 {
            d[c] -= l * (d[c] / l).round();
        }
        d
    }

    #[inline]
    pub fn wrap(&self, p: [f64; 3]) -> [f64; 3] {
        let l = self.box_size;
        [
            p[0] - l * (p[0] / l).floor(),
            p[1] - l * (p[1] / l).floor(),
            p[2] - l * (p[2] / l).floor(),
        ]
    }
}

#[inline]
fn angle_of(b1: [f64; 3], b2: [f64; 3]) -> f64 {
    let n1 = (b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2]).sqrt();
    let n2 = (b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2]).sqrt();
    if n1 < 1e-12 || n2 < 1e-12 {
        return 0.0;
    }
    let c = ((b1[0] * b2[0] + b1[1] * b2[1] + b1[2] * b2[2]) / (n1 * n2))
        .clamp(-1.0, 1.0);
    c.acos()
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
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
pub fn norm_pub(a: [f64; 3]) -> f64 {
    norm(a)
}

/// Rodrigues rotation with precomputed cos/sin.
#[inline]
pub fn rotate_pub(v: [f64; 3], u: [f64; 3], ct: f64, st: f64) -> [f64; 3] {
    let kv = dot(u, v);
    let cx = cross(u, v);
    [
        v[0] * ct + cx[0] * st + u[0] * kv * (1.0 - ct),
        v[1] * ct + cx[1] * st + u[1] * kv * (1.0 - ct),
        v[2] * ct + cx[2] * st + u[2] * kv * (1.0 - ct),
    ]
}

/// Dihedral angle phi(i,j,k,l) via the standard atan2 formula
/// (sign-irrelevant for the OPLS cosine series).
fn dihedral_phi(b12: [f64; 3], b23: [f64; 3], b34: [f64; 3]) -> f64 {
    // b12 = r_i - r_j? we pass disp(j,i)=i-j etc; use standard:
    // phi = atan2(dot(cross(n1, n2), b23/|b23|), dot(n1, n2))
    let n1 = cross(b12, b23);
    let n2 = cross(b23, b34);
    let n1n = norm(n1);
    let n2n = norm(n2);
    if n1n < 1e-12 || n2n < 1e-12 {
        return 0.0;
    }
    let b23u = [b23[0] / norm(b23), b23[1] / norm(b23), b23[2] / norm(b23)];
    let x = dot(n1, n2) / (n1n * n2n);
    let y = dot(cross(n1, n2), b23u) / (n1n * n2n);
    y.atan2(x)
}

pub struct AtomisticEngine {
    pub state: AtomisticState,
    pub params: AtomisticParams,
    pub temperature: f64,
}

impl AtomisticEngine {
    pub fn new(state: AtomisticState, params: AtomisticParams, temperature: f64) -> Self {
        AtomisticEngine {
            state,
            params,
            temperature,
        }
    }

    // ---------- per-term energies ----------

    #[inline]
    fn bond_e(&self, i: usize, j: usize, t: usize) -> f64 {
        let d = self.state.disp(i, j);
        let r = norm(d);
        let dr = r - self.params.bond_r0[t];
        self.params.bond_k[t] * dr * dr
    }

    #[inline]
    fn angle_e(&self, i: usize, j: usize, k: usize, t: usize) -> f64 {
        let b1 = self.state.disp(j, i);
        let b2 = self.state.disp(j, k);
        let th = angle_of(b1, b2);
        let dth = th - self.params.angle_t0[t];
        self.params.angle_k[t] * dth * dth
    }

    #[inline]
    fn dihedral_e(&self, i: usize, j: usize, k: usize, l: usize, t: usize) -> f64 {
        let b12 = self.state.disp(j, i); // i - j
        let b23 = self.state.disp(k, j); // j - k
        let b34 = self.state.disp(l, k); // k - l
        let phi = dihedral_phi(b12, b23, b34);
        let ks = self.params.dih_k[t];
        0.5 * ks[0] * (1.0 + phi.cos())
            + 0.5 * ks[1] * (1.0 - (2.0 * phi).cos())
            + 0.5 * ks[2] * (1.0 + (3.0 * phi).cos())
            + 0.5 * ks[3] * (1.0 - (4.0 * phi).cos())
    }

    #[inline]
    pub fn pair_e_pub(&self, i: usize, j: usize) -> f64 {
        self.pair_e(i, j)
    }

    /// kBT in kcal/mol (units real). `temperature` is stored as kBT.
    #[inline]
    pub fn temperature_kbt(&self) -> f64 {
        self.temperature
    }

    /// Bonded terms (bonds/angles/dihedrals) touching `moved`, once each.
    pub fn local_bonded_energy(&self, moved: &[usize]) -> f64 {
        let st = &self.state;
        let mut e = 0.0;
        let mut seen_bonds: HashSet<usize> = HashSet::new();
        let mut seen_angles: HashSet<usize> = HashSet::new();
        let mut seen_dih: HashSet<usize> = HashSet::new();
        for &i in moved {
            for &t in &st.bonds_of[i] {
                if seen_bonds.insert(t) {
                    let (a, b, bt) = st.bonds[t];
                    e += self.bond_e(a, b, bt);
                }
            }
            for &t in &st.angles_of[i] {
                if seen_angles.insert(t) {
                    let (a, b, c, at) = st.angles[t];
                    e += self.angle_e(a, b, c, at);
                }
            }
            for &t in &st.dihedrals_of[i] {
                if seen_dih.insert(t) {
                    let (a, b, c, d, dt) = st.dihedrals[t];
                    e += self.dihedral_e(a, b, c, d, dt);
                }
            }
        }
        e
    }

    #[inline]
    fn pair_e(&self, i: usize, j: usize) -> f64 {
        if self.state.excl[i].contains(&j) {
            return 0.0;
        }
        let d = self.state.disp(i, j);
        let r2 = dot(d, d);
        let p = &self.params;
        let (ti, tj) = (self.state.types[i], self.state.types[j]);
        let sig = (p.pair_sig[ti] * p.pair_sig[tj]).sqrt();
        let eps = (p.pair_eps[ti] * p.pair_eps[tj]).sqrt();
        let scaled = self
            .state
            .scaled14
            .contains(&(i.min(j), i.max(j)));
        let (lj_scale, coul_scale) = if scaled {
            (p.scale14_lj, p.scale14_coul)
        } else {
            (1.0, 1.0)
        };
        let mut e = 0.0;
        if r2 < p.lj_cut * p.lj_cut {
            let sr2 = sig * sig / r2;
            let sr6 = sr2 * sr2 * sr2;
            e += lj_scale * 4.0 * eps * (sr6 * sr6 - sr6);
        }
        if r2 < p.coul_cut * p.coul_cut {
            e += coul_scale * COULOMB_REAL * self.state.charges[i] * self.state.charges[j]
                / r2.sqrt();
        }
        e
    }

    /// Total energy (all terms, straightforward O(N^2) pair loop).
    pub fn total_energy(&self) -> f64 {
        let mut e = 0.0;
        let st = &self.state;
        for &(i, j, t) in &st.bonds {
            e += self.bond_e(i, j, t);
        }
        for &(i, j, k, t) in &st.angles {
            e += self.angle_e(i, j, k, t);
        }
        for &(i, j, k, l, t) in &st.dihedrals {
            e += self.dihedral_e(i, j, k, l, t);
        }
        for i in 0..st.pos.len() {
            for j in (i + 1)..st.pos.len() {
                e += self.pair_e(i, j);
            }
        }
        e
    }

    /// Energy of every term touching the moved atom set (exact ΔU set):
    /// pairs with at least one moved atom (counted once), bonds/angles/
    /// dihedrals with at least one moved atom (counted once).
    pub fn local_energy(&self, moved: &[usize]) -> f64 {
        let st = &self.state;
        let mut e = 0.0;
        let mut moved_sorted = moved.to_vec();
        moved_sorted.sort_unstable();
        moved_sorted.dedup();
        let in_moved = |x: usize| moved_sorted.binary_search(&x).is_ok();

        for &i in &moved_sorted {
            for j in 0..st.pos.len() {
                if j == i || in_moved(j) {
                    continue; // moved-moved handled below
                }
                e += self.pair_e(i, j);
            }
        }
        for (a, &i) in moved_sorted.iter().enumerate() {
            for &j in &moved_sorted[a + 1..] {
                e += self.pair_e(i, j);
            }
        }

        let mut seen_bonds: HashSet<usize> = HashSet::new();
        let mut seen_angles: HashSet<usize> = HashSet::new();
        let mut seen_dih: HashSet<usize> = HashSet::new();
        for &i in &moved_sorted {
            for &t in &st.bonds_of[i] {
                if seen_bonds.insert(t) {
                    let (a, b, bt) = st.bonds[t];
                    e += self.bond_e(a, b, bt);
                }
            }
            for &t in &st.angles_of[i] {
                if seen_angles.insert(t) {
                    let (a, b, c, at) = st.angles[t];
                    e += self.angle_e(a, b, c, at);
                }
            }
            for &t in &st.dihedrals_of[i] {
                if seen_dih.insert(t) {
                    let (a, b, c, d, dt) = st.dihedrals[t];
                    e += self.dihedral_e(a, b, c, d, dt);
                }
            }
        }
        e
    }
}
