//! Energy evaluation: pair potentials, bond models, angle potential,
//! cell lists, and local (moved-set) energy differences.
//!
//! Conventions (matching `AutoPoly.models.bead_spring`, Kremer-Grest):
//! * LJ: 4*eps*[(sig/r)^12 - (sig/r)^6], optionally shifted or WCA
//!   (truncated at 2^(1/6)*sig and shifted). Unshifted/untruncated tail
//!   beyond cutoff matches the numpy reference.
//! * Bonds: harmonic 0.5*k*(r-r0)^2 or FENE -0.5*k*R0^2*ln(1-(r/R0)^2).
//! * Angles: k*(1 - cos(theta)), derived from chain contour order.
//! * Nonbonded exclusion: 1-2 (directly bonded) pairs only, the standard
//!   KG convention (LAMMPS special_bonds lj 0 1 1).

use crate::state::MeltState;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug)]
pub struct PairParams {
    pub epsilon: f64,
    pub sigma: f64,
    /// Dimensionless cutoff in units of sigma (ignored when `wca`).
    pub cutoff: f64,
    pub shifted: bool,
    pub wca: bool,
}

impl Default for PairParams {
    fn default() -> Self {
        PairParams {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff: 2.5,
            shifted: false,
            wca: false,
        }
    }
}

impl PairParams {
    /// Actual interaction distance cutoff in length units.
    pub fn r_cut(&self) -> f64 {
        if self.wca {
            2f64.powf(1.0 / 6.0) * self.sigma
        } else {
            self.cutoff * self.sigma
        }
    }

    #[inline]
    pub fn r_cut2(&self) -> f64 {
        let rc = self.r_cut();
        rc * rc
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BondModel {
    Harmonic { k: f64, r0: f64 },
    Fene { k: f64, r0max: f64 },
}

impl Default for BondModel {
    fn default() -> Self {
        BondModel::Harmonic { k: 100.0, r0: 1.0 }
    }
}

#[inline]
pub fn pair_energy(r2: f64, p: &PairParams) -> f64 {
    let rc2 = p.r_cut2();
    if r2 >= rc2 || r2 < 1e-20 {
        return 0.0;
    }
    let sr2 = (p.sigma * p.sigma) / r2;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;
    let base = 4.0 * p.epsilon * (sr12 - sr6);
    if p.wca {
        base + p.epsilon
    } else if p.shifted {
        let src2 = (p.sigma * p.sigma) / rc2;
        let src6 = src2 * src2 * src2;
        base - 4.0 * p.epsilon * (src6 * src6 - src6)
    } else {
        base
    }
}

#[inline]
pub fn bond_energy(r: f64, m: &BondModel) -> f64 {
    match *m {
        BondModel::Harmonic { k, r0 } => {
            let dr = r - r0;
            0.5 * k * dr * dr
        }
        BondModel::Fene { k, r0max } => {
            if r >= r0max {
                f64::INFINITY
            } else {
                let x = r / r0max;
                -0.5 * k * r0max * r0max * (1.0 - x * x).ln()
            }
        }
    }
}

/// Spatial cell list for nonbonded neighbor queries.
/// Cell width >= pair cutoff, so only the 27 neighboring cells need
/// scanning. Falls back to all-pairs when the box is too small for
/// >= 3 cells per axis (queries then return every bead).
pub struct CellList {
    n: [usize; 3],
    w: [f64; 3],
    cells: Vec<Vec<usize>>,
    cell_of: Vec<usize>,
    brute: bool,
    n_beads: usize,
}

impl CellList {
    pub fn build(state: &MeltState, cutoff: f64) -> Self {
        let l = state.box_size;
        let mut n = [0usize; 3];
        let mut brute = false;
        for a in 0..3 {
            let nc = (l / cutoff).floor() as usize;
            n[a] = nc.max(1);
            if n[a] < 3 {
                brute = true;
            }
        }
        let w = [l / n[0] as f64, l / n[1] as f64, l / n[2] as f64];
        let total = n[0] * n[1] * n[2];
        let mut cl = CellList {
            n,
            w,
            cells: vec![Vec::new(); total],
            cell_of: vec![0; state.n_beads()],
            brute,
            n_beads: state.n_beads(),
        };
        for b in 0..state.n_beads() {
            let idx = cl.cell_index(state.pos[b]);
            cl.cells[idx].push(b);
            cl.cell_of[b] = idx;
        }
        cl
    }

    #[inline]
    fn cell_index(&self, p: [f64; 3]) -> usize {
        let mut c = [0usize; 3];
        for a in 0..3 {
            let mut i = (p[a] / self.w[a]).floor() as isize;
            if i < 0 {
                i = 0;
            }
            if i >= self.n[a] as isize {
                i = self.n[a] as isize - 1;
            }
            c[a] = i as usize;
        }
        (c[2] * self.n[1] + c[1]) * self.n[0] + c[0]
    }

    /// Re-assign a bead's cell after its position changed.
    pub fn update(&mut self, bead: usize, pos: [f64; 3]) {
        if self.brute {
            return;
        }
        let idx = self.cell_index(pos);
        let old = self.cell_of[bead];
        if idx != old {
            if let Some(p) = self.cells[old].iter().position(|&x| x == bead) {
                self.cells[old].swap_remove(p);
            }
            self.cells[idx].push(bead);
            self.cell_of[bead] = idx;
        }
    }

    /// All beads in the 27 cells neighboring the cell containing `p`
    /// (periodic). In brute mode: every bead in the system.
    pub fn candidates_into(&self, p: [f64; 3], out: &mut Vec<usize>) {
        if self.brute {
            out.extend(0..self.n_beads);
            return;
        }
        // Cell coordinates of p.
        let mut c = [0usize; 3];
        for a in 0..3 {
            let mut i = (p[a] / self.w[a]).floor() as isize;
            if i < 0 {
                i = 0;
            }
            if i >= self.n[a] as isize {
                i = self.n[a] as isize - 1;
            }
            c[a] = i as usize;
        }
        for dz in -1isize..=1 {
            let cz = (c[2] as isize + dz).rem_euclid(self.n[2] as isize) as usize;
            for dy in -1isize..=1 {
                let cy = (c[1] as isize + dy).rem_euclid(self.n[1] as isize) as usize;
                for dx in -1isize..=1 {
                    let cx = (c[0] as isize + dx).rem_euclid(self.n[0] as isize) as usize;
                    let idx = (cz * self.n[1] + cy) * self.n[0] + cx;
                    out.extend_from_slice(&self.cells[idx]);
                }
            }
        }
    }
}

/// Energy of every term touching the moved set: nonbonded pairs,
/// bonds, and (optional) angles. Each term is counted exactly once
/// whether one or several of its beads moved.
pub fn local_energy(
    state: &MeltState,
    cells: &CellList,
    pair: &PairParams,
    bond: &BondModel,
    angle_k: f64,
    moved: &[usize],
) -> f64 {
    let mut moved_sorted = moved.to_vec();
    moved_sorted.sort_unstable();
    moved_sorted.dedup();
    let in_moved = |x: usize| moved_sorted.binary_search(&x).is_ok();

    let mut e = 0.0;
    let rc2 = pair.r_cut2();
    let mut cand: Vec<usize> = Vec::with_capacity(64);

    // Nonbonded pairs.
    for &i in &moved_sorted {
        cand.clear();
        cells.candidates_into(state.pos[i], &mut cand);
        for &j in &cand {
            if j == i {
                continue;
            }
            // Count pairs inside the moved set once (i < j).
            if in_moved(j) && j < i {
                continue;
            }
            // 1-2 exclusion. This uses the *topology*, not the distance,
            // so the pair is excluded in both the old and the new
            // evaluation even when a stretched bond (r > cutoff) makes
            // it LJ-active. Excluding it consistently keeps ΔU exact:
            // a hard overlap then registers through the bond term alone.
            if state.bonded(i, j) {
                continue;
            }
            let d = state.disp(i, j);
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 < rc2 {
                e += pair_energy(r2, pair);
            }
        }
    }

    // Bonds.
    for &i in &moved_sorted {
        for &j in &state.bonds_adj[i] {
            if in_moved(j) && j < i {
                continue;
            }
            let d = state.disp(i, j);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            e += bond_energy(r, bond);
        }
    }

    // Angles: dedupe triplets shared between moved beads.
    if angle_k != 0.0 {
        let mut seen: HashSet<(usize, usize, usize)> = HashSet::new();
        for &i in &moved_sorted {
            for t in state.angles_touching(i) {
                let key = (t[0].min(t[2]), t[1], t[0].max(t[2]));
                if seen.insert(key) {
                    let b1 = state.disp(t[1], t[0]);
                    let b2 = state.disp(t[1], t[2]);
                    let n1 = (b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2]).sqrt();
                    let n2 = (b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2]).sqrt();
                    if n1 < 1e-12 || n2 < 1e-12 {
                        continue;
                    }
                    let dot = b1[0] * b2[0] + b1[1] * b2[1] + b1[2] * b2[2];
                    let cos_theta = (dot / (n1 * n2)).clamp(-1.0, 1.0);
                    e += angle_k * (1.0 - cos_theta);
                }
            }
        }
    }

    e
}

/// Full system energy (all pairs via cell list, all bonds, all angles).
pub fn total_energy(
    state: &MeltState,
    cells: &CellList,
    pair: &PairParams,
    bond: &BondModel,
    angle_k: f64,
) -> f64 {
    let mut e = 0.0;
    let rc2 = pair.r_cut2();
    let mut cand: Vec<usize> = Vec::with_capacity(128);

    for i in 0..state.n_beads() {
        cand.clear();
        cells.candidates_into(state.pos[i], &mut cand);
        for &j in &cand {
            if j <= i {
                continue;
            }
            if state.bonded(i, j) {
                continue;
            }
            let d = state.disp(i, j);
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 < rc2 {
                e += pair_energy(r2, pair);
            }
        }
    }

    for i in 0..state.n_beads() {
        for &j in &state.bonds_adj[i] {
            if j > i {
                let d = state.disp(i, j);
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                e += bond_energy(r, bond);
            }
        }
    }

    if angle_k != 0.0 {
        for t in state.all_angles() {
            let b1 = state.disp(t[1], t[0]);
            let b2 = state.disp(t[1], t[2]);
            let n1 = (b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2]).sqrt();
            let n2 = (b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2]).sqrt();
            if n1 < 1e-12 || n2 < 1e-12 {
                continue;
            }
            let dot = b1[0] * b2[0] + b1[1] * b2[1] + b1[2] * b2[2];
            let cos_theta = (dot / (n1 * n2)).clamp(-1.0, 1.0);
            e += angle_k * (1.0 - cos_theta);
        }
    }

    e
}
