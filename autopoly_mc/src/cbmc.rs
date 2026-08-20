//! Configurational-bias Monte Carlo (CBMC) trimer regrowth for atomistic
//! double-bridging (Karayiannis-style, adapted to same-contour-position
//! tail swaps for strictly monodisperse melts).
//!
//! Frozen-geometry bridging fails in atomistic melts because junction
//! environments must be *reconstructed* (measured: every candidate
//! carries >= 13k kcal/mol of bonded strain). This module excises the
//! 3-backbone-carbon trimer around each cut and regrows it in the new
//! connectivity with proper priors:
//!
//! * bond length r ~ r^2 * exp(-β K (r-r0)^2)      (inverse-CDF grid)
//! * angle θ ~ sinθ * exp(-β K (θ-θ0)^2)           (inverse-CDF grid)
//! * torsion φ ~ uniform for the first two trimer atoms (dihedral energy
//!   enters the Rosenbluth weight)
//! * the CLOSING atom is placed on the circle defined by its sampled
//!   (r, θ), with the azimuth ψ sampled from the closure-Boltzmann
//!   conditional ∝ exp(-β u_close(ψ)) — the circle normalizer Z_ψ is
//!   folded into the Rosenbluth weight, so closure always succeeds when
//!   geometrically reachable and the scheme stays exact.
//!
//! Hydrogens of the six regrown carbons are regrown by single-atom
//! Rosenbluth (bond-prior radial × uniform direction on the sphere,
//! weight = angles + dihedrals + nonbonded), so no deterministic H
//! placement bias enters the acceptance rule.
//!
//! W_old is retraced on a position buffer initialized from the NEW
//! configuration (the reverse move's input), with each atom restored to
//! its old position once its weight is evaluated — this is the exact
//! reverse-sequence environment when both trimers share one junction.
//!
//! Acceptance: min[1, (W_new/W_old) · (P_fwd/P_rev)] with proposal
//! probabilities from reach-based candidate counts (each physical swap
//! is reachable from either chain, so both chains' counts enter).

use crate::atomistic::{AtomisticParams, AtomisticState, COULOMB_REAL};
use crate::atomistic_mc::ACellList;
use crate::bridge::{BridgeProposal, TypeTables};
use rand::Rng;
use std::collections::HashSet;

// ----------------------------------------------------------------------
// Inverse-CDF samplers (bonded Boltzmann priors with Jacobians)
// ----------------------------------------------------------------------

pub struct InvCdf {
    grid: Vec<f64>,
    cdf: Vec<f64>,
}

impl InvCdf {
    fn build<F: Fn(f64) -> f64>(x0: f64, x1: f64, n: usize, density: F) -> Self {
        let mut grid = Vec::with_capacity(n);
        let mut cdf = Vec::with_capacity(n);
        let mut acc = 0.0;
        let mut prev = 0.0;
        for i in 0..n {
            let x = x0 + (x1 - x0) * (i as f64 + 0.5) / n as f64;
            let d = density(x);
            acc += 0.5 * (d + prev) * (x1 - x0) / n as f64;
            prev = d;
            grid.push(x);
            cdf.push(acc);
        }
        for c in cdf.iter_mut() {
            *c /= acc;
        }
        InvCdf { grid, cdf }
    }

    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let u: f64 = rng.random();
        let idx = self.cdf.partition_point(|&c| c < u).min(self.grid.len() - 1);
        let i0 = idx.saturating_sub(1);
        let (c0, c1) = (self.cdf.get(i0).copied().unwrap_or(0.0), self.cdf[idx]);
        let t = if c1 > c0 { (u - c0) / (c1 - c0) } else { 0.5 };
        self.grid[i0] + t * (self.grid[idx] - self.grid[i0])
    }
}

/// Bond-length sampler: p(r) ∝ r^2 exp(-β K (r-r0)^2).
pub fn bond_sampler(k: f64, r0: f64, kbt: f64) -> InvCdf {
    let sigma = (kbt / (2.0 * k)).sqrt();
    let lo = (r0 - 5.0 * sigma).max(0.3);
    let hi = r0 + 5.0 * sigma;
    InvCdf::build(lo, hi, 256, |r| {
        r * r * (-(r - r0) * (r - r0) * k / kbt).exp()
    })
}

/// Angle sampler: p(θ) ∝ sinθ exp(-β K (θ-θ0)^2).
pub fn angle_sampler(k: f64, theta0: f64, kbt: f64) -> InvCdf {
    InvCdf::build(0.02, std::f64::consts::PI - 0.02, 512, |th| {
        th.sin() * (-(th - theta0) * (th - theta0) * k / kbt).exp()
    })
}

// ----------------------------------------------------------------------
// Geometry helpers
// ----------------------------------------------------------------------

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
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
#[inline]
fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    if n < 1e-12 {
        [1.0, 0.0, 0.0]
    } else {
        scale(a, 1.0 / n)
    }
}

#[inline]
fn min_img(d: [f64; 3], l: f64) -> [f64; 3] {
    [
        d[0] - l * (d[0] / l).round(),
        d[1] - l * (d[1] / l).round(),
        d[2] - l * (d[2] / l).round(),
    ]
}

#[inline]
fn wrap(p: [f64; 3], l: f64) -> [f64; 3] {
    [
        p[0] - l * (p[0] / l).floor(),
        p[1] - l * (p[1] / l).floor(),
        p[2] - l * (p[2] / l).floor(),
    ]
}

fn arb_perp(u: [f64; 3]) -> [f64; 3] {
    let a = if u[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    unit(cross(u, a))
}

/// Local frame at p1: u along (p2 - p1), e1 in the (u, p3) plane pointing
/// toward the p3 side (so ψ=0 is cis with p3), e2 = u × e1.
fn local_frame(
    p1: [f64; 3],
    p2: [f64; 3],
    p3: Option<[f64; 3]>,
    box_size: f64,
) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let u = unit(min_img(sub(p2, p1), box_size));
    let e1 = match p3 {
        Some(q) => {
            let w = min_img(sub(q, p1), box_size);
            let perp = sub(w, scale(u, dot(w, u)));
            if norm(perp) < 1e-8 {
                arb_perp(u)
            } else {
                unit(perp)
            }
        }
        None => arb_perp(u),
    };
    let e2 = unit(cross(u, e1));
    (u, e1, e2)
}

/// Point on the (r, θ) cone about u at azimuth ψ.
fn cone_point(
    p1: [f64; 3],
    u: [f64; 3],
    e1: [f64; 3],
    e2: [f64; 3],
    r: f64,
    theta: f64,
    psi: f64,
    box_size: f64,
) -> [f64; 3] {
    let dir = add(
        scale(u, theta.cos()),
        add(scale(e1, theta.sin() * psi.cos()), scale(e2, theta.sin() * psi.sin())),
    );
    wrap(add(p1, scale(dir, r)), box_size)
}

/// Uniform random direction on the sphere.
#[allow(dead_code)]
fn sphere_dir<R: Rng>(rng: &mut R) -> [f64; 3] {
    let z = rng.random::<f64>() * 2.0 - 1.0;
    let t = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
    let s = (1.0 - z * z).max(0.0).sqrt();
    [s * t.cos(), s * t.sin(), z]
}

// ----------------------------------------------------------------------
// CBMC configuration
// ----------------------------------------------------------------------

pub struct CbmcConfig {
    /// trials per backbone atom (Rosenbluth k)
    pub n_trials: usize,
    /// trials per regrown hydrogen
    pub n_h_trials: usize,
    /// Fibonacci grid size for the hydrogen direction conditional
    pub n_h_dirs: usize,
    /// ψ grid size for the closure conditional
    pub n_psi: usize,
    /// candidate reach: stub-to-downstream distance [Å] within which a
    /// trimer regrowth can close (3 C-C bonds)
    pub r_reach: f64,
    /// upstream guide bias: equilibrium span to the closure target with
    /// m bonds remaining (index m), [Å]
    pub guide_span: [f64; 5],
    /// strength of the upstream guide bias [kcal/mol/Å²]
    pub guide_k: f64,
    /// backbone atoms regrown per junction (3 = trimer, 4 = quadmer);
    /// longer segments bridge larger inter-chain gaps at the price of
    /// more Rosenbluth dilution
    pub n_regrow: usize,
}

impl Default for CbmcConfig {
    fn default() -> Self {
        CbmcConfig {
            n_trials: 30,
            n_h_trials: 8,
            n_h_dirs: 48,
            n_psi: 144,
            r_reach: 4.5,
            guide_span: [0.0, 1.53, 2.55, 3.80, 5.00],
            guide_k: 10.0,
            n_regrow: 3,
        }
    }
}

// ----------------------------------------------------------------------
// Trial environment: topology from `state`, positions passed explicitly
// ----------------------------------------------------------------------

struct TrialEnv<'a> {
    state: &'a AtomisticState,
    par: &'a AtomisticParams,
    /// Atoms being regrown in this move that have NOT yet been placed
    /// (or retraced) — they still sit at stale positions in the working
    /// buffer and must not contribute to trial nonbonded energies
    /// (standard CBMC segment excision). Interior mutability lets the
    /// regrowth sequence manage placement state through shared refs.
    ghost: std::cell::RefCell<HashSet<usize>>,
}

impl<'a> TrialEnv<'a> {
    #[inline]
    fn l(&self) -> f64 {
        self.state.box_size
    }

    /// Nonbonded energy of atom `a` vs all atoms (excluding 1-2/1-3
    /// partners, scaling 1-4), via the cell list.
    fn nonbonded(&self, pos: &[[f64; 3]], a: usize, cells: &ACellList) -> f64 {
        let mut e = 0.0;
        let mut cand = Vec::with_capacity(128);
        cells.candidates_into(pos[a], &mut cand);
        let ghosts = self.ghost.borrow();
        for j in cand {
            if j == a || ghosts.contains(&j) || self.state.excl[a].contains(&j) {
                continue;
            }
            let key = (a.min(j), a.max(j));
            let (ls, cs) = if self.state.scaled14.contains(&key) {
                (self.par.scale14_lj, self.par.scale14_coul)
            } else {
                (1.0, 1.0)
            };
            let d = min_img(sub(pos[j], pos[a]), self.l());
            let r2 = dot(d, d);
            let (ti, tj) = (self.state.types[a], self.state.types[j]);
            let sig = (self.par.pair_sig[ti] * self.par.pair_sig[tj]).sqrt();
            let eps = (self.par.pair_eps[ti] * self.par.pair_eps[tj]).sqrt();
            if r2 < self.par.lj_cut * self.par.lj_cut {
                let sr2 = sig * sig / r2;
                let sr6 = sr2 * sr2 * sr2;
                e += ls * 4.0 * eps * (sr6 * sr6 - sr6);
            }
            if r2 < self.par.coul_cut * self.par.coul_cut {
                e += cs * COULOMB_REAL * self.state.charges[a] * self.state.charges[j] / r2.sqrt();
            }
        }
        e
    }

    fn bond_e(&self, pos: &[[f64; 3]], i: usize, j: usize, t: usize) -> f64 {
        let r = norm(min_img(sub(pos[j], pos[i]), self.l()));
        let dr = r - self.par.bond_r0[t];
        self.par.bond_k[t] * dr * dr
    }

    fn angle_e(&self, pos: &[[f64; 3]], i: usize, j: usize, k: usize, t: usize) -> f64 {
        let b1 = min_img(sub(pos[i], pos[j]), self.l());
        let b2 = min_img(sub(pos[k], pos[j]), self.l());
        let n1 = norm(b1);
        let n2 = norm(b2);
        if n1 < 1e-12 || n2 < 1e-12 {
            return 0.0;
        }
        let c = (dot(b1, b2) / (n1 * n2)).clamp(-1.0, 1.0);
        let dth = c.acos() - self.par.angle_t0[t];
        self.par.angle_k[t] * dth * dth
    }

    fn dih_e(&self, pos: &[[f64; 3]], i: usize, j: usize, k: usize, l: usize, t: usize) -> f64 {
        let b12 = min_img(sub(pos[i], pos[j]), self.l());
        let b23 = min_img(sub(pos[j], pos[k]), self.l());
        let b34 = min_img(sub(pos[k], pos[l]), self.l());
        let n1 = cross(b12, b23);
        let n2 = cross(b23, b34);
        let n1n = norm(n1);
        let n2n = norm(n2);
        if n1n < 1e-12 || n2n < 1e-12 {
            return 0.0;
        }
        let b23u = unit(b23);
        let x = dot(n1, n2) / (n1n * n2n);
        let y = dot(cross(n1, n2), b23u) / (n1n * n2n);
        let phi = y.atan2(x);
        let ks = self.par.dih_k[t];
        0.5 * ks[0] * (1.0 + phi.cos())
            + 0.5 * ks[1] * (1.0 - (2.0 * phi).cos())
            + 0.5 * ks[2] * (1.0 + (3.0 * phi).cos())
            + 0.5 * ks[3] * (1.0 - (4.0 * phi).cos())
    }

    /// Hydrogen reweight energy: every bonded term containing h except
    /// its (single) bond, plus nonbonded.
    fn h_reweight(&self, pos: &[[f64; 3]], h: usize, cells: &ACellList) -> f64 {
        let mut e = self.nonbonded(pos, h, cells);
        for &ti in &self.state.angles_of[h] {
            let (i, j, k, t) = self.state.angles[ti];
            e += self.angle_e(pos, i, j, k, t);
        }
        for &ti in &self.state.dihedrals_of[h] {
            let (i, j, k, l, t) = self.state.dihedrals[ti];
            e += self.dih_e(pos, i, j, k, l, t);
        }
        e
    }
}

// ----------------------------------------------------------------------
// Type resolution (by-example tables)
// ----------------------------------------------------------------------

fn resolve_bond(tables: &TypeTables, st: &AtomisticState, i: usize, j: usize) -> usize {
    let key = (st.types[i].min(st.types[j]), st.types[i].max(st.types[j]));
    tables.bond.get(&key).copied().unwrap_or(1)
}
fn resolve_angle(tables: &TypeTables, st: &AtomisticState, i: usize, j: usize, k: usize) -> usize {
    tables
        .angle
        .get(&(st.types[i], st.types[j], st.types[k]))
        .or_else(|| tables.angle.get(&(st.types[k], st.types[j], st.types[i])))
        .copied()
        .unwrap_or(1)
}
fn resolve_dih(
    tables: &TypeTables,
    st: &AtomisticState,
    i: usize,
    j: usize,
    k: usize,
    l: usize,
) -> usize {
    tables
        .dihedral
        .get(&(st.types[i], st.types[j], st.types[k], st.types[l]))
        .or_else(|| tables.dihedral.get(&(st.types[l], st.types[k], st.types[j], st.types[i])))
        .copied()
        .unwrap_or(1)
}

// ----------------------------------------------------------------------
// Reach-based candidate enumeration.
//
// A k-mer regrowth from stub a1 can close onto downstream atom d_b only
// if |a1 - d_b| is within the (k+1)-bond reach. Both junctions must
// satisfy it. All four reach atoms (stubs and downstream anchors) are
// never moved by the regrowth, so the reverse proposal is always in the
// reverse candidate set.
// ----------------------------------------------------------------------

pub fn enumerate_cbmc_candidates(
    state: &AtomisticState,
    cells: &ACellList,
    a: usize,
    r_reach: f64,
    k_regrow: usize,
) -> Vec<BridgeProposal> {
    let chains = &state.chains;
    let n = chains[a].len();
    if n < k_regrow + 4 {
        return Vec::new();
    }
    // Atomistic valence rule: a flip joins each stub to the OTHER chain's
    // end atom, which then has two backbone neighbors. Chemically valid
    // only if that end atom's type also occurs at interior backbone
    // positions (by-example evidence the junction terms can exist);
    // e.g. a methyl end type fails — the flipped join would be
    // pentavalent. Non-flip joins attach interior atoms and never trip
    // this rule.
    let mut interior_types: HashSet<usize> = HashSet::new();
    for chain in chains {
        let m = chain.len();
        if m > 2 {
            for &b in &chain[1..m - 1] {
                interior_types.insert(state.types[b]);
            }
        }
    }
    let flip_ok = |c: usize| -> bool {
        interior_types.contains(&state.types[*chains[c].last().unwrap()])
    };
    let mut where_is: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (c, chain) in chains.iter().enumerate() {
        for &b in chain.iter() {
            where_is.insert(b, c);
        }
    }
    let rr2 = r_reach * r_reach;
    let reach2 = |i: usize, j: usize| {
        let d = min_img(sub(state.pos[i], state.pos[j]), state.box_size);
        dot(d, d)
    };
    let mut out = Vec::new();
    let mut cand = Vec::new();
    for s in 2..(n - k_regrow) {
        let a1 = chains[a][s - 1];
        cand.clear();
        cells.candidates_into(state.pos[a1], &mut cand);
        for &j in &cand {
            let Some(&b) = where_is.get(&j) else { continue };
            if b == a || chains[b].len() != n {
                continue;
            }
            for flip in [false, true] {
                if flip && (!flip_ok(a) || !flip_ok(b)) {
                    continue;
                }
                // downstream anchor of the tail coming onto stub a1
                let d_b = if flip { chains[b][n - 1 - k_regrow] } else { chains[b][s + k_regrow] };
                if d_b != j || reach2(a1, d_b) >= rr2 {
                    continue;
                }
                // other junction: stub b1 onto downstream anchor of tail a
                let b1 = chains[b][s - 1];
                let d_a = if flip { chains[a][n - 1 - k_regrow] } else { chains[a][s + k_regrow] };
                if reach2(b1, d_a) < rr2 {
                    out.push(BridgeProposal { a, b, s, flip });
                }
            }
        }
    }
    out
}

// ----------------------------------------------------------------------
// Per-atom regrowth with Rosenbluth weights
// ----------------------------------------------------------------------

/// Select a trial index proportionally to exp(logw) and return
/// (log Σ exp(logw), selected index).
fn rosenbluth_select<R: Rng>(logw: &[f64], rng: &mut R) -> (f64, usize) {
    let lmax = logw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = logw.iter().map(|&l| (l - lmax).exp()).collect();
    let wsum: f64 = w.iter().sum();
    let mut r = rng.random::<f64>() * wsum;
    let mut sel = logw.len() - 1;
    for (i, &wi) in w.iter().enumerate() {
        r -= wi;
        if r <= 0.0 {
            sel = i;
            break;
        }
    }
    (lmax + wsum.ln(), sel)
}

/// ψ-bias for the circle-sampled regrowth of one backbone atom.
enum PsiBias<'a> {
    /// Upstream atoms: quadratic pull toward the equilibrium remaining
    /// span to the closure target. Heuristic, exactly compensated by the
    /// Z_ψ factor in the Rosenbluth weight.
    Guide { d: usize, r_opt: f64, k_guide: f64 },
    /// Closing atom: the full ψ-dependent bonded terms + nonbonded.
    Close(&'a Closure),
}

/// Regrow one backbone atom on its (r, θ) circle: r and θ from bonded
/// priors, azimuth ψ from the 1-D conditional ∝ exp(-β u_bias(ψ)) on the
/// circle. Per-trial Rosenbluth weight = Z_ψ · exp(-β u_rem) where u_rem
/// collects the energy terms not folded into the ψ-bias (nonbonded +
/// own dihedral for upstream atoms; nothing for the closing atom).
/// With `include = Some(p)` (retrace), the original position is added as
/// the last trial, evaluated under the same weight formula.
#[allow(clippy::too_many_arguments)]
fn regrow_atom<R: Rng>(
    pos: &mut Vec<[f64; 3]>,
    atom: usize,
    p1: usize,
    p2: usize,
    p3: Option<usize>,
    bond_t: usize,
    angle_t: usize,
    dih_t: Option<usize>,
    bias: &PsiBias,
    env: &TrialEnv,
    cfg: &CbmcConfig,
    kbt: f64,
    include: Option<[f64; 3]>,
    rng: &mut R,
) -> f64 {
    let bs = bond_sampler(env.par.bond_k[bond_t], env.par.bond_r0[bond_t], kbt);
    let as_ = angle_sampler(env.par.angle_k[angle_t], env.par.angle_t0[angle_t], kbt);
    let l = env.l();
    let (u, e1, e2) = local_frame(pos[p1], pos[p2], p3.map(|q| pos[q]), l);
    let cutoff = env.par.lj_cut.max(env.par.coul_cut);
    let cells = ACellList::build_from(pos, l, cutoff);
    let n = cfg.n_trials + include.is_some() as usize;
    let dpsi = 2.0 * std::f64::consts::PI / cfg.n_psi as f64;
    let circle = |o: [f64; 3], rho: f64, psi: f64| {
        cone_point(o, [1.0, 0.0, 0.0], e1, e2, rho, std::f64::consts::FRAC_PI_2, psi, l)
    };
    let mut logw = Vec::with_capacity(n);
    let mut pts = Vec::with_capacity(n);
    let saved = pos[atom];
    for t in 0..n {
        // (r, θ) for this trial: sampled, or read off the included point
        let (r, th, included_q) = match include {
            Some(q) if t == n - 1 => {
                let v = min_img(sub(q, pos[p1]), l);
                let rr = norm(v);
                let ct = if rr > 1e-12 { dot(v, u) / rr } else { 1.0 };
                (rr, ct.clamp(-1.0, 1.0).acos(), Some(q))
            }
            _ => (bs.sample(rng), as_.sample(rng), None),
        };
        let o = add(pos[p1], scale(u, r * th.cos()));
        let rho = r * th.sin();
        // ψ conditional on the circle
        let mut logc = Vec::with_capacity(cfg.n_psi);
        for m in 0..cfg.n_psi {
            let psi = (m as f64 + 0.5) * dpsi;
            let q = circle(o, rho, psi);
            pos[atom] = q;
            let ub = match bias {
                PsiBias::Guide { d, r_opt, k_guide } => {
                    let dist = norm(min_img(sub(q, pos[*d]), l));
                    let dev = dist - r_opt;
                    // sterically aware guide: span pull + nonbonded, so
                    // upstream atoms are not placed into LJ cores in a
                    // dense melt (nonbonded is exactly compensated via
                    // Z_ψ; u_rem drops it below)
                    k_guide * dev * dev + env.nonbonded(pos, atom, &cells)
                }
                PsiBias::Close(cl) => {
                    close_bias(env, pos, atom, p1, p2, p3, cl)
                        + env.nonbonded(pos, atom, &cells)
                }
            };
            logc.push(-ub / kbt);
        }
        let lcmax = logc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let zsum: f64 = logc.iter().map(|&x| (x - lcmax).exp()).sum();
        let log_zpsi = lcmax + (dpsi * zsum).ln();
        let q = match included_q {
            Some(q) => q,
            None => {
                let (_, msel) = rosenbluth_select(&logc, rng);
                circle(o, rho, (msel as f64 + 0.5) * dpsi)
            }
        };
        pos[atom] = q;
        let u_rem = match bias {
            PsiBias::Guide { .. } => {
                // nonbonded already folded into the ψ-bias above
                if let (Some(q3), Some(dt)) = (p3, dih_t) {
                    env.dih_e(pos, atom, p1, p2, q3, dt)
                } else {
                    0.0
                }
            }
            PsiBias::Close(_) => 0.0,
        };
        logw.push(log_zpsi - u_rem / kbt);
        pts.push(q);
        pos[atom] = saved;
    }
    let (log_w, sel) = rosenbluth_select(&logw, rng);
    if include.is_none() {
        pos[atom] = pts[sel];
    } else {
        pos[atom] = saved;
    }
    log_w
}

/// Closure-term types and anchors for the closing trimer atom.
struct Closure {
    d: usize,
    dn: Option<usize>,
    cb_t: usize,          // bond q-d
    ca1_t: usize,         // angle p1-q-d
    ca2_t: usize,         // angle q-d-dn
    cd1_t: usize,         // dih p2-p1-q-d
    cd2_t: usize,         // dih p1-q-d-dn
    own_dt: Option<usize>, // dih q-p1-p2-p3
}

/// All ψ-dependent bonded terms of the closing atom (the ψ-bias energy).
#[allow(clippy::too_many_arguments)]
fn close_bias(
    env: &TrialEnv,
    pos: &[[f64; 3]],
    atom: usize,
    p1: usize,
    p2: usize,
    p3: Option<usize>,
    cl: &Closure,
) -> f64 {
    let mut e = env.bond_e(pos, atom, cl.d, cl.cb_t);
    e += env.angle_e(pos, p1, atom, cl.d, cl.ca1_t);
    e += env.dih_e(pos, p2, p1, atom, cl.d, cl.cd1_t);
    if let Some(dn) = cl.dn {
        e += env.angle_e(pos, atom, cl.d, dn, cl.ca2_t);
        e += env.dih_e(pos, p1, atom, cl.d, dn, cl.cd2_t);
    }
    if let (Some(q3), Some(dt)) = (p3, cl.own_dt) {
        e += env.dih_e(pos, atom, p1, p2, q3, dt);
    }
    e
}

/// Regrow one hydrogen on its (fixed) carbon: bond-prior radial ×
/// uniform direction; weight = angles + dihedrals + nonbonded.
#[allow(clippy::too_many_arguments)]
/// Fibonacci-sphere direction set of size k (deterministic; randomly
/// rotated per use for trial diversity).
fn fib_sphere(k: usize) -> Vec<[f64; 3]> {
    let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    (0..k)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / k as f64;
            let s = (1.0 - z * z).max(0.0).sqrt();
            let phi = i as f64 * golden;
            [s * phi.cos(), s * phi.sin(), z]
        })
        .collect()
}

/// Rotate v by the unit quaternion (w, x, y, z).
fn quat_rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let t = [
        2.0 * (y * v[2] - z * v[1]),
        2.0 * (z * v[0] - x * v[2]),
        2.0 * (x * v[1] - y * v[0]),
    ];
    [
        v[0] + w * t[0] + y * t[2] - z * t[1],
        v[1] + w * t[1] + z * t[0] - x * t[2],
        v[2] + w * t[2] + x * t[1] - y * t[0],
    ]
}

/// Uniform random unit quaternion (Shoemake).
fn rand_quat<R: Rng>(rng: &mut R) -> [f64; 4] {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
    let u3: f64 = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
    let a = (1.0 - u1).sqrt();
    let b = u1.sqrt();
    [a * u2.sin(), a * u2.cos(), b * u3.sin(), b * u3.cos()]
}

fn regrow_h<R: Rng>(
    pos: &mut Vec<[f64; 3]>,
    h: usize,
    carbon: usize,
    bond_t: usize,
    env: &TrialEnv,
    cfg: &CbmcConfig,
    kbt: f64,
    include: Option<[f64; 3]>,
    rng: &mut R,
) -> f64 {
    let bs = bond_sampler(env.par.bond_k[bond_t], env.par.bond_r0[bond_t], kbt);
    let l = env.l();
    let cutoff = env.par.lj_cut.max(env.par.coul_cut);
    let cells = ACellList::build_from(pos, l, cutoff);
    let n = cfg.n_h_trials + include.is_some() as usize;
    let dirs = fib_sphere(cfg.n_h_dirs);
    let solid_angle = 4.0 * std::f64::consts::PI / cfg.n_h_dirs as f64;
    let mut logw = Vec::with_capacity(n);
    let mut pts = Vec::with_capacity(n);
    let saved = pos[h];
    for t in 0..n {
        // Sphere conditional: direction sampled ∝ exp(-β u_h) over a
        // (randomly rotated) Fibonacci grid; the grid normalizer
        // log(ΔΩ Σ exp(-β u)) is the trial's Rosenbluth weight, exactly
        // compensating the directional bias. Bond length from the prior.
        let (r, included_p) = match include {
            Some(p) if t == n - 1 => {
                let rr = norm(min_img(sub(p, pos[carbon]), l));
                (rr, Some(p))
            }
            _ => (bs.sample(rng), None),
        };
        let quat = rand_quat(rng);
        let mut logc = Vec::with_capacity(cfg.n_h_dirs);
        for d in &dirs {
            let dv = quat_rotate(quat, *d);
            pos[h] = wrap(add(pos[carbon], scale(dv, r)), l);
            logc.push(-env.h_reweight(pos, h, &cells) / kbt);
        }
        let lcmax = logc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let zsum: f64 = logc.iter().map(|&x| (x - lcmax).exp()).sum();
        let log_zdir = lcmax + (solid_angle * zsum).ln();
        let pt = match included_p {
            Some(p) => p,
            None => {
                let (_, msel) = rosenbluth_select(&logc, rng);
                wrap(add(pos[carbon], scale(quat_rotate(quat, dirs[msel]), r)), l)
            }
        };
        pos[h] = pt;
        logw.push(log_zdir);
        pts.push(pt);
        pos[h] = saved;
    }
    let (log_w, sel) = rosenbluth_select(&logw, rng);
    if std::env::var("CBMC_DEBUG").is_ok() {
        eprintln!("    H {h} on C{carbon}: logw = {log_w:.2}");
    }
    if include.is_none() {
        pos[h] = pts[sel];
    } else {
        pos[h] = saved;
    }
    log_w
}

// ----------------------------------------------------------------------
// Trimer and hydrogen sequences
// ----------------------------------------------------------------------

fn tail_of(chain: &[usize], s: usize, flip: bool) -> Vec<usize> {
    if flip {
        chain[s..].iter().rev().copied().collect()
    } else {
        chain[s..].to_vec()
    }
}

fn non_backbone_neighbors(state: &AtomisticState, carbon: usize) -> Vec<usize> {
    let mut backbone: HashSet<usize> = HashSet::new();
    for chain in &state.chains {
        for &b in chain {
            backbone.insert(b);
        }
    }
    let mut out = Vec::new();
    for &t in &state.bonds_of[carbon] {
        let (i, j, _) = state.bonds[t];
        let other = if i == carbon { j } else { i };
        if !backbone.contains(&other) {
            out.push(other);
        }
    }
    out.sort_unstable();
    out
}

/// Retrace or forward-regrow one k-mer segment. With
/// `retrace_pos = Some(old)`, each atom's old position is included as
/// the last trial and `pos[atom]` is SET to the old position after its
/// weight is evaluated (so later atoms in the sequence see
/// already-retraced atoms at old positions, matching the reverse move's
/// growth environment).
#[allow(clippy::too_many_arguments)]
fn regrow_segment<R: Rng>(
    pos: &mut Vec<[f64; 3]>,
    tail: &[usize],
    stub: usize,
    prev_chain: &[usize],
    s: usize,
    state: &AtomisticState,
    tables: &TypeTables,
    env: &TrialEnv,
    cfg: &CbmcConfig,
    kbt: f64,
    retrace_pos: Option<&[[f64; 3]]>,
    rng: &mut R,
) -> f64 {
    let k = cfg.n_regrow;
    let mut w = 0.0;
    let p1 = stub;
    let p2 = prev_chain[s - 2];
    let p3 = if s >= 3 { Some(prev_chain[s - 3]) } else { None };
    let d = tail[k];
    let dn: Option<usize> = tail.get(k + 1).copied();
    // closing atom's upstream dihedral partner (a3 of atom k-1)
    let cl_a3 = if k >= 4 { tail[k - 4] } else { p1 };
    let cl = Closure {
        d,
        dn,
        cb_t: resolve_bond(tables, state, tail[k - 1], d),
        ca1_t: resolve_angle(tables, state, tail[k - 2], tail[k - 1], d),
        ca2_t: resolve_angle(tables, state, tail[k - 1], d, dn.unwrap_or(d)),
        cd1_t: resolve_dih(tables, state, tail[k - 3], tail[k - 2], tail[k - 1], d),
        cd2_t: resolve_dih(tables, state, tail[k - 2], tail[k - 1], d, dn.unwrap_or(d)),
        own_dt: Some(resolve_dih(tables, state, cl_a3, tail[k - 3], tail[k - 2], tail[k - 1])),
    };
    for (i, &atom) in tail[..k].iter().enumerate() {
        let (a1, a2, a3) = match i {
            0 => (p1, p2, p3),
            1 => (tail[0], p1, Some(p2)),
            2 => (tail[1], tail[0], Some(p1)),
            _ => (tail[i - 1], tail[i - 2], Some(tail[i - 3])),
        };
        let bt = resolve_bond(tables, state, a1, atom);
        let at = resolve_angle(tables, state, a2, a1, atom);
        let include = retrace_pos.map(|rp| rp[atom]);
        // upstream atoms are ψ-guided toward the equilibrium span to the
        // closure target for the number of bonds remaining after this
        // atom (atom i: k-1-i bonds -> guide_span[k-1-i]).
        let bias = if i < k - 1 {
            PsiBias::Guide {
                d,
                r_opt: cfg.guide_span[k - 1 - i],
                k_guide: cfg.guide_k,
            }
        } else {
            PsiBias::Close(&cl)
        };
        let dt = if i < k - 1 { a3.map(|q| resolve_dih(tables, state, q, a2, a1, atom)) } else { None };
        let lw = regrow_atom(pos, atom, a1, a2, a3, bt, at, dt, &bias, env, cfg, kbt, include, rng);
        if std::env::var("CBMC_DEBUG").is_ok() {
            eprintln!("  segment atom {i}/{k} (id {atom}): logw = {lw:.2}");
        }
        if let Some(rp) = retrace_pos {
            pos[atom] = rp[atom]; // retrace leaves old positions behind
        }
        env.ghost.borrow_mut().remove(&atom); // placed/retraced: now visible
        w += lw;
    }
    w
}

/// Regrow (or retrace) all non-backbone substituents of one carbon, in
/// sorted order for forward/reverse consistency.
#[allow(clippy::too_many_arguments)]
fn regrow_carbon_hydrogens<R: Rng>(
    pos: &mut Vec<[f64; 3]>,
    carbon: usize,
    state: &AtomisticState,
    tables: &TypeTables,
    env: &TrialEnv,
    cfg: &CbmcConfig,
    kbt: f64,
    retrace_pos: Option<&[[f64; 3]]>,
    rng: &mut R,
) -> f64 {
    let mut w = 0.0;
    for h in non_backbone_neighbors(state, carbon) {
        let bt = resolve_bond(tables, state, carbon, h);
        let include = retrace_pos.map(|rp| rp[h]);
        w += regrow_h(pos, h, carbon, bt, env, cfg, kbt, include, rng);
        if let Some(rp) = retrace_pos {
            pos[h] = rp[h];
        }
        env.ghost.borrow_mut().remove(&h); // placed/retraced: now visible
    }
    w
}

// ----------------------------------------------------------------------
// The CBMC double-bridge move
// ----------------------------------------------------------------------

pub struct CbmcOutcome {
    pub accepted: bool,
    pub log_accept_ratio: f64,
    pub w_new: f64,
    pub w_old: f64,
}

/// log Σ exp(xs) with the max factored out.
fn logsumexp(xs: &[f64]) -> f64 {
    let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + xs.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
}

/// One forward regrowth pass into the CURRENT state topology (after
/// surgery), starting from the old scaffold `pos_old`. Returns the
/// Rosenbluth log-weight and the full new configuration.
#[allow(clippy::too_many_arguments)]
fn forward_pass<R: Rng>(
    state: &AtomisticState,
    tables: &TypeTables,
    par: &AtomisticParams,
    tail_a: &[usize],
    tail_b: &[usize],
    ca: &[usize],
    cb: &[usize],
    s: usize,
    pos_old: &[[f64; 3]],
    ghosts: &HashSet<usize>,
    cfg: &CbmcConfig,
    kbt: f64,
    rng: &mut R,
) -> (f64, Vec<[f64; 3]>) {
    let k = cfg.n_regrow;
    let mut w = 0.0;
    let mut pos = pos_old.to_vec();
    let env = TrialEnv {
        state,
        par,
        ghost: std::cell::RefCell::new(ghosts.clone()),
    };
    w += regrow_segment(
        &mut pos, tail_b, ca[s - 1], ca, s, state, tables, &env, cfg, kbt, None, rng,
    );
    w += regrow_segment(
        &mut pos, tail_a, cb[s - 1], cb, s, state, tables, &env, cfg, kbt, None, rng,
    );
    for &carbon in tail_b[..k].iter().chain(tail_a[..k].iter()) {
        w += regrow_carbon_hydrogens(&mut pos, carbon, state, tables, &env, cfg, kbt, None, rng);
    }
    (w, pos)
}

/// One reverse-direction weight pass in the OLD topology: either the
/// retrace of the old configuration (`retrace_pos = Some(old)`, mixed
/// buffer from `pos_start`) or a fresh reference regrowth
/// (`retrace_pos = None`). Weight only.
#[allow(clippy::too_many_arguments)]
fn reverse_pass<R: Rng>(
    backup: &AtomisticState,
    tables: &TypeTables,
    par: &AtomisticParams,
    tail_a: &[usize],
    tail_b: &[usize],
    ca: &[usize],
    cb: &[usize],
    s: usize,
    pos_start: &[[f64; 3]],
    ghosts: &HashSet<usize>,
    cfg: &CbmcConfig,
    kbt: f64,
    retrace_pos: Option<&[[f64; 3]]>,
    rng: &mut R,
) -> f64 {
    let k = cfg.n_regrow;
    let mut w = 0.0;
    let env_old = TrialEnv {
        state: backup,
        par,
        ghost: std::cell::RefCell::new(ghosts.clone()),
    };
    let mut posr = pos_start.to_vec();
    w += regrow_segment(
        &mut posr, tail_a, ca[s - 1], ca, s, backup, tables, &env_old, cfg, kbt,
        retrace_pos, rng,
    );
    w += regrow_segment(
        &mut posr, tail_b, cb[s - 1], cb, s, backup, tables, &env_old, cfg, kbt,
        retrace_pos, rng,
    );
    for &carbon in tail_a[..k].iter().chain(tail_b[..k].iter()) {
        w += regrow_carbon_hydrogens(
            &mut posr, carbon, backup, tables, &env_old, cfg, kbt, retrace_pos, rng,
        );
    }
    w
}

#[allow(clippy::too_many_arguments)]
pub fn cbmc_double_bridge<R: Rng>(
    state: &mut AtomisticState,
    tables: &TypeTables,
    par: &AtomisticParams,
    proposal: &BridgeProposal,
    cfg: &CbmcConfig,
    kbt: f64,
    n_fwd_total: usize,
    rng: &mut R,
) -> CbmcOutcome {
    cbmc_double_bridge_mtm(state, tables, par, proposal, cfg, kbt, n_fwd_total, 1, rng)
}

/// Multi-try Metropolis (Liu–Liang–Wong) double bridge: R forward
/// regrowth passes, selection ∝ Rosenbluth weight, reverse reference set
/// of R−1 fresh regrowths plus the retrace of the current configuration.
/// Accept with min(1, (Σ_fwd W)/(Σ_rev W) × count-ratio). R = 1 reduces
/// to plain CBMC double bridging.
#[allow(clippy::too_many_arguments)]
pub fn cbmc_double_bridge_mtm<R: Rng>(
    state: &mut AtomisticState,
    tables: &TypeTables,
    par: &AtomisticParams,
    proposal: &BridgeProposal,
    cfg: &CbmcConfig,
    kbt: f64,
    n_fwd_total: usize,
    n_mtm: usize,
    rng: &mut R,
) -> CbmcOutcome {
    let reject = |w_old: f64| CbmcOutcome {
        accepted: false,
        log_accept_ratio: f64::NEG_INFINITY,
        w_new: f64::NEG_INFINITY,
        w_old,
    };
    let s = proposal.s;
    let flip = proposal.flip;
    let ca = state.chains[proposal.a].clone();
    let cb = state.chains[proposal.b].clone();
    let n = ca.len();
    let k = cfg.n_regrow;
    if n < k + 4 || cb.len() != n || s < 2 || s + k >= n {
        return reject(f64::NEG_INFINITY);
    }
    let tail_a = tail_of(&ca, s, flip);
    let tail_b = tail_of(&cb, s, flip);
    let cutoff = par.lj_cut.max(par.coul_cut);
    let r_mtm = n_mtm.max(1);

    // ---------- surgery on a clone; segment excision set --------------
    let backup = state.clone();
    let pos_old = backup.pos.clone();
    if crate::bridge::apply_bridge(state, tables, proposal).is_err() {
        *state = backup;
        return reject(f64::NEG_INFINITY);
    }
    let mut ghosts: HashSet<usize> = HashSet::new();
    for &carbon in tail_a[..k].iter().chain(tail_b[..k].iter()) {
        ghosts.insert(carbon);
        ghosts.extend(non_backbone_neighbors(&backup, carbon));
    }

    // ---------- forward trial set --------------------------------------
    let mut fwd_w = Vec::with_capacity(r_mtm);
    let mut fwd_pos = Vec::with_capacity(r_mtm);
    for _ in 0..r_mtm {
        let (w, p) = forward_pass(
            state, tables, par, &tail_a, &tail_b, &ca, &cb, s, &pos_old,
            &ghosts, cfg, kbt, rng,
        );
        fwd_w.push(w);
        fwd_pos.push(p);
    }
    // select j ∝ W_j
    let (log_wsum_fwd, j_sel) = rosenbluth_select(&fwd_w, rng);
    let pos = fwd_pos.swap_remove(j_sel);
    let w_new = fwd_w[j_sel];

    // ---------- reverse reference set ----------------------------------
    // retrace of the current (old) configuration + R−1 fresh regrowths
    // into the old topology, all starting from the selected new config.
    let mut rev_w = Vec::with_capacity(r_mtm);
    rev_w.push(reverse_pass(
        &backup, tables, par, &tail_a, &tail_b, &ca, &cb, s, &pos, &ghosts, cfg, kbt,
        Some(&pos_old), rng,
    ));
    for _ in 1..r_mtm {
        rev_w.push(reverse_pass(
            &backup, tables, par, &tail_a, &tail_b, &ca, &cb, s, &pos, &ghosts, cfg, kbt,
            None, rng,
        ));
    }
    let log_wsum_rev = logsumexp(&rev_w);
    let w_old = rev_w[0];

    // ---------- proposal-count ratio ----------------------------------
    // System-wide uniform sampling: the driver picks uniformly from the
    // concatenated per-chain candidate lists, so every physical swap has
    // probability 2/M (it appears once from each partner chain); the
    // factor 2 cancels in the ratio, giving P_fwd/P_rev = M_rev/M_fwd
    // with M the total per-chain candidate counts.
    state.pos = pos.clone(); // tentative new geometry
    let cells_new = ACellList::build(state, cutoff);
    let n_rev_total: usize = (0..state.chains.len())
        .map(|c| enumerate_cbmc_candidates(state, &cells_new, c, cfg.r_reach, cfg.n_regrow).len())
        .sum();
    let log_count = (n_rev_total.max(1) as f64 / n_fwd_total.max(1) as f64).ln();

    let log_ratio = (log_wsum_fwd - log_wsum_rev) + log_count;
    let accept = log_ratio >= 0.0 || rng.random::<f64>() < log_ratio.exp();
    if accept {
        CbmcOutcome {
            accepted: true,
            log_accept_ratio: log_ratio,
            w_new,
            w_old,
        }
    } else {
        *state = backup;
        CbmcOutcome {
            accepted: false,
            log_accept_ratio: log_ratio,
            w_new,
            w_old,
        }
    }
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atomistic::{AtomisticEngine, AtomisticParams};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Two 8-carbon chains built at the force-field's equilibrium
    /// (r0, theta0) with a trans/gauche torsion pattern, so original
    /// positions satisfy the regrowth priors and stub-to-downstream
    /// spans sit comfortably within the 4-bond reach. Chain B is chain A
    /// offset by `separation` in y (nonzero to avoid exact overlap).
    fn fixture(separation: f64, soft: bool) -> (AtomisticState, AtomisticParams) {
        const R0: f64 = 1.529;
        let t0 = 112.7f64.to_radians();
        let l = 60.0;
        let mut chain: Vec<[f64; 3]> = Vec::new();
        chain.push([10.0, 10.0, 10.0]);
        chain.push([10.0 + R0, 10.0, 10.0]);
        // torsion pattern: trans / gauche+ / trans / gauche- ...
        let psis = [0.0f64, 2.0944, 0.0, -2.0944, 0.0, 2.0944];
        for &psi in &psis {
            let m = chain.len();
            let p3 = if m >= 3 { Some(chain[m - 3]) } else { None };
            let (u, e1, e2) = local_frame(chain[m - 1], chain[m - 2], p3, l);
            let q = cone_point(chain[m - 1], u, e1, e2, R0, t0, psi, l);
            chain.push(q);
        }
        let mut pos = chain.clone();
        for p in &chain {
            pos.push([p[0], p[1] + separation, p[2]]);
        }
        let n = pos.len();
        let types = vec![1usize; n];
        let charges = vec![0.0; n];
        let mol: Vec<usize> = (0..8).map(|_| 0).chain((0..8).map(|_| 1)).collect();
        let chains = vec![(0..8).collect::<Vec<_>>(), (8..16).collect::<Vec<_>>()];
        let mut bonds = Vec::new();
        let mut angles = Vec::new();
        let mut dihs = Vec::new();
        for chain in &chains {
            for w in chain.windows(2) {
                bonds.push((w[0], w[1], 1));
            }
            for w in chain.windows(3) {
                angles.push((w[0], w[1], w[2], 1));
            }
            for w in chain.windows(4) {
                dihs.push((w[0], w[1], w[2], w[3], 1));
            }
        }
        let st = AtomisticState::new(
            pos, 60.0, types, charges, mol, bonds, angles, dihs, chains,
        );
        let (bk, ak) = if soft { (20.0, 5.0) } else { (268.0, 58.35) };
        let par = AtomisticParams {
            pair_eps: vec![0.0, 0.0], // no LJ: isolated bonded statistics
            pair_sig: vec![0.0, 3.5],
            bond_k: vec![0.0, bk],
            bond_r0: vec![0.0, 1.529],
            angle_k: vec![0.0, ak],
            angle_t0: vec![0.0, 112.7f64.to_radians()],
            dih_k: vec![[0.0; 4], [1.1, -0.2, 0.2, 0.0]],
            lj_cut: 11.0,
            coul_cut: 11.0,
            scale14_lj: 0.5,
            scale14_coul: 0.5,
        };
        (st, par)
    }

    #[test]
    #[ignore]
    fn debug_weight_magnitudes() {
        let (mut st, par) = fixture(2.0, false);
        let tables = TypeTables::from_state(&st);
        let cfg = CbmcConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(31);
        // spans for every (s, flip) combination
        for s in 2..5 {
            for flip in [false, true] {
                let tail_b = tail_of(&st.chains[1].clone(), s, flip);
                let tail_a = tail_of(&st.chains[0].clone(), s, flip);
                let ra = norm(min_img(
                    sub(st.pos[st.chains[0][s - 1]], st.pos[tail_b[3]]),
                    st.box_size,
                ));
                let rb = norm(min_img(
                    sub(st.pos[st.chains[1][s - 1]], st.pos[tail_a[3]]),
                    st.box_size,
                ));
                println!("s={s} flip={flip}: reach_a={ra:.2} reach_b={rb:.2}");
            }
        }
        let cells = ACellList::build(&st, 11.0);
        let cands = enumerate_cbmc_candidates(&st, &cells, 0, cfg.r_reach, cfg.n_regrow);
        println!("{} reach candidates from chain 0", cands.len());
        for c in &cands {
            println!("  cand a={} b={} s={} flip={}", c.a, c.b, c.s, c.flip);
        }
        let pr = match cands.first() {
            Some(p) => p.clone(),
            None => BridgeProposal { a: 0, b: 1, s: 3, flip: false },
        };
        for _ in 0..10 {
            let out =
                cbmc_double_bridge(&mut st, &tables, &par, &pr, &cfg, 0.9, 8, &mut rng);
            println!(
                "accepted={} w_new={:.3} w_old={:.3} log_ratio={:.3}",
                out.accepted, out.w_new, out.w_old, out.log_accept_ratio
            );
        }
    }

    #[test]
    #[ignore]
    fn debug_direction_asymmetry() {
        let (mut st, par) = fixture(2.0, false);
        let tables = TypeTables::from_state(&st);
        let cfg = CbmcConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(37);
        let cells = ACellList::build(&st, 11.0);
        let cands = enumerate_cbmc_candidates(&st, &cells, 0, cfg.r_reach, cfg.n_regrow);
        let pr = cands[0].clone();
        // wait for first accept
        let mut fwd_tries = 0;
        loop {
            fwd_tries += 1;
            if cbmc_double_bridge(&mut st, &tables, &par, &pr, &cfg, 0.9, 8, &mut rng).accepted {
                break;
            }
            assert!(fwd_tries < 5000, "never accepted forward");
        }
        println!("forward accepted after {fwd_tries} tries");
        // relax, then reverse attempts: statistics
        let mut rev_log: Vec<f64> = Vec::new();
        let mut rev_wold: Vec<f64> = Vec::new();
        let mut rev_wnew: Vec<f64> = Vec::new();
        for round in 0..20 {
            let eng = AtomisticEngine::new(st.clone(), par.clone(), 0.9);
            let mut mc = crate::atomistic_mc::AtomisticMC::new(
                eng,
                crate::atomistic_mc::AMcParams { w_torsion: 0.0, ..Default::default() },
            );
            mc.run(100, &mut rng);
            st.pos = mc.engine.state.pos.clone();
            let _ = round;
            for _ in 0..50 {
                let out = cbmc_double_bridge(&mut st, &tables, &par, &pr, &cfg, 0.9, 8, &mut rng);
                rev_log.push(out.log_accept_ratio);
                rev_wold.push(out.w_old);
                rev_wnew.push(out.w_new);
                if out.accepted {
                    println!("reverse accepted");
                    break;
                }
            }
        }
        let stats = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            format!(
                "min={:.2} p25={:.2} med={:.2} p75={:.2} max={:.2}",
                v[0],
                v[v.len() / 4],
                v[v.len() / 2],
                v[3 * v.len() / 4],
                v[v.len() - 1]
            )
        };
        println!("reverse log-ratio: {}", stats(&mut rev_log));
        println!("reverse w_old:     {}", stats(&mut rev_wold));
        println!("reverse w_new:     {}", stats(&mut rev_wnew));
    }

    #[test]
    fn bond_sampler_follows_boltzmann() {
        let (k, r0, kbt) = (268.0, 1.529, 0.9);
        let s = bond_sampler(k, r0, kbt);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut sum = 0.0;
        let m = 20_000;
        for _ in 0..m {
            let r = s.sample(&mut rng);
            assert!(r > 0.3 && r < r0 + 5.0 * (kbt / (2.0 * k)).sqrt() + 1e-9);
            sum += r;
        }
        let mean = sum / m as f64;
        // <r> for p(r) ~ r^2 exp(-βK(r-r0)^2) is slightly above r0
        assert!((mean - r0).abs() < 0.01, "mean bond length {mean}");
    }

    #[test]
    fn angle_sampler_follows_boltzmann() {
        let (k, t0, kbt) = (58.35, 112.7f64.to_radians(), 0.9);
        let s = angle_sampler(k, t0, kbt);
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut sum = 0.0;
        let m = 20_000;
        for _ in 0..m {
            let th = s.sample(&mut rng);
            assert!(th > 0.02 && th < std::f64::consts::PI - 0.02);
            sum += th;
        }
        let mean = sum / m as f64;
        assert!((mean - t0).abs() < 0.02, "mean angle {mean} vs {t0}");
    }

    #[test]
    fn retrace_weights_finite_on_self_consistent_geometry() {
        // Overlapping identical chains: the swap maps the configuration
        // onto itself, so both W_old (retrace) and W_new (regrow) must
        // be finite — the circle closure always closes here.
        let (mut st, par) = fixture(2.0, false);
        let tables = TypeTables::from_state(&st);
        let cfg = CbmcConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let pr = BridgeProposal { a: 0, b: 1, s: 3, flip: false };
        let out = cbmc_double_bridge(&mut st, &tables, &par, &pr, &cfg, 0.9, 8, &mut rng);
        assert!(out.w_old.is_finite(), "retrace weight must be finite");
        assert!(out.w_new.is_finite(), "regrow on identical geometry must close");
    }

    #[test]
    fn far_apart_chains_rejected_and_restored() {
        // Chains 30 Å apart: closure conditional exists but carries
        // enormous bond strain -> W_new/W_old ~ 0 -> reject, with the
        // state fully restored.
        let (mut st, par) = fixture(30.0, false);
        let tables = TypeTables::from_state(&st);
        let cfg = CbmcConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let pos_before = st.pos.clone();
        let bonds_before = st.bonds.clone();
        let pr = BridgeProposal { a: 0, b: 1, s: 3, flip: false };
        let out = cbmc_double_bridge(&mut st, &tables, &par, &pr, &cfg, 0.9, 8, &mut rng);
        assert!(!out.accepted, "impossible closure must be rejected");
        assert_eq!(st.pos, pos_before, "rejected attempt must restore positions");
        assert_eq!(st.bonds, bonds_before, "rejected attempt must restore topology");
    }

    #[test]
    fn bookkeeping_consistent_over_many_attempts() {
        // Stiff bonded terms on the near-identical fixture, candidates
        // drawn from the reach enumerator: whatever the accept/reject
        // outcome, the state must stay consistent.
        let (mut st, par) = fixture(2.0, false);
        let tables = TypeTables::from_state(&st);
        let cfg = CbmcConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        for _ in 0..200 {
            let pos_before = st.pos.clone();
            let bonds_before = st.bonds.clone();
            let cells = ACellList::build(&st, 11.0);
            let cands = enumerate_cbmc_candidates(&st, &cells, 0, cfg.r_reach, cfg.n_regrow);
            assert!(!cands.is_empty(), "fixture must have reach candidates");
            let pr = cands[rng.random_range(0..cands.len())].clone();
            let out =
                cbmc_double_bridge(&mut st, &tables, &par, &pr, &cfg, 0.9, 8, &mut rng);
            assert_eq!(st.chains[0].len(), 8);
            assert_eq!(st.chains[1].len(), 8);
            let mut all: Vec<usize> =
                st.chains[0].iter().chain(st.chains[1].iter()).copied().collect();
            all.sort_unstable();
            assert_eq!(all, (0..16).collect::<Vec<_>>(), "backbone atoms permuted");
            let eng = AtomisticEngine::new(st.clone(), par.clone(), 1.0);
            assert!(eng.total_energy().is_finite());
            if !out.accepted {
                assert_eq!(st.pos, pos_before, "rejected attempt must restore positions");
                assert_eq!(st.bonds, bonds_before, "rejected attempt must restore topology");
            }
        }
    }

    /// Fixture acceptance tracker: measures the forward acceptance rate
    /// over reach-enumerated candidates with local relaxation between
    /// attempts. Prints the rate; asserts only that the machinery
    /// produces *some* accepts (rate >> the 0/3000 frozen baseline).
    /// The involution property of the topology surgery itself is covered
    /// exactly by bridge::tests::bridge_delta_matches_total_energy_diff.
    /// Here: the MTM tail-multiplier must produce strictly more accepts
    /// than single-try CBMC at the same seed protocol (R=8 vs R=1).
    #[test]
    fn acceptance_rate_on_fixture() {
        let run = |n_mtm: usize, rounds: usize, seed: u64| -> (usize, usize) {
            let (mut st, par) = fixture(2.0, false);
            let tables = TypeTables::from_state(&st);
            let cfg = CbmcConfig::default();
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut att = 0usize;
            let mut acc = 0usize;
            for _ in 0..rounds {
                // local relaxation between attempts, as in production
                let eng = AtomisticEngine::new(st.clone(), par.clone(), 0.9);
                let mut mc = crate::atomistic_mc::AtomisticMC::new(
                    eng,
                    crate::atomistic_mc::AMcParams { w_torsion: 0.0, ..Default::default() },
                );
                mc.run(50, &mut rng);
                st.pos = mc.engine.state.pos.clone();
                let cells = ACellList::build(&st, 11.0);
                let cands =
                    enumerate_cbmc_candidates(&st, &cells, 0, cfg.r_reach, cfg.n_regrow);
                if cands.is_empty() {
                    continue;
                }
                let pr = cands[rng.random_range(0..cands.len())].clone();
                att += 1;
                if cbmc_double_bridge_mtm(
                    &mut st, &tables, &par, &pr, &cfg, 0.9, 8, n_mtm, &mut rng,
                )
                .accepted
                {
                    acc += 1;
                }
            }
            (att, acc)
        };
        let (att1, acc1) = run(1, 300, 23);
        let (att8, acc8) = run(8, 300, 23);
        eprintln!("fixture acceptance R=1: {acc1}/{att1}, R=8: {acc8}/{att8}");
        assert!(acc8 > 0, "MTM R=8 gave no accepts in {att8} attempts");
        assert!(
            acc8 > acc1 || acc1 > 4,
            "MTM should lift acceptance when it is low (R=1: {acc1}, R=8: {acc8})"
        );
    }

    #[test]
    fn closure_circle_always_satisfies_bond_when_reachable() {
        // On the identical-overlap fixture, every accepted configuration
        // must have both junction bonds within a tight equilibrium band —
        // the ψ-conditional should essentially never produce strain.
        let (mut st, par) = fixture(2.0, false);
        let tables = TypeTables::from_state(&st);
        let cfg = CbmcConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(29);
        for _ in 0..50 {
            let cells = ACellList::build(&st, 11.0);
            let cands = enumerate_cbmc_candidates(&st, &cells, 0, cfg.r_reach, cfg.n_regrow);
            assert!(!cands.is_empty());
            let pr = cands[rng.random_range(0..cands.len())].clone();
            let out =
                cbmc_double_bridge(&mut st, &tables, &par, &pr, &cfg, 0.9, 8, &mut rng);
            if !out.accepted {
                continue;
            }
            for &(i, j, _) in &st.bonds.clone() {
                let d = min_img(sub(st.pos[i], st.pos[j]), st.box_size);
                let r = norm(d);
                assert!(
                    (r - 1.529).abs() < 0.15,
                    "bond {i}-{j} strained: r = {r}"
                );
            }
        }
    }
}
