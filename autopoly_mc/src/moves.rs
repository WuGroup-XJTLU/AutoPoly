//! Monte Carlo moves. Every move proposes a new configuration; the engine
//! evaluates the *local* energy difference (old state first, then trial)
//! and commits or rolls back via Metropolis.
//!
//! Phase 0 ports the connectivity-preserving suite from
//! `AutoPoly.models.bead_spring`: displacement, pivot, crankshaft,
//! reptation, chain translation, chain rotation.

use crate::state::MeltState;
use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MoveKind {
    Displacement,
    Pivot,
    Crankshaft,
    Reptation,
    Translation,
    Rotation,
}

impl MoveKind {
    #[allow(dead_code)] // convenience for drivers
    pub const ALL: [MoveKind; 6] = [
        MoveKind::Displacement,
        MoveKind::Pivot,
        MoveKind::Crankshaft,
        MoveKind::Reptation,
        MoveKind::Translation,
        MoveKind::Rotation,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MoveKind::Displacement => "displacement",
            MoveKind::Pivot => "pivot",
            MoveKind::Crankshaft => "crankshaft",
            MoveKind::Reptation => "reptation",
            MoveKind::Translation => "translation",
            MoveKind::Rotation => "rotation",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MoveParams {
    pub max_displacement: f64,
    pub max_angle: f64,
}

impl Default for MoveParams {
    fn default() -> Self {
        MoveParams {
            max_displacement: 0.5,
            max_angle: 0.3,
        }
    }
}

pub struct Proposal {
    /// Beads whose energy must be re-evaluated (sorted/deduped later).
    pub moved: Vec<usize>,
    pub new_pos: Vec<(usize, [f64; 3])>,
    /// Pre-move positions for rollback (same beads as new_pos).
    pub old_pos: Vec<(usize, [f64; 3])>,
    /// Reverse the chain's contour order after committing (reptation).
    pub reverse_chain: Option<usize>,
}

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
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Rodrigues rotation: v rotated by `theta` about unit axis `u`.
#[inline]
fn rotate(v: [f64; 3], u: [f64; 3], theta: f64) -> [f64; 3] {
    let (c, s) = (theta.cos(), theta.sin());
    let kv = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let cx = cross(u, v);
    add(add(scale(v, c), scale(cx, s)), scale(u, kv * (1.0 - c)))
}

/// Isotropic random unit vector (Gaussian projection via Box-Muller).
fn random_unit<R: Rng>(rng: &mut R) -> [f64; 3] {
    let g = |rng: &mut R| -> f64 {
        let u1: f64 = rng.random::<f64>().max(1e-300);
        let u2: f64 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let mut v = [g(rng), g(rng), g(rng)];
    let mut n = norm(v);
    while n < 1e-12 {
        v = [g(rng), g(rng), g(rng)];
        n = norm(v);
    }
    scale(v, 1.0 / n)
}

/// Attempt to generate a move of `kind` on chain `c`.
/// Returns None if the move is geometrically inapplicable.
pub fn propose<R: Rng>(
    kind: MoveKind,
    state: &MeltState,
    c: usize,
    params: &MoveParams,
    rng: &mut R,
) -> Option<Proposal> {
    let chain = &state.chains[c];
    let n = chain.len();
    match kind {
        MoveKind::Displacement => {
            let i = rng.random_range(0..n);
            let bead = chain[i];
            let d = [
                rng.random_range(-params.max_displacement..params.max_displacement),
                rng.random_range(-params.max_displacement..params.max_displacement),
                rng.random_range(-params.max_displacement..params.max_displacement),
            ];
            Some(Proposal {
                moved: vec![bead],
                new_pos: vec![(bead, state.wrap(add(state.pos[bead], d)))],
                old_pos: vec![(bead, state.pos[bead])],
                reverse_chain: None,
            })
        }

        MoveKind::Pivot => {
            // Rotate the contour arm after a pivot bead by a small random
            // rotation (symmetric => no proposal bias).
            if n < 2 {
                return None;
            }
            let pivot_local = rng.random_range(0..n - 1);
            let pivot_bead = chain[pivot_local];
            let pivot_pos = state.pos[pivot_bead];
            let axis = random_unit(rng);
            let theta = rng.random_range(-params.max_angle..params.max_angle);
            let mut moved = Vec::with_capacity(n - pivot_local - 1);
            let mut new_pos = Vec::with_capacity(n - pivot_local - 1);
            for &bead in &chain[pivot_local + 1..] {
                let rel = state.min_image(sub(state.pos[bead], pivot_pos));
                let np = state.wrap(add(pivot_pos, rotate(rel, axis, theta)));
                moved.push(bead);
                new_pos.push((bead, np));
            }
            let old_pos = moved.iter().map(|&b| (b, state.pos[b])).collect();
            Some(Proposal {
                moved,
                new_pos,
                old_pos,
                reverse_chain: None,
            })
        }

        MoveKind::Crankshaft => {
            if n < 4 {
                return None;
            }
            let i = rng.random_range(0..n - 3);
            let j = rng.random_range(i + 3..n);
            let bi = chain[i];
            let bj = chain[j];
            let pi = state.pos[bi];
            let axis_d = state.disp(bi, bj);
            let alen = norm(axis_d);
            if alen < 1e-10 {
                return None;
            }
            let u = scale(axis_d, 1.0 / alen);
            let theta = rng.random_range(-params.max_angle..params.max_angle);
            let mut moved = Vec::with_capacity(j - i - 1);
            let mut new_pos = Vec::with_capacity(j - i - 1);
            for &bead in &chain[i + 1..j] {
                let rel = state.min_image(sub(state.pos[bead], pi));
                let np = state.wrap(add(pi, rotate(rel, u, theta)));
                moved.push(bead);
                new_pos.push((bead, np));
            }
            let old_pos = moved.iter().map(|&b| (b, state.pos[b])).collect();
            Some(Proposal {
                moved,
                new_pos,
                old_pos,
                reverse_chain: None,
            })
        }

        MoveKind::Reptation => {
            if n < 2 {
                return None;
            }
            let forward = rng.random::<bool>();
            // Every bead shifts one contour slot toward the depleted end;
            // a fresh bead is appended at the other end at unit bond
            // length in an isotropic direction. Internal bond lengths are
            // preserved by the shift, so only the new bond and the moved
            // beads' nonbonded terms enter the acceptance test.
            let anchor = if forward { chain[n - 1] } else { chain[0] };
            let new_end = state.wrap(add(state.pos[anchor], random_unit(rng)));

            let mut moved = Vec::with_capacity(n);
            let mut new_pos = Vec::with_capacity(n);
            if forward {
                for k in 0..n - 1 {
                    moved.push(chain[k]);
                    new_pos.push((chain[k], state.pos[chain[k + 1]]));
                }
                moved.push(chain[n - 1]);
                new_pos.push((chain[n - 1], new_end));
            } else {
                for k in (1..n).rev() {
                    moved.push(chain[k]);
                    new_pos.push((chain[k], state.pos[chain[k - 1]]));
                }
                moved.push(chain[0]);
                new_pos.push((chain[0], new_end));
            }
            let old_pos = moved.iter().map(|&b| (b, state.pos[b])).collect();
            Some(Proposal {
                moved,
                new_pos,
                old_pos,
                reverse_chain: Some(c),
            })
        }

        MoveKind::Translation => {
            let d = [
                rng.random_range(-params.max_displacement..params.max_displacement),
                rng.random_range(-params.max_displacement..params.max_displacement),
                rng.random_range(-params.max_displacement..params.max_displacement),
            ];
            let mut moved = Vec::with_capacity(n);
            let mut new_pos = Vec::with_capacity(n);
            for &bead in chain {
                moved.push(bead);
                new_pos.push((bead, state.wrap(add(state.pos[bead], d))));
            }
            let old_pos = moved.iter().map(|&b| (b, state.pos[b])).collect();
            Some(Proposal {
                moved,
                new_pos,
                old_pos,
                reverse_chain: None,
            })
        }

        MoveKind::Rotation => {
            if n < 2 {
                return None;
            }
            // Center of mass under PBC: accumulate minimum-image offsets
            // from the first bead (valid when the chain spans < box/2,
            // which holds for any chain that fits in the box).
            let p0 = state.pos[chain[0]];
            let mut acc = [0.0f64; 3];
            for &bead in chain {
                acc = add(acc, state.min_image(sub(state.pos[bead], p0)));
            }
            let com = state.wrap(add(p0, scale(acc, 1.0 / n as f64)));

            let axis = random_unit(rng);
            let theta = rng.random_range(-params.max_angle..params.max_angle);
            let mut moved = Vec::with_capacity(n);
            let mut new_pos = Vec::with_capacity(n);
            for &bead in chain {
                let rel = state.min_image(sub(state.pos[bead], com));
                let np = state.wrap(add(com, rotate(rel, axis, theta)));
                moved.push(bead);
                new_pos.push((bead, np));
            }
            let old_pos = moved.iter().map(|&b| (b, state.pos[b])).collect();
            Some(Proposal {
                moved,
                new_pos,
                old_pos,
                reverse_chain: None,
            })
        }
    }
}
