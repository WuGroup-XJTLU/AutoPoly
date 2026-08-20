//! Directed double-bridging for atomistic chains (clone-mutate-diff).
//!
//! Cut two chains at the same backbone contour position and swap tails.
//! Because atomistic bonded terms are stiff, candidates are prefiltered
//! by junction geometry (new bond length and junction angle windows).
//!
//! ΔU strategy: clone the state, apply the full topology mutation
//! (bonds, angles, dihedrals, pair classification, chains), diff the two
//! states to find the exact affected set, and evaluate
//!   ΔU = local_energy(new, affected) - local_energy(old, affected).
//! This is exact by construction and verified against total-energy
//! differences in the test suite. Slower per attempt than the KG path —
//! bridges are attempted at low frequency, so this is the right trade.

use crate::atomistic::{AtomisticEngine, AtomisticParams, AtomisticState};
use rand::Rng;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct BridgeProposal {
    pub a: usize,
    pub b: usize,
    pub s: usize,
    pub flip: bool,
}

/// By-example type tables for new bonded terms (homopolymer-friendly).
pub struct TypeTables {
    pub bond: HashMap<(usize, usize), usize>,
    pub angle: HashMap<(usize, usize, usize), usize>,
    pub dihedral: HashMap<(usize, usize, usize, usize), usize>,
}

impl TypeTables {
    pub fn from_state(state: &AtomisticState) -> Self {
        let mut bond = HashMap::new();
        let mut angle = HashMap::new();
        let mut dihedral = HashMap::new();
        for &(i, j, t) in &state.bonds {
            let key = (state.types[i].min(state.types[j]), state.types[i].max(state.types[j]));
            bond.entry(key).or_insert(t);
        }
        for &(i, j, k, t) in &state.angles {
            angle
                .entry((state.types[i], state.types[j], state.types[k]))
                .or_insert(t);
        }
        for &(i, j, k, l, t) in &state.dihedrals {
            dihedral
                .entry((state.types[i], state.types[j], state.types[k], state.types[l]))
                .or_insert(t);
        }
        TypeTables { bond, angle, dihedral }
    }
}

/// Geometry gate for a candidate junction: both new bonds within
/// [jb_min, jb_max] and both junction angles within [ja_min, ja_max].
pub struct BridgeGate {
    pub jb_min: f64,
    pub jb_max: f64,
    pub ja_min: f64,
    pub ja_max: f64,
    pub r_max: f64,
}

impl Default for BridgeGate {
    fn default() -> Self {
        BridgeGate {
            jb_min: 1.30,
            jb_max: 1.75,
            ja_min: 1.65, // ~94 deg
            ja_max: 2.35, // ~135 deg
            r_max: 1.9,
        }
    }
}

// ----------------------------------------------------------------------
// Topology mutation
// ----------------------------------------------------------------------

/// Apply the bridge swap to `state` in place. Returns Err (leaving the
/// state possibly partially mutated — callers must not reuse it) if a new
/// term cannot be type-resolved.
pub fn apply_bridge(
    state: &mut AtomisticState,
    tables: &TypeTables,
    pr: &BridgeProposal,
) -> Result<(), String> {
    let n = state.chains[pr.a].len();
    let s = pr.s;
    let ca = state.chains[pr.a].clone();
    let cb = state.chains[pr.b].clone();
    let a1 = ca[s - 1];
    let a2 = ca[s];
    let b1 = cb[s - 1];
    let b2 = cb[s];
    let (x, y) = if pr.flip {
        (*cb.last().unwrap(), *ca.last().unwrap())
    } else {
        (b2, a2)
    };

    // --- add terms containing the new bonds ---
    // adjacency after swap (computed from the pre-mutation bond list)
    let adj = |u: usize| -> Vec<usize> {
        let mut v: Vec<usize> = neighbors_of(state, u)
            .into_iter()
            .filter(|&w| !((u == a1 && w == a2) || (u == a2 && w == a1)
                || (u == b1 && w == b2) || (u == b2 && w == b1)))
            .collect();
        if u == a1 { v.push(x); }
        if u == x { v.push(a1); }
        if u == b1 { v.push(y); }
        if u == y { v.push(b1); }
        v
    };
    let mut new_angles: HashSet<(usize, usize, usize)> = HashSet::new();
    let mut new_dih: HashSet<(usize, usize, usize, usize)> = HashSet::new();
    for &(u, v) in &[(a1, x), (b1, y)] {
        for &w in &adj(u) {
            if w != v {
                new_angles.insert((w, u, v));
            }
        }
        for &w in &adj(v) {
            if w != u {
                new_angles.insert((u, v, w));
            }
        }
        // dihedrals with (u,v) central: (w,u,v,z)
        for &w in &adj(u) {
            if w == v { continue; }
            for &z in &adj(v) {
                if z == u || z == w { continue; }
                new_dih.insert((w, u, v, z));
            }
        }
        // dihedrals with (u,v) outer: (p,q,u,v) and (u,v,q,p)
        for &q in &adj(u) {
            if q == v { continue; }
            for &p in &adj(q) {
                if p == u || p == q || p == v { continue; }
                new_dih.insert((p, q, u, v));
            }
        }
        for &q in &adj(v) {
            if q == u { continue; }
            for &p in &adj(q) {
                if p == v || p == q || p == u { continue; }
                new_dih.insert((u, v, q, p));
            }
        }
    }
    // --- pre-validate: resolve every new term's type BEFORE mutating ---
    let t_ax = resolve(&tables.bond, state, a1, x, "bond")?;
    let t_by = resolve(&tables.bond, state, b1, y, "bond")?;
    let mut typed_angles = Vec::with_capacity(new_angles.len());
    for (i, j, k) in &new_angles {
        let key = (state.types[*i], state.types[*j], state.types[*k]);
        let key_r = (key.2, key.1, key.0);
        let t = tables
            .angle
            .get(&key)
            .or_else(|| tables.angle.get(&key_r))
            .copied()
            .ok_or_else(|| format!("no angle type for {key:?}"))?;
        typed_angles.push((*i, *j, *k, t));
    }
    let mut typed_dih = Vec::with_capacity(new_dih.len());
    for (i, j, k, l) in &new_dih {
        let key = (state.types[*i], state.types[*j], state.types[*k], state.types[*l]);
        let key_r = (key.3, key.2, key.1, key.0);
        let t = tables
            .dihedral
            .get(&key)
            .or_else(|| tables.dihedral.get(&key_r))
            .copied()
            .ok_or_else(|| format!("no dihedral type for {key:?}"))?;
        typed_dih.push((*i, *j, *k, *l, t));
    }

    // --- mutate (nothing below can fail) ---
    state.bonds.retain(|&(i, j, _)| {
        !(i == a1 && j == a2) && !(i == a2 && j == a1)
            && !(i == b1 && j == b2) && !(i == b2 && j == b1)
    });
    state.bonds.push((a1, x, t_ax));
    state.bonds.push((b1, y, t_by));
    state.angles.retain(|&(i, j, k, _)| {
        !contains_pair3(i, j, k, a1, a2) && !contains_pair3(i, j, k, b1, b2)
    });
    state.dihedrals.retain(|&(i, j, k, l, _)| {
        !contains_pair4(i, j, k, l, a1, a2) && !contains_pair4(i, j, k, l, b1, b2)
    });
    for (i, j, k, t) in typed_angles {
        state.angles.push((i, j, k, t));
    }
    for (i, j, k, l, t) in typed_dih {
        state.dihedrals.push((i, j, k, l, t));
    }

    // --- chains ---
    let mut new_a: Vec<usize> = ca[..s].to_vec();
    let tail_b: Vec<usize> = if pr.flip {
        cb[s..].iter().rev().copied().collect()
    } else {
        cb[s..].to_vec()
    };
    new_a.extend_from_slice(&tail_b);
    let mut new_b: Vec<usize> = cb[..s].to_vec();
    let tail_a: Vec<usize> = if pr.flip {
        ca[s..].iter().rev().copied().collect()
    } else {
        ca[s..].to_vec()
    };
    new_b.extend_from_slice(&tail_a);
    state.chains[pr.a] = new_a;
    state.chains[pr.b] = new_b;

    // --- rebuild derived structures globally (simple + exact) ---
    rebuild_index(state);
    Ok(())
}

/// Full rebuild of per-atom term lists and pair classifications.
pub fn rebuild_index(state: &mut AtomisticState) {
    let n = state.pos.len();
    state.bonds_of = vec![Vec::new(); n];
    state.angles_of = vec![Vec::new(); n];
    state.dihedrals_of = vec![Vec::new(); n];
    state.excl = vec![HashSet::new(); n];
    state.scaled14 = HashSet::new();
    for (t, &(i, j, _)) in state.bonds.iter().enumerate() {
        state.bonds_of[i].push(t);
        state.bonds_of[j].push(t);
        state.excl[i].insert(j);
        state.excl[j].insert(i);
    }
    for (t, &(i, j, k, _)) in state.angles.iter().enumerate() {
        state.angles_of[i].push(t);
        state.angles_of[j].push(t);
        state.angles_of[k].push(t);
        state.excl[i].insert(k);
        state.excl[k].insert(i);
    }
    for (t, &(i, j, k, l, _)) in state.dihedrals.iter().enumerate() {
        state.dihedrals_of[i].push(t);
        state.dihedrals_of[j].push(t);
        state.dihedrals_of[k].push(t);
        state.dihedrals_of[l].push(t);
        state.scaled14.insert((i.min(l), i.max(l)));
    }
}

fn resolve(
    table: &HashMap<(usize, usize), usize>,
    state: &AtomisticState,
    i: usize,
    j: usize,
    kind: &str,
) -> Result<usize, String> {
    let key = (state.types[i].min(state.types[j]), state.types[i].max(state.types[j]));
    table
        .get(&key)
        .copied()
        .ok_or_else(|| format!("no {kind} type for atom types {key:?}"))
}

pub fn neighbors_of(state: &AtomisticState, u: usize) -> Vec<usize> {
    let mut v = Vec::new();
    for &t in &state.bonds_of[u] {
        let (i, j, _) = state.bonds[t];
        v.push(if i == u { j } else { i });
    }
    v
}

fn contains_pair3(i: usize, j: usize, k: usize, a: usize, b: usize) -> bool {
    (i == a && j == b) || (i == b && j == a) || (j == a && k == b) || (j == b && k == a)
}

fn contains_pair4(i: usize, j: usize, k: usize, l: usize, a: usize, b: usize) -> bool {
    (i == a && j == b)
        || (i == b && j == a)
        || (j == a && k == b)
        || (j == b && k == a)
        || (k == a && l == b)
        || (k == b && l == a)
}

// ----------------------------------------------------------------------
// ΔU via clone-mutate-diff
// ----------------------------------------------------------------------

/// Exact ΔU of a bridge proposal, or None if the move is not applicable
/// (type resolution failure).
pub fn bridge_delta(
    engine: &AtomisticEngine,
    tables: &TypeTables,
    pr: &BridgeProposal,
) -> Option<f64> {
    let old = &engine.state;
    let mut new = old.clone();
    apply_bridge(&mut new, tables, pr).ok()?;

    // affected set: atoms appearing in the symmetric difference of the
    // bond/angle/dihedral term lists, plus atoms whose 1-2/1-3/1-4 pair
    // classification changed
    let mut affected: HashSet<usize> = HashSet::new();
    diff_terms(&old.bonds, &new.bonds, 2, &mut affected);
    diff_terms(&old.angles, &new.angles, 3, &mut affected);
    diff_terms(&old.dihedrals, &new.dihedrals, 4, &mut affected);
    // pair-class changes: compare excl/scaled14
    for u in 0..old.pos.len() {
        if old.excl[u] != new.excl[u] {
            affected.insert(u);
            affected.extend(old.excl[u].symmetric_difference(&new.excl[u]).copied());
        }
    }
    let all14: HashSet<(usize, usize)> =
        old.scaled14.union(&new.scaled14).copied().collect();
    for &(i, j) in &all14 {
        let o = old.scaled14.contains(&(i, j));
        let nn = new.scaled14.contains(&(i, j));
        if o != nn {
            affected.insert(i);
            affected.insert(j);
        }
    }
    let moved: Vec<usize> = affected.into_iter().collect();
    let eng_new = AtomisticEngine {
        state: new,
        params: engine.params.clone(),
        temperature: engine.temperature,
    };
    let e_old = engine.local_energy(&moved);
    let e_new = eng_new.local_energy(&moved);
    Some(e_new - e_old)
}

fn diff_terms<T>(old: &[T], new: &[T], width: usize, out: &mut HashSet<usize>)
where
    T: TupleAtoms + Eq + std::hash::Hash,
{
    let old_set: HashSet<&T> = old.iter().collect();
    let new_set: HashSet<&T> = new.iter().collect();
    for t in old_set.symmetric_difference(&new_set) {
        for &a in t.atoms().iter().take(width) {
            out.insert(a);
        }
    }
}

trait TupleAtoms {
    fn atoms(&self) -> Vec<usize>;
}
impl TupleAtoms for (usize, usize, usize) {
    fn atoms(&self) -> Vec<usize> {
        vec![self.0, self.1]
    }
}
impl TupleAtoms for (usize, usize, usize, usize) {
    fn atoms(&self) -> Vec<usize> {
        vec![self.0, self.1, self.2]
    }
}
impl TupleAtoms for (usize, usize, usize, usize, usize) {
    fn atoms(&self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3]
    }
}

// ----------------------------------------------------------------------
// Directed candidate search
// ----------------------------------------------------------------------

/// Loose contact-gate enumeration for HMC healing: junction backbone
/// atoms within `gate.r_max`, no bond/angle windows (the junction forms
/// stretched and is healed by MD). Returns proposals only (no ΔU).
pub fn enumerate_bridges_loose(
    engine: &AtomisticEngine,
    gate: &BridgeGate,
    a: usize,
    cell_cand: &dyn Fn([f64; 3]) -> Vec<usize>,
) -> Vec<BridgeProposal> {
    let state = &engine.state;
    let chains = &state.chains;
    let n = chains[a].len();
    if n < 5 {
        return Vec::new();
    }
    let mut where_is: HashMap<usize, (usize, usize)> = HashMap::new();
    for (c, chain) in chains.iter().enumerate() {
        for (i, &b) in chain.iter().enumerate() {
            where_is.insert(b, (c, i));
        }
    }
    let rc2 = gate.r_max * gate.r_max;
    let mut out = Vec::new();
    for s in 2..n - 2 {
        let a1 = chains[a][s - 1];
        for &j in &cell_cand(state.pos[a1]) {
            let Some(&(b, _)) = where_is.get(&j) else { continue };
            if b == a || chains[b].len() != n {
                continue;
            }
            for flip in [false, true] {
                let x = if flip { *chains[b].last().unwrap() } else { chains[b][s] };
                if x != j {
                    continue;
                }
                let d = dist(state, a1, x);
                if d < rc2 {
                    out.push(BridgeProposal { a, b, s, flip });
                }
            }
        }
    }
    out
}

/// Enumerate geometry-feasible bridge candidates for chain `a`
/// (all cut positions, both orientations), with ΔU precomputed.
/// `cell_cand` provides neighbor atom indices near a point (cell list).
pub fn enumerate_bridges<R: Rng>(
    engine: &AtomisticEngine,
    tables: &TypeTables,
    gate: &BridgeGate,
    a: usize,
    cell_cand: &dyn Fn([f64; 3]) -> Vec<usize>,
    _rng: &mut R,
) -> Vec<(BridgeProposal, f64)> {
    let state = &engine.state;
    let chains = &state.chains;
    let n = chains[a].len();
    if n < 5 {
        return Vec::new();
    }
    // bead -> (chain, contour index) lookup
    let mut where_is: HashMap<usize, (usize, usize)> = HashMap::new();
    for (c, chain) in chains.iter().enumerate() {
        for (i, &b) in chain.iter().enumerate() {
            where_is.insert(b, (c, i));
        }
    }
    let rc2 = gate.r_max * gate.r_max;
    let mut out = Vec::new();
    for s in 2..n - 2 {
        let a1 = chains[a][s - 1];
        let a2 = chains[a][s];
        let pa = chains[a][s - 2];
        for &j in &cell_cand(state.pos[a1]) {
            let Some(&(b, ij)) = where_is.get(&j) else { continue };
            if b == a || chains[b].len() != n {
                continue;
            }
            for flip in [false, true] {
                let (x, y) = if flip {
                    (*chains[b].last().unwrap(), *chains[a].last().unwrap())
                } else {
                    (chains[b][s], chains[a][s])
                };
                if x != j {
                    continue;
                }
                let b1 = chains[b][s - 1];
                let pb = chains[b][s - 2];
                // geometry gate
                let r1 = dist(state, a1, x);
                let r2 = dist(state, b1, y);
                if r1 < gate.jb_min || r1 > gate.jb_max || r2 < gate.jb_min || r2 > gate.jb_max {
                    continue;
                }
                let th1 = angle_at(state, pa, a1, x);
                let th2 = angle_at(state, pb, b1, y);
                if th1 < gate.ja_min || th1 > gate.ja_max || th2 < gate.ja_min || th2 > gate.ja_max {
                    continue;
                }
                let pr = BridgeProposal { a, b, s, flip };
                if let Some(d) = bridge_delta(engine, tables, &pr) {
                    if d.is_finite() {
                        out.push((pr, d));
                    }
                }
            }
        }
        let _ = rc2;
    }
    out
}

fn dist(state: &AtomisticState, i: usize, j: usize) -> f64 {
    let d = state.disp(i, j);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn angle_at(state: &AtomisticState, i: usize, j: usize, k: usize) -> f64 {
    let b1 = state.disp(j, i);
    let b2 = state.disp(j, k);
    let n1 = (b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2]).sqrt();
    let n2 = (b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2]).sqrt();
    if n1 < 1e-12 || n2 < 1e-12 {
        return 0.0;
    }
    ((b1[0] * b2[0] + b1[1] * b2[1] + b1[2] * b2[2]) / (n1 * n2))
        .clamp(-1.0, 1.0)
        .acos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atomistic::{AtomisticEngine, AtomisticParams};

    /// Two 6-carbon zigzag chains, close enough to interact.
    fn fixture() -> (AtomisticState, AtomisticParams) {
        let mut pos = Vec::new();
        // chain A: zigzag along x at z=0
        for i in 0..6 {
            let x = i as f64 * 1.529 * 0.9;
            let z = if i % 2 == 0 { 0.0 } else { 0.6 };
            pos.push([x, 0.0, z]);
        }
        // chain B: parallel, offset in y by 2.0 (within LJ cutoff)
        for i in 0..6 {
            let x = i as f64 * 1.529 * 0.9;
            let z = if i % 2 == 0 { 0.6 } else { 0.0 };
            pos.push([x, 2.5, z]);
        }
        let n = pos.len();
        let types = vec![1usize; n];
        let charges = vec![0.0; n];
        let mol: Vec<usize> = (0..6).map(|_| 0).chain((0..6).map(|_| 1)).collect();
        let chains = vec![(0..6).collect::<Vec<_>>(), (6..12).collect::<Vec<_>>()];
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
            pos, 40.0, types, charges, mol, bonds, angles, dihs, chains,
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

    #[test]
    fn bridge_delta_matches_total_energy_diff() {
        let (st, par) = fixture();
        let engine = AtomisticEngine::new(st, par, 1.0);
        let tables = TypeTables::from_state(&engine.state);
        for flip in [false, true] {
            for s in [2, 3, 4] {
                let pr = BridgeProposal { a: 0, b: 1, s, flip };
                let e0 = engine.total_energy();
                let d = bridge_delta(&engine, &tables, &pr).expect("delta");
                let new_state = engine.state.clone();
                let mut eng2 = AtomisticEngine::new(new_state, engine.params.clone(), 1.0);
                apply_bridge(&mut eng2.state, &tables, &pr).expect("apply");
                let e1 = eng2.total_energy();
                assert!(
                    (d - (e1 - e0)).abs() < 1e-8,
                    "flip={flip} s={s}: delta={d} vs total diff={}",
                    e1 - e0
                );
                // involution: apply again, expect original topology+energy
                let pr_rev = BridgeProposal { a: 0, b: 1, s, flip };
                apply_bridge(&mut eng2.state, &tables, &pr_rev).expect("revert");
                assert!(
                    (eng2.total_energy() - e0).abs() < 1e-8,
                    "flip={flip} s={s}: roundtrip energy"
                );
            }
        }
    }
}
