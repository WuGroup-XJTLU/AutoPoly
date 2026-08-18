//! Melt state: positions, chain connectivity, and per-chain contour order.
//!
//! Design notes
//! ------------
//! * Chains are stored as *ordered bead ids* along the contour
//!   (`chains[c][i]`), never as contiguous slices. Connectivity-altering
//!   moves (reptation now; join / segment-exchange later) rewrite chain
//!   membership, so contiguity cannot be assumed.
//! * `bonds_adj` is the ground truth for bonded terms; it is derived from
//!   `chains` at construction and maintained incrementally by moves.
//! * Phase 0 scope: linear open chains (valence <= 2). Rings and branched
//!   architectures are rejected by `validate` for now.

#[derive(Clone)]
pub struct MeltState {
    /// Bead positions, wrapped into [0, box_size)^3.
    pub pos: Vec<[f64; 3]>,
    /// Cubic periodic box edge length.
    pub box_size: f64,
    /// Per-chain ordered bead ids along the contour.
    pub chains: Vec<Vec<usize>>,
    /// bead -> chain index.
    pub chain_of: Vec<usize>,
    /// bead -> position within its chain's contour.
    pub idx_in_chain: Vec<usize>,
    /// bead -> bonded neighbors (len <= 2 for linear chains).
    pub bonds_adj: Vec<Vec<usize>>,
}

impl MeltState {
    pub fn new(
        pos: Vec<[f64; 3]>,
        chains: Vec<Vec<usize>>,
        box_size: f64,
    ) -> Result<Self, String> {
        if box_size <= 0.0 {
            return Err(format!("box_size must be positive, got {box_size}"));
        }
        let n = pos.len();
        let mut chain_of = vec![usize::MAX; n];
        let mut idx_in_chain = vec![usize::MAX; n];
        let mut bonds_adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (c, chain) in chains.iter().enumerate() {
            if chain.len() < 2 {
                return Err(format!("chain {c} has {} beads (need >= 2)", chain.len()));
            }
            for (i, &b) in chain.iter().enumerate() {
                if b >= n {
                    return Err(format!("chain {c} references bead {b} >= n_beads {n}"));
                }
                if chain_of[b] != usize::MAX {
                    return Err(format!("bead {b} appears in more than one chain"));
                }
                chain_of[b] = c;
                idx_in_chain[b] = i;
            }
            for w in chain.windows(2) {
                bonds_adj[w[0]].push(w[1]);
                bonds_adj[w[1]].push(w[0]);
            }
        }
        for b in 0..n {
            if chain_of[b] == usize::MAX {
                return Err(format!("bead {b} does not belong to any chain"));
            }
        }

        let state = MeltState {
            pos,
            box_size,
            chains,
            chain_of,
            idx_in_chain,
            bonds_adj,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn n_beads(&self) -> usize {
        self.pos.len()
    }

    pub fn n_chains(&self) -> usize {
        self.chains.len()
    }

    /// Minimum-image displacement.
    #[inline]
    pub fn min_image(&self, d: [f64; 3]) -> [f64; 3] {
        let l = self.box_size;
        [
            d[0] - l * (d[0] / l).round(),
            d[1] - l * (d[1] / l).round(),
            d[2] - l * (d[2] / l).round(),
        ]
    }

    /// Minimum-image vector pos[j] - pos[i].
    #[inline]
    pub fn disp(&self, i: usize, j: usize) -> [f64; 3] {
        let a = self.pos[i];
        let b = self.pos[j];
        self.min_image([b[0] - a[0], b[1] - a[1], b[2] - a[2]])
    }

    /// Wrap a position into [0, box_size)^3.
    #[inline]
    pub fn wrap(&self, p: [f64; 3]) -> [f64; 3] {
        let l = self.box_size;
        [
            p[0] - l * (p[0] / l).floor(),
            p[1] - l * (p[1] / l).floor(),
            p[2] - l * (p[2] / l).floor(),
        ]
    }

    #[inline]
    pub fn bonded(&self, i: usize, j: usize) -> bool {
        self.bonds_adj[i].contains(&j)
    }

    /// Angle triplets [a, b, c] along the contour that include `bead`.
    /// Derived on the fly from the chain ordering, so it is always
    /// consistent with the current topology.
    pub fn angles_touching(&self, bead: usize) -> Vec<[usize; 3]> {
        let c = self.chain_of[bead];
        let i = self.idx_in_chain[bead];
        let chain = &self.chains[c];
        let l = chain.len();
        let mut out = Vec::with_capacity(3);
        if i >= 1 && i + 1 < l {
            out.push([chain[i - 1], chain[i], chain[i + 1]]);
        }
        if i + 2 < l {
            out.push([chain[i], chain[i + 1], chain[i + 2]]);
        }
        if i >= 2 {
            out.push([chain[i - 2], chain[i - 1], chain[i]]);
        }
        out
    }

    /// All angle triplets in the system (each once).
    pub fn all_angles(&self) -> Vec<[usize; 3]> {
        let mut out = Vec::new();
        for chain in &self.chains {
            for w in chain.windows(3) {
                out.push([w[0], w[1], w[2]]);
            }
        }
        out
    }

    /// Re-derive idx_in_chain for one chain after its order changed.
    pub fn reindex_chain(&mut self, c: usize) {
        for (i, &b) in self.chains[c].iter().enumerate() {
            self.idx_in_chain[b] = i;
        }
    }

    #[allow(dead_code)] // Phase 1: connectivity moves
    pub fn add_bond(&mut self, i: usize, j: usize) {
        self.bonds_adj[i].push(j);
        self.bonds_adj[j].push(i);
    }

    #[allow(dead_code)] // Phase 1: connectivity moves
    pub fn remove_bond(&mut self, i: usize, j: usize) {
        if let Some(p) = self.bonds_adj[i].iter().position(|&x| x == j) {
            self.bonds_adj[i].swap_remove(p);
        }
        if let Some(p) = self.bonds_adj[j].iter().position(|&x| x == i) {
            self.bonds_adj[j].swap_remove(p);
        }
    }

    /// True when every chain has the same contour length.
    #[allow(dead_code)] // Phase 1: connectivity moves
    pub fn monodisperse(&self) -> bool {
        match self.chains.first() {
            None => true,
            Some(first) => self.chains.iter().all(|c| c.len() == first.len()),
        }
    }

    /// Structural invariants. Returns Err on the first violation found.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.pos.len();
        // Bead conservation: every bead appears exactly once across chains.
        let mut seen = vec![false; n];
        for (c, chain) in self.chains.iter().enumerate() {
            if chain.len() < 2 {
                return Err(format!("chain {c} shorter than 2 beads"));
            }
            for (i, &b) in chain.iter().enumerate() {
                if b >= n {
                    return Err(format!("chain {c} references bead {b} out of range"));
                }
                if seen[b] {
                    return Err(format!("bead {b} appears twice in chain storage"));
                }
                seen[b] = true;
                if self.chain_of[b] != c || self.idx_in_chain[b] != i {
                    return Err(format!("bead {b}: chain index bookkeeping stale"));
                }
            }
        }
        if let Some(b) = seen.iter().position(|&s| !s) {
            return Err(format!("bead {b} is not part of any chain"));
        }
        // Adjacency must match chain contour exactly (valence <= 2).
        for (i, adj) in self.bonds_adj.iter().enumerate() {
            if adj.len() > 2 {
                return Err(format!("bead {i} has valence {} (> 2)", adj.len()));
            }
            for &j in adj {
                if !self.bonds_adj[j].contains(&i) {
                    return Err(format!("bond {i}-{j} is not symmetric"));
                }
                let (ci, ii) = (self.chain_of[i], self.idx_in_chain[i]);
                let (cj, ij) = (self.chain_of[j], self.idx_in_chain[j]);
                if ci != cj || ii.abs_diff(ij) != 1 {
                    return Err(format!("bond {i}-{j} inconsistent with chain contour"));
                }
            }
        }
        Ok(())
    }
}
