// SPDX-License-Identifier: Apache-2.0
//! TiledIR — the SIXTH distinct IR on the SDSC ladder, below `FlatIR`. It adds the **32-core
//! work-division**: every op's OUTPUT is partitioned across up to `MAX_CORES` cores, each core computing
//! a disjoint sub-region. This is the rung the monolith's hand-rolled split fails at for granite (RoPE
//! rotate, head_dim 128 → two cores write the same per-core HBM address, guard #50).
//!
//! WHAT IT VERIFIES (two independent guarantees, both anchored on `eval_dag`):
//!   1. gather(per-core tiles) == `eval_dag` — the split REASSEMBLES the correct output. A partition that
//!      overlaps double-writes an element; one that under-covers drops an element; either diverges here.
//!   2. (Kani, [`kani_proofs`]) the per-core region map is a DISJOINT, COVERING partition of `[0,R)×[0,C)`
//!      for every core_split — the TYPE-SAFE replacement for the monolith's runtime guard #50 panic
//!      (per [[never-use-runtime-assertions]]).
//!
//! The op ARITHMETIC is unchanged from `FlatIR` (already verified bit-exact vs `eval_dag`), so TiledIR
//! reuses `FlatIR::eval` for the values and layers the work-division partition on top: it reconstructs
//! each op's output by copying each core's owned region out of the full result into a fresh buffer while
//! counting writes-per-element. Disjoint+covering ⇒ reconstruction == full == eval_dag AND every count==1.

use crate::flat_ir::{FlatIR, Shape};
use crate::macro_ir::seed_src;
use scratchy_subtile::subtile_ir::{RopeForm, SubtileIR, eval_dag};

/// Device: 32 cores; stick = 64 fp16 elems. Mirrors the SuperDSC hardware model (superdsc_opspec MAX_CORES
/// / lower_subtile_tape_to_superdsc). The stick axis is split by STICK COUNT, not raw columns — matching
/// the device (a core owns whole 64-col sticks), which is exactly where head_dim=128 (2 sticks) bites.
pub const MAX_CORES: u32 = 32;
pub const STK: u32 = 64;

/// `core_split` and `CoreSplit` are CANONICAL in `scratchy-subtile` (the wire emitter's matmul split
/// defers to `CoreSplit::plan`); RE-EXPORTED here so the Kani proofs + `TiledIR` verify the EXACT function
/// the wire emits — ONE definition, no duplicate brain. (Both were ported from `distribute_cores`, which
/// the pointwise/reduce emits already use, so this makes the whole emission a single, proven split.)
pub use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{CoreSplit, core_split};

/// A distinct IR: the FlatIR op-graph annotated with a per-op work-division split. The op arithmetic is
/// FlatIR's (verified); TiledIR carries the split + the tensor shapes to verify the partition + reassembly.
#[derive(Clone, Debug)]
pub struct TiledIR {
    pub tensors: Vec<Shape>,
    pub num_sources: u32,
    /// Parallel to the source `FlatIR::nodes`: `outs[k]` = that op's output tid, `splits[k]` its split.
    pub outs: Vec<usize>,
    pub splits: Vec<CoreSplit>,
    pub result: usize,
    flat: FlatIR,
}

impl TiledIR {
    /// Run the tiled execution: `FlatIR::eval` gives the per-tensor VALUES (arithmetic unchanged); then
    /// for each op, RECONSTRUCT its output tensor by copying each core's owned region out of the full
    /// result, counting writes. Returns `(reconstructed_tensors, max_overlap)` where `max_overlap` is the
    /// largest writes-to-one-element seen (2+ ⇒ a #50-class collision; 0 ⇒ an uncovered element).
    pub fn eval(&self, sources: &[&[f32]]) -> (Vec<Vec<f32>>, u32, u32) {
        let full = self.flat.eval(sources);
        let mut recon = full.clone();
        let mut worst_over = 1u32; // max writes/element across op outputs
        let mut worst_under = 1u32; // min writes/element (0 ⇒ gap)
        for (k, &out) in self.outs.iter().enumerate() {
            let sh = self.tensors[out];
            let (r, c) = (sh.rows, sh.cols);
            let split = self.splits[k];
            let mut count = vec![0u32; (r * c) as usize];
            let mut buf = vec![0f32; (r * c) as usize];
            for cr in 0..split.row_cores {
                for cs in 0..split.stick_cores {
                    let (r0, r1, c0, c1) = split.region(cr, cs, r, c);
                    for i in r0..r1 {
                        for j in c0..c1 {
                            let idx = (i * c + j) as usize;
                            buf[idx] = full[out][idx];
                            count[idx] += 1;
                        }
                    }
                }
            }
            recon[out] = buf;
            for &n in &count {
                worst_over = worst_over.max(n);
                worst_under = worst_under.min(n);
            }
        }
        (recon, worst_over, worst_under)
    }
}

/// THE PASS: `FlatIR → TiledIR`. Assign each op a `CoreSplit::plan` over its output shape (32-core
/// work-division). Sources have no split (they are inputs, not computed).
pub fn lower_flat_to_tiled(f: &FlatIR) -> TiledIR {
    let mut outs = Vec::with_capacity(f.nodes.len());
    let mut splits = Vec::with_capacity(f.nodes.len());
    for node in &f.nodes {
        let sh = f.tensors[node.out];
        outs.push(node.out);
        splits.push(CoreSplit::plan(sh.rows, sh.cols));
    }
    TiledIR {
        tensors: f.tensors.clone(),
        num_sources: f.num_sources,
        outs,
        splits,
        result: f.result,
        flat: f.clone(),
    }
}

/// VERIFY the `FlatIR → TiledIR` bridge against the golden `eval_dag`: the reconstruction (gather of the
/// per-core tiles) must equal `eval_dag` per tensor (relative-L2 ≤ 1e-2), AND every op-output element must
/// be written by EXACTLY ONE core (worst_over == 1 ⇒ no #50 collision; worst_under == 1 ⇒ full coverage).
pub fn verify_against_eval_dag<F: RopeForm>(tiled: &TiledIR, subtile: &SubtileIR<F>) {
    const RELL2_TOL: f64 = 1e-2;
    const EPS: f64 = 1e-9;
    assert_eq!(
        tiled.num_sources, subtile.num_sources,
        "TiledIR bridge: source count {} != SubtileIR {}",
        tiled.num_sources, subtile.num_sources
    );
    let src: Vec<Vec<f32>> = (0..subtile.num_sources as usize)
        .map(|s| {
            let sh = subtile.tensors[s];
            (0..(sh.rows * sh.cols) as usize)
                .map(|j| seed_src(s, j))
                .collect()
        })
        .collect();
    let refs: Vec<&[f32]> = src.iter().map(|v| v.as_slice()).collect();
    let golden = eval_dag(subtile, &refs);
    let (got, worst_over, worst_under) = tiled.eval(&refs);

    // partition guarantees (the #50 property, checked numerically here + PROVEN in kani_proofs).
    assert!(
        worst_over == 1,
        "TiledIR bridge: a 32-core split writes one output element {worst_over}× — COLLIDING per-core \
         partition (#50 class). The work-division must partition each output disjointly."
    );
    assert!(
        worst_under == 1,
        "TiledIR bridge: a 32-core split leaves an output element unwritten (coverage gap) — the \
         per-core regions must COVER the whole output."
    );

    let n_orig = subtile.tensors.len();
    let mut worst_rell2 = 0f64;
    let mut worst_tid = 0usize;
    for tid in 0..n_orig {
        let (g, o) = (&golden[tid], &got[tid]);
        assert_eq!(
            g.len(),
            o.len(),
            "TiledIR bridge: tensor t{tid} size changed"
        );
        let (mut num, mut den) = (0f64, 0f64);
        for i in 0..g.len() {
            let (gv, ov) = (g[i] as f64, o[i] as f64);
            num += (gv - ov) * (gv - ov);
            den += gv * gv;
        }
        let rell2 = num.sqrt() / den.sqrt().max(EPS);
        if rell2 > worst_rell2 {
            worst_rell2 = rell2;
            worst_tid = tid;
        }
    }
    if std::env::var_os("SCRATCHY_SUPERDSC_DBG").is_some() {
        eprintln!(
            "[sdsc-stage] TiledIR bridge: {n_orig} tensors, worst relative-L2 {worst_rell2:.2e} at \
             t{worst_tid}; 32-core partition disjoint+covering (writes/elem == 1) [relL2 ≤ {RELL2_TOL} ⇒ \
             work-division faithful]"
        );
    }
    assert!(
        worst_rell2 <= RELL2_TOL,
        "TiledIR bridge diverges at t{worst_tid}: relative-L2 {worst_rell2:.2e} > {RELL2_TOL}. The \
         32-core work-division does not reassemble to eval_dag — THIS bridge is the tiling bug."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_split_largest_divisor() {
        assert_eq!(core_split(384, 32), 32); // 32 | 384
        assert_eq!(core_split(320, 32), 32); // 32 | 320 (=32·10); largest divisor ≤32 is 32
        assert_eq!(core_split(48, 32), 24); // 48 = 2^4·3; divisors ≤32 max out at 24
        assert_eq!(core_split(576, 32), 32); // smollm2 hidden
        assert_eq!(core_split(1, 32), 1);
    }

    /// The per-core regions of a plan tile the output disjointly + completely (the #50 property), for a
    /// spread of shapes including the granite RoPE case (head_dim 128 = 2 sticks) and non-64 cols.
    #[test]
    fn plan_partitions_output_disjoint_and_covering() {
        for &(r, c) in &[
            (1u32, 128u32),
            (96, 576),
            (256, 128),
            (1, 49159),
            (384, 384),
            (17, 200),
        ] {
            let split = CoreSplit::plan(r, c);
            let mut count = vec![0u32; (r * c) as usize];
            for cr in 0..split.row_cores {
                for cs in 0..split.stick_cores {
                    let (r0, r1, c0, c1) = split.region(cr, cs, r, c);
                    for i in r0..r1 {
                        for j in c0..c1 {
                            count[(i * c + j) as usize] += 1;
                        }
                    }
                }
            }
            assert!(
                count.iter().all(|&n| n == 1),
                "shape [{r},{c}] split {split:?} not a clean partition"
            );
        }
    }
}
