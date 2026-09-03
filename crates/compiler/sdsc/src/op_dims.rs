// SPDX-License-Identifier: Apache-2.0
//! OpDims — the NINTH island: map one op's `CoreSplit` onto its OUTPUT's per-DIM `(per_core, nsplits,
//! is_stick)`, the input `FoldIR::dim_fold` needs. This is where op structure enters: a matmul OUTPUT is
//! `[mb, out]` — `mb` (rows) split by `row_cores` (non-stick), `out` (cols) split by `stick_cores` (the
//! 64-stick dim). Reductions/inputs/kernels are later sub-islands; this one owns the output, which is what
//! `AddrIR` addressed and the collision-critical write.
//!
//! ISLAND obligation (Kani, concrete granite shapes so the mapping's `full/nsplits` runs CONCRETELY — no
//! symbolic ÷, the `core_split` CBMC-hostility): each output dim's fold RECONSTRUCTS its extent — the tile
//! the card iterates (`FoldIR`) is exactly the op's output extent — and the stick dim tiles into whole
//! 64-sticks. Requires the OUTPUT `cols` pre-padded to a 64-multiple (a separate padding sub-island owns
//! that); every real granite matmul output is (hidden 4096, kv 1024, mlp intermediate, padded vocab 49664).

use crate::fold_ir::{DimFold, dim_fold, reconstruct};
use crate::tiled_ir::{CoreSplit, MAX_CORES, STK};

/// One arg-dim's split: its per-core extent, the split across cores, and whether it is the 64-stick dim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArgDim {
    pub per_core: u32,
    pub nsplits: u32,
    pub is_stick: bool,
}

impl ArgDim {
    pub fn fold(&self) -> DimFold {
        dim_fold(self.per_core, self.nsplits, self.is_stick)
    }
}

/// The matmul OUTPUT `[m, n]`'s two dims under `split` (`n` a multiple of 64; `row_cores | m`,
/// `stick_cores | n/64` — both hold by `CoreSplit::plan` on a 64-padded `n`). `mb` = rows (non-stick),
/// `out` = cols (the 64-stick dim). Divisions here run CONCRETELY in the proofs (concrete shape).
pub fn matmul_output_dims(m: u32, n: u32, split: CoreSplit) -> (ArgDim, ArgDim) {
    let mb = ArgDim {
        per_core: m / split.row_cores,
        nsplits: split.row_cores,
        is_stick: false,
    };
    let out = ArgDim {
        per_core: n / split.stick_cores,
        nsplits: split.stick_cores,
        is_stick: true,
    };
    (mb, out)
}

/// The matmul INPUT `[m, k]`'s dims: `mb`=rows split by `row_cores` (non-stick), `in`=k the reduction —
/// the INPUT 64-stick dim, fully RESIDENT per core (nsplits=1). `k` a multiple of 64 (hidden 4096 / hd 128).
pub fn matmul_input_dims(m: u32, k: u32, split: CoreSplit) -> (ArgDim, ArgDim) {
    let mb = ArgDim {
        per_core: m / split.row_cores,
        nsplits: split.row_cores,
        is_stick: false,
    };
    let in_ = ArgDim {
        per_core: k,
        nsplits: 1,
        is_stick: true,
    };
    (mb, in_)
}

/// The matmul KERNEL `[k, n]`'s dims: `in`=k unsplit (the kernel sticks on `out`, not `in`, so `in` is
/// non-stick here) and fully resident; `out`=cols split by `stick_cores` (the 64-stick dim). `n` 64-padded.
pub fn matmul_kernel_dims(k: u32, n: u32, split: CoreSplit) -> (ArgDim, ArgDim) {
    let in_ = ArgDim {
        per_core: k,
        nsplits: 1,
        is_stick: false,
    };
    let out = ArgDim {
        per_core: n / split.stick_cores,
        nsplits: split.stick_cores,
        is_stick: true,
    };
    (in_, out)
}

/// A shape-preserving POINTWISE op's OUTPUT `[rows, cols]` dims (Mul / Add / Silu / ScalarMul / Exp /
/// SubRowBcast / DivRowBcast / …): the SAME split shape as a matmul output — `rows` by `row_cores`
/// (non-stick), `cols` by `stick_cores` (the 64-stick dim). No reduction. `cols` 64-padded.
pub fn pointwise_dims(rows: u32, cols: u32, split: CoreSplit) -> (ArgDim, ArgDim) {
    matmul_output_dims(rows, cols, split)
}

/// A ROW-REDUCE op's INPUT `[rows, cols]` dims (RowSum / RowMax / RowSum-of-Square): `rows` split by
/// `row_cores` (non-stick), the reduced `cols` (the input 64-stick) fully RESIDENT per core (nsplits=1).
/// The output is `[rows, 1]` (split on rows only). Split from `CoreSplit::plan(rows, 1)`. `cols` 64-padded.
pub fn reduce_input_dims(rows: u32, cols: u32, split: CoreSplit) -> (ArgDim, ArgDim) {
    let row = ArgDim {
        per_core: rows / split.row_cores,
        nsplits: split.row_cores,
        is_stick: false,
    };
    let col = ArgDim {
        per_core: cols,
        nsplits: 1,
        is_stick: true,
    }; // reduced ⇒ fully resident
    (row, col)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointwise_and_reduce_dims_reconstruct() {
        for &(r, c) in &[(1u32, 4096u32), (1, 11008), (256, 128)] {
            let ps = CoreSplit::plan(r, c);
            let (pr, pc) = pointwise_dims(r, c, ps);
            assert_eq!(reconstruct(&pr.fold()), r);
            assert_eq!(reconstruct(&pc.fold()), c);
            let rs = CoreSplit::plan(r, 1);
            let (rr, rc) = reduce_input_dims(r, c, rs);
            assert_eq!(reconstruct(&rr.fold()), r);
            assert_eq!(reconstruct(&rc.fold()), c);
        }
    }

    #[test]
    fn matmul_input_kernel_dims_reconstruct() {
        for &(m, k, n) in &[
            (1u32, 4096u32, 4096u32),
            (1, 4096, 1024),
            (1, 4096, 49664),
            (256, 128, 128),
        ] {
            let split = CoreSplit::plan_capped(m, n, MAX_CORES);
            let (imb, iin) = matmul_input_dims(m, k, split);
            assert_eq!(reconstruct(&imb.fold()), m);
            assert_eq!(reconstruct(&iin.fold()), k);
            let (kin, kout) = matmul_kernel_dims(k, n, split);
            assert_eq!(reconstruct(&kin.fold()), k);
            assert_eq!(reconstruct(&kout.fold()), n);
        }
    }

    #[test]
    fn matmul_output_dims_reconstruct() {
        // (m, n) — real granite decode matmul outputs (m=1) + the RoPE-rotate shape.
        for &(m, n) in &[
            (1u32, 4096u32),
            (1, 1024),
            (1, 49664),
            (256, 128),
            (384, 384),
        ] {
            assert_eq!(n % STK, 0, "test shape n={n} must be 64-padded");
            let split = CoreSplit::plan_capped(m, n, MAX_CORES);
            let (mb, out) = matmul_output_dims(m, n, split);
            assert_eq!(reconstruct(&mb.fold()), m, "[{m},{n}] mb fold ≠ m");
            assert_eq!(reconstruct(&out.fold()), n, "[{m},{n}] out fold ≠ n");
            assert_eq!(out.fold().elem_arr_0, STK, "[{m},{n}] out inner ≠ stick");
        }
    }
}

// ── Kani: concrete shapes ⇒ the mapping's `full/nsplits` is concrete (no symbolic ÷). Each closes in ~1s. ──
#[cfg(kani)]
mod kani_opdims {
    use super::*;

    fn matmul_output_reconstructs(m: u32, n: u32) {
        let split = CoreSplit::plan_capped(m, n, MAX_CORES);
        let (mb, out) = matmul_output_dims(m, n, split);
        assert!(reconstruct(&mb.fold()) == m); // mb fold covers all rows
        assert!(reconstruct(&out.fold()) == n); // out fold covers all (padded) cols
        assert!(out.fold().elem_arr_0 == STK); // out tiles into whole 64-sticks
    }

    /// INPUT + KERNEL dims of a matmul with contraction `k` also reconstruct their extents (m, k, n).
    fn matmul_in_kernel_reconstructs(m: u32, k: u32, n: u32) {
        let split = CoreSplit::plan_capped(m, n, MAX_CORES);
        let (imb, iin) = matmul_input_dims(m, k, split);
        assert!(reconstruct(&imb.fold()) == m);
        assert!(reconstruct(&iin.fold()) == k);
        let (kin, kout) = matmul_kernel_dims(k, n, split);
        assert!(reconstruct(&kin.fold()) == k);
        assert!(reconstruct(&kout.fold()) == n);
    }
    #[kani::proof]
    fn matmul_in_kernel_hidden() {
        matmul_in_kernel_reconstructs(1, 4096, 4096); // o/q proj: k=hidden
    }
    #[kani::proof]
    fn matmul_in_kernel_lmhead() {
        matmul_in_kernel_reconstructs(1, 4096, 49664); // lm_head: k=hidden, n=padded vocab
    }
    #[kani::proof]
    fn matmul_output_hidden() {
        matmul_output_reconstructs(1, 4096); // decode o/q projection
    }
    #[kani::proof]
    fn matmul_output_kv() {
        matmul_output_reconstructs(1, 1024); // decode kv projection (8 kv-heads × 128)
    }
    #[kani::proof]
    fn matmul_output_lmhead() {
        matmul_output_reconstructs(1, 49664); // decode lm_head: vocab 49159 padded to 776·64
    }
    #[kani::proof]
    fn matmul_output_rope() {
        matmul_output_reconstructs(256, 128); // RoPE-rotate (the #50 shape), multi-row
    }

    fn pointwise_reconstructs(r: u32, c: u32) {
        let split = CoreSplit::plan_capped(r, c, MAX_CORES);
        let (pr, pc) = pointwise_dims(r, c, split);
        assert!(reconstruct(&pr.fold()) == r);
        assert!(reconstruct(&pc.fold()) == c);
    }
    #[kani::proof]
    fn pointwise_hidden() {
        pointwise_reconstructs(1, 4096); // decode residual / silu / scalarmul over hidden
    }

    fn reduce_reconstructs(r: u32, c: u32) {
        let split = CoreSplit::plan_capped(r, 1, MAX_CORES); // reduce output is [rows, 1]
        let (rr, rc) = reduce_input_dims(r, c, split);
        assert!(reconstruct(&rr.fold()) == r); // rows tiled
        assert!(reconstruct(&rc.fold()) == c); // reduced cols fully resident
    }
    #[kani::proof]
    fn reduce_hidden() {
        reduce_reconstructs(1, 4096); // rmsnorm / softmax row-reduce over hidden
    }
}
