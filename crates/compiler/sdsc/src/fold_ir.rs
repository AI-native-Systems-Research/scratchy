// SPDX-License-Identifier: Apache-2.0
//! FoldIR — the EIGHTH island, one pass below `AddrIR`. It turns each op-dim's `(per_core_extent, nsplits,
//! is_stick)` into the `coordInfo` FOLD FACTORS the wire's `SdscOp` carries (the card's per-dim tile-fold —
//! ABI = the monolith's `gen_coord_info_value`): `core_fold` = the split, then `elem_arr_*` = the per-core
//! tile (a non-stick dim is one `[per_core]` run; a stick dim is `[per_core/64]` sticks of `64`).
//!
//! ISLAND obligation (Kani, closes in <1s): the folds RECONSTRUCT the dim — `core_fold · Π elem_arr ==
//! full_extent` — so the tile the card iterates is exactly the tile `AddrIR` addressed; and a stick dim's
//! inner run is exactly one 64-stick. CRUCIAL for the 5-10s bar: `dim_fold` takes `per_core` DIRECTLY (the
//! caller divides `full/nsplits` ONCE, and that division's evenness is already core_split's proven property),
//! so the proof is pure MULTIPLICATION (`per_core · nsplits`) with only a CONCRETE `/64` — no symbolic ÷/%
//! (the `core_split` CBMC-hostility), hence instant.

use crate::tiled_ir::STK;

/// The `coordInfo` fold of one op-dim: `core_fold` (= the split across cores) × the per-core tile
/// `elem_arr_1 · elem_arr_0`. Mirrors `gen_coord_info_value`'s `dim_prop_attr`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DimFold {
    pub core_fold: u32,
    pub elem_arr_1: u32,
    pub elem_arr_0: u32,
}

/// Build a dim's fold from its PER-CORE extent + split. `per_core` is passed in (not divided here) so the
/// caller owns the single `full/nsplits` (proven even by core_split) and this stays division-free except a
/// CONCRETE `/STK`. A STICK dim tiles into `per_core/64` sticks of `64`; a non-stick dim is one run.
pub fn dim_fold(per_core: u32, nsplits: u32, is_stick: bool) -> DimFold {
    if is_stick {
        DimFold {
            core_fold: nsplits,
            elem_arr_1: per_core / STK,
            elem_arr_0: STK,
        }
    } else {
        DimFold {
            core_fold: nsplits,
            elem_arr_1: 1,
            elem_arr_0: per_core,
        }
    }
}

/// The full extent the fold reconstructs: `core_fold · elem_arr_1 · elem_arr_0`.
pub fn reconstruct(f: &DimFold) -> u32 {
    f.core_fold * f.elem_arr_1 * f.elem_arr_0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_reconstruct_examples() {
        // non-stick: rows=256 split 16 → per_core 16, one run of 16, ×16 cores = 256.
        let f = dim_fold(16, 16, false);
        assert_eq!(reconstruct(&f), 256);
        // stick: out=128 (2 sticks) split 2 → per_core 64 = 1 stick, ×2 cores = 128.
        let g = dim_fold(64, 2, true);
        assert_eq!(reconstruct(&g), 128);
        assert_eq!(g.elem_arr_0, STK);
        // lm_head-ish: per_core 6208 (97 sticks) split 8 → 8·97·64 = 49664 (padded vocab).
        let h = dim_fold(6208, 8, true);
        assert_eq!(reconstruct(&h), 49664);
    }
}

// ── Kani proofs (co-located with the island; discovered by `#[kani::proof]`). DIVISION-FREE construction
//    (parametrise by the per-core building blocks, build `full` by MULTIPLICATION) so CBMC never sees a
//    symbolic ÷/% — closes in <1s, honoring the 5-10s bar. ──
#[cfg(kani)]
mod kani_folds {
    use super::*;
    use crate::tiled_ir::MAX_CORES;

    /// A NON-STICK dim's fold reconstructs its full extent (`nsplits · per_core`) exactly.
    #[kani::proof]
    fn fold_reconstructs_nonstick() {
        let per_core: u32 = kani::any::<u16>() as u32; // ≤ hidden 4096 fits; bounded for a tiny domain
        let nsplits: u32 = kani::any();
        kani::assume(per_core >= 1 && per_core <= 4096);
        kani::assume(nsplits >= 1 && nsplits <= MAX_CORES);
        let f = dim_fold(per_core, nsplits, false);
        assert!(reconstruct(&f) == per_core * nsplits); // == the full extent
        assert!(f.core_fold == nsplits);
    }

    /// A STICK dim's fold reconstructs its full extent AND its inner run is exactly one 64-stick. Built from
    /// `sticks` so `per_core = sticks·64` by construction (`per_core/64 == sticks`, a concrete-divisor div).
    #[kani::proof]
    fn fold_reconstructs_stick() {
        let sticks: u32 = kani::any::<u8>() as u32; // per-core stick count, tiny domain
        let nsplits: u32 = kani::any();
        kani::assume(sticks >= 1 && sticks <= 128);
        kani::assume(nsplits >= 1 && nsplits <= MAX_CORES);
        let per_core = sticks * STK;
        let f = dim_fold(per_core, nsplits, true);
        assert!(f.elem_arr_0 == STK); // inner run is one whole stick
        assert!(f.elem_arr_1 == sticks); // per_core/64 == sticks (concrete /64)
        assert!(reconstruct(&f) == per_core * nsplits); // == the full extent
    }
}
