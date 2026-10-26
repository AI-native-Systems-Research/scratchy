//! ⭐ THE `dl16tofp32` LANE STAGGER — a TARGET-ABI FACT, stated as CONST TABLES.
//!
//! MEASURED on the card (2026-10-26, granite-2b fp8, probe-8): when BOTH operands are addressed
//! STANDARD, the device's `dl16tofp32` writes value-exact but LANE-PERMUTED output — within each
//! 64-element block of the f32 word stream, the sixteen 4-lane groups permute by a 1-bit RIGHT
//! ROTATION of the 4-bit group index: logical lane `4g + r` lands at buffer position
//! `4·ror(g) + r`, `ror(g) = (g>>1) | ((g&1)<<3)`. Lane 1318 lands at buffer 1330; lanes 0-3 and
//! 60-63 are fixed; the map is block-invariant and the global multiset is equal. torch-spyre
//! knows the same fact host-side (`ElementArrangement.DL16_TO_FP32`) and its own KTIR emitter
//! REFUSES to state it as numbers — this module is the statement it never wrote down.
//!
//! The stagger is NOT expressible as strides (it is a bit rotation of a sub-index), so it cannot
//! ride a layout; it is a fact about the OPCODE, and a fact about an opcode is a TARGET-ABI CONST
//! TABLE ROW (the in-place-kernels/write-bindings law): [`OP_OUTPUT_ARRANGEMENT`] maps an op-func
//! to the arrangement its output carries when both operands are standard, and the bake-side
//! propagation ([`crate::emit::whole_function::staggered_sources`]) takes that table as INPUT —
//! never logic of its own. Everything downstream is decided at expansion: the staged tables are
//! EMITTED already in their arrangement ([`crate::wiring`]'s builders are pure in the
//! arrangement), the worker only indexes.
//!
//! ⛔ PADS/REDUCTIONS END IT. A reduction over a staggered operand is order-free (the multiset is
//! unchanged), so the arrangement never escapes a program whose consumers reduce — and a
//! staggered value reaching a host read-back is a build-time refusal at the propagation, never a
//! permuted answer on the card.

/// The element arrangement a value's buffer carries: the order of elements within each
/// 64-element block of the flat word stream.
///
/// A TYPE, not a string: it rides the baked facts ([`crate::wiring::LoadConst`],
/// [`crate::wiring::SampleSeat`]) as a field, so a consumer that stages or binds a buffer
/// states the arrangement it stages — and the propagation's refusals (a staggered value meeting
/// a standard one where order matters) fire at EXPANSION, which is a `cargo build` error.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ElementArrangement {
    /// Logical lane `i` at buffer position `i` — every buffer the card's ops address unless a
    /// table row says otherwise.
    Standard,
    /// Logical lane `4g + r` at buffer position `4·ror(g) + r` within each 64-element block —
    /// the measured `dl16tofp32` output stagger ([`DL16TOFP32_PERMUTATION`]).
    Dl16ToFp32,
}

/// The 64-entry permutation: `PERMUTATION[i]` is the BUFFER POSITION logical lane `i` of a
/// 64-element block occupies in a [`ElementArrangement::Dl16ToFp32`] buffer. A const table, not
/// a derivation — the stager that writes lane `i` reads this row and never computes.
pub const DL16TOFP32_PERMUTATION: [u8; 64] = dl16tofp32_permutation();

/// The const constructor behind [`DL16TOFP32_PERMUTATION`] — evaluated at compile time, so the
/// table on disk IS the formula (the right rotation of the 4-bit group index), stated once.
const fn dl16tofp32_permutation() -> [u8; 64] {
    let mut table = [0u8; 64];
    let mut g = 0;
    while g < 16 {
        // ror(g) = (g >> 1) | ((g & 1) << 3) — a 1-bit right rotation of the 4-bit group index.
        let rotated = (g >> 1) | ((g & 1) << 3);
        let mut r = 0;
        while r < 4 {
            table[g * 4 + r] = (rotated * 4 + r) as u8;
            r += 1;
        }
        g += 1;
    }
    table
}

/// ⭐ THE TARGET-ABI CONST TABLE — opcode → the arrangement the op's OUTPUT carries when both
/// operands are addressed standard. The shared propagation takes this table as INPUT; it never
/// decides an arrangement by logic of its own (the in-place-kernels/write-bindings law: target
/// facts are const tables, never scattered match arms).
///
/// The `fp32todl16` row states the INVERSE stagger — the narrow-back permutes the same way
/// round, mapping standard-order f32 words into staggered dl16 lanes. ⛔ NOT YET MEASURED as
/// numbers: no producer this tree compiles emits the narrow-back, so the propagation refuses a
/// program that would need its staging rather than guessing a table for it. The row exists so
/// the ABI fact is stated where the ABI facts live; the day a card probe measures it, the
/// refusal is replaced by the measured table and nothing else moves.
pub const OP_OUTPUT_ARRANGEMENT: &[(&'static str, ElementArrangement)] = &[
    ("dl16tofp32", ElementArrangement::Dl16ToFp32),
    ("fp32todl16", ElementArrangement::Dl16ToFp32),
];

/// The arrangement an op-func's output carries, or `None` when the op-func is not in the table
/// (an op not in the table addresses its operands and output standard — the default every
/// buffer starts from). ONE lookup, straight off the const table.
pub fn output_arrangement_of(op_func: &str) -> Option<ElementArrangement> {
    OP_OUTPUT_ARRANGEMENT
        .iter()
        .find(|(op, _)| *op == op_func)
        .map(|(_, arrangement)| *arrangement)
}

/// Stage a logical-order f32 word stream into its [`ElementArrangement::Dl16ToFp32`] buffer
/// order: lane `i` of each 64-word block is written at `DL16TOFP32_PERMUTATION[i]`. The ONE
/// staging derivation — the baked load-const tables ([`crate::wiring::LoadConst::build`]) and
/// the worker's per-step noise bind both call here, so the two legs cannot stage differently.
///
/// Block-wise over the FLAT word sequence, the same blocking the device permutes by; a stream
/// whose length is not a 64 multiple keeps its tail verbatim (the tail is past every whole
/// block, and every buffer this tree stages staggered is a whole-block `[·, PAD_V]` row).
pub fn dl16tofp32_stagger(words: &[f32]) -> Vec<f32> {
    let mut out = words.to_vec();
    for block in out.chunks_exact_mut(64) {
        let logical: [f32; 64] = block.try_into().expect("chunks_exact_mut is 64 wide");
        for (i, word) in logical.iter().enumerate() {
            block[DL16TOFP32_PERMUTATION[i] as usize] = *word;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⛔ THE TABLE IS PINNED AGAINST THE MEASURED MAP (probe-8, granite-2b fp8, 2026-10-26):
    /// the f16 row's lane 1318 (block 20, local lane 38) was measured at buffer position 1330
    /// (local 50) of the f32 widen; lanes 0-3 and 60-63 were measured FIXED; the map was
    /// block-invariant with the global multiset equal. A change to the const constructor that
    /// moves any pinned lane is a regression against the card, caught here at build time.
    #[test]
    fn the_permutation_is_pinned_to_the_measured_map() {
        // Lane 1318 → buffer 1330: within the block, local lane 38 → position 50.
        assert_eq!(
            DL16TOFP32_PERMUTATION[1318 % 64] as usize,
            1330 % 64,
            "the measured stagger put lane 1318 at buffer 1330 — local 38 at 50"
        );
        // Lanes 0-3 (group 0) and 60-63 (group 15) are the rotation's fixed points.
        for i in 0..4 {
            assert_eq!(DL16TOFP32_PERMUTATION[i], i as u8, "lane {i} is fixed");
        }
        for i in 60..64 {
            assert_eq!(DL16TOFP32_PERMUTATION[i], i as u8, "lane {i} is fixed");
        }
        // A permutation, not a projection: every buffer position is written exactly once.
        let mut seen = [false; 64];
        for &pos in DL16TOFP32_PERMUTATION.iter() {
            assert!(!seen[pos as usize], "position {pos} written twice");
            seen[pos as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "every position is written");
    }

    /// The staging is the table applied block-wise: a word at logical lane `i` of a block lands
    /// at the table's position, and the multiset is unchanged.
    #[test]
    fn the_staging_writes_lane_i_at_the_baked_index() {
        let logical: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let staged = dl16tofp32_stagger(&logical);
        for block in 0..2 {
            for i in 0..64 {
                assert_eq!(
                    staged[block * 64 + DL16TOFP32_PERMUTATION[i] as usize],
                    (block * 64 + i) as f32,
                    "logical lane {i} of block {block} must sit at the baked index"
                );
            }
        }
        let mut a = staged.clone();
        let mut b = logical.clone();
        a.sort_by(|x, y| x.total_cmp(y));
        b.sort_by(|x, y| x.total_cmp(y));
        assert_eq!(a, b, "the staging permutes, never projects");
    }
}
