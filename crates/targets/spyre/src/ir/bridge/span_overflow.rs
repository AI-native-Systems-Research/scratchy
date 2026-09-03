//! Bridge 2 (the tiler) — SPAN-OVERFLOW tiling, ported from torch-spyre
//! `torch_spyre/_inductor/span_overflow_hint_analysis.py` (`plan_span_overflow_tile` /
//! `_split_candidates_for_host_dim` / `_combo_cost` / `_search_min_cost_tile_plan`).
//!
//! torch-spyre's `coarse_tile` pass is really TWO separate hardware-limit checks, composed
//! together (its own docstring on `plan_span_overflow_tile` says so verbatim: "make a common
//! planner for Work Division and Working Set Reduction together" is still a TODO there too):
//!
//!   1. WORKING-SET REDUCTION — does the per-core resident tile fit the on-chip LX scratchpad?
//!      Already ported as [`scratchy_subtile::superdsc_opspec::WorkPlan::time_tile_for_lx`]
//!      (`USABLE_LX_BYTES` ≈ 1.6 MB).
//!   2. SPAN OVERFLOW — does any operand's touched PHYSICAL BYTE ADDRESS RANGE (from its lowest
//!      to its highest touched byte) exceed the hardware's addressable span — a 16-bit count of
//!      4096-byte pages, `MAX_SPAN_BYTES = 65535 * 4096` ≈ 256 MB (torch-spyre
//!      `work_division.py:71`)? This is a DIFFERENT hardware limit from LX capacity: an operand
//!      can have a tiny per-trip LX footprint while its full un-tiled physical address range
//!      still spans hundreds of MB — e.g. scratchy's OWN stick-major `dev_off` formula
//!      (`sdsc_abstract::dev_off_stk`'s rank-2 form, `(j/stk)*(a*stk) + i*stk + (j%stk)`) strides
//!      by the tensor's FULL outer extent `a` per column-stick, so a large `a` (e.g. a big KV
//!      cache `cap`, or a wide prefill row count) inflates the touched span independent of how
//!      the WORK is core-split. **This was never checked in scratchy before this file — there
//!      was no span-overflow port at all, only the LX-capacity one.**
//!
//! What's ported vs. adapted:
//!   - `MAX_SPAN_BYTES`, `_MAX_AUTO_TILE_SPLIT_COUNT` (renamed `MAX_AUTO_TILE_SPLIT_COUNT`): exact
//!     constants, unchanged.
//!   - `_split_candidates_for_host_dim`: same algorithm (enumerate divisors of the dim's full
//!     size, keep those `<= MAX_AUTO_TILE_SPLIT_COUNT` that don't cut a physical stick), adapted
//!     to scratchy's `ItDim`/stick-count vocabulary instead of torch-spyre's generic
//!     `FixedTiledLayout`.
//!   - `_combo_cost` / `_search_min_cost_tile_plan`: torch-spyre searches jointly over up to
//!     `_MAX_TILE_DIMS` (3) candidate host dims via a Cartesian-product combo search, because an
//!     arbitrary Inductor op can have several independently-overflowing output dims. Scratchy's
//!     iteration space has exactly ONE dim whose extent legitimately grows with prefill length
//!     (`mb` — the row/token count; `out`/`in`/`y` are per-op shape constants independent of
//!     `mq`), so the search here is the single-dim DEGENERATE case of that same combo search: a
//!     1-element candidate list, cost-ordered ascending (fewer tiles first, matching
//!     `_combo_cost`'s primary key `math.prod(combo)`), first legal split that clears the span
//!     wins. Kept as an explicit search loop (not simplified to "smallest split that clears")
//!     because — exactly as in the reference — this file does not assume the span is monotonic
//!     in the split count; it tries candidates in cost order and validates each one.
//!   - Physical span computation: torch-spyre walks an arbitrary sympy affine index expression
//!     per input/output `MemoryDep`, because a generic Inductor op can have arbitrary index math.
//!     Scratchy's device addressing is the single CLOSED-FORM formula in
//!     `sdsc_abstract::dev_off_stk` (a 2-D tensor sticked on its last dim is `[outer/stk, inner,
//!     stk]`; everything else is flat row-major) — so the span here is computed directly from
//!     that formula's known extremes, not by re-deriving a symbolic coordinate walk for a
//!     generality scratchy's addressing model doesn't have.

use scratchy_subtile::superdsc_opspec::ItDim;

/// = 65535 * 4096 bytes (~256 MB): torch-spyre `work_division.MAX_SPAN_BYTES`
/// (`torch_spyre/_inductor/work_division.py:71`). A hardware addressing-span limit — independent
/// of, and NOT to be confused with, [`scratchy_subtile::superdsc_opspec::USABLE_LX_BYTES`] (the on-chip
/// scratchpad CAPACITY limit that `time_tile_for_lx` already checks).
pub const MAX_SPAN_BYTES: u64 = 65_535 * 4_096;

/// torch-spyre `_MAX_AUTO_TILE_SPLIT_COUNT` (`span_overflow_hint_analysis.py:149`): automatic
/// (non-manual-hint) span tiling never requests more than this many trips, keeping the
/// one-`scf.for`-loop-per-split lowering conservative.
pub const MAX_AUTO_TILE_SPLIT_COUNT: u32 = 64;

/// The per-core physical byte span of a rank-2 stick-major tensor `[outer, inner]` (word size
/// `word_bytes`, `stick_elems` elements/stick — [`scratchy_subtile::superdsc_opspec::DataFormat::
/// ELEMS_PER_STICK`]) under `sdsc_abstract::dev_off_stk`'s rank-2 formula:
/// `off(i,j) = (j/stk)*(outer*stk) + i*stk + (j%stk)`.
///
/// Mirrors torch-spyre's `_coordinate_span_elems` (evaluate the affine coordinate at its extreme
/// indices and take `max - min + 1`), specialized to this one closed-form address function instead
/// of a symbolic walk: the formula is monotonically increasing in both `i` and `j`, so the extremes
/// are exactly `(i=0,j=0)` and `(i=outer-1, j=inner-1)` — no `Mod`-wraparound case to search,
/// because `j % stk` is bounded by construction (`stk` is the SAME modulus dev_off_stk itself
/// divides `j` by, so `j % stk` and `j / stk` cannot both simultaneously grow past their extremes
/// independent of `j`'s own extreme, unlike the reference's general "another symbol's stride
/// happens to share this coordinate" case).
///
/// `outer` is the tensor's FULL declared outer extent (e.g. `cap` for a KV cache, `mq_pad` for a
/// prefill activation) — NOT the per-core split extent: this is exactly the hazard this module
/// exists to catch, since `dev_off_stk`'s column-stick stride is `outer*stk` regardless of how the
/// row dimension is core-split.
pub fn physical_span_bytes(outer: u32, inner: u32, stick_elems: u32, word_bytes: u32) -> u64 {
    if outer == 0 || inner == 0 {
        return 0;
    }
    let off = |i: u64, j: u64| -> u64 {
        let stk = stick_elems as u64;
        (j / stk) * (outer as u64 * stk) + i * stk + (j % stk)
    };
    let max_off = off((outer - 1) as u64, (inner - 1) as u64);
    let min_off = off(0, 0);
    (max_off - min_off + 1) * word_bytes as u64
}

/// torch-spyre `_split_candidates_for_host_dim` (`span_overflow_hint_analysis.py:1100`), adapted
/// to scratchy's `ItDim` vocabulary: every divisor of `full_size` that is `<=
/// MAX_AUTO_TILE_SPLIT_COUNT` and, if this dim IS the stick-carrying dim (`is_stick`), leaves a
/// whole number of `stick_elems`-sized sticks per tile (torch-spyre
/// `_post_tile_stick_alignment_error`: "coarse-tile boundaries would cut through physical
/// sticks"). Non-stick dims skip that check, matching the reference (`split == 1 or (...)`, where
/// the stick check is a no-op for non-stick dims). Ascending, like the reference's `sorted({...})`.
pub fn split_candidates_for_dim(full_size: u32, is_stick: bool, stick_elems: u32) -> Vec<u32> {
    if full_size == 0 {
        return vec![1];
    }
    let mut candidates: Vec<u32> = Vec::new();
    let mut i = 1u32;
    while i.saturating_mul(i) <= full_size {
        if full_size.is_multiple_of(i) {
            candidates.push(i);
            let pair = full_size / i;
            if pair != i {
                candidates.push(pair);
            }
        }
        i += 1;
    }
    candidates.sort_unstable();
    candidates.retain(|&split| {
        split <= MAX_AUTO_TILE_SPLIT_COUNT
            && (!is_stick || (full_size / split).is_multiple_of(stick_elems))
    });
    candidates
}

/// torch-spyre `_combo_cost` (`span_overflow_hint_analysis.py:1181`), degenerate single-dim case:
/// rank candidates by fewest total tiles first (`math.prod(combo)`, here just `split` itself since
/// there is one dim), matching the reference's primary sort key. The reference's secondary keys
/// (fewer tiled dims, smaller max split) are constant/redundant in the single-dim case and are
/// therefore omitted rather than reproduced as dead tie-breakers.
fn combo_cost(split: u32) -> u32 {
    split
}

/// torch-spyre `_search_min_cost_tile_plan` (`span_overflow_hint_analysis.py:1269`), degenerate
/// single-dim case (torch-spyre's own `host_dims` candidate list has exactly one entry here — the
/// only scratchy `ItDim` whose extent scales with prefill length). Tries every legal split of
/// `dim`'s full size, cheapest first (fewest tiles), and returns the first whose resulting
/// per-core span — computed by the caller-supplied `span_at_split`, since which operand's layout
/// matters is a call-site concern exactly as it is in the reference (`op`'s own input/output
/// deps) — clears [`MAX_SPAN_BYTES`]. `Err` (matching the reference's final `raise Unsupported`)
/// if no legal split clears it, naming the dim and the best span found.
pub fn cheapest_split_clearing_span(
    dim: &ItDim,
    span_at_split: impl Fn(u32) -> u64,
) -> Result<u32, String> {
    let stick_elems = dim.df.elems_per_stick();
    let mut candidates = split_candidates_for_dim(dim.size, dim.is_stick, stick_elems);
    candidates.sort_unstable_by_key(|&split| combo_cost(split));

    let mut best_span = u64::MAX;
    for split in candidates {
        let span = span_at_split(split);
        best_span = best_span.min(span);
        if span <= MAX_SPAN_BYTES {
            return Ok(split);
        }
    }
    Err(format!(
        "span-overflow: no legal split of dim '{}' (size {}) brings its physical span under \
         {MAX_SPAN_BYTES} B ({:.2} MB) — best achieved was {best_span} B ({:.2} MB) at the \
         largest legal split (<= {MAX_AUTO_TILE_SPLIT_COUNT})",
        dim.name,
        dim.size,
        MAX_SPAN_BYTES as f64 / (1024.0 * 1024.0),
        best_span as f64 / (1024.0 * 1024.0),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_tensor_never_overflows() {
        // A realistic decode/prefill-sized tensor (cap=4096, hd=64, fp16) is nowhere near 256 MB.
        let span = physical_span_bytes(4096, 64, 64, 2);
        assert!(
            span < MAX_SPAN_BYTES,
            "span={span} should be well under the limit"
        );
    }

    #[test]
    fn split_candidates_respect_stick_alignment() {
        // size=256, stick=64: legal stick-dim splits must leave a multiple of 64 per tile ⇒ only
        // splits where 256/split is itself a multiple of 64 ⇒ split in {1,2,4} (256/4=64).
        let candidates = split_candidates_for_dim(256, true, 64);
        assert_eq!(candidates, vec![1, 2, 4]);
    }

    #[test]
    fn split_candidates_non_stick_dim_allows_any_divisor() {
        let candidates = split_candidates_for_dim(12, false, 64);
        assert_eq!(candidates, vec![1, 2, 3, 4, 6, 12]);
    }

    #[test]
    fn cheapest_split_picks_smallest_that_clears() {
        let dim = ItDim {
            name: "mb",
            size: 8,
            is_reduction: false,
            is_stick: false,
            df: scratchy_subtile::superdsc_opspec::Df::Fp16,
        };
        // Span shrinks by exactly the split factor; only split>=4 clears an 8-unit budget with a
        // 32-unit baseline span (32/4=8 <= MAX budget of, say, we use a real tiny stand-in here).
        let span_at = |split: u32| -> u64 { 32 / split as u64 };
        let got = cheapest_split_clearing_span(&dim, |split| {
            // Force a real MAX_SPAN_BYTES-scale check by inflating both sides equally --
            // the ratio is what matters for "smallest that clears".
            span_at(split) * (MAX_SPAN_BYTES / 8)
        })
        .expect("split=8 clears (32/8=4 units <= 4-unit-equivalent budget)");
        assert_eq!(
            got, 4,
            "smallest split with span <= budget should be chosen (4: 32/4*budget/8=budget)"
        );
    }
}
