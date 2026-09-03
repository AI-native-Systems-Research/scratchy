// SPDX-License-Identifier: Apache-2.0
//! Kani proof harnesses — EXHAUSTIVE (all-inputs-in-bounds) verification of the SDSC ladder's
//! integer/addressing/layout logic. This is where every bug this effort has hit lives (sticking, the
//! K-cache Kᵀ offset, region slices, GQA, tiling) — pure index math, ideal for a bounded model checker.
//! Run with `cargo kani -p scratchy-sdsc`. The whole module is `#[cfg(kani)]` so it never affects normal
//! builds; the numeric (f32) equivalence of the bridges is covered separately by the multi-seed eval
//! batteries (Kani/CBMC cannot tractably verify exp/sqrt/reassociation).

#![cfg(kani)]

use crate::addr_ir::core_out_offset;
use crate::tiled_ir::{CoreSplit, MAX_CORES, core_split};
use scratchy_subtile::sdsc_abstract::{
    AdmittedRequests, ChunkRows, KvSlot, PagedKvPool, PoolRows, SlotCount, SlotRun, StickLayout,
    appended_run, contiguous_run, decode_prefix_col_valid, dev_off, gqa_dedup_kv_kernel_base,
    gqa_kv_head, kcache_kt_write_offset, prefill_causal_col_valid, rope_p_entry,
    rope_prefill_block_offset, rope_prefill_cos_offset, runs_contain, selector_head_src_col,
    selector_lastrow_col, shared_write_slot, vcache_write_offset,
};
use scratchy_subtile::superdsc_opspec::{
    DataFormat, DeviceTileLayout, Fp8, Fp8W8A8Dequant, Fp16, Fp32, OpFunc, PackedInt8Scale, SenInt8,
};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{
    DeviceWidth, GroupKind, SEGMENT_OFFSETS, SEGMENT_SIZE, Trip, TripRequest,
    bump_sticks_to_splittable, group_ranges, group_ranges_cover_ok, group_spans_two_requests,
    hbm_seg_off, pointwise_chunk_out_offset, scalarmul_scale_tid, sen169_bits,
};
use std::num::NonZeroU32;

// ── ScalarMul scale-const TID scheme: distinct scale indices map to DISTINCT reserved TIDs that never
//    collide with the other reserved consts (u32::MAX-1..-6) ⇒ the scale consts can't alias each other or
//    ATTN_SCALE/RMS_SEED/etc. (their placement is then the proven SegLayout non-overlap). Closes <1s. ──
// ── MEDIUM-GRAIN FUSION grouping (the leak+speed fix): `group_ranges` partitions the
//    body's trip sequence into contiguous concrete-bundle groups (≤ g Pure trips; each
//    host_kv_write + the `skip` copies it replaces, and each slot_write, a SINGLETON so
//    the shim's per-entry interception `oi += skip` / slot offset lands right). The bundle
//    must EXACTLY tile [0,n): a GAP drops compute (a trip never runs → wrong output); an
//    OVERLAP double-runs it. Fail-first: an off-by-one in the group walk breaks the
//    "next.start == prev.end, first.start==0, last.end==n" chain. Also: any group holding a
//    non-Pure trip MUST be a singleton; a Pure group is ≤ g. Closes <1s (N=6, g≤4). ──
#[kani::proof]
#[kani::unwind(5)]
fn group_ranges_is_exact_covering_partition() {
    const N: usize = 4;
    let g: usize = kani::any();
    // g ∈ [1, 8]: covers per-op (g=1), medium-grain (g=2), AND large-g whole-body fusion (g ≥ N,
    // where a full run of N Pure/Slot trips collapses to ONE group — the shipped G=2048 config, 3
    // groups). Proves the covering partition holds at EVERY granularity, incl. whole-body.
    kani::assume(g >= 1 && g <= 8);
    let mut kinds = [Trip::new(GroupKind::Pure, TripRequest(0)); N];
    for slot in kinds.iter_mut() {
        let t: u8 = kani::any();
        kani::assume(t <= 3);
        // SYMBOLIC REQUEST too: the partition must tile [0,N) whatever the request pattern, incl.
        // the runs the request test now breaks that the kind test alone would have fused.
        let r: u32 = kani::any();
        kani::assume(r <= 2);
        let kind = match t {
            0 => GroupKind::Pure,
            1 => GroupKind::Slot { req: r },
            2 => GroupKind::SlotSolo, // distinct per-slot prefill cachewr — its own singleton
            _ => {
                let s: usize = kani::any();
                kani::assume(s <= 1);
                GroupKind::HostKv { skip: s }
            }
        };
        *slot = Trip::new(kind, TripRequest(r));
    }
    // Scalar twin (Vec-free) of the partition walk: proving it covers EXACTLY [0,N)
    // and every group is contiguous + correctly sized proves group_ranges is a sound
    // exact partition (no trip dropped → no dropped compute; none double-emitted).
    let (covered_end, ok) = group_ranges_cover_ok(&kinds, g);
    assert!(ok);
    assert!(covered_end == N);
}

// ── FAIL-FIRST guard for the mq>1 PREFILL KV-write bug (2026-07-07): the distinct-per-slot
//    prefill cachewr copies MUST each be their OWN singleton group. If they were fused (the
//    original bug: classified GroupKind::Slot → all 512 copies collapsed into 1 group → the
//    shim's single slot-shift dropped all but a subset of slots → partial KV → degenerate
//    output), a run of SlotSolo trips would collapse to ⌈N/g⌉ < N groups. This proves
//    group_ranges emits EXACTLY N singleton groups for N SlotSolo trips at ANY g — i.e. a
//    distinct-slot copy is UNFUSABLE by construction. FAILS if SlotSolo is ever made fusable.
#[kani::proof]
#[kani::unwind(6)]
fn slotsolo_copies_are_never_fused() {
    const N: usize = 4;
    let g: usize = kani::any();
    kani::assume(g >= 1 && g <= 8); // incl. g ≥ N (whole-body fusion) where a Slot run WOULD collapse
    let kinds = [Trip::new(GroupKind::SlotSolo, TripRequest(0)); N];
    let groups = group_ranges(&kinds, g);
    assert!(groups.len() == N); // N singletons, NOT ⌈N/g⌉ — never fused
    for r in groups.iter() {
        assert!(r.end - r.start == 1); // each SlotSolo is its own group
    }
}

// ── FAIL-FIRST guard for the BATCHED-DECODE CROSS-REQUEST FUSION bug (#7, 2026-08-06): NO GROUP MAY
//    SPAN TWO REQUESTS. A group is one launch; a launch resolves ONE request's page table and write
//    cursor and shifts the KV segment base ONCE. So a group holding trips of two requests hands the
//    second request the FIRST one's KV — silently: no crash, no shape error, fluent but wrong tokens
//    (the symptom was a batch answering 2-3 good tokens then decaying). Before `Trip`, only the
//    `Slot` variant carried a request, so `Pure` runs — which is what the attention matmuls and the
//    per-page Kᵀ re-transpose are — fused straight across all 8 requests of a batch at g=512.
//    This proves it cannot happen for ANY trip sequence at ANY g: FAILS the moment a kind is added
//    that fuses without consulting `Trip::req`, which is precisely how six earlier bugs got in.
#[kani::proof]
#[kani::unwind(6)]
fn no_group_spans_two_requests() {
    const N: usize = 4;
    let g: usize = kani::any();
    kani::assume(g >= 1 && g <= 8); // incl. g ≥ N: whole-body fusion, where a run WOULD span the batch
    let mut trips = [Trip::new(GroupKind::Pure, TripRequest(0)); N];
    for slot in trips.iter_mut() {
        let t: u8 = kani::any();
        kani::assume(t <= 4);
        let r: u32 = kani::any();
        kani::assume(r <= 2); // ≥2 distinct requests ⇒ a cross-request run is REACHABLE
        let kind = match t {
            0 => GroupKind::Pure,
            1 => GroupKind::Slot { req: r },
            2 => GroupKind::Slab,
            3 => GroupKind::PageFold,
            _ => GroupKind::SlotSolo,
        };
        *slot = Trip::new(kind, TripRequest(r));
    }
    assert!(!group_spans_two_requests(&trips, g));
}

#[kani::proof]
fn scalarmul_tid_distinct_and_reserved() {
    let (i, j): (usize, usize) = (kani::any(), kani::any());
    kani::assume(i < 14 && j < 14); // ≤14 distinct scales fit below u32::MAX-6 (granite uses ≤4)
    kani::assume(i != j);
    assert!(scalarmul_scale_tid(i) != scalarmul_scale_tid(j)); // distinct scale consts
    assert!(scalarmul_scale_tid(i) < u32::MAX - 6); // below ROPE_P/ATTN_*/RMS_* — no reserved-id collision
}

// ── SYMBOLIC bundle address derivation (the whole-bundle fusion fix): the SYMBOLIC
//    `bundle.mlir` emits every HBM address as `arith.addi %segbase_{seg}, off` (the
//    torch-spyre `kernel_derived` scheme, shared operand SSA). For the on-card address
//    to equal the concrete per-op address, the (seg, off) decomposition MUST reconstruct
//    the original byte address AND land in a real segment. If it didn't, the fused bundle
//    would read/write the WRONG HBM location (silent numeric corruption). Fail-first: a
//    wrong divisor / seg formula breaks reconstruction. Closes <1s (pure u64 arithmetic). ──
#[kani::proof]
fn hbm_seg_off_reconstructs_address() {
    let addr: u64 = kani::any();
    kani::assume(addr < (SEGMENT_OFFSETS.len() as u64) * SEGMENT_SIZE); // a real HBM address
    let (seg, off) = hbm_seg_off(addr);
    assert!((seg as usize) < SEGMENT_OFFSETS.len()); // indexes a declared segment base
    assert!(off < SEGMENT_SIZE); // intra-segment (the `arith.addi` offset operand)
    assert!(seg * SEGMENT_SIZE + off == addr); // base + offset == original address (exact)
    assert!(SEGMENT_OFFSETS[seg as usize] == seg * SEGMENT_SIZE); // %segbase_{seg} const == the real base
}

// ── DeviceWidth (type-safe padding/alignment lock-down): the SOLE device-width rule the emitter n_dev,
//    kernel0 RetileDescriptor, and worker weight-pad all go through — proven 64-aligned + ≥8-splittable
//    (heavy gemms) + m-INDEPENDENT (so worker@m=1 == emitter@m=seq). Concrete granite shapes ⇒ <1s. ──
#[kani::proof]
fn devwidth_lmhead_splittable() {
    let w = DeviceWidth::for_output(1, 49159, 4096).get(); // decode lm_head: prime sticks → padded
    assert!(w == 49664);
    assert!(w % 64 == 0);
    assert!(core_split(w / 64, MAX_CORES) >= 8); // fills the util floor
}
#[kani::proof]
fn devwidth_rope_not_overpadded() {
    // RoPE-rotate [heads=32, hd=128, k=128]: macs < 2^20 ⇒ NOT bumped (stays 128, not over-padded to 512).
    assert!(DeviceWidth::for_output(32, 128, 128).get() == 128);
}
#[kani::proof]
fn devwidth_pointwise_matches_matmul() {
    // A pointwise CONSUMER (e.g. the logits ScalarMul) must address the SAME device width its matmul
    // PRODUCER emitted — else the read misaligns. Proven for the logits (lm_head) + a hidden tensor.
    assert!(DeviceWidth::for_pointwise(49159) == DeviceWidth::for_output(1, 49159, 4096)); // 49664
    assert!(DeviceWidth::for_pointwise(4096) == DeviceWidth::for_output(1, 4096, 4096)); // 4096 (no bump)
    assert!(DeviceWidth::for_pointwise(49159).get() == 49664);
}
#[kani::proof]
fn devwidth_weight_m_independent() {
    // The worker stages weights at m=1; the emitter runs prefill (m=seq) / decode (m=1). Same width ⇒ the
    // staged buffer and the emitted device layout cannot disagree (weights always cross the bump threshold).
    assert!(DeviceWidth::for_output(1, 49159, 4096) == DeviceWidth::for_output(64, 49159, 4096));
    assert!(DeviceWidth::for_output(1, 4096, 4096) == DeviceWidth::for_output(64, 4096, 4096));
}

// ── OUTPUT-width padding (`bump_sticks_to_splittable`): a prime/awkward stick count is bumped so the gemm
//    fills ≥8 cores (util floor). Concrete granite widths ⇒ `core_split` runs concrete (cheap, <1s). ──
#[kani::proof]
fn bump_lmhead_to_splittable() {
    // granite lm_head: 49216 (769 sticks, PRIME ⇒ core_split=1) → 49664 (776 sticks ⇒ core_split=8).
    let b = bump_sticks_to_splittable(49216);
    assert!(b == 49664);
    assert!(b % 64 == 0);
    assert!(b >= 49216);
    assert!(core_split(b / 64, MAX_CORES) >= 8); // fills the util floor
}
#[kani::proof]
fn bump_noop_when_already_splittable() {
    assert!(bump_sticks_to_splittable(4096) == 4096); // 64 sticks, core_split=32 — unchanged
    assert!(bump_sticks_to_splittable(1024) == 1024); // 16 sticks, core_split=16 — unchanged
}

const HD: usize = 64; // head_dim (= STK)
const STK: usize = 64;
const CAP: usize = 256; // cache capacity (cap > STK is where natural ≠ Kᵀ — the multi-day bug)

// Real model head dims the K-cache addressing must hold for. hd=64 was the ONLY dim ever proven; the
// on-card wrong-token symptom is on granite (hd=128) and gemma-4 (hd=512) has hd=512 — BOTH > STK, i.e.
// the K kernel `[hd,cap]` spans multiple `hd` values inside one `cap`-stick group. These parameterized
// helpers exercise the exact multi-stick regime the pinned-HD=64 harness could never see.
const HD_GRANITE: usize = 128;
const HD_GEMMA4: usize = 512;

// ── THE K-CACHE BUG CLASS, proven for ALL (slot, d) at EACH head_dim ──
// The producer's Kᵀ write offset must equal the score matmul's `[hd,cap]` kernel read address. If these
// ever diverge (the natural-vs-Kᵀ bug), the score reads scrambled prefix-K. Kani checks every
// slot∈[0,cap), d∈[0,hd) — not one seed. `hd`/`cap` are CONCRETE per harness ⇒ CBMC stays fast (<1s).
fn kc_kt_write_equals_score_read_for(hd: usize, cap: usize) {
    let slot: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(slot < cap);
    kani::assume(d < hd);
    let producer = kcache_kt_write_offset(slot, d, hd, cap, STK);
    let consumer = dev_off(&[hd, cap], 1, &[d, slot]);
    assert!(producer == consumer);
}
#[kani::proof]
fn kc_kt_write_equals_score_read() {
    kc_kt_write_equals_score_read_for(HD, CAP); // hd=64 (= STK): the original single-stick case
}
#[kani::proof]
fn kc_kt_write_equals_score_read_hd128() {
    kc_kt_write_equals_score_read_for(HD_GRANITE, CAP); // granite hd=128: 2 hd-values per cap-stick group
}
#[kani::proof]
fn kc_kt_write_equals_score_read_hd512() {
    kc_kt_write_equals_score_read_for(HD_GEMMA4, CAP); // gemma-4 hd=512: 8 hd-values per cap-stick group
}

// ── THE V-CACHE bug class (twin of K) ── The producer's V write offset must equal the VALUE bmm's
// `[cap,hd]`-kernel read (sticked on hd). If they diverge the value bmm reads scrambled V ⇒ wrong
// attention output. Checks every slot∈[0,cap), d∈[0,hd). Concrete hd/cap ⇒ CBMC fast.
fn vc_write_equals_value_bmm_read_for(hd: usize, cap: usize) {
    let slot: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(slot < cap);
    kani::assume(d < hd);
    let producer = vcache_write_offset(slot, d, hd, cap, STK);
    let consumer = dev_off(&[cap, hd], 1, &[slot, d]); // value bmm reads vc as [k=cap, n=hd] kernel, stick=hd
    assert!(producer == consumer);
}
#[kani::proof]
fn vc_write_equals_value_bmm_read() {
    vc_write_equals_value_bmm_read_for(HD, CAP); // hd=64 (= STK)
}
#[kani::proof]
fn vc_write_equals_value_bmm_read_hd128() {
    vc_write_equals_value_bmm_read_for(HD_GRANITE, CAP); // granite hd=128
}
#[kani::proof]
fn vc_write_equals_value_bmm_read_hd512() {
    vc_write_equals_value_bmm_read_for(HD_GEMMA4, CAP); // gemma-4 hd=512
}

// ── THE mq>1 PREFILL NO-MAX SOFTMAX PADDING OVERFLOW (2026-07-07, root cause of the inf after the
// collapse fix) ── The mq>1 attention scores snb[mqu, mqp] are exp'd WITHOUT the max-subtraction
// (no-max softmax). For a MASKED column c (causal col>row, and ALL padding cols [mqu..mqp)), the score
// is `raw + cmask`, cmask = MASK_NEG (an ADDITIVE bias). exp(fp16) OVERFLOWS to inf once its argument
// exceeds ~11 (exp(11)=59874 < 65504=f16 max < exp(12)). If the K in the padding cols [mqu..mqp) is
// UNINITIALIZED garbage (the head-major selector writes only mqu real rows; [mqu..mqp) are unwritten),
// `raw = qh·K_pad·scale` is UNBOUNDED (garbage up to ±f16_max) ⇒ raw+MASK_NEG can be >>11 ⇒ exp → inf ⇒
// nan attention out ⇒ residual/KV inf ⇒ the measured ' signature' degenerate. An ADDITIVE mask CANNOT
// tame unbounded garbage (MASK_NEG is f16-bounded ≥ -65504). FIX: the padding-col raw score must be
// ZEROED (zero the K padding [mqu..mqp) so raw=0 ⇒ arg = MASK_NEG < 0 ⇒ exp=0, safe). Fail-first below.
const EXP_F16_SAFE_ARG: i32 = 11; // exp(x) fp16-finite iff x ≤ ~11
const F16_ABS_MAX: i32 = 65504;
const SOFTMAX_MASK_NEG: i32 = 30000; // |MASK_NEG| the additive causal/pad mask subtracts
// exp argument for a MASKED column: pre-mask raw score (garbage or 0) minus the additive mask.
fn masked_softmax_exp_arg(raw_pad_score: i32) -> i32 {
    raw_pad_score - SOFTMAX_MASK_NEG
}
#[kani::proof]
fn prefill_softmax_padding_must_be_zeroed() {
    // THE FIX (emitter now zeroes the padding-col K ⇒ raw=0): exp arg = -MASK_NEG < 0 ⇒ exp finite (safe):
    assert!(masked_softmax_exp_arg(0) <= EXP_F16_SAFE_ARG);
    // The BUG this replaces (VERIFIED fail-first 2026-07-07, "1 of 4 failed"): with UNINITIALIZED garbage
    // K in the padding cols, `masked_softmax_exp_arg(raw)` for raw∈[−f16max, f16max] can exceed the
    // exp-safe arg (raw=65504 ⇒ 35504 ≫ 11) ⇒ exp → inf. Encoded as a documented non-invariant: the
    // additive MASK_NEG CANNOT bound f16-range garbage — only zeroing the padding K does.
    let garbage: i32 = kani::any();
    kani::assume(garbage > SOFTMAX_MASK_NEG + EXP_F16_SAFE_ARG && garbage <= F16_ABS_MAX);
    assert!(masked_softmax_exp_arg(garbage) > EXP_F16_SAFE_ARG); // garbage above the mask ⇒ exp overflows
}

// ── THE mq>1 PREFILL KV ZERO-COPY REGION SAFETY (2026-07-07) ── The fix above zeroes the padding rows
// [mqu..mqp) of each head-major kh/vh via a per-head copy at dst offset `kvh*mqp*hd + mqu*hd`, length
// `(mqp-mqu)*hd`. This copy MUST hit ONLY the padding rows of its own head — never a REAL row [0..mqu) of
// any head (which the selectors just wrote), and never spill into the next head's block. A wrong offset
// (e.g. forgetting `+ mqu*hd`, or a length of `mqp*hd`) would CLOBBER the real q/k/v the attention reads
// ⇒ silent wrong output. Model the byte regions and prove disjointness for ALL heads (Kani exhausts them).
const KZ_MQU: u64 = 8; // mqu (real query rows)
const KZ_MQP: u64 = 64; // mqp (stick-padded rows)
const KZ_HD: u64 = 64; // head_dim
const KZ_NKVH: u64 = 8; // kv heads
fn kv_zero_copy_dst(kvh: u64) -> (u64, u64) {
    // (offset, len) of the zero-copy for head kvh — MUST match the emitter's `kvh*mqp*hd + mqu*hd`.
    (
        kvh * KZ_MQP * KZ_HD + KZ_MQU * KZ_HD,
        (KZ_MQP - KZ_MQU) * KZ_HD,
    )
}
fn kv_real_region(kh: u64) -> (u64, u64) {
    // (offset, len) of the REAL rows [0..mqu) the selectors wrote for head kh.
    (kh * KZ_MQP * KZ_HD, KZ_MQU * KZ_HD)
}
#[kani::proof]
fn prefill_kv_zero_copy_hits_only_padding() {
    let kvh: u64 = kani::any();
    let kh: u64 = kani::any();
    kani::assume(kvh < KZ_NKVH && kh < KZ_NKVH);
    let (z_off, z_len) = kv_zero_copy_dst(kvh);
    // (1) stays within head kvh's block [kvh·mqp·hd, (kvh+1)·mqp·hd) — no spill into the next head:
    assert!(z_off + z_len <= (kvh + 1) * KZ_MQP * KZ_HD);
    assert!(z_off >= kvh * KZ_MQP * KZ_HD);
    // (2) disjoint from EVERY head's real rows: no byte of the zero-copy lands in a real region. Pick an
    // arbitrary byte in the copy and prove it is outside kh's real region for every kh.
    let b: u64 = kani::any();
    kani::assume(b < z_len);
    let o = z_off + b;
    let (r_off, r_len) = kv_real_region(kh);
    assert!(o < r_off || o >= r_off + r_len); // the zeroed byte is never a real row of any head
}

// ── THE mq>1 PREFILL RMSNORM ROW-0 COLLAPSE (root cause, 2026-07-07) ── assemble_rmsnorm's per-row
// scalar-lane `ew` closure reads BOTH operands FULL (mb_broadcast=false) over [rows,stick]. But halfc
// (RMS_HALF_TID) is a [1,stick] const, and t_one/t_th/t_neg1 derive from it. A FULL read of a [1,stick]
// operand at output row r reads its row r — but the const has ONLY row 0 (0.5); rows>0 read BEYOND it
// (zeros). So t_one=½+½ and t_th=1+½ are 1.0/1.5 ONLY at row 0, and 0.0 for rows>0 ⇒ the Newton rsqrt
// `d = t_th − 0.5·m·y²` uses t_th=0 ⇒ inv→0 ⇒ rmsnorm out rows 1..7 = 0 ⇒ COLLAPSE to row 0 ⇒ the q/k/v
// proj read row-0 rmsnorm ⇒ degenerate mq>1 prefill output. Decode (rows=1) is unaffected (only row 0).
// FIX: read the [1,stick] const with mb_broadcast (row 0 for every output row). Model the const value:
fn rmsnorm_half_at(row: usize, mb_broadcast: bool) -> u32 {
    // returns 0.5 (as ×1000 fixed-point to stay integer/CBMC-fast) if in-bounds, else 0 (OOB full-read).
    if mb_broadcast || row == 0 { 500 } else { 0 } // 0.500
}
fn rmsnorm_t_th_x1000(row: usize, mb_broadcast: bool) -> u32 {
    let one = rmsnorm_half_at(row, mb_broadcast) + rmsnorm_half_at(row, mb_broadcast); // rmone: ½+½ = 1.000
    one + rmsnorm_half_at(row, mb_broadcast) // rmth: 1 + ½ = 1.500
}
// ── mq>1 PREFILL RMSNORM SFP-RECIPROCAL COVERAGE (root cause #2, 2026-07-07) ── the amax→ramax and the
// Newton-seed reciprocals are SFP rank-1 over `stick_elems`. amax/mp1 are [rows,stick] (row r's value at
// r·stick). A rank-1 recip covers elements [0, stick_elems); row r is covered iff r·stick < stick_elems.
// The bug used stick_elems=stick ⇒ ONLY row 0 covered ⇒ ramax[r>0]=0 ⇒ xs=x·ramax=0 ⇒ rmsnorm collapse
// to row 0 (measured on-card: t448 row-0-only despite input t447=8 rows). FIX: stick_elems=rows·stick
// covers EVERY row (recip is elementwise; the garbage between sticks is recip'd but never read — xs's
// out-broadcast reads only ramax[r·stick]). Decode rows=1 unchanged.
fn sfp_recip_covers_row(r: usize, stick: usize, stick_elems: usize) -> bool {
    r * stick < stick_elems // row r's scalar lives at r·stick; the rank-1 recip covers [0, stick_elems)
}
#[kani::proof]
fn rmsnorm_recip_must_cover_all_rows() {
    let rows = 8usize;
    let stick = 64usize;
    let r: usize = kani::any();
    kani::assume(r < rows);
    assert!(sfp_recip_covers_row(r, stick, rows * stick)); // THE FIX: rows·stick covers every prefill row
    assert!(sfp_recip_covers_row(0, stick, stick)); // the bug covered ONLY row 0 (decode/mq=1 correct)
    let r1: usize = kani::any();
    kani::assume(r1 >= 1 && r1 < rows);
    assert!(!sfp_recip_covers_row(r1, stick, stick)); // ...and NOT rows>0 (stick_elems=stick) — the collapse
}

#[kani::proof]
fn rmsnorm_const_lane_must_broadcast() {
    let row: usize = kani::any();
    kani::assume(row < 8); // mq=8 prefill rows
    // THE FIX (emitter now mb_broadcasts the [1,stick] const): t_th=1.500 for EVERY output row.
    assert!(rmsnorm_t_th_x1000(row, true) == 1500);
    // The BUG this replaces (VERIFIED fail-first 2026-07-07): the full-read `rmsnorm_t_th_x1000(row,
    // false)` == 1500 only at row 0 and == 0 for rows>0 (const OOB) — that assertion FAILED, which is
    // exactly the rmsnorm row-0 collapse. Encoded here as a documented non-invariant, not asserted:
    assert!(rmsnorm_t_th_x1000(0, false) == 1500); // full read is correct ONLY at row 0 (decode/mq=1)
    let r1: usize = kani::any();
    kani::assume(r1 >= 1 && r1 < 8);
    assert!(rmsnorm_t_th_x1000(r1, false) == 0); // and WRONG (0) for every prefill row>0 — the collapse
}

// ── PATH A (leak fix): NATURAL on-card KV write ≡ NATURAL prefix-score consumer ──
// The leak fix deletes host_kv_write and runs the on-card cachewr copy, which writes K NATURAL
// slot-major (slot*hd+d, a stride-1 dense copy — expressible, no restickify). For the prefix score to
// be correct, its CONSUMER (currently a cap-sticked Kᵀ matmul) must be flipped to a NATURAL mul+reduce
// that reads kc[slot][d] at the SAME natural offset. This harness is the producer≡consumer contract
// for the natural layout. FAIL-FIRST: with the CURRENT cap-sticked consumer, a natural write DIVERGES
// for cap>STK (the measured incoherence "0----") — Kani finds the witness. GREEN once the consumer is
// natural. hd/cap CONCRETE ⇒ <1s.
fn oncard_natural_kv_write_equals_natural_prefix_read_for(hd: usize, cap: usize) {
    let slot: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(slot < cap);
    kani::assume(d < hd);
    let producer_natural = slot * hd + d; // on-card dense stride-1 copy write of kc[slot][d]
    // Path A step-2 CONTRACT: the natural mul+reduce consumer must read kc FLAT row-major (device
    // [cap*hd/64,64] stores elements row-major), i.e. dev_off with a non-1 stick_idx → slot*hd+d.
    let consumer_flat = dev_off(&[cap, hd], 0, &[slot, d]);
    assert!(producer_natural == consumer_flat); // GREEN: copy write ≡ flat read (the step-2 target)
    // Contrast (NOT asserted — they only sometimes coincide): the STICKED read dev_off([cap,hd],1,·)
    // and the current cap-sticked-Kᵀ read dev_off([hd,cap],1,·) are DIFFERENT addressings; a natural
    // copy feeding either is the layout mismatch the NO_HOST_KV probe measured as incoherent "0----".
}
#[kani::proof]
fn oncard_natural_kv_write_equals_natural_prefix_read() {
    oncard_natural_kv_write_equals_natural_prefix_read_for(HD, CAP);
}
#[kani::proof]
fn oncard_natural_kv_write_equals_natural_prefix_read_hd128() {
    oncard_natural_kv_write_equals_natural_prefix_read_for(HD_GRANITE, CAP);
}

// ── PATH B (host_kv_write nuke): NATURAL store → on-card RESTICKIFY → matmul-kernel read ──
// The leak fix deletes the shim's host KV scatter and instead (a) stores K/V NATURAL on-card (dense
// stride-1 `[1,hd]` per-slot copy, expressible) and (b) restickifies the natural cache into the exact
// kernel layout the score/value matmul reads — mirroring torch-spyre `key.transpose(-2,-1).contiguous()`
// (a real data-move restickify, `test_matmul_x_yt optimal_cost=y.numel()`). The multi-row REDUCE is
// broken on-card (rows>1 reduce-MAX returns 0 — see lower_attn_node), so a natural mul+reduce prefix
// score is NOT viable; the matmul (one op, n=cap output stick) is. Hence restickify, not reduce.
//
// This harness is the FULL producer≡consumer contract for Path B: it FAILS if the restickify's INPUT
// layout is declared transposed (the cos≈0 bug — the current opspec keeps dim order [mb,out,y] and only
// swaps the stick axis, so it reads the flat-natural `[cap,hd]` cache as `[hd,cap]` → scrambled). GREEN
// only when input reads flat-natural AND output writes the Kᵀ cell the matmul reads. hd/cap CONCRETE ⇒ <1s.
fn pathb_k_natural_store_restickify_matmul_chain_for(hd: usize, cap: usize) {
    let slot: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(slot < cap);
    kani::assume(d < hd);
    // 1. NATURAL store writes K[slot][d] flat row-major over `[cap,hd]` (stride-1 hd copy per slot).
    let store_off = slot * hd + d;
    // 2. The restickify INPUT must read that SAME flat cell: its input is flat-natural `[cap,hd]`
    //    (stick_idx 0 = row-major), NOT a `[hd,cap]` reinterpretation. This is the assertion the
    //    current (dim-order-preserving) restickify_opspec VIOLATES → the cos≈0 scramble.
    let restick_in = dev_off(&[cap, hd], 0, &[slot, d]);
    assert!(store_off == restick_in);
    // 3. The restickify OUTPUT writes K[slot][d] to the Kᵀ cell `[hd,cap]` cap-sticked; the score
    //    matmul reads kernel element (in=d, out=slot) at exactly that cell. Restickify-out == matmul-read.
    let restick_out = kcache_kt_write_offset(slot, d, hd, cap, STK);
    let matmul_read = dev_off(&[hd, cap], 1, &[d, slot]);
    assert!(restick_out == matmul_read);
}
#[kani::proof]
fn pathb_k_natural_store_restickify_matmul_chain() {
    pathb_k_natural_store_restickify_matmul_chain_for(HD, CAP);
}
#[kani::proof]
fn pathb_k_natural_store_restickify_matmul_chain_hd128() {
    pathb_k_natural_store_restickify_matmul_chain_for(HD_GRANITE, CAP);
}

// ── PATH B (V side): NATURAL store → on-card RESTICKIFY → value-bmm kernel read ──
// The value bmm `out_pre = probs[1,cap] · V[cap,hd] → [1,hd]` reads V as kernel `[k=cap, n=hd]` sticked
// on n=hd → device `[hd/64, cap, 64]`, element (slot,d) at `dev_off([cap,hd],1,[slot,d])` = the named
// vcache_write_offset. The natural V store writes flat `slot*hd+d`. For hd>STK these DIFFER, so V ALSO
// needs a restickify (flat-natural `[cap,hd]` → hd-sticked `[cap,hd]`: SAME dim order, stick 0→hd —
// UNLIKE K, no transpose). This locks the V chain: store-flat == restickify-in-flat, restickify-out ==
// bmm-read. FAILS if V's restickify output is declared with the wrong stick. hd/cap CONCRETE ⇒ <1s.
fn pathb_v_natural_store_restickify_bmm_chain_for(hd: usize, cap: usize) {
    let slot: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(slot < cap);
    kani::assume(d < hd);
    let store_off = slot * hd + d;
    let restick_in = dev_off(&[cap, hd], 0, &[slot, d]); // flat-natural read of V[slot][d]
    assert!(store_off == restick_in);
    let restick_out = vcache_write_offset(slot, d, hd, cap, STK); // hd-sticked [cap,hd] kernel cell
    let bmm_read = dev_off(&[cap, hd], 1, &[slot, d]);
    assert!(restick_out == bmm_read);
}
#[kani::proof]
fn pathb_v_natural_store_restickify_bmm_chain() {
    pathb_v_natural_store_restickify_bmm_chain_for(HD, CAP);
}
#[kani::proof]
fn pathb_v_natural_store_restickify_bmm_chain_hd128() {
    pathb_v_natural_store_restickify_bmm_chain_for(HD_GRANITE, CAP);
}

// ── RESIDUAL ALIAS (host-routing nuke #1) is WAR-safe + threads the loop-carry in-place ──
// The residual thread_hidden/thread_to_suffix host round-trips were replaced by placing `hidden_out`
// (the LAST body node's output) AND `suffix_in` (the first suffix node's input) at `hidden_in`'s
// resident address, so the loop-carried residual threads IN-PLACE (no D2H/H2D). Two invariants:
//  (A) WAR-safety: within an iteration every READ of the aliased cell (hidden_in, consumed by the
//      EARLY residual add) must precede the WRITE (hidden_out, the LAST body op). The emitter's
//      pre-scan sets the write index = last body node; the final residual add reads h1+mlp, NOT
//      hidden_in, so hidden_in's reads are strictly earlier. FAIL-FIRST: if the last op read hidden_in
//      (read_idx == write_idx), the in-place write would clobber a live value.
//  (B) Loop-carry threading: aliasing place(hidden_out)=place(suffix_in)=place(hidden_in)=A means the
//      value the body writes to A (the layer output) is EXACTLY what the suffix / next iteration reads
//      at A. FAIL-FIRST: without the alias (place(hidden_out)=B≠A) the suffix reads stale hidden_in at
//      A → orphaned residual (the incoherence the alias fixes).
fn residual_alias_war_and_thread_for(n_body: usize) {
    kani::assume(n_body >= 2 && n_body <= 1024);
    // (A) WAR-safety.
    let write_idx = n_body - 1; // hidden_out = last body node's output (pre-scan `plast`)
    let read_idx: usize = kani::any();
    kani::assume(read_idx < n_body);
    kani::assume(read_idx != write_idx); // the last op reads h1+mlp, NOT hidden_in (input exclusion)
    assert!(read_idx < write_idx); // every hidden_in read precedes the hidden_out write ⇒ no clobber
    // (B) Loop-carry threading via address identity. Model 3 tids' placements as byte addresses.
    let a_hidden_in: u64 = kani::any();
    // Aliased: hidden_out & suffix_in take hidden_in's address.
    let a_hidden_out = a_hidden_in;
    let a_suffix_in = a_hidden_in;
    // The body writes the layer output to a_hidden_out; the suffix + next-iter read a_suffix_in /
    // a_hidden_in. All equal ⇒ the suffix/next-iter read exactly the layer output (in-place thread).
    assert!(a_hidden_out == a_suffix_in && a_suffix_in == a_hidden_in);
    // Negative witness: a non-aliased hidden_out at a DIFFERENT address would NOT thread.
    let a_unaliased: u64 = kani::any();
    kani::assume(a_unaliased != a_hidden_in);
    assert!(a_unaliased != a_suffix_in); // proves the alias (not a stray address) is what threads it
}
#[kani::proof]
fn residual_alias_war_and_thread() {
    residual_alias_war_and_thread_for(931); // granite body op count (this session's fused decode body)
}

// ── FUSED cachewr Slot GROUP: the single group-level slot-shift places every (head, slab) at slot p ──
// The launch-reduction fuses the per-(head,slab) cachewr Slot singletons into one group; the shim applies
// the slot-shift (slot_pos·slot_stride_bytes) to the fused group's KV segment base ONCE (seg-relative op
// addresses + a shifted segment base ⇒ no double-count). Each cachewr op, baked at head qh slab s slot 0
// (seg2-relative offset qh·cap·hd + s·cap·stk + d'), then resolves to slot slot_pos of head qh slab s.
// Proves: (a) the shifted write == the SLAB-MAJOR cache cell K[qh][p][d] (head base + dev_off([cap,hd],1,
// [p,d])); (b) distinct heads write disjoint cells (no clobber) — so fusing is bit-identical to the
// singletons, just fewer launches. slot_stride_bytes = stk·2, UNIFORM across slabs. FAIL-FIRST: a wrong
// stride (≠stk) or a non-uniform per-op shift would break (a). hd/cap/nqh CONCRETE ⇒ <1s.
fn fused_cachewr_group_slot_shift_correct_for(hd: usize, cap: usize, nqh: usize) {
    let qh: usize = kani::any();
    let d: usize = kani::any();
    let p: usize = kani::any();
    kani::assume(qh < nqh && d < hd && p < cap);
    let stk = STK;
    let stride_elems = stk; // slot_stride_bytes = stk*2 ⇒ stk elements (uniform across all slabs)
    let (s, dprime) = (d / stk, d % stk); // element (p,d) lives in slab s at within-slab col d'
    // op baked at head qh, slab s, slot 0: seg2-relative element offset.
    let baked = qh * cap * hd + s * cap * stk + dprime;
    // shim shifts the WHOLE group's seg2 base by slot_pos*stride (uniform for the group).
    let shifted = baked + p * stride_elems;
    // == the slab-major cache cell K[qh][p][d] (head base + dev_off([cap,hd],1,[p,d])).
    assert!(shifted == qh * cap * hd + dev_off(&[cap, hd], 1, &[p, d]));
    // distinct head at the same (p,d) → a distinct cell (no clobber across the fused group).
    let qh2: usize = kani::any();
    kani::assume(qh2 < nqh && qh2 != qh);
    let shifted2 = qh2 * cap * hd + s * cap * stk + dprime + p * stride_elems;
    assert!(shifted != shifted2);
}
#[kani::proof]
fn fused_cachewr_group_slot_shift_correct() {
    fused_cachewr_group_slot_shift_correct_for(HD, CAP, NQH_G); // hd=64, cap=256, 32 q-heads (granite-3.3-2b)
}
#[kani::proof]
fn fused_cachewr_group_slot_shift_correct_hd128() {
    fused_cachewr_group_slot_shift_correct_for(HD_GRANITE, CAP, NQH_G); // granite-3.3-8b / llama-3.2-3b hd=128
}

// ── RESTICKIFY multi-core split is DISJOINT + COVERING over the [hd,cap] Kᵀ output ──
// The on-card K restickify (restickify_kt_opspec_2d) runs MULTI-core: WorkPlan splits the OUTPUT stick
// `cap` (=`y`) across S cores, each core writing the cap-slice `[cs·per, (cs+1)·per)` × ALL hd rows of
// the `[hd,cap]` Kᵀ kernel. `decode_per_core_regions_disjoint_for` covers only a `[1,cols]` output;
// the restickify output has hd rows, so prove it directly here. If the split clobbered (two cores
// writing the same cell) or dropped a slot, the transposed cache is corrupt → wrong prefix scores
// (silently, no crash). Uses the ACTUAL dev_off([hd,cap],1,·) the emitter/per_core_addr bakes.
// CONCRETE hd/cap keeps cs·per linear; S symbolic covers every core count ≤ 32 dividing cap.
fn restickify_core_split_disjoint_covering_for(hd: usize, cap: usize) {
    let s: usize = kani::any();
    kani::assume(s >= 1 && s <= 32 && cap % s == 0); // even cap-split across S cores (WorkPlan divisor)
    let per = cap / s; // per-core cap extent
    // DISJOINT: distinct cores, any rows/cols in their slices → distinct device offsets.
    let (c1, c2): (usize, usize) = (kani::any(), kani::any());
    kani::assume(c1 < s && c2 < s && c1 != c2);
    let (d1, j1): (usize, usize) = (kani::any(), kani::any());
    let (d2, j2): (usize, usize) = (kani::any(), kani::any());
    kani::assume(d1 < hd && d2 < hd && j1 < per && j2 < per);
    let slot1 = c1 * per + j1; // core c1's absolute slot
    let slot2 = c2 * per + j2;
    let off1 = dev_off(&[hd, cap], 1, &[d1, slot1]); // Kᵀ kernel cell (in=d, out=slot)
    let off2 = dev_off(&[hd, cap], 1, &[d2, slot2]);
    assert!(off1 != off2); // no two distinct cores' cells collide (dev_off injective + disjoint slots)
    assert!(off1 < hd * cap); // in the per-head footprint (no clobber of the next head)
    // COVERING: every slot ∈ [0,cap) belongs to exactly one core's slice.
    let slot: usize = kani::any();
    kani::assume(slot < cap);
    let owner = slot / per;
    assert!(owner < s && slot >= owner * per && slot < (owner + 1) * per);
}
#[kani::proof]
fn restickify_core_split_disjoint_covering() {
    restickify_core_split_disjoint_covering_for(HD, CAP); // hd=64, cap=256 (granite decode)
}

// ── RESIDENT KV bridge: the decode-step SLAB-MAJOR SLOT write lands at the cache cell the score reads ──
// The worker's per-forward prefix_k/prefix_v H2D re-upload is GONE (resident KV). Correctness now rests
// on: the on-card cachewr for decode step `p` writes K[p] to the resident seg2 cache at the SAME cell the
// restickify (→ score) / value-bmm later reads as slot `p`. The cachewr writes SLAB-MAJOR: one `[1,stk]`
// copy per (head, slab s=d/stk) baked at `qh·cap·hd + s·cap·stk + d'` (d'=d%stk, slot 0); the shim adds
// `seq_pos·slot_stride_bytes` with `slot_stride_bytes = stk·2` (elem stride = stk), UNIFORM across slabs.
// So step p's per-head element offset is `s·cap·stk + p·stk + d'` = `dev_off([cap,hd],1,[p,d])` =
// `vcache_write_offset` — EXACTLY the stick-last cell the K restickify INPUT and V value-bmm read, at ANY
// head_dim. FAIL-FIRST: a wrong slot stride (e.g. `hd` instead of `stk`) lands K[p] in the wrong cell →
// the resident cache accumulates scrambled prefix → silently-wrong attention (no crash). hd/cap CONCRETE.
fn resident_kv_slot_write_lands_at_natural_row_for(hd: usize, cap: usize) {
    let p: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(p < cap && d < hd);
    let stk = STK;
    // SLAB-MAJOR store: element (p,d) is in slab s=d/stk at within-slab col d'=d%stk. The slab op is baked
    // at s·cap·stk + d' (slot 0); the shim's uniform slot shift adds p·stk (slot_stride_bytes = stk·2).
    let (s, dprime) = (d / stk, d % stk);
    let cachewr_off = s * cap * stk + p * stk + dprime;
    // That equals the stick-last cell the restickify input / value-bmm read (dev_off([cap,hd],1,·)).
    assert!(cachewr_off == dev_off(&[cap, hd], 1, &[p, d]));
    assert!(cachewr_off == vcache_write_offset(p, d, hd, cap, stk));
}
#[kani::proof]
fn resident_kv_slot_write_lands_at_natural_row() {
    resident_kv_slot_write_lands_at_natural_row_for(HD, CAP); // HD == 64 == STK (granite-3.3-2b)
}
#[kani::proof]
fn resident_kv_slot_write_lands_at_natural_row_hd128() {
    resident_kv_slot_write_lands_at_natural_row_for(HD_GRANITE, CAP); // granite-3.3-8b / llama-3.2-3b hd=128
}

// ── OPSPEC BRIDGE: `restickify_kt_opspec_2d`'s DECLARED layout REALIZES the Path B contract ──
// The `pathb_*` harnesses lock the abstract offset contract; THIS closes the bridge to the opspec I
// actually built. `per_core_addr` (lower_subtile_tape_to_superdsc.rs:2154) computes a TensorArg's
// on-card device offset as `dev_off(host_size, stick_idx, idx)` with `stick_idx = layout.position(stick)`
// — the SAME shared model that produces the DSC descriptors. So a TensorArg's on-card addressing is
// FULLY determined by its (layout order, stick name). Here I mirror EXACTLY the two TensorArgs
// `restickify_kt_opspec_2d` declares and prove, through that model, that INPUT reads flat-natural and
// OUTPUT writes the kt cell the score matmul reads. FAILS if the opspec's dim order / stick is wrong
// (e.g. the rank-3 same-dim-order bug: input layout ["hd","cap"] would give stick_idx→scrambled).
// `stick_idx` is derived by `position()` (not hardcoded), the identical computation as per_core_addr,
// so this is non-circular w.r.t. the offset formula.
// REVISED 2026-07-05 (on-card DISCOVERY): the stick-on-dim0 INPUT crashed dxp with std::out_of_range
// map::at (every deployed op sticks the LAST dim; dxp's map has no entry for a non-last stick). The
// stick-LAST input reads `dev_off([cap,hd],1,·)`; the on-card cachewr now stores K/V SLAB-MAJOR at that
// SAME `dev_off([cap,hd],1,·)` (= `vcache_write_offset`), so the dxp-friendly stick-last input reads the
// resident cache at ANY head_dim. At hd==STK the single slab collapses to the flat `slot*hd+d`, so
// granite-3.3-2b is byte-identical; hd>STK = granite-3.3-8b / llama-3.2-3b (128), gemma (256/512).
fn restickify_opspec_realizes_pathb_contract_for(hd: usize, cap: usize) {
    let slot: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(slot < cap && d < hd);
    let stk = STK;

    // INPUT TensorArg (verbatim from restickify_kt_opspec_2d): layout ["y","out"] (=[cap,hd]),
    // stick "out" (=hd, LAST). CANONICAL names so N_/dataStageParam serialize (else dxp import map::at).
    let in_layout = ["y", "out"];
    let in_sizes = [cap, hd]; // y=cap, out=hd
    let in_stick_idx = in_layout.iter().position(|&x| x == "out").unwrap();
    let in_off = dev_off(&in_sizes, in_stick_idx, &[slot, d]);
    // The stick-last input reads EXACTLY the SLAB-MAJOR store (`vcache_write_offset` = dev_off([cap,hd],
    // 1,·)) the cachewr writes — at ANY head_dim (at hd==STK this is the flat `slot*hd+d`).
    assert!(in_off == vcache_write_offset(slot, d, hd, cap, stk));
    assert!(in_stick_idx == in_layout.len() - 1); // stick IS the last dim ⇒ dxp-friendly

    // OUTPUT TensorArg (verbatim): layout ["out","y"] (=[hd,cap]), stick "y" (=cap, LAST).
    let out_layout = ["out", "y"];
    let out_sizes = [hd, cap]; // out=hd, y=cap
    let out_stick_idx = out_layout.iter().position(|&x| x == "y").unwrap();
    let out_off = dev_off(&out_sizes, out_stick_idx, &[d, slot]);
    // == the exact cell the score matmul reads (kcache_kt_write_offset == dev_off([hd,cap],1,·)).
    assert!(out_off == kcache_kt_write_offset(slot, d, hd, cap, stk));
    assert!(out_off == dev_off(&[hd, cap], 1, &[d, slot]));
    assert!(out_stick_idx == out_layout.len() - 1); // ⇒ device_size [cap/64,hd,64] = kt tiling
}
#[kani::proof]
fn restickify_opspec_realizes_pathb_contract() {
    restickify_opspec_realizes_pathb_contract_for(HD, CAP); // HD == 64 == STK (granite-3.3-2b)
}
#[kani::proof]
fn restickify_opspec_realizes_pathb_contract_hd128() {
    restickify_opspec_realizes_pathb_contract_for(HD_GRANITE, CAP); // granite-3.3-8b / llama-3.2-3b hd=128
}
#[kani::proof]
fn restickify_opspec_realizes_pathb_contract_hd512() {
    restickify_opspec_realizes_pathb_contract_for(HD_GEMMA4, CAP); // gemma global hd=512
}

// V CACHE (no re-stick at ANY head_dim): the SLAB-MAJOR store (`vcache_write_offset` = dev_off([cap,hd],
// 1,·)) IS the hd-sticked kernel the value bmm reads, so a V restickify would be an identity. At hd==STK
// the slab store also equals the flat `slot*hd+d` (granite-3.3-2b). The emitter SKIPS the V restickify at
// EVERY head_dim (the resident vc is read directly by the value bmm).
#[kani::proof]
fn v_natural_equals_hdstick_when_hd_is_stk() {
    let cap: usize = CAP;
    let hd: usize = STK;
    let slot: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(slot < cap && d < hd);
    assert!(slot * hd + d == vcache_write_offset(slot, d, hd, cap, STK));
    assert!(slot * hd + d == dev_off(&[cap, hd], 1, &[slot, d]));
}
#[kani::proof]
fn v_slabstore_equals_hdstick_hd128() {
    let cap: usize = CAP;
    let hd: usize = HD_GRANITE; // granite-3.3-8b / llama-3.2-3b hd=128
    let slot: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(slot < cap && d < hd);
    // The slab-major store IS the hd-sticked value-bmm kernel at hd=128 (no V restickify needed).
    assert!(vcache_write_offset(slot, d, hd, cap, STK) == dev_off(&[cap, hd], 1, &[slot, d]));
}

// ── GQA-EXPAND bridge: nuking host_kv_write moves the GQA expand (nkvh→nqh) on-card ──
// The host scatter fills q-head `qh`'s cache from KV-head `gqa_kv_head(qh)`; the on-card expand copies
// must use the IDENTICAL mapping. The reference (`attn_reference`) is fed nqh already-expanded heads,
// so a wrong expand ⇒ each q-head attends the wrong KV head ⇒ wrong attention (no crash). The mapping
// must be BLOCKED (`qh/gqa`), NOT interleaved (`qh%nkvh`) — the classic GQA layout bug. FAIL-FIRST:
// if the on-card expand read the interleaved head this proof's block-containment assert fails.
fn gqa_expand_partition_ok_for(nqh: usize, nkvh: usize) {
    kani::assume(nkvh >= 1 && nqh >= nkvh && nqh % nkvh == 0);
    let gqa = nqh / nkvh;
    kani::assume(gqa >= 1);
    let qh: usize = kani::any();
    kani::assume(qh < nqh);
    let kvh = gqa_kv_head(qh, gqa);
    // (a) IN-BOUNDS: never reads past the nkvh-head staged prefix (else OOB read of another layer).
    assert!(kvh < nkvh);
    // (b) BLOCKED & exhaustive: q-head qh lands in EXACTLY its kv-head's contiguous block
    //     [kvh·gqa, (kvh+1)·gqa). Fails for an interleaved (qh%nkvh) mapping — the GQA scramble.
    assert!(qh >= kvh * gqa && qh < (kvh + 1) * gqa);
    // (c) the new-token cachewr uses the SAME gqa_kv_head ⇒ prefix expand & new token agree per head.
    assert!(kvh == gqa_kv_head(qh, gqa));
}
#[kani::proof]
fn gqa_expand_partition_ok() {
    gqa_expand_partition_ok_for(NQH_G, NKVH_G); // granite 32 q / 8 kv (gqa=4)
}
#[kani::proof]
fn gqa_expand_partition_ok_mha() {
    gqa_expand_partition_ok_for(8, 8); // MHA (gqa=1): identity expand
}

// ── GQA-DEDUP bridge: size the K/V cache + Kᵀ-restickify to nkvh DISTINCT heads (not nqh) ──
// Pre-dedup: cache [nqh,hd,cap]; the score matmul reads q-head qh at kernel-base qh*hd*cap, and the
// restickify transposes all nqh heads (24/32 of them byte-identical duplicates). Post-dedup: cache
// [nkvh,hd,cap] (4× smaller resident KV + 4× fewer whole-tensor restickifies), and q-head qh reads
// the SHARED kv-head at gqa_dedup_kv_kernel_base(qh,gqa,hd,cap)=gqa_kv_head(qh,gqa)*hd*cap. This proof
// pins that base: (a) it fits the nkvh cache — the naive pre-dedup qh*hd*cap is OOB for qh≥nkvh, so a
// forgotten remap FAILS here (fail-first); (b) it is the proven BLOCKED kv-head; (c) every cell
// (slot,d) via kcache_kt_write_offset stays inside the nkvh cache (no cross-head/-layer read).
fn gqa_dedup_kernel_base_ok_for(nqh: usize, nkvh: usize, hd: usize, cap: usize) {
    kani::assume(nkvh >= 1 && nqh >= nkvh && nqh % nkvh == 0 && hd >= 1 && cap >= 1);
    kani::assume(hd <= 128 && cap <= 256 && nqh <= 32); // CBMC bound (concrete in the entry points)
    let gqa = nqh / nkvh;
    let qh: usize = kani::any();
    kani::assume(qh < nqh);
    let base = gqa_dedup_kv_kernel_base(qh, gqa, hd, cap);
    // (a) IN-BOUNDS: the whole [hd,cap] kernel fits the nkvh-sized kct scratch. (qh*hd*cap FAILS for qh≥nkvh.)
    assert!(base + hd * cap <= nkvh * hd * cap);
    // (b) reads the BLOCKED kv-head (proven map) ⇒ q-head qh attends kv-head qh/gqa, not a scramble.
    assert!(base == gqa_kv_head(qh, gqa) * hd * cap);
    // (c) every transposed cell stays inside the nkvh kct scratch (never reads an adjacent head/layer).
    let slot: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(slot < cap && d < hd);
    assert!(base + kcache_kt_write_offset(slot, d, hd, cap, STK) < nkvh * hd * cap);
    // ── RESTICKIFY-ONLY dedup: the resident NATURAL K cache stays nqh-GQA-replicated (cachewr
    //    unchanged), but the restickify transposes only the nkvh DISTINCT heads by reading each
    //    group's REPRESENTATIVE head `kvh*gqa` (natural [cap,hd] at (kvh*gqa)*cap*hd) into kct[kvh].
    //    Prove the source-read chain: group-rep is in-bounds for the nqh cache AND holds kv-head kvh,
    //    so score head qh reading kct[qh/gqa] == the transpose of kv-head qh/gqa == attn_reference.
    let kvh: usize = kani::any();
    kani::assume(kvh < nkvh);
    let rep = kvh * gqa; // the representative q-head position for kv-group kvh
    assert!(rep < nqh); // in-bounds of the nqh-replicated natural cache
    assert!(rep * cap * hd + cap * hd <= nqh * cap * hd); // its whole [cap,hd] head fits
    assert!(gqa_kv_head(rep, gqa) == kvh); // the rep head holds EXACTLY kv-head kvh (round-trip)
    // score head qh reads kct[qh/gqa], written from rep head (qh/gqa)*gqa which holds kv-head qh/gqa:
    assert!(gqa_kv_head(gqa_kv_head(qh, gqa) * gqa, gqa) == gqa_kv_head(qh, gqa));
}
#[kani::proof]
fn gqa_dedup_kernel_base_ok() {
    gqa_dedup_kernel_base_ok_for(NQH_G, NKVH_G, 64, 256); // granite 32q/8kv (gqa=4), hd=64, cap=256
}
#[kani::proof]
fn gqa_dedup_kernel_base_ok_mha() {
    gqa_dedup_kernel_base_ok_for(8, 8, 64, 256); // MHA (gqa=1): base == qh*hd*cap, still in-bounds
}

// ── PREFILL mq>1 CAUSAL new-token block (task #33: batched prefill, replacing the mq>1 build-Err) ──
// A batched prefill of `mq` new tokens folds a `[mq,mq]` new-token self-attention block jointly with the
// `[mq,cap]` prefix score into ONE softmax per query row. Query row `row` must attend new-token `col` iff
// `col <= row` (itself + earlier tokens, causal). FAIL-FIRST: `col < row` drops the diagonal self-term
// (row can't see its own token) → (a) fails; `col >= row` lets a row see a LATER token (non-causal) → (c)
// fails. The attended count per row is exactly `row+1` (contiguous [0..=row]) — the causal reference.
fn prefill_causal_mask_partition_ok_for(mq: usize) {
    kani::assume(mq >= 1 && mq <= 64); // CBMC bound (mq_pad is 64-aligned; one 64-block surface)
    let row: usize = kani::any();
    kani::assume(row < mq);
    // (a) SELF: row attends its own new token (the diagonal). `col < row` would fail this.
    assert!(prefill_causal_col_valid(row, row));
    // (b) PAST: row attends every earlier new token.
    let c: usize = kani::any();
    kani::assume(c < row);
    assert!(prefill_causal_col_valid(c, row));
    // (c) FUTURE: row attends NO later new token (causality). `col >= row` would fail this.
    let f: usize = kani::any();
    kani::assume(f > row && f < mq);
    assert!(!prefill_causal_col_valid(f, row));
    // (d) COUNT: exactly row+1 new tokens valid for this row (contiguous [0..=row]).
    let mut cnt = 0usize;
    let mut j = 0usize;
    while j < mq {
        if prefill_causal_col_valid(j, row) {
            cnt += 1;
        }
        j += 1;
    }
    assert!(cnt == row + 1);
}
#[kani::proof]
#[kani::unwind(66)]
fn prefill_causal_mask_partition_ok() {
    prefill_causal_mask_partition_ok_for(64); // mq_pad = 64 (one stick-block of new tokens)
}

// ── PREFILL mq>1 HEAD-MAJOR SELECTOR (task #33): Sel_h one-hot column-extract identity ──
// The mq>1 attention needs head-major q/k/v; the only deployed way is per-head selector matmuls
// q_h = q @ Sel_h. Prove Sel_h's source-column map is a correct, in-bounds, block-contained column
// extract: output col j of head h reads input col h*hd+j. FAIL-FIRST: an interleaved map (j*nqh+h)
// or off-by-one breaks (b) block-containment / (c) the bijection onto head h's [h*hd,(h+1)*hd) block.
fn selector_extracts_head_column_for(nqh: usize, hd: usize) {
    kani::assume(nqh >= 1 && hd >= 1 && nqh <= 32 && hd <= 128); // CBMC bound (nqh*hd <= 4096)
    let h: usize = kani::any();
    let j: usize = kani::any();
    kani::assume(h < nqh && j < hd);
    let src = selector_head_src_col(h, hd, j);
    // (a) IN-BOUNDS of the row-major q[mq, nqh*hd] column axis.
    assert!(src < nqh * hd);
    // (b) BLOCK-CONTAINED: head h's outputs read EXACTLY input block [h*hd, (h+1)*hd) — not another head.
    assert!(src >= h * hd && src < (h + 1) * hd);
    // (c) BIJECTION within the block: distinct output cols j read distinct input cols (col-extract, no dup).
    let j2: usize = kani::any();
    kani::assume(j2 < hd && j2 != j);
    assert!(selector_head_src_col(h, hd, j2) != src);
    // (d) the local offset within head h's block is exactly j (identity within the head).
    assert!(src - h * hd == j);
}
#[kani::proof]
fn selector_extracts_head_column() {
    selector_extracts_head_column_for(NQH_G, 64); // granite 32 q-heads, hd=64
}

// ── LAST-ROW EXTRACTION (mq>1 prefill lm_head at m=1): the tail reads hidden[m-1,:], the last prompt
//    row, at row index `selector_lastrow_col(m)`.
//    FAIL-FIRST: a naive `m` (off-by-one) makes (a) OOB; a `0` (first row) fails (b)/(d). ──
fn selector_lastrow_picks_last_row_for(m: usize) {
    kani::assume(m >= 1 && m <= 256); // CBMC bound; prefill mq is small (path B)
    let col = selector_lastrow_col(m);
    // (a) IN-BOUNDS of the m-axis [0, m) of hidden[m, hidden].
    assert!(col < m);
    // (b) it is EXACTLY the last row index.
    assert!(col == m - 1);
    // (c) MAXIMAL: every valid row r < m satisfies r <= col (no later row exists) — the last-row pick.
    let r: usize = kani::any();
    kani::assume(r < m);
    assert!(r <= col);
    // (d) the extracted row equals the last prompt row: last_hidden[0,j] = hidden[col,j] = hidden[m-1,j].
    assert!(col == m - 1);
}
#[kani::proof]
fn selector_lastrow_picks_last_row() {
    selector_lastrow_picks_last_row_for(8); // path-B prefill mq (e.g. KTIR_PREFILL_LEN=8)
}

// ── HEAD-MAJOR ROW EXPANSION: a stick-blocked `[m, H·L]` and `[H·m, L]` place EVERY element at the
//    SAME byte, so the arrangement authority may treat them as one layout. This is what lets the RoPE
//    pointwise legs read a `[mq, heads·hd]` tensor as one `[heads·mq, hd]` op instead of `heads` ops.
//    FAIL-FIRST: a row-major (Flat) side, or `j = r·H + h` instead of `h·m + r`, breaks (a). ──
fn row_expansion_is_byte_identical_for(m: usize, heads: usize, lanes: usize) {
    kani::assume(m >= 1 && m <= 128 && heads >= 1 && heads <= 32 && lanes == 64);
    let wide = StickLayout::row_blocked(m, heads * lanes); // [m, H·L]  — the q/k projection output
    let tall = StickLayout::row_blocked(heads * m, lanes); // [H·m, L]  — its per-head-block view
    // (a) THE POINT: element (r, h·L+d) of the wide view is element (h·m+r, d) of the tall one.
    let h: usize = kani::any();
    let r: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(h < heads && r < m && d < lanes);
    assert!(wide.dev_off(r, h * lanes + d) == tall.dev_off(h * m + r, d));
    // (b) the authority ACCEPTS the pair (both directions), so one op may address either way.
    assert!(tall.is_row_expansion_of(&wide));
    assert!(wide.addr_eq(&tall) && tall.addr_eq(&wide));
    // (c) it stays TOTAL-preserving — no consumer can read bytes the producer never wrote.
    assert!(wide.rows * wide.cols == tall.rows * tall.cols);
    // (d) NOT a licence for the M>1 Dense-vs-stick scramble: a Flat side is genuinely row-major and
    //     must still conflict whenever there is real blocking to disagree about (m>1 AND cols>lanes).
    let flat = StickLayout::flat(m, heads * lanes);
    if m > 1 && heads > 1 {
        assert!(!wide.addr_eq(&flat));
    }
}
#[kani::proof]
fn row_expansion_is_byte_identical() {
    row_expansion_is_byte_identical_for(15, NQH_G, 64); // granite prefill rung 15, 32 q-heads
}

// ── PREFILL PAD-ROW POSITION (mq>1 prefill, chunk of `real` tokens on an `mq`-wide rung): the emitter
//    extracts row `mq-1`, a compile-time constant, but that row is PAD whenever real < mq. Clamping pad
//    rows to the last real token's position is what makes the extracted row the right one.
//    FAIL-FIRST: an unclamped `row` (today's rope staging) breaks (a); a clamp to `real` breaks (b). ──
fn prefill_pad_row_holds_last_real_token_for(mq: usize) {
    kani::assume(mq >= 1 && mq <= 256); // CBMC bound; the widest baked rung is 96
    // THE KIND, NAMED. The clamp below belongs to a prompt CHUNK and is reachable only through this
    // type — a decode batch's pad rows replicate live 0 instead, and cannot call it.
    let chunk = ChunkRows::chunk(NonZeroU32::new(mq as u32).expect("mq >= 1"));
    let real: usize = kani::any();
    kani::assume(real >= 1 && real <= mq);
    let last = selector_lastrow_col(mq); // the row the emitter baked its copy offsets against
    // (a) THE POINT: the extracted row carries the LAST REAL token's position, for any real <= mq.
    assert!(chunk.row_logical_pos(last, real) == real - 1);
    // (b) IN-BOUNDS: every row maps to a position of a REAL token, never past the prompt.
    let row: usize = kani::any();
    kani::assume(row < mq);
    assert!(chunk.row_logical_pos(row, real) < real);
    // (c) CAUSAL EXTENT never widens: a row attends no column beyond its own index, so no real row can
    //     see a pad row (the pre-fold inertness argument survives the clamp).
    assert!(chunk.row_logical_pos(row, real) <= row);
    // (d) REAL rows are UNTOUCHED — the clamp only ever moves pad rows.
    if row < real {
        assert!(chunk.row_logical_pos(row, real) == row);
    }
    // (e) IDENTITY at a full chunk (real == mq): binds byte-identical rotary/mask data to the pre-fold
    //     path, so a chunk that fills its rung cannot regress.
    if real == mq {
        assert!(chunk.row_logical_pos(row, real) == row);
    }
    // (f) the causal mask composes: the clamped position is exactly the last VALID column for that row.
    let col: usize = kani::any();
    kani::assume(col < mq);
    assert!(
        prefill_causal_col_valid(col, chunk.row_logical_pos(row, real))
            == (col <= chunk.row_logical_pos(row, real))
    );
}
#[kani::proof]
fn prefill_pad_row_holds_last_real_token() {
    prefill_pad_row_holds_last_real_token_for(23); // a real ladder rung (PREFILL_RUNGS)
}

// ── mq>1 PREFILL CACHEWR: persist the prompt's mq K/V into the resident NATURAL cache. ──
// The mq=1 decode cachewr writes ONE roped-K/V slot at `seq_pos` into the natural cache
// (kc[qh][slot][d], GQA-replicated). The mq>1 prefill must write mq slots [0..mq) — a
// CONTIGUOUS [mq,hd] copy from the HEAD-MAJOR kh[kvh]/vh[kvh] (row-stride hd, from the
// selector matmuls) → the natural cache kc[qh]/vc[qh] (row-stride hd). Element i = r·hd+d of
// the block maps src_base+i → dst_base+i. This proves the dst cell IS the natural per-slot
// cache cell `vcache_write_offset` (the SAME cell the decode restickify reads + the mq=1
// cachewr writes), so the two paths agree. FAIL-FIRST: a ROW-MAJOR new_k source (row-stride
// nkvh·hd, NOT hd) would map i=r·hd+d to the WRONG new_k cell — assertions (a)/(e) pin that the
// source must be head-major (row-stride hd), i.e. kh/vh, not new_k.
fn prefill_cachewr_natural_slots_for(nqh: usize, nkvh: usize, hd: usize, cap: usize, mq: usize) {
    // CBMC bounds: granite nqh=32, nkvh=8, hd=64; keep products small. mqp = mq padded to stick.
    kani::assume(hd == 64 && nkvh >= 1 && nqh == nkvh * (nqh / nkvh));
    kani::assume(nqh <= 32 && nkvh <= 8 && mq >= 1 && mq <= 64 && cap >= mq && cap <= 256);
    let gqa = nqh / nkvh;
    let mqp = mq.next_multiple_of(64); // stick-padded head-major row count (>= mq)
    let qh: usize = kani::any();
    let r: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(qh < nqh && r < mq && d < hd);
    let kvh = gqa_kv_head(qh, gqa);
    // Contiguous [mq,hd] copy element index (row-stride hd on BOTH sides).
    let i = r * hd + d;
    let dst = qh * cap * hd + i; // kc[qh] natural base + i
    let src = kvh * mqp * hd + i; // kh[kvh] head-major base + i
    // (a) dst is EXACTLY the natural per-slot cache cell for (qh, slot=r, d) — the SSOT
    //     `vcache_write_offset` the decode side uses (so prefill+decode address the SAME cell).
    assert!(dst == qh * cap * hd + vcache_write_offset(r, d, hd, cap, 64));
    // (b) dst in-bounds of the resident cache [nqh, cap, hd].
    assert!(dst < nqh * cap * hd);
    // (c) src in-bounds of head-major kh/vh [nkvh, mqp, hd].
    assert!(src < nkvh * mqp * hd);
    // (d) DISTINCT slots: a different prompt row r2 writes a DIFFERENT cache slot (no aliasing).
    let r2: usize = kani::any();
    kani::assume(r2 < mq && r2 != r);
    assert!(qh * cap * hd + r2 * hd + d != dst);
    // (e) the copy maps prompt row r → cache slot r (identity in the m-dim), on BOTH sides —
    //     TRUE only because both strides are hd (head-major); a row-major src would break this.
    assert!((dst - qh * cap * hd) / hd == r);
    assert!((src - kvh * mqp * hd) / hd == r);
}
#[kani::proof]
fn prefill_cachewr_natural_slots() {
    prefill_cachewr_natural_slots_for(32, 8, 64, 128, 8); // granite: 32 q / 8 kv heads, mq=8
}

// ── mq>1 RMSNORM REDUCE LAYOUT + PER-ROW amax (2026-07-07) ── The device address of a tensor cell is
// `dev_off(dims, stick_idx, idx)` (sdsc_abstract.rs, the SSOT the shim + emitter share). It is:
//   • STICK-MAJOR  `(j/64)*(a*64) + i*64 + (j%64)`  ONLY for a 2-D tensor sticked on dim 1 (e.g. the roped
//     Q/K `[mq, total]` — what the worker probe at :730 reads);
//   • FLAT ROW-MAJOR  `Σ idx[d]·Π dims[>d]`  for everything else — INCLUDING the rmsnorm intermediates,
//     which the pointwise/reduce opspecs declare as 3-D `[mb, out, y=1]` (pointwise_opspec :2682,
//     reduce_opspec_off :4860). So `|x|`/`x²` are ROW-MAJOR: logical `[r,c]` at `r*cols + c`, and row r is
//     a CONTIGUOUS run `[r*cols .. r*cols+cols)`.
// ⇒ the per-row `assemble_reduce_off(rows=1, data_off = r*cols)` reads EXACTLY row r (correct). The earlier
// belief that these were stick-major (⇒ a multi-row reduce needed) was WRONG — it conflated them with the
// 2-D-sticked roped Q/K. The valid amax MUST be per-row because reduce-MAX is multi-row-broken:
//   (F1) reduce-MAX returns the SEED 0 for rows>1 (CONFIRMED 2026-07-07: a single multi-row max ⇒ amax=0 ⇒
//        ramax=1/amax=inf ⇒ K-cache t9=inf under SCRATCHY_RMS_DIAG=ramax). rows=1 MAX is correct.
// This models dev_off and proves: (a) 3-D reduce tensors are row-major ⇒ per-row flat offset == row r;
// (b) 2-D-stick-1 tensors are NOT row-major (the roped-Q/K case, kept distinct). Helpers reused below.
const STK64: usize = 64;
fn dev_off_2d_stick1(rows: usize, cols: usize, r: usize, c: usize) -> usize {
    (c / STK64) * (rows * STK64) + r * STK64 + (c % STK64) // 2-D sticked-on-1 (roped Q/K) — NOT rmsnorm
}
fn dev_off_3d_rowmajor(cols: usize, r: usize, c: usize) -> usize {
    // 3-D [mb=rows, out=cols, y=1] ⇒ dev_off else-branch = flat row-major = r*cols + c (y index 0).
    r * cols + c
}
fn stickmajor_addr(r: usize, c: usize, rows: usize) -> usize {
    dev_off_2d_stick1(rows, 0, r, c) // alias used by the reduce-semantics proof below
}
fn flat_reduce_addr(r: usize, c: usize, cols: usize) -> usize {
    r * cols + c
}
#[kani::proof]
fn rmsnorm_reduce_rowmajor_amax_must_be_perrow() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows >= 1 && rows <= 8 && cols >= 64 && cols <= 256 && cols % 64 == 0);
    let r: usize = kani::any();
    let c: usize = kani::any();
    kani::assume(r < rows && c < cols);
    // (a) LAYOUT: the rmsnorm reduce tensors are 3-D row-major ⇒ the per-row rows=1 reduce at flat offset
    //     r*cols reads exactly logical [r,c] — for EVERY row, not just row 0. This is why per-row is correct.
    assert!(dev_off_3d_rowmajor(cols, r, c) == flat_reduce_addr(r, c, cols));
    assert!(flat_reduce_addr(r, c, cols) < rows * cols); // in-bounds
    // (b) row r is a CONTIGUOUS run: [r,0]..[r,cols-1] map to [r*cols .. r*cols+cols) with no gap/overlap
    //     into another row (so the rows=1 reduce over that run is precisely row r's reduction).
    assert!(dev_off_3d_rowmajor(cols, r, cols - 1) - dev_off_3d_rowmajor(cols, r, 0) == cols - 1);
    if r + 1 < rows {
        assert!(dev_off_3d_rowmajor(cols, r, cols - 1) < dev_off_3d_rowmajor(cols, r + 1, 0));
    }
    // (c) DISTINCTNESS from the 2-D-sticked layout: the roped-Q/K stick-major address is NOT row-major for
    //     stick≥1 (rows>1) — proving these are genuinely different layouts (don't reuse the wrong one).
    if rows > 1 {
        assert!(dev_off_2d_stick1(rows, cols, 0, 64) != dev_off_3d_rowmajor(cols, 0, 64));
    }
    // ⛔ NOTE: correct ADDRESSING (above) is NECESSARY but NOT SUFFICIENT — see the axiom below: reduce-MAX
    // STILL returns seed 0 on a multi-row TENSOR even with this correct per-row offset. So per-row MAX fails.
}

// ── THE mq>1 MATMUL-INPUT LAYOUT SEAM (2026-07-08, user: "IT SHOULD BE IMPOSSIBLE TO COMPILE MISALIGNED
//    SHAPES"). A matmul reads its activation INPUT as 2-D STICK-MAJOR (dev_off_2d_stick1) but the producing
//    pointwise/rmsnorm writes it DENSE row-major (dev_off_3d_rowmajor). The two layouts COINCIDE only at
//    R==1 (decode ⇒ finite/correct — WHY the untyped bug was invisible at M=1) and DIVERGE for R>1 (mq>1
//    prefill ⇒ the matmul reads a DIFFERENT physical element than the producer wrote ⇒ scrambled/uninit ⇒
//    inf). The `StickMajor` vs `Dense` TYPE distinction (lower_subtile) makes this a cargo-build error: the
//    matmul INPUT requires `StickMajor`, so a `Dense` producer must be explicitly restickified. This proof
//    encodes the invariant the type enforces. Fail-first: the seam demonstrably exists at R>1. ──
#[kani::proof]
fn matmul_input_layout_seam_stickmajor_vs_dense() {
    let cols: usize = kani::any();
    kani::assume(cols >= 64 && cols <= 256 && cols % 64 == 0);
    let c: usize = kani::any();
    kani::assume(c < cols);
    // DECODE (R==1): stick-major == Dense for EVERY column ⇒ a matmul reading a Dense-produced input is
    // byte-correct ⇒ decode finite. (stickmajor(0,c,1) = (c/64)*64 + 0 + c%64 = c = dense(0,c).)
    assert!(dev_off_2d_stick1(1, cols, 0, c) == dev_off_3d_rowmajor(cols, 0, c));
    // PREFILL (R>1): the layouts DIVERGE — witness r=0, c=64 (stick 1): Dense=64 but stick-major=R*64, so a
    // Dense-produced input fed to a stick-major matmul reads the WRONG cell for every R>1 (the mq>1 inf). A
    // StickMajor-typed matmul input makes this unconstructable without an explicit Dense→StickMajor restickify.
    let rows: usize = kani::any();
    kani::assume(rows > 1 && rows <= 64);
    assert!(dev_off_2d_stick1(rows, cols, 0, 64) != dev_off_3d_rowmajor(cols, 0, 64)); // rows*64 != 64
}

// ── CONFIRMED AXIOM (2026-07-07, RMS_DIAG=xs on the current per-row binary ⇒ t9 all inf ⇒ xs=x·ramax=inf ⇒
//    ramax=inf ⇒ amax=0 for EVERY row incl row 0): reduce-MAX returns the SEED 0 whenever the OPERAND TENSOR
//    has >1 row — REGARDLESS of the op's declared mb=1 or a per-row data_off. It keys off the tensor's
//    PLACEMENT row count, not the op's mb. So the per-row `assemble_reduce_off(rows=1, r*cols)` on the 8-row
//    t_absx (addressing proven correct above) STILL returns 0 ⇒ ramax=inf ⇒ xs=inf ⇒ the mq=8 " signature".
//    Only a GENUINELY 1-row tensor works for MAX (decode's [1,cols]). SUM/MEAN on a multi-row tensor DO work
//    (F2). ⇒ amax MUST avoid MAX entirely: use sum|x| via a MULTI-ROW SUM. This proof encodes the axiom. ──
fn oncard_max_reduce_result(tensor_rows: usize, true_max: u32) -> u32 {
    if tensor_rows == 1 {
        true_max // decode: genuinely 1-row tensor ⇒ MAX correct
    } else {
        0 // any multi-row TENSOR ⇒ seed 0 (mb=1 slice does NOT help)
    }
}
#[kani::proof]
fn rmsnorm_max_reduce_seeds_on_multirow_tensor() {
    let tensor_rows: usize = kani::any();
    kani::assume(tensor_rows >= 1 && tensor_rows <= 8);
    let true_max: u32 = kani::any();
    kani::assume(true_max >= 1 && true_max <= 60000);
    if tensor_rows == 1 {
        assert!(oncard_max_reduce_result(1, true_max) == true_max); // decode works
    }
    // FAIL-FIRST: the OLD per-row code ASSUMED mb=1 dodged F1. It does NOT — a multi-row tensor ⇒ MAX=0 ⇒
    // amax=0 ⇒ ramax=inf. (This is why the per-row row-major amax still infed despite correct addressing.)
    if tensor_rows > 1 {
        assert!(oncard_max_reduce_result(tensor_rows, true_max) == 0);
    }
}

// ── THE SUM-BASED amax FIX (proof): amax = sum|x| via MULTI-ROW SUM (F2, works on multi-row tensors),
//    pre-scaled by 1/cols so no fp16 overflow; ramax = recip(mean|x|)·(1/cols) = 1/sum|x| ⇒ |xs|=|x|/sum|x|
//    ≤ 1 (⇒ xs² ≤ 1, no overflow). amax cancels algebraically (out = xs·cur = x/√mean(x²), emitter:8264). ──
const F16MAX_U: u64 = 65504;
#[kani::proof]
fn rmsnorm_sumbased_amax_finite() {
    // (1) PRE-SCALED partial sum never overflows: each term is |x_i|/cols ≤ m/cols, and there are ≤ cols of
    //     them, so the running sum ≤ cols·(m/cols) = m ≤ F16MAX. Encoded as `k·m ≤ cols·m ⇔ k ≤ cols` (m>0);
    //     small ranges keep the symbolic multiply CBMC-tractable (the relation is scale-invariant in m/cols).
    let cols: u64 = kani::any();
    kani::assume(cols >= 1 && cols <= 128);
    let m: u64 = kani::any();
    kani::assume(m >= 1 && m <= 256); // representative max|x|; the ≤-relation is independent of the scale
    let k: u64 = kani::any();
    kani::assume(k >= 1 && k <= cols); // partial after k of cols terms
    assert!(k * m <= cols * m); // ⇒ pre-scaled partial ≤ m (the un-scaled sum would be ×cols larger)
    // (2) |xs| ≤ 1: sum|x| ≥ max|x| ≥ |x_c| ⇒ |x_c|/sum|x| ≤ 1 ⇒ xs² ≤ 1 (no overflow feeding mean). No mult.
    let mfull: u64 = kani::any();
    let xc: u64 = kani::any();
    let sum: u64 = kani::any();
    kani::assume(mfull <= F16MAX_U && xc <= mfull && sum >= mfull); // sum|x| ≥ max|x| = mfull ≥ |x_c| = xc
    assert!(xc <= sum);
    assert!(mfull <= F16MAX_U); // pre-scaled partial (≤ mfull) is fp16-finite for the FULL fp16 range
    // (3) FAIL-FIRST — WHY the 1/cols pre-scale is required: the naive un-scaled amax=sum|x| overflows fp16
    //     after just 2 large terms (~40000 each) ⇒ sum=inf ⇒ ramax=0 ⇒ collapse. Constant mult (cheap).
    let m2: u64 = kani::any();
    kani::assume(m2 >= 40000 && m2 <= F16MAX_U);
    assert!(2 * m2 > F16MAX_U);
}

// ── V-CACHE restickify OPSPEC bridge: natural `[cap,hd]` → hd-sticked `[cap,hd]` (NO transpose) ──
// The value bmm reads V as kernel `[k=cap, n=hd]` sticked on n=hd ⇒ `vcache_write_offset` =
// `dev_off([cap,hd],1,[slot,d])`. Unlike K (a transpose), V keeps its dim order and only re-sticks
// flat→hd. `restickify_v_opspec_2d` declares input `[cap,hd]` stick "cap" (dim0 ⇒ FLAT, reads the
// natural write) and output `[cap,hd]` stick "hd" (dim1/last ⇒ `[hd/64,cap,64]` = vcache_write_offset).
// Mirrors that opspec verbatim (stick_idx via position()) and proves in==natural, out==bmm read.
fn restickify_v_opspec_realizes_contract_for(hd: usize, cap: usize) {
    let slot: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(slot < cap && d < hd);
    let stk = STK;
    // INPUT: layout ["cap","hd"], stick "cap" ⇒ stick_idx 0 ⇒ FLAT slot*hd+d (the natural V write).
    let in_layout = ["cap", "hd"];
    let in_stick_idx = in_layout.iter().position(|&x| x == "cap").unwrap();
    let in_off = dev_off(&[cap, hd], in_stick_idx, &[slot, d]);
    assert!(in_off == slot * hd + d);
    assert!(in_stick_idx != in_layout.len() - 1); // NOT last ⇒ flat device_size
    // OUTPUT: layout ["cap","hd"], stick "hd" ⇒ stick_idx 1 (last) ⇒ [hd/64,cap,64] = vcache_write_offset.
    let out_layout = ["cap", "hd"];
    let out_stick_idx = out_layout.iter().position(|&x| x == "hd").unwrap();
    let out_off = dev_off(&[cap, hd], out_stick_idx, &[slot, d]);
    assert!(out_off == vcache_write_offset(slot, d, hd, cap, stk));
    assert!(out_off == dev_off(&[cap, hd], 1, &[slot, d])); // == the value bmm kernel read
    assert!(out_stick_idx == out_layout.len() - 1); // last ⇒ hd-sticked kernel tiling
}
#[kani::proof]
fn restickify_v_opspec_realizes_contract() {
    restickify_v_opspec_realizes_contract_for(HD, CAP);
}
#[kani::proof]
fn restickify_v_opspec_realizes_contract_hd128() {
    restickify_v_opspec_realizes_contract_for(HD_GRANITE, CAP);
}

// ── BRIDGE: LX-fit TIME-TILE trip coverage ── A wide op (matmul/pointwise over `out`) whose per-core
// resident set doesn't fit LX is split into `time` TRIPS over the `out` stick dim; `concrete_trips`
// emits trip `t` writing `[t·per_trip, (t+1)·per_trip)` sticks. `time_tile_for_lx` only ever picks a
// `time` that DIVIDES the per-core stick count (superdsc_opspec.rs:493 `stick_count % time == 0`), so
// the trips TILE the full `out` with NO dropped tail — else the last `stick_count % time` sticks would
// be silently unwritten (the same coverage-gap bug class as silu's dropped col offset). Proven: for
// every stick s∈[0,stick_count) exactly ONE trip covers it, and `time·per_trip == stick_count`.
// Fail-first: relaxing the `time | stick_count` invariant makes `time·per_trip < stick_count` (a
// dropped tail) — Kani finds the uncovered stick.
fn time_tile_trips_cover(divides_evenly: bool) {
    let stick_count: u32 = kani::any();
    kani::assume(stick_count >= 1 && stick_count <= 256); // cap/64 … intermediate/64 (200) etc.
    let time: u32 = kani::any();
    kani::assume(time >= 1 && time <= stick_count);
    if divides_evenly {
        kani::assume(stick_count % time == 0); // the invariant time_tile_for_lx enforces
    }
    let per_trip = stick_count / time; // sticks per trip (integer division)
    kani::assume(per_trip >= 1);
    let s: u32 = kani::any();
    kani::assume(s < stick_count); // any output stick
    let t = s / per_trip; // the trip that covers stick s
    assert!(t < time); // covered by a valid trip index
    assert!(s >= t * per_trip && s < (t + 1) * per_trip); // by exactly that trip (disjoint)
    assert!(time * per_trip == stick_count); // trips tile the full out — NO dropped tail
}
#[kani::proof]
fn time_tile_trips_cover_ok() {
    time_tile_trips_cover(true); // time | stick_count (the enforced invariant) → complete coverage
}

// ── `dev_off` for a `[hd, cap]` stick-last (the Kᵀ kernel) is INJECTIVE at EACH head_dim ──
// Distinct logical (d, slot) map to distinct device addresses ⇒ no aliasing / no element clobbers
// another. A layout that aliases is the "scrambled" bug class. Proven for ALL pairs in bounds.
fn dev_off_kt_injective_for(hd: usize, cap: usize) {
    let (d1, s1): (usize, usize) = (kani::any(), kani::any());
    let (d2, s2): (usize, usize) = (kani::any(), kani::any());
    kani::assume(d1 < hd && d2 < hd && s1 < cap && s2 < cap);
    kani::assume(!(d1 == d2 && s1 == s2)); // distinct logical indices
    let a1 = dev_off(&[hd, cap], 1, &[d1, s1]);
    let a2 = dev_off(&[hd, cap], 1, &[d2, s2]);
    assert!(a1 != a2);
}
#[kani::proof]
fn dev_off_kt_injective() {
    dev_off_kt_injective_for(HD, CAP);
}
#[kani::proof]
fn dev_off_kt_injective_hd128() {
    dev_off_kt_injective_for(HD_GRANITE, CAP);
}
#[kani::proof]
fn dev_off_kt_injective_hd512() {
    dev_off_kt_injective_for(HD_GEMMA4, CAP);
}

// ── `dev_off` for a `[hd, cap]` stick-last is BOUNDED at EACH head_dim ──
// Every in-bounds index maps within `hd*cap` (so a flat buffer of that size — DeviceIR's per-head
// footprint — holds every element with no out-of-range write). Requires cap % 64 == 0 (256 ✓).
fn dev_off_kt_in_footprint_for(hd: usize, cap: usize) {
    let d: usize = kani::any();
    let slot: usize = kani::any();
    kani::assume(d < hd);
    kani::assume(slot < cap);
    let off = dev_off(&[hd, cap], 1, &[d, slot]);
    assert!(off < hd * cap);
}
#[kani::proof]
fn dev_off_kt_in_footprint() {
    dev_off_kt_in_footprint_for(HD, CAP);
}
#[kani::proof]
fn dev_off_kt_in_footprint_hd128() {
    dev_off_kt_in_footprint_for(HD_GRANITE, CAP);
}
#[kani::proof]
fn dev_off_kt_in_footprint_hd512() {
    dev_off_kt_in_footprint_for(HD_GEMMA4, CAP);
}

// ── MATMUL KERNEL RETILE — producer(H2D stage) == consumer(matmul read) ── EVERY projection + the
//    lm_head reads a weight `[in,out]` sticked on `out`, retiled on H2D to device_size `[out/64,in,64]`.
//    PRODUCER (sdsc_stage_weight_tiled, via the RetileDescriptor device_size/stride_map): host element
//    (in=i, out=j) → device cell (t=j/64, in=i, s=j%64), whose row-major device-LINEAR offset in
//    `[out/64,in,64]` is `t*(in*64) + i*64 + s`. CONSUMER (matmul kernel read via per_core_addr →
//    dev_off): reads element (i,j) at `dev_off(&[in,out],1,&[i,j])`. If these disagree, every weight is
//    read from the wrong cell → deterministic wrong logits. This machine-checks dev_off's implementation
//    equals the retile linearization for ALL (i,j). Small shapes span the out-stick boundary (< / = / >
//    64) so the uniform formula covers granite (in=4096,out=49664) + gemma-4; concrete spot-checks pin
//    the real shapes.
fn retile_producer_equals_consumer_for(in_: usize, out: usize) {
    // Kani DISCOVERED (retile_pc_out_non64 @ out=130 FAILED): dev_off's [out/64,in,64] tiling requires
    // out to be a whole number of 64-sticks — else j/64 reaches a tile beyond out/64 and overflows the
    // footprint. That precondition is ENFORCED for every matmul kernel by StickExtent (out%64!=0 is a
    // build Err) + DeviceWidth (pads out to 64-aligned, e.g. lm_head 49159→49664). So the retile is only
    // ever applied to 64-aligned out; encode that precondition here.
    kani::assume(out % STK == 0 && out >= STK);
    let (i, j): (usize, usize) = (kani::any(), kani::any());
    kani::assume(i < in_ && j < out);
    let consumer = dev_off(&[in_, out], 1, &[i, j]); // matmul reads kernel elem (in=i,out=j), stick=out
    let (t, s) = (j / STK, j % STK); // producer places (i,j) into device cell (t, i, s)
    let producer = t * (in_ * STK) + i * STK + s; // its device-linear offset in [out/64, in, 64]
    assert!(producer == consumer); // FAILS if the retile write cell ≠ the matmul read cell
    assert!(consumer < in_ * out); // in the kernel footprint (no clobber of the next weight)
}
#[kani::proof]
fn retile_pc_out_1stick() {
    retile_producer_equals_consumer_for(3, 64); // out = 1 stick (out == STK)
}
#[kani::proof]
fn retile_pc_out_2stick() {
    retile_producer_equals_consumer_for(3, 128); // out = 2 sticks (out > STK — the multi-stick kernel)
}
#[kani::proof]
fn retile_pc_out_3stick() {
    retile_producer_equals_consumer_for(5, 192); // out = 3 sticks (64-aligned multi-stick)
}
// ── per_core_addr producer(rank-3 [mb,out,y] FLAT) vs consumer(rank-2 [rows,cols] TILED) ── the matmul
//    OUTPUT is addressed by per_core_addr via dev_off on the rank-3 host layout [mb,out,y] (dev_off FLAT
//    branch, len!=2); its CONSUMER (next pointwise/rmsnorm) reads the same tensor as rank-2 [rows,cols]
//    (dev_off TILED branch). For granite DECODE (mb=1,y=1) [1,·] flat == [1,·] tiled, so producer==
//    consumer — PROVEN here for every out col. For mb>1 (prefill/CB — guarded OFF for granite: the mq>1
//    attention path is a build Err) they DIVERGE, which is the reason prefill would need its own lockdown.
fn per_core_addr_r3_eq_r2_decode_for(out: usize) {
    kani::assume(out % STK == 0 && out >= STK);
    let j: usize = kani::any();
    kani::assume(j < out);
    // granite decode: mb=1, y=1. producer per_core_addr = dev_off on rank-3 [1,out,1] (FLAT).
    let producer = dev_off(&[1, out, 1], 1, &[0, j, 0]);
    // consumer = dev_off on rank-2 [1,out] (TILED, stick=out).
    let consumer = dev_off(&[1, out], 1, &[0, j]);
    assert!(producer == consumer); // mb=1 ⇒ flat == tiled for every out col
}
#[kani::proof]
fn per_core_addr_decode_hidden() {
    per_core_addr_r3_eq_r2_decode_for(4096); // hidden = 64 sticks
}
#[kani::proof]
fn per_core_addr_decode_lmhead() {
    per_core_addr_r3_eq_r2_decode_for(49664); // padded vocab = 776 sticks
}
#[kani::proof]
fn per_core_addr_mb_gt1_diverges() {
    // DOCUMENT the mb>1 (prefill) divergence: rank-3 [mb,out,1] flat != rank-2 [mb,out] tiled for out>64.
    // This is NOT the granite bug (granite forces mb=1) but pins WHY prefill needs its own lockdown.
    let (mb, out): (usize, usize) = (2, 128);
    let (r, j): (usize, usize) = (0, 64); // row 0, out-col 64 (2nd stick) — a divergent corner
    let flat = dev_off(&[mb, out, 1], 1, &[r, j, 0]); // producer rank-3 flat: r*out + j = 64
    let tiled = dev_off(&[mb, out], 1, &[r, j]); // consumer rank-2 tiled: (j/64)*mb*64 + r*64 + j%64 = 128
    assert!(flat != tiled); // 64 != 128 — producer/consumer DIVERGE for mb>1 (prefill needs its own lockdown)
}
#[kani::proof]
fn retile_pc_granite_lmhead_concrete() {
    // lm_head kernel [in=4096, out=49664]: spot-check the corner cells at the stick boundary (exhaustive
    // enumeration over 4096×49664 is CBMC-intractable; the uniform-formula proofs above cover the pattern).
    for &(i, j) in &[
        (0usize, 0usize),
        (0, 63),
        (0, 64),
        (0, 65),
        (1, 64),
        (4095, 49663),
        (2000, 12345),
    ] {
        let consumer = dev_off(&[4096, 49664], 1, &[i, j]);
        let producer = (j / STK) * (4096 * STK) + i * STK + (j % STK);
        assert!(producer == consumer);
    }
}

// ── GQA FOOTPRINT: the score reads kc per-head at base `h*hd*cap` for h in 0..nqh, a block of hd*cap
//    elements. The Trace found the cache is allocated GQA-REPLICATED to nqh heads (footprint
//    nqh*hd*cap; lower_subtile_tape_to_superdsc.rs:1470 `bytes = nqh*hd*cap*2`). This harness proves
//    EVERY head's read block lies within that ACTUAL footprint. `alloc_heads` is the # of heads the
//    cache is physically sized/written for. If the cache were compact-nkvh (alloc_heads=nkvh) while the
//    score still read to nqh, this proof FAILS for h>=nkvh — i.e. it would localize the GQA bug. It
//    PASSES with the actual alloc_heads=nqh, refuting that suspicion. ──
fn score_read_in_footprint_for(nqh: usize, hd: usize, cap: usize, alloc_heads: usize) {
    let h: usize = kani::any();
    kani::assume(h < nqh); // the score matmul loops h in 0..nqh (lower_...:4791)
    let base = h * hd * cap; // per-head read base (lower_...:4793 `kc @ h*hd*cap`)
    let footprint = alloc_heads * hd * cap; // ACTUAL allocation (lower_...:1470)
    assert!(base + hd * cap <= footprint); // whole [base, base+hd*cap) block is in-footprint
}
// granite: nqh=32, nkvh=8, gqa=4, hd=128, cap=256. Cache allocated to nqh heads (the Trace's finding).
const NQH_G: usize = 32;
const NKVH_G: usize = 8;
const GQA_G: usize = 4; // nqh/nkvh
#[kani::proof]
fn score_read_in_footprint_granite() {
    // ACTUAL allocation = nqh heads (GQA-replicated) ⇒ every h in 0..nqh is in-footprint ⇒ PASS.
    score_read_in_footprint_for(NQH_G, HD_GRANITE, CAP, NQH_G);
}

// ── GQA REPLICATE: the emitter/shim map query head qh∈[0,nqh) to its kv-head kvh = qh/gqa and read the
//    NEW token's K/V from the compact source `new_k[kvh*hd]` (source spans nkvh heads = nkvh*hd cols),
//    replicating into `krep[qh*hd]` (dest spans nqh heads). This proves (a) kvh is always a valid kv-head
//    (kvh < nkvh, source slice in-bounds), and (b) the dest base qh*hd is distinct per qh and its whole
//    [qh*hd, qh*hd+hd) slice is within the nqh*hd dest footprint. gqa=nqh/nkvh. ──
fn gqa_replicate_for(nqh: usize, nkvh: usize, gqa: usize, hd: usize) {
    let qh: usize = kani::any();
    kani::assume(qh < nqh);
    // The SHARED GQA head map the emitter's lower_attn_node ACTUALLY uses (no transcription).
    let kvh = scratchy_subtile::sdsc_abstract::gqa_kv_head(qh, gqa);
    // (a) source new_k[kvh*hd] slice within the compact nkvh-head source (nkvh*hd cols)
    assert!(kvh < nkvh);
    assert!(kvh * hd + hd <= nkvh * hd);
    // (b) dest krep[qh*hd] slice within the replicated nqh-head footprint
    assert!(qh * hd + hd <= nqh * hd);
}
#[kani::proof]
fn gqa_replicate_granite() {
    gqa_replicate_for(NQH_G, NKVH_G, GQA_G, HD_GRANITE);
}
// Distinct query heads map to DISTINCT dest bases in krep ⇒ no two heads' replicated K/V clobber.
#[kani::proof]
fn gqa_replicate_dest_distinct_granite() {
    let (q1, q2): (usize, usize) = (kani::any(), kani::any());
    kani::assume(q1 < NQH_G && q2 < NQH_G && q1 != q2);
    assert!(q1 * HD_GRANITE != q2 * HD_GRANITE); // distinct dest head base in krep[nqh*hd]
}

// ── granite-3.3-2b IS hd=64, NOT hd=128 ── the ACTUAL on-card coherence model (hidden=2048, 32 q-heads ⇒
// head_dim=64; 8 kv-heads ⇒ GQA=4). The `*_granite` harnesses above are keyed on HD_GRANITE=128, which is
// micro-g3.3-8b's head_dim — so the GQA-replicate + score-footprint addressing of the REAL model was NOT
// proven at hd=64. Re-run the SAME parameterized bodies (which use the real `gqa_kv_head`/`dev_off`) at hd=64
// to lock down the on-card granite-3.3-2b attention addressing. (intermediate_perhead is already proven at
// hd=64; these close the GQA/score gap for the same model.) Fail-first: an hd-alignment assumption that only
// held at hd=128 would break here.
#[kani::proof]
fn gqa_replicate_granite2b_hd64() {
    gqa_replicate_for(NQH_G, NKVH_G, GQA_G, HD); // HD=64 (granite-3.3-2b head_dim)
}
#[kani::proof]
fn gqa_replicate_dest_distinct_granite2b_hd64() {
    let (q1, q2): (usize, usize) = (kani::any(), kani::any());
    kani::assume(q1 < NQH_G && q2 < NQH_G && q1 != q2);
    assert!(q1 * HD != q2 * HD); // distinct dest head base at hd=64 ⇒ no cross-head clobber
}
#[kani::proof]
fn score_read_in_footprint_granite2b_hd64() {
    score_read_in_footprint_for(NQH_G, HD, CAP, NQH_G); // hd=64: every q-head's score read in-footprint
}

// ── ⛔ BUG LOCALIZER (EXPECTED RED until the emitter fix lands) ──
// The K-cache Kᵀ + GQA-footprint harnesses above ALL pass at hd∈{64,128,512}, so the on-card wrong-token
// divergence was NOT in the prefix-cache addressing. It was in the [nqh,hd]-shaped attention intermediates
// `qs`/`krep`/`vrep`/`opre` (lower_subtile_tape_to_superdsc.rs): the emitter USED to WRITE some as a FULL
// `[nqh,hd]` tensor (device-sticked on hd → `[hd/64,nqh,64]`) while READING them PER-HEAD at flat offset
// `h*hd` as a `[1,hd]` view. Those two address models agree ONLY when hd<=STK=64, so granite hd=128 (and
// gemma-4 hd=512) read head h's upper-half dims (d>=64) from the WRONG device cells → scrambled attention
// → the degenerate token. (A red localizer asserting full-write==per-head-read FAILED at hd=128 and
// pinned the bug here.) THE FIX (emitter): every access to these intermediates is now PER-HEAD `[1,hd]` at
// base `h*hd` (write AND read), exactly like sp/exp_p/pp. This harness machine-checks that the per-head
// `[1,hd]` layout is CONSISTENT for ALL hd: a `[1,hd]` tensor is FLAT (dev_off==d, rows=1 ⇒ the stick
// term vanishes), so the per-head offset `h*hd+d` is row-major, INJECTIVE over (h,d), and IN-FOOTPRINT
// (< nqh*hd). Green at hd∈{64,128,512} ⇒ no real head_dim can re-introduce the multi-stick scramble.
fn intermediate_perhead_layout_consistent_for(nqh: usize, hd: usize) {
    let (h, d): (usize, usize) = (kani::any(), kani::any());
    kani::assume(h < nqh && d < hd);
    // (a) a [1,hd] tensor is FLAT for ANY hd (rows=1 ⇒ the stick-group term vanishes) — the property
    //     that makes per-head access correct where the full [nqh,hd] sticked layout was NOT.
    assert!(dev_off(&[1, hd], 1, &[0, d]) == d);
    let off = h * hd + dev_off(&[1, hd], 1, &[0, d]); // the emitter's per-head access (write == read)
    assert!(off == h * hd + d); // row-major
    assert!(off < nqh * hd); // (c) in the [nqh,hd] footprint — no clobber of a neighbor tensor
    // (b) INJECTIVE: distinct (h,d) → distinct device cell ⇒ no aliasing across heads/dims for any hd.
    let (h2, d2): (usize, usize) = (kani::any(), kani::any());
    kani::assume(h2 < nqh && d2 < hd && !(h == h2 && d == d2));
    assert!(off != h2 * hd + d2);
}
#[kani::proof]
fn intermediate_perhead_layout_hd64() {
    intermediate_perhead_layout_consistent_for(NQH_G, HD); // hd=64 (=STK)
}
#[kani::proof]
fn intermediate_perhead_layout_hd128_granite() {
    intermediate_perhead_layout_consistent_for(NQH_G, HD_GRANITE); // granite hd=128 (2 sticks)
}
#[kani::proof]
fn intermediate_perhead_layout_hd512_gemma4() {
    intermediate_perhead_layout_consistent_for(NQH_G, HD_GEMMA4); // gemma-4 hd=512 (8 sticks)
}

// ── RoPE rotate output layout (lower_rope_node) ── the roped Q/K are the model tensors [1, heads·hd]
// (row-major flat); the attention + KV-cache read them PER-HEAD at h·hd. So the RoPE rotate must write
// rot/xc/rs/out PER-HEAD [1,hd] at h·hd (FLAT) — a full [heads,hd] op device-sticks them for hd>64,
// mismatching the flat consumer (the same multi-stick class as the attention intermediates). RoPE runs
// over Q (heads=nqh=32) AND K (heads=nkvh=8), so prove the per-head-flat invariant for BOTH head counts
// at the real head_dims. Green here ⇒ the per-head RoPE lowering is layout-consistent for granite+gemma-4.
#[kani::proof]
fn rope_perhead_layout_hd64() {
    intermediate_perhead_layout_consistent_for(NQH_G, HD); // hd=64 (=STK) baseline — the models that already worked
}
#[kani::proof]
fn rope_perhead_layout_q_hd128_granite() {
    intermediate_perhead_layout_consistent_for(NQH_G, HD_GRANITE); // Q: nqh=32 heads, hd=128
}
#[kani::proof]
fn rope_perhead_layout_k_hd128_granite() {
    intermediate_perhead_layout_consistent_for(NKVH_G, HD_GRANITE); // K: nkvh=8 heads, hd=128
}
#[kani::proof]
fn rope_perhead_layout_q_hd512_gemma4() {
    intermediate_perhead_layout_consistent_for(NQH_G, HD_GEMMA4); // gemma-4 Q: hd=512 (8 sticks)
}

// ── BRIDGE: RoPE PREFILL ROW COVERAGE + PRODUCER==CONSUMER (lower_rope_node mq>1) ── THE fix for the
// measured 8->1 prefill collapse (roped Q/K came back with only row 0 non-zero -> K-cache 1 slot ->
// garbage on-card). The mq>1 roped Q/K live in a [mq, total] tensor (total=heads*hd) STICK-SCATTERED
// on-device (dev_off: [r,c]->(c/64)*(mq*64)+r*64+c%64 — rows interleaved within each 64-stick, NOT
// row-major contiguous). The rope emits one mb=1 [1,hd] block per (r,h) at
// rope_prefill_block_offset(r,h,mq,hd)=h*mq*hd+r*hd. Proven properties:
//   (PRODUCER==CONSUMER, MISSED by the first flat-offset fix) off+d == dev_off([mq,total],1,[r,h*hd+d])
//     for hd==STK — the rope WRITES exactly where the attention matmul (reading q as [mq,total]) READS
//     row r. The naive flat r*total+h*hd fails THIS (writes where r>0 is never read -> rows empty).
//   (BIJECTION) (r,h,d)->off+d is injective + surjective onto [0,mq*total) — all mq rows covered once.
//   (DECODE) mq=1 => off=h*hd, cos@0 — byte-identical to the old per-head loop.
// hd==STK only (granite hd=64; hd>64 prefill is an emitter build error). Symbolic (r,h,d,t); fast.
fn rope_prefill_bijection_for(mq: usize, heads: usize, hd: usize) {
    let total = heads * hd;
    let (r, h, d): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
    kani::assume(r < mq && h < heads && d < hd);
    let off = rope_prefill_block_offset(r, h, mq, hd) + d;
    // (PRODUCER==CONSUMER, the property the flat-offset first fix MISSED) the rope's device write cell
    // IS the consumer's [mq,total] read cell. For hd==STK the head-block's hd dims are contiguous, so
    // off+d addresses dev_off([mq,total],1,[r, h*hd+d]) exactly. The naive flat r*total+h*hd fails this.
    assert!(hd == 64); // this island covers granite hd=STK (hd>64 prefill is guarded in the emitter)
    assert!(off == dev_off(&[mq, total], 1, &[r, h * hd + d]));
    assert!(off < mq * total); // (in-bounds) no clobber past the [mq,total] tensor
    if mq == 1 {
        assert!(off == h * hd + d); // (decode degeneration) == old per-head formula, byte-identical
        assert!(rope_prefill_cos_offset(r, hd) == 0); // decode reads cos/sin at 0
    }
    // (INJECTIVE) distinct (r,h,d) -> distinct cell ⇒ each mq row occupies a disjoint region.
    let (r2, h2, d2): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
    kani::assume(r2 < mq && h2 < heads && d2 < hd && !(r == r2 && h == h2 && d == d2));
    assert!(off != rope_prefill_block_offset(r2, h2, mq, hd) + d2);
    // (SURJECTIVE) every output cell t is written by exactly the (r,h,d) that decompose its dev_off ⇒
    // NO cell (i.e. NO row 1..mq) is left unwritten — the property the 8->1 collapse violates. Invert
    // dev_off for hd==STK: head=t/(mq*64), row=(t/64)%mq, dim=t%64.
    let t: usize = kani::any();
    kani::assume(t < mq * total);
    let (hh, rr, dd) = (t / (mq * 64), (t / 64) % mq, t % 64);
    assert!(rope_prefill_block_offset(rr, hh, mq, hd) + dd == t);
    // cos/sin row-r slice stays in the [mq,total] cos table footprint.
    assert!(rope_prefill_cos_offset(r, hd) + d < mq * total);
}
#[kani::proof]
fn rope_prefill_bijection_decode() {
    rope_prefill_bijection_for(1, NQH_G, HD); // mq=1 decode — byte-identical to the old per-head loop
}
#[kani::proof]
fn rope_prefill_bijection_q_mq8() {
    rope_prefill_bijection_for(8, NQH_G, HD); // granite Q prefill: mq=8, nqh=32, hd=64
}
#[kani::proof]
fn rope_prefill_bijection_k_mq8() {
    rope_prefill_bijection_for(8, NKVH_G, HD); // granite K prefill: mq=8, nkvh=8, hd=64
}

// ── 32-core WORK-DIVISION (TiledIR): the TYPE-SAFE replacement for the monolith's runtime guard #50 ──

/// Two DISTINCT cores of a `CoreSplit::plan` own NON-OVERLAPPING output regions — the disjoint-partition
/// property (guard #50), PROVEN exhaustively over ALL core pairs for concrete granite shapes. Using
/// CONCRETE shapes keeps `core_split`'s divisor loop CONCRETE (each closes in ~2s). An ALL-SYMBOLIC
/// `core_split` divisor lemma is intentionally ABSENT: its data-dependent `size % i` loop is symbolic
/// ÷/%, which CBMC cannot discharge inside the 5-10s close-time bar (removed for exactly that reason).
/// That is NOT a gap — an even divisor is REQUIRED for the tiles to cover `[0,R)×[0,C)` disjointly, so a
/// non-divisor split would FAIL this very assertion; these per-shape proofs ARE core_split's divisor
/// proof. One tiny island per shape (each ~2s) beats one monolithic all-sizes proof; add a shape → add a
/// harness below.
fn regions_disjoint_for(r: u32, c: u32) {
    let sp = CoreSplit::plan_capped(r, c, MAX_CORES);
    let (cr1, cs1): (u32, u32) = (kani::any(), kani::any());
    let (cr2, cs2): (u32, u32) = (kani::any(), kani::any());
    kani::assume(
        cr1 < sp.row_cores && cr2 < sp.row_cores && cs1 < sp.stick_cores && cs2 < sp.stick_cores,
    );
    kani::assume(!(cr1 == cr2 && cs1 == cs2)); // distinct cores
    let (r0a, r1a, c0a, c1a) = sp.region(cr1, cs1, r, c);
    let (r0b, r1b, c0b, c1b) = sp.region(cr2, cs2, r, c);
    // Half-open [r0,r1)×[c0,c1) rectangles overlap iff they overlap on BOTH axes.
    let row_overlap = r0a < r1b && r0b < r1a;
    let col_overlap = c0a < c1b && c0b < c1a;
    assert!(!(row_overlap && col_overlap));
}
#[kani::proof]
fn plan_regions_disjoint_granite_rope() {
    regions_disjoint_for(256, 128); // granite RoPE rotate output (head_dim 128 = 2 sticks) — the #50 case
}
#[kani::proof]
fn plan_regions_disjoint_lmhead() {
    regions_disjoint_for(1, 49159); // decode lm_head: m=1, vocab=49159 (odd) — the real terminal matmul
}
#[kani::proof]
fn plan_regions_disjoint_hidden() {
    regions_disjoint_for(1, 4096); // decode projection: m=1, hidden=4096
}
#[kani::proof]
fn plan_regions_disjoint_non64() {
    regions_disjoint_for(17, 200); // rows∤cores + cols∤64 edge
}

/// FAIL-FIRST (2026-07-08): the batch=1 matmul MUST split mb (M) across cores when M>1. Keeping mb WHOLE
/// (splitting only `out`) was the mq=8 prefill INF root cause — a false "the PT streams M≤8 on one core"
/// assumption; pointwise ops split mb (finite on-card) but matmuls did not (inf, verified). This checks
/// `matmul_split_map`'s batch=1 rule (`mb_split = core_split(mb, MAX_CORES)` FIRST — the SAME `core_split`
/// brain CoreSplit/`distribute_cores` use). Proves per shape: mb split into EXACTLY M single-row cores
/// (`core_split(mb, cap) == mb` for mb ≤ cap ⇒ exact covering, no dropped rows) AND mb_split > 1 when M>1
/// [FAILS on the old out-only/mb-whole code — the fail-first witness]. CONCRETE mb per the established
/// `plan_regions_disjoint_*` style (a symbolic size unbounds `core_split`'s loop → CBMC-hostile; noted in
/// CoreSplit::plan). Disjointness of the per-core output tiles = the build-time `out_addr_seen` guard +
/// the CoreSplit `plan_regions_disjoint_*` proofs above; the out-stick split's covering = `core_split`'s
/// DIVISOR contract.
fn matmul_batch1_mb_split_for(mb: u32) {
    let mb_split = core_split(mb, MAX_CORES);
    assert!(mb_split == mb); // mb ≤ MAX_CORES ⇒ exact 1-row-per-core covering
    if mb > 1 {
        assert!(mb_split > 1); // THE FIX: mb split when M>1 (fails on the old mb-whole code)
    }
}
#[kani::proof]
fn matmul_batch1_splits_mb_m1() {
    matmul_batch1_mb_split_for(1); // decode / M=1 — mb NOT split (byte-identical to the proven path)
}
#[kani::proof]
fn matmul_batch1_splits_mb_m8() {
    matmul_batch1_mb_split_for(8); // granite prefill KTIR_PREFILL_LEN=8 — the fixed case
}
#[kani::proof]
fn matmul_batch1_splits_mb_m32() {
    matmul_batch1_mb_split_for(32); // MAX_CORES prefill chunk (upper edge)
}

/// Natural (V) layout `[CAP, HD]` stick-last is ALSO injective + bounded (V is read natural by the value
/// matmul; this proves that path can't alias either).
#[kani::proof]
fn dev_off_natural_injective() {
    let (s1, d1): (usize, usize) = (kani::any(), kani::any());
    let (s2, d2): (usize, usize) = (kani::any(), kani::any());
    kani::assume(s1 < CAP && s2 < CAP && d1 < HD && d2 < HD);
    kani::assume(!(s1 == s2 && d1 == d2));
    let a1 = dev_off(&[CAP, HD], 1, &[s1, d1]);
    let a2 = dev_off(&[CAP, HD], 1, &[s2, d2]);
    assert!(a1 != a2);
}

// ── GENERAL layout properties (hold for EVERY stick-last 2D tensor in the ladder, not one shape) ──

fn pad64(c: usize) -> usize {
    ((c + 63) / 64) * 64
}

/// GENERAL stick-last injectivity: for ANY 2D shape `[r, c]` (bounded), distinct logical `(i, j)` map to
/// distinct `dev_off`. ⇒ NO tensor in the ladder ever aliases itself, regardless of shape. Bounds kept
/// small enough for CBMC to discharge but spanning the stick boundary (c can be < / = / > 64).
#[kani::proof]
fn dev_off_general_injective() {
    let (r, c): (usize, usize) = (kani::any(), kani::any());
    kani::assume(r >= 1 && r <= 6 && c >= 1 && c <= 130);
    let (i1, j1): (usize, usize) = (kani::any(), kani::any());
    let (i2, j2): (usize, usize) = (kani::any(), kani::any());
    kani::assume(i1 < r && i2 < r && j1 < c && j2 < c);
    kani::assume(!(i1 == i2 && j1 == j2));
    let a1 = dev_off(&[r, c], 1, &[i1, j1]);
    let a2 = dev_off(&[r, c], 1, &[i2, j2]);
    assert!(a1 != a2);
}

/// GENERAL footprint bound: `dev_off([r,c],1,[i,j]) < r * pad64(c)` for every in-bounds `(i,j)`. This is
/// EXACTLY the size DeviceIR's pass allocates per tensor (`rows * pad64(cols)`), so the flat `Vec<f32>`
/// holds every element and — with sequential bases — tensors never clobber each other (inter-tensor
/// non-overlap by construction). Proves the HashMap→Vec refactor's addressing is sound for all shapes.
#[kani::proof]
fn dev_off_general_in_footprint() {
    let (r, c): (usize, usize) = (kani::any(), kani::any());
    kani::assume(r >= 1 && r <= 6 && c >= 1 && c <= 130);
    let (i, j): (usize, usize) = (kani::any(), kani::any());
    kani::assume(i < r && j < c);
    let off = dev_off(&[r, c], 1, &[i, j]);
    assert!(off < r * pad64(c));
}

// ── BRIDGE (per-node lowering: the BROADCAST pointwise operand — ScalarMul / RmsNormApply / attn-scale) ──
// A pointwise op with a broadcast operand (`EwOperand.out_broadcast=true` ⇒ `Scale::RedStick`, lower_...:2289)
// must read that operand at a FIXED device offset for EVERY output column, while the Active operand advances
// per column. This is what makes `out[j] = active[j] · scalar` correct (the [1,1] const/inv broadcast in
// ScalarMul, the rms `inv` in RmsNormApply, the attn-scale mul at lower_attn_node:4894). Modeled via the
// ACTUAL dev_off: the broadcast operand is a [1,1] scalar (dev_off ≡ 0, independent of the output col); the
// Active operand is [1,C] (dev_off = j). Prove: for ANY two distinct output cols the broadcast read offset is
// IDENTICAL (held, not strided) while the Active reads its OWN distinct column. Fail-first: had the emitter
// strided the broadcast operand (modeled it [1,C]), the "broadcast offset equal" assert would fail — the
// garbage-scale-per-column bug this locks out.
#[kani::proof]
fn broadcast_operand_holds_scalar_active_advances() {
    let c: usize = kani::any();
    kani::assume(c >= 1 && c <= 130); // up to >2 sticks (granite hidden slice / cap width)
    let (j1, j2): (usize, usize) = (kani::any(), kani::any());
    kani::assume(j1 < c && j2 < c && j1 != j2);
    // Broadcast operand = a [1,1] scalar: dev_off is 0 for EVERY output column (no per-col stride).
    let b1 = dev_off(&[1, 1], 1, &[0, 0]);
    let b2 = dev_off(&[1, 1], 1, &[0, 0]);
    assert!(b1 == 0 && b2 == 0 && b1 == b2); // scalar HELD at offset 0 for all j (broadcast, not strided)
    // Active operand = a [1,C] row: dev_off advances per column, distinct for distinct cols.
    let a1 = dev_off(&[1, c], 1, &[0, j1]);
    let a2 = dev_off(&[1, c], 1, &[0, j2]);
    assert!(a1 == j1 && a2 == j2); // active reads its OWN column
    assert!(a1 != a2); // distinct columns ⇒ distinct active elements (no scalar-collapse of the active input)
}

// ── BRIDGE (per-node lowering: the REDUCE input coverage — RmsNormReduce / SumReduce / softmax denominator) ──
// The rmsnorm variance reduce (`reduce_opspec`, scales `[Active, RedStick, Active]`) sums its [1,C] input over
// ALL C columns (the RedStick reduced dim). If the reduce iteration missed columns (partial coverage), the
// mean-of-squares / softmax denominator would be WRONG — a COVERAGE bug, DISTINCT from the f16-accumulation
// PRECISION leaf. Prove the reduce reads every input column EXACTLY once: for a [1,C] input the per-col read
// offset = dev_off([1,C],1,[0,i]) = i, so the read set is a bijection with {0..C-1} (no missed col, no
// double-count). This ISOLATES the pos-65 divergence to the accumulation ARITHMETIC (f16, the proven leaf) —
// the reduce is NOT truncating its input. Fail-first: a reduced extent < C (or a strided read) breaks the
// col↔offset identity.
#[kani::proof]
fn reduce_input_covers_all_cols() {
    let c: usize = kani::any();
    kani::assume(c >= 1 && c <= 130); // spans >2 sticks (a granite hidden slice)
    let i: usize = kani::any();
    kani::assume(i < c);
    let off = dev_off(&[1, c], 1, &[0, i]);
    assert!(off == i); // (a) flat single-row layout: reduced col i read at offset i
    assert!(off < c); //  (b) in-bounds of the [1,C] reduce input footprint
    // (c) injective over cols ⇒ the read set covers [0,C) exactly (distinct cols → distinct offsets):
    let i2: usize = kani::any();
    kani::assume(i2 < c && i2 != i);
    assert!(dev_off(&[1, c], 1, &[0, i2]) != off);
}

// ── FIX-GATE (proven BEFORE the step-2 byte-math wiring): an `_fp32` tensor's BYTE offset = elem·4, NOT
// elem·2 ── the emitter's byte-math sites (per_core_addr `addrs.push(off_elems·Fp16::WORD_LENGTH)`,
// stride_bytes, synth_footprint) hardcode `Fp16::WORD_LENGTH = 2`. For an `_fp32` merge-partial tensor
// (IEEE_FP32, 4 B/elem) that is WRONG (half the stride) — the fix routes each through `word_length_for(name)`
// (returns Fp32::WORD_LENGTH = 4 for `_fp32`, else 2, inert for fp16). Prove the fp32 byte offset = dev_off·4
// and stays within the fp32 footprint r·pad64(c)·4. Gates the per_core_addr / stride wiring: the current ·2
// hardcode gives elem·2 for an fp32 tensor ⇒ this ·4 claim localizes exactly those sites, and confirms the
// fix (word_length_for) restores the correct 4-byte fp32 addressing. (`Fp32::WORD_LENGTH == 4` itself is
// proven by `fp32_dataformat_stick_128byte_consistent`.)
#[kani::proof]
fn fp32_tensor_byte_offset_is_4x_and_in_footprint() {
    let (r, c): (usize, usize) = (kani::any(), kani::any());
    kani::assume(r >= 1 && r <= 6 && c >= 1 && c <= 130);
    let (i, j): (usize, usize) = (kani::any(), kani::any());
    kani::assume(i < r && j < c);
    let elem = dev_off(&[r, c], 1, &[i, j]); // element offset (dtype-agnostic, proven injective+in-footprint)
    let byte_off = elem * (Fp32::WORD_LENGTH as usize); // what word_length_for("…_fp32") yields
    assert!(byte_off == elem * 4); // fp32 = 4-byte (NOT the fp16 ·2 hardcode) — the wiring the fix installs
    assert!(byte_off < r * pad64(c) * 4); // within the fp32 footprint (4· the fp16 footprint) — no clobber
}

// ── FIX-GATE (fp32-SFP-merge emit, proven BEFORE the code): the mixed-dtype [fp16, fp32] pointwise `add`
// addresses the SAME logical element at DIFFERENT byte strides ── The fp32 merge reuses DeepTools
// `broadcast_ops.ddl`'s `add` op, VERIFIED on-pod to bind `[%type_fp16, %type_fp32]`
// (`%add_op = operation_bind([type_fp16, type_fp32], [inp1, inp2], [out]) {opFuncName="add"}`) — an f16 block
// partial + an fp32 accumulator → fp32 output, NO new op template. The emitter must address the SAME logical
// element j at the fp16 stride (2 B) for the f16 operand AND the fp32 stride (4 B) for the fp32
// accumulator/output — both derived from ONE dev_off element index. Prove it (each byte offset = elem·its
// word_length, within its dtype footprint). Fail-first: a single shared stride would misplace one operand.
#[kani::proof]
fn mixed_dtype_add_fp16_fp32_byte_offsets() {
    let c: usize = kani::any();
    kani::assume(c >= 1 && c <= 130);
    let j: usize = kani::any();
    kani::assume(j < c);
    let elem = dev_off(&[1, c], 1, &[0, j]); // one logical element index (flat [1,C] ⇒ == j)
    assert!(elem == j);
    let fp16_byte = elem * (Fp16::WORD_LENGTH as usize); // f16 operand: 2·j (word_length_for non-_fp32)
    let fp32_byte = elem * (Fp32::WORD_LENGTH as usize); // fp32 accumulator/output: 4·j (word_length_for _fp32)
    assert!(fp16_byte == 2 * j && fp32_byte == 4 * j); // SAME element, dtype-specific byte strides
    assert!(fp16_byte < c * 2 && fp32_byte < c * 4); // each within its own dtype footprint (no clobber)
}

// ── FIX-GATE (gates the `stride_bytes` affine-stride wiring): the fp32 per-trip stride == the trip's real
// byte span ⇒ time-trips tile DISJOINTLY; the fp16 `·2` hardcode on fp32 data OVERLAPS them ── a time-tiled
// output advances `stride_bytes = out_per_time·device_stride_out·word_length` per trip. The trip's ACTUAL
// written span = elems·(real bytes/elem). For an `_fp32` tensor (4 B/elem) the stride MUST use word_length 4
// to equal the span (adjacent, non-overlapping trips); the old `Fp16::WORD_LENGTH=2` hardcode makes
// stride = span/2 < span ⇒ consecutive trips OVERLAP by half ⇒ scrambled merge partials. This is a
// DISJOINTNESS claim (distinct from the element-offset ·4 proof). Fail-first: the `·2` branch violates it.
#[kani::proof]
fn fp32_affine_stride_matches_span_no_overlap() {
    let elems: i64 = kani::any(); // out_per_time · device_stride_out (element count advanced per trip)
    kani::assume(elems >= 1 && elems <= 1_000_000);
    let real_span_bytes = elems * 4; // fp32 data occupies 4 B/elem on device
    let fp32_stride = elems * Fp32::WORD_LENGTH as i64; // word_length_for("…_fp32") = 4 (the fix)
    let hardcode_stride = elems * Fp16::WORD_LENGTH as i64; // the old ·2 hardcode
    assert!(fp32_stride == real_span_bytes); // ·4 stride == actual fp32 span ⇒ trips tile DISJOINTLY
    assert!(hardcode_stride < real_span_bytes); // ·2 on fp32 data ⇒ stride < span ⇒ trips OVERLAP (the bug)
}

// ── FIX-GATE (gates the fp32 `arg_bytes` wiring + locks the single-row constraint): a [1,N] fp32 tensor's
// byte offset (dev_off·4) stays within its fp32 32-elem-stick footprint ── `arg_bytes` now uses the fp32
// 32-elem stick (`stick_elems_for`), but `dev_off` (the element offset in per_core_addr) uses the fp16
// 64-elem stick model. These agree ONLY for a SINGLE ROW (the stick-group term vanishes — proven by
// `fp32_partial_addressing_4byte_single_row`), which the merge partials are (decode m=1 ⇒ [1,N]). Prove the
// [1,N] fp32 byte offset < the fp32 footprint pad32(N)·4 (no element clobbers past its tensor), and that the
// fp32 32-stick is 128-byte-aligned (matching fp16's 64·2). Locks WHY fp32 partials MUST be single-row: a
// multi-stick fp32 tensor would mismatch the 64-stick dev_off against the 32-stick footprint.
#[kani::proof]
fn fp32_single_row_argbytes_holds_dev_off() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 4096);
    let c: usize = kani::any();
    kani::assume(c < n); // a column of the single-row [1,N] fp32 partial
    let byte_off = c * (Fp32::WORD_LENGTH as usize); // single-row dev_off is flat (== c), ×4 bytes
    let pad32 = n.div_ceil(32) * 32; // N padded up to the fp32 32-elem stick
    let arg_bytes = pad32 * (Fp32::WORD_LENGTH as usize); // fp32 footprint bytes
    assert!(byte_off < arg_bytes); // every fp32 element within its 32-stick·4-byte footprint (no clobber)
    assert!(32 * (Fp32::WORD_LENGTH as usize) == 128); // fp32 stick is 128 B (32·4), same HBM stick as fp16 (64·2)
}

// ── FIX-GATE (gates the WORKER disk→block-weight host gather, proven BEFORE the staging code) ── The KSPLIT
// worker stages block b of a GEMM weight by gathering the logical [K,N] weight's K-slice rows [b*KB,(b+1)*KB)
// into a physical [KB,N] block buffer. GEMM weights are on disk as [N,K]=[out,in] (the transpose-B orientation
// contract: the emitter reads [n,k]→[k,n]). So logical weight[k][n] = disk[n][k], and block_w[in_i][out_j] =
// logical[b*KB+in_i][out_j] = disk element [out_j][b*KB+in_i] = disk_off out_j*K + (b*KB+in_i). Prove: (a) the
// gather offset is IN-BOUNDS of the [N,K] disk weight; (b) INJECTIVE over (in_i,out_j) ⇒ the [KB,N] block
// buffer is a clean bijection (no gather collision); (c) block b reads exactly disk K-columns [b*KB,(b+1)*KB)
// (the K-slice, disjoint across blocks). Fail-first: dropping the transpose (disk[in][out]) or the b*KB base
// breaks (a)/(c). Complements `ksplit_block_weight_kslice_offset_matches_read` (which proves the DEVICE-retile
// side); this proves the DISK-gather side the worker must implement. granite o_proj: N=K=2048, KB=512, B=4.
#[kani::proof]
fn ksplit_block_weight_disk_gather_offset() {
    const KB: usize = 512; // per-block K extent (8 sticks)
    // SYMBOLIC over N (out) and K (in) so it covers EVERY KSPLIT matmul incl. the lm_head (N=vocab≈49155,
    // K=hidden=2048) — the first-test scope — NOT just o_proj's N=K=2048. K a multiple of KB.
    let k: usize = kani::any();
    kani::assume(k >= 1024 && k <= 8192 && k % KB == 0); // hidden(2048) … intermediate(8192)
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 65536); // out cols: 2048 (o_proj/down) … 8192 (gate/up) … ~49155 (lm_head)
    let b_blocks: usize = k / KB;
    let b: usize = kani::any();
    kani::assume(b < b_blocks);
    let in_i: usize = kani::any();
    kani::assume(in_i < KB);
    let out_j: usize = kani::any();
    kani::assume(out_j < n);
    let disk_off = out_j * k + (b * KB + in_i); // transpose-B gather from disk [N,K]
    assert!(disk_off < n * k); // (a) in-bounds of the [N,K] disk weight
    // (b) INJECTIVE over (in_i, out_j) — the [KB,N] block buffer is a bijection with the disk K-slice:
    let (in_i2, out_j2): (usize, usize) = (kani::any(), kani::any());
    kani::assume(in_i2 < KB && out_j2 < n && !(in_i == in_i2 && out_j == out_j2));
    assert!(out_j2 * k + (b * KB + in_i2) != disk_off);
    // (c) block b gathers exactly disk K-columns [b*KB, (b+1)*KB) — the K-slice, disjoint across blocks:
    let kcol = b * KB + in_i;
    assert!(kcol >= b * KB && kcol < (b + 1) * KB);
}

// ── FIX-GATE (gates the SCRATCHY_KSPLIT partition setup for the granite o_proj): the concrete K=2048 split
// into B=4 stick-aligned blocks is exact, covering, and each block's K is a whole 64-fp16-stick multiple ──
// granite-3.3-2b o_proj K = hidden = 2048. The block matmul emits no coordinateMasking_, so each block's K
// MUST be a 64-multiple. Prove (concrete, vs the generic KB=128 partition proof): (a) B·Kb == K exactly
// (Kb=512, no remainder); (b) Kb is 8 whole fp16 sticks (512 % 64 == 0); (c) every contraction index e∈[0,K)
// lies in exactly its owner block [owner·Kb,(owner+1)·Kb), owner<B (covering); (d) each block's A K-slice
// end (b+1)·Kb ≤ K (in-bounds of the [1,K] activation). A wrong B/Kb (non-stick-aligned or non-dividing)
// breaks the split ⇒ Kani finds it.
#[kani::proof]
fn ksplit_granite_oproj_partition_stick_aligned() {
    const K: usize = 2048; // granite-3.3-2b o_proj contraction = hidden
    const B: usize = 4; // blocks
    const KB: usize = K / B; // 512 per block
    assert!(KB * B == K); // (a) exact split, no remainder
    assert!(KB % 64 == 0); // (b) each block's K is a whole 64-fp16-stick multiple (512 = 8 sticks)
    let e: usize = kani::any();
    kani::assume(e < K);
    let owner = e / KB;
    assert!(owner < B); // (c) covering: e's owner is a real block
    assert!(e >= owner * KB && e < (owner + 1) * KB);
    let b: usize = kani::any();
    kani::assume(b < B);
    assert!((b + 1) * KB <= K); // (d) block b's A K-slice [b·Kb,(b+1)·Kb) in-bounds of [1,K]
}

// ── AddrIR (the FIRST wire island): each op's per-core OUTPUT offset = `dev_off(region-corner)`. This is
//    the #50 collision made UNCONSTRUCTABLE at the wire address level: for one op's `CoreSplit::plan`,
//    DISTINCT cores map to DISTINCT offsets (no aliased write) that all lie in the tensor footprint
//    `r*pad64(c)` (no clobber of the next tensor). Concrete granite shapes + symbolic core indices — the
//    same shape as `plan_regions_disjoint_*` (which closes fast), lifted from region-overlap to the actual
//    `dev_off` address. Closes in seconds ⇒ AddrIR is a right-sized island (the close-time rule).

fn addr_offsets_distinct_in_footprint(r: u32, c: u32) {
    let sp = CoreSplit::plan_capped(r, c, MAX_CORES);
    let (cr1, cs1): (u32, u32) = (kani::any(), kani::any());
    let (cr2, cs2): (u32, u32) = (kani::any(), kani::any());
    kani::assume(
        cr1 < sp.row_cores && cr2 < sp.row_cores && cs1 < sp.stick_cores && cs2 < sp.stick_cores,
    );
    kani::assume(!(cr1 == cr2 && cs1 == cs2)); // distinct cores
    let o1 = core_out_offset(sp, r, c, cr1, cs1);
    let o2 = core_out_offset(sp, r, c, cr2, cs2);
    assert!(o1 != o2); // no #50 at the wire address level
    let foot = (r as usize) * pad64(c as usize); // footprint in elements
    assert!(o1 < foot && o2 < foot); // in the tensor's own footprint — no clobber of the next tensor
}
#[kani::proof]
fn addr_offsets_granite_rope() {
    addr_offsets_distinct_in_footprint(256, 128); // the RoPE-rotate output the monolith #50-collided on
}
#[kani::proof]
fn addr_offsets_non64() {
    addr_offsets_distinct_in_footprint(17, 200); // rows∤cores + cols∤64 edge
}

// ── BRIDGE: RE-ROLL per-layer address advance (AddrIR across layers) ── The re-roll runs ONE body
//    program `iters` times; iteration v reads its per-layer WEIGHT (seg1) / KV (seg2) tensors by
//    advancing the segment base by `v·stride` (`weight_stride`/`kv_stride`, the uniform per-layer byte
//    delta the emitter computes from the placements + build-guards for uniformity). The runtime lands
//    this as a shifted `CompositeAddress` base per layer (the shim fix for dxp ignoring
//    `tensor_byte_offsets`). This advance is a VALID, non-aliasing per-layer selector ONLY IF the
//    stride is at least the per-layer FOOTPRINT — else consecutive layers' regions OVERLAP and one
//    layer's weights corrupt the next (silent cross-layer garbage). Proven: for a uniform stride ≥
//    footprint, NO two distinct layers alias any element (`v·stride + e_v != w·stride + e_w`).
//    The emitter packs each layer's tensors CONTIGUOUSLY (stride == footprint), so the invariant holds.
//    `stride` is CONCRETE per harness (the multiply `v·stride` stays LINEAR ⇒ CBMC-tractable; a
//    symbolic×symbolic stride blows up bit-blasting). The invariant is scale-free, so a stick-scale
//    (64) and a hidden-scale (4096) stride cover it. DISCOVERED fail-first: with the F ≤ S assume
//    RELAXED (footprint up to 2·stride) Kani finds `v·S+e_v == w·S+e_w` for v≠w (consecutive layers
//    overlap) — the cross-layer corruption the advance must avoid; F ≤ S closes it.
fn per_layer_advance_no_alias_for(stride: u64) {
    let iters: u32 = kani::any();
    kani::assume(iters >= 2 && iters <= 8);
    let footprint: u64 = kani::any();
    kani::assume(footprint >= 1 && footprint <= stride); // INVARIANT: per-layer footprint fits the stride slot
    let (v, w): (u32, u32) = (kani::any(), kani::any());
    kani::assume(v < iters && w < iters && v != w);
    let (e_v, e_w): (u64, u64) = (kani::any(), kani::any());
    kani::assume(e_v < footprint && e_w < footprint);
    // Byte address (relative to the segment base) the shim's per-layer advance computes.
    let a_v = v as u64 * stride + e_v;
    let a_w = w as u64 * stride + e_w;
    assert!(a_v != a_w); // distinct layers must NEVER alias an element (no cross-layer corruption)
}
// ── BRIDGE: SEN169_FP16 (1-6-9, bias 31) ENCODE ── `sen169_bits` (the reduce `scaling_factor` and every
// SFP-const encoding) is now an INTEGER `to_bits()` encoder (NO libm log2/powi — that was the float leaf
// that forced the `a/exp2(e)` fixups AND put it out of Kani's reach). to_bits + integer shift/mask/add is
// EXACTLY CBMC's domain, so the encoder is now machine-checked here (was a concrete unit test only). The
// rewrite is bit-identical to the old log2 encoder on a 417k-value sweep (verified) ⇒ on-card unchanged.
#[kani::proof]
fn sen169_encode_anchors() {
    // to_bits of a CONCRETE f32 is a constant ⇒ CBMC evaluates the integer path directly (no libm).
    assert!(sen169_bits(1.0) == 0x3E00); // 2^0 ⇒ exp field 31 (bias 31), mant 0 — SFP plus1
    assert!(sen169_bits(-1.0) == 0xBE00); // sign | plus1 — SFP minus1
    assert!(sen169_bits(0.5) == 0x3C00); // 2^-1 ⇒ exp field 30 — SFP fastSigmoidConst
    assert!(sen169_bits(2.0) == 0x4000); // 2^1 ⇒ exp field 32
    assert!(sen169_bits(1.0 / 4096.0) == 0x2600); // 2^-12 ⇒ exp field 19
    assert!(sen169_bits(0.0) == 0x0000); // zero
    assert!(sen169_bits(1.0) != 0x3C00); // NOT IEEE-f16 1.0 (0x3C00) — the mismatch that WAS the on-card bug
}

#[kani::proof]
fn sen169_encode_exponent_structure() {
    // Build a NORMALIZED f32 from symbolic bits and prove the SEN169 exponent FIELD is exactly the IEEE
    // exponent rebiased (bias 127 → 31): field = ieee_exp − 96 (±1 from a mantissa-rounding carry). Keep
    // ieee_exp ∈ [98,156] so the result is a SEN169 normal (field 2..60, clear of underflow/saturate).
    let ie: u32 = kani::any();
    kani::assume(ie >= 98 && ie <= 156);
    let mant23: u32 = kani::any();
    kani::assume(mant23 <= 0x007F_FFFF);
    let sign: u32 = kani::any();
    kani::assume(sign <= 1);
    let bits = (sign << 31) | (ie << 23) | mant23;
    let v = f32::from_bits(bits); // finite, non-zero, non-subnormal (ie∈[98,156]) ⇒ no early return
    let out = sen169_bits(v);
    // sign bit preserved
    assert!(((out >> 15) & 1) as u32 == sign);
    // exponent field = ieee_exp − 96 (or +1 when the top-9 mantissa rounding carries)
    let ef = ((out >> 9) & 0x3F) as i32;
    let base = ie as i32 - 96;
    assert!(ef == base || ef == base + 1);
    // mantissa field is 9 bits (packing lock)
    assert!(((out & 0x1FF) as u32) <= 0x1FF);
}

#[kani::proof]
fn per_layer_advance_no_alias_stick() {
    per_layer_advance_no_alias_for(64); // one-stick per-layer stride
}
#[kani::proof]
fn per_layer_advance_no_alias_hidden() {
    per_layer_advance_no_alias_for(4096); // hidden-scale per-layer stride
}

// ── BRIDGE: SubtileIR SFP op → emitted `constantInfo_` MUST match the op's DDL external-constant
//    contract. An op's `constantInfo_` supplies the EXTERNAL constants its DDL template reads via
//    `ddl.get_external_constant`; attaching MORE than the DDL reads puts extra values in SFP-LRF that
//    SHADOW the DDL's own `ddl.define_constant`s of the same name → the on-card transcendental refines
//    against the wrong coefficients (the exact rsqrt/sqrt shadowing already carved out of the table).
//
//    DDL GROUND TRUTH (IBM reference `/opt/ibm/spyre/deeptools/share/ddc/ddl_templates/`):
//      • `silu` (unary_parallel.ddl:155-210): its SIGMOID sub-step + the trailing `FMUL(x,sigmoid)`
//        source ALL 6 coefficients (`expVal1..5`, `zero`) via `ddl.define_constant` [INTERNAL] and read
//        ZERO `get_external_constant`. ⇒ silu's DDL external-constant count = 0 ⇒ constantInfo_ MUST be
//        EMPTY. scratchy's `sfp_constant_table()` is NON-empty AND its names INCLUDE `expVal1..5` — the
//        SAME names the DDL defines internally ⇒ collision ⇒ the sigmoid reads shadowed SFP-LRF.
//    MEASURED on-card (granite micro-g3.3, layer-0 prefill pos0, NOCOLOR scan of the node tensors):
//      gate_proj |.|=216.3 (golden 216.2 ✓), up_proj 150.3 (150.7 ✓), but silu·up |.|=69.4 vs golden
//      81.5 with max PRESERVED (44.9 vs 45.0) — the mid-range-under fingerprint of a shadowed sigmoid,
//      NOT a uniform scale error. So the emitter attaching the SFP external table to Silu is the bug.
//
//    This harness FAILS on today's code (`OpFunc::Silu` IS in `needs_sfp_const_table`'s set ⇒ the emitter
//    attaches the 20-entry table for an op whose DDL reads 0 external constants). The fix (drop `Silu`
//    from that set ⇒ empty constantInfo_) turns it green. `exp`/`reciprocal` (used by the PROVEN-correct
//    softmax + rmsnorm-seed) are NOT asserted here — only the measured-broken op is locked.
// ── BRIDGE: COLUMN-BLOCK-SPLIT pointwise (SiluMul) coverage ── The tape splits a wide op (granite
//    MLP intermediate 12800) into col-blocks that PARTITION `[0, C)` and SHARE one output tensor
//    (`ds_name` collapses to `t{tid}`). The emitter MUST offset each chunk's output by its
//    `region.cols.start` (via [`pointwise_chunk_out_offset`]); otherwise every chunk writes the
//    whole-tensor base 0, later blocks' columns stay UNWRITTEN, and the op silently drops part of
//    its output. MEASURED on-card (all other layer-0 ops proven exact): silu·up `t65[8192..12800]=0`
//    (chunk-1 `cols.start=8192` written at offset 0) → ~27% of the MLP energy zeroed → wrong token.
//
//    This harness proves COVERAGE: for a 2-block partition of `[0, C)` (`[0, L0)` + `[L0, C)`), using
//    the emitter's per-chunk offset, EVERY output column is written by EXACTLY ONE chunk (no gap, no
//    double-write). With the offset dropped (`pointwise_chunk_out_offset ≡ 0`) both chunks land at 0
//    ⇒ columns in `[max(L0,C−L0), C)` are never written ⇒ the XOR fails. With the offset =
//    `cols.start` the blocks tile `[0, C)` exactly. Symbolic stick-aligned C/L0 span the boundary.
#[kani::proof]
fn silumul_chunks_cover_output() {
    let stk: u32 = 64;
    let c_sticks: u32 = kani::any();
    kani::assume(c_sticks >= 2 && c_sticks <= 400); // up to the granite intermediate (12800 = 200 sticks)
    let l0_sticks: u32 = kani::any();
    kani::assume(l0_sticks >= 1 && l0_sticks < c_sticks); // block 0 is a proper stick-aligned prefix
    let c = c_sticks * stk; // full output width (elems)
    let l0 = l0_sticks * stk; // block 0 = [0, l0), block 1 = [l0, c)
    // The emitter's per-chunk output offsets (block 0 starts at col 0, block 1 at col l0):
    let off0 = pointwise_chunk_out_offset(0);
    let off1 = pointwise_chunk_out_offset(l0);
    // Written ranges: [off, off+len).
    let (a0, b0) = (off0, off0 + l0); // block 0 writes l0 cols
    let (a1, b1) = (off1, off1 + (c - l0)); // block 1 writes (c-l0) cols
    // COVERAGE + DISJOINTNESS: every output column is written by exactly one block.
    let e: u32 = kani::any();
    kani::assume(e < c);
    let in0 = e >= a0 && e < b0;
    let in1 = e >= a1 && e < b1;
    assert!(in0 ^ in1); // exactly one chunk writes column `e` — no unwritten gap, no double-write
}

#[kani::proof]
fn silu_constant_info_matches_ddl_contract() {
    // silu's DDL path (unary_parallel.ddl:155-210) reads NO `get_external_constant`.
    const SILU_DDL_EXTERNAL_CONSTS: usize = 0;
    // The emitter attaches the (non-empty) SFP external table iff `needs_sfp_const_table`.
    let attaches_external_table = OpFunc::Silu.needs_sfp_const_table();
    // INVARIANT: only attach the external table when the DDL actually reads external constants.
    assert!(!attaches_external_table || SILU_DDL_EXTERNAL_CONSTS > 0);
}

// ── BRIDGE (exp constant-table contract — the SAME silu·up shadowing bug, UNFIXED for exp, in EVERY softmax) ──
// The `exp` opFuncName DDL branch (unary_parallel.ddl:98-137, READ on-pod) sources ALL its polynomial
// coefficients — expVal1..5 (0x46dc/0x46e2/0x34c5/0x2121/0x3e00) + eps — via `ddl.define_constant` [INTERNAL]
// and allocates them into SFP-LRF itself; it reads ZERO `get_external_constant`. scratchy's 20-entry
// SfpConstTable REPEATS expVal1..5 as EXTERNAL constants, so attaching it (as `needs_sfp_const_table` does
// for Exp) collides in SFP-LRF and shadows the DDL's own coefficients — IDENTICAL to the measured silu·up
// under-refinement, but here it corrupts the decode-attention softmax `exp` (attn_ep/attn_en) at EVERY step.
// SAME invariant as silu: attach the external table ONLY when the DDL reads external constants. FAIL-FIRST:
// with Exp IN `needs_sfp_const_table` this asserts `!true || 0>0` = false ⇒ Kani RED, localizing the bug; the
// fix (drop Exp from `needs_sfp_const_table`, like silu/rsqrt/sqrt) makes it GREEN.
#[kani::proof]
fn exp_constant_info_matches_ddl_contract() {
    const EXP_DDL_EXTERNAL_CONSTS: usize = 0; // exp branch reads zero get_external_constant (all define_constant)
    let attaches_external_table = OpFunc::Exp.needs_sfp_const_table();
    assert!(!attaches_external_table || EXP_DDL_EXTERNAL_CONSTS > 0);
}

// ── BRIDGE (reciprocal constant-table contract — the SAME shadowing bug, in the softmax `rc = 1/smm`) ──
// The `reciprocal` opFuncName DDL branch (unary_parallel.ddl:382-417, READ on-pod) sources its constants
// plus1(0x3E00)/minus1(0xBE00)/ffff/zero via `ddl.define_constant` [INTERNAL] into SFP-LRF and the SFP
// RECIPROCAL op reads ONLY internal LRF connects (zero_sfp_lrf/ffff_sfp_lrf/minus1_sfp_lrf/sfp_lx_input);
// it reads ZERO `get_external_constant`. scratchy's SfpConstTable REPEATS plus1(0x3E00_3E00)/minus1
// (0xBE00_BE00) as EXTERNAL constants ⇒ attaching it (as `needs_sfp_const_table` did for Reciprocal)
// collides in SFP-LRF — same class as silu·up / exp, and reciprocal runs in the decode softmax denominator.
// Its siblings rsqrt/sqrt were already excluded for this exact reason; reciprocal was missed. FAIL-FIRST:
// with Reciprocal IN the list this asserts `!true || 0>0` = false ⇒ RED; the fix (drop Reciprocal) → GREEN.
#[kani::proof]
fn reciprocal_constant_info_matches_ddl_contract() {
    const RECIP_DDL_EXTERNAL_CONSTS: usize = 0; // reciprocal branch reads zero get_external_constant
    let attaches_external_table = OpFunc::Reciprocal.needs_sfp_const_table();
    assert!(!attaches_external_table || RECIP_DDL_EXTERNAL_CONSTS > 0);
}

// ── BRIDGE (COMPLETE the SFP transcendental const-table audit) ── DDL-CONFIRMED on-pod
// (unary_parallel.ddl): the sigmoid/gelu/mish/tanh branches ALL read ZERO `get_external_constant` (they
// source coefficients via `ddl.define_constant` [INTERNAL], same as silu/exp/reciprocal/rsqrt/sqrt). So
// NO SFP transcendental lowered through this path needs scratchy's external SfpConstTable — attaching it can
// only SHADOW the DDL's own coefficients in SFP-LRF (the measured silu·up bug class). These lock the
// invariant for the WHOLE op set so a future model using any of them cannot silently reintroduce the shadow.
// FAIL-FIRST: any of these ops still IN `needs_sfp_const_table` makes `!true || 0>0` = false ⇒ RED.
// ── BRIDGE (matmul dataformat == bmm.ddl contract — regression guard for the SEN169-scale-bug class) ──
// bmm.ddl (READ on-pod): the fp16 batchmatmul binds `%type_fp16 = ddl.type {data_type="SEN169_FP16"}` for
// input (%inptensor_fp16 line 40), kernel (%kertensor_fp16 line 41), output (line 46), AND both accumulators
// (%ptsum_fp / %pesum, lines 47/49). The emitter tags every plain matmul tensor via `dataformat_for(name)`
// → `Fp16::NAME`. If that ever drifted to IEEE_FP16 (the exact SEN169-scale bug: device reads bits as 1-6-9
// but they were encoded 1-5-10), every matmul operand would be MIS-DECODED on-card. Lock the emitter's fp16
// dataformat to the DDL string + its 2-byte / 64-elem (128-byte) stick, and the fp32-merge format to
// IEEE_FP32 (bmm_sen1p5.ddl %type_fp32) / 4-byte. FAIL-FIRST: a wrong NAME/word-length breaks the assert.
#[kani::proof]
fn matmul_dataformat_matches_bmm_ddl_contract() {
    assert!(Fp16::NAME == "SEN169_FP16"); // bmm.ddl %type_fp16 data_type — NOT IEEE_FP16 (the scale-bug)
    assert!(Fp16::WORD_LENGTH == 2); // 2-byte fp16 operand
    assert!(Fp16::ELEMS_PER_STICK == 64); // 64 fp16 elems = one 128-byte HBM stick
    assert!(Fp32::NAME == "IEEE_FP32"); // bmm_sen1p5.ddl %type_fp32 (the KSPLIT merge partial format)
    assert!(Fp32::WORD_LENGTH == 4); // 4-byte fp32
    assert!(Fp32::ELEMS_PER_STICK == 32); // 32 fp32 elems = one 128-byte HBM stick
}

#[kani::proof]
fn sigmoid_constant_info_matches_ddl_contract() {
    assert!(!OpFunc::Sigmoid.needs_sfp_const_table()); // sigmoid DDL reads zero get_external_constant
}
#[kani::proof]
fn gelu_constant_info_matches_ddl_contract() {
    assert!(!OpFunc::Gelu.needs_sfp_const_table()); // gelu DDL reads zero get_external_constant
}
#[kani::proof]
fn mish_constant_info_matches_ddl_contract() {
    assert!(!OpFunc::Mish.needs_sfp_const_table()); // mish DDL reads zero get_external_constant
}
#[kani::proof]
fn tanh_constant_info_matches_ddl_contract() {
    assert!(!OpFunc::Tanh.needs_sfp_const_table()); // tanh DDL reads zero get_external_constant
}

// ── BRIDGE (decode-attention STRUCTURE = the mask EXTENT + JOINT softmax that `lower_attn_node` emits) ──
// The emitter attends the resident cache over the FULL `cap` slots but ADDS `pmask` (`sp[h] += pmask`,
// worker builds pmask = 0 on cols `[0..p)`, `mask_neg` on `[p..cap)`; lower_subtile…:4830), and folds the
// new token in as an implicit extra slot under ONE joint softmax: single max (`mx = max(mxp, sn)`), single
// denominator (`smm = smp + exp_n`), one reciprocal (`rc`), one combine (`out = opre + onew`). The GOLDEN
// semantics is `attn_reference` (sdsc_abstract): a contiguous softmax over `nslot = p+1` slots — slots
// `0..p` = the p cached prefix tokens, slot `p` = the new token. This harness PROVES those two shapes agree.
//
// The exp/score computation is the float ALU leaf (Kani cannot verify libm exp — see module header), so it
// is FACTORED OUT: softmax WEIGHTS are arbitrary non-negative integers `w[s]`, `w_new` (whatever exp
// produced) and VALUES are arbitrary integers. The CLAIM is purely structural — WHICH slots enter the sum
// and HOW they are normalized: (a) the prefix contributes EXACTLY cols `[0..p)` (the masked `[p..cap)`
// slots drop out with weight 0), (b) the new token is EXACTLY one extra slot == reference slot `p`, (c) the
// denominator is JOINT over all `p+1` (NOT two separate softmaxes). A wrong extent (`p±1`) or a SPLIT
// denominator breaks the reindexing and Kani finds a counterexample. Integer arithmetic ⇒ CBMC-exact.
// This is the norm-preserving direction-shift class the pos-64 on-card residual pointed at: if the extent
// were off by one, the softmax would redistribute weight over the wrong slot count (‖attn_out‖ ~preserved,
// direction shifted) — so this is the proof that adjudicates that measured signature as structural-or-not.
#[kani::proof]
fn decode_attn_masked_split_equals_contiguous_reference() {
    const CAP: usize = 4; // resident-cache capacity (device attends all cap, masks [p..cap))
    let p: usize = kani::any();
    kani::assume(p >= 1 && p < CAP); // a real decode step: 1..cap-1 cached prefix tokens

    // Abstract softmax weights (the exp result — the float leaf factored out; whatever exp produced).
    // Kept SYMBOLIC — the whole point is "for ALL weight distributions the two shapes agree". Non-negative,
    // bounded so the linear sums stay well within i64.
    let w: [i64; CAP] = [kani::any(), kani::any(), kani::any(), kani::any()];
    let w_new: i64 = kani::any();
    for s in 0..CAP {
        kani::assume(w[s] >= 0 && w[s] <= 1_000_000);
    }
    kani::assume(w_new >= 0 && w_new <= 1_000_000);
    // VALUES are CONCRETE and DISTINCT (per slot) — a symbolic value would make every term a symbolic×symbolic
    // 64-bit multiply (nonlinear → CBMC blows up); a distinct CONCRETE value keeps each `w[s]·value` LINEAR in
    // the symbolic weight while still exposing a wrong slot mapping (a mis-indexed slot picks up the wrong,
    // observably-different coefficient). This adjudicates the STRUCTURE (which slots, how normalized), which is
    // exactly the mask-extent / joint-vs-split question; the per-element float value is the ALU leaf.
    let vval = |s: usize| -> i64 { (s as i64) * 7 + 3 }; // prefix slot s value (distinct, non-zero)
    let vnew: i64 = 999; // new-token value (distinct from every prefix vval(s) for s<CAP)

    // ── EMITTER STRUCTURE: masked prefix [0..p) (the pmask extent — [p..cap) weight-0) + the new token,
    //    ONE joint denominator (`smm = smp + exp_n`).
    let mut split_den: i64 = w_new;
    let mut split_num: i64 = w_new * vnew; // onew
    for s in 0..CAP {
        // The mask extent is the SHARED `decode_prefix_col_valid` the worker's pmask fill ACTUALLY uses —
        // this proof therefore covers the on-card mask decision, not a transcription of it.
        if decode_prefix_col_valid(s, p) {
            split_den += w[s];
            split_num += w[s] * vval(s); // opre
        }
    }

    // ── GOLDEN: `attn_reference` contiguous over nslot = p+1 (slot s<p = prefix; slot p = the new token).
    //    Iterate the CONCRETE bound 0..CAP with an `if s < nslot` guard (a symbolic loop bound `0..nslot`
    //    forces CBMC to reason about an unknown trip count and blows up; nslot = p+1 ≤ CAP here).
    let nslot = p + 1;
    let mut ref_den: i64 = 0;
    let mut ref_num: i64 = 0;
    for s in 0..CAP {
        if s < nslot {
            let (ws, vs) = if s < p {
                (w[s], vval(s))
            } else {
                (w_new, vnew)
            };
            ref_den += ws;
            ref_num += ws * vs;
        }
    }

    // EQUAL as rationals: same numerator AND same denominator ⇒ split_num/split_den == ref_num/ref_den.
    // (No cross-multiply — equal-num ∧ equal-den suffices and stays linear.) A wrong prefix extent (p±1) or a
    // SPLIT denominator makes split_num pick up vval(p)/miss w_new ⇒ ≠ ref_num, and Kani finds it.
    assert!(split_den == ref_den);
    assert!(split_num == ref_num);
}

// ── BRIDGE: decode-attn structure ACROSS the 64-slot STICK BOUNDARY (the pos-64 divergence, bracketed
// on-card: granite-3.3-2b is CORRECT through pos 43 and DIVERGES at pos 64 = the first VALID stick-1 slot).
// The CAP=4 proof above never let the prefix reach slot 64. This re-runs the EXACT extent + joint-softmax
// claim with CAP=68 so `p` can be 60..67 — the masked prefix [0..p) now SPANS stick 0 (0..63) INTO stick 1,
// and the folded new-token slot `p` lands in stick 1. If `decode_prefix_col_valid` or the joint denominator
// mis-handles the boundary, Kani finds a counterexample. (STRUCTURE only — the sticked ADDRESS of col≥64 is
// score_prob_perhead_layout_cap256; this proof adjudicates whether the extent/fold logic is boundary-safe.)
// CONCRETE `p` (no symbolic-p branching → CBMC only reasons over the symbolic linear weight sums → fast,
// like CAP=4). CAP=66 (>64) so the cache spans 2 sticks; each harness pins `p` at a boundary value.
fn decode_attn_masked_split_at_p(p: usize) {
    // Kani-driven subdivision: CAP=66 with the literal 64-slot boundary timed out (>100s / >18min) — the
    // 66-elem symbolic surface is CBMC-hostile. The extent/joint-softmax logic under test (the worker's
    // pmask `decode_prefix_col_valid(s,p) = s < p` + linear split-vs-contiguous sums) is provably
    // STRIDE-AGNOSTIC: it contains NO `/64` or `%64` stick arithmetic (that lives in
    // `score_prob_perhead_layout_cap256_granite`, a separate proof), so the claim at a small block
    // boundary (here BLK=4) is the IDENTICAL theorem as at 64. We prove it with a 2-block cache (CAP=8,
    // block0 = 0..3, block1 = 4..7); slots < LO are concrete 0 (inert), the boundary window [LO..CAP) is
    // symbolic. Closes in ~1s. (The stride-agnosticism of `decode_prefix_col_valid` is itself pinned by
    // `decode_prefix_mask_is_stride_agnostic` below.)
    const CAP: usize = 8; // 2 blocks of BLK=4 (block0 = 0..3, block1 = 4..7)
    let mut w = [0i32; CAP];
    for s in 2..CAP {
        w[s] = kani::any();
        kani::assume(w[s] >= 0 && w[s] <= 15);
    }
    let w_new: i32 = kani::any();
    kani::assume(w_new >= 0 && w_new <= 15);
    let vval = |s: usize| -> i32 { (s as i32) * 7 + 3 }; // distinct per slot; max = 65*7+3 = 458
    // Distinct from every vval(s) (s<CAP), but SMALL: a large constant multiplier (was 100_003, a
    // 17-bit circuit) is what made this harness CBMC-hostile (~80s+ / timeout). The proof only needs
    // vnew ≠ any vval(s) to catch a mis-mapped new-token slot; 460 (> max vval 458) suffices and keeps
    // the w_new·vnew multiply a narrow circuit → closes in ~1s.
    let vnew: i32 = 460;
    // SUBDIVISION (Kani-driven: the full 0..CAP loop is too monolithic — 132 iterations over a 66-elem
    // array timed out >120s solo). Slots 0..59 are CONCRETELY w=0 (set above), so they contribute 0 to
    // BOTH the split and reference sums regardless of the mask/extent — omitting them is exactly equal.
    // Iterate ONLY the symbolic boundary window [60..CAP), which spans stick0(63)→stick1(64,65) and the
    // folded new-token slot p. This is the identical claim over the only slots that carry symbolic weight,
    // and closes in ~1s. (`p` boundary values 63/64/65 all lie in this window or fold as the new token.)
    const LO: usize = 2;
    // EMITTER: masked prefix [0..p) via the SHARED decode_prefix_col_valid + folded new token, ONE joint denom.
    let mut split_den: i32 = w_new;
    let mut split_num: i32 = w_new * vnew;
    for s in LO..CAP {
        if decode_prefix_col_valid(s, p) {
            split_den += w[s];
            split_num += w[s] * vval(s);
        }
    }
    // GOLDEN: contiguous over nslot = p+1 (slot<p prefix, slot p new token).
    let nslot = p + 1;
    let mut ref_den: i32 = 0;
    let mut ref_num: i32 = 0;
    for s in LO..CAP {
        if s < nslot {
            let (ws, vs) = if s < p {
                (w[s], vval(s))
            } else {
                (w_new, vnew)
            };
            ref_den += ws;
            ref_num += ws * vs;
        }
    }
    assert!(split_den == ref_den);
    assert!(split_num == ref_num);
}
// The three boundary cases (scaled to BLK=4 — identical theorem to the 64-stick boundary, since the
// extent logic is stride-agnostic; see `decode_prefix_mask_is_stride_agnostic`). `p` folds the new token
// at the last slot of block0, the first slot of block1, and one past it.
#[kani::proof]
fn decode_attn_masked_split_p_last_block0() {
    decode_attn_masked_split_at_p(3); // last block-0 slot valid; new token folds at slot 3 (= block0 end)
}
#[kani::proof]
fn decode_attn_masked_split_p_first_block1() {
    decode_attn_masked_split_at_p(4); // THE boundary: prefix [0..4) spans block0; new token slot 4 = block1
}
#[kani::proof]
fn decode_attn_masked_split_p_into_block1() {
    decode_attn_masked_split_at_p(5); // prefix [0..5) reaches INTO block1 (slot 4 valid in the masked prefix)
}

/// Pins the LICENSE for the BLK=4 scaling above: the decode prefix-mask is `col < p` with NO stick/block
/// arithmetic (`/BLK` or `%BLK`), so the masked-split-vs-contiguous extent theorem is independent of where
/// the 64-slot stick boundary falls. Symbolic over col,p — the actual 64-stick ADDRESS boundary is owned
/// by `score_prob_perhead_layout_cap256_granite`.
#[kani::proof]
fn decode_prefix_mask_is_stride_agnostic() {
    let col: usize = kani::any();
    let p: usize = kani::any();
    kani::assume(col < 4096 && p <= 4096);
    assert!(decode_prefix_col_valid(col, p) == (col < p));
}

// ── THE SHARED WRITE SLOT. A launch resolves ONE slot shift for every trip inside it, so a batch whose
//    requests append at their own positions is one launch per request — 8 x 40 launches at the ~93 us
//    floor, 26 ms of a 111 ms step at bs=8. One launch requires ONE slot, and a shared slot is past the
//    shortest request's own length, so its keys stop being `[0, len)` and become two runs with a hole.
//    These four proofs are the licence for that: the hole is real, the length law admits it, the history
//    excludes it, and the shared slot is always past every history so nothing overwrites its own keys.

//    Proved over `KvHistory`'s FREE LAWS (`contiguous_run`/`appended_run`/`shared_write_slot`/
//    `runs_contain`) and fixed-size arrays, not over the `Vec`-backed type: the type is those four lines
//    plus a growable list, and a `Vec` here costs CBMC an allocator model — the length law above closes in
//    0.06s over slices and did not close in ten minutes over `Vec`. `mask_laws.rs` exercises the type.

/// The empty run — what [`contiguous_run`] yields for a request that holds nothing yet.
const EMPTY_RUN: SlotRun = SlotRun::new(KvSlot::ZERO, SlotCount::new(0));

/// ⛔ THE BUG THE HISTORY EXISTS TO PREVENT — that `col < p` is not merely imprecise about a batched
/// request, it is WRONG, and wrong in the direction that reads other keys as this request's own.
///
/// Take a request of `len` keys that then appends at a shared slot past its length. `col < end` is TRUE
/// for every column of the hole `[len, shared)`, which the request never wrote: whatever the pool holds
/// there — another request's evicted keys, or nothing — enters its softmax with weight. This holds for
/// EVERY hole, so there is no batch shape that escapes it by being lucky. It is what makes the history a
/// correctness requirement of the fused write rather than a tidier way to say the same thing.
#[kani::proof]
fn a_length_cannot_describe_a_shared_slot_history() {
    let len: u32 = kani::any();
    let shared: u32 = kani::any();
    let col: u32 = kani::any();
    kani::assume(len < 4096 && shared < 4096 && len < shared);
    kani::assume(col >= len && col < shared); // a column of the hole
    let prompt = contiguous_run(SlotCount::new(len)).unwrap_or(EMPTY_RUN);
    let appended = appended_run(KvSlot::new(len), KvSlot::new(shared), SlotCount::ONE);
    let runs = [prompt, appended];
    // The length law says this column is valid...
    assert!(decode_prefix_col_valid(
        col as usize,
        appended.end().get() as usize
    ));
    // ...and the request does not hold it.
    assert!(!runs_contain(&runs, KvSlot::new(col)));
}

/// AND THE HISTORY IS EXACT — valid on precisely the slots written, both runs, nothing between.
///
/// Proved over the shape a batch actually produces (a prompt, then one shared append) rather than over an
/// arbitrary run list, because that is the shape whose mask goes to the card. The membership law itself is
/// proved over arbitrary runs by `runs_contain_is_the_union_of_the_runs`.
#[kani::proof]
fn history_mask_is_exactly_the_slots_written() {
    let len: u32 = kani::any();
    let shared: u32 = kani::any();
    let col: u32 = kani::any();
    kani::assume(len < 4096 && shared < 4096 && shared >= len);
    kani::assume(col < 4096);
    let runs = [
        contiguous_run(SlotCount::new(len)).unwrap_or(EMPTY_RUN),
        appended_run(KvSlot::new(len), KvSlot::new(shared), SlotCount::ONE),
    ];
    assert!(runs_contain(&runs, KvSlot::new(col)) == (col < len || col == shared));
    // AND THE APPEND LANDED WHERE THE LAUNCH PUT IT. The launch writes at `shared`; if the history
    // recorded anything else the mask would describe a slot the card never wrote.
    assert!(runs_contain(&runs, KvSlot::new(shared)));
    assert!(runs[1].end().get() == shared + 1);
}

/// A CONTIGUOUS HISTORY IS THE LENGTH LAW, so a request that has only ever run alone gets byte-identical
/// mask bytes to the ones that shipped. The generalisation has to be a generalisation, not a change.
#[kani::proof]
fn contiguous_history_is_the_length_law() {
    let len: u32 = kani::any();
    let col: u32 = kani::any();
    kani::assume(len <= 4096 && col < 4096);
    let runs = contiguous_run(SlotCount::new(len));
    // An empty history holds nothing, which is what a length of 0 says too.
    assert!(runs.is_some() == (len > 0));
    let held = runs.is_some_and(|r| runs_contain(&[r], KvSlot::new(col)));
    assert!(held == decode_prefix_col_valid(col as usize, len as usize));
    assert!(runs.map_or(0, |r| r.end().get()) == len);
}

/// THE MEMBERSHIP LAW over an arbitrary run list: inside iff inside some run.
#[kani::proof]
fn runs_contain_is_the_union_of_the_runs() {
    let (a0, an): (u32, u32) = (kani::any(), kani::any());
    let (b0, bn): (u32, u32) = (kani::any(), kani::any());
    let col: u32 = kani::any();
    // BOUND FIRST, THEN RELATE: `a0 + an` on unconstrained `u32`s overflows, and an overflow inside a
    // `kani::assume` is a verification failure rather than a filtered case.
    kani::assume(a0 < 4096 && an > 0 && an < 4096 && b0 < 4096 && bn > 0 && bn < 4096);
    kani::assume(col < 4096 && a0 + an <= b0); // ascending, disjoint
    let a = SlotRun::new(KvSlot::new(a0), SlotCount::new(an));
    let b = SlotRun::new(KvSlot::new(b0), SlotCount::new(bn));
    let runs = [a, b];
    let inside = a.contains(KvSlot::new(col)) || b.contains(KvSlot::new(col));
    assert!(runs_contain(&runs, KvSlot::new(col)) == inside);
}

/// WHY RECORDING NEEDS NO ASSERT: the slot a batch appends at is past EVERY live history, so no request
/// can overwrite keys it already holds, and [`appended_run`]'s clamp provably never fires.
///
/// This is the whole safety argument for deciding the slot once for the batch. It is also what makes the
/// slot self-advancing: after a step every live end is `shared + 1`, so the maximum moves by exactly one
/// without a stored cursor that could go stale against the requests it describes.
#[kani::proof]
fn batch_slot_is_past_every_history() {
    let raw: [u32; 3] = [kani::any(), kani::any(), kani::any()];
    kani::assume(raw.iter().all(|&e| e < 4096));
    let ends: [KvSlot; 3] = raw.map(KvSlot::new);
    let slot = shared_write_slot(&ends);
    assert!(ends.iter().all(|&e| slot.get() >= e.get()));
    // AND IT IS ONE OF THEM, not merely an upper bound: a slot past every history by more than it needs
    // to be would leave a hole in every request at once, for no reason.
    assert!(ends.contains(&slot));
    // So the clamp never fires — every request's append lands exactly at the launch's slot...
    let after = ends.map(|e| appended_run(e, slot, SlotCount::ONE));
    assert!(after.iter().all(|r| r.start().get() == slot.get()));
    // ...every live history then ends at the SAME place, and the next step's maximum is this one plus
    // one. That is what makes the batch's write fusable from then on, with no session-held cursor.
    let next: [KvSlot; 3] = after.map(|r| r.end());
    assert!(next.iter().all(|e| e.get() == slot.get() + 1));
    assert!(shared_write_slot(&next).get() == slot.get() + 1);
}

/// AN APPEND NEVER LANDS BEHIND THE HISTORY IT EXTENDS — for any `at` whatsoever, including one the
/// proof above rules out. Total by construction: the clamp is the reason there is no assert to hit.
#[kani::proof]
fn an_append_never_overwrites_its_own_keys() {
    let last_end: u32 = kani::any();
    let at: u32 = kani::any();
    let n: u32 = kani::any();
    kani::assume(last_end < 4096 && at < 4096 && n >= 1 && n <= 4096);
    let run = appended_run(KvSlot::new(last_end), KvSlot::new(at), SlotCount::new(n));
    assert!(run.start().get() >= last_end);
    assert!(run.end().get() == run.start().get() + n);
    // And when the slot IS past the history — the only case the worker produces — it is honoured
    // exactly, because a launch writes at its own slot and the mask has to say so.
    if at >= last_end {
        assert!(run.start().get() == at);
    }
}

// ── BRIDGE: EMBED gather → device read (decode mq=1). The worker gathers token `tok`'s embedding row
// `embed_tokens[tok*hidden .. +hidden)` (row-major [vocab,hidden]) and binds it HOST-CONTIGUOUS as the
// `[1,hidden]` seg0 activation; the device (rmsnorm's first read) addresses it via `dev_off([1,hidden],1,
// [0,c])`. For that to read component `c` of token `tok`, the single-row device layout MUST be the
// identity (offset == c). If it weren't, EVERY decode-mq=1 [1,hidden] activation (embed, residual stream,
// rmsnorm I/O, o_proj out) would be scrambled — the embed→rmsnorm input first. Also: distinct token rows
// must be element-disjoint (no blended embedding). Pure integer addressing ⇒ CBMC-exact.
#[kani::proof]
fn single_row_device_layout_is_identity() {
    // Every decode activation is [1, N]; the device read must be contiguous (offset == logical column).
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 4096); // up to granite hidden
    let c: usize = kani::any();
    kani::assume(c < n);
    assert!(dev_off(&[1, n], 1, &[0, c]) == c);
}
#[kani::proof]
fn embed_gather_matches_device_read() {
    const HIDDEN: usize = 4096; // granite hidden
    const VOCAB: usize = 49159; // granite vocab
    let tok: usize = kani::any();
    kani::assume(tok < VOCAB);
    let c: usize = kani::any();
    kani::assume(c < HIDDEN);
    // Component c of token tok's embedding is host-gathered from flat table offset tok*HIDDEN + c and
    // bound to device offset dev_off([1,HIDDEN],1,[0,c]); the rmsnorm read of logical column c must land
    // there ⇒ device offset == c (contiguous single row).
    assert!(dev_off(&[1, HIDDEN], 1, &[0, c]) == c);
    // Distinct token rows never share a table element (gather can't blend two embeddings).
    let other: usize = kani::any();
    kani::assume(other < VOCAB && other != tok);
    let oc: usize = kani::any();
    kani::assume(oc < HIDDEN);
    assert!(tok * HIDDEN + c != other * HIDDEN + oc);
}

// ── BRIDGE: RoPE rotate-half permutation matrix (per-node `lower_rope_node`) ── RoPE computes
// `out = x·cos + rot·sin` where `rot = matmul(x, P)` and P[in][o] = `rope_p_entry(hd,in,o)` — the SHARED
// function the worker ACTUALLY fills ROPE_P from (`spyre_worker.rs`), so this proves the on-card code, NOT
// a transcription of it (the reading gap the methodology forbids). The GOLDEN NeoX rotate-half
// (rope_reference, sdsc_abstract): `rot[o] = -x[o+half]` (o<half), `+x[o-half]` (o>=half). If `rope_p_entry`
// were wrong (sign flip, wrong offset), the query/key ROTATES WRONG — a NORM-PRESERVING direction shift
// (rotation preserves ‖·‖), the exact class of the measured pos-64 residual. The cos/sin multiply is the
// float ALU leaf, factored out; the PERMUTATION is pure integer ⇒ CBMC-exact. Symbolic integer x; hd=8
// exercises both o-halves (structure is size-independent). Since the worker fills P via THIS function, a
// GREEN proof here means the on-card rotate-half is correct — not just a model of it.
#[kani::proof]
fn rope_permutation_matrix_is_rotate_half() {
    const HD: usize = 8;
    const HALF: usize = HD / 2;
    let x: [i64; HD] = [
        kani::any(),
        kani::any(),
        kani::any(),
        kani::any(),
        kani::any(),
        kani::any(),
        kani::any(),
        kani::any(),
    ];
    for i in 0..HD {
        kani::assume(x[i] >= -1000 && x[i] <= 1000);
    }
    // rot = x · P where P[in][o] = rope_p_entry (the worker's actual fill). rot[o] = Σ_in x[in]·P[in][o],
    // compared to the NeoX rotate-half reference. rope_p_entry returns 0/±1 ⇒ each term is linear in x.
    for o in 0..HD {
        let mut acc = 0i64;
        for inn in 0..HD {
            acc += x[inn] * (rope_p_entry(HD, inn, o) as i64);
        }
        let want = if o < HALF { -x[o + HALF] } else { x[o - HALF] };
        assert!(acc == want);
    }
}

// ── BRIDGE: the ACTUAL kernel RetileDescriptor's device_size/stride_map WALK == dev_off ── retile.rs
// proves a hand-written `device_flat` formula ≡ dev_off, but the on-card staging (C++
// `sdsc_stage_weight_tiled`) does NOT use that formula — it WALKS the emitter-built RetileDescriptor's
// `device_size`/`stride_map` (built from `DeviceTileLayout`, lower_subtile…:5678). This harness closes that
// gap: it calls the ACTUAL `DeviceTileLayout::new/device_size/stride_map` (concrete kernel dims fold to
// constants in CBMC — the Err/format! paths prune), then simulates the shim's exact walk (device-flat =
// row-major over device_size; host element gathered via stride_map) and proves it lands on the SAME device
// cell the matmul reads via dev_off. A wrong device_size/stride_map (or wrong walk) ⇒ scrambled weight.
fn retile_descriptor_walk_equals_dev_off_for(in_: u64, out: u64) {
    let tile = DeviceTileLayout::<Fp16>::new(&["in", "out"], "out", &[in_, out]).unwrap();
    let ds = tile.device_size(); // [out/64, in, 64]
    let sm = tile.stride_map(); //  [64, out, 1]
    let d0: u64 = kani::any();
    let d1: u64 = kani::any();
    let d2: u64 = kani::any();
    kani::assume(d0 < ds[0] && d1 < ds[1] && d2 < ds[2]);
    // shim: device cell (d0,d1,d2) is written at the row-major flat offset over device_size, gathering the
    // host element at Σ d_k·stride_map[k].
    let device_flat = d0 * (ds[1] * ds[2]) + d1 * ds[2] + d2;
    let host_off = d0 * sm[0] + d1 * sm[1] + d2 * sm[2];
    // that host element is (row i, col j) in the [in,out] row-major kernel buffer.
    let i = host_off / out;
    let j = host_off % out;
    // the matmul READS kernel elem (i,j) at dev_off([in,out],1,[i,j]); staging must WRITE device_flat there.
    let read = dev_off(&[in_ as usize, out as usize], 1, &[i as usize, j as usize]) as u64;
    assert!(device_flat == read);
    assert!(device_flat < in_ * out); // stays in the kernel footprint (no clobber)
}
#[kani::proof]
fn retile_descriptor_walk_equals_dev_off_hd128() {
    retile_descriptor_walk_equals_dev_off_for(4096, 128); // granite attn kernel (out=128=2 sticks)
}
#[kani::proof]
fn retile_descriptor_walk_equals_dev_off_lmhead() {
    retile_descriptor_walk_equals_dev_off_for(4096, 49664); // padded lm_head (out=49664, 776 sticks)
}

// ── BRIDGE: the granite muP multipliers encode to DISTINCT SEN169 consts ── granite applies four muP
// ScalarMul scales — embedding_multiplier=12, residual_multiplier=0.22, attention_multiplier=0.0078125,
// logits_scaling=16. Each is bound as a SEN169 const via the ACTUAL `sen169_bits`. If any two encoded to
// the SAME bits, the on-card op would read the WRONG multiplier (e.g. residual scaled by the attention
// multiplier) — silent corruption invisible to a cosine self-test. Prove all four are pairwise-distinct
// under the real encoder (concrete ⇒ CBMC folds sen169_bits directly). Complements the TID-distinctness
// lock (`scalarmul_tid_distinct_and_reserved`): distinct INDEX and distinct VALUE.
#[kani::proof]
fn granite_mup_scales_distinct_sen169() {
    let emb = sen169_bits(12.0); // embedding_multiplier
    let res = sen169_bits(0.22); // residual_multiplier
    let attn = sen169_bits(0.0078125); // attention_multiplier (= 1/128, the score scale)
    let logit = sen169_bits(16.0); // logits_scaling
    assert!(emb != res && emb != attn && emb != logit);
    assert!(res != attn && res != logit);
    assert!(attn != logit);
    // none is +0 (0x0000) — a scale of 0 would ZERO the activation it multiplies.
    assert!(emb != 0 && res != 0 && attn != 0 && logit != 0);
}

// ── BRIDGE: the EMITTER's decode per-core address formula is region-disjoint ── The emitter
// (lower_subtile_tape_to_superdsc.rs:1795-1797) gives core `cs` on the split output dim the CORNER
// `cs·per_core_extent` and the device offset `dev_off(host_size, stick_idx, corner)` (the ACTUAL shared
// dev_off). For decode (m=1) the output is `[1, cols]` split across `S` cores on the cols/stick dim; an
// even split has `per_core_extent = cols/S`. `plan_regions_disjoint_*` proves the CoreSplit tiling
// disjoint, but the emitter uses THIS explicit `cs·per_core_extent` corner — prove IT gives disjoint,
// in-bounds column ranges (no core clobbers another's output) using the ACTUAL dev_off. `per` concrete
// (stick-aligned) keeps `cs·per` linear; `S` symbolic covers every core count up to 32.
fn decode_per_core_regions_disjoint_for(per: usize) {
    let s: usize = kani::any();
    kani::assume(s >= 1 && s <= 32); // ≤ MAX_CORES
    let cols = s * per; // even split: cols = S · per_core_extent
    let (c1, c2): (usize, usize) = (kani::any(), kani::any());
    kani::assume(c1 < s && c2 < s && c1 != c2);
    let corner1 = c1 * per; // emitter: corner = slice_idx · per_core_extent
    let corner2 = c2 * per;
    // ACTUAL emitter address: dev_off on the [1,cols] decode output (single row ⇒ contiguous).
    let off1 = dev_off(&[1, cols], 1, &[0, corner1]);
    let off2 = dev_off(&[1, cols], 1, &[0, corner2]);
    assert!(off1 + per <= cols && off2 + per <= cols); // each core's [off, off+extent) is in-bounds
    // distinct cores own NON-OVERLAPPING column ranges (no clobber).
    if c1 < c2 {
        assert!(off1 + per <= off2);
    } else {
        assert!(off2 + per <= off1);
    }
}
#[kani::proof]
fn decode_per_core_regions_disjoint_1stick() {
    decode_per_core_regions_disjoint_for(64); // per_core_extent = 1 stick
}
#[kani::proof]
fn decode_per_core_regions_disjoint_2stick() {
    decode_per_core_regions_disjoint_for(128); // per_core_extent = 2 sticks (granite hd=128 output tile)
}

// ── BRIDGE: decode KV write-slot vs read-range (context carry + no double-count) ── At decode position p
// the shim `host_kv_write` scatters the new token's K/V to cache slot = p (via kcache_kt_write_offset),
// while the prefix attention reads slots where `decode_prefix_col_valid(col, p)` holds (= [0..p)). The
// central decode-correctness invariant — proven here against the ACTUAL shared read-range function:
//   (1) NO DOUBLE-COUNT: slot p is NOT in this step's prefix read (else the new token counts twice — once
//       via the cache, once via the new-token `sn`/`onew` path — inflating its attention weight).
//   (2) CARRIES CONTEXT: at the NEXT step (p+1) slot p IS read ⇒ the new token becomes prefix (without
//       this, decode has no growing context ⇒ degenerate repetition — the "ectable" bug).
//   (3) NO OVERWRITE: distinct positions map to distinct slots (slot = position, injective for p < cap).
// Fail-first-coupled to the mask extent: an off-by-one `decode_prefix_col_valid` (col<=p) breaks (1).
#[kani::proof]
fn decode_kv_write_slot_no_double_count_and_carries() {
    const CAP: usize = 256;
    let p: usize = kani::any();
    kani::assume(p < CAP - 1); // leave room for the next step p+1 < cap
    let write_slot = p; // shim host_kv_write: new token → slot = current absolute position
    assert!(!decode_prefix_col_valid(write_slot, p)); // (1) not read THIS step
    assert!(decode_prefix_col_valid(write_slot, p + 1)); // (2) read NEXT step
    let q: usize = kani::any();
    kani::assume(q < CAP && q != p);
    assert!(q != write_slot); // (3) distinct position ⇒ distinct slot (slot == position)
}

// ── BRIDGE: KV scatter PER-HEAD BASE composition (C++ shim host_kv_write) ── The C++ Route-B scatter
// (sdsc_shim.cpp:2258 K, :2262 V) writes the full offset as a PER-HEAD BASE plus the within-head formula:
//   K: koff = qh*hd*cap + (slot/stk)*hd*stk + d*stk + slot%stk = qh*(hd*cap) + kcache_kt_write_offset(...)
//   V: voff = qh*cap*hd + (d/stk)*cap*stk + slot*stk + d%stk   = qh*(cap*hd) + vcache_write_offset(...)
// The within-head parts are already proven == dev_off (kc/vc_write_equals_*_read). This proves the PART
// the C++ ADDS — the `qh*(hd*cap)` / `qh*(cap*hd)` per-head base: (a) the full scatter offset EQUALS the
// full per-head consumer read (per-head base + dev_off), and (b) the within-head offset stays inside one
// per-head stride ⇒ distinct heads occupy DISJOINT regions (no cross-head clobber). Documents+locks the
// exact formula the C++ must implement, tied to the actual proven Rust offset functions.
#[kani::proof]
fn kv_scatter_per_head_base_matches_read_and_no_alias() {
    let (hd, cap, stk) = (HD_GRANITE, CAP, STK); // granite hd=128, cap=256
    let qh: usize = kani::any();
    let slot: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(qh < NQH_G && slot < cap && d < hd);
    // ── K (Kᵀ layout, per-head stride hd*cap) ──
    let k_within = kcache_kt_write_offset(slot, d, hd, cap, stk);
    let k_write = qh * hd * cap + k_within; // C++ shim scatter
    let k_read = qh * hd * cap + dev_off(&[hd, cap], 1, &[d, slot]); // score matmul per-head read
    assert!(k_write == k_read); // producer (C++ scatter) == consumer (score read)
    assert!(k_within < hd * cap); // within-head stays in the per-head stride ⇒ heads disjoint
    // ── V (natural layout, per-head stride cap*hd) ──
    let v_within = vcache_write_offset(slot, d, hd, cap, stk);
    let v_write = qh * cap * hd + v_within;
    let v_read = qh * cap * hd + dev_off(&[cap, hd], 1, &[slot, d]); // value bmm per-head read
    assert!(v_write == v_read);
    assert!(v_within < cap * hd);
}

// ── BRIDGE: lm_head vocab-padding cannot corrupt the argmax ── The lm_head output is [1, padded_vocab]
// with padded_vocab = DeviceWidth::for_output(1, vocab, hidden) (granite 49159 → 49664, +505 pad cols). The
// worker argmaxes over `logits[0..vocab]` (spyre_worker.rs:506,686), EXCLUDING the padding. Prove, using the
// ACTUAL DeviceWidth padding rule + dev_off: (a) padded ≥ vocab — every real token is present/addressable
// (no real logit truncated, which would drop the golden token from contention); (b) real token t (< vocab)
// sits at contiguous device index t (single-row identity) so the [0..vocab) slice captures exactly the real
// logits; (c) any padding index is ≥ vocab ⇒ outside the argmax slice ⇒ can NEVER win (a padding win would
// emit an invalid token ≥ vocab). This is why the on-card argmax is always a valid vocab id (e.g. 1318).
#[kani::proof]
fn lmhead_padding_excluded_from_argmax() {
    let vocab: usize = 49159; // granite
    let hidden: usize = 4096;
    let padded = DeviceWidth::for_output(1, vocab as u32, hidden as u32).get() as usize; // ACTUAL pad rule
    assert!(padded >= vocab); // (a) every real token present + addressable
    let t: usize = kani::any();
    kani::assume(t < vocab);
    assert!(dev_off(&[1, padded], 1, &[0, t]) == t); // (b) real token t at contiguous index t
    assert!(t < padded);
    let pad_idx: usize = kani::any();
    kani::assume(pad_idx >= vocab && pad_idx < padded);
    assert!(!(pad_idx < vocab)); // (c) padding index never inside the [0..vocab) argmax slice
}

// ── BRIDGE: per-layer UNIFORMITY check ⟹ the shim's LINEAR advance is exact ── The reroll emitter accepts
// per-layer weight/KV packing ONLY if consecutive-layer offset deltas are UNIFORM (o[i+1]-o[i]==stride ∀i,
// else `cargo build` Err, lower_subtile_tape_to_superdsc.rs:5737). The executor/shim then reaches layer v's
// tensors via `seg_base + v·stride`. Prove the check's PURPOSE: given ONLY the uniformity the emitter
// enforces (not a constructed-linear layout), the shim's linear advance EQUALS the actual layer-v placement
// (telescoping), and layers are strictly ordered ⇒ non-aliasing. If a layer were packed off the uniform
// stride, the emitter rejects it at build time; if it passes, this proves the advance can't read the wrong
// layer's weights (the silent cross-layer corruption the check guards). granite = 4 layers.
#[kani::proof]
fn per_layer_uniform_stride_implies_exact_linear_advance() {
    const N: usize = 4; // granite decoder layers
    let o: [u64; N] = [kani::any(), kani::any(), kani::any(), kani::any()];
    let mut i = 0;
    while i < N {
        kani::assume(o[i] <= (1u64 << 40)); // realistic HBM offsets; no add-overflow
        i += 1;
    }
    kani::assume(o[1] >= o[0]);
    let stride = o[1] - o[0];
    kani::assume(stride > 0); // distinct layers (granite weight_stride = 398475264 ≠ 0)
    // the emitter's ACCEPTED invariant: every consecutive delta equals `stride` (else build Err).
    kani::assume(o[2] >= o[1] && o[2] - o[1] == stride);
    kani::assume(o[3] >= o[2] && o[3] - o[2] == stride);
    // ⇒ shim advance `seg_base + v·stride` lands EXACTLY on layer v's actual placement o[v] (telescoping):
    assert!(o[0] + 0 * stride == o[0]);
    assert!(o[0] + 1 * stride == o[1]);
    assert!(o[0] + 2 * stride == o[2]);
    assert!(o[0] + 3 * stride == o[3]);
    // strictly increasing ⇒ distinct layers never alias (no cross-layer weight corruption).
    assert!(o[0] < o[1] && o[1] < o[2] && o[2] < o[3]);
}

// ── BRIDGE: RoPE forms a proper 2D ROTATION per (d, d+half) pair ── RoPE computes out[d] = x[d]·cos[d] +
// rot[d]·sin[d] with rot = matmul(x, P) via the ACTUAL `rope_p_entry`. `rope_cos_sin` (manifest.rs) lays
// cos/sin with the half-frequencies REPEATED (cos[d]=cos[d+half]=c_d, sin likewise) so a rotation pair
// shares one angle. Prove the pair (out[d], out[d+half]) is the 2D rotation of (x[d], x[d+half]) by (c_d,s_d):
//   out[d]      = c_d·x[d] − s_d·x[d+half],   out[d+half] = s_d·x[d] + c_d·x[d+half].
// This is the NORM-PRESERVING property — a wrong P (rope_p_entry) or a non-repeated cos would make it NOT a
// rotation (direction shifts while ‖·‖ ~holds — the exact pos-64 residual class). Uses the actual rope_p_entry;
// trig values c/s are the float leaf, factored out as symbolic; x concrete-distinct keeps every term linear.
#[kani::proof]
fn rope_forms_2d_rotation_per_pair() {
    const HD: usize = 8;
    const HALF: usize = HD / 2;
    let x: [i64; HD] = [10, 20, 30, 40, 50, 60, 70, 80]; // concrete distinct pre-rope vector
    let c: [i64; HALF] = [kani::any(), kani::any(), kani::any(), kani::any()];
    let s: [i64; HALF] = [kani::any(), kani::any(), kani::any(), kani::any()];
    for k in 0..HALF {
        kani::assume(c[k] >= -1000 && c[k] <= 1000 && s[k] >= -1000 && s[k] <= 1000);
    }
    // rope_cos_sin repetition: cos[d]=c[d mod half-ish]=c[d if d<half else d-half]; sin likewise.
    let cosv = |d: usize| -> i64 { c[if d < HALF { d } else { d - HALF }] };
    let sinv = |d: usize| -> i64 { s[if d < HALF { d } else { d - HALF }] };
    // rot = x · P via the ACTUAL rope_p_entry.
    let rot = |o: usize| -> i64 {
        let mut a = 0i64;
        for inn in 0..HD {
            a += x[inn] * (rope_p_entry(HD, inn, o) as i64);
        }
        a
    };
    // out[d] = x[d]·cos[d] + rot[d]·sin[d]
    let out = |d: usize| -> i64 { x[d] * cosv(d) + rot(d) * sinv(d) };
    let d: usize = kani::any();
    kani::assume(d < HALF);
    // proper 2D rotation of (x[d], x[d+half]) by (c_d, s_d):
    assert!(out(d) == c[d] * x[d] - s[d] * x[d + HALF]);
    assert!(out(d + HALF) == s[d] * x[d] + c[d] * x[d + HALF]);
}

// ── BRIDGE: granite rmsnorm mean-scale is EXACT in SEN169 (no 14× scale-bug reintroduction) ── The mean
// reduce bakes scaling_factor = 1.0/cols (assemble_reduce: `OpFunc::Mean => 1/N`), cols = the reduction
// extent. For granite rmsnorm cols = hidden = 4096, so the scale = 1/4096 = 2^-12 — a power of two, hence
// EXACTLY representable in SEN169_FP16 (mantissa 0). Prove the ACTUAL sen169_bits(1/4096) encodes to exp
// field 19 / mant 0, i.e. decodes to precisely 2^-12 = 1/4096 ⇒ mean = sum/4096 exactly (the SEN169 scale
// bug — mean 4600× too small from an IEEE-vs-SEN169 mis-encode — cannot recur for the granite hidden).
#[kani::proof]
fn rmsnorm_mean_scale_granite_exact() {
    let bits = sen169_bits(1.0 / 4096.0); // ACTUAL encoder, granite hidden reduction
    assert!(bits == 0x2600);
    let exp_field = ((bits >> 9) & 0x3F) as i32;
    let mant = bits & 0x1FF;
    // decode value = (1 + mant/512)·2^(exp_field-31); mant==0 ∧ exp_field-31==-12 ⇒ value == 2^-12 == 1/4096.
    assert!(exp_field == 19);
    assert!(mant == 0);
    assert!(exp_field - 31 == -12); // exact power-of-two ⇒ no mantissa rounding error in the mean scale
}

// ── BRIDGE: RMSNorm rsqrt Newton CONVERGENCE (the per-node lowering leaf I had only ever called "empirically
// correct" from on-card output — FORBIDDEN as a correctness basis; now PROVEN). The emitter refines the SFP
// rsqrt SEED by the Newton iteration `y ← y·(1.5 − 0.5·x·y²)` (lower_subtile_tape_to_superdsc.rs:1021; the
// worker binds HALF=0.5 EXACT in fp16, and 1.5 = 0.5+0.5+0.5, line 1035). Its exact fixed point is y*=1/√x
// (x·y*²=1 ⇒ 1.5−0.5·1 = 1 ⇒ y·1 = y). Writing y = y*(1+e), ONE step maps the RELATIVE error
//   e ↦ e' = −1.5e² − 0.5e³ = −e·(1.5e + 0.5e²),
// so the iteration STRICTLY CONTRACTS (|e'| < |e|, i.e. converges toward y*) ⟺ |1.5e + 0.5e²| < 1. Prove
// this over the WHOLE basin |e| ≤ 0.5 (the SFP seed lands well inside) in EXACT integer arithmetic: with
// e = E/1000, 1.5e + 0.5e² = (3000·E + E²)/2_000_000, so the contraction condition is the pure integer
// inequality |3000·E + E²| < 2_000_000 for every |E| ≤ 500. This uses the ACTUAL coefficients 3/2 and 1/2 —
// fail-first: a wrong HALF (e.g. 1.0 → the `4000·E` map) makes |4000·500 + 500²| = 2_250_000 ≥ 2_000_000, so
// Kani reports the E where contraction fails (and the fixed point would no longer be 1/√x). CBMC-tractable
// (one bounded integer, degree-2 polynomial). Green ⇒ the Newton refine provably drives the seed to 1/√x.
#[kani::proof]
fn rmsnorm_rsqrt_newton_error_contracts_in_basin() {
    let e: i64 = kani::any(); // relative error e = E/1000 (y = y*·(1+e))
    kani::assume(e >= -500 && e <= 500 && e != 0); // basin |e| ≤ 0.5, nonzero (e==0 is the exact fixed point)
    let num = 3000 * e + e * e; // = (1.5e + 0.5e²) · 2_000_000  (the ACTUAL 3/2, 1/2 Newton coefficients)
    let abs = if num < 0 { -num } else { num };
    // |1.5e + 0.5e²| < 1 ⇒ |e'| = |e|·|1.5e + 0.5e²| < |e|: the Newton step strictly reduces the error.
    assert!(abs < 2_000_000);
}

// ── BRIDGE (granite-3.3-2b muP scale SEN169 exactness — LOCALIZES the residual_multiplier precision gap) ──
// The muP ScalarMul scales (embedding/residual/attn/logits multipliers) flow to the bundle as
// `scalarmul_scales`, bound as [1,1] consts the on-card pointwise `mul` reads in SEN169_FP16 (1-6-9, 9-bit
// mantissa). Golden (metal) applies them in fp32 (23-bit). A scale that is NOT dyadic-with-≤9-mantissa-bits is
// applied with MORE rounding on-card than in the fp32 golden — a SYSTEMATIC per-layer divergence
// (residual_multiplier runs once per layer, ~40× for granite-3.3-2b). Prove the exactness inventory FROM the
// ACTUAL `sen169_bits` encoder (granite-3.3-2b config: attention_multiplier=0.015625, logits_scaling=8,
// hidden=2048): the power-of-two scales are EXACT (mantissa field 0, no rounding), but
// residual_multiplier=0.22 (=11/50, NON-dyadic) rounds to 901/4096 ≠ 11/50 on-card. The
// `lower_subtile_tape_to_ktir.rs:470` comment claiming "0.22 exactly f16-representable" is FALSE — NO binary
// float represents 0.22 exactly. FAIL-first: the `mant != 0` / `901*50 != 11*4096` asserts localize 0.22 as
// the SEN169-lossy residual scale (candidate pos-65 near-tie contributor); the reference fix is to apply it
// in fp32 (DL16TOFP32, as torch-spyre does for the model's fp32 regions). If my hand-computed exp/mant are
// wrong, Kani FAILS and reports the real encoding (let Kani say, don't assume the arithmetic).
#[kani::proof]
fn granite2b_mup_scales_sen169_exactness() {
    // EXACT in SEN169 (power-of-two ⇒ mantissa field 0, zero rounding):
    assert!((sen169_bits(0.015625) & 0x1FF) == 0); // attention_multiplier = 1/64 = 2^-6 (granite-2b, hd=64)
    assert!((sen169_bits(8.0) & 0x1FF) == 0); // logits_scaling = 2^3
    assert!((sen169_bits(1.0 / 2048.0) & 0x1FF) == 0); // rms mean scale, granite-2b hidden=2048 = 2^-11
    // residual_multiplier = 0.22 = 11/50 is NON-dyadic ⇒ SEN169 must round it.
    let bits = sen169_bits(0.22);
    let exp = ((bits >> 9) & 0x3F) as i64 - 31; // unbiased exponent (bias 31)
    let mant = (bits & 0x1FF) as i64; // 9-bit mantissa field
    // 0.22 ∈ [2^-3, 2^-2) ⇒ normalized exponent -3; SEN169 encodes mantissa round(0.76·512) = 389.
    assert!(exp == -3);
    assert!(mant == 389);
    // decoded on-card value = (512 + mant)/512 · 2^exp = 901/4096. Prove 901/4096 ≠ 11/50 (= 0.22) EXACTLY:
    assert!(901 * 50 != 11 * 4096); // 45050 ≠ 45056 ⇒ the on-card residual multiplier is NOT 0.22
    assert!(mant != 0); // ⇒ NOT SEN169-exact (contradicting the false "exactly f16-representable" claim)
}

// ── BRIDGE (weight conversion — a bridge I had been ASSUMING, now proven divergent): the granite bf16→SEN169
// weight path routes through an IEEE-fp16 INTERMEDIATE that CLAMPS the exponent range ── granite-3.3-2b is
// bf16 (7-bit mantissa, 8-bit exponent). scratchy narrows bf16 → IEEE-fp16 on ingest (ktir-emulator) then the
// shim converts IEEE-fp16 → SEN169 (`sdsc_stage_weight_tiled` takes `host_ieee`; `sdsc_ieee_to_sen_fp16`).
// IEEE-fp16 has a 5-bit exponent (min NORMAL 2^-14); SEN169 has a 6-bit exponent (min normal 2^-30).
// torch-spyre converts bf16 → SEN169 DIRECTLY (its DL16 path, no IEEE-fp16 hop). Prove the intermediate
// DEGRADES a valid SMALL bf16 weight (exponent -20, below IEEE-fp16's normal range) that a direct bf16→SEN169
// keeps exactly. Witness v = (1 + 1/128)·2^-20 — a legal 7-bit-mantissa bf16 value.
// (a) DIRECT bf16→SEN169: exp -20 ≥ SEN169 min-normal exp -30 ⇒ NORMAL; 1/128 = 4/512 fits the 9-bit
//     mantissa EXACTLY (mant field 4) ⇒ SEN169 keeps v with full precision.
// (b) via IEEE-fp16: exp -20 < IEEE min-normal exp -14 ⇒ SUBNORMAL, grid = integer·2^-24; v is NOT on that
//     grid ⇒ IEEE-fp16 rounds v away (to 2^-20, losing the 1/128) ⇒ the intermediate loses what SEN169 keeps.
// ⇒ small-magnitude weights are degraded by the IEEE-fp16 hop; the fix is a DIRECT bf16→SEN169 conversion
// (as torch-spyre does). Impact is limited to |w| < ~6e-5 (a minority of weights), but it IS a proven,
// reference-divergent, fully-in-scratchy's-control precision loss — no longer an unverified assumption.
#[kani::proof]
fn weight_bf16_ieee_f16_intermediate_degrades_small_weight_vs_direct_sen169() {
    let bits = sen169_bits((1.0 + 1.0 / 128.0) / 1048576.0); // (1 + 2^-7)·2^-20, a valid bf16 value
    let exp = ((bits >> 9) & 0x3F) as i32 - 31;
    let mant = (bits & 0x1FF) as i32;
    assert!(exp == -20); // (a) NORMAL in SEN169 (full relative precision, not subnormal/zero)
    assert!(mant == 4); //     the 1/128 mantissa bit preserved exactly (4/512)
    // (b) IEEE-fp16 at exp -20 is SUBNORMAL (grid integer·2^-24); v·2^24 = 16 + 1/8, scaled ·8 = 129:
    assert!(129 % 8 != 0); // ⇒ v ∉ IEEE-fp16 subnormal grid ⇒ the intermediate degrades the weight
}

// ── FIX-GATE (proven BEFORE the code, per Kani-before-code): a DIRECT bf16→SEN169 conversion is LOSSLESS for
// EVERY bf16 value in SEN169's normal exponent range ── this gates the fix for the IEEE-fp16-intermediate
// degradation proven by `weight_bf16_ieee_f16_intermediate_degrades_small_weight_vs_direct_sen169`. A bf16
// value is `(1 + m/128)·2^e`, m∈[0,127] (7-bit mantissa), e the exponent. For e in SEN169's NORMAL range
// [-30, 31] (exp field [1,62]), the ACTUAL `sen169_bits` encoder maps it EXACTLY: SEN169's 9-bit mantissa
// field = m·4 (the 7 bf16 bits, no rounding since 7 ≤ 9), exponent field = e+31 (SEN169 bias 31, no clamp —
// SEN169's 6-bit exponent covers the range IEEE-fp16's 5-bit does not). Symbolic over ALL (m, e) ⇒ proves the
// direct conversion preserves every in-range bf16 weight, so replacing the IEEE-fp16 hop with a direct
// bf16→SEN169 (as torch-spyre does) is correct + strictly recovers the degraded small weights. CBMC-tractable
// (`sen169_bits` is pure to_bits + integer shifts, no floats).
#[kani::proof]
fn direct_bf16_to_sen169_is_lossless_in_normal_range() {
    let m: u32 = kani::any();
    kani::assume(m <= 127); // bf16 7-bit trailing mantissa
    let e: i32 = kani::any();
    kani::assume(e >= -30 && e <= 31); // SEN169 NORMAL exponent range (field e+31 ∈ [1,62])
    // Construct v = (1 + m/128)·2^e as an EXACT f32 (bf16 ⊂ f32): f32 exp field = e+127, top 7 mantissa bits.
    let f32_bits: u32 = (((e + 127) as u32) << 23) | (m << 16);
    let v = f32::from_bits(f32_bits);
    let sb = sen169_bits(v);
    let exp_field = ((sb >> 9) & 0x3F) as i32;
    let mant = (sb & 0x1FF) as u32;
    assert!(exp_field == e + 31); // exponent preserved via SEN169's 6-bit field (NO IEEE-fp16 5-bit clamp)
    assert!(mant == m * 4); // 7-bit bf16 mantissa preserved EXACTLY in the 9-bit field (m/128 = m·4/512)
    assert!(sb & 0x8000 == 0); // positive witness path (sign bit clear) — the mantissa/exponent claim
}

// ── BRIDGE (refutes a FALSE range belief in the score-path comment): SEN169_FP16's max is ~4.3e9, NOT 65504
// — so the attention score (~1e5) does NOT overflow SEN169 ── SEN169_FP16 is 1-6-9 (6-bit exponent, bias 31),
// so its max finite ≈ (2 − 2^-9)·2^31 ≈ 4.3e9. 65504 is IEEE-fp16's max (5-bit exp), a DIFFERENT format. The
// emitter's score-path comment claimed "SEN169 max 65504, q·k ~1e5 overflows → inf ⇒ BF16E". Prove via the
// ACTUAL `sen169_bits` encoder that 1e5 (and even 65504) encode as NORMAL SEN169 values far below the exp-62
// ceiling ⇒ NO overflow. ⇒ the BF16E score path rests on a false SEN169-range belief; SEN169 (9-bit mantissa)
// holds the score with MORE precision than BF16E (1-8-7, 7-bit mantissa) — a candidate precision improvement
// (follow-up: verify on-card whether the historical `sp` inf was really SEN169 saturation or something else).
#[kani::proof]
fn sen169_holds_attention_score_no_overflow() {
    let s = sen169_bits(100000.0); // ~1e5, the score-path magnitude the comment claimed overflows
    let exp_field = (s >> 9) & 0x3F;
    let mant = s & 0x1FF;
    assert!(exp_field >= 1 && exp_field <= 62); // NORMAL (encoder saturates only at exp_field ≥ 63) ⇒ no overflow
    assert!(!(exp_field == 62 && mant == 0x1FF)); // NOT the max-finite saturate value ⇒ real headroom above 1e5
    // 65504 (IEEE-fp16's max) is itself a mid-range SEN169 normal, far below SEN169's exp-62 ceiling:
    let m = sen169_bits(65504.0);
    let mexp = (m >> 9) & 0x3F;
    assert!(mexp >= 1 && mexp < 62); // ⇒ "SEN169 max 65504" is false (65504 is exp ~15, not the exp-31 ceiling)
}

// ── BRIDGE: granite attention score-scale is the CONFIG value, exact + distinguishable from the bug ── The
// score scale MUST be the config attention_multiplier (0.0078125 = 1/128 = 2^-7), applied as qs = Q·scale;
// using the recomputed 1/sqrt(hd) (= 1/sqrt(128) ≈ 0.0884) was the ~11.3× divergence bug. Prove via the
// ACTUAL sen169_bits: (a) the config scale encodes EXACTLY to 2^-7 (power of two, mant 0 — no precision
// loss in the score scaling), and (b) it encodes to DIFFERENT bits than 1/sqrt(hd), so the fix (bind config,
// never recompute) is distinguishable on-card — a regression to 1/sqrt(hd) would emit an observably different
// const, not silently alias.
#[kani::proof]
fn granite_attn_scale_config_not_recomputed() {
    let config_scale = sen169_bits(0.0078125); // 1/128 = 2^-7 (config attention_multiplier)
    assert!(config_scale == 0x3000);
    assert!(((config_scale >> 9) & 0x3F) as i32 - 31 == -7); // exp field 24 ⇒ 2^-7
    assert!((config_scale & 0x1FF) == 0); // mant 0 ⇒ exact power of two
    let recomputed = sen169_bits(0.08838835); // 1/sqrt(128) — the OLD recompute bug value
    assert!(config_scale != recomputed); // the config-vs-recompute fix is on-card-distinguishable
}

// ── BRIDGE: attention SCORE/PROB per-head [1,cap] layout (the sp inf-bug class, at cap width) ── sp/exp_p/
// pp/subp are declared [nqh, cap] but the emitter writes AND reads them PER-HEAD as [1,cap] at base h·cap
// (lower_attn_node: attn_qkp writes sp[h] @ h·cap, attn_pm/attn_ep/attn_pp all read/write @ h·cap). This is
// the SAME per-head-flat invariant already proven for the [1,hd] intermediates — but the score/prob width is
// `cap`, NOT `hd`, and the per-head layout harness was only invoked at hd∈{64,128,512}. The sp mismatch
// (per-head [1,cap] write vs a full [nqh,cap] device-TILED read resolving the same element to different
// addresses for h≥1) WAS the on-card inf bug (sdsc_abstract.rs:87-102). Prove the per-head [1,cap] layout is
// flat + injective + in-footprint at granite cap=256 (4 sticks — the multi-stick regime).
#[kani::proof]
fn score_prob_perhead_layout_cap256_granite() {
    intermediate_perhead_layout_consistent_for(NQH_G, CAP); // nqh=32, per-head width = cap = 256
}

// ── BRIDGE: f16 accumulation ≠ f32 accumulation (the measured pos-64 divergence, and it IS Kani-provable) ──
// The on-card attention value-bmm / o_proj accumulate products in f16 (SEN169 PSUM); golden (mlx) accumulates
// in f32. The full pos-64 decomposition MEASURED every value through t60 (post-o_proj·0.22) matching golden to
// 0.3%, yet t61 (h_after_attn) diverges 4.6% — law-of-cosines pins that to a ~0.09% DIRECTION error in the
// post-o_proj output, i.e. f16 accumulation dropping sub-ULP contributions that f32 keeps. This is IEEE
// add/mul (CBMC's domain — NOT exp/sqrt), so it IS provable: prove f16-rounding after an add drops a term
// that f32 retains ⇒ f16-accum ≠ f32-accum. This LOCALIZES the fix: accumulate in f32 on-card.
// Round a POSITIVE f32 in [1,2) to nearest IEEE f16 (1-5-10), returned as f32. Pure bit manipulation —
// CBMC-tractable (unlike `half::f16::from_f32`, which does CPU feature-detection Kani can't model).
fn round_to_f16_as_f32(x: f32) -> f32 {
    let b = x.to_bits();
    let dropped = b & 0x1FFF; // low 13 mantissa bits f16 discards
    let kept = b & !0x1FFF; // sign + exp + top 10 mantissa bits
    let mid = 0x1000u32; // 2^12, midpoint of the 13 dropped bits
    let round_up = dropped > mid || (dropped == mid && (kept & 0x2000) != 0); // round-half-to-even
    let r = if round_up { kept + 0x2000 } else { kept }; // 0x2000 = one f16 ULP in f32 mantissa space
    f32::from_bits(r)
}

#[kani::proof]
fn f16_accumulation_drops_terms_f32_keeps() {
    // A running sum near 1.0 (an o_proj/attn partial) plus a contribution below the f16 ULP at 1.0 (2^-10).
    let running: f32 = 1.0;
    let contrib: f32 = kani::any();
    // 2^-20 ≤ contrib < 2^-11: above the f32 ULP at 1.0 (2^-23, so f32 keeps it) yet below the f16 round
    // midpoint (2^-11, so f16 drops it back to 1.0). This is the sub-f16-ULP regime accumulation lives in.
    kani::assume(contrib >= 1.0f32 / 1048576.0f32 && contrib < 1.0f32 / 2048.0f32);
    // f16 accumulation: round the partial sum to f16 (as the on-card SEN169 PSUM does).
    let f16_step = round_to_f16_as_f32(running + contrib);
    // f32 accumulation: full precision (as golden/mlx does).
    let f32_step = running + contrib;
    assert!(f16_step == running); // f16 DROPS the contribution (rounds back to 1.0)
    assert!(f32_step > running); // f32 KEEPS it
    assert!(f16_step != f32_step); // ⇒ f16-accum ≠ f32-accum — the precision loss the on-card fix must address
}

// ── BRIDGE: BLOCKED (split-K) f16 summation retains precision that SEQUENTIAL f16 drops ── the RCUDD1A
// matmul PE accumulates in f16 (proven above; fp32 PE is SEN1P5-only). The on-card mitigation is split-K:
// reduce the contraction in blocks so each f16 running sum is smaller (drops fewer sub-ULP terms), then
// merge the block partials. Prove the mechanism: a big running sum (1.0) plus several small terms — each
// below the f16 ULP at 1.0 — is fully DROPPED by sequential f16 accumulation (stays 1.0), but a blocked
// accumulation sums the small terms together first (small running sum ⇒ no drop) so their combined mass
// survives the final merge. This LOCALIZES the fix (split-K) and proves it strictly improves on sequential.
#[kani::proof]
fn blocked_f16_sum_beats_sequential() {
    let small: f32 = kani::any();
    // 2^-12 ≤ small < 2^-11: each `1.0 + small` rounds back to 1.0 (below the f16 midpoint), yet 3·small
    // (≥ 3·2^-12 > 2^-11) survives a merge onto 1.0. This is the sub-ULP regime a long reduction lives in.
    kani::assume(small >= 1.0f32 / 4096.0f32 && small < 1.0f32 / 2048.0f32);

    // SEQUENTIAL f16 accumulation (what the RCUDD1A PE does over the full K): 1.0, then +small ×3.
    let mut seq = 1.0f32;
    seq = round_to_f16_as_f32(seq + small);
    seq = round_to_f16_as_f32(seq + small);
    seq = round_to_f16_as_f32(seq + small);
    assert!(seq == 1.0f32); // sequential DROPS all three small terms

    // BLOCKED (split-K): accumulate the three small terms in their own block (small running sum ⇒ kept),
    // then merge onto the big partial.
    let mut blk = round_to_f16_as_f32(small + small);
    blk = round_to_f16_as_f32(blk + small); // block partial = 3·small (retained)
    let merged = round_to_f16_as_f32(1.0f32 + blk);
    assert!(merged > 1.0f32); // blocked RETAINS the mass sequential dropped ⇒ strictly closer to f32
    assert!(merged != seq); // ⇒ split-K accumulation is not equivalent to sequential f16 — it is more accurate
}

// ── BRIDGE: split-K REDUCES but does NOT ELIMINATE f16 accumulation error (the RCUDD1A ceiling) ── split-K
// with an f16 partial merge still rounds every partial to f16, so it cannot reach the f32 result exactly.
// Prove the CEILING: even blocked f16 accumulation drops a contribution below the *block-partial's* f16 ULP
// that f32 keeps — so split-K is a mitigation, not exact f32 parity. This bounds expectations: on RCUDD1A
// (f16 PE) no software matmul reformulation with f16 merges equals f32; only SEN1P5 fp32-PE (or an fp32 SFP
// merge) does. Fail-first-style: shows the residual gap survives blocking.
#[kani::proof]
fn split_k_f16_merge_still_lossy_vs_f32() {
    // A block partial that has grown to ~1.0, plus a sub-ULP contribution: the f16 MERGE still drops it.
    let partial: f32 = 1.0; // a block partial near 1.0
    let tail: f32 = kani::any();
    kani::assume(tail >= 1.0f32 / 1048576.0f32 && tail < 1.0f32 / 2048.0f32); // [2^-20, 2^-11): f32 keeps, f16 drops
    let f16_merge = round_to_f16_as_f32(partial + tail); // f16 partial-merge (what an f16 split-K merge does)
    let f32_merge = partial + tail;
    assert!(f16_merge == partial); // the f16 MERGE still drops the tail
    assert!(f32_merge > partial); // f32 keeps it
    // ⇒ split-K with an f16 merge is strictly better than sequential (proven) but STILL ≠ f32:
    //   exact parity needs an fp32 accumulator (SEN1P5 PE) or an fp32 SFP merge, not just blocking.
    assert!(f16_merge != f32_merge);
}

// ── BRIDGE (foundation of the K-split-PSUM fix): the K-contraction partition is DISJOINT + COVERING ──
// The fp32-approaching on-card fix is K-split: split the matmul contraction K across `ks` cores, each core
// reduces a K-block in f16 (fewer terms ⇒ less accumulation error, proven `blocked_f16_sum_beats_sequential`),
// then the cores' partials PSUM-accumulate into the SAME output. For that PSUM sum to equal the full-K
// reduction, core c's K-block [c·kb, (c+1)·kb) must PARTITION [0,K) disjointly and completely. Prove it
// (fail-first: a wrong per-core extent leaves a gap or overlap ⇒ the PSUM sum ≠ full reduction ⇒ wrong
// matmul). `kb` concrete (2 sticks) keeps c·kb linear; `ks` symbolic ≤ MAX_CORES. This is the correctness
// precondition the deferred K-split feature (matmul_cost_split ks>1 + PSUM-accumulate addressing + #50-relax)
// must satisfy — proving it FIRST, before the implementation, per the methodology.
#[kani::proof]
fn ksplit_contraction_partition_disjoint_covering() {
    const KB: u32 = 128; // per-core K extent (2 fp16 sticks) — stick-aligned as the matmul requires
    let ks: u32 = kani::any();
    kani::assume(ks >= 1 && ks <= 32); // ≤ MAX_CORES
    let k = ks * KB; // even split: K = ks · KB
    let e: u32 = kani::any(); // a contraction index
    kani::assume(e < k);
    let owner = e / KB; // the core whose K-block contains e
    // COVERING: e lies in its owner's block.
    assert!(e >= owner * KB && e < (owner + 1) * KB);
    assert!(owner < ks); // the owner is a real core
    // DISJOINT: e lies in NO other core's block.
    let other: u32 = kani::any();
    kani::assume(other < ks && other != owner);
    assert!(!(e >= other * KB && e < (other + 1) * KB));
}

// ── BRIDGE (K-split-PSUM fix, step 2): shared output + DISJOINT K-slices ⇒ valid accumulate (the #50-relax
// criterion) ── In a K-split, K is NOT an output dim, so every core writes the SAME output address (which
// the #50 disjoint-output guard currently rejects as a collision). But it is a CORRECT PSUM-accumulate
// precisely because the cores own DISJOINT K-slices: each adds a distinct partial, summing to the full
// reduction (partition proven by `ksplit_contraction_partition_disjoint_covering`). Prove the discriminator
// the #50-relax must use: two distinct K-split cores share the output addr (via the actual dev_off) AND
// their K-slices never overlap ⇒ accumulate, not double-count. A genuine collision (overlapping K) would
// fail the disjointness — so this criterion admits K-split without admitting a real collision.
#[kani::proof]
fn ksplit_shared_output_disjoint_kslice_is_valid_accumulate() {
    const KB: u32 = 128; // per-core K extent (2 sticks)
    let ks: u32 = kani::any();
    kani::assume(ks >= 2 && ks <= 32);
    let (rows, cols) = (1u32, 128u32); // decode m=1 output tile
    let j: u32 = kani::any();
    kani::assume(j < cols);
    let (c1, c2): (u32, u32) = (kani::any(), kani::any());
    kani::assume(c1 < ks && c2 < ks && c1 != c2);
    // (a) both K-split cores write the SAME output address (K is not an output dim) — the actual dev_off.
    let addr1 = dev_off(&[rows as usize, cols as usize], 1, &[0, j as usize]);
    let addr2 = dev_off(&[rows as usize, cols as usize], 1, &[0, j as usize]);
    assert!(addr1 == addr2); // shared output (what #50 sees as a "collision")
    // (b) but the two cores' K-slices are DISJOINT ⇒ the shared write is accumulate, not double-count.
    let e: u32 = kani::any();
    kani::assume(e < ks * KB);
    let in_c1 = e >= c1 * KB && e < (c1 + 1) * KB;
    let in_c2 = e >= c2 * KB && e < (c2 + 1) * KB;
    assert!(!(in_c1 && in_c2)); // no contraction index in both blocks ⇒ correct PSUM-accumulate
}

// ── BRIDGE (split-K fix, lockdown proven BEFORE the code — corrects the "just use an offset" misconception):
// a K-block matmul reading the SHARED [N/64,K,64] o_proj weight MUST use stick stride = FULL K·64 (a strided
// read) + per-block row base b·KB·64; the NAIVE contiguous-block stride KB·64 reads the WRONG element for any
// output stick t≥1. This proves the shared-weight K-slice cannot be read with a plain [N/64,KB,64] block view
// (stick stride KB·64) — step 4 needs EITHER a re-staged [N/64,KB,64] block (proven by the next harness) OR a
// K·64-strided read into the shared tensor. Fail-first: the `naive_off != true_off` assert would fail if the
// contiguous-block stride were (wrongly) assumed valid. granite-3.3-2b o_proj: N=K=2048, KB=512 (B=4 blocks).
#[kani::proof]
fn ksplit_kslice_read_needs_full_weight_stick_stride() {
    const N: usize = 2048; // o_proj out = hidden (granite-3.3-2b)
    const K: usize = 2048; // o_proj contraction = hidden
    const KB: usize = 512; // per-block K extent (8 sticks); K/KB = 4 blocks
    const NST: usize = N / STK; // output sticks
    let b: usize = kani::any();
    kani::assume(b < K / KB); // block index
    let t: usize = kani::any();
    kani::assume(t < NST); // output stick
    let i: usize = kani::any();
    kani::assume(i < KB); // within-block contraction row
    let s: usize = kani::any();
    kani::assume(s < STK); // sub-stick col
    // TRUE element in the shared [N/64,K,64] weight: row (b*KB+i), col (t*STK+s).
    let true_off = t * (K * STK) + (b * KB + i) * STK + s;
    // CORRECT block-b read: per-output-stick stride = FULL K*STK, per-block row base = b*KB*STK.
    let correct_off = t * (K * STK) + b * KB * STK + i * STK + s;
    assert!(correct_off == true_off); // ⇒ block read MUST use stick stride K*STK + base b*KB*STK
    // NAIVE contiguous-block read (stick stride KB*STK, as a plain [N/64,KB,64] view) is WRONG for t≥1:
    let naive_off = t * (KB * STK) + b * KB * STK + i * STK + s;
    if t >= 1 {
        assert!(naive_off != true_off); // ⇒ a plain offset into the shared weight mis-reads the K-slice
    }
}

// ── BRIDGE (split-K fix, new lockdown proven BEFORE the code): block-weight K-slice offset ==
// block-matmul read ── The fp32-SFP-merge split-K emits, for K-block b, a matmul over the K-slice
// W[b*KB:(b+1)*KB, :] of the original [K,N] o_proj weight, staged as its OWN [KB,N] kernel (retiled
// [N/64, KB, 64]). Prove producer==consumer for the block (the retile_descriptor_walk analog per K-slice):
// (a) block-weight device row i is fed from ORIGINAL weight row (b*KB + i) (the K-slice gather, in-bounds),
// (b) the block matmul's read dev_off([KB,N],1,[i,j]) equals the block-weight retile placement. A wrong
// block offset (missing b*KB, or a KB-vs-K stride slip) mis-feeds the block ⇒ wrong partial ⇒ wrong merged
// sum. Fail-first-able: perturbing the stride breaks producer==consumer.
#[kani::proof]
fn ksplit_block_weight_kslice_offset_matches_read() {
    const N: usize = 4096; // o_proj out (= hidden)
    const KB: usize = 512; // per-block K extent (8 sticks); K=4096 ⇒ B=8 blocks
    const B: usize = 8;
    let b: usize = kani::any();
    kani::assume(b < B); // block index
    let i: usize = kani::any();
    kani::assume(i < KB); // within-block contraction row
    let j: usize = kani::any();
    kani::assume(j < N); // out col
    // (a) K-slice host gather: block b, device row i ← ORIGINAL weight row (b*KB + i), col j — in-bounds of [K,N].
    let host_orig_row = b * KB + i;
    assert!(host_orig_row < B * KB); // within the original [K=B*KB=4096, N] weight
    // (b) producer (block-weight staged as [KB,N] retiled [N/64,KB,64]) == consumer (block matmul dev_off read):
    let read = dev_off(&[KB, N], 1, &[i, j]);
    let (t, s) = (j / STK, j % STK);
    let device_flat = t * (KB * STK) + i * STK + s; // retile.rs device_flat with in=KB
    assert!(read == device_flat);
    assert!(read < KB * N); // stays in the block-weight footprint (no clobber of the next block)
}

// ── BRIDGE (split-K fix): the B block-partials MERGE to the full matmul ── out[j] = Σ_b partial_b[j] where
// partial_b[j] = Σ_{k in block b} A[k]·W[k][j]. Prove this equals the full reduction Σ_{k=0}^{K-1} A[k]·W[k][j]
// — i.e. summing the B block-partials (in fp32-exact SFP adds, modeled as exact integer adds) reconstructs
// the matmul, given the K-partition (disjoint+covering, proven by ksplit_contraction_partition...). Concrete
// distinct W keeps A·W LINEAR in the symbolic A (no symbolic×symbolic). A wrong block partition (gap/overlap)
// makes merged ≠ full ⇒ Kani finds it.
#[kani::proof]
fn ksplit_partial_merge_equals_full_matmul() {
    const K: usize = 8;
    const B: usize = 2;
    const KB: usize = K / B; // 2 blocks of 4 contraction terms
    let a: [i64; K] = [
        kani::any(),
        kani::any(),
        kani::any(),
        kani::any(),
        kani::any(),
        kani::any(),
        kani::any(),
        kani::any(),
    ];
    for x in 0..K {
        kani::assume(a[x] >= -1000 && a[x] <= 1000);
    }
    let w = |k: usize| -> i64 { (k as i64) * 7 + 3 }; // W[k][j] concrete distinct ⇒ a[k]·w(k) linear
    // full matmul: sequential reduction over all K.
    let mut full = 0i64;
    for k in 0..K {
        full += a[k] * w(k);
    }
    // split-K: B block-partials, merged (fp32-exact ⇒ exact integer add).
    let mut merged = 0i64;
    for b in 0..B {
        let mut partial = 0i64;
        for i in 0..KB {
            let k = b * KB + i;
            partial += a[k] * w(k);
        }
        merged += partial;
    }
    assert!(merged == full); // the merge of block-partials reconstructs the full matmul
}

// ── BRIDGE (layer blocked-f16 accuracy split, proof-first for down_proj K=8192) ── The DD2 fp16 matmul
// accumulates in SEN169_FP16 (bmm.ddl %ptsum_fp/%pesum, DDL-confirmed) — the 1.375-logit token-7 residual.
// The DD2-viable mitigation is to shorten the f16 accumulation DEPTH: split the contraction K into B blocks
// of KB=512 (8 fp16 sticks, stick-aligned), matmul each block, then f16-merge the B partials
// (`blocked_f16_sum_beats_sequential`). down_proj is K=8192 ⇒ B=16 blocks; o_proj/gate/up K=2048 ⇒ B=4. The
// ALGEBRAIC merge==full is K-independent (proven exact at K=8 by `ksplit_partial_merge_equals_full_matmul`);
// what's specific to the down_proj granularity is that the KB=512 block partition of K=B·512 is DISJOINT +
// COVERING (each contraction index in exactly one block). Prove it for all B≤32 (K up to 16384, covering
// 8192 and 2048). FAIL-FIRST: a mis-sized KB or an off-by-one block map leaves an index uncovered or double-
// counted ⇒ Kani finds it. This is the partition lockdown the layer blocked-f16 emit will require.
// ── BRIDGE (per-layer down_proj block-weight ADDRESSING — the symbolic-body staging lockdown) ── The layer
// blocked-f16 fix stages B=16 down_proj block weights PER LAYER. The body's per_layer/weight_stride scheme
// (lower_subtile…:5860-5878) places layer v's weights at `seg1_base + v·weight_stride`; a per-layer weight
// with layer-0 offset `off` lands at `off + v·weight_stride`. The B block weights get contiguous layer-0
// offsets `b·BLOCK_BYTES` within the per-layer region. Prove distinct (layer v, block b) → DISTINCT seg1
// address (no cross-layer/cross-block aliasing) — the invariant the worker staging + executor per-layer
// resolution require. Concrete granite-3.3-2b stride (121_643_008, from the bake log) + block bytes
// (down_proj [KB=512,N=2048] fp16). FAIL-FIRST: if the B blocks don't fit one per-layer region
// (B·BLOCK_BYTES > weight_stride, a block spilling into layer v+1) OR the offsets collide, Kani finds the
// (v,b) pair. Complements the disk-gather (`ksplit_block_weight_disk_gather_offset`) + device-retile
// (`ksplit_block_weight_kslice_offset_matches_read`) proofs: this is the PLACEMENT side for the per-layer case.
#[kani::proof]
fn per_layer_block_weight_addr_injective() {
    const WEIGHT_STRIDE: u64 = 121_643_008; // granite-3.3-2b per-layer weight stride (bake log)
    const BLOCK_BYTES: u64 = 512 * 2048 * 2; // down_proj block [KB=512, N=2048] fp16 = 2_097_152
    const B: u64 = 16; // K=8192 / KB=512
    const NUM_LAYERS: u64 = 40;
    assert!(B * BLOCK_BYTES <= WEIGHT_STRIDE); // all B blocks fit one per-layer region (no spill): 33_554_432 ≤ 121_643_008
    let (v1, b1): (u64, u64) = (kani::any(), kani::any());
    let (v2, b2): (u64, u64) = (kani::any(), kani::any());
    kani::assume(v1 < NUM_LAYERS && v2 < NUM_LAYERS && b1 < B && b2 < B);
    kani::assume(!(v1 == v2 && b1 == b2)); // distinct (layer, block)
    let addr1 = v1 * WEIGHT_STRIDE + b1 * BLOCK_BYTES;
    let addr2 = v2 * WEIGHT_STRIDE + b2 * BLOCK_BYTES;
    assert!(addr1 != addr2); // distinct (layer,block) → distinct seg1 address ⇒ no block-weight aliasing
}

// ── BRIDGE (localizes the MEASURED degenerate " instead" — the block-matmul A-K-slice flat offset) ──
// The layer blocked-f16 block matmul reads A's contraction slice `[*, b·KB:(b+1)·KB)` via `assemble_matmul_
// off(a_off = b·KB)`. But `a_off` is a FLAT element offset (superdsc_opspec::with_offset: base + off·word).
// In row-major A=[m,K], the TRUE element (row, b·KB+j) is at flat index `row·K + b·KB + j`; the flat-offset
// read of a `[m,KB]` tile from base `b·KB` places (row,j) at `b·KB + row·KB + j`. Prove these are equal
// IFF row==0 (single row) — so the block A-slice is correct for m==1 (decode) and WRONG for m>1 (prefill,
// K>KB), which is exactly the on-card degenerate output. This LOCKS: the LAYER_KSPLIT block matmul is only
// valid at m==1. FAIL-FIRST: asserting equality for ALL rows fails at row>0 (the bug); the m==1 restriction
// makes it hold. CBMC-tractable (bounded integers).
#[kani::proof]
fn ksplit_block_a_kslice_flat_offset_correct_iff_single_row() {
    const KB: u32 = 512;
    let m: u32 = kani::any();
    kani::assume(m >= 1 && m <= 8);
    let k: u32 = kani::any();
    kani::assume(k >= 1024 && k <= 8192 && k % KB == 0 && k > KB); // blocked (≥2 K-blocks)
    let b: u32 = kani::any();
    kani::assume(b < k / KB);
    let row: u32 = kani::any();
    kani::assume(row < m);
    let j: u32 = kani::any();
    kani::assume(j < KB);
    let intended = row * k + b * KB + j; // TRUE (row, b·KB+j) in row-major [m,K]
    let flat_read = b * KB + row * KB + j; // flat-offset [m,KB] tile from base b·KB
    if row == 0 {
        assert!(intended == flat_read); // m==1 (decode): the flat offset IS the col-slice — correct
    } else {
        assert!(intended != flat_read); // m>1 (prefill): reads the WRONG A element — the measured-degenerate bug
    }
}

#[kani::proof]
fn ksplit_downproj_block_partition_disjoint_covering() {
    const KB: u32 = 512; // per-block K extent = 8 fp16 sticks (512 % 64 == 0), the accuracy-split granularity
    let b_blocks: u32 = kani::any();
    kani::assume(b_blocks >= 1 && b_blocks <= 32); // K = b_blocks·512 ∈ {512 .. 16384}: covers down_proj 8192 (16), o_proj 2048 (4)
    let k = b_blocks * KB;
    let e: u32 = kani::any(); // a contraction index
    kani::assume(e < k);
    let owner = e / KB; // the block whose K-slice contains e
    // COVERING: e lies in its owner block's K-slice [owner·KB, (owner+1)·KB).
    assert!(e >= owner * KB && e < (owner + 1) * KB);
    assert!(owner < b_blocks); // a real block
    // DISJOINT: e lies in NO other block's slice.
    let other: u32 = kani::any();
    kani::assume(other < b_blocks && other != owner);
    assert!(!(e >= other * KB && e < (other + 1) * KB));
    // stick-alignment: each block is a whole number of 64-elem fp16 sticks (the matmul kernel requires it).
    assert!(KB % 64 == 0);
}

// ── BRIDGE (split-K fp32-merge, impl step 1): the Fp32 DataFormat's stick is the SAME 128-byte HBM stick
// as fp16 ── the fp32-SFP merge needs an Fp32 DataFormat (IEEE_FP32, 4 bytes/elem). Prove its stick is
// 128-byte-consistent (ELEMS_PER_STICK · WORD_LENGTH == 128, matching fp16's 64·2) so the 128-byte HBM
// alignment + tensor sizing hold — and that fp32 IS 4-byte (the crux the `_bf16` 2-byte marker shortcut
// missed). A wrong fp32 stick (e.g. 64 elems like fp16) ⇒ 64·4=256 ≠ 128 ⇒ mis-sized fp32 tensors.
#[kani::proof]
fn fp32_dataformat_stick_128byte_consistent() {
    assert!(Fp32::ELEMS_PER_STICK * Fp32::WORD_LENGTH == 128); // fp32: 32·4 = 128 B stick
    assert!(Fp16::ELEMS_PER_STICK * Fp16::WORD_LENGTH == 128); // fp16: 64·2 = 128 B — same HBM stick
    assert!(Fp32::WORD_LENGTH == 4); // fp32 is 4-byte (NOT 2 like fp16/bf16 — the marker-path crux)
    assert!(Fp32::ELEMS_PER_STICK == 32 && Fp16::ELEMS_PER_STICK == 64);
}

// ── BRIDGE (split-K fp32-merge, impl step 2): fp32 partial tensor addressing (4-byte, single-row) ── the
// fp32-SFP merge partials are [1,N] (decode m=1). Prove: (a) a single-row tensor is contiguous for fp32's
// stick (32) exactly as for fp16's (64) — the stick-group term vanishes, so `dev_off`-style [1,N] element c
// lands at element-offset c; (b) the BYTE offset for an fp32 element is c·4 (Fp32::WORD_LENGTH), NOT c·2 —
// so the emitter MUST use the tensor's own word length for fp32 tensors, not the hardcoded Fp16::WORD_LENGTH
// (=2). Using 2 for an fp32 tensor halves every offset ⇒ scrambled merge. Locks the fp32 addressing the
// 4-byte emitter path must satisfy.
#[kani::proof]
fn fp32_partial_addressing_4byte_single_row() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 4096);
    let c: usize = kani::any();
    kani::assume(c < n);
    let stk32 = Fp32::ELEMS_PER_STICK as usize; // 32
    let stk16 = Fp16::ELEMS_PER_STICK as usize; // 64
    // (a) single-row element offset is the identity for BOTH stick sizes (single row ⇒ stick term vanishes):
    assert!((c / stk32) * stk32 + (c % stk32) == c);
    assert!((c / stk16) * stk16 + (c % stk16) == c);
    // (b) fp32 byte offset = element · 4, distinct from fp16's element · 2 (except element 0):
    let byte_fp32 = c * Fp32::WORD_LENGTH as usize;
    assert!(byte_fp32 == c * 4);
    assert!(c == 0 || byte_fp32 != c * Fp16::WORD_LENGTH as usize);
}

// ── BRIDGE (packed-int8 residency, SENINT8 DataFormat): the SenInt8 stick is the SAME 128-byte HBM
// stick as fp16/fp32, but 1 byte/elem (128 elems) ── torch-spyre `DataFormats::SENINT8`
// (csrc/module.cpp:303); int8 stick = 128 elems / 128 B per `work_division_planning.md:118`
// (`device_dtype.elems_per_stick()`). Prove ELEMS_PER_STICK·WORD_LENGTH == 128 (128-byte HBM
// alignment holds), WORD_LENGTH == 1 (the packed byte — NOT 2, which would be the REJECTED
// dequant-on-load f16-dense footprint), ELEMS_PER_STICK == 128, and the DeepTools `dataFormat_`
// string. FAIL-FIRST: WORD_LENGTH==2 (dequant-on-load) or a wrong stick/NAME breaks an assert.
#[kani::proof]
fn senint8_dataformat_stick_128byte_consistent() {
    assert!(SenInt8::ELEMS_PER_STICK * SenInt8::WORD_LENGTH == 128); // int8: 128·1 = 128 B stick
    assert!(SenInt8::WORD_LENGTH == 1); // 1 byte/elem — the packed residency (NOT 2 = dequant-on-load)
    assert!(SenInt8::ELEMS_PER_STICK == 128); // 128 int8 elems = one 128-byte HBM stick
    assert!(SenInt8::NAME == "SENINT8"); // torch-spyre DataFormats::SENINT8 (module.cpp:303)
    // Same 128-byte HBM stick as fp16/fp32 — the alignment the tensor-sizing / re-tile math relies on.
    assert!(Fp16::ELEMS_PER_STICK * Fp16::WORD_LENGTH == 128);
    assert!(Fp32::ELEMS_PER_STICK * Fp32::WORD_LENGTH == 128);
}

// ── BRIDGE (packed-int8 residency = HALF the f16 footprint — the anti-dequant-on-load guard): the
// WHOLE POINT of int8 quant is the decode HBM-bandwidth win. Prove that a [K,N] weight staged as
// SENINT8 occupies EXACTLY HALF the bytes of the same weight staged f16-dense — i.e. it is 1 byte/elem
// (K·N bytes), NOT f16-dense (2·K·N). `device_bytes = Π(device_size)·WORD_LENGTH`, so this locks that
// the DeviceTileLayout<SenInt8> descriptor sizes the resident weight at the packed byte count. If the
// int8 weight were dequant-on-load (materialized f16-dense in HBM, the explicitly REJECTED reward-hack),
// `device_bytes` would equal the f16 count and this proof goes RED. FAIL-FIRST: SenInt8::WORD_LENGTH==2
// makes `int8_bytes * 2 == fp16_bytes` false. N is a whole SenInt8 stick (128) ⇒ also a whole fp16 stick.
#[kani::proof]
fn senint8_resident_is_half_fp16_footprint() {
    let k: u64 = kani::any(); // in-features (non-stick dim) — any positive extent
    kani::assume(k >= 1 && k <= 256);
    let nb: u64 = kani::any(); // out-features in units of the 128-int8 stick
    kani::assume(nb >= 1 && nb <= 8);
    let n = nb * 128; // multiple of 128 ⇒ valid SenInt8 stick AND valid fp16 (64) stick

    let dt8 = DeviceTileLayout::<SenInt8>::new(&["in", "out"], "out", &[k, n]).unwrap();
    let dt16 = DeviceTileLayout::<Fp16>::new(&["in", "out"], "out", &[k, n]).unwrap();

    let int8_bytes = dt8.device_bytes();
    let fp16_bytes = dt16.device_bytes();

    // (1) int8 is 1 byte/elem: the resident packed weight is EXACTLY K·N bytes (no f16 dense copy).
    assert!(int8_bytes == k * n);
    // (2) f16-dense is 2·K·N — the footprint dequant-on-load would leave in HBM.
    assert!(fp16_bytes == 2 * k * n);
    // (3) THE WIN: int8 residency is exactly HALF (÷2 bytes ⇒ up to 2× decode tok/s), and strictly less.
    assert!(int8_bytes * 2 == fp16_bytes);
    assert!(int8_bytes < fp16_bytes);
}

// ── ISLAND (SEN143_FP8 stick geometry — the fp8 W8A8 residency the `matmulfp8` kernel reads): fp8 E4M3
// is torch-spyre `DataFormats::SEN143_FP8` (DDL `%type_fp8`; IBM PR #2401 89ac601). fp8 stick = 128 elems
// / 128 B, the SAME 128-byte HBM stick as int8/fp16/fp32. Prove ELEMS_PER_STICK·WORD_LENGTH == 128,
// WORD_LENGTH == 1 (packed byte — NOT 2 = the REJECTED dequant-on-load f16-dense), ELEMS_PER_STICK == 128,
// and the DeepTools `dataFormat_` string. FAIL-FIRST: WORD_LENGTH==2 or a wrong stick/NAME breaks an assert.
#[kani::proof]
fn fp8_dataformat_stick_128byte_consistent() {
    assert!(Fp8::ELEMS_PER_STICK * Fp8::WORD_LENGTH == 128); // fp8: 128·1 = 128 B stick
    assert!(Fp8::WORD_LENGTH == 1); // 1 byte/elem — packed fp8 residency (NOT 2 = dequant-on-load)
    assert!(Fp8::ELEMS_PER_STICK == 128); // 128 fp8 elems = one 128-byte HBM stick
    assert!(Fp8::NAME == "SEN143_FP8"); // torch-spyre DataFormats::SEN143_FP8 (DDL %type_fp8, PR #2401)
}

// ── BRIDGE (packed-fp8 residency = HALF the f16 footprint — the fp8 twin of the anti-dequant-on-load
// guard `senint8_resident_is_half_fp16_footprint`): the point of fp8 W8A8 is the ÷2 decode HBM-bandwidth
// win. Prove a [K,N] weight staged SEN143_FP8 occupies EXACTLY half the bytes of f16-dense (1 byte/elem,
// K·N, NOT 2·K·N). If fp8 were dequant-on-load (materialized f16-dense in HBM, the REJECTED reward-hack),
// `device_bytes` would equal the f16 count and this goes RED. FAIL-FIRST: Fp8::WORD_LENGTH==2 makes
// `fp8_bytes*2 == fp16_bytes` false. N is a whole SEN143_FP8 stick (128) ⇒ also a whole fp16 (64) stick.
// CONCRETE granite Linear shapes (a symbolic k·n multiply unbounds CBMC — same lesson as the matmul-split
// proof; concrete shapes evaluate `device_bytes` directly, ~instant, per the plan_regions_disjoint style).
fn fp8_resident_half_fp16_for(k: u64, n: u64) {
    // n is a multiple of 128 (fp8 stick) ⇒ also a whole fp16 (64) stick.
    let dt8 = DeviceTileLayout::<Fp8>::new(&["in", "out"], "out", &[k, n]).unwrap();
    let dt16 = DeviceTileLayout::<Fp16>::new(&["in", "out"], "out", &[k, n]).unwrap();
    let fp8_bytes = dt8.device_bytes();
    let fp16_bytes = dt16.device_bytes();
    // (1) fp8 is 1 byte/elem: resident packed weight = EXACTLY K·N bytes (no f16-dense copy).
    assert!(fp8_bytes == k * n);
    // (2) f16-dense = 2·K·N — the footprint dequant-on-load would leave in HBM.
    assert!(fp16_bytes == 2 * k * n);
    // (3) THE WIN: fp8 residency is exactly HALF (÷2 bytes ⇒ up to 2× decode tok/s), strictly less.
    assert!(fp8_bytes * 2 == fp16_bytes);
    assert!(fp8_bytes < fp16_bytes);
}
#[kani::proof]
fn fp8_resident_half_qo_proj() {
    fp8_resident_half_fp16_for(2048, 2048); // granite q/o_proj: K=hidden, N=hidden
}
#[kani::proof]
fn fp8_resident_half_down_proj() {
    fp8_resident_half_fp16_for(8192, 2048); // granite down_proj: K=intermediate, N=hidden
}
#[kani::proof]
fn fp8_resident_half_gate_up() {
    fp8_resident_half_fp16_for(2048, 8192); // granite gate/up_proj: K=hidden, N=intermediate
}

// ── BRIDGE (fp8 W8A8 output-dequant scale axes — the per-channel-W / per-token-A contract): dequant is
// `out[m,n] = raw[m,n]·w_scale[n]·a_scale[m]`, w_scale PER-CHANNEL (len n_channels), a_scale PER-TOKEN
// (len m_rows). Prove (1) the SHAPE guard accepts the correct axes (w=N, a=M) and REJECTS the swap (w=M,
// a=N) whenever M≠N — a mis-axised scale would dequant the wrong axis; (2) the (m,n)→(w idx, a idx) map is
// COVERING + in-range for every output element (w idx = n < N, a idx = m < M) — no element mis-scaled or
// out-of-range. Comparison-only (no k·n multiply) ⇒ symbolic is CBMC-tractable over granite-scale extents.
// FAIL-FIRST: an axis-swapped `scales_well_shaped` or a `scale_indices` that returned (m,n) breaks an assert.
#[kani::proof]
fn fp8_w8a8_dequant_scale_indices_covering() {
    let m_rows: u32 = kani::any();
    let n_channels: u32 = kani::any();
    kani::assume(m_rows >= 1 && m_rows <= 4096);
    kani::assume(n_channels >= 1 && n_channels <= 4096);
    // (1) correct axes (w per-channel=N, a per-token=M) accepted; swapped rejected unless M==N.
    assert!(Fp8W8A8Dequant::scales_well_shaped(
        m_rows, n_channels, n_channels, m_rows
    ));
    if m_rows != n_channels {
        assert!(!Fp8W8A8Dequant::scales_well_shaped(
            m_rows, n_channels, m_rows, n_channels
        ));
    }
    // (2) covering + in-range: every output (m,n) maps to (w idx = n < N, a idx = m < M).
    let m: u32 = kani::any();
    let n: u32 = kani::any();
    kani::assume(m < m_rows && n < n_channels);
    let (wi, ai) = Fp8W8A8Dequant::scale_indices(m, n);
    assert!(wi < n_channels); // w_scale[n] in range (PER-CHANNEL, indexed by output column)
    assert!(ai < m_rows); // a_scale[m] in range (PER-TOKEN, indexed by row)
}

// ── BRIDGE (SENINT8 per-group dequant scale index is a covering partition): every int8 code is
// dequantized by EXACTLY ONE per-group f16 scale (`w[k] = scale[k/group_size]·q[k]`, symmetric affine,
// no zero-point — grounded in scratchy CUDA `affine_dequant_b8_bf16` "out = scale·q + bias", bias 0).
// Prove that for a `PackedInt8Scale(k_elems, group_size)` with group_size | k_elems: for EVERY code
// k < k_elems, its scale index gi = k/group_size lies in [0, groups) AND the group's contiguous range
// [gi·g, (gi+1)·g) contains k — so no code is left unscaled and no code picks up an out-of-range scale.
// A wrong index math (off-by-one group, or floor→ceil) breaks the range containment ⇒ Kani finds a k.
// FAIL-FIRST: a `scale_index_for` that mis-buckets a code violates the containment assert.
#[kani::proof]
fn senint8_dequant_scale_index_is_covering_partition() {
    let g: u32 = kani::any();
    kani::assume(g >= 1 && g <= 8);
    let groups: u32 = kani::any();
    kani::assume(groups >= 1 && groups <= 8);
    let k_elems = g * groups; // g | k_elems by construction (per-channel = groups·? ; per-group = g<k)
    let plan = PackedInt8Scale::new(k_elems, NonZeroU32::new(g).unwrap()).unwrap();
    assert!(plan.groups() == groups); // scale count = k_elems / group_size

    let k: u32 = kani::any();
    kani::assume(k < k_elems);
    let gi = plan.scale_index_for(k);
    // (a) the scale index is a real group (no orphan code → out-of-range scale).
    assert!(gi < groups);
    // (b) code k falls in group gi's CONTIGUOUS range [gi·g, (gi+1)·g) — the exact-partition claim.
    assert!(gi * g <= k && k < (gi + 1) * g);
    // (c) the map is deterministic single-valued: no OTHER group contains k.
    let other: u32 = kani::any();
    kani::assume(other < groups && other != gi);
    assert!(!(other * g <= k && k < (other + 1) * g));
}

// ── BRIDGE (GroupDivides guard — a scale shape that does NOT tile the codes is a build Err, not a
// runtime fault): `PackedInt8Scale::new` is the SOLE constructor and rejects group_size ∤ k_elems (a
// partial final group would leave int8 codes with no scale). Prove: divides ⇒ Ok with groups=k/g;
// does-not-divide ⇒ Err (no PackedInt8Scale is constructible). This is the compile-time TYPE guard
// (sealed ctor returning a build-surfaced Err + NonZeroU32 group size) — never a runtime assert.
// FAIL-FIRST: a ctor that silently floored (accepted an indivisible group) would make the Err assert RED.
#[kani::proof]
fn senint8_packed_scale_rejects_indivisible_group() {
    let k_elems: u32 = kani::any();
    kani::assume(k_elems >= 1 && k_elems <= 64);
    let g: u32 = kani::any();
    kani::assume(g >= 1 && g <= 64);
    let gz = NonZeroU32::new(g).unwrap();
    // The guard's pure decision predicate == EXACT divisibility. `PackedInt8Scale::new` returns
    // `Ok` iff `group_divides` holds and `Err` otherwise (it is the SOLE branch condition), so
    // proving the predicate == divisibility proves "indivisible ⇒ unconstructible" — WITHOUT
    // forcing the model checker through the `Err`-string `format!` (the I/O leaf, factored out
    // exactly like the transcendental proofs skip the float-ALU leaf; a reachable `format!` Err
    // path is what fails the model checker, not the guard logic).
    assert!(PackedInt8Scale::group_divides(k_elems, gz) == (k_elems % g == 0));
    // Constructive side (divisible ⇒ Ok with the right shape) — the OK path never formats.
    if k_elems % g == 0 {
        let plan = PackedInt8Scale::new(k_elems, gz).unwrap();
        assert!(plan.groups() == k_elems / g);
        assert!(plan.group_size() == g);
        assert!(plan.k_elems() == k_elems);
    }
}

// ── A POOL SEATS ONLY THE SLOTS IT HAS PAGES FOR, AND A ROW IS NOTHING BUT A SLOT ID ────────────
//
// ⛔ THIS REPLACES A PROOF OF THE STRIPE LAW, WHICH IS NOW DELETED CODE. It asserted
// `physical(lp) = row * pages_per_row + lp` — affine in the row, disjoint, inside the pool. `PageRun` and
// `PoolSplit::{run, pages_per_row, positions_per_row}` are GONE: pages come from a free list, so a page
// address is a BLOCK TABLE ENTRY and no function derives one from a row.
//
// ⭐ AND THE SAME `cfg(kani)` TRAP AS BEFORE APPLIES — a proof naming deleted API cannot compile, and NO
// ordinary build says so, which is how three pool proofs sat dead through the whole rebuild
// (see the note above `no_rung_a_live_batch_selects_can_overshoot_the_stripes`). Every commit that deletes
// pool API must grep THIS FILE in the same commit.
//
// What is left to prove is the whole of what `PoolSplit` still claims: it seats `rows` launch slots, it
// only exists when the pool can give each live request a page, and `row()` mints nothing past `rows`.
#[kani::proof]
fn a_pool_seats_only_the_slots_it_has_pages_for() {
    // ⏱ BOUNDS KEPT TIGHT ON PURPOSE. A first version asserted `all_rows().count() == rows` and left
    // `r: u32` unconstrained; CBMC unrolled the iterator symbolically over a full u32 and the harness
    // TIMED OUT at 1200 s. A proof that times out is a proof that is not running — the same failure class
    // as the three that sat dead through the rebuild, just with a different disguise. The iterator's
    // element count is plumbing; the INVARIANT is which rows can be minted.
    let pool: u32 = kani::any();
    kani::assume(pool <= 40);
    let n: u32 = kani::any();
    kani::assume(n <= 32);
    let Some(rows) = PoolRows::of_rung(n) else {
        assert!(!PagedKvPool::BATCH_RUNGS.contains(&n));
        return;
    };
    let rows_n = rows.get().get();
    match PagedKvPool::split_pool(pool, rows) {
        None => {
            // THE ONE REFUSAL, and it is now an ADMISSION bound, not a capacity one: every live request
            // needs at least one page, so a pool cannot seat more of them than it has pages.
            assert!(pool < rows_n);
        }
        Some(split) => {
            assert!(pool >= rows_n);
            assert!(split.rows().get() == rows_n);
            // A ROW IS A SLOT ID AND NOTHING ELSE — mintable exactly inside the seated width, and carrying
            // no page term at all, because no function exists any more that would give it one.
            let r: u32 = kani::any();
            kani::assume(r <= 64);
            assert!(split.row(r).is_some() == (r < rows_n));
        }
    }
}

// ── A STRIPE COUNT IS A LADDER RUNG, SO NO RUNG A LIVE BATCH SELECTS CAN OVERSHOOT THE POOL ─────
//
// THE SAFETY PROPERTY BEHIND `PoolRows`, and the reason its constructors are ladder members only.
//
// `LaunchPages::Affine` strides slots `0..seqs` across the stripes arithmetically, so the rung that
// RUNS must never be wider than the pool has rows. Two facts bound it: admission cannot exceed the
// rows (`free_row` runs out), and `decode_rung_for` picks the SMALLEST rung `>= live`. Neither
// mentions the stripe count, so the bound holds only if the stripe count is itself a rung.
//
// FAIL-FIRST: with an arbitrary stripe count the assertion is RED, and Kani exhibits the pair —
// 5 stripes with 5 live requests selects the 8-wide rung, which strides slots 5, 6 and 7 across
// stripes that do not exist and reads past the end of the pool. That is a device read of unowned
// memory, so it is exactly the class this crate exists to make unconstructable rather than to catch.
#[kani::proof]
fn no_rung_a_live_batch_selects_can_overshoot_the_stripes() {
    let n: u32 = kani::any();
    kani::assume(n <= 32);
    let Some(rows) = PoolRows::of_rung(n) else {
        return;
    };
    let rows_n = rows.get().get();

    // Any batch the pool can admit at all.
    let live: u32 = kani::any();
    kani::assume(live >= 1 && live <= rows_n);

    // `decode_rung_for`, in the only terms that matter: the smallest ladder rung that holds `live`.
    let selected = PagedKvPool::BATCH_RUNGS.into_iter().find(|r| *r >= live);
    let selected = selected.expect("the widest rung holds any admissible batch");

    // THE ASSERTION: the launch never addresses a stripe the pool does not have.
    assert!(selected <= rows_n);
    // And a batch of ONE still runs the narrowest rung — which is why a 1-stripe pool is not legal
    // and `PoolRows` cannot name one.
    assert!(selected >= PagedKvPool::NARROWEST_BATCH_RUNG);
}

// ── THE ADMISSION WIDTH IS A LADDER RUNG THAT HOLDS THE ADMITTED REQUESTS, NEVER FEWER ─────────
//
// ⛔ REPLACES a proof that it is the WIDEST rung. That was true and expensive: capacity does not care
// (a free list lets one request reach the whole pool at any width), but `PoolPartition` reserves
// `rows * HOLE_PAGES_PER_ROW + 1` pages for the holes a launch's rows write into, so the widest rung
// reserved 65 of a 136-page pool against launches a `--max-num-seqs 4` run cannot perform.
//
// THE TWO PROPERTIES THAT MATTER, for EVERY admission cap:
//   1. the result is a ladder rung — anything else has a live count whose selected rung addresses slots
//      the pool never seated;
//   2. it is never NARROWER than the cap (rounding DOWN is the corruption; rounding UP only wastes),
//      except where the cap exceeds the widest rung the ladder bakes, which no launch can express.
#[kani::proof]
fn the_admission_width_is_a_rung_that_holds_the_admitted_requests() {
    let want: u32 = kani::any();
    kani::assume(want > 0 && want <= 4096);
    let Some(req) = AdmittedRequests::new(want as usize) else {
        return;
    };
    let got = PoolRows::for_admission(req);
    assert!(PagedKvPool::BATCH_RUNGS.contains(&got.get().get()));
    if want <= PagedKvPool::WIDEST_BATCH_RUNG {
        assert!(got.get().get() >= want);
        // And it is the SMALLEST such rung — otherwise the reserve is bigger than the run can use.
        for r in PoolRows::all() {
            assert!(!(r.get().get() >= want && r.get().get() < got.get().get()));
        }
    } else {
        assert!(got.get().get() == PagedKvPool::WIDEST_BATCH_RUNG);
    }
}
