// SPDX-License-Identifier: Apache-2.0
// ⛔ RUN THIS FILE WITH `SCRATCHY_PLAN_ONLY_BAKE=1` when the surrounding sweep sets it —
// this file itself runs no bake (it binds ONE hand-minted program, no door, no lowering),
// but it lives in the spyre-emu suite that does.
#![cfg(feature = "spyre-emu")]
//! THE COMPOSITE ARGMAX'S NUMERIC PARITY — the KTIR program and the HOST argmax are two
//! expressions of ONE computation; this file executes the walk-mint `IRFunction` through
//! the REAL serving path (`ktir_groups` → `SpyreSession::new_multi` → `run_step`, the
//! emulator's fused driver, not the per-node interpreter) and checks the row it gathers
//! against `manifest::argmax` (#631's first-max) on the SAME logits.
//!
//! The opus ruling this file exists for: the parity gate has a structural half (the door
//! test — the descriptors the door emits are byte-identical to the direct assembly's) and
//! a NUMERIC half, and this is it. The first-max law is pinned the way #631's own unit
//! pins pin it, with crafted equal maxima:
//!
//!   • a UNIQUE maximum (with negative values in the row — the running max must start
//!     below them, as both the device's fp16-floor init and `manifest::argmax`'s do);
//!   • TIED MAXIMA at i < j — the EARLIER index must win (the ramp `V − iota` makes the
//!     first maximum's weight unique by construction; this is that arithmetic executed);
//!   • an ALL-EQUAL row — every lane ties, index 0 must win.
//!
//! The program under test is `mint_argmax_node`'s — the same construction the rolled walk
//! mints (the walk test pins the mint fires and binds the graph's own slots; this file
//! pins the FUNC's numerics). The func is SELF-CONTAINED: the ramp and the scalar V are
//! computed in-func (`linalg.generic` over the seed, the V immediate), so the reserved-tid
//! consts the DESCRIPTOR chain reads are not staged here — two expressions of one
//! computation, and this is the program's own.
//!
//! The logits and table values are pinned through the emulator's own f16 codec (the
//! single rounding `Tile::compute` applies), and the host oracle runs over the SAME
//! images — the comparison is exact, never within-an-epsilon.

use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::emit::argmax::{TiedLmHeadEmbedding, mint_argmax_node};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{FoldGrouping, ktir_groups};
use scratchy_target_spyre::manifest;
use scratchy_target_spyre::runner::SpyreSession;

/// Stick-legal small geometry — the door and walk tests' own (two fp16 sticks each way;
/// the probes hold granite's 49152-wide geometry on the assembly side).
const VOCAB: usize = 128;
const HIDDEN: usize = 128;
/// The two PAD-LANE vocabs, granite's own shapes scaled: 131 = 2·64 + 3 (61 pad lanes in
/// the 192-wide staging — granite-3.1's 49155 = 768·64 + 3) and 135 = 2·64 + 7 (57 pad
/// lanes — granite-3.3-2b-instruct's 49159 = 768·64 + 7). Two widths, because a preset
/// name would have missed that the pad count is the only thing that differs.
const PAD_VOCAB_A: usize = 131;
const PAD_VOCAB_B: usize = 135;
/// The three boundary tids — distinct, as a real bundle's are. The table rides the
/// resident-tensor channel (the session's `weights`), the logits the per-step source
/// channel (which narrows to f16 — the images are pinned first), the row is the output.
const LOGITS_TID: u64 = 2;
const TABLE_TID: usize = 1;
const ROW_TID: u64 = 3;

/// The staging width a vocab pads to — the lm_head's own out-axis law.
fn stick_pad(vocab: usize) -> usize {
    vocab.div_ceil(64) * 64
}

/// f16 round-trip through the emulator's storage: the value an f16 element holds after
/// `Tile::compute`'s single rounding — the moe chain test's own helper.
fn f16(v: f32) -> f32 {
    let bits = ktir_emulator::codec::encode(&[v], ktir_emulator::dtypes::DType::F16);
    ktir_emulator::codec::decode(&bits, 1, ktir_emulator::dtypes::DType::F16)[0]
}

/// The tied embedding table, row-major `[vocab + 1, hidden]` — the orientation the mint's
/// table VIEW states (out-major: token v's embedding is out-slot v), with the LAST row
/// the ZERO SENTINEL the `idx = V` gather reads when nothing real won (the staging's
/// zero-filled shadow, mirrored here). Row `v < V` is `(131·v + 17·h) mod 251 − 125`:
/// every value f16-exact, and rows pairwise DISTINCT (131 is invertible mod 251, so
/// `v ≠ v'` differ at every lane — a wrong gather index cannot pass as a right one, and
/// neither can the sentinel pass as token 0's row).
fn table(vocab: usize) -> Vec<f32> {
    let mut t = Vec::with_capacity((vocab + 1) * HIDDEN);
    for v in 0..vocab {
        for h in 0..HIDDEN {
            let x = (131 * v + 17 * h) % 251;
            t.push(f16(x as f32 - 125.0));
        }
    }
    t.resize((vocab + 1) * HIDDEN, 0.0);
    t
}

/// Execute the walk-mint composite over one logits row and return the gathered row. The
/// logits slice is the FULL stick-padded staging row (the caller owns the pad lanes);
/// the table carries its sentinel row.
fn composite_row(vocab: usize, logits: &[f32], table: &[f32]) -> Vec<f32> {
    let tied = TiedLmHeadEmbedding::from_tied_config(true).expect("granite ties");
    let k = mint_argmax_node(
        vocab as u32,
        HIDDEN as u32,
        LOGITS_TID as u32,
        TABLE_TID as u32,
        ROW_TID as u32,
        tied,
    );
    // The walk's own mint form: KTIR-first, `op: None`, the program is the whole op.
    let mut e = EmittedOp::bare(k.func.name.to_string());
    e.ktir = Some(k);
    let ops = vec![e];
    let groups = ktir_groups(&ops, FoldGrouping::Split).expect("bind the composite's program");
    // The table as ONE resident f16 tensor, row-major [vocab + 1, hidden] LE — the
    // sentinel row included, zeros, exactly what the staging's shadow holds.
    let table_bytes: Vec<u8> = table
        .iter()
        .flat_map(|v| ktir_emulator::codec::encode(&[*v], ktir_emulator::dtypes::DType::F16))
        .collect();
    let weights = vec![(
        TABLE_TID,
        table_bytes,
        scratchy_tensors::DType::F16,
        vec![vocab + 1, HIDDEN],
    )];
    let mut session = SpyreSession::new_multi(&[(&groups, &[ROW_TID])], weights)
        .expect("build the composite's session");
    let out = session
        .run_step(
            0,
            vec![(LOGITS_TID, logits.to_vec(), vec![1, stick_pad(vocab)])],
            &[(ROW_TID, 0)],
        )
        .expect("run the composite");
    out[&ROW_TID].clone()
}

/// One parity leg: run the composite, run `manifest::argmax` over the SAME REAL images
/// (the host never sees the pad lanes — its row is the unpadded one), and demand the
/// gathered row IS the oracle's index's row — the index itself is never read back (the
/// composite's output is the ROW), so a wrong index can only pass by gathering a row
/// that happens to equal the right one, and the table's rows are pairwise distinct.
fn assert_parity(what: &str, vocab: usize, logits: &[f32], table: &[f32]) {
    let got = composite_row(vocab, logits, table);
    let want_idx = manifest::argmax(&logits[..vocab]);
    let want = &table[want_idx * HIDDEN..(want_idx + 1) * HIDDEN];
    assert_eq!(
        got.len(),
        HIDDEN,
        "{what}: the composite writes one [1, hidden] row"
    );
    for (h, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g, w,
            "{what}: lane {h} — the gathered row is token {want_idx}'s"
        );
    }
}

/// A UNIQUE maximum, with negative values in the row — both oracles start their running
/// max below the data (the device at the fp16 floor, the host at −65504), and a negative
/// row would expose an init above the data.
#[test]
fn a_unique_maximum_gathers_its_own_row() {
    let table = table(VOCAB);
    let mut logits: Vec<f32> = (-100..(VOCAB as i64 - 100))
        .map(|i| f16(i as f32))
        .collect();
    logits[41] = f16(500.0); // strictly above every other lane
    assert_parity("the unique maximum", VOCAB, &logits, &table);
}

/// TIED MAXIMA at i < j — the EARLIER index wins. This is the ramp arithmetic's whole
/// point: `w = (logits == m) · (V − iota)` makes the first maximum's weight unique, so
/// the second max lands on the earliest lane — #631's first-max law made structural, and
/// here EXECUTED, exactly the way `manifest::argmax`'s strict `>` keeps index 7.
#[test]
fn tied_maxima_take_the_earlier_index() {
    let table = table(VOCAB);
    let mut logits: Vec<f32> = (0..VOCAB).map(|i| f16((i % 13) as f32)).collect();
    logits[7] = f16(99.0);
    logits[90] = f16(99.0); // the tie: equal in fp16, and 7 < 90
    assert_eq!(
        manifest::argmax(&logits),
        7,
        "the host oracle agrees: first wins"
    );
    assert_parity("the tied maximum", VOCAB, &logits, &table);
}

/// An ALL-EQUAL row — every lane ties with every other, index 0 must win. The mask is
/// all-ones, so `w` IS the ramp and its maximum is `V` at lane 0, giving `idx = V − V = 0`.
#[test]
fn an_all_equal_row_gathers_token_zero() {
    let table = table(VOCAB);
    let logits: Vec<f32> = (0..VOCAB).map(|_| f16(5.0)).collect();
    assert_eq!(manifest::argmax(&logits), 0);
    assert_parity("the all-equal row", VOCAB, &logits, &table);
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
//  THE PAD-LANE FORM — opus's ruling, executed: ONE staged bias (0 real / −65504 pad)
//  before the max, pinned at BOTH granite pad widths (61 lanes and 57).
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐⭐⭐ OPUS'S PINNED TEST — garbage in the pad lanes EXCEEDING the real maximum must
/// still pick a real lane. Without the bias, the pad lanes (unwritten staging, whatever
/// the buffer held) would own the max and the gather would answer a pad row; with it,
/// a pad lane holds `garbage − 65504` — at or below the fp16 floor for any garbage the
/// buffer can carry — and the real maximum wins. granite-3.1's own shape scaled: 61 pad
/// lanes in the 192-wide staging, the garbage pinned at f16's max finite (the worst a
/// pad lane can hold short of inf).
#[test]
fn garbage_pad_lanes_exceeding_the_real_max_still_pick_a_real_lane() {
    let vocab = PAD_VOCAB_A;
    let table = table(vocab);
    let mut logits: Vec<f32> = (-100..(vocab as i64 - 100))
        .map(|i| f16(i as f32))
        .collect();
    logits[41] = f16(500.0); // the real maximum
    logits.resize(stick_pad(vocab), f16(65504.0)); // 61 pad lanes of the WORST garbage
    assert_parity("the garbage-padded row", vocab, &logits, &table);
    // The control: the host oracle over the REAL lanes picks 41, so the leg above demands
    // the composite does too — against pads that beat 500 by two orders of magnitude.
    assert_eq!(manifest::argmax(&logits[..vocab]), 41);
}

/// The SAME form at the SECOND pad width — 135 = 2·64 + 7, granite-3.3-2b-instruct's
/// 49159 scaled: 57 pad lanes instead of 61. TIED maxima on the real lanes, garbage
/// pads: the earlier index still wins — first-max semantics are the bias's OWN promise
/// ("unchanged"), and this leg pins that the promise holds at a second pad count (the
/// structural difference a preset-name list would have missed).
#[test]
fn the_second_pad_width_keeps_first_max_semantics() {
    let vocab = PAD_VOCAB_B;
    let table = table(vocab);
    let mut logits: Vec<f32> = (0..vocab).map(|i| f16((i % 13) as f32)).collect();
    logits[7] = f16(99.0);
    logits[90] = f16(99.0); // the tie: equal in fp16, and 7 < 90
    assert_eq!(manifest::argmax(&logits), 7, "the host oracle: first wins");
    logits.resize(stick_pad(vocab), f16(-30000.0)); // 57 pad lanes of garbage
    assert_parity("the tied maximum at 57 pad lanes", vocab, &logits, &table);
}

/// ⭐ THE NaN SENTINEL, executed — an all-NaN REAL row (the host's own refusal shape:
/// `manifest::argmax` floors to token 0) gathers the ZERO row at index V, both forms.
/// The device's answer is the sentinel row, which the executor REFUSES (`idx ≥ vocab`)
/// — the host answers token 0; that host/device divergence is the filed follow-up, and
/// this test pins the device's half: the gathered row is the sentinel, NOT token 0's
/// row (the table's rows are pairwise distinct, so zeros cannot pass as token 0).
#[test]
fn an_all_nan_row_lands_on_the_sentinel_row_not_token_zero() {
    // The pad-lane form, the harsher leg: NaN real lanes AND finite garbage pads, so the
    // biased max is a PAD value — only the clamped ramp's zero weight keeps the index at
    // the sentinel instead of walking past it.
    let vocab = PAD_VOCAB_A;
    let tab = table(vocab);
    let mut logits: Vec<f32> = vec![f32::NAN; vocab];
    logits.resize(stick_pad(vocab), f16(30000.0));
    let got = composite_row(vocab, &logits, &tab);
    assert_eq!(
        got,
        vec![0.0; HIDDEN],
        "the all-NaN row gathers the ZERO sentinel"
    );
    assert_ne!(
        got,
        tab[..HIDDEN].to_vec(),
        "the sentinel is NOT token 0's row — the divergence the follow-up issue owns"
    );
    assert_eq!(
        manifest::argmax(&logits[..vocab]),
        0,
        "the host floors to token 0"
    );

    // The whole-stick form's own leg: all-NaN everywhere, the mask all-false, `w ≡ 0`,
    // `idx = V − 0 = V` — the sentinel row the landed extension reserved, now IN the
    // view (the [V, H] view made this gather out-of-bounds; [V + 1, H] is the fix).
    let whole_table = table(VOCAB);
    let logits: Vec<f32> = vec![f32::NAN; VOCAB];
    let got = composite_row(VOCAB, &logits, &whole_table);
    assert_eq!(
        got,
        vec![0.0; HIDDEN],
        "the whole-stick all-NaN row gathers the ZERO sentinel too"
    );
}

/// The residual hole, pinned as the sentinel's own: every real lane at the fp16 floor
/// (the only row where a biased pad lane — `garbage − 65504` — can strictly beat every
/// real one) also lands on the sentinel. The host floors such a row to token 0; the
/// device refuses it. That is the bias's stated boundary — "a pad lane can win only
/// what an all-floor real row could" — made executable.
#[test]
fn an_all_floor_row_with_winning_garbage_lands_on_the_sentinel_too() {
    let vocab = PAD_VOCAB_A;
    let table = table(vocab);
    let mut logits: Vec<f32> = vec![f16(-65504.0); vocab];
    logits.resize(stick_pad(vocab), f16(65504.0)); // biased pads = 0 > the all-floor reals
    let got = composite_row(vocab, &logits, &table);
    assert_eq!(
        got,
        vec![0.0; HIDDEN],
        "the all-floor row gathers the ZERO sentinel — the executor refuses it"
    );
    assert_eq!(
        manifest::argmax(&logits[..vocab]),
        0,
        "the host floors the same row to token 0 — the documented divergence"
    );
}
