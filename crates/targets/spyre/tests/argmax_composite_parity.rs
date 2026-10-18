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
/// The three boundary tids — distinct, as a real bundle's are. The table rides the
/// resident-tensor channel (the session's `weights`), the logits the per-step source
/// channel (which narrows to f16 — the images are pinned first), the row is the output.
const LOGITS_TID: u64 = 2;
const TABLE_TID: usize = 1;
const ROW_TID: u64 = 3;

/// f16 round-trip through the emulator's storage: the value an f16 element holds after
/// `Tile::compute`'s single rounding — the moe chain test's own helper.
fn f16(v: f32) -> f32 {
    let bits = ktir_emulator::codec::encode(&[v], ktir_emulator::dtypes::DType::F16);
    ktir_emulator::codec::decode(&bits, 1, ktir_emulator::dtypes::DType::F16)[0]
}

/// The tied embedding table, row-major `[vocab, hidden]` — the orientation the mint's
/// table VIEW states (out-major: token v's embedding is out-slot v). Row `v` is
/// `(131·v + 17·h) mod 251 − 125`: every value f16-exact, and rows pairwise DISTINCT
/// (131 is invertible mod 251, so `v ≠ v'` differ at every lane — a wrong gather index
/// cannot pass as a right one).
fn table() -> Vec<f32> {
    (0..VOCAB)
        .flat_map(|v| {
            (0..HIDDEN).map(move |h| {
                let x = (131 * v + 17 * h) % 251;
                f16(x as f32 - 125.0)
            })
        })
        .collect()
}

/// Execute the walk-mint composite over one logits row and return the gathered row.
fn composite_row(logits: &[f32], table: &[f32]) -> Vec<f32> {
    let tied = TiedLmHeadEmbedding::from_tied_config(true).expect("granite ties");
    let k = mint_argmax_node(
        VOCAB as u32,
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
    // The table as ONE resident f16 tensor, row-major [vocab, hidden] LE.
    let table_bytes: Vec<u8> = table
        .iter()
        .flat_map(|v| ktir_emulator::codec::encode(&[*v], ktir_emulator::dtypes::DType::F16))
        .collect();
    let weights = vec![(
        TABLE_TID,
        table_bytes,
        scratchy_tensors::DType::F16,
        vec![VOCAB, HIDDEN],
    )];
    let mut session = SpyreSession::new_multi(&[(&groups, &[ROW_TID])], weights)
        .expect("build the composite's session");
    let out = session
        .run_step(
            0,
            vec![(LOGITS_TID, logits.to_vec(), vec![1, VOCAB])],
            &[(ROW_TID, 0)],
        )
        .expect("run the composite");
    out[&ROW_TID].clone()
}

/// One parity leg: run the composite, run `manifest::argmax` over the SAME images, and
/// demand the gathered row IS the oracle's index's row — the index itself is never read
/// back (the composite's output is the ROW), so a wrong index can only pass by gathering
/// a row that happens to equal the right one, and the table's rows are pairwise distinct.
fn assert_parity(what: &str, logits: &[f32], table: &[f32]) {
    let got = composite_row(logits, table);
    let want_idx = manifest::argmax(logits);
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
    let table = table();
    let mut logits: Vec<f32> = (-100..(VOCAB as i64 - 100))
        .map(|i| f16(i as f32))
        .collect();
    logits[41] = f16(500.0); // strictly above every other lane
    assert_parity("the unique maximum", &logits, &table);
}

/// TIED MAXIMA at i < j — the EARLIER index wins. This is the ramp arithmetic's whole
/// point: `w = (logits == m) · (V − iota)` makes the first maximum's weight unique, so
/// the second max lands on the earliest lane — #631's first-max law made structural, and
/// here EXECUTED, exactly the way `manifest::argmax`'s strict `>` keeps index 7.
#[test]
fn tied_maxima_take_the_earlier_index() {
    let table = table();
    let mut logits: Vec<f32> = (0..VOCAB).map(|i| f16((i % 13) as f32)).collect();
    logits[7] = f16(99.0);
    logits[90] = f16(99.0); // the tie: equal in fp16, and 7 < 90
    assert_eq!(
        manifest::argmax(&logits),
        7,
        "the host oracle agrees: first wins"
    );
    assert_parity("the tied maximum", &logits, &table);
}

/// An ALL-EQUAL row — every lane ties with every other, index 0 must win. The mask is
/// all-ones, so `w` IS the ramp and its maximum is `V` at lane 0, giving `idx = V − V = 0`.
#[test]
fn an_all_equal_row_gathers_token_zero() {
    let table = table();
    let logits: Vec<f32> = (0..VOCAB).map(|_| f16(5.0)).collect();
    assert_eq!(manifest::argmax(&logits), 0);
    assert_parity("the all-equal row", &logits, &table);
}
