// SPDX-License-Identifier: Apache-2.0
//! ⛔⛔⛔ THE LAST TWO HAND-SUMMED ADDRESSES IN THE ATTENTION EMITTER, PINNED BEFORE THEY ARE REPLACED.
//!
//! `attn.rs:728` and `:982` both read the kv stream as
//!     `rc_of(mq, nkvh * hd, j * stick, kvh * hd + sl * stick)`
//! — a COLUMN built by hand from two products of DIFFERENT UNITS (kv-head x head_dim, slab x lanes) plus a
//! ROW built from another (sub-block x lanes). That is the exact expression shape this file has already been
//! burned by twice:
//!   * `token_stream` was `rc_of(mq, heads*hd, 0, h*hd + s*stick)`, and transposed to `h*stick + s*hd` it is
//!     IDENTICAL at hd == 64 and wrong at hd == 128 — invisible on every model but the one that fails.
//!   * the restickify DEST was `j*stick*hd` where the law wants `j*stick*stick`, "the same number only at
//!     hd == stick", per the comment still at that site.
//!
//! So before converting these to a named-axis `Nest` view, this pins that the two spellings agree over a
//! sweep that INCLUDES the cases where the coincidences break: hd > stick, nkvh > 1, sl > 0, j > 0. If the
//! conversion changes any emitted address, this test says so instead of a card run saying it three bakes later.
use scratchy_subtile::addr::{Idx, Nest, Row};
use scratchy_subtile::superdsc_opspec::Df;

/// The kv stream is rank-2 `[mq, nkvh*hd]`; kv-head `kvh`'s slab `sl` is the column `kvh*hd + sl*stick`,
/// and sub-block `j` is the row window `[j*stick, ...)`. Named axes put both products inside the law.
fn nest_form(mq: u32, nkvh: u32, hd: u32, j: u32, kvh: u32, sl: u32, stick: u32) -> u32 {
    let _ = stick;
    Nest::new(&["row", "head", "feat"], &[mq, nkvh, hd], Df::Fp16)
        .view()
        .at(Idx::<Row>::n(j * stick))
        .at(Idx::<scratchy_subtile::addr::Head>::n(kvh))
        .slab(sl)
        .dev()
        .into_raw_elems()
}

fn hand_form(mq: u32, nkvh: u32, hd: u32, j: u32, kvh: u32, sl: u32, stick: u32) -> u32 {
    scratchy_subtile::addr::rc_of(mq, nkvh * hd, j * stick, kvh * hd + sl * stick, Df::Fp16)
        .into_raw_elems()
}

#[test]
fn the_nest_view_reproduces_the_hand_summed_kv_stream_address() {
    let stick = 64u32;
    let mut checked = 0usize;
    for hd in [64u32, 128, 256] {
        for nkvh in [1u32, 2, 8] {
            for mq in [1u32, 2, 4, 8, 96] {
                let nslab = hd / stick;
                let nsub = mq.div_ceil(stick);
                for kvh in 0..nkvh {
                    for sl in 0..nslab {
                        for j in 0..nsub {
                            let a = hand_form(mq, nkvh, hd, j, kvh, sl, stick);
                            let b = nest_form(mq, nkvh, hd, j, kvh, sl, stick);
                            assert_eq!(
                                a, b,
                                "hd={hd} nkvh={nkvh} mq={mq} kvh={kvh} slab={sl} sub={j}: \
                                 hand-summed {a} vs named-axis {b}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    // ⛔ A SWEEP THAT PROVES NOTHING IS WORSE THAN NO SWEEP: this must include hd > stick and slab > 0.
    assert!(
        checked > 200,
        "only {checked} case(s) — the sweep collapsed"
    );
    println!("{checked} (mq, nkvh, hd, kvh, slab, sub) cases agree");
}

/// The IDENTITY-matrix slab offset at `lower_subtile_tape_to_superdsc.rs:8551`:
/// `rc_of(hd, hd, s * stick, s * stick)` — the SAME product on BOTH axes, which is the spelling where a
/// transposition is undetectable by construction. Pinned before conversion for the same reason as the kv
/// stream: at one stick every slab term is 0.
fn ident_hand(hd: u32, s: u32, stick: u32) -> u32 {
    scratchy_subtile::addr::rc_of(hd, hd, s * stick, s * stick, Df::Fp16).into_raw_elems()
}

fn ident_nest(hd: u32, s: u32, stick: u32) -> u32 {
    Nest::new(&["row", "feat"], &[hd, hd], Df::Fp16)
        .view()
        .at(Idx::<Row>::n(s * stick))
        .slab(s)
        .dev()
        .into_raw_elems()
}

#[test]
fn the_nest_view_reproduces_the_identity_slab_offset() {
    let stick = 64u32;
    let mut checked = 0usize;
    for hd in [64u32, 128, 256] {
        for s in 0..(hd / stick) {
            assert_eq!(
                ident_hand(hd, s, stick),
                ident_nest(hd, s, stick),
                "hd={hd} slab={s}"
            );
            checked += 1;
        }
    }
    assert!(checked >= 6, "only {checked} case(s) — the sweep collapsed");
    println!("{checked} identity-slab cases agree");
}
