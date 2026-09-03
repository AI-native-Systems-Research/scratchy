// SPDX-License-Identifier: Apache-2.0
//! Standalone verification for `ir::bridge::span_overflow` — bypasses the bit-rotted
//! `#[cfg(test)]` module in `lower_subtile_tape_to_superdsc.rs` (pre-existing, unrelated
//! signature drift; `cargo test -p scratchy-subtile` fails to even compile independent of this
//! file — confirmed via `git stash` earlier this session).
//!
//!   cargo run -p scratchy-subtile --features superdsc --example verify_span_overflow

use scratchy_subtile::superdsc_opspec::{Df, ItDim};
use scratchy_target_spyre::ir::bridge::span_overflow::{
    MAX_SPAN_BYTES, cheapest_split_clearing_span, physical_span_bytes, split_candidates_for_dim,
};

fn main() {
    // A realistic decode/prefill KV-cache-sized tensor (cap=4096, hd=64, fp16) is nowhere near
    // the 256 MB span limit -- an honest, verifiable NEGATIVE result for this workload's real
    // shapes, not a claim that this port fixes batched-prefill coherence.
    let span = physical_span_bytes(4096, 64, 64, 2);
    assert!(
        span < MAX_SPAN_BYTES,
        "span={span} should be well under the limit"
    );
    println!(
        "cap=4096,hd=64,fp16 span = {span} B ({:.4} MB of {:.0} MB budget)",
        span as f64 / 1e6,
        MAX_SPAN_BYTES as f64 / 1e6
    );

    // Stick-alignment filtering: size=256, stick=64 -> only splits where 256/split stays a
    // multiple of 64 survive (1, 2, 4).
    let candidates = split_candidates_for_dim(256, true, 64);
    assert_eq!(candidates, vec![1, 2, 4]);
    println!("split_candidates_for_dim(256, is_stick=true, stick=64) = {candidates:?}");

    // Non-stick dim: every divisor is legal.
    let candidates = split_candidates_for_dim(12, false, 64);
    assert_eq!(candidates, vec![1, 2, 3, 4, 6, 12]);
    println!("split_candidates_for_dim(12, is_stick=false, stick=64) = {candidates:?}");

    // cheapest_split_clearing_span: an artificially inflated span (mimicking a genuine
    // overflow) that only clears once mb is split 4-way.
    let dim = ItDim {
        name: "mb",
        size: 8,
        is_reduction: false,
        is_stick: false,
        df: Df::Fp16,
    };
    let unit = MAX_SPAN_BYTES / 8; // so "clears" happens exactly at split=4 (32/4=8 units)
    let got = cheapest_split_clearing_span(&dim, |split| (32 / split as u64) * unit)
        .expect("split=4 should clear (32/4=8 units * unit = MAX_SPAN_BYTES)");
    assert_eq!(got, 4, "smallest legal split whose span <= MAX_SPAN_BYTES");
    println!("cheapest_split_clearing_span(mb size=8, inflated span) = {got}");

    // The genuine-overflow-with-no-fix case: even the largest legal split still overflows.
    let unsplittable = ItDim {
        name: "mb",
        size: 7,
        is_reduction: false,
        is_stick: false,
        df: Df::Fp16,
    };
    let err = cheapest_split_clearing_span(&unsplittable, |_| MAX_SPAN_BYTES * 2)
        .expect_err("size=7 is prime; only split=1 or 7 are legal, neither clears 2x budget");
    println!("expected Err (no legal split clears): {err}");

    println!("OK: span_overflow constant/divisor-search/cost-search all behave as ported");
}
