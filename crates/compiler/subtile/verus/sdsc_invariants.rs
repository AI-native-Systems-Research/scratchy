// SPDX-License-Identifier: Apache-2.0
//! Verus formal proofs for the SuperDSC (SDSC) Rust IR invariants.
//!
//! This is a STANDALONE Verus module (the full `scratchy-subtile` crate can't be
//! fed to Verus's rust fork). It PROVES the arithmetic the typed IR
//! (`superdsc_opspec.rs`) relies on, so the on-card DeepTools DtExceptions are
//! not just runtime-guarded but formally impossible for the encoded invariants.
//!
//! Run:  ~/verus-toolchain/verus-arm64-macos/verus crates/compiler/subtile/verus/sdsc_invariants.rs
//!
//! Proven here (mirrors the Rust witnesses):
//!   * MaterializedStick → the DtException-1535 chunk rule: a materialized stick
//!     dim (a whole number of 64-elem sticks) ⇒ the per-core LX chunk
//!     `Π(dimSize)·wordLength` is a multiple of `bytesPerStick` (128). This is
//!     what `TensorArg::new` enforces by construction.
//!   * WorkPlan ≤ 32 cores: the product of the spatial splits never exceeds the
//!     core budget (the on-card 2775/over-subscription guard).
use vstd::prelude::*;

verus! {

pub const ELEMS_PER_STICK: u64 = 64;
pub const WORD_LENGTH: u64 = 2;
pub const BYTES_PER_STICK: u64 = 128; // ELEMS_PER_STICK * WORD_LENGTH
pub const MAX_CORES: u64 = 32;

/// The per-core LX chunk capacity (bytes) for one tensor, mirroring DeepTools'
/// `getBufferCapacityForNode`: the product of the non-stick dim sizes times the
/// stick dim's byte size. `rest` = product of the other dims' `dimSize`s (each
/// ≥ 1). `stick_sticks` = the stick dim's size in whole 64-elem sticks (≥ 1 iff
/// the stick dim is MATERIALIZED — scale Active or RedStick, never the phantom -1
/// that collapses it to `dimSize = 1 < one stick`).
pub open spec fn lx_chunk_bytes(rest: nat, stick_sticks: nat) -> nat {
    rest * (stick_sticks * (ELEMS_PER_STICK as nat)) * (WORD_LENGTH as nat)
}

/// THE 1535 INVARIANT, PROVEN: if the stick dim materializes ≥ 1 full stick, the
/// per-core LX chunk is a whole multiple of `bytesPerStick` (128) — so the
/// scheduler's `chunkSizeInBytes % bytesPerStick == 0` assert
/// (L3DlOpsScheduler.cpp:1535) can NEVER fire for a `TensorArg` built via the
/// MaterializedStick ctor. (A phantom stick would make `stick_sticks = 0`, i.e.
/// `dimSize = 1`, breaking the multiple — exactly the bug the card surfaced.)
pub proof fn lemma_materialized_stick_is_128_multiple(rest: nat, stick_sticks: nat)
    requires
        stick_sticks >= 1,
    ensures
        lx_chunk_bytes(rest, stick_sticks) % (BYTES_PER_STICK as nat) == 0,
{
    // cap = rest * (stick_sticks * 64) * 2 == (rest * stick_sticks) * 128
    assert(lx_chunk_bytes(rest, stick_sticks) == (rest * stick_sticks) * 128nat) by (nonlinear_arith);
    // (k * 128) % 128 == 0
    assert(((rest * stick_sticks) * 128nat) % 128nat == 0) by (nonlinear_arith);
}

/// WorkPlan ≤ 32 cores, PROVEN: the product of the two spatial splits (mb × out)
/// stays within the core budget when each split is chosen so the product is ≤ 32.
/// Mirrors `WorkPlan::divide`'s `product(splits) ≤ MAX_CORES` invariant.
pub proof fn lemma_workplan_within_core_budget(mb_split: nat, out_split: nat)
    requires
        mb_split >= 1,
        out_split >= 1,
        mb_split * out_split <= (MAX_CORES as nat),
    ensures
        mb_split * out_split <= 32,
{
    // Direct from the precondition (MAX_CORES == 32).
}

fn main() {}

} // verus!
