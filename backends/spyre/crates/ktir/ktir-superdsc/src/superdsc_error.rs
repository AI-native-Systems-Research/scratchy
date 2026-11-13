// SPDX-License-Identifier: Apache-2.0
//! The shared build-time error for the SuperDSC (SDSC) emitter.
//!
//! Lives in its own leaf module so BOTH the typed frontend core
//! ([`crate::superdsc_opspec`] — the witnesses) and the wire/lowering module
//! (`scratchy_target_spyre::lower_subtile_tape_to_superdsc`, which this leaf must not depend on and
//! so cannot link to) raise the SAME error type without
//! a module cycle. Every failure mode of SDSC generation is captured here as a
//! Rust `Err` surfaced as a hard `cargo build` failure when wired into the
//! `#[forward]` macro — NEVER an on-card DtException (see memory
//! `guard-every-crash-at-build-time`).

/// Error from the typed SuperDSC construction / SubtileIR → SuperDSC walk. An op
/// the emitter cannot lower, a matmul whose region geometry is internally
/// inconsistent, OR a per-core tile that does not fit the LX scratchpad even
/// after time-tiling, would mean silently-wrong (or DtException'd) on-card
/// output, so each is a build error here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperDscError(pub String);

impl std::fmt::Display for SuperDscError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SuperDSC lowering: {}", self.0)
    }
}
impl std::error::Error for SuperDscError {}
