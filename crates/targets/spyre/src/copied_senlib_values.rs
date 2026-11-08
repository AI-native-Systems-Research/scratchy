// SPDX-License-Identifier: Apache-2.0

//! ⭐ COPIES of senlib-rs's generation values, cross-checking the
//! device-facts table against them.
//!
//! These are COPIES of the values senlib-rs states in its snt1p0
//! generation table (`~/git/senlib-rs/crates/chip/src/generation/
//! snt1p0.rs`, at a86bbe8), each with its path and line. The
//! const-asserts below check the device-facts table
//! ([`ktir_core::ddl_binds::DeviceGen`], baked as
//! [`ktir_core::ddl_binds::Baked`]) against these copies: a table value
//! that disagrees with the copies is a BUILD FAILURE.
//!
//! ⛔ A SENLIB CHANGE IS **NOT** DETECTED HERE. scratchy does not depend
//! on senlib-chip (the port repo lives on github.ibm.com, which CI
//! cannot fetch; and the compiler core must not build-depend on the
//! driver), so a senlib revision that changes geometry changes nothing
//! in this build — two literals in one repo compared with each other
//! test the typist, not silicon. The copies are refreshed by hand, with
//! their lines re-checked at the revision named above. The follow-up is
//! a DIRECT dependency on senlib-chip's public half once it is
//! CI-fetchable, at which point these asserts compare against the real
//! thing and a senlib change again fails the build.
//!
//! The LX capacity is not among the copies: senlib has no name for it
//! (senlib is the host/wire layer; the LX is device-internal) — the
//! table states it against the vendor's `sysdef.h` default and
//! torch-spyre's confirmation under its own citation.

/// senlib `RCU_CORES: u32 = 32` (snt1p0.rs:225 — "probe-pinned at 32 on
/// snt1p0 silicon; chip geometry, not a per-card discovery").
const COPIED_SENLIB_RCU_CORES: u32 = 32;

/// senlib `FLIT_SHIFT: u32 = 7` (snt1p0.rs:292 — one flit is
/// `1 << FLIT_SHIFT` bytes).
const COPIED_SENLIB_FLIT_SHIFT: u32 = 7;

/// senlib `XLAT_LEN_MAX_FLITS: u64 = 1 << 27` (snt1p0.rs:535 — "a stored
/// length of zero is the field maximum, 2^27 flits (16 GiB)").
const COPIED_SENLIB_XLAT_LEN_MAX_FLITS: u64 = 1 << 27;

/// senlib `BASE_ADDR_TABLE_ENTRIES: usize = 64` (snt1p0.rs:246).
const COPIED_SENLIB_BASE_ADDR_TABLE_ENTRIES: usize = 64;

/// The table's core count equals the copied `RCU_CORES`.
const _: () = assert!(
    <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::CORES
        == COPIED_SENLIB_RCU_CORES,
    "the baked generation's CORES must equal the copied senlib RCU_CORES (snt1p0.rs:225)"
);

/// The table's flit equals the copied `1 << FLIT_SHIFT`.
const _: () = assert!(
    <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::FLIT_BYTES
        == (1u64 << COPIED_SENLIB_FLIT_SHIFT),
    "the baked generation's FLIT_BYTES must equal the copied senlib 1 << FLIT_SHIFT (snt1p0.rs:292)"
);

/// The table's segment ceiling equals the copied translation-window
/// maximum: `XLAT_LEN_MAX_FLITS` flits of `FLIT_SHIFT`-sized flits.
const _: () = assert!(
    <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::MAX_SEGMENT_BYTES
        == COPIED_SENLIB_XLAT_LEN_MAX_FLITS << COPIED_SENLIB_FLIT_SHIFT,
    "the baked generation's MAX_SEGMENT_BYTES must equal the copied senlib XLAT_LEN_MAX_FLITS flits (snt1p0.rs:535)"
);

/// The table's sid count equals the copied base-address-table width.
const _: () = assert!(
    <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::MAX_SIDS as usize
        == COPIED_SENLIB_BASE_ADDR_TABLE_ENTRIES,
    "the baked generation's MAX_SIDS must equal the copied senlib BASE_ADDR_TABLE_ENTRIES (snt1p0.rs:246)"
);

#[cfg(test)]
mod tests {
    /// The asserts above are `const` asserts — they fail the BUILD, so no
    /// test can observe them failing. This test states the same equality
    /// at run time for the reader, and keeps the module compiled in every
    /// test leg.
    #[test]
    fn the_copied_senlib_values_match_the_table() {
        use ktir_core::ddl_binds::{Baked, DeviceGen};
        assert_eq!(<Baked as DeviceGen>::CORES, super::COPIED_SENLIB_RCU_CORES);
        assert_eq!(
            <Baked as DeviceGen>::FLIT_BYTES,
            1u64 << super::COPIED_SENLIB_FLIT_SHIFT
        );
        assert_eq!(
            <Baked as DeviceGen>::MAX_SEGMENT_BYTES,
            super::COPIED_SENLIB_XLAT_LEN_MAX_FLITS << super::COPIED_SENLIB_FLIT_SHIFT
        );
        assert_eq!(
            <Baked as DeviceGen>::MAX_SIDS as usize,
            super::COPIED_SENLIB_BASE_ADDR_TABLE_ENTRIES
        );
    }
}
