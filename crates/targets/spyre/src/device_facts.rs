// SPDX-License-Identifier: Apache-2.0

//! ⭐ THE SENLIB PINS — the device-facts table cross-checked against the
//! driver's own generation table.
//!
//! The compiler core's generation constants
//! ([`ktir_core::ddl_binds::DeviceGen`], baked as
//! [`ktir_core::ddl_binds::Baked`]) state gen1's geometry with citations to
//! the vendor's deeptools headers. senlib — the host/driver layer — states
//! the same facts under its OWN names in its snt1p0 generation table
//! (`~/git/senlib-rs/crates/chip/src/generation/snt1p0.rs`, selected by the
//! `snt1p0` feature's `current` alias; `GEN: SpyreGeneration = Snt1P0` at
//! snt1p0.rs:24 is the generation id these pins are taken at).
//!
//! scratchy does NOT depend on senlib-rs (the compiler core must not
//! build-depend on the driver; and the port repo is private upstream for
//! now), so this module COPIES senlib's values with their paths and lines
//! and const-asserts them equal to the table's. A wrong copy on EITHER
//! side — a table that drifts from silicon, or a senlib revision that
//! changes geometry — is a BUILD FAILURE here, not a silent drift. When
//! senlib-rs's public half lands, these pins upgrade to direct reads.
//!
//! The LX capacity is NOT pinned here: senlib has no name for it (senlib
//! is the host/wire layer; the LX is device-internal) — the table states it
//! against the vendor's `sysdef.h` default and torch-spyre's confirmation
//! under its own citation.

/// senlib `RCU_CORES: u32 = 32` (snt1p0.rs:225 — "probe-pinned at 32 on
/// snt1p0 silicon; chip geometry, not a per-card discovery").
const SENLIB_RCU_CORES: u32 = 32;

/// senlib `FLIT_SHIFT: u32 = 7` (snt1p0.rs:292 — one flit is
/// `1 << FLIT_SHIFT` bytes).
const SENLIB_FLIT_SHIFT: u32 = 7;

/// senlib `XLAT_LEN_MAX_FLITS: u64 = 1 << 27` (snt1p0.rs:535 — "a stored
/// length of zero is the field maximum, 2^27 flits (16 GiB)").
const SENLIB_XLAT_LEN_MAX_FLITS: u64 = 1 << 27;

/// senlib `BASE_ADDR_TABLE_ENTRIES: usize = 64` (snt1p0.rs:232).
const SENLIB_BASE_ADDR_TABLE_ENTRIES: usize = 64;

/// The table's core count is senlib's `RCU_CORES`.
const _: () = assert!(
    <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::CORES == SENLIB_RCU_CORES,
    "the baked generation's CORES must equal senlib's RCU_CORES (snt1p0.rs:225)"
);

/// The table's flit is senlib's `1 << FLIT_SHIFT`.
const _: () = assert!(
    <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::FLIT_BYTES
        == (1u64 << SENLIB_FLIT_SHIFT),
    "the baked generation's FLIT_BYTES must equal senlib's 1 << FLIT_SHIFT (snt1p0.rs:292)"
);

/// The table's segment ceiling is senlib's translation-window maximum:
/// `XLAT_LEN_MAX_FLITS` flits of `FLIT_SHIFT`-sized flits.
const _: () = assert!(
    <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::MAX_SEGMENT_BYTES
        == SENLIB_XLAT_LEN_MAX_FLITS << SENLIB_FLIT_SHIFT,
    "the baked generation's MAX_SEGMENT_BYTES must equal senlib's XLAT_LEN_MAX_FLITS flits (snt1p0.rs:535)"
);

/// The table's sid count is senlib's base-address-table width.
const _: () = assert!(
    <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::MAX_SIDS as usize
        == SENLIB_BASE_ADDR_TABLE_ENTRIES,
    "the baked generation's MAX_SIDS must equal senlib's BASE_ADDR_TABLE_ENTRIES (snt1p0.rs:232)"
);

#[cfg(test)]
mod tests {
    /// The pins above are `const` asserts — they fail the BUILD, so no test
    /// can observe them failing. This test exists to state the same equality
    /// at run time for the reader, and to keep the module compiled in every
    /// test leg.
    #[test]
    fn the_senlib_pins_hold() {
        use ktir_core::ddl_binds::{Baked, DeviceGen};
        assert_eq!(<Baked as DeviceGen>::CORES, super::SENLIB_RCU_CORES);
        assert_eq!(
            <Baked as DeviceGen>::FLIT_BYTES,
            1u64 << super::SENLIB_FLIT_SHIFT
        );
        assert_eq!(
            <Baked as DeviceGen>::MAX_SEGMENT_BYTES,
            super::SENLIB_XLAT_LEN_MAX_FLITS << super::SENLIB_FLIT_SHIFT
        );
        assert_eq!(
            <Baked as DeviceGen>::MAX_SIDS as usize,
            super::SENLIB_BASE_ADDR_TABLE_ENTRIES
        );
    }
}
