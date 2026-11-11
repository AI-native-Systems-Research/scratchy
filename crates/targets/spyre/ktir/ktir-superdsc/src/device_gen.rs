// SPDX-License-Identifier: Apache-2.0
//! The card generation, as the core division reads it — #1158's [`DeviceGen`] / [`Baked`]
//! (`ktir_core::ddl_binds`), re-exported, plus the three spellings the work division reads.
//!
//! ⛔ NO SECOND STATEMENT OF ANY FACT. The trait, the `Gen1` impl and every value live in
//! `ktir_core::ddl_binds` alone; this module only names them for the lowering. Lowering code reads
//! the baked generation ([`Baked`]), never a generation by name — `Gen1` is spelled in
//! `ddl_binds.rs` (its impl and the `Baked` alias) and in tests.

pub use ktir_core::ddl_binds::{Baked, DeviceGen, Gen1};

/// The baked generation's core count — `<Baked as DeviceGen>::CORES`, spelled once.
pub const BUILD_CORES: u32 = <Baked as DeviceGen>::CORES;

/// The baked generation's util-floor core count — `<Baked as DeviceGen>::UTIL_FLOOR_CORES`.
pub const BUILD_UTIL_FLOOR_CORES: u32 = <Baked as DeviceGen>::UTIL_FLOOR_CORES;

/// Every core of the baked generation — the budget of a division that is a law of its own.
pub const fn whole() -> crate::superdsc_opspec::CoreBudget {
    crate::superdsc_opspec::CoreBudget::whole()
}
