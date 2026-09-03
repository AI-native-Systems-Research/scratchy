//! The ONLY senlib boundary needed by the config/util subsystem: card-count
//! and per-card PCIe-address lookups used by `FlexConfig::FlexSpyreDevices` /
//! `FlexConfig::RdmaGetPCIeAddress`.
//!
//! Verified call sites (the C++ line numbers and method names below are
//! verified against the real source; whether `SenPci` and `SenPciShared`'s
//! `ncards()`/`pf(...)` are the exact same underlying senlib call rather than
//! two distinct-but-similar ones is NOT independently verified — `flex-cxx`
//! does not vendor senlib itself, so no senlib header was available to check
//! this during the audit; it is carried here as an assumption, not a fact):
//! - `senlib::v2::SenPci::ncards()` / `senlib::v2::SenPciShared::ncards()`
//!   in `flex/src/runtime_graph/flex_config.cpp:379` and
//!   `flex/src/util/flex_config.cpp:408` (`FlexConfig::FlexSpyreDevices`).
//! - `senlib::v2::SenPci::pf(unsigned int).to_string()` /
//!   `senlib::v2::SenPciShared::pf(...)` in
//!   `flex/src/runtime_graph/flex_config.cpp:442` and
//!   `flex/src/util/flex_config.cpp:471` (`FlexConfig::RdmaGetPCIeAddress`).
//!
//! These are irreducible: both are hardware/driver topology queries (how many
//! physical Spyre cards senlib's PCIe enumeration found, and that card's PCIe
//! bus address) with no flex-owned bookkeeping above them — flex only
//! consumes the returned count/address. See `SENLIB_BOUNDARY_config-util.md`.

use std::ffi::CStr;
use std::os::raw::{c_char, c_uint};

// NOTE: `senlib::v2::SenPci::ncards()` used to be redeclared here as
// `senlib_senpci_ncards`. It is ASSUMED to be the same real senlib call as
// `senlib_ffi_runtime::flex_senlib_pci_ncards` (used by
// `RuntimeContext`/`getNumDevices`) — the Assembly phase consolidated on
// that one canonical declaration; see `senlib_ffi.rs` module doc. This is an
// unverified assumption, not a confirmed fact: `flex-cxx` does not vendor
// senlib itself, so no header was available to check that `SenPci::ncards()`
// and `SenPciShared::ncards()` are really the exact same call rather than
// two distinct-but-similar methods.
#[allow(non_snake_case)]
unsafe extern "C" {
    /// `senlib::v2::SenPci::pf(card_index).to_string()` /
    /// `SenPciShared::pf(...)`: PCIe bus address string for the physical
    /// function of card `card_index`. Returns a senlib-owned, null-terminated
    /// buffer valid for the duration of the call (no ownership transfer).
    fn senlib_senpci_pf_address(card_index: c_uint) -> *const c_char;
}

/// Safe wrapper over `senlib::v2::SenPci::ncards()`. Delegates to the
/// canonical declaration in `senlib_ffi_runtime.rs` (see `senlib_ffi.rs`
/// module doc for why this used to have its own duplicate extern).
pub fn sen_pci_card_count() -> i64 {
    // SAFETY: `flex_senlib_pci_ncards` takes no arguments and returns a plain
    // integer; senlib guarantees it is callable at any point after library
    // init (card enumeration happens once at senlib load time).
    unsafe { crate::senlib_ffi_runtime::flex_senlib_pci_ncards() as i64 }
}

/// Safe wrapper over `senlib::v2::SenPci::pf(card_index).to_string()`.
pub fn pcie_address_for_card(card_index: u32) -> String {
    // SAFETY: `card_index` is caller-checked (by `FlexConfig`) to be less than
    // `sen_pci_card_count()` before this is called; senlib returns a valid
    // null-terminated C string for any in-range index, and the buffer is not
    // retained by us beyond this call.
    unsafe {
        let ptr = senlib_senpci_pf_address(card_index);
        if ptr.is_null() {
            return String::new();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}
