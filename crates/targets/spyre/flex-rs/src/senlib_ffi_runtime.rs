//! Runtime-core phase's slice of the senlib boundary — see
//! `SENLIB_BOUNDARY_runtime-core.md` at the crate root for the full rationale.
//! Per-phase file (to avoid symbol/module collisions with parallel agents'
//! own `senlib_ffi_<label>.rs` files); the Assembly phase consolidates all of
//! them later.
//!
//! `RuntimeContext`'s and `RuntimeStream`'s OWN bookkeeping (singleton
//! lifecycle, stream table, in-flight counters, shutdown flag, deferred-error
//! storage, completion-callback lifecycle — ported natively in `runtime.rs`
//! and `stream.rs`) makes exactly ONE direct senlib call of its own:
//! `senlib::v2::SenPci::ncards()`, used by `detail::getNumDevicesImpl` /
//! `flex::getNumDevices()` (`runtime_context.hpp`) to learn the total card
//! count without opening any device — verified at
//! `flex/src/runtime_stream/runtime_context.cpp:44` and (identically) `:109`.
//! This is irreducible: card enumeration is a PCI/hardware-topology fact that
//! only senlib (backed by the driver) can answer; there is no flex-side
//! algorithm to port for it.
//!
//! Everything else `RuntimeContext` touches — device interface construction
//! (`DeviceInterfaceFactory`), the allocator (`FlexAllocator`), and the
//! scheduler backend (`RuntimeScheduler`/`PfRuntimeScheduler`) — belongs to
//! other subsystems' own senlib boundaries (or, for the scheduler, is out of
//! scope for this phase entirely per the task brief) and is deliberately not
//! declared here. `RuntimeStream::launchOperation*` itself never touches
//! senlib directly in the C++ source — it validates parameters and calls
//! `scheduler_->schedule(...)`; the actual hardware submission lives inside
//! the scheduler backend, which is a parallel agent's file
//! (`crate::scheduler`), not this one.
//!
//! Not built/linked on this machine (no senlib headers/libs present): an
//! honest `extern "C"` declaration with no body, not a stub implementation.

unsafe extern "C" {
    /// Corresponds to `senlib::v2::SenPci::ncards()`
    /// (`flex/src/runtime_stream/runtime_context.cpp:44`), which returns the
    /// number of Spyre cards physically present in the system, independent
    /// of any process-level device filtering (`SPYRE_DEVICES`/
    /// `AIU_WORLD_SIZE`, applied by flex-side logic in `runtime.rs`, not
    /// here).
    pub fn flex_senlib_pci_ncards() -> u64;
}
