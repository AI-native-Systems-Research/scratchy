//! Native Rust port of the flex-owned call-graph slice reachable from
//! scratchy's Spyre SDK ABI shim: 134 flex-owned types / 320 flex-owned
//! functions across 17 ABI entrypoints, per CodeQL's class-hierarchy-aware
//! transitive closure (see `/tmp/flex_scope_final.json` for the master
//! scope this crate is checked against — the ORIGINAL closure, including
//! the multi_device (RDMA/HDMA) slice that #530 deleted and #550 re-added
//! against the real cross-port P2P surface; see `senlib_ffi`'s module doc
//! and `SENLIB_BOUNDARY.md` for both records). See `SENLIB_BOUNDARY.md` at
//! the crate root for the FFI boundary rationale.
//!
//! Module map:
//! - `address`             — LogicalAddress, Chunk, CompositeAddress
//! - `domain`               — MemoryDomain, PlacementPolicy
//! - `topology`             — DeviceTopology
//! - `memory_region`        — MemoryType, MemoryBlock, MemoryRegion, AllocationBackingStore
//! - `multi_device`         — RDMA/HDMA peer-to-peer exchange, barrier/broadcast
//!   rendezvous, bitmask/bitset bookkeeping, cross-port P2P mapping (#550
//!   re-add; transport bottoms out on `senlib_ffi`'s multi-device section)
//! - `device_memory_allocator` — DeviceMemoryAllocation, DeviceMemoryAllocator (senlib boundary)
//! - `allocator`            — AllocationDirective, FlexAllocator
//! - `runtime`              — RuntimeContext, initializeRuntime, getFlexRuntimeContext
//! - `stream`               — RuntimeStream, RuntimeStreamMode/Priority, StreamId
//! - `dma`                  — DmaParams, create/destroyDmaParams
//! - `compute`              — ComputeParams, create/destroyComputeParams, setPipelineBarrier
//! - `control_blocks`       — control-block construction/fixup layer (SBF translation)
//! - `scheduler`            — RuntimeScheduler and its Pf/Vf/Mock backends
//! - `host_interface`       — RaiiBuffer
//! - `flex_config`          — FlexConfig, RuntimeConfig
//! - `util`                 — sendnn::Status, generic numeric/env helpers
//! - `telemetry`            — FlexLogExtra, FlexConfig subset, Histogram, LatencyBreakdown,
//!   HistTimer, Profiler/GlobalProfiler, UnifiedProfiler, TimestampCalibrator math
//! - `senlib_ffi`           — THE senlib FFI boundary: every `extern "C"`
//!   declaration against senlib in this crate (runtime bring-up, topology,
//!   IOMMU, device-memory allocator, scheduler CBI/RBI submission +
//!   completion, config/util), plus the opaque handle types and safe
//!   wrappers around them. Formerly six per-phase `senlib_ffi_*` modules +
//!   a re-export hub, folded in issue #530.

// A dropped `Result`/`#[must_use]` value in this crate is not a style nit: the
// long-standing hardware hang was exactly that shape — `FreeResponses` credit
// received and never returned, and a senlib submission failure observed and
// discarded. `ResponseCredit`/`PipelineCredit` encode those obligations in the
// type system, and this makes ignoring one a COMPILE ERROR rather than a
// warning scrolled past in a 6-million-line build log.
#![deny(unused_must_use)]

pub mod address;
pub mod allocator;
pub mod compute;
pub mod control_block_wire;
pub mod control_blocks;
pub mod device_memory_allocator;
pub mod dma;
pub mod domain;
pub mod flex_config;
pub mod host_interface;
pub mod iommu;
pub mod memory_region;
pub mod multi_device;
pub mod rb_telemetry;
pub mod runtime;
pub mod scheduler;
pub mod senlib_ffi;
pub mod stream;
pub mod telemetry;
pub mod topology;
pub mod util;
