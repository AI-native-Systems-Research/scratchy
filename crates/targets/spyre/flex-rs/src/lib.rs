//! Native Rust port of the flex-owned call-graph slice reachable from
//! scratchy's Spyre SDK ABI shim: 134 flex-owned types / 320 flex-owned
//! functions across 17 ABI entrypoints, per CodeQL's class-hierarchy-aware
//! transitive closure (see `/tmp/flex_scope_final.json` for the master
//! scope this crate is checked against). See `SENLIB_BOUNDARY.md` at the
//! crate root for the FFI boundary rationale.
//!
//! Module map:
//! - `address`             — LogicalAddress, Chunk, CompositeAddress
//! - `domain`               — MemoryDomain, PlacementPolicy
//! - `topology`             — DeviceTopology
//! - `memory_region`        — MemoryType, MemoryBlock, MemoryRegion, AllocationBackingStore
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
//! - `multi_device`         — RDMA/HDMA peer-to-peer exchange, barrier/broadcast
//! - `telemetry`            — FlexLogExtra, FlexConfig subset, Histogram, LatencyBreakdown,
//!   HistTimer, Profiler/GlobalProfiler, UnifiedProfiler, TimestampCalibrator math
//! - `senlib_ffi`           — consolidated re-export hub for all senlib FFI boundaries
//! - `senlib_ffi_allocator`, `senlib_ffi_config_util`, `senlib_ffi_controlblocks`,
//!   `senlib_ffi_multi_device`, `senlib_ffi_runtime`, `senlib_ffi_scheduler` —
//!   per-phase `extern "C"` declarations (see `senlib_ffi` module doc for the
//!   one cross-phase duplicate found and consolidated during Assembly)

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
pub mod runtime;
pub mod scheduler;
pub mod senlib_ffi;
pub mod senlib_ffi_allocator;
pub mod senlib_ffi_config_util;
pub mod senlib_ffi_controlblocks;
pub mod senlib_ffi_multi_device;
pub mod senlib_ffi_runtime;
pub mod senlib_ffi_scheduler;
pub mod stream;
pub mod telemetry;
pub mod topology;
pub mod util;
