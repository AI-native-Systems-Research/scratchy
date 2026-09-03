//! senlib FFI boundary for the DMA/compute-params + control_blocks subsystem
//! (`crate::dma`, `crate::compute`, `crate::control_blocks`).
//!
//! **This file intentionally declares no `extern "C"` functions.**
//!
//! Verified by reading every file assigned to this phase:
//!   - `flex/include/flex/runtime_stream/runtime_submission_params.hpp` +
//!     `.cpp` — `DmaParams`/`ComputeParams`/`FillParams` construction is
//!     pure host-side parameter validation (null-pointer/length checks
//!     only); no senlib call anywhere in the file.
//!   - `flex/src/control_blocks/control_block_stream/control_block_stream.hpp`
//!     (fixup structs, `TranslateDmvaToDmpa`) — pure arithmetic over
//!     caller-supplied segment tables.
//!   - `flex/src/control_blocks/control_block_stream/segment_addressing.hpp`
//!     — pure bit arithmetic (`SegmentOffset`/`dmvaToSegmentId`).
//!   - `flex/src/control_blocks/dma/dma_descriptor_pack_allocation.{hpp,cpp}`,
//!     `dma_descriptor_pack_fixup.{hpp,cpp}` — plain value types + `operator<<`.
//!   - `flex/src/control_blocks/dma/dma_control_block_handler.{hpp,cpp}` —
//!     `DmaControlBlockHandler` is the **mock/software device** CB executor
//!     (`RuntimeConfig::FlexCompute() == "NULL"`/`"SENULATOR"` backend
//!     selection in `control_block_handler.cpp:45-65`). Every "device
//!     access" it performs (`Copy`, `ParseFill`, `ParseDefrag`,
//!     `TranslateVirtualToPhysicalAddress`) is a `std::memcpy`/pointer
//!     arithmetic into a flex-owned `DeviceMemory` buffer
//!     (`device_types/device_memory.hpp`), not a senlib call. The one
//!     senlib-*adjacent* type it names, `senlib::DmaControlBlockSectionSBF`
//!     (aliased from `SentientSoc::V1::DmaControlBlockSectionSBF`), is a
//!     read-only on-wire struct layout it reads fields out of — a type
//!     definition, not a call.
//!   - `flex/src/control_blocks/control_block_handler/control_block_handler.{hpp,cpp}`
//!     — same: `ControlBlockHandlerBase`/`AsyncControlBlockHandlerBase` are
//!     pure host bookkeeping (a worker thread + queue) over the same mock
//!     `DeviceMemory`.
//!   - `flex/src/control_blocks/control_block_handler/compute_pipeline_control_block_handler.{hpp,cpp}`
//!     — orchestrates the three mock handlers above; no senlib call.
//!
//! `flex/src/control_blocks/message_processing/response_worker.{hpp,cpp}`
//! (read for boundary context, not owned by this phase) is where the
//! *real* hardware submission lives — `senlib::ControlBlockInterface`/
//! `ResponseBlockInterface` calls inside `ResponseWorker::LaunchCbs`/
//! `FetchResponseBlocks`. That is the scheduler/graph-compiler phase's
//! concern; nothing in this file duplicates it. See
//! `SENLIB_BOUNDARY_dma-compute-controlblocks.md` for the full accounting.
