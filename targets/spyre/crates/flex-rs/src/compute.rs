//! Port of `flex::ComputeParams` construction from
//! `flex/include/flex/runtime_stream/runtime_submission_params.hpp` +
//! `flex/src/runtime_stream/runtime_submission_params.cpp`.
//!
//! Pure parameter/value object, no senlib involvement — see `crate::dma` for
//! the sibling `DmaParams`/`FillParams` port and the file-level note on
//! what's out of scope for this phase.

use std::sync::Arc;

use crate::address::{ByteOffset, CompositeAddress};
use crate::control_block_wire::{ControlBlockWire, ResponseTag};

/// Name of the kernel to execute. Newtype over the C++ `std::string
/// kernel_name`; empty string in C++ has no special meaning documented in
/// the header beyond "unset", modeled the same way here via `String::is_empty`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KernelName(pub String);

/// Offset within the program allocation (segment 7) where kernel execution
/// begins. Newtype over the C++ `uint64_t bootstrap_offset` (`0` = base of
/// the allocation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BootstrapOffset(pub u64);

/// Byte offset of the reserved Program segment (segment 7) within the
/// device's flat segment-addressed space. Port of
/// `flex::PROG_OFFSET_BASE` (`flex/include/flex/memory_interface/segment_table.hpp:245`:
/// `const std::size_t PROG_OFFSET_BASE = PROG_SEGMENT << SEGMENT_SIZE_BITS;`,
/// with `PROG_SEGMENT = 7` (`:242`) and `SEGMENT_SIZE_BITS = 34` (`:230`)),
/// added here rather
/// than in `address.rs`/`allocator.rs` since its only real use site is
/// alongside `ComputeParams::bootstrap_offset` (both used together to derive
/// a program DMVA in the scheduler backend, e.g.
/// `PROG_OFFSET_BASE + op.getBootstrapOffset()`). Modeled as `ByteOffset`
/// (rather than a bare `u64`) to match that field's own type and this
/// crate's newtype convention.
pub const PROG_OFFSET_BASE: ByteOffset = ByteOffset(7u64 << 34);

/// Port of `flex::ComputeParams`.
///
/// `RuntimeOperationParams` base fields `callback`/`callback_user_data`
/// (`runtime_submission_params.hpp:72-73`) are not represented here — see
/// `crate::dma::DmaParams`' doc comment for why (they are consumed by
/// `RuntimeStream`, `runtime_stream.cpp:148`/`:178`, never by anything
/// holding a params object). `pipeline_barrier` *is* represented. Unlike
/// `DmaParams` — which redeclares (shadows) its own
/// `bool pipeline_barrier = false` at `runtime_submission_params.hpp:124` —
/// `ComputeParams` (`:378-415`) declares no such member and simply uses the
/// inherited `RuntimeOperationParams::pipeline_barrier`
/// (`:71`, default `false`).
///
/// The C++ constructor (`:403-414`) defaults every argument after
/// `device_addr` (`tensor_allocations = {}`, `kernel = ""`,
/// `bootstrap_off = 0`, `tensor_offsets = {}`), as does the
/// `createComputeParams` factory declaration (`:594-598`); `new` below takes
/// them all explicitly since each Rust type here is `Default` and callers can
/// pass `Vec::new()`/`KernelName::default()`/`BootstrapOffset::default()`.
/// No validation is performed in either the C++ constructor or the factory
/// (`runtime_submission_params.cpp:116-121`) — the null-program-address check
/// lives in the scheduler instead (`runtime_scheduler.cpp:346-349`,
/// `RAS::RUNTIMESCHEDULER::NullProgramAddress`), which is why there is no
/// `Result` here and no error type for this module.
#[derive(Debug, Clone)]
pub struct ComputeParams {
    device_address: Arc<CompositeAddress>,
    tensor_allocs: Vec<Arc<CompositeAddress>>,
    kernel_name: KernelName,
    bootstrap_offset: BootstrapOffset,
    tensor_byte_offsets: Vec<ByteOffset>,
    pipeline_barrier: bool,
}

impl ComputeParams {
    /// Port of `createComputeParams`/the private `ComputeParams`
    /// constructor (`runtime_submission_params.hpp:403-414`,
    /// `runtime_submission_params.cpp:116-121`). The C++ constructor performs
    /// no validation beyond defaulting; this mirrors that (no `Result`
    /// needed). `pipeline_barrier` defaults to `false`, matching
    /// `RuntimeOperationParams::pipeline_barrier`'s default
    /// (`runtime_submission_params.hpp:71`); use `with_pipeline_barrier` (port of `setPipelineBarrier`) to
    /// override it.
    pub fn new(
        device_address: Arc<CompositeAddress>,
        tensor_allocs: Vec<Arc<CompositeAddress>>,
        kernel_name: KernelName,
        bootstrap_offset: BootstrapOffset,
        tensor_byte_offsets: Vec<ByteOffset>,
    ) -> Self {
        Self {
            device_address,
            tensor_allocs,
            kernel_name,
            bootstrap_offset,
            tensor_byte_offsets,
            pipeline_barrier: false,
        }
    }

    /// Builder-style setter for `pipeline_barrier`, since `ComputeParams` is
    /// otherwise constructed in one shot via `new`. No direct 1:1 C++ method
    /// analog: in the real source `pipeline_barrier` is a plain
    /// constructor-initialized field on `RuntimeOperationParams`
    /// (`flex/include/flex/runtime_stream/runtime_submission_params.hpp:71`),
    /// not something set through a params-object method. `setPipelineBarrier`
    /// is a *different*, unrelated thing: a virtual method on the
    /// `RuntimeOperation` class (`runtime_stream/operations/runtime_operation.hpp:123`)
    /// used elsewhere to *propagate* `params->pipeline_barrier` onto the
    /// operation object once it's been built (e.g.
    /// `op.setPipelineBarrier(params->pipeline_barrier);` in
    /// `runtime_stream.cpp:146`), not to set the field on the params struct
    /// itself.
    pub fn with_pipeline_barrier(mut self, pipeline_barrier: bool) -> Self {
        self.pipeline_barrier = pipeline_barrier;
        self
    }

    pub fn device_address(&self) -> &Arc<CompositeAddress> {
        &self.device_address
    }
    pub fn tensor_allocs(&self) -> &[Arc<CompositeAddress>] {
        &self.tensor_allocs
    }
    pub fn kernel_name(&self) -> &KernelName {
        &self.kernel_name
    }
    pub fn bootstrap_offset(&self) -> BootstrapOffset {
        self.bootstrap_offset
    }
    pub fn tensor_byte_offsets(&self) -> &[ByteOffset] {
        &self.tensor_byte_offsets
    }
    pub fn pipeline_barrier(&self) -> bool {
        self.pipeline_barrier
    }
}

/// The stamp a replay submission runs on the backend's canvas copy: given
/// the tags the `ResponseWorker` leased (one per CB, in CB order) and the
/// mutable copy, write this step's bytes — see
/// [`ComputeWireBatch::patch`] for the contract.
pub type WireStamp = dyn Fn(&[ResponseTag], &mut [ControlBlockWire]) + Send;

/// A compute batch submitted as PREBUILT wire bytes — #603 PR2's replay
/// payload, scratchy's own extension on top of the vendor surface (no C++
/// analog: the vendor re-encodes every CB from `ComputeParams` per
/// submission; this crate's `ControlBlockWire` made the bytes portable, and
/// the parity gate made them provable, which is what lets a whole step's
/// CBs travel as data).
///
/// * `canvas` — the image: one `ControlBlockWire` per CB, rendered once
///   through the same `build_compute_cb` the `Vec<ComputeParams>` path runs
///   per CB per submission. An `Arc` because the rendering is shared with
///   the caller's image cache: the submit path takes its copy from HERE
///   (one memcpy per step), never from the cache directly.
/// * `patch` — stamps one step onto that copy: the tags the
///   `ResponseWorker` leased (`ControlBlockWire::set_tag`) plus whatever
///   else moves with the step, at offsets the caller derived once. Runs
///   inside `prepare_cbs`, under the submission lock, between the tag
///   lease and the pending-slot publish — on the launching thread, never a
///   completion thread — so it must be cheap, deterministic and
///   side-effect-free: it is the last host code that touches these bytes
///   before the doorbell.
///
/// Submitted through `RuntimeStream::launch_compute_wires`, which carries
/// the barrier decision the same way `launch_compute_batch` does (the OR
/// of what the caller would have passed per op). The hazard tracker sees
/// this payload exactly as it sees the `Vec<ComputeParams>` batch it
/// replaces — no `ComputeParams` to read addresses from, so no addresses,
/// the same "neither checked against nor recorded as" disposition that
/// batch already has (see `scheduler::hazards::address_sets`).
pub struct ComputeWireBatch {
    pub canvas: Arc<Vec<ControlBlockWire>>,
    pub patch: Box<WireStamp>,
}
