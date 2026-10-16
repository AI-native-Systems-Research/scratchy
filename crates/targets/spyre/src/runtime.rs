// SPDX-License-Identifier: Apache-2.0
//! The typed card surface: streams, device addresses, launches, and the card's memory capacity —
//! flex-rs's own Rust types held directly, with no C ABI between this crate and its runtime.
//!
//! Errors are typed ([`HwError`]) and carry the refusing layer's own error; addresses are owned
//! `Arc<CompositeAddress>` values, not raw pointers; stream creation reports the bring-up or
//! context refusal; and [`DevAddr::from_chunk`] is infallible because building a descriptor value
//! cannot fail.
//!
//! Laws this surface holds:
//! - [`Stream::h2d`] takes its transfer length FROM THE SLICE — the over-read that motivated it
//!   is unrepresentable, not documented against.
//! - STREAM CREATION IS SERIALIZED — see the ⛔ doc on [`Stream::create`].
//! - Session-lifetime device handles are leaked ON PURPOSE at their drop sites (`OpProg`/
//!   `Executor` in [`crate::superdsc_exec`]) — see the ⛔ doc on [`DevAddr`].
//! - One process-wide `RuntimeContext`, built lazily by [`init_runtime`] / [`Stream::create`] /
//!   [`DevAddr::alloc`], over the real senlib scheduler.

use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use flex_rs::address::{Chunk, CompositeAddress, LogicalAddress, RegionId};
use flex_rs::allocator::{
    AllocationDirective, AllocationDirectiveError, FlexAllocator, FlexAllocatorError,
    FlexAllocatorInit,
};
use flex_rs::compute::{BootstrapOffset, ComputeParams, KernelName};
use flex_rs::device_memory_allocator::DeviceMemoryAllocator;
use flex_rs::dma::{DmaDirection, DmaParams, DmaPipelineShape, DmaShaping, HostVirtualAddress};
use flex_rs::flex_config::DeviceKind;
use flex_rs::memory_region::{DomainId, MemoryType};
use flex_rs::runtime::{RuntimeContext, RuntimeContextError};
use flex_rs::scheduler::{
    DeviceAddressResolver, FlexAllocatorAddressResolver, HardwareSubmit, QueueCapacitySource,
    SchedulerBackendConfig, SchedulerFactoryError, SenlibHardwareSubmit, SenlibQueueCapacity,
    create_scheduler,
};
use flex_rs::stream::{
    RuntimeStream, RuntimeStreamError, RuntimeStreamMode, RuntimeStreamPriority, StreamId,
};
use flex_rs::topology::populate_domain_topology;

type RuntimeInit = Result<Arc<RuntimeContext>, Arc<BringUpError>>;

fn runtime_cell() -> &'static Mutex<Option<RuntimeInit>> {
    static CELL: OnceLock<Mutex<Option<RuntimeInit>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(None))
}

/// Idempotent bring-up: real senlib device init (`flex_shim_init_runtime`)
/// then a real `RuntimeContext` over a `Pf` backend + a real `FlexAllocator`
/// whose address resolution is `FlexAllocatorAddressResolver` (production —
/// see the `flex_rs::scheduler` module doc).
///
/// Single on-card domain (`DomainId(0)`), unconditionally: nothing in this codebase has ever
/// asked the topology which domain is real, and this module does not invent a multi-domain
/// topology query that has no call site anywhere else.
fn build_runtime() -> Result<Arc<RuntimeContext>, BringUpError> {
    let rc = unsafe { flex_rs::senlib_ffi::flex_shim_init_runtime(0) };
    if rc != 0 {
        return Err(BringUpError::ShimInit(rc));
    }

    // Real single on-card domain, sized off the card's actual HBM capacity
    // and RCU core count — `flex::populateDomainTopology`
    // (`device_memory_topology.cpp`), ported natively as
    // `topology::populate_domain_topology`. A hardcoded placeholder here once
    // mis-sized the domain and caused a real `FlexAllocator` OOM during
    // hardware testing — the capacity must come from the card, not a literal.
    // SAFETY: `flex_shim_init_runtime` above already succeeded, so the
    // device handle this reads from is live.
    let device_memory_size = unsafe { flex_rs::senlib_ffi::flex_senlib_device_memory_size() };
    let topo = populate_domain_topology(device_memory_size);

    let init = FlexAllocatorInit::default();
    let dma = DeviceMemoryAllocator::new(flex_rs::allocator::DEVICE_ALIGNMENT);
    let allocator = Arc::new(Mutex::new(
        FlexAllocator::new(init, topo, dma).map_err(BringUpError::AllocatorInit)?,
    ));

    let addr_resolver: Arc<dyn DeviceAddressResolver> =
        Arc::new(FlexAllocatorAddressResolver::new(allocator.clone()));
    let hw: Arc<dyn HardwareSubmit> = Arc::new(SenlibHardwareSubmit::new());
    let capacity: Arc<dyn QueueCapacitySource> = Arc::new(SenlibQueueCapacity);
    // `PfIommuMapper::new()` derives the page size itself via `sysconf(_SC_PAGESIZE)`,
    // matching the real C++'s `PfIommuMapper::PfIommuMapper`'s
    // `page_size_(common::page_size())` (`pf_iommu_mapper.cpp:41-42`).
    let iommu_mapper = flex_rs::iommu::PfIommuMapper::new();
    // `DeviceKind::Pf` is hardcoded, not read from `RuntimeConfig::flex_device()`:
    // `flex_shim_init_runtime` above (the real senlib bring-up call) itself
    // only ever opens a PF device (`senlib::v2::SenPci::pf()`,
    // unconditionally, in cxx-shim/senlib_ffi.cpp) — there is no VF/MOCK
    // bring-up path here yet for a config-selected `DeviceKind` to route to.
    // `create_scheduler` still earns its keep over the old hardcoded
    // `Backend::Pf(PfBackend::new(...))`: it's the one call site that
    // checks `SPYRE_CHIP` against what `RuntimeSchedulerFactory` actually
    // supports (`1P0`), which nothing here checked before.
    let scheduler = create_scheduler(
        &DeviceKind::Pf,
        SchedulerBackendConfig::Real(hw, capacity, addr_resolver, iommu_mapper),
    )
    .map_err(BringUpError::Scheduler)?;

    RuntimeContext::create(None, scheduler, allocator).map_err(BringUpError::Context)
}

fn get_or_init_runtime() -> Result<Arc<RuntimeContext>, Arc<BringUpError>> {
    let mut guard = runtime_cell().lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        // The failure is cached on purpose: bring-up costs seconds, so a card that refused once
        // must not be re-probed by every subsequent `Stream::create` / `DevAddr::alloc`.
        *guard = Some(build_runtime().map_err(Arc::new));
    }
    guard.as_ref().unwrap().clone()
}

/// Bring the process-wide flex runtime up (idempotent, ~5.7s cold). `Err` carries the
/// bring-up failure verbatim.
pub fn init_runtime() -> Result<(), HwError> {
    get_or_init_runtime().map(|_| ()).map_err(HwError::BringUp)
}

/// Kick the runtime bring-up on a detached background thread (overlaps host weight load).
pub fn prewarm_runtime() {
    std::thread::spawn(|| {
        if let Err(e) = get_or_init_runtime() {
            tracing::error!("prewarm failed (prepare will report it): {e}");
        }
    });
}

/// Tear the whole RuntimeContext down — the one-shot leak probe. BREAKS every live session.
///
/// Also drops this module's cached handle, so the next [`init_runtime`] / [`Stream::create`] /
/// [`DevAddr::alloc`] rebuilds a fresh runtime rather than failing `NotCreated` against the dead
/// context for the rest of the process.
pub fn runtime_reset() -> Result<(), HwError> {
    RuntimeContext::reset().map_err(HwError::Reset)?;
    *runtime_cell().lock().unwrap_or_else(|e| e.into_inner()) = None;
    Ok(())
}

/// `flex::PROG_OFFSET_BASE` — the device VA base every `job_bin_ptr` is rebased by at launch.
pub fn prog_offset_base() -> u64 {
    flex_rs::compute::PROG_OFFSET_BASE.0
}

/// THE OPEN CARD'S REAL MEMORY CAPACITY IN BYTES, or `None` if the device cannot be brought up.
///
/// The same `flex_senlib_device_memory_size()` ([`DeviceHandle::GetDmpaSize`]) that
/// [`build_runtime`] sizes the allocator's single domain from — exposed so callers that must size a
/// device-resident allocation (the superdsc paged KV pool) can ask the CARD instead of guessing.
///
/// 🛑 GUESSING IS THE BUG THIS EXISTS TO KILL, AND IT HAS ALREADY BEEN PAID FOR ONCE. `build_runtime`
/// used a 32 GiB placeholder and a real `FlexAllocator` OOM on hardware is what found it. The KV pool
/// then repeated the mistake one layer up with an 8 GiB constant of its own, which is why a bare
/// `scr serve` refused to start over arithmetic that never involved the card.
///
/// `None` rather than a plausible default ON PURPOSE: a caller that cannot learn the real capacity
/// must say so, not substitute a number that will size an allocation the card may not be able to
/// serve. Bring-up is idempotent (`flex_shim_init_runtime` is `std::call_once` on the C++ side), so
/// this is safe to call before or after any other runtime use.
pub fn device_memory_bytes() -> Option<CardMemory> {
    // Reuse the ONE bring-up path: `build_runtime` is what the rest of this module goes through, and
    // its `flex_shim_init_runtime` is the documented prerequisite for the capacity call. Going
    // through it (rather than calling `flex_shim_init_runtime` directly) also means a capacity query
    // can never be the thing that races a half-built runtime into existence.
    get_or_init_runtime().ok()?;
    // SAFETY: `get_or_init_runtime` succeeded above, so `flex_shim_init_runtime` returned 0 and the
    // cached `flex::DeviceHandle` this reads from is live — exactly the precondition the extern
    // declaration documents.
    let bytes = unsafe { flex_rs::senlib_ffi::flex_senlib_device_memory_size() };
    // The FFI returns 0 (and logs) when the handle is not live. Zero is not a capacity, and a
    // caller must not divide a pool out of it.
    (bytes > 0).then_some(CardMemory(bytes))
}

/// ⛔⛔⛔ THE CARD'S OWN MEMORY CAPACITY — AND THE ONLY THING A KV BUDGET MAY BE DERIVED FROM.
///
/// 🛑 THE BUG THIS TYPE EXISTS TO MAKE UNREPRESENTABLE. The superdsc KV pool was sized from
/// `const POOL_BUDGET_BYTES: u64 = 8 GiB` — a literal with no relationship to the device it sized a
/// real allocation on. The card reports 114688 MB, so it was **14x too small**, and a bare
/// `scr serve` refused itself with arithmetic that never mentioned the card. A second copy of the
/// same literal (`determine_available_memory` returning `8usize << 30`) then capped the pool the
/// first one had already mis-sized. Both were plain integers, so nothing could tell them apart from
/// a measurement.
///
/// ⛔ AND THE SAME MISTAKE HAD ALREADY BEEN PAID FOR ONE LAYER DOWN: `build_runtime` sized the
/// allocator's domain off a 32 GiB placeholder until a real `FlexAllocator` OOM on hardware found
/// it. Fixing that one did not stop this one, because a `u64` from
/// `flex_senlib_device_memory_size()` and a `u64` written `8 * 1024 * 1024 * 1024` are the same
/// type. So they are not the same type any more.
///
/// The private field is the whole guard: [`device_memory_bytes`] is the ONLY constructor in the
/// crate, and it can only answer from `DeviceHandle::GetDmpaSize()`. A future literal cannot be
/// spent where a `CardMemory` is demanded — `8 * 1024 * 1024 * 1024` does not typecheck as one — so
/// re-introducing a hardcoded KV budget is a BUILD ERROR, not a silently mis-sized pool. That is the
/// [[guard-every-crash-at-build-time]] discipline applied to the quantity that actually regressed.
///
/// The guard, as a build failure — the exact literal that was there, in the slot it was there in:
///
/// ```compile_fail,E0423
/// // `POOL_BUDGET_BYTES = 8 GiB` cannot be reconstituted: the tuple field is private, so a
/// // literal has no way into the type the budget arithmetic demands.
/// let _budget = scratchy_target_spyre::runtime::CardMemory(8 * 1024 * 1024 * 1024);
/// ```
///
/// ```compile_fail,E0308
/// // ...and it cannot be smuggled in as a bare integer either.
/// fn wants_the_card(_: scratchy_target_spyre::runtime::CardMemory) {}
/// wants_the_card(8 * 1024 * 1024 * 1024);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct CardMemory(u64);

// ⛔ THE EMITTER'S CEILING AND THE ALLOCATOR'S ARE ONE NUMBER. `scratchy_spyre_bundle` is a leaf
// (it may depend on neither end of the pipeline), so it states `MAX_SEGMENT_BYTES` itself; this is
// the only crate that sees both, and it refuses to compile if they ever diverge — the build-time
// segment guard would otherwise silently stop matching what the card can serve.
const _: () =
    assert!(crate::bundle_code::MAX_SEGMENT_BYTES == flex_rs::allocator::MAX_REGION_BYTES);

impl CardMemory {
    /// The card's TOTAL capacity in bytes — the number to REPORT, not the number to size an
    /// allocation from. See [`Self::tensor_capacity`] / [`Self::max_single_allocation`], which are
    /// the two quantities an allocation actually has to fit.
    ///
    /// Deliberately not `From<u64>`/`new`: the value leaves this type freely, but nothing outside
    /// `device_memory_bytes` can put one IN.
    pub const fn bytes(self) -> u64 {
        self.0
    }

    /// ⛔⛔⛔ THE BYTES A TENSOR ALLOCATION CAN ACTUALLY BE SERVED FROM — **not** the card.
    ///
    /// `FlexAllocator::pre_allocate_regions` carves the card into
    /// [`flex_rs::allocator::DEFAULT_MAX_REGIONS`] equal regions and reserves the first
    /// [`flex_rs::allocator::DEFAULT_NUM_PROGRAM_REGIONS`] of them for PROGRAM memory. A tensor
    /// allocation may only be served from the rest, so a budget derived from the whole card
    /// over-states what exists by one region's worth.
    ///
    /// 🛑 MEASURED, and it is the same class of bug [`CardMemory`] was created to stop — provenance
    /// was guarded, MEANING was not. `bytes()` (114,688 MB on dd2) was spent where this belonged, so
    /// the paged KV budget printed `86491 MB` while the allocator's own OOM diagnostic for the very
    /// next allocation reported `total_capacity_bytes=103079215104` — 96 GiB, i.e. 6 of 7 regions.
    /// The pool that budget sizes is one allocation on that same 96 GiB, so the missing 16 GiB was
    /// never the KV cache's to spend.
    pub const fn tensor_capacity(self) -> u64 {
        let regions = flex_rs::allocator::DEFAULT_MAX_REGIONS as u64;
        let program = flex_rs::allocator::DEFAULT_NUM_PROGRAM_REGIONS as u64;
        self.region_bytes() * (regions - program)
    }

    /// ⛔ THE LARGEST **SINGLE** ALLOCATION THIS CARD CAN SERVE = one region.
    ///
    /// A `FlexAllocator` allocation is satisfied from ONE pre-allocated region
    /// (`allocate_in_region`): the free space of several regions never combines. So an allocation
    /// larger than a region fails with plenty free — the exact fault shape that reads as a
    /// self-contradictory OOM (`requested_bytes=17365082112, free_space_bytes=103048964224`), and
    /// the reason the granite-3.1-8b fp16 weight segment (16.17 GiB) cannot load on a card whose
    /// regions are 16 GiB.
    pub const fn max_single_allocation(self) -> u64 {
        self.region_bytes()
    }

    /// One region's size, exactly as `pre_allocate_regions` computes it: an equal share of the card,
    /// capped at the hardware ceiling.
    const fn region_bytes(self) -> u64 {
        let per = self.0 / flex_rs::allocator::DEFAULT_MAX_REGIONS as u64;
        if per < flex_rs::allocator::MAX_REGION_BYTES {
            per
        } else {
            flex_rs::allocator::MAX_REGION_BYTES
        }
    }
}

/// Every failure the card surface can report. Distinct failures stay distinct: the runtime's own
/// refusal ([`HwError::Stream`]), the H2D length door refusing BEFORE submission
/// ([`HwError::OverrunsRegion`]), the handle outliving its runtime ([`HwError::StreamGone`]),
/// bring-up ([`HwError::BringUp`]), stream creation ([`HwError::StreamCreate`]), allocation
/// ([`HwError::Alloc`]) and reset ([`HwError::Reset`]) are different facts, not renderings of
/// `rc=-1`.
#[derive(Debug)]
pub enum HwError {
    /// The runtime rejected the launch, or `synchronize()` rethrew a deferred device error.
    Stream(RuntimeStreamError),
    /// The host slice is longer than the device region it targets — refused before submission.
    OverrunsRegion { len: u64, region: u64 },
    /// The stream no longer exists in its runtime: destroyed, or the `RuntimeContext` was reset.
    StreamGone,
    /// Constructing the process-wide runtime failed. The failure is cached, so every later
    /// caller reports the same bring-up refusal.
    BringUp(Arc<BringUpError>),
    /// `RuntimeContext::create_stream` refused the new stream.
    StreamCreate(RuntimeContextError),
    /// A device-memory allocation was refused.
    Alloc(AllocError),
    /// `RuntimeContext::reset` refused.
    Reset(RuntimeContextError),
}

impl fmt::Display for HwError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stream(e) => write!(f, "{e}"),
            Self::OverrunsRegion { len, region } => write!(
                f,
                "refused before submission: the {len} B host slice is longer than its {region} B \
                 device region — this H2D would write past what was allocated"
            ),
            Self::StreamGone => write!(
                f,
                "the stream no longer exists in its runtime (destroyed, or the RuntimeContext was \
                 reset)"
            ),
            Self::BringUp(e) => write!(f, "runtime bring-up failed: {e}"),
            Self::StreamCreate(e) => write!(f, "{e}"),
            Self::Alloc(e) => write!(f, "{e}"),
            Self::Reset(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for HwError {}

/// Why the process-wide runtime could not be constructed — the refusing layer's own error.
#[derive(Debug)]
pub enum BringUpError {
    /// `flex_shim_init_runtime` — the senlib device init — returned a nonzero rc.
    ShimInit(i32),
    /// `FlexAllocator::new` refused the domain topology.
    AllocatorInit(FlexAllocatorError),
    /// `create_scheduler` refused the device kind / chip.
    Scheduler(SchedulerFactoryError),
    /// `RuntimeContext::create` refused.
    Context(RuntimeContextError),
}

impl fmt::Display for BringUpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShimInit(rc) => write!(f, "flex_shim_init_runtime failed: rc={rc}"),
            Self::AllocatorInit(e) => write!(f, "{e}"),
            Self::Scheduler(e) => write!(f, "{e}"),
            Self::Context(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for BringUpError {}

/// Why a device-memory allocation was refused — the refusing layer's own error.
#[derive(Debug)]
pub enum AllocError {
    /// `AllocationDirective::new` refused the placement.
    Directive(AllocationDirectiveError),
    /// The runtime's allocator handle could not be fetched — the context is inactive.
    Context(RuntimeContextError),
    /// `FlexAllocator::allocate` refused the size.
    Allocator(FlexAllocatorError),
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Directive(e) => write!(f, "{e}"),
            Self::Context(e) => write!(f, "{e}"),
            Self::Allocator(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for AllocError {}

/// One created `flex::RuntimeStream`. Dropping it destroys the stream, which drains the
/// scheduler's per-stream fence deques (the resident-session leak fix recreates one per forward).
///
/// `Send` by construction: the handle is an `Arc` clone of the shared `RuntimeContext` plus a
/// `StreamId` — both plain shared state. The stream is only ever used from the session's own
/// thread, but the session moves across worker threads between forwards.
pub struct Stream {
    rt: Arc<RuntimeContext>,
    id: StreamId,
}

/// ⛔⛔⛔ STREAM CREATION IS SERIALIZED, AND IT IS NOT A PERFORMANCE HEDGE.
///
/// `Stream::create` lazily initialises the process-wide flex runtime ([`init_runtime`]) and then
/// calls `RuntimeContext::create_stream`. The prefill ladder builds its rungs IN PARALLEL —
/// measured on granite-3.1-8b: `prefill ladder: 20 rung(s) ready in 6.19s wall — built in parallel`
/// — so twenty-odd threads enter that function at once on every load.
///
/// MEASURED: model load aborts intermittently, roughly one run in three, with `Signal Received: 6`
/// and NO message — the abort fires inside flex during concurrent stream creation (reached from
/// `Executor::prepare` ← `SuperDscSession::new_inner`), so nothing can be caught or retried, and
/// the process dies before serving a single token, taking a `scr batch` run's output file with it.
///
/// A concurrent lazy-init plus a runtime call with no documented thread-safety is the shape of that
/// abort, so the calls are taken one at a time. The cost is bounded and small — 20 creations against
/// a 6.19 s parallel build — and correctness here is not tradeable: an aborted load is a lost run.
static STREAM_CREATE: Mutex<()> = Mutex::new(());

impl Stream {
    /// Create a dedicated stream: `Normal` priority, `STRICT_ORDERING` — the values every scratchy
    /// session has ever run. The knobs flex-rs exposes (`RuntimeContext::create_stream` takes
    /// priority and mode) are the door when a caller needs otherwise (#500).
    pub fn create() -> Result<Stream, HwError> {
        // A poisoned lock still serializes (nothing is protected but the CALL), so recover rather than
        // panic: a panic here would turn one thread's failure into the whole load's.
        let _serialize = STREAM_CREATE.lock().unwrap_or_else(|e| e.into_inner());
        let rt = get_or_init_runtime().map_err(HwError::BringUp)?;
        let id = rt
            .create_stream(
                RuntimeStreamPriority::Normal,
                RuntimeStreamMode::StrictOrdering,
            )
            .map_err(HwError::StreamCreate)?;
        Ok(Stream { rt, id })
    }

    /// The live `RuntimeStream` this handle names. `Err(StreamGone)` after the stream was destroyed
    /// or the runtime reset.
    fn live(&self) -> Result<Arc<RuntimeStream>, HwError> {
        match self.rt.get_stream(self.id) {
            Ok(Some(s)) => Ok(s),
            _ => Err(HwError::StreamGone),
        }
    }

    pub fn synchronize(&self) -> Result<(), HwError> {
        self.live()?.synchronize().map_err(HwError::Stream)
    }

    /// ⛔⛔⛔ ASYNC H2D OF `bytes` INTO `addr`. THE LENGTH IS THE HOST SLICE'S — THERE IS NO OTHER FORM.
    ///
    /// A slice is the only form of door — `h2d` and `h2d_on_compute` differ only in which
    /// hardware queue the transfer rides — and that is the point. A raw pointer plus an
    /// independent length
    /// cannot be made safe by documentation: a caller can source the length from a DIFFERENT
    /// object than the buffer, and when it comes from `addr.total_size()` — the DEVICE region's
    /// alignment-padded extent — the DMA pins past the host buffer's end whenever the pad exceeds
    /// it. That over-read was real (`alloc_ops_at` sending `addr.total_size()` bytes out of
    /// `op.code.init_binary`), though it was NOT the cause of the failure that found it — see
    /// that function, where a granite-3.1-8b program's VFIO pin returned `EFAULT` and cost
    /// prefill rung m=11; fixing the length alone moved that failure zero bytes.
    ///
    /// A slice carries its own length, so the over-read is UNREPRESENTABLE: a caller that wants
    /// fewer bytes than the region slices the buffer, and the slice bound is the proof.
    ///
    /// The device side is the other half of the same pairing, so it is checked here rather than
    /// trusted: a transfer LONGER than the region would write past what was allocated. That refusal
    /// is [`HwError::OverrunsRegion`] — a distinct variant, so a caller rendering the error cannot
    /// confuse "we refused" with "the card refused".
    pub fn h2d(&self, bytes: &[u8], addr: &DevAddr) -> Result<(), HwError> {
        self.h2d_shaped(bytes, addr, DmaShaping::default())
    }

    /// ⭐ H2D ON THE COMPUTE PIPELINE — same transfer, same slice-length law, different hardware
    /// queue. Port of the vendor's `DmaParams::use_compute_pipeline = true` submission shape:
    /// the DMA rides `PipelineId::COMPUTE` with the step's compute ops instead of `ASYNC_DMAI`
    /// (`runtime_scheduler.cpp:440-461`). The CB itself is unchanged — flex-rs's DMA control
    /// block has always been the compute-path `CreateComputeDataTransfer` shape; only the queue
    /// and its bookkeeping move.
    ///
    /// For the player's per-step preamble this buys two things. The step's `[preamble H2D,
    /// compute...]` junction stops being a pipeline switch at all (the H2D records `Compute` as
    /// the stream's last pipeline), so the compute submission no longer auto-barriers — and no
    /// longer HOST-BLOCKS — on the preamble's fence. And the cross-step barrier that remains
    /// (prev `D2H` → this `H2D`, still a switch by tag) fences COMPUTE, so it drains the
    /// DMAI/DMAO deques — empty once every DMA rides COMPUTE — instead of waiting out the
    /// previous step's readback. Ordering itself is carried by the COMPUTE queue's FIFO: the
    /// preamble CB is submitted ahead of the step's compute CBs on the same queue.
    pub fn h2d_on_compute(&self, bytes: &[u8], addr: &DevAddr) -> Result<(), HwError> {
        self.h2d_shaped(
            bytes,
            addr,
            DmaShaping {
                pipeline_shape: DmaPipelineShape::ComputePipeline,
                ..Default::default()
            },
        )
    }

    /// The one body behind [`Self::h2d`] and [`Self::h2d_on_compute`]: the slice-length law and
    /// the `DmaParams` construction, parameterized only by the shaping the door names.
    fn h2d_shaped(&self, bytes: &[u8], addr: &DevAddr, shaping: DmaShaping) -> Result<(), HwError> {
        let region = addr.total_size();
        if ByteSize(bytes.len() as u64) > region {
            return Err(HwError::OverrunsRegion {
                len: bytes.len() as u64,
                region: region.as_u64(),
            });
        }
        let stream = self.live()?;
        // `bytes.len()` is the caller's own explicit DMA byte count — NOT `addr.total_size()`, whose
        // alignment pad can exceed the logical byte count the slice carries. Using
        // `DmaParams::from_device_allocation` here would silently substitute the region's padded
        // extent for the slice's length, over-reading past the caller's host buffer whenever the
        // two differ — the exact defect the slice-typed door above exists to make unrepresentable.
        // `Some(...)` here can never trip `DmaParamsError::NullDeviceAddress`: `addr` is an owned
        // `DevAddr`, not a nullable pointer.
        let params = DmaParams::with_explicit_size(
            HostVirtualAddress(bytes.as_ptr() as usize),
            ByteSize(bytes.len() as u64),
            DmaDirection::HostToDevice,
            Some(addr.0.clone()),
            shaping,
        )
        .expect("device_address is Some, constructed from an owned DevAddr");
        stream.launch_h2d(params, None).map_err(HwError::Stream)
    }

    /// Async D2H into `bytes` from the device window `addr`.
    pub fn d2h(&self, bytes: &mut [u8], addr: &DevAddr) -> Result<(), HwError> {
        self.d2h_shaped(bytes, addr, DmaShaping::default())
    }

    /// ⭐ D2H ON THE COMPUTE PIPELINE — the readback twin of [`Self::h2d_on_compute`]: same
    /// transfer on `PipelineId::COMPUTE` instead of `ASYNC_DMAO`
    /// (`runtime_scheduler.cpp:519-540`). The `[compute..., logits D2H]` junction is still a
    /// pipeline switch by tag (the D2H's own tag vs the recorded `Compute`), so an
    /// auto-barrier still fires — but it fences COMPUTE, draining the DMAI/DMAO deques (empty
    /// once every DMA rides COMPUTE) instead of HOST-BLOCKING on the step's compute fences
    /// before the readback can even be submitted. The readback CB then rides the same
    /// COMPUTE queue, FIFO-behind the step's compute CBs — the queue carries the ordering the
    /// host-side wait used to.
    pub fn d2h_on_compute(&self, bytes: &mut [u8], addr: &DevAddr) -> Result<(), HwError> {
        self.d2h_shaped(
            bytes,
            addr,
            DmaShaping {
                pipeline_shape: DmaPipelineShape::ComputePipeline,
                ..Default::default()
            },
        )
    }

    /// The one body behind [`Self::d2h`] and [`Self::d2h_on_compute`].
    fn d2h_shaped(
        &self,
        bytes: &mut [u8],
        addr: &DevAddr,
        shaping: DmaShaping,
    ) -> Result<(), HwError> {
        let stream = self.live()?;
        // See `h2d`: the slice's own length is the explicit transfer size — substituting
        // `addr.total_size()` would risk writing past the caller's host buffer.
        let params = DmaParams::with_explicit_size(
            HostVirtualAddress(bytes.as_mut_ptr() as usize),
            ByteSize(bytes.len() as u64),
            DmaDirection::DeviceToHost,
            Some(addr.0.clone()),
            shaping,
        )
        .expect("device_address is Some, constructed from an owned DevAddr");
        stream.launch_d2h(params, None).map_err(HwError::Stream)
    }

    /// One ComputeOnDevice launch (async). `tensor_allocs` is POSITIONAL — entry i backs the
    /// program's logical segment i.
    pub fn compute(
        &self,
        prog: &DevAddr,
        tensor_allocs: &[&DevAddr],
        kernel_name: &str,
        bootstrap_offset: u64,
        tensor_byte_offsets: &[u64],
        pipeline_barrier: bool,
    ) -> Result<(), HwError> {
        let stream = self.live()?;
        let allocs: Vec<Arc<CompositeAddress>> =
            tensor_allocs.iter().map(|a| a.0.clone()).collect();
        let tbo: Vec<ByteOffset> = tensor_byte_offsets.iter().map(|o| ByteOffset(*o)).collect();
        let params = ComputeParams::new(
            prog.0.clone(),
            allocs,
            KernelName(kernel_name.to_owned()),
            BootstrapOffset(bootstrap_offset),
            tbo,
        )
        .with_pipeline_barrier(pipeline_barrier);
        stream.launch_compute(params, None).map_err(HwError::Stream)
    }

    /// N compute launches as ONE submission — one doorbell for the whole list (the vendor's
    /// multi-CB branch; see `flex_rs::stream::RuntimeStream::launch_compute_batch` for the
    /// semantics, including why `pipeline_barrier` is the OR of what the caller would have
    /// passed per op). Device-side order is the list order, same as N `compute` calls.
    ///
    /// ⛔ THE ITEMS OWN THEIR HANDLES. A batch outlives the loop that builds it (the whole
    /// rung is collected, then flushed once), and that loop's shifted bases and `&DevAddr`
    /// borrows are loop-local — so an item carries `DevAddr` clones (descriptor refcounts,
    /// see [`DevAddr`]'s Clone note) rather than borrows.
    ///
    /// `kernel_name` and `tensor_byte_offsets` are not per-item parameters: the only caller
    /// of this door launches exactly the shape `compute` launches at its every call site —
    /// `kernel_name=""` and no tensor byte offsets, matching torch-spyre's
    /// JobPlanStepCompute (the resident binary runs by its bootstrap address; dxp does not
    /// apply `tensor_byte_offsets` to per-op addresses). Widening this to a general batch
    /// door means growing [`BatchedCompute`], not a new method.
    pub fn compute_batch(
        &self,
        launches: &[BatchedCompute],
        pipeline_barrier: bool,
    ) -> Result<(), HwError> {
        let stream = self.live()?;
        let params: Vec<ComputeParams> = launches
            .iter()
            .map(|l| {
                ComputeParams::new(
                    l.prog.0.clone(),
                    l.tensor_allocs.iter().map(|a| a.0.clone()).collect(),
                    KernelName(String::new()),
                    BootstrapOffset(l.bootstrap_offset),
                    Vec::new(),
                )
            })
            .collect();
        stream
            .launch_compute_batch(params, pipeline_barrier, None)
            .map_err(HwError::Stream)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // `destroy_stream` synchronizes (draining the scheduler's per-stream fence deques — the
        // leak fix `predict` relies on when it recreates the stream each forward) and itself
        // logs-and-swallows any deferred error.
        let _ = self.rt.destroy_stream(self.id);
    }
}

/// The byte types [`DevAddr`]'s signature speaks, re-exported so the player imports every device
/// address type from this one module — the typed card surface hands out the types it takes.
pub use flex_rs::address::{ByteOffset, ByteSize};

/// One entry of a [`Stream::compute_batch`] list — the owned handles one compute launch
/// needs. The handles are owned (not borrowed) because a batch outlives the loop that
/// builds it; see `compute_batch`'s ⛔ note. Field semantics match [`Stream::compute`]'s
/// parameters of the same names.
pub struct BatchedCompute {
    /// The program allocation (segment 7) the bootstrap offset is relative to.
    pub prog: DevAddr,
    /// POSITIONAL — entry i backs the program's logical segment i.
    pub tensor_allocs: Vec<DevAddr>,
    /// Offset within the program allocation where kernel execution begins.
    pub bootstrap_offset: u64,
}

/// One chunk's identity — what a shifted or sliced sub-address is built from. The fields are
/// flex-rs's own newtypes, so a chunk cannot be rebuilt with a region id in the offset's place
/// or a domain silently truncated.
#[derive(Clone, Copy, Debug)]
pub struct Chunk0 {
    pub region: RegionId,
    pub offset: ByteOffset,
    pub size: ByteSize,
    pub domain: DomainId,
}

/// An owned `flex::CompositeAddress` — the descriptor of a device allocation or window.
///
/// ⛔ DEVICE REGIONS ARE NEVER FREED, ONLY DESCRIBED. Dropping a `DevAddr` releases the descriptor
/// value (an `Arc<CompositeAddress>` refcount); the underlying device memory lives until process
/// exit. Freeing a region needs a live `RuntimeContext`'s allocator, which a plain descriptor drop
/// cannot reach — and session-lifetime handles are `mem::forget`d ON PURPOSE at their drop sites
/// (`OpProg`/`Executor` in [`crate::superdsc_exec`]; see the ⛔ doc there), so their drop never
/// becomes load-bearing against runtime lifetime. Per-step windows (`shifted`/`window`, built
/// per layer per launch) drop normally as owned values while the runtime is alive.
///
/// `Send` by construction: the descriptor is plain shared data.
///
/// `Clone` is a descriptor refcount bump and nothing more: the field is an `Arc`, the
/// underlying device memory is never freed (see above), so two handles describe the same
/// allocation exactly as one handle and a borrow do. The batch door ([`Stream::compute_batch`])
/// needs it — a batch item outlives the loop that built its shifted bases, so it must own its
/// handles instead of borrowing them.
#[derive(Clone)]
pub struct DevAddr(Arc<CompositeAddress>);

impl DevAddr {
    /// Allocate `nbytes` of device memory of the given [`MemoryType`]. The region is never freed
    /// (see the struct doc); this handle describes it.
    pub fn alloc(nbytes: ByteSize, kind: MemoryType) -> Result<DevAddr, HwError> {
        let rt = get_or_init_runtime().map_err(HwError::BringUp)?;
        let directive = AllocationDirective::new(
            flex_rs::domain::PlacementPolicy::Bind,
            vec![DomainId(0)],
            None,
            kind,
        )
        .map_err(|e| HwError::Alloc(AllocError::Directive(e)))?;
        let allocator = rt
            .get_allocator()
            .map_err(|e| HwError::Alloc(AllocError::Context(e)))?;
        let mut guard = allocator.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .allocate(nbytes, &directive)
            .map(|a| DevAddr(Arc::new(a)))
            .map_err(|e| HwError::Alloc(AllocError::Allocator(e)))
    }

    /// A new single-chunk address naming the window (region, offset, size, domain) — the shifted
    /// per-layer base / sliced-DMA form. Infallible: building a descriptor value cannot fail.
    pub fn from_chunk(c: Chunk0) -> DevAddr {
        let addr = LogicalAddress::new(c.region, c.offset);
        let chunk = Chunk {
            addr,
            size: c.size,
            domain_id: c.domain,
        };
        DevAddr(Arc::new(CompositeAddress::from_chunk(chunk)))
    }

    pub fn total_size(&self) -> ByteSize {
        self.0.total_size()
    }

    pub fn is_single_chunk(&self) -> bool {
        self.0.is_single_chunk()
    }

    pub fn chunk0(&self) -> Option<Chunk0> {
        let c0 = self.0.chunks().first()?;
        Some(Chunk0 {
            region: c0.addr.region_id,
            offset: c0.addr.offset,
            size: c0.size,
            domain: c0.domain_id,
        })
    }

    /// A shifted single-chunk window `sh` bytes into this region — size = the bytes remaining
    /// past `sh`, or the whole chunk when `sh` is beyond it.
    pub fn shifted(&self, sh: ByteSize) -> Option<DevAddr> {
        let c0 = self.chunk0()?;
        let nsz = if c0.size > sh {
            ByteSize(c0.size.0 - sh.0)
        } else {
            c0.size
        };
        Some(DevAddr::from_chunk(Chunk0 {
            region: c0.region,
            offset: ByteOffset(c0.offset.0 + sh.0),
            size: nsz,
            domain: c0.domain,
        }))
    }

    /// A sliced single-chunk window `[off, off+len)` of this region.
    pub fn window(&self, off: ByteOffset, len: ByteSize) -> Option<DevAddr> {
        let c0 = self.chunk0()?;
        Some(DevAddr::from_chunk(Chunk0 {
            region: c0.region,
            offset: ByteOffset(c0.offset.0 + off.0),
            size: len,
            domain: c0.domain,
        }))
    }
}
