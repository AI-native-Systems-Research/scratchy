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
//! - No process-wide card: one [`CardRuntime`] per open card, over the real senlib scheduler, and
//!   every stream and allocation names the `CardRuntime` it acts on.
//! - ONE LAUNCH STREAM PER CARD, minted at bring-up ([`CardRuntime::launches`]) with the step
//!   queue over it ([`CardRuntime::steps`]) — executors borrow it; the card's teardown drains it.

use std::fmt;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

// ⭐ THE COLLECTIVE PLAN'S TWO WIRE FACTS ARE THE RUNTIME'S. The bundle crate is a leaf and cannot
// name flex-rs, so it declares the message cap and the flit itself; this is the one place both
// crates are in scope, and a disagreement fails the build here rather than cutting a fragment the
// card refuses or misaligning a window.
const _: () = {
    assert!(
        scratchy_target_spyre::bundle_code::collective::RDMA_MESSAGE_CAP_BYTES
            == flex_rs::scheduler::P2P_RDMA_WINDOW_SIZE
    );
    assert!(
        scratchy_target_spyre::bundle_code::collective::FLIT_BYTES
            == flex_rs::control_block_wire::Flits::BYTES
    );
};

use scratchy_target_spyre::bundle_code::collective::{
    PerRank, Rank, RecvWindow, Sid, WINDOW_SEGMENT,
};

use flex_rs::address::{Chunk, CompositeAddress, LogicalAddress, RegionId};
use flex_rs::allocator::{
    AllocationDirective, AllocationDirectiveError, FlexAllocator, FlexAllocatorError,
    FlexAllocatorInit,
};
pub use flex_rs::card::Card;
pub use flex_rs::card_table::{CardTable, CardTableError, EnumeratedCard};
use flex_rs::compute::{BootstrapOffset, ComputeParams, KernelName};
use flex_rs::device_memory_allocator::DeviceMemoryAllocator;
use flex_rs::dma::{
    DmaDirection, DmaParams, DmaPipelineShape, DmaShaping, FillParams, FillPattern, FillPipeline,
    HostVirtualAddress, IovaAddress,
};
use flex_rs::flex_config::DeviceKind;
use flex_rs::iommu::{IommuMapping, PfIommuMapper};
use flex_rs::memory_region::{DomainId, MemoryType};
use flex_rs::numa::{CardCpus, PinError, PlacementError};
use flex_rs::runtime::{RuntimeContext, RuntimeContextError};
use flex_rs::scheduler::{
    DeviceAddressResolver, FlexAllocatorAddressResolver, HardwareSubmit, QueueCapacitySource,
    RdmaOperand, RdmaSendParams, SchedulerBackendConfig, SchedulerFactoryError,
    SenlibHardwareSubmit, SenlibQueueCapacity, build_compute_cb, build_rdma_send_cb,
    create_scheduler,
};
use flex_rs::shared_host_pool::{SharedHostPool, SharedHostPoolError};
use flex_rs::stream::{
    DeviceId, RuntimeStream, RuntimeStreamError, RuntimeStreamMode, RuntimeStreamPriority,
    SchedulerError, StreamId, UserCallback,
};
use flex_rs::topology::populate_domain_topology;
use flex_rs::util::Status;

/// ⭐ ONE OPEN SPYRE CARD AND THE FLEX RUNTIME DRIVING IT. Nothing card-scoped is process-global:
/// every stream, allocation and capacity query goes through the `CardRuntime` it acts on, and a
/// process that drives several cards holds several (senlib serves them all in one process —
/// nickm5 held 8 open and linked at once, 2026-11-06). A one-card process is the same code with
/// one `CardRuntime`.
///
/// Built in two phases over a card its owner opened ([`WorldCards::open_cards`] then
/// [`WorldCards::build_all`]): the card BORROWED for `'a` — never cloned — its capacity read from
/// the card, then a real `RuntimeContext` over a `Pf` backend and a real `FlexAllocator` whose
/// address resolution is `FlexAllocatorAddressResolver` (production — see the
/// `flex_rs::scheduler` module doc). Every flex object the runtime holds (allocator, hardware
/// submitter, capacity source, IOMMU mapper, scheduler, context) borrows the card with the same
/// `'a`, so one card has one owner and the compiler — not a refcount — checks that nothing
/// outlives it.
pub struct CardRuntime<'a, T: ktir_superdsc::collective::Tp> {
    card: &'a flex_rs::card::Card,
    /// ⭐ THE CPUS THIS CARD'S HOST THREADS RUN ON — its NUMA node's, read from sysfs at open
    /// (`flex_rs::numa`). Its response worker, its rank's submit thread and its bring-up thread
    /// each pin themselves to it first.
    cpus: CardCpus,
    memory: CardMemory,
    rt: Arc<RuntimeContext<'a>>,
    /// ⭐ THE CARD'S ONE LAUNCH STREAM — minted once here, at bring-up, and
    /// BORROWED by every executor that launches on this card. Before the
    /// pipelining front each `Executor::prepare` minted (and dropped) its own
    /// stream, so a card's sessions serialized on per-session streams and no
    /// host side could state what the card still owed. One stream per card is
    /// the law the step cadence rides: its FIFO is the submission order the
    /// [`steps`](Self::steps) queue's landings are drained in, and its
    /// teardown drain (this runtime's [`shutdown`](Self::shutdown), plus the
    /// handle's own Drop) is the one place a card's outstanding work is
    /// waited out.
    launches: Stream<'a>,
    /// ⭐ THE CARD'S STEP QUEUE — the landings of the steps submitted on
    /// [`launches`](Self::launches), in submission order (the flex step
    /// door's own law). ONE queue exists per card because one thread arms
    /// it: the worker's pipeline thread, which is also the only submitter —
    /// so the arm order IS the submission order, and draining pops landings
    /// in exactly the order the card retires them. In a sharded world only
    /// the LEAD rank's queue is ever armed (a step's landing is armed on the
    /// lead's doorbell; the collectives tie the peers, which settle through
    /// the lead's pick) — a peer's queue exists and stays empty, the same
    /// shape every card gets.
    steps: flex_rs::step::StepQueue<'a>,
    /// This card's p2p link, when it runs one rank of a sharded world — set once, by
    /// [`link_peers`]. A card's `XSEG` entries point at one window's slots, so the card holds at
    /// most one link for its life.
    link: std::sync::OnceLock<PeerLink<T>>,
}

impl<'a, T: ktir_superdsc::collective::Tp> fmt::Debug for CardRuntime<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CardRuntime")
            .field("pci", &self.card.pci())
            .field("cpus", &self.cpus)
            .field("memory", &self.memory)
            .finish_non_exhaustive()
    }
}

impl<'a, T: ktir_superdsc::collective::Tp> CardRuntime<'a, T> {
    /// Build one card's runtime over a card its owner already opened ([`WorldCards::open_cards`])
    /// — the card BORROWED for `'a`, its capacity read from the card.
    ///
    /// Single on-card domain (`DomainId(0)`), unconditionally: nothing in this codebase has ever
    /// asked the topology which domain is real, and this module does not invent a multi-domain
    /// topology query that has no call site anywhere else.
    ///
    /// ⭐ THE CALLING THREAD IS THE CARD'S BRING-UP THREAD: it is pinned to the card's NUMA node
    /// before anything is built, so the host buffers the allocator's first touch lands on that
    /// node. [`WorldCards::build_all`] opens each card's build on its own scoped thread, so each
    /// thread lands on its card's node.
    ///
    /// `scope` is the SERVE SPAN's scope: `create_scheduler` parks its completion threads on it,
    /// so they run until the serve span ends and are joined at its close — the flex-rs law that
    /// the completion thread rides a caller's scope, never a thread of its own. That is why
    /// `'a` (the card's lifetime) must outlive `'scope`: the threads the scope joins borrow the
    /// hardware objects built here.
    fn build<'scope>(
        index: u32,
        pci: &str,
        card: &'a flex_rs::card::Card,
        scope: &'scope std::thread::Scope<'scope, '_>,
    ) -> Result<Self, BringUpError>
    where
        'a: 'scope,
    {
        let cpus = CardCpus::read(pci).map_err(BringUpError::Placement)?;
        cpus.pin_current_thread().map_err(BringUpError::Pin)?;
        tracing::info!("[spyre-card] card {index} ({pci}): host threads on {cpus}");
        // Real single on-card domain, sized off the card's actual HBM capacity and RCU core count —
        // `flex::populateDomainTopology` (`device_memory_topology.cpp`), ported natively as
        // `topology::populate_domain_topology`. A hardcoded placeholder here once mis-sized the
        // domain and caused a real `FlexAllocator` OOM during hardware testing — the capacity must
        // come from the card, not a literal.
        let memory = CardMemory(card.memory_bytes().map_err(BringUpError::Memory)?);
        let topo = populate_domain_topology(memory.0);

        let init = FlexAllocatorInit::default();
        let dma = DeviceMemoryAllocator::new(card, flex_rs::allocator::DEVICE_ALIGNMENT);
        let allocator = Arc::new(Mutex::new(
            FlexAllocator::new(init, topo, dma).map_err(BringUpError::AllocatorInit)?,
        ));

        let addr_resolver: Arc<dyn DeviceAddressResolver> =
            Arc::new(FlexAllocatorAddressResolver::new(allocator.clone()));
        let hw: Arc<dyn HardwareSubmit + 'a> =
            Arc::new(SenlibHardwareSubmit::new(card, cpus.clone()));
        let capacity: Arc<dyn QueueCapacitySource + 'a> = Arc::new(SenlibQueueCapacity::new(card));
        // `PfIommuMapper::new` derives the page size itself via `sysconf(_SC_PAGESIZE)`, matching
        // the real C++'s `PfIommuMapper::PfIommuMapper`'s `page_size_(common::page_size())`
        // (`pf_iommu_mapper.cpp:41-42`).
        let iommu_mapper = flex_rs::iommu::PfIommuMapper::new(card);
        // `DeviceKind::Pf` is hardcoded, not read from `RuntimeConfig::flex_device()`:
        // `Card::open` (the real senlib bring-up call) only ever opens a PF device
        // (`SenPci::pf(addr)`, in cxx-shim/senlib_ffi.cpp) — there is no VF/MOCK bring-up path
        // here for a config-selected `DeviceKind` to route to. `create_scheduler` still checks
        // `SPYRE_CHIP` against what `RuntimeSchedulerFactory` actually supports (`1P0`).
        let scheduler = create_scheduler(
            &DeviceKind::Pf,
            SchedulerBackendConfig::Real(hw, capacity, addr_resolver, iommu_mapper),
            scope,
        )
        .map_err(BringUpError::Scheduler)?;

        let rt = RuntimeContext::create(DeviceId(i64::from(index)), scheduler, allocator);
        // ⭐ THE ONE LAUNCH STREAM, minted at bring-up — serialized with every
        // other stream creation (see the ⛔ doc on [`Stream::create`]), with
        // the step queue over it in the same breath: a card whose launch
        // stream cannot open does not open at all ([`BringUpError::LaunchStream`]),
        // so an executor never reaches for a launch stream that is not there.
        let launches = Stream::open(
            &rt,
            RuntimeStreamPriority::Normal,
            RuntimeStreamMode::StrictOrdering,
            false,
        )
        .map_err(BringUpError::LaunchStream)?;
        let steps = launches.step_queue().map_err(BringUpError::LaunchStream)?;
        Ok(Self {
            card,
            cpus,
            memory,
            rt,
            launches,
            steps,
            link: std::sync::OnceLock::new(),
        })
    }

    /// The senlib card this runtime drives — borrowed for the card's own `'a`, not refcounted.
    pub fn card(&self) -> &'a flex_rs::card::Card {
        self.card
    }

    /// The CPUs this card's host threads run on — its NUMA node's, read at [`Self::open`]. A
    /// thread that drives this card calls `pin_current_thread` on it first.
    pub fn cpus(&self) -> &CardCpus {
        &self.cpus
    }

    /// This card's key in the RB telemetry ring (`flex_rs::rb_telemetry::RbTiming::source`) —
    /// the same identity its hardware submitter records completions under, from the one
    /// definition both read ([`flex_rs::card::Card::telemetry_key`]).
    pub fn telemetry_key(&self) -> usize {
        self.card.telemetry_key()
    }

    /// This card's p2p link, when it runs one rank of a sharded world ([`link_peers`]).
    pub fn link(&self) -> Option<&PeerLink<T>> {
        self.link.get()
    }

    /// The flex `RuntimeContext` driving this card — for flex-level probes that drive the runtime's
    /// own types directly (`examples/shared_host_pool_probe.rs`).
    pub fn context(&self) -> &Arc<RuntimeContext<'a>> {
        &self.rt
    }

    /// ⭐ THE CARD'S ONE LAUNCH STREAM — the stream every executor on this
    /// card borrows to launch. Executors do NOT mint streams: one stream per
    /// card is the law (see the field's ⭐ doc), so a card's submissions all
    /// ride one FIFO and the [`steps`](Self::steps) queue's order is that
    /// FIFO's order.
    pub fn launches(&self) -> &Stream<'a> {
        &self.launches
    }

    /// ⭐ THE CARD'S STEP QUEUE — the landings of this card's submitted
    /// steps, in submission order. The worker's pipeline thread (the one
    /// submitter) arms a step's landing right after the submit that carries
    /// it; the drain pops from the front. See the field's ⭐ doc for the
    /// single-submitter law and the lead-rank law.
    pub fn steps(&self) -> &flex_rs::step::StepQueue<'a> {
        &self.steps
    }

    /// THE CARD'S REAL MEMORY CAPACITY, read from the card at [`Self::open`] — the one
    /// [`CardMemory`] constructor. A card that cannot report its capacity does not open, so a
    /// caller holding a `CardRuntime` never has a capacity to guess.
    pub fn memory(&self) -> CardMemory {
        self.memory
    }

    /// Tear this card's RuntimeContext down — the one-shot leak probe. BREAKS every live session
    /// on this card: their streams and allocations answer `NotCreated` from here on.
    pub fn reset(&self) -> Result<(), HwError> {
        self.rt.reset().map_err(HwError::Reset)
    }

    /// Stop this card's completion thread — the vendor's own teardown door (`Shutdown`), reached
    /// through the runtime context. Bounded by that door's own law, unchanged: a 5 s drain
    /// window for what is still pending, then a force flag and a wait for the exit signal.
    /// After this the card's sessions are dead; [`Self::reset`] is the mid-process probe, this
    /// is teardown — the serve span calls it on every card before it releases the scope the
    /// completion threads live on.
    ///
    /// ⭐ THE LAUNCH STREAM DRAINS FIRST: the one stream every executor on
    /// this card borrowed is the card's own, so the card's teardown — not
    /// each borrower's — is where its outstanding work is waited out
    /// (`drain_for_teardown`, the door that also waits on an errored
    /// stream). Its deferred error, if any, is logged by the stream's own
    /// teardown path, never swallowed into this void return's silence on
    /// purpose: teardown is past the point a caller could act on it.
    pub fn shutdown(&self) {
        let _ = self.launches.drain_for_teardown();
        self.rt.shutdown();
    }
}

/// `flex::PROG_OFFSET_BASE` — the device VA base every `job_bin_ptr` is rebased by at launch.
pub fn prog_offset_base() -> u64 {
    flex_rs::compute::PROG_OFFSET_BASE.0
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
/// The private field is the whole guard: [`CardRuntime::build`] is the ONLY constructor in the
/// crate, and it can only answer from `DeviceHandle::GetDmpaSize()`. A future literal cannot be
/// spent where a `CardMemory` is demanded — `8 * 1024 * 1024 * 1024` does not typecheck as one — so
/// re-introducing a hardcoded KV budget is a BUILD ERROR, not a silently mis-sized pool. That is the
/// [[guard-every-crash-at-build-time]] discipline applied to the quantity that actually regressed.
///
/// The guard, as a build failure — the exact literal that was there, in the slot it was there in:
///
/// ```compile_fail,E0603
/// // `POOL_BUDGET_BYTES = 8 GiB` cannot be reconstituted: the tuple field is private, so a
/// // literal has no way into the type the budget arithmetic demands.
/// let _budget = scratchy_target_spyre_card::runtime::CardMemory(8 * 1024 * 1024 * 1024);
/// ```
///
/// ```compile_fail,E0308
/// // ...and it cannot be smuggled in as a bare integer either.
/// fn wants_the_card(_: scratchy_target_spyre_card::runtime::CardMemory) {}
/// wants_the_card(8 * 1024 * 1024 * 1024);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct CardMemory(u64);

// ⛔ THE EMITTER'S CEILING AND THE ALLOCATOR'S ARE ONE NUMBER. `scratchy_spyre_bundle` is a leaf
// (it may depend on neither end of the pipeline), so it states `MAX_SEGMENT_BYTES` itself; this is
// the only crate that sees both, and it refuses to compile if they ever diverge — the build-time
// segment guard would otherwise silently stop matching what the card can serve.
const _: () = assert!(
    scratchy_target_spyre::bundle_code::MAX_SEGMENT_BYTES == flex_rs::allocator::MAX_REGION_BYTES
);

impl CardMemory {
    /// The card's TOTAL capacity in bytes — the number to REPORT, not the number to size an
    /// allocation from. See [`Self::tensor_capacity`] / [`Self::max_single_allocation`], which are
    /// the two quantities an allocation actually has to fit.
    ///
    /// Deliberately not `From<u64>`/`new`: the value leaves this type freely, but nothing outside
    /// [`CardRuntime::build`] can put one IN.
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
    /// Opening a card's runtime failed.
    BringUp(Arc<BringUpError>),
    /// `RuntimeContext::create_stream` refused the new stream.
    StreamCreate(RuntimeContextError),
    /// A device-memory allocation was refused.
    Alloc(AllocError),
    /// `RuntimeContext::reset` refused.
    Reset(RuntimeContextError),
    /// Rendering compute-CB wire images OFF the submit path refused — the CB
    /// encoder cannot express an operand ([`Stream::render_batch`]).
    /// Never a submission: nothing was sent to the device.
    CbRender(SchedulerError),
    /// A #606 pooled copy ([`Stream::pooled_copy_on_compute`]) was refused by the
    /// pool's own validation set — slot range, single-chunk device, the
    /// slot-capacity transfer law — or the submission it built was rejected.
    PooledCopy(SharedHostPoolError),
    /// Pinning a host shadow into the IOMMU for a pre-mapped H2D
    /// ([`Stream::pin_h2d`]) was refused by the vendor mapper. Never a
    /// submission: nothing was sent on any stream.
    IommuPin(Status),
    /// A pre-mapped H2D ([`H2dPin::send_window`]) was refused before
    /// submission — the window is not one chunk, the host slice is not
    /// exactly the window's extent, or the pin does not cover the shadow.
    PremappedH2d(PremappedH2dRefusal),
}

/// Why a pre-mapped H2D was refused before submission ([`HwError::PremappedH2d`]):
/// the facts [`H2dPin::send_window`]'s recipe states about the transfer it is
/// handed — the device window resolves one chunk, the host slice is exactly
/// that window's extent, the byte range is inside the pinned shadow, and the
/// pin covers the whole shadow it was asked to map.
#[derive(Debug)]
pub enum PremappedH2dRefusal {
    /// The window spans more than one chunk. The pre-mapped recipe resolves one
    /// chunk per transfer (the pool's `copyRaw` law); a caller walking a
    /// multi-chunk region must cut it with [`DevAddr::windows`] first.
    NotOneChunk { chunks: usize },
    /// The host slice's length is not the window's extent. The transfer length
    /// is the window's total size — the pin's iova maps the slice exactly, so
    /// any other length would DMA pages the mapping does not own.
    LengthMismatch { len: u64, window: u64 },
    /// The IOMMU mapping the mapper handed back covers fewer bytes than the
    /// shadow it was asked to pin. A mapping carries its FIRST scatter-gather
    /// entry's length, not the size asked for (the shim reads only
    /// `dev_shm_sglist.front()`), so a pin the driver split does not own an
    /// iova for every byte of the shadow — a window past the split would DMA
    /// from an iova this mapping does not own, and the unpin would leave the
    /// rest pinned for the process's life. Refused by name, never sent.
    ShortPin { asked: u64, pinned: u64 },
    /// The byte range is not inside the pinned shadow: the window walk and the
    /// shadow disagree about the transfer's extent, so the host side and the
    /// iova side of the transfer would not be the same bytes.
    RangeOutsideShadow { start: u64, end: u64, shadow: u64 },
}

impl fmt::Display for PremappedH2dRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotOneChunk { chunks } => write!(
                f,
                "refused before submission: the device window spans {chunks} chunks and the \
                 pre-mapped recipe resolves one — cut the region into single-chunk windows with \
                 DevAddr::windows"
            ),
            Self::LengthMismatch { len, window } => write!(
                f,
                "refused before submission: the {len} B host slice is not the {window} B window — \
                 the pre-mapped transfer length is the window's extent, the span its iova maps"
            ),
            Self::ShortPin { asked, pinned } => write!(
                f,
                "refused before submission: the mapper pinned {pinned} B of a {asked} B shadow — \
                 the mapping owns an iova for only part of it, so windows past the split would \
                 DMA from an iova the mapping does not own"
            ),
            Self::RangeOutsideShadow { start, end, shadow } => write!(
                f,
                "refused before submission: byte range {start}..{end} is outside the {shadow} B \
                 pinned shadow — the window walk and the shadow disagree about the transfer's \
                 extent"
            ),
        }
    }
}

/// The pre-mapped pairing [`H2dPin::send_window`] demands of its (slice, window)
/// pair, stated apart from the door so it is checkable without a card: the window
/// resolves one chunk, and the slice is exactly that window's extent. Everything
/// past this check needs a live stream; the check itself is a fact about the pair.
fn premapped_pairing(bytes: &[u8], addr: &DevAddr) -> Result<(), PremappedH2dRefusal> {
    if !addr.is_single_chunk() {
        return Err(PremappedH2dRefusal::NotOneChunk {
            chunks: addr.0.chunks().len(),
        });
    }
    let window = addr.total_size();
    if bytes.len() as u64 != window.as_u64() {
        return Err(PremappedH2dRefusal::LengthMismatch {
            len: bytes.len() as u64,
            window: window.as_u64(),
        });
    }
    Ok(())
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
            Self::CbRender(e) => write!(f, "CB image render (no submission): {e}"),
            Self::PooledCopy(e) => write!(f, "pooled copy: {e}"),
            Self::IommuPin(e) => write!(f, "pinning host memory into the IOMMU: {e}"),
            Self::PremappedH2d(e) => write!(f, "pre-mapped H2D: {e}"),
        }
    }
}
impl std::error::Error for HwError {}

/// ⭐ THE WORLD'S CARDS — one distinct card per rank of world `T`, claimed from the host's card table
/// ([`CardTable`], senlib's enumeration read once). Rank `r` runs on `cards.get(r)`: an index into a
/// value with exactly `T::RANKS` cards, not an integer re-derived per rank.
///
/// ⭐ THE SOURCE, STATED: the world is the FIRST `T::RANKS` cards of senlib's enumeration, in its
/// order — what every TP build has run on. A different subset is a second, typed source; never an
/// environment variable.
///
/// ⭐ DISTINCT BY CONSTRUCTION: [`Self::claim`] is the only constructor and refuses a repeated
/// address, so no two ranks hold one card and nothing downstream checks it again.
#[derive(Debug)]
pub struct WorldCards<T: ktir_superdsc::collective::Tp>(PerRank<T, WorldCard<T>>);

/// One rank's card of a [`WorldCards`] — only a claim mints one, so a card opened as a rank's is a
/// card its world claimed.
#[derive(Debug)]
pub struct WorldCard<T: ktir_superdsc::collective::Tp>(EnumeratedCard, std::marker::PhantomData<T>);

impl<T: ktir_superdsc::collective::Tp> WorldCard<T> {
    /// The claimed card's PCIe address.
    pub fn pci(&self) -> &str {
        self.0.pci()
    }
}

/// Why a world's cards could not be claimed from the host's table — a fact of this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimRefusal {
    /// The world needs more cards than senlib enumerates.
    TooFewCards { needed: u32, enumerated: usize },
    /// senlib names one PCIe address for two enumerated cards.
    RepeatedCard { pci: String },
}

impl fmt::Display for ClaimRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewCards { needed, enumerated } => write!(
                f,
                "this build's world needs {needed} Spyre card(s); senlib enumerates {enumerated}"
            ),
            Self::RepeatedCard { pci } => write!(
                f,
                "senlib names card {pci} twice — a world's ranks need distinct cards"
            ),
        }
    }
}

impl std::error::Error for ClaimRefusal {}

impl<T: ktir_superdsc::collective::Tp> WorldCards<T> {
    /// Claim world `T`'s cards from `table`: rank `r` is the table's `r`-th card.
    ///
    /// # Errors
    /// [`ClaimRefusal`], by name.
    pub fn claim(table: &CardTable) -> Result<Self, ClaimRefusal> {
        let cards = table.cards();
        let needed = <T as scratchy_target_spyre::bundle_code::collective::World>::RANKS;
        if cards.len() < needed as usize {
            return Err(ClaimRefusal::TooFewCards {
                needed,
                enumerated: cards.len(),
            });
        }
        let world = &cards[..needed as usize];
        for (i, c) in world.iter().enumerate() {
            if world[..i].iter().any(|p| p.pci() == c.pci()) {
                return Err(ClaimRefusal::RepeatedCard {
                    pci: c.pci().to_string(),
                });
            }
        }
        Ok(WorldCards(PerRank::from_fn(|r| {
            WorldCard(world[r.get() as usize].clone(), std::marker::PhantomData)
        })))
    }

    /// Rank by rank, each rank's card.
    pub fn per_rank(&self) -> &PerRank<T, WorldCard<T>> {
        &self.0
    }

    /// PHASE 1 of the two-phase bring-up: open the world's cards, each on its own plain thread
    /// (their ~5.7 s bring-ups overlap). Returns the per-rank open THREADS, unjoined — a caller
    /// overlaps the opens with its own work (the worker opens the cards while the host loads
    /// weights) and joins through [`Self::join_opens`], which hands the cards back BY VALUE —
    /// their owner's to place: the worker leaks them for the process, a function-scoped runner
    /// keeps them as locals — because phase 2 ([`Self::build_all`]) borrows each card for the
    /// life of the runtime it builds over it.
    ///
    /// ⭐ EACH THREAD PINS ITSELF to its card's NUMA node before opening it, so the host buffers
    /// `Card::open` touches first land on that node.
    pub fn open_cards(
        &self,
    ) -> PerRank<T, std::thread::JoinHandle<Result<flex_rs::card::Card, BringUpError>>> {
        PerRank::<T, _>::from_fn(|r| {
            let claimed = self.0.get(r);
            let (index, pci) = (claimed.0.index(), claimed.0.pci().to_string());
            std::thread::spawn(move || open_card(index, &pci))
        })
    }

    /// Join phase 1's open threads, in rank order.
    ///
    /// # Errors
    /// The first rank whose card did not open — named.
    pub fn join_opens(
        open: PerRank<T, std::thread::JoinHandle<Result<flex_rs::card::Card, BringUpError>>>,
    ) -> Result<PerRank<T, flex_rs::card::Card>, String> {
        open.try_map_into(|r, h| {
            h.join()
                .map_err(|_| format!("card of rank {}: open thread panicked", r.get()))?
                .map_err(|e| format!("open the card of rank {}: {e}", r.get()))
        })
    }

    /// PHASE 2 of the two-phase bring-up: build every card's runtime over the cards phase 1
    /// opened, each on its own scoped thread (per-rank construction parallelism — those threads
    /// close when this returns; they are NOT the serve span), joined in rank order.
    ///
    /// `scope` is the SERVE SPAN's scope — the one `create_scheduler` parks each card's
    /// completion threads on (see [`CardRuntime::build`]), so it must outlive every runtime
    /// built here.
    ///
    /// # Errors
    /// The first rank whose runtime does not build — named.
    pub fn build_all<'a, 'scope>(
        &self,
        cards: &'a PerRank<T, flex_rs::card::Card>,
        scope: &'scope std::thread::Scope<'scope, '_>,
    ) -> Result<PerRank<T, CardRuntime<'a, T>>, String>
    where
        'a: 'scope,
    {
        std::thread::scope(|build| {
            let handles = PerRank::<T, _>::from_fn(|r| {
                let card = cards.get(r);
                let claimed = self.0.get(r);
                let (index, pci) = (claimed.0.index(), claimed.0.pci().to_string());
                build.spawn(move || CardRuntime::build(index, &pci, card, scope))
            });
            handles.try_map_into(|r, h| {
                h.join()
                    .map_err(|_| format!("card of rank {}: build thread panicked", r.get()))?
                    .map_err(|e| format!("build the card of rank {}: {e}", r.get()))
            })
        })
    }

    /// PHASE 2 FOR THE WORKER: build the world's runtimes on a serve-span scope a dedicated
    /// thread hosts ([`ServeSpan`] — the worker's serve span has no frame of its own to run a
    /// scope in). The scope that `create_scheduler` parks every card's completion threads on is
    /// the one this thread's body IS; the runtimes it returns borrow only the leaked `cards`,
    /// never the scope, so they may be leaked in turn and outlive the call.
    ///
    /// Returns the span — drop it to stop every card and end the span — together with the build
    /// result. On failure the span is still live and must still be dropped (its [`Drop`] handles
    /// a failed build: the runtimes died with the error, so there is nothing to stop and nothing
    /// parked).
    pub fn build_on_serve_span(
        world: WorldCards<T>,
        cards: &'static PerRank<T, flex_rs::card::Card>,
    ) -> SpanBuild<T> {
        let (built_tx, built_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let host = std::thread::Builder::new()
            .name("serve-span".to_string())
            .spawn(move || {
                std::thread::scope(|span| {
                    let built = world.build_all(cards, span);
                    // The channel-drop law: if the worker died before taking this, the send
                    // fails and DROPS the runtimes — their completion threads see their
                    // scheduler references go and exit — and the park below returns at once,
                    // because the release sender died with the span handle.
                    let _ = built_tx.send(built);
                    // Park for the serve span: every completion thread spawned on `span` runs
                    // until this scope closes, and it closes only when the span handle is
                    // dropped (or died with the worker, which the channel-drop case above
                    // already covered).
                    let _ = release_rx.recv();
                })
            })
            .expect(
                "spawn the serve-span hosting thread — a refusal here is boot-time resource \
                     exhaustion, not a load error to route around",
            );
        let built = built_rx.recv().expect(
            "the serve-span host returns the build result before it parks, or dies with \
                     the worker — neither leaves the channel silently empty",
        );
        (
            ServeSpan {
                release: Some(release_tx),
                host: Some(host),
                runtimes: None,
            },
            built,
        )
    }

    /// One line naming every rank's card — the world's card set, for the log.
    ///
    /// (A world of one is the same claim: its card is the first senlib enumerates.)
    pub fn describe(&self) -> String {
        self.0
            .iter()
            .map(|(r, c)| format!("r{}={}", r.get(), c.pci()))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Open ONE card by its PCIe address — [`WorldCards::open_cards`]'s per-rank thread body. The
/// CALLING THREAD (phase 1's per-rank thread) pins itself to the card's NUMA node first, so the
/// host buffers `Card::open` touches first land on that node; the card comes back BY VALUE for
/// the owner to place, because phase 2 borrows it for the life of the runtime built over it.
fn open_card(index: u32, pci: &str) -> Result<flex_rs::card::Card, BringUpError> {
    let cpus = CardCpus::read(pci).map_err(BringUpError::Placement)?;
    cpus.pin_current_thread().map_err(BringUpError::Pin)?;
    tracing::info!("[spyre-card] card {index} ({pci}): opened on {cpus}");
    flex_rs::card::Card::open(pci).map_err(BringUpError::Open)
}

/// The pair [`WorldCards::build_on_serve_span`] returns: the span hosting every card's
/// completion threads, and the world's runtimes (or the first rank's build failure, named).
pub type SpanBuild<T> = (
    ServeSpan<T>,
    Result<PerRank<T, CardRuntime<'static, T>>, String>,
);

/// ⭐ THE WORKER'S SERVE SPAN — the one `thread::scope` that hosts every card's completion
/// threads. A worker driven by the shared engine (`Box<dyn Worker>`, held for the process) has
/// no frame of its own to run a scope in: the engine calls in from its own threads. So the serve
/// span lives HERE, on a dedicated hosting thread whose body IS the scope:
/// [`WorldCards::build_on_serve_span`] builds the runtimes nested inside it (per-rank
/// construction threads, each pinned to its card's node), the built runtimes — whose data
/// borrows only the leaked cards, never the scope — return over a channel, and the thread parks
/// for the span. Only the completion THREADS live in the scope; that is why the runtimes may
/// outlive the frame that built them.
///
/// Dropping this handle ends the span, in order:
///
/// 1. **The join law.** Every card's completion thread is STOPPED first
///    ([`CardRuntime::shutdown`] — the vendor's bounded-drain door), so the scope's own join
///    at exit drains threads that are already stopping, never parked ones. Ranks stop in rank
///    order; each stop is bounded by the door's own 5 s drain + force law.
/// 2. **The every-path release.** [`Drop`] IS the release — it runs on every path after the
///    runtimes were built (a failed later load step, a worker dropped without serving, a panic
///    unwind), not only happy teardown. A span whose build FAILED releases the same way: its
///    runtimes were already dropped with the error, their completion threads saw their scheduler
///    references go, and the hosting thread's park returned when this handle's sender dropped.
/// 3. **The channel-drop law.** If the worker dies before the build result is taken, the
///    hosting thread's send fails — the send itself DROPS the runtimes, so their completion
///    threads exit — and the park below it returns immediately (the release sender died with
///    the handle). No deadlock, no stranded thread.
pub struct ServeSpan<T: ktir_superdsc::collective::Tp> {
    /// The hosting thread's release signal — `send(())` ends the park, the scope closes, the
    /// completion threads join. `None` only inside [`Drop`], which takes it.
    release: Option<std::sync::mpsc::Sender<()>>,
    /// The hosting thread — joined in [`Drop`] after the release, so its scope (and the join it
    /// performs) is finished before the drop returns.
    host: Option<std::thread::JoinHandle<()>>,
    /// The LEAKED runtimes whose completion threads this span hosts — adopted once the caller
    /// has leaked them, shut down in rank order at [`Drop`] before the hosting thread is
    /// released. `None` when the build failed (nothing was leaked; nothing to stop).
    runtimes: Option<&'static PerRank<T, CardRuntime<'static, T>>>,
}

impl<T: ktir_superdsc::collective::Tp> ServeSpan<T> {
    /// Adopt the leaked runtimes this span hosts — the one mint, called once by the owner that
    /// leaked them. Their completion threads are stopped at [`Drop`], before the span releases.
    pub fn adopt(&mut self, runtimes: &'static PerRank<T, CardRuntime<'static, T>>) {
        self.runtimes = Some(runtimes);
    }
}

impl<T: ktir_superdsc::collective::Tp> Drop for ServeSpan<T> {
    fn drop(&mut self) {
        // The join law: stop every card's completion thread BEFORE the hosting thread is
        // released, so the scope's join drains already-stopping threads.
        if let Some(runtimes) = self.runtimes.take() {
            for (_, card) in runtimes.iter() {
                card.shutdown();
            }
        }
        // The every-path release: this runs however the worker got here. The send errors when
        // the hosting thread already ended (a failed build — nothing parks), and the join then
        // returns with it.
        if let Some(release) = self.release.take() {
            let _ = release.send(());
        }
        if let Some(host) = self.host.take() {
            let _ = host.join();
        }
    }
}

/// Why a card's runtime could not be constructed — the refusing layer's own error.
#[derive(Debug)]
pub enum BringUpError {
    /// `Card::open` refused: senlib could not open the card, or opened a different one.
    Open(String),
    /// The open card could not report its memory capacity.
    Memory(String),
    /// `FlexAllocator::new` refused the domain topology.
    AllocatorInit(FlexAllocatorError),
    /// `create_scheduler` refused the device kind / chip.
    Scheduler(SchedulerFactoryError),
    /// The card's NUMA placement could not be read from sysfs.
    Placement(PlacementError),
    /// The bring-up thread could not be run on the card's NUMA node.
    Pin(PinError),
    /// The card's ONE LAUNCH STREAM ([`CardRuntime::launches`]) could not be
    /// created — a card that cannot open its launch stream does not open at
    /// all, so no executor is ever left to mint its own.
    LaunchStream(HwError),
}

impl fmt::Display for BringUpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open(e) => write!(f, "card open: {e}"),
            Self::Memory(e) => write!(f, "card memory capacity: {e}"),
            Self::AllocatorInit(e) => write!(f, "{e}"),
            Self::Scheduler(e) => write!(f, "{e}"),
            Self::Placement(e) => write!(f, "card NUMA placement: {e}"),
            Self::Pin(e) => write!(f, "bring-up thread: {e}"),
            Self::LaunchStream(e) => write!(f, "card launch stream: {e}"),
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

/// ⭐ #32 — ONE DEFERRED READ'S OWN COMPLETION: armed by the D2H op's user
/// callback (the per-op completion hook every `launch_*` has always carried and
/// every seat door has always passed `None`), parked on by the read's finish
/// half instead of a whole-stream synchronize.
///
/// ⛔ THE LATCH ITSELF IS flex-rs's STEP DOOR's ([`flex_rs::step::Landing`]) —
/// refounded, not restated. The completion form ([`UserCallback::Latch`]), the
/// park, and the timeout tick whose only job is the dead-stream escape (an op
/// that FAILS never fires its user callback — the stream records a deferred
/// error and goes sticky-shutdown instead — so a waiter with no tick would
/// hang on a dead device) are stated ONCE, in `flex_rs::step`; this type names
/// that latch as a read's own landing and adds nothing the latch does not
/// already carry. Before the step door existed this file carried its own copy
/// of the pair, the tick, and the park loop — the same law stated twice, which
/// is the defect the refounding deletes.
///
/// One signal names one transfer, and the finish half that parks on it is
/// its only WAITER (the ring law — a slot's bytes are read exactly once, by
/// the drain that owns the handle). The overlap probe gets its non-blocking
/// view through a separate HANDLE TYPE ([`ReadSignal::peek_handle`]) whose
/// whole surface is `fired` — the no-waiting law carried by the TYPE, not
/// by a comment: a handle with no `wait` method cannot break the ring law,
/// so the signal itself is not `Clone`.
pub struct ReadSignal {
    landing: flex_rs::step::Landing,
}

/// ⭐ #32 (E) — THE PEEK HANDLE: a read-only view of one transfer's
/// completion latch, minted only by [`ReadSignal::peek_handle`]. Its whole
/// surface is [`ReadPeek::fired`] — one latch read, never a park, never a
/// byte of the slot — so the overlap probe can hold it across a step
/// without being able to wait on it. The waiter's law stays on
/// `ReadSignal` itself, which this type deliberately cannot become.
pub struct ReadPeek(flex_rs::step::Peek);

impl ReadPeek {
    /// Has the transfer's callback fired? The card-busy answer the overlap
    /// probe asks at each submit about the PREVIOUS step's read — "not yet
    /// fired" is the card owing work, the overlap fraction the depth row
    /// cannot prove (depth counts submits-outstanding, which the
    /// pre-schedule could inflate while the card sat idle; this cannot lie
    /// that way).
    pub fn fired(&self) -> bool {
        self.0.fired()
    }
}

impl ReadSignal {
    pub fn new() -> Self {
        Self {
            landing: flex_rs::step::Landing::new(),
        }
    }

    /// The latch handed to `launch_d2h`/`launch_h2d` as the op's user
    /// callback: the landing's own `(flag, condvar)` pair, by `Arc` clone —
    /// the LAST per-crossing heap allocation on the seam, gone. Firing it
    /// (flex-rs's `UserCallback::Latch` arm, under the same catch_unwind
    /// protection the boxed closure got) sets the flag and wakes the parker;
    /// it runs on the scheduler's completion thread, touches nothing but
    /// this latch, and cannot borrow the transfer's host buffer — the buffer
    /// outlives the call site by the caller's own laws (the ring slot's
    /// index, the shadow's seat ownership). The waiter parks on the same
    /// pair the callback fires, with no intermediary state to agree on.
    fn arm(&self) -> UserCallback {
        self.landing.callback()
    }

    /// ⭐ #32 (E) — MINT THE PEEK HANDLE: the overlap probe's non-blocking
    /// view of this transfer's completion. [`ReadPeek::fired`] is the whole
    /// of what the holder can do — by type, not by comment, the handle
    /// cannot wait, so the drain's park on this signal stays its one waiter
    /// no matter who holds a peek.
    pub fn peek_handle(&self) -> ReadPeek {
        ReadPeek(self.landing.peek())
    }

    /// Park until the read's own callback fires — the step door's one park,
    /// over this read's landing. `dead` runs ONLY on the latch's timeout
    /// tick and answers "can the op this wait is for still complete at
    /// all". Returns `true` when the transfer landed, `false` when the
    /// stream died underneath it.
    pub fn wait_or(&self, dead: impl Fn() -> bool) -> bool {
        self.landing.wait_or(dead)
    }
}

impl Default for ReadSignal {
    fn default() -> Self {
        Self::new()
    }
}

/// One created `flex::RuntimeStream`. Dropping it destroys the stream, which synchronizes it and
/// erases its scheduler bookkeeping (per-stream fence deques included — though a resolved fence
/// now leaves its deque at completion, `SchedulerState::retire_fence`, so those hold only
/// in-flight fences and the stream can stay alive for a whole resident session).
///
/// `Send` by construction: the handle is an `Arc` clone of the shared `RuntimeContext` plus a
/// `StreamId` — both plain shared state. The stream is only ever used from the session's own
/// thread, but the session moves across worker threads between forwards.
pub struct Stream<'a> {
    rt: Arc<RuntimeContext<'a>>,
    id: StreamId,
}

/// ⛔⛔⛔ STREAM CREATION IS SERIALIZED, AND IT IS NOT A PERFORMANCE HEDGE.
///
/// `Stream::create` calls `RuntimeContext::create_stream`. The prefill ladder builds its rungs IN PARALLEL —
/// measured on granite-3.1-8b: `prefill ladder: 20 rung(s) ready in 6.19s wall — built in parallel`
/// — so twenty-odd threads enter that function at once on every load.
///
/// MEASURED: model load aborts intermittently, roughly one run in three, with `Signal Received: 6`
/// and NO message — the abort fires inside flex during concurrent stream creation (reached from
/// `Executor::prepare` ← `SuperDscSession::new_inner`), so nothing can be caught or retried, and
/// the process dies before serving a single token, taking a `scr batch` run's output file with it.
///
/// A concurrent runtime call with no documented thread-safety is the shape of that abort, so the calls are taken one at a time. The cost is bounded and small — 20 creations against
/// a 6.19 s parallel build — and correctness here is not tradeable: an aborted load is a lost run.
static STREAM_CREATE: Mutex<()> = Mutex::new(());

impl<'a> Stream<'a> {
    /// Create a dedicated stream: `Normal` priority, `STRICT_ORDERING`, untracked — the values
    /// every scratchy session has ever run. Sessions stay UNTRACKED deliberately: their streams
    /// are user-owned and order through their own submissions; the hazard tracker's conservative
    /// Compute address sets would cross-edge concurrent sessions sharing resident weights into
    /// full serialization. Delegates to [`Self::create_with`].
    pub fn create<T: ktir_superdsc::collective::Tp>(
        card: &CardRuntime<'a, T>,
    ) -> Result<Stream<'a>, HwError> {
        Self::create_with(
            card,
            RuntimeStreamPriority::Normal,
            RuntimeStreamMode::StrictOrdering,
            false,
        )
    }

    /// ⭐ The #546 door: a stream with the vendor's full creation surface — priority, ordering
    /// mode, and `track_hazards` (`runtime_context.hpp:70-78`). `track_hazards = true`
    /// registers the stream with the scheduler's cross-stream hazard tracker: its ops
    /// participate in RAW/WAR/WAW edge insertion against every OTHER tracked stream
    /// (same-stream ordering is unaffected; untracked streams are never edged, in either
    /// direction). This is the shape for concurrent pipelines over shared device memory —
    /// `OP_ORDERING` streams whose cross-stream ordering the tracker carries instead of a
    /// host-side lock; see `examples/hazard_concurrency.rs`, the #546 concurrency leg.
    pub fn create_with<T: ktir_superdsc::collective::Tp>(
        card: &CardRuntime<'a, T>,
        priority: RuntimeStreamPriority,
        mode: RuntimeStreamMode,
        track_hazards: bool,
    ) -> Result<Stream<'a>, HwError> {
        Self::open(&card.rt, priority, mode, track_hazards)
    }

    /// The one creation body behind every stream door, over a context the
    /// caller already holds — factored out so [`CardRuntime::build`] can
    /// mint the card's launch stream from its own freshly built context
    /// without going through a `CardRuntime` that does not exist yet.
    /// Carries the ⛔ serialization law's lock (see the doc above): every
    /// stream creation in the process, the launch stream's included, is one
    /// at a time.
    fn open(
        rt: &Arc<RuntimeContext<'a>>,
        priority: RuntimeStreamPriority,
        mode: RuntimeStreamMode,
        track_hazards: bool,
    ) -> Result<Stream<'a>, HwError> {
        // A poisoned lock still serializes (nothing is protected but the CALL), so recover rather than
        // panic: a panic here would turn one thread's failure into the whole load's.
        let _serialize = STREAM_CREATE.lock().unwrap_or_else(|e| e.into_inner());
        let id = rt
            .create_stream(priority, mode, track_hazards)
            .map_err(HwError::StreamCreate)?;
        Ok(Stream { rt: rt.clone(), id })
    }

    /// ⭐ THE STEP QUEUE OVER THIS STREAM — the flex step door's landings,
    /// in this stream's submission order. Minted once per card, over the
    /// card's launch stream ([`CardRuntime::steps`]); a stream handle cannot
    /// outlive the queue it minted, so the queue borrows nothing the stream
    /// does not already hold.
    pub fn step_queue(&self) -> Result<flex_rs::step::StepQueue<'a>, HwError> {
        Ok(flex_rs::step::StepQueue::on(&self.live()?))
    }

    /// The live `RuntimeStream` this handle names. `Err(StreamGone)` after the stream was destroyed
    /// or the runtime reset.
    fn live(&self) -> Result<Arc<RuntimeStream<'a>>, HwError> {
        match self.rt.get_stream(self.id) {
            Ok(Some(s)) => Ok(s),
            _ => Err(HwError::StreamGone),
        }
    }

    pub fn synchronize(&self) -> Result<(), HwError> {
        self.live()?.synchronize().map_err(HwError::Stream)
    }

    /// Teardown drain — port of the vendor's `drainForTeardown`
    /// (`runtime_stream.cpp:893`), exposed because it is the safety
    /// prerequisite for recycling memory a stream's outstanding operations
    /// still reference (#609): `synchronize` does NOT wait on a stream that
    /// took a device error (its `needsShutdown` guard returns without
    /// waiting), so "synchronize said it was idle" is not evidence a failed
    /// stream's pinned slots are free. This sets shutdown, waits for
    /// in-flight unconditionally, and surfaces the deferred error exactly
    /// once. The teardown paths (`RuntimeContext::destroy_stream`, `reset`,
    /// `RuntimeStream`'s drop) already call it; it is public for callers
    /// that recycle pinned slots after an errored step and need the same
    /// guarantee before the stream leaves their hands (#603 PR3's
    /// in-flight ring).
    pub fn drain_for_teardown(&self) -> Result<(), HwError> {
        self.live()?.drain_for_teardown().map_err(HwError::Stream)
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
        self.h2d_shaped(bytes, addr, DmaShaping::default(), None)
    }

    /// ⭐ DEVICE-SIDE ZERO FILL of `addr` — the vendor's `RuntimeOperationFill`
    /// (`SetHAFill`): the card writes the pattern itself, NO host bytes cross
    /// the wire. `addr` must be a single-chunk window (the scheduler's fill CB
    /// resolves one chunk — `DevAddr::windows` yields exactly that shape). The
    /// host→device path is bandwidth-bound (~3.5 GB/s effective), so zeroing a
    /// large region by host-pushed H2D costs seconds where the card's own fill
    /// costs a fraction of that — `fill_zero_probe` measures the real rate on
    /// the card it runs on.
    pub fn fill_zero(&self, addr: &DevAddr) -> Result<(), HwError> {
        let stream = self.live()?;
        // `FillParams::new` takes the caller's size verbatim (the C++ member-init
        // list has no validation), so the window's own extent is the byte count —
        // the same pairing `h2d` gets from a slice's length.
        let params = FillParams::new(
            addr.0.clone(),
            addr.total_size(),
            FillPattern(0),
            FillPipeline::Dmai,
        );
        stream.launch_fill(params).map_err(HwError::Stream)
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
            None,
        )
    }

    /// ⭐ #32 — THE H2D TWIN of [`Self::d2h_on_compute_signaled`]: the same
    /// compute-pipeline transfer with the op's completion callback armed. The
    /// caller this exists for is the seat's own refill flush: an H2D reads the
    /// HOST SHADOW at device-execution time (the shadow-write law), so a host
    /// thread that overwrites the shadow must first know every in-flight H2D
    /// has executed. The LAST H2D a submit enqueues is the one the drain waits
    /// — FIFO retires the whole prefix behind it — which is what lets the
    /// read-scoped drain (no whole-stream synchronize) keep the shadow-write
    /// law intact: the drain returns when the read landed AND the shadow is
    /// quiet, and the steps' compute behind those DMAs keeps executing.
    pub fn h2d_on_compute_signaled(
        &self,
        bytes: &[u8],
        addr: &DevAddr,
    ) -> Result<ReadSignal, HwError> {
        let signal = ReadSignal::new();
        self.h2d_shaped(
            bytes,
            addr,
            DmaShaping {
                pipeline_shape: DmaPipelineShape::ComputePipeline,
                // ⭐ #32 (E) — THE CB BARRIER BIT, SET. ⛔ CARD-OBSERVED
                // HARDWARE FACT (the v4/E1/E2 isolation triple, 2026-10-26,
                // golden-pinned ×2 per arm, back-to-back on one card): a
                // compute-pipeline H2D CB does NOT wait for the preceding
                // compute CB to finish reading its operands — the pipelined
                // loop queues step j+2's preamble H2D behind step j+1's
                // RUNNING compute, and with this bit clear the H2D overwrote
                // the fixed-placement per-step activations (rope cos/sin,
                // masks, KV index) mid-read: byte-wrong picks, fluent
                // garbage, deterministic within a session. The WAR edge
                // (compute READ → H2D WRITE) needs CB bit 32 — the vendor's
                // own SetBarrier site on CreateComputeDataTransfer
                // (control_block_wire.rs:1113-1119). The RAW edge (compute
                // WRITE → D2H read) held WITHOUT the bit — CARD-OBSERVED,
                // not vendor-stated: v3's per-step D2H read fresh sticks
                // behind running compute with the bit clear, and E2 (no
                // overlap) pinned clean with the bit clear too. They are
                // opposite edges; only the WAR one orders for free.
                // Host side the bit costs nothing: the barrier drains only
                // the DMAI/DMAO deques, which are empty once every DMA
                // rides COMPUTE.
                pipeline_barrier: true,
                ..Default::default()
            },
            Some(&signal),
        )?;
        Ok(signal)
    }

    /// The one body behind [`Self::h2d`], [`Self::h2d_on_compute`], and
    /// [`Self::h2d_on_compute_signaled`]: the slice-length law and the
    /// `DmaParams` construction, parameterized by the shaping the door names
    /// and the per-op completion hook (`None` leaves the op silent, the doors'
    /// behavior before #32). One body, so a new guard cannot land on one
    /// door's shape and skip the others.
    fn h2d_shaped(
        &self,
        bytes: &[u8],
        addr: &DevAddr,
        shaping: DmaShaping,
        signal: Option<&ReadSignal>,
    ) -> Result<(), HwError> {
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
        stream
            .launch_h2d(params, signal.map(ReadSignal::arm))
            .map_err(HwError::Stream)
    }

    /// ⭐ PIN A HOST SHADOW FOR PRE-MAPPED H2D — the constructor of the guard that carries
    /// the transfer's safety ([`H2dPin`]). The mapper pins and IOMMU-maps the slice's own
    /// pages ONCE (`PfIommuMapper::map`), and the guard that comes back owns that mapping
    /// and borrows the slice it was pinned from — so the iova a window is sent from and
    /// the bytes it reads are derived from ONE value and cannot disagree, and the borrow
    /// cannot end while a transfer may still be reading the bytes (the guard's [`Drop`]
    /// synchronizes before it unpins). Those per-call costs are the plain door's rate —
    /// 6-7 GB/s host-blocking, against ~59 GB/s pre-mapped (card-measured,
    /// `h2d_streams_probe`, 2026-11-10) — which is why boot weight staging, a one-shot
    /// multi-GiB transfer from a THP-backed region, wants this door.
    ///
    /// A mapping that does not cover the whole slice is refused BY NAME before any window
    /// is sent ([`PremappedH2dRefusal::ShortPin`]): the shim's mapping carries its first
    /// scatter-gather entry's length, not the size asked for, so a pin the driver split
    /// does not own an iova for every byte of the shadow.
    pub fn pin_h2d<'p>(
        &'p self,
        mapper: &std::sync::Arc<PfIommuMapper<'a>>,
        src: &'p [u8],
    ) -> Result<H2dPin<'p, 'a>, HwError> {
        let mapping = mapper
            .map(src.as_ptr() as usize, src.len(), true)
            .map_err(HwError::IommuPin)?;
        if (mapping.size_bytes() as u64) < src.len() as u64 {
            // Nothing was submitted from this mapping, so its drop-side unpin
            // races no transfer — other pins' transfers are guarded by their
            // own guards.
            return Err(HwError::PremappedH2d(PremappedH2dRefusal::ShortPin {
                asked: src.len() as u64,
                pinned: mapping.size_bytes() as u64,
            }));
        }
        Ok(H2dPin {
            src,
            stream: self,
            mapping,
        })
    }

    /// Async D2H into `bytes` from the device window `addr`.
    pub fn d2h(&self, bytes: &mut [u8], addr: &DevAddr) -> Result<(), HwError> {
        self.d2h_shaped(bytes, addr, DmaShaping::default(), None)
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
            None,
        )
    }

    /// ⭐ #32 — THE READ-SCOPED TWIN of [`Self::d2h_on_compute`]: the same
    /// compute-pipeline transfer (same slice-length law, same FIFO-behind-the-
    /// step's-compute-CBs ordering), with the op's completion callback ARMED —
    /// the seat's first user of the hook every `launch_*` has carried and every
    /// door until now passed `None`. The returned [`ReadSignal`] parks the
    /// finish half on THIS transfer's own retirement, so a drain waits the one
    /// read it asked for and not the whole stream — the whole-stream
    /// synchronize behind it waited out the very step the pipelined loop had
    /// just submitted for overlap, which is why depth 2 measured as queue
    /// structure and bought nothing (fence mean = one full step).
    ///
    /// ⛔ THE HOST BUFFER OUTLIVES THIS CALL while the DMA is in flight — same
    /// law the unsignaled door already rides (the caller's ring slot or shadow
    /// window stays put until the finish half returns it), now with the
    /// callback as the proof of landing instead of a stream synchronize.
    pub fn d2h_on_compute_signaled(
        &self,
        bytes: &mut [u8],
        addr: &DevAddr,
    ) -> Result<ReadSignal, HwError> {
        let signal = ReadSignal::new();
        self.d2h_shaped(
            bytes,
            addr,
            DmaShaping {
                pipeline_shape: DmaPipelineShape::ComputePipeline,
                ..Default::default()
            },
            Some(&signal),
        )?;
        Ok(signal)
    }

    /// The stream's sticky error state (`RuntimeStream::needs_shutdown`) — a
    /// read-only query for a waiter that must know whether the op it is parked
    /// on can still complete: a FAILED op never fires its user callback (the
    /// stream records a deferred error and goes sticky-shutdown instead), so
    /// [`ReadSignal::wait_or`]'s timeout tick asks this before waiting forever.
    /// A gone stream is dead by definition.
    pub fn is_dead(&self) -> bool {
        self.live().map(|s| s.needs_shutdown()).unwrap_or(true)
    }

    /// ⭐ #603 PR3b — a #606-POOLED, pre-mapped copy on the COMPUTE pipeline: the decode
    /// window's idx-stick D2H into (and any future preamble H2D out of) a
    /// [`SharedHostPool`] slot. The compute shaping is [`Self::d2h_on_compute`]'s own law
    /// (#544: the copy is the step's LAST op, FIFO-behind the step's compute CBs, its
    /// barrier draining the empty DMA deques instead of host-blocking on compute); the
    /// slot's STABLE IOVA is the pool's whole point — the copy rides the scheduler's
    /// pre-mapped path with no per-step pin, no fresh mapping, no shadow buffer, which is
    /// what keeps the per-step DMA a replay candidate instead of a re-render.
    ///
    /// The transfer length is the device object's own `total_size` (the pool door's law),
    /// so the caller passes the PLACEMENT WINDOW, never the whole segment — a window that
    /// cannot form must be refused by the caller, not papered over with the region.
    ///
    /// # Errors
    /// [`HwError`]: [`HwError::StreamGone`] when the stream is no longer live, or
    /// [`HwError::PooledCopy`] carrying the pool's own validation set (slot range,
    /// single-chunk device, the slot-capacity transfer law) or the submission's
    /// rejection.
    pub fn pooled_copy_on_compute(
        &self,
        pool: &SharedHostPool,
        slot: usize,
        addr: &DevAddr,
        to_device: bool,
    ) -> Result<(), HwError> {
        let stream = self.live()?;
        pool.enqueue_copy_on_compute(&stream, slot, Some(addr.0.clone()), to_device)
            .map_err(HwError::PooledCopy)
    }

    /// The one body behind [`Self::d2h`], [`Self::d2h_on_compute`], and
    /// [`Self::d2h_on_compute_signaled`] — `signal` is the only difference
    /// between the doors, and it is the per-op completion hook: `None` leaves
    /// the op silent (the doors' behavior until #32), `Some` arms the read's
    /// own landing signal. One body, so a new guard cannot land on one door's
    /// shape and skip the others.
    fn d2h_shaped(
        &self,
        bytes: &mut [u8],
        addr: &DevAddr,
        shaping: DmaShaping,
        signal: Option<&ReadSignal>,
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
        stream
            .launch_d2h(params, signal.map(ReadSignal::arm))
            .map_err(HwError::Stream)
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
    ///
    /// ⭐ THE `*_signaled` TWIN OF THE DMA DOORS: `land` is the per-op
    /// completion hook (`launch_compute_batch`'s `user_cb`) — `None` leaves
    /// the submission silent (every door's behavior until now), `Some` arms
    /// the given landing on the ONE callback the whole-list submission
    /// carries. One callback per submission is the vendor's own semantics
    /// (one in-flight increment, fired when the whole list retires), so the
    /// landing armed here is the fact "this step's last control block has
    /// retired" — a READLESS completion door: no bytes cross, no D2H is
    /// owed, and the submitter returns while the card still owes the work.
    /// The submitter arms the card's step queue with the landing right
    /// after this call returns (the step door's arm-order law).
    fn compute_batch<'l>(
        &self,
        launches: impl Iterator<Item = &'l BatchedCompute>,
        pipeline_barrier: bool,
        land: Option<&ReadSignal>,
    ) -> Result<(), HwError> {
        let stream = self.live()?;
        let params: Vec<ComputeParams> = launches.map(BatchedCompute::params).collect();
        stream
            .launch_compute_batch(params, pipeline_barrier, land.map(ReadSignal::arm))
            .map_err(HwError::Stream)
    }

    /// Render the wire bytes of a staged batch WITHOUT submitting it (#603
    /// PR2's image-render door): the same `flex_rs::scheduler::build_compute_cb`
    /// the submit path runs per compute CB, and `build_rdma_send_cb` for a
    /// collective's send, over the same resolver TYPE the Pf backend
    /// resolves with (`FlexAllocatorAddressResolver` on the runtime's own
    /// allocator — the very `Arc` `build_runtime` handed the backend, so the
    /// region map both read is one map), at CALLER-chosen tags. Tags only
    /// reach the CTRL section's tag field; a caller comparing two renders
    /// feeds both sides the same `first_tag` and the field cancels — the
    /// 2b patch stage stamps real pool-leased tags there.
    ///
    /// Nothing here rings a doorbell, leases a tag, or touches a queue: the
    /// images are plain bytes until a caller submits them. The replay canvas
    /// (`cb_plan::WalkCanvas`) renders the walk's own staged batch THROUGH
    /// THIS ONE DOOR, so what it caches is exactly what the device would be
    /// handed.
    pub fn render_batch(
        &self,
        items: &[BatchedCb],
        first_tag: u32,
    ) -> Result<Vec<ControlBlockWire>, HwError> {
        let resolver = FlexAllocatorAddressResolver::new(
            self.rt
                .get_allocator()
                .map_err(|e| HwError::Alloc(AllocError::Context(e)))?,
        );
        let compute = |l: &BatchedCompute, tag: ResponseTag| {
            build_compute_cb(&resolver, &l.params(), tag).map_err(HwError::CbRender)
        };
        let mut cbs = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let tag = ResponseTag(first_tag + i as u32);
            cbs.push(match item {
                BatchedCb::Compute(l) => compute(l, tag)?,
                BatchedCb::Consume(l) => compute(l, tag)?.with_ctrl_barrier(),
                // ⭐ THE SEND WAITS FOR ITS PRODUCER: the barrier keeps the DMA engine from
                // reading the operand before the compute that writes it has retired.
                BatchedCb::Send(s) => build_rdma_send_cb(&resolver, &s.params(), tag, s.barrier)
                    .map_err(HwError::CbRender)?,
                // The wait is the scheduler submit's own CB (`submit_p2p_rdma_wait`): the Rdma10
                // wdone-barrier words, sync set, on an empty segment table.
                BatchedCb::Wait { sids } => ControlBlockWire::p2p_rdma_wait(
                    tag,
                    flex_rs::hal_sbf::rdma10_wdone_wait(
                        &sids.iter().map(|&s| u64::from(s)).collect::<Vec<_>>(),
                        true,
                    )
                    .map_err(|e| HwError::CbRender(format!("rdma10_wdone_wait: {e}").into()))?,
                ),
            });
        }
        Ok(cbs)
    }

    /// Submit a staged batch as ONE doorbell. A batch of plain compute launches takes
    /// `Self::compute_batch`, the door it always took; a batch that carries a collective's
    /// control blocks is rendered through [`Self::render_batch`] and submitted as wires
    /// ([`Self::compute_wires`]) with the leased tags stamped on, so the sends, waits and consumes
    /// ride the same FIFO as the compute that produces and consumes their bytes.
    ///
    /// ⭐ `land` is the `*_signaled` twin's hook — see [`Self::compute_batch`]'s
    /// ⭐ doc: `Some` arms the given landing on this submission's ONE
    /// completion callback, a readless door for a step whose pick (if it
    /// owes one) rides a LATER read. Rides both arms unchanged.
    pub fn submit_batch(
        &self,
        items: &[BatchedCb],
        pipeline_barrier: bool,
        land: Option<&ReadSignal>,
    ) -> Result<(), HwError> {
        if items.iter().all(|i| matches!(i, BatchedCb::Compute(_))) {
            return self.compute_batch(
                items.iter().filter_map(|i| match i {
                    BatchedCb::Compute(l) => Some(l),
                    _ => None,
                }),
                pipeline_barrier,
                land,
            );
        }
        let canvas = Arc::new(self.render_batch(items, 0)?);
        self.compute_wires(
            ComputeWireBatch {
                canvas,
                patch: Box::new(|tags, cbs| {
                    for (cb, tag) in cbs.iter_mut().zip(tags.iter()) {
                        cb.set_tag(*tag);
                    }
                }),
            },
            pipeline_barrier,
            land,
        )
    }

    /// #603 PR2's replay submit door — the write side of
    /// [`Self::render_batch`]'s images: submit a canvas the render
    /// door produced (and the prepare gates proved) as ONE doorbell, with
    /// the batch's stamp closure applying the step's patches and the
    /// pool-leased tags to the copy the backend takes under the submission
    /// lock. Same one-operation shape as `Self::compute_batch` — one
    /// barrier decision (`pipeline_barrier` is the OR of the step's per-op
    /// host fences), one fence, one capacity gate, in-flight accounting per
    /// CB — so the step is ordered on the ring exactly as its per-rung
    /// `compute_batch` calls were, with the same CBs in the same order.
    ///
    /// ⭐ `land` is the `*_signaled` twin's hook — see [`Self::compute_batch`]'s
    /// ⭐ doc: `Some` arms the given landing on this submission's ONE
    /// completion callback, a readless door for a step whose pick (if it
    /// owes one) rides a LATER read.
    pub fn compute_wires(
        &self,
        batch: ComputeWireBatch,
        pipeline_barrier: bool,
        land: Option<&ReadSignal>,
    ) -> Result<(), HwError> {
        let stream = self.live()?;
        stream
            .launch_compute_wires(batch, pipeline_barrier, land.map(ReadSignal::arm))
            .map_err(HwError::Stream)
    }
}

impl Drop for Stream<'_> {
    fn drop(&mut self) {
        // `destroy_stream` teardown-drains (`drain_for_teardown`: set
        // shutdown, wait for in-flight unconditionally — including on a
        // stream that took a device error, where `synchronize` returns
        // without waiting — resolved fences leave their deque at
        // completion now, `retire_fence`) and itself logs-and-swallows any
        // deferred error.
        let _ = self.rt.destroy_stream(self.id);
    }
}

/// ⭐ THE GUARD THAT CARRIES A PRE-MAPPED H2D'S SAFETY: it owns the IOMMU
/// mapping and borrows the host bytes that mapping was pinned from, so the two
/// facts a pre-mapped transfer states — which host bytes it reads, and which
/// iova those bytes map to — are DERIVED from one value and cannot disagree.
/// [`H2dPin::send_window`] takes a byte range and derives both: `src[range]`
/// for the bytes, `iova + range.start` for the address, the same linear
/// arithmetic the #606 pool's slot resolution rides and the law the scheduler
/// states by advancing a caller iova with its host address.
///
/// The borrow is the lifetime proof, and the guard's [`Drop`] is what makes it
/// hold on EVERY exit, not just the caller's happy path: a refused later
/// window, a failed synchronize, a panic unwind — each one drops the guard,
/// the guard SYNCHRONIZES THE STREAM BEFORE THE MAPPING UNPINS, and only then
/// does the pin go away. A transfer can never read an iova that is already
/// unpinned, or host pages the borrow no longer covers. A caller that holds
/// pins across its own synchronize and drops them after pays one extra
/// (already-drained) sync per guard, never a race.
///
/// Built by [`Stream::pin_h2d`], which refuses a mapping that does not cover
/// the whole slice ([`PremappedH2dRefusal::ShortPin`] — the shim's mapping
/// carries its first scatter-gather entry's length, not the size asked for).
pub struct H2dPin<'a, 'ctx> {
    src: &'a [u8],
    stream: &'a Stream<'ctx>,
    mapping: IommuMapping<'ctx>,
}

impl H2dPin<'_, '_> {
    /// Send byte range `range` of the pinned shadow into `window` — the same
    /// transfer as [`Stream::h2d`] on the same consumer path the #606 pool
    /// rides: `DmaShaping::iova` tells the backend the caller already pinned
    /// and IOMMU-mapped the host pages, so it skips its per-call DmaBuf setup
    /// and VFIO pin and DMAs straight from the pin's mapping.
    ///
    /// The pairing the recipe states, refused before submission (cardless,
    /// like [`HwError::OverrunsRegion`]): `window` is a SINGLE-CHUNK window —
    /// the transfer resolves one chunk, so a caller walking a multi-chunk
    /// region cuts it with [`DevAddr::windows`] first — the range's slice IS
    /// the window's extent, the pool's `copyRaw` law, and the range is inside
    /// the shadow the pin borrows. The slice and the iova both come from the
    /// guard, so they cannot disagree.
    pub fn send_window(
        &self,
        window: &DevAddr,
        range: std::ops::Range<usize>,
    ) -> Result<(), HwError> {
        let bytes = self.src.get(range.clone()).ok_or(HwError::PremappedH2d(
            PremappedH2dRefusal::RangeOutsideShadow {
                start: range.start as u64,
                end: range.end as u64,
                shadow: self.src.len() as u64,
            },
        ))?;
        premapped_pairing(bytes, window).map_err(HwError::PremappedH2d)?;
        // The mapping's iova maps `src[0]` (the constructor refused a shorter
        // pin), so `iova + range.start` maps `bytes` — the mapping's iova
        // advanced by the host offset. `bytes.len()` equals the window's
        // extent by the refusal above, so the explicit size is the window's
        // total size — the padded-extent over-read `h2d_shaped`'s doc guards
        // against cannot happen here: the equality is the proof the slice
        // spans the window.
        let params = DmaParams::with_explicit_size(
            HostVirtualAddress(bytes.as_ptr() as usize),
            ByteSize(bytes.len() as u64),
            DmaDirection::HostToDevice,
            Some(window.0.clone()),
            DmaShaping {
                iova: Some(IovaAddress(self.mapping.iova() + range.start)),
                ..Default::default()
            },
        )
        .expect("device_address is Some, constructed from an owned DevAddr");
        self.stream
            .live()?
            .launch_h2d(params, None)
            .map_err(HwError::Stream)
    }
}

impl Drop for H2dPin<'_, '_> {
    fn drop(&mut self) {
        // ⭐ THE PIN OUTLIVES ITS DMA, ON EVERY EXIT: synchronize BEFORE the
        // mapping unpins (`IommuMapping`'s own drop), so an in-flight transfer
        // never reads an iova that is already unpinned. Best-effort by
        // necessity — a refusal here is logged and the unpin proceeds, which
        // is no worse than the vendor's own RAII on a dying stream.
        if let Err(e) = self.stream.synchronize() {
            tracing::debug!("pre-mapped H2D pin: sync before unpin refused: {e}");
        }
    }
}

/// The byte types [`DevAddr`]'s signature speaks, re-exported so the player imports every device
/// address type from this one module — the typed card surface hands out the types it takes.
pub use flex_rs::address::{ByteOffset, ByteSize};
/// The replay payload [`Stream::compute_wires`] takes and the wire it is made of — same re-export
/// rule: the door hands out the types it takes, so the player names every submission type through
/// this one module.
pub use flex_rs::compute::ComputeWireBatch;
pub use flex_rs::control_block_wire::{ControlBlockWire, ResponseTag};
/// The world's placement the worker builds from its cards' ([`CardRuntime::cpus`]) and pins its
/// load and decode-loop threads to.
pub use flex_rs::numa::WorldCpus;

/// One compute launch of a [`Stream::submit_batch`] list — the owned handles one compute launch
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

impl BatchedCompute {
    /// The launch's parameters — `kernel_name=""` and no tensor byte offsets, the one shape every
    /// batched launch takes (see `Stream::compute_batch`).
    fn params(&self) -> ComputeParams {
        ComputeParams::new(
            self.prog.0.clone(),
            self.tensor_allocs.iter().map(|a| a.0.clone()).collect(),
            KernelName(String::new()),
            BootstrapOffset(self.bootstrap_offset),
            Vec::new(),
        )
    }
}

/// One control block of a staged batch ([`Stream::submit_batch`], [`Stream::render_batch`]).
pub enum BatchedCb {
    /// A compute launch.
    Compute(BatchedCompute),
    /// A collective's consume: the compute launch with its CTRL barrier bit set, so it cannot start
    /// before the write-done wait in front of it retires (#1007 §3).
    Consume(BatchedCompute),
    /// A collective's F-send into the peer's receive window.
    Send(RdmaSend),
    /// Wait for one message from each of several peers to land: one write-done count on each sid
    /// the PEERS signal for that message's landing slot — the wdone barrier's own sid set.
    Wait { sids: Vec<u8> },
}

/// One end of a send, addressed as the launch it is aimed from addresses it: the allocation backing
/// program segment `segment` for that launch, and the byte offset inside it.
pub struct RdmaEnd {
    pub base: DevAddr,
    pub segment: u8,
    pub offset: u64,
}

/// ONE MESSAGE of a collective's F-send, its endpoints still allocations (rendered by
/// `flex_rs::scheduler::build_rdma_send_cb`).
pub struct RdmaSend {
    /// The SENDING rank's write-done sid.
    pub sid: u8,
    /// The peer's BAT index.
    pub dest_tid: u8,
    pub bytes: scratchy_target_spyre::bundle_code::collective::WireBytes,
    pub source: RdmaEnd,
    pub window: RdmaEnd,
    /// This message's place in its send's run.
    pub place: flex_rs::scheduler::RunPlace,
    /// Whether this send waits for everything ahead of it to retire (the CTRL barrier). The FIRST
    /// send of a collective takes it — its operand is the compute just before it — and every later
    /// send of the same collective reads that same operand, already waited for, so it rides the
    /// FIFO behind the first without a barrier of its own.
    pub barrier: bool,
}

impl RdmaSend {
    fn params(&self) -> RdmaSendParams {
        let end = |e: &RdmaEnd| RdmaOperand {
            base: e.base.0.clone(),
            segment: e.segment,
            offset: e.offset,
        };
        RdmaSendParams {
            sid: self.sid,
            dest_tid: self.dest_tid,
            length: flex_rs::control_block_wire::TransferSize(self.bytes.get()),
            source: end(&self.source),
            window: end(&self.window),
            place: self.place,
        }
    }
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
/// allocation exactly as one handle and a borrow do. The batch door ([`Stream::submit_batch`])
/// needs it — a batch item outlives the loop that built its shifted bases, so it must own its
/// handles instead of borrowing them.
#[derive(Clone)]
pub struct DevAddr(Arc<CompositeAddress>);

impl DevAddr {
    /// Allocate `nbytes` of device memory of the given [`MemoryType`]. The region is never freed
    /// (see the struct doc); this handle describes it.
    pub fn alloc<T: ktir_superdsc::collective::Tp>(
        card: &CardRuntime<'_, T>,
        nbytes: ByteSize,
        kind: MemoryType,
    ) -> Result<DevAddr, HwError> {
        let rt = &card.rt;
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

    /// The WHOLE region as single-chunk windows of at most `step` bytes, chunk by chunk in the
    /// region's chunk order — one chunk for a Bind allocation, one per stripe for an interleaved one.
    /// Every window lies inside one chunk and together they cover each chunk's bytes exactly once.
    /// [`Self::window`] cannot build this walk: it addresses chunk 0 only, so past chunk 0's end it
    /// names bytes the region does not own.
    pub fn windows(&self, step: NonZeroU64) -> impl Iterator<Item = DevAddr> {
        let step = step.get();
        self.0.chunks().iter().flat_map(move |c| {
            (0..c.size.0).step_by(step as usize).map(move |off| {
                DevAddr::from_chunk(Chunk0 {
                    region: c.addr.region_id,
                    offset: ByteOffset(c.addr.offset.0 + off),
                    size: ByteSize(step.min(c.size.0 - off)),
                    domain: c.domain_id,
                })
            })
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

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  The p2p link — tensor parallelism's one load-time fact
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Why a sharded world's cards could not be linked. Each is a load refusal: a topology without
/// device p2p RDMA is refused, never served over the host.
#[derive(Debug)]
pub enum AttachRefusal<T: ktir_superdsc::collective::Tp> {
    /// A card is already linked, as another rank or for another window.
    AlreadyLinked {
        card: String,
        linked: (Rank<T>, RecvWindow<T>),
        asked: (Rank<T>, RecvWindow<T>),
    },
    /// A peer card's region-2 bus address could not be read.
    PeerRegion(String),
    /// The cross-port P2P route to a peer under another PCIe root could not be established
    /// ([`flex_rs::card::Card::p2p_route_peer`]). Without it that peer's region-2 address has
    /// no route from this card, so the world cannot run: this is the load-time refusal of a
    /// cross-root link that did not attach (#1192).
    P2pMap {
        card: String,
        peer: String,
        why: String,
    },
    /// A base-address-table entry could not be installed.
    Bat { tid: u8, why: String },
    /// A landing slot's xseg word did not compose, or its `XSEG` entry could not be programmed.
    Xseg(String),
}

impl<T: ktir_superdsc::collective::Tp> fmt::Display for AttachRefusal<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyLinked {
                card,
                linked,
                asked,
            } => write!(
                f,
                "the card at {card} is linked as {linked:?} and cannot link again as {asked:?} — one \
                 card's XSEG entries point at one window's slots, so every bundle it runs shares \
                 one window"
            ),
            Self::PeerRegion(e) => write!(f, "a peer card's region 2 could not be read: {e}"),
            Self::P2pMap { card, peer, why } => write!(
                f,
                "the cross-port p2p route from the card at {card} to the peer at {peer} could \
                 not be established: {why} — a peer under another PCIe root has no route to the \
                 region-2 address the base-address table names until the cross-port IOMMU mapping \
                 exists"
            ),
            Self::Bat { tid, why } => write!(f, "base-address-table entry {tid}: {why}"),
            Self::Xseg(e) => write!(f, "programming a landing slot's XSEG entry failed: {e}"),
        }
    }
}

impl<T: ktir_superdsc::collective::Tp> std::error::Error for AttachRefusal<T> {}

/// ⭐ ONE CARD'S P2P LINK — tensor parallelism's one load-time fact per card: which rank the card
/// runs as and where its peer's sends land. Minted only by [`link_peers`], which installs every
/// peer card's region-2 address in this card's base-address table (device p2p RDMA — never host
/// DMA) and points this card's `XSEG` entry for each of the peer's landing slots at that slot of
/// the receive window. A session on a linked card has done both, so its sends have somewhere to
/// land and its waits have something to count.
///
/// ONE PER CARD, held by its [`CardRuntime`]: the card's `XSEG` entries point at one window's slots,
/// so every bundle the card runs (decode, prefill, every rung) lands its peer's sends in the one
/// window the link programmed — [`crate::superdsc_exec::Executor::load`] refuses a bundle whose
/// window is elsewhere.
#[derive(Debug)]
pub struct PeerLink<T: ktir_superdsc::collective::Tp> {
    me: Rank<T>,
    window: RecvWindow<T>,
    /// The window's first byte as the ALLOCATOR placed it on this card — set once, by
    /// [`point_landing_slots`], when the owning session's weight segment exists.
    landing: std::sync::OnceLock<u64>,
}

impl<T: ktir_superdsc::collective::Tp> PeerLink<T> {
    /// The rank this card runs as.
    pub fn identity(&self) -> Rank<T> {
        self.me
    }

    /// Where every peer's sends land on this card.
    pub fn window(&self) -> RecvWindow<T> {
        self.window
    }

    /// The window's first byte as the allocator placed it — `None` until [`point_landing_slots`].
    pub fn landing(&self) -> Option<u64> {
        self.landing.get().copied()
    }
}

/// Link a sharded world's cards — `cards.get(r)` runs rank `r` of world `T` — for sends landing
/// in `window` on every card. One process, every card: each card's base-address table gets every
/// peer's region-2 address (entry `tid = peer rank`), and for every peer `s` its `XSEG` entry for
/// each of `s`'s landing slots (`Sid::of(s, slot)`) points at that slot of `s`'s block of its own
/// window. Linking a card again as the same rank for the same window keeps the link it has.
///
/// ⭐ THE CROSS-ROOT ROUTE (#1192): the BAT entry names the peer's region-2 bus address, which a
/// peer under the same PCIe root is reachable at directly. A peer under another root has no route
/// to that address until the cross-port P2P IOMMU mapping exists — so linking establishes the
/// route ([`flex_rs::card::Card::p2p_route_peer`], senlib's
/// `create_iommu_mapping_for_cross_port_p2p_rdma`) for exactly the cross-root peers, at link,
/// before the entry that names the address is installed. Same-root pairs take the direct route
/// they have always taken; nothing about their path changes.
///
/// # Errors
/// [`AttachRefusal`], by name.
pub fn link_peers<T: ktir_superdsc::collective::Tp>(
    cards: &PerRank<T, CardRuntime<'_, T>>,
    window: RecvWindow<T>,
) -> Result<(), AttachRefusal<T>> {
    let region2: PerRank<T, u64> =
        PerRank::try_from_fn(|r| flex_rs::card::pci_addr_for_region(cards.get(r).card.pci(), 2))
            .map_err(AttachRefusal::PeerRegion)?;
    for (me, c) in cards.iter() {
        if let Some(link) = c.link.get() {
            if (link.me, link.window) == (me, window) {
                continue;
            }
            return Err(AttachRefusal::AlreadyLinked {
                card: c.card.pci().to_string(),
                linked: (link.me, link.window),
                asked: (me, window),
            });
        }
        // ⭐ THE SYMMETRIC BAT, AS THE tp37 CELL INSTALLS IT (#1008): entry `tid` — a peer's rank,
        // the destination every send names (`cb_plan::base_addr_entry`) — holds that peer card's
        // region-2 PCI address, from senlib's own region-address door (the vendor's
        // `initialize_p2p_address_table` value source), for the card the address names, never an
        // enumeration order. The peer's route to that address is established first: a peer under
        // another PCIe root is unreachable until its cross-port mapping exists.
        for peer in me.peers() {
            c.card
                .p2p_route_peer(cards.get(peer).card())
                .map_err(|why| AttachRefusal::P2pMap {
                    card: c.card.pci().to_string(),
                    peer: cards.get(peer).card.pci().to_string(),
                    why,
                })?;
            let addr = *region2.get(peer);
            let tid = crate::cb_plan::base_addr_entry(peer);
            c.card
                .set_base_addr_table(tid, addr)
                .map_err(|why| AttachRefusal::Bat { tid, why })?;
        }
        // Set only here, after the card is fully programmed; `get` above saw none.
        let _ = c.link.set(PeerLink {
            me,
            window,
            landing: std::sync::OnceLock::new(),
        });
    }
    Ok(())
}

/// ⭐ POINT EVERY PEER'S LANDING SLOTS AT THIS CARD'S WINDOW — one `XSEG` entry per (peer, slot),
/// each programmed once: an F-send lands where its sid's entry points and nowhere else (#1009, E8a),
/// so peer `from`'s message into slot `s` signals `Sid::of(from, s)` and that entry names the slot —
/// `hal_sbf::rdma_xseg` of the slot's flat VA plus the one-flit L4 compensation, the VA from
/// [`flex_rs::scheduler::landing_va_bytes`], the same compose the send's own word takes.
///
/// ⛔ THE VA IS THE ALLOCATOR'S, SO THIS RUNS ONCE THE WINDOW IS ALLOCATED. The window is a placement
/// in the weight segment, and where that segment's bytes ARE — the region's segment and the
/// allocation's offset inside it — exists only once the session that OWNS it has allocated it
/// (`window_region`; every other session aliases it). MEASURED, #1009 run 5 and again here on
/// 2026-11-06: entries pointed at `(program seg 1, layout offset)` name an address the allocator gave
/// to something else, and every collective runs clean over a window the peers' bytes never reach.
///
/// Every rank's window must land at ONE address: each send names its own window's address for its
/// peers' ([`flex_rs::scheduler::build_rdma_send_cb`]). The cards of one process allocate one
/// sequence, so they do; [`refuse_if_windows_apart`] checks it once every rank is pointed.
///
/// # Errors
/// [`AttachRefusal`]: an unlinked card, a slot past the allocation, a second pointing at another
/// address, or a word that will not compose or program.
pub fn point_landing_slots<T: ktir_superdsc::collective::Tp>(
    card: &CardRuntime<'_, T>,
    window_region: &DevAddr,
) -> Result<(), AttachRefusal<T>> {
    let Some(link) = card.link.get() else {
        return Err(AttachRefusal::Xseg(
            "point_landing_slots: the card is not linked".to_string(),
        ));
    };
    let resolver = FlexAllocatorAddressResolver::new(
        card.rt
            .get_allocator()
            .map_err(|e| AttachRefusal::Xseg(format!("allocator: {e:?}")))?,
    );
    let at = |offset: u64, len: u64| {
        flex_rs::scheduler::landing_va_bytes(
            &resolver,
            &RdmaOperand {
                base: window_region.0.clone(),
                segment: WINDOW_SEGMENT as u8,
                offset,
            },
            len,
        )
        .map_err(|e| AttachRefusal::Xseg(e.to_string()))
    };
    let window = link.window;
    let landing = at(window.offset(), window.size())?;
    if link.landing.get().is_some_and(|l| *l != landing) {
        return Err(AttachRefusal::Xseg(format!(
            "the window is pointed at 0x{landing:x}, but this card already pointed it at {:x?}",
            link.landing.get()
        )));
    }
    if link.landing.get().is_some() {
        return Ok(());
    }
    let cap = scratchy_target_spyre::bundle_code::collective::RDMA_MESSAGE_CAP_BYTES;
    for (from, (slot, off)) in link
        .me
        .peers()
        .flat_map(|from| window.slots_from(from).map(move |sa| (from, sa)))
    {
        // The slot's own extent: one message, or the block's remainder for its last.
        let len = (window.block().get() - u64::from(slot.message()) * cap).min(cap);
        let va = at(off, len)?;
        let word = flex_rs::hal_sbf::rdma_xseg(va + flex_rs::control_block_wire::Flits::BYTES)
            .map_err(AttachRefusal::Xseg)?;
        card.card
            .write_xseg(Sid::of(from, slot).get(), word)
            .map_err(AttachRefusal::Xseg)?;
    }
    let _ = link.landing.set(landing);
    tracing::info!(
        "[spyre-tp] rank {}: landing slots pointed at the receive window, landing 0x{landing:x}",
        link.me.get()
    );
    Ok(())
}

/// The ranks of a world placed their receive windows at different device addresses: a send names
/// its own window's address for its peers', so this world would send where its peers do not land.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowsApart {
    pub landings: Vec<Option<u64>>,
}

impl fmt::Display for WindowsApart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the ranks' receive windows land at {:x?} — every rank of a world must place its window \
             at one address, which each send names for its peers'",
            self.landings
        )
    }
}

impl std::error::Error for WindowsApart {}

/// ⭐ EVERY RANK'S WINDOW AT ONE ADDRESS — checked once every card is pointed, before a step.
///
/// # Errors
/// [`WindowsApart`], naming every rank's landing.
pub fn refuse_if_windows_apart<T: ktir_superdsc::collective::Tp>(
    cards: &PerRank<T, CardRuntime<'_, T>>,
) -> Result<u64, WindowsApart> {
    let landings: Vec<Option<u64>> = cards
        .values()
        .map(|c| c.link().and_then(PeerLink::landing))
        .collect();
    match landings.first().copied().flatten() {
        Some(l) if landings.iter().all(|x| *x == Some(l)) => Ok(l),
        _ => Err(WindowsApart { landings }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(region: u64, offset: u64, size: u64, domain: u32) -> Chunk {
        Chunk::new(
            LogicalAddress::new(RegionId(region), ByteOffset(offset)),
            ByteSize(size),
            DomainId(domain),
        )
    }

    fn walk(addr: &DevAddr, step: u64) -> Vec<(u64, u64, u64, u32)> {
        addr.windows(NonZeroU64::new(step).expect("a nonzero step"))
            .map(|w| {
                assert!(w.is_single_chunk(), "a window is one chunk");
                let c = w.chunk0().expect("a window has a chunk");
                (c.region.0, c.offset.0, c.size.0, c.domain.0)
            })
            .collect()
    }

    /// ⭐ THE ZERO-FILL WALK OVER AN INTERLEAVED REGION (#672). `zero_seg` fills exactly these
    /// windows on the card, so this is the address half of zeroing a multi-chunk KV pool — the
    /// half a card is not needed for. Each chunk is cut at the step in its OWN
    /// region/offset/domain, a short tail stays short, a zero-length chunk sends nothing, and
    /// the bytes sum to the region's size.
    #[test]
    fn windows_cover_each_chunk_of_an_interleaved_region_once() {
        let addr = DevAddr(Arc::new(CompositeAddress::from_chunks(vec![
            chunk(7, 4096, 20, 0),
            chunk(9, 0, 8, 1),
            chunk(3, 64, 0, 1),
            chunk(7, 1 << 20, 13, 0),
        ])));
        assert!(!addr.is_single_chunk());
        let got = walk(&addr, 8);
        assert_eq!(
            got,
            vec![
                (7, 4096, 8, 0),
                (7, 4104, 8, 0),
                (7, 4112, 4, 0),
                (9, 0, 8, 1),
                (7, 1 << 20, 8, 0),
                (7, (1 << 20) + 8, 5, 0),
            ]
        );
        assert_eq!(
            got.iter().map(|w| w.2).sum::<u64>(),
            addr.total_size().as_u64()
        );
    }

    /// The Bind shape — the only one `alloc` makes today — walks exactly as `zero_seg`'s old
    /// `window(off, min(step, total - off))` loop did, so the change moves no transfer on the card.
    #[test]
    fn windows_of_a_single_chunk_region_match_the_offset_loop() {
        let addr = DevAddr(Arc::new(CompositeAddress::from_chunk(chunk(5, 256, 17, 0))));
        let mut old = Vec::new();
        let mut off = 0;
        while off < 17 {
            let sz = 8.min(17 - off);
            let c = addr
                .window(ByteOffset(off), ByteSize(sz))
                .and_then(|w| w.chunk0())
                .expect("window builds a value");
            old.push((c.region.0, c.offset.0, c.size.0, c.domain.0));
            off += sz;
        }
        assert_eq!(walk(&addr, 8), old);
        assert_eq!(old.len(), 3);
    }

    /// ⭐ THE PRE-MAPPED DOOR'S REFUSALS ARE CARDLESS — the two facts of
    /// [`H2dPin::send_window`]'s pairing, stated in [`premapped_pairing`] apart
    /// from the door so they are checkable without a runtime: a window that
    /// spans chunks, and a slice that is not the window's extent, are both
    /// refused BY NAME before any stream is touched. The paired case — one
    /// chunk, slice == extent — is the walk `Executor::prepare` feeds the door,
    /// covered by the window-walk tests above.
    #[test]
    fn premapped_pairing_refuses_a_multi_chunk_window_and_a_mislength_slice() {
        let multi = DevAddr(Arc::new(CompositeAddress::from_chunks(vec![
            chunk(7, 4096, 20, 0),
            chunk(9, 0, 8, 1),
        ])));
        let bytes = vec![0u8; 28];
        match premapped_pairing(&bytes, &multi) {
            Err(PremappedH2dRefusal::NotOneChunk { chunks }) => assert_eq!(chunks, 2),
            other => panic!("a 2-chunk window is refused by name, got {other:?}"),
        }

        let one = DevAddr(Arc::new(CompositeAddress::from_chunk(chunk(5, 256, 17, 0))));
        match premapped_pairing(&bytes, &one) {
            Err(PremappedH2dRefusal::LengthMismatch { len, window }) => {
                assert_eq!((len, window), (28, 17));
            }
            other => panic!("a 28 B slice into a 17 B window is refused by name, got {other:?}"),
        }

        let one28 = DevAddr(Arc::new(CompositeAddress::from_chunk(chunk(5, 256, 28, 0))));
        assert!(premapped_pairing(&bytes, &one28).is_ok());
    }

    /// The two refusals that need a live mapper to fire — a pin that came back
    /// shorter than the shadow it was asked to map, and a byte range outside
    /// the pinned shadow — still NAME their facts when they do: the numbers a
    /// split pin or an overruning walk would have to be diagnosed from are in
    /// the refusal's own text. The refusals themselves are constructed by the
    /// card ([`Stream::pin_h2d`], [`H2dPin::send_window`]); their words are a
    /// fact about the types, checkable cardlessly.
    #[test]
    fn premapped_refusals_name_a_short_pin_and_an_outside_range() {
        let short = PremappedH2dRefusal::ShortPin {
            asked: 5_130_000_000,
            pinned: 4_294_967_296,
        };
        let text = short.to_string();
        assert!(
            text.contains("5130000000"),
            "names the bytes asked for: {text}"
        );
        assert!(
            text.contains("4294967296"),
            "names the bytes pinned: {text}"
        );

        let outside = PremappedH2dRefusal::RangeOutsideShadow {
            start: 64,
            end: 128,
            shadow: 96,
        };
        let text = outside.to_string();
        assert!(text.contains("64..128"), "names the range: {text}");
        assert!(text.contains("96"), "names the shadow's extent: {text}");
    }
}

#[cfg(test)]
mod world_cards_tests {
    use super::{CardTable, ClaimRefusal, WorldCards};
    use scratchy_target_spyre::bundle_code::collective::{Rank, TpWorld};

    fn table(pcis: &[&str]) -> CardTable {
        CardTable::of(
            pcis.iter()
                .enumerate()
                .map(|(i, p)| (i as u32, p.to_string())),
        )
    }

    /// ⭐ THE WORLD IS THE TABLE'S FIRST `RANKS` CARDS, IN ITS ORDER: rank `r` holds the `r`-th.
    #[test]
    fn a_world_claims_the_first_cards_in_order() {
        let t = table(&["0000:1a:00.0", "0000:1b:00.0", "0000:1c:00.0"]);
        let w = WorldCards::<TpWorld<2>>::claim(&t).unwrap();
        assert_eq!(w.per_rank().get(Rank::of::<0>()).pci(), "0000:1a:00.0");
        assert_eq!(w.per_rank().get(Rank::of::<1>()).pci(), "0000:1b:00.0");
        assert_eq!(w.describe(), "r0=0000:1a:00.0 r1=0000:1b:00.0");
    }

    /// A world bigger than the host refuses, by name.
    #[test]
    fn a_world_bigger_than_the_host_is_refused() {
        let t = table(&["0000:1a:00.0"]);
        assert_eq!(
            WorldCards::<TpWorld<2>>::claim(&t).unwrap_err(),
            ClaimRefusal::TooFewCards {
                needed: 2,
                enumerated: 1
            }
        );
    }

    /// ⭐ DISTINCT BY CONSTRUCTION: a table naming one address twice inside the world refuses; past
    /// the world's cards, a repeat is not the world's concern.
    #[test]
    fn a_repeated_card_inside_the_world_is_refused() {
        let t = table(&["0000:1a:00.0", "0000:1a:00.0", "0000:1c:00.0"]);
        assert_eq!(
            WorldCards::<TpWorld<2>>::claim(&t).unwrap_err(),
            ClaimRefusal::RepeatedCard {
                pci: "0000:1a:00.0".to_string()
            }
        );
        let t = table(&["0000:1a:00.0", "0000:1b:00.0", "0000:1b:00.0"]);
        assert!(WorldCards::<TpWorld<2>>::claim(&t).is_ok());
    }
}
