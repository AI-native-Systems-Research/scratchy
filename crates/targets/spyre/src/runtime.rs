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
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, OnceLock};

// ⭐ THE COLLECTIVE PLAN'S TWO WIRE FACTS ARE THE RUNTIME'S. The bundle crate is a leaf and cannot
// name flex-rs, so it declares the message cap and the flit itself; this is the one place both
// crates are in scope, and a disagreement fails the build here rather than cutting a fragment the
// card refuses or misaligning a window.
const _: () = {
    assert!(
        crate::bundle_code::collective::RDMA_MESSAGE_CAP_BYTES
            == flex_rs::multi_device::MAX_HDMA_MESSAGE_BYTES
    );
    assert!(
        crate::bundle_code::collective::FLIT_BYTES == flex_rs::control_block_wire::Flits::BYTES
    );
};

use crate::bundle_code::collective::{LandingSlot, PeerRegion, RecvWindow, Sid, TpIdentity};
use flex_rs::address::{Chunk, CompositeAddress, LogicalAddress, RegionId};
use flex_rs::allocator::{
    AllocationDirective, AllocationDirectiveError, FlexAllocator, FlexAllocatorError,
    FlexAllocatorInit,
};
use flex_rs::compute::{BootstrapOffset, ComputeParams, KernelName};
use flex_rs::device_memory_allocator::DeviceMemoryAllocator;
use flex_rs::dma::{
    DmaDirection, DmaParams, DmaPipelineShape, DmaShaping, FillParams, FillPattern, FillPipeline,
    HostVirtualAddress,
};
use flex_rs::flex_config::DeviceKind;
use flex_rs::memory_region::{DomainId, MemoryType};
use flex_rs::runtime::{RuntimeContext, RuntimeContextError};
use flex_rs::scheduler::{
    DeviceAddressResolver, FlexAllocatorAddressResolver, HardwareSubmit, QueueCapacitySource,
    RdmaOperand, RdmaSendParams, SchedulerBackendConfig, SchedulerFactoryError,
    SenlibHardwareSubmit, SenlibQueueCapacity, build_compute_cb, build_rdma_send_cb,
    create_scheduler,
};
use flex_rs::shared_host_pool::{SharedHostPool, SharedHostPoolError};
use flex_rs::stream::{
    RuntimeStream, RuntimeStreamError, RuntimeStreamMode, RuntimeStreamPriority, SchedulerError,
    StreamId,
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
    match card_cell().get() {
        // ⭐ THE BIND, AND ITS PROOF (#1008's doors, as the tp37 cell runs them): open the card
        // by its address, then hold senlib's own resolution of the open card to it before
        // anything else touches the device.
        Some(pci) => {
            flex_rs::senlib_ffi::init_runtime_bdf(pci.as_str()).map_err(|why| {
                BringUpError::OpenAt {
                    pci: pci.clone(),
                    why,
                }
            })?;
            let resolved = flex_rs::senlib_ffi::resolved_pci();
            if !resolved
                .as_deref()
                .is_ok_and(|r| r.eq_ignore_ascii_case(pci.as_str()))
            {
                return Err(BringUpError::WrongCard {
                    asked: pci.clone(),
                    resolved,
                });
            }
        }
        // No card was chosen: the shim's first available device, as every one-card host runs.
        None => {
            let rc = unsafe { flex_rs::senlib_ffi::flex_shim_init_runtime(0) };
            if rc != 0 {
                return Err(BringUpError::ShimInit(rc));
            }
        }
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

/// ⭐ A CARD'S PCI ADDRESS — `dddd:bb:dd.f` (domain, bus, device, function), lowercase: what senlib
/// selects a card by (`flex_rs::senlib_ffi::init_runtime_bdf`, `SenPci::pf(const std::string&)`)
/// and reports back (`resolved_pci`, the same lowercase form). Minted only by
/// [`PciAddress::parse`], so a value is always one address the doors can be handed whole.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PciAddress(String);

/// A string that is not a `dddd:bb:dd.f` PCI address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotAPciAddress(pub String);

impl fmt::Display for NotAPciAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} is not a PCI address — a card is named `dddd:bb:dd.f` (domain, bus, device, \
             function), e.g. `0000:3b:00.0`",
            self.0
        )
    }
}

impl std::error::Error for NotAPciAddress {}

impl PciAddress {
    /// `Ok` iff `s` is `dddd:bb:dd.f`: four, two and two hex digits and a function 0–7. Hex
    /// digits are taken in either case and kept lowercase.
    ///
    /// # Errors
    /// [`NotAPciAddress`].
    pub fn parse(s: &str) -> Result<Self, NotAPciAddress> {
        let hex = |p: &str, n: usize| p.len() == n && p.bytes().all(|b| b.is_ascii_hexdigit());
        let well_formed = match s.split([':', '.']).collect::<Vec<_>>().as_slice() {
            [domain, bus, device, function] => {
                hex(domain, 4)
                    && hex(bus, 2)
                    && hex(device, 2)
                    && matches!(function.as_bytes(), [b'0'..=b'7'])
                    && s.as_bytes()[4] == b':'
                    && s.as_bytes()[7] == b':'
                    && s.as_bytes()[10] == b'.'
            }
            _ => false,
        };
        if well_formed {
            Ok(Self(s.to_ascii_lowercase()))
        } else {
            Err(NotAPciAddress(s.to_string()))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PciAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Which card a process's runtime opens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrivenCard {
    /// The shim's first available device — what a process that chose none opens.
    FirstAvailable,
    /// The card at this address ([`drive_card`]).
    At(PciAddress),
}

impl fmt::Display for DrivenCard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FirstAvailable => f.write_str("the first available card"),
            Self::At(pci) => write!(f, "the card at {pci}"),
        }
    }
}

fn card_cell() -> &'static OnceLock<PciAddress> {
    static CARD: OnceLock<PciAddress> = OnceLock::new();
    &CARD
}

/// The card this process's runtime opens: the one [`drive_card`] chose, else the first available.
pub fn driven_card() -> DrivenCard {
    card_cell().get().map_or(DrivenCard::FirstAvailable, |pci| {
        DrivenCard::At(pci.clone())
    })
}

/// The process already drives another card. A process holds ONE runtime (senlib's PF handle is
/// process-global), so it drives one card for its life.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardTaken {
    pub driving: DrivenCard,
    pub asked: PciAddress,
}

impl fmt::Display for CardTaken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "this process already drives {} and cannot open the card at {} — one runtime per \
             process, one card per runtime",
            self.driving, self.asked
        )
    }
}

impl std::error::Error for CardTaken {}

/// Choose the card this process drives, by its PCI address, BEFORE anything brings the runtime up
/// (a bring-up with no choice opens the first available card and fixes it). Choosing the card
/// already chosen is a no-op.
///
/// # Errors
/// [`CardTaken`] when the process already drives (or chose) another card.
pub fn drive_card(card: &PciAddress) -> Result<(), CardTaken> {
    // Held across the choice: a bring-up reads the choice under this same lock, so none can start
    // between the check and the choice.
    let runtime = runtime_cell().lock().unwrap_or_else(|e| e.into_inner());
    let driving = if runtime.is_some() {
        driven_card()
    } else {
        DrivenCard::At(card_cell().get_or_init(|| card.clone()).clone())
    };
    drop(runtime);
    if driving == DrivenCard::At(card.clone()) {
        Ok(())
    } else {
        Err(CardTaken {
            driving,
            asked: card.clone(),
        })
    }
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
    /// Rendering compute-CB wire images OFF the submit path refused — the CB
    /// encoder cannot express an operand ([`Stream::render_batch`]).
    /// Never a submission: nothing was sent to the device.
    CbRender(SchedulerError),
    /// A #606 pooled copy ([`Stream::pooled_copy_on_compute`]) was refused by the
    /// pool's own validation set — slot range, single-chunk device, the
    /// slot-capacity transfer law — or the submission it built was rejected.
    PooledCopy(SharedHostPoolError),
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
        }
    }
}
impl std::error::Error for HwError {}

/// Why the process-wide runtime could not be constructed — the refusing layer's own error.
#[derive(Debug)]
pub enum BringUpError {
    /// `flex_shim_init_runtime` — the senlib device init — returned a nonzero rc.
    ShimInit(i32),
    /// The card the process chose ([`drive_card`]) would not open at its address.
    OpenAt { pci: PciAddress, why: String },
    /// The open card is not the one asked for: senlib resolved another address (or none).
    WrongCard {
        asked: PciAddress,
        resolved: Result<String, String>,
    },
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
            Self::OpenAt { pci, why } => write!(f, "the card at {pci} did not open: {why}"),
            Self::WrongCard { asked, resolved } => match resolved {
                Ok(r) => write!(
                    f,
                    "asked for the card at {asked}, but senlib opened {r} — the bind did not \
                     select the card"
                ),
                Err(why) => write!(
                    f,
                    "asked for the card at {asked}, but senlib resolves no open card: {why}"
                ),
            },
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

/// ⭐ #32 — ONE DEFERRED READ'S OWN COMPLETION: armed by the D2H op's user
/// callback (the per-op completion hook every `launch_*` has always carried and
/// every seat door has always passed `None`), parked on by the read's finish
/// half instead of a whole-stream synchronize.
///
/// NOT flex-rs's `Event`, and the difference is load-bearing: an op that FAILS
/// never fires its user callback — the stream records a deferred error and goes
/// sticky-shutdown instead (`make_completion_callback`'s Err arm) — so a waiter
/// with no timeout would hang on a dead device. This latch parks on a condvar
/// with a TIMEOUT tick whose only job is that escape: in the healthy path the
/// callback's notify returns the park and the tick never fires; when the tick
/// does fire it hands the wait back so the caller can surface the stream's
/// deferred error by the same door the whole-stream wait used.
///
/// One signal names one transfer, and the finish half that parks on it is
/// its only WAITER (the ring law — a slot's bytes are read exactly once, by
/// the drain that owns the handle). The overlap probe gets its non-blocking
/// view through a separate HANDLE TYPE ([`ReadSignal::peek_handle`]) whose
/// whole surface is `fired` — the no-waiting law carried by the TYPE, not
/// by a comment: a handle with no `wait` method cannot break the ring law,
/// so the signal itself is not `Clone`.
pub struct ReadSignal {
    done: std::sync::Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}

/// ⭐ #32 (E) — THE PEEK HANDLE: a read-only view of one transfer's
/// completion latch, minted only by [`ReadSignal::peek_handle`]. Its whole
/// surface is [`ReadPeek::fired`] — one latch read, never a park, never a
/// byte of the slot — so the overlap probe can hold it across a step
/// without being able to wait on it. The waiter's law stays on
/// `ReadSignal` itself, which this type deliberately cannot become.
pub struct ReadPeek(std::sync::Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>);

impl ReadPeek {
    /// Has the transfer's callback fired? The card-busy answer the overlap
    /// probe asks at each submit about the PREVIOUS step's read — "not yet
    /// fired" is the card owing work, the overlap fraction the depth row
    /// cannot prove (depth counts submits-outstanding, which the
    /// pre-schedule could inflate while the card sat idle; this cannot lie
    /// that way).
    pub fn fired(&self) -> bool {
        let (mtx, _cv) = &*self.0;
        *mtx.lock().unwrap()
    }
}

/// The timeout tick — the error-escape cadence. An order past a decode step
/// (~24 ms measured, ~48 ms at depth 2), so it never fires on a healthy wait;
/// it exists only to notice a dead stream (≤ this much latency on a path that
/// is already a hard failure).
const READ_SIGNAL_DEAD_TICK: std::time::Duration = std::time::Duration::from_millis(250);

impl ReadSignal {
    pub fn new() -> Self {
        Self {
            done: std::sync::Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new())),
        }
    }

    /// The closure handed to `launch_d2h` as the op's user callback: set the
    /// flag and wake the parker. Runs on the scheduler's completion thread,
    /// touches nothing but this latch, and cannot borrow the transfer's host
    /// buffer — the buffer outlives the call site by the caller's own laws
    /// (the ring slot's index, the shadow's seat ownership).
    fn arm(&self) -> Box<dyn FnOnce() + Send> {
        let done = self.done.clone();
        Box::new(move || {
            let (mtx, cv) = &*done;
            *mtx.lock().unwrap() = true;
            cv.notify_all();
        })
    }

    /// ⭐ #32 (E) — MINT THE PEEK HANDLE: the overlap probe's non-blocking
    /// view of this transfer's completion. [`ReadPeek::fired`] is the whole
    /// of what the holder can do — by type, not by comment, the handle
    /// cannot wait, so the drain's park on this signal stays its one waiter
    /// no matter who holds a peek.
    pub fn peek_handle(&self) -> ReadPeek {
        ReadPeek(self.done.clone())
    }

    /// Park until the read's own callback fires. `dead` runs ONLY on the
    /// timeout tick — the error escape above — and answers "can the op this
    /// wait is for still complete at all". Returns `true` when the transfer
    /// landed, `false` when the stream died underneath it.
    pub fn wait_or(&self, dead: impl Fn() -> bool) -> bool {
        let (mtx, cv) = &*self.done;
        let mut guard = mtx.lock().unwrap();
        loop {
            if *guard {
                return true;
            }
            let (guard_next, _timeout) = cv.wait_timeout(guard, READ_SIGNAL_DEAD_TICK).unwrap();
            guard = guard_next;
            if *guard {
                return true;
            }
            if dead() {
                return false;
            }
        }
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
    /// Create a dedicated stream: `Normal` priority, `STRICT_ORDERING`, untracked — the values
    /// every scratchy session has ever run. Sessions stay UNTRACKED deliberately: their streams
    /// are user-owned and order through their own submissions; the hazard tracker's conservative
    /// Compute address sets would cross-edge concurrent sessions sharing resident weights into
    /// full serialization. Delegates to [`Self::create_with`].
    pub fn create() -> Result<Stream, HwError> {
        Self::create_with(
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
    pub fn create_with(
        priority: RuntimeStreamPriority,
        mode: RuntimeStreamMode,
        track_hazards: bool,
    ) -> Result<Stream, HwError> {
        // A poisoned lock still serializes (nothing is protected but the CALL), so recover rather than
        // panic: a panic here would turn one thread's failure into the whole load's.
        let _serialize = STREAM_CREATE.lock().unwrap_or_else(|e| e.into_inner());
        let rt = get_or_init_runtime().map_err(HwError::BringUp)?;
        let id = rt
            .create_stream(priority, mode, track_hazards)
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
    fn compute_batch<'l>(
        &self,
        launches: impl Iterator<Item = &'l BatchedCompute>,
        pipeline_barrier: bool,
    ) -> Result<(), HwError> {
        let stream = self.live()?;
        let params: Vec<ComputeParams> = launches.map(BatchedCompute::params).collect();
        stream
            .launch_compute_batch(params, pipeline_barrier, None)
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
    /// images are plain bytes until a caller submits them. The parity gate
    /// (`superdsc_exec::Executor::prepare`) renders one step's batch both the
    /// walk way and the CbPlan way THROUGH THIS ONE DOOR, so what it compares
    /// is exactly what the device would be handed.
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
                BatchedCb::Send(s) => build_rdma_send_cb(&resolver, &s.params(), tag, true)
                    .map_err(HwError::CbRender)?,
                // `submit_p2p_rdma_wait`'s CB, in the stream: #1008's wdone door (`sync` set),
                // placed by its builder.
                BatchedCb::Wait { sid } => ControlBlockWire::p2p_rdma_wait(
                    tag,
                    flex_rs::hal_sbf::rdma10_wdone_wait(&[u64::from(*sid)], true)
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
    pub fn submit_batch(&self, items: &[BatchedCb], pipeline_barrier: bool) -> Result<(), HwError> {
        if items.iter().all(|i| matches!(i, BatchedCb::Compute(_))) {
            return self.compute_batch(
                items.iter().filter_map(|i| match i {
                    BatchedCb::Compute(l) => Some(l),
                    _ => None,
                }),
                pipeline_barrier,
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
    pub fn compute_wires(
        &self,
        batch: ComputeWireBatch,
        pipeline_barrier: bool,
    ) -> Result<(), HwError> {
        let stream = self.live()?;
        stream
            .launch_compute_wires(batch, pipeline_barrier, None)
            .map_err(HwError::Stream)
    }
}

impl Drop for Stream {
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

/// The byte types [`DevAddr`]'s signature speaks, re-exported so the player imports every device
/// address type from this one module — the typed card surface hands out the types it takes.
pub use flex_rs::address::{ByteOffset, ByteSize};
/// The replay payload [`Stream::compute_wires`] takes and the wire it is made of — same re-export
/// rule: the door hands out the types it takes, so the player names every submission type through
/// this one module.
pub use flex_rs::compute::ComputeWireBatch;
pub use flex_rs::control_block_wire::{ControlBlockWire, ResponseTag};

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
    /// Wait for one of the peer's messages to land: one write-done count on the sid the PEER
    /// signals for that message's landing slot.
    Wait { sid: u8 },
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
    pub bytes: crate::bundle_code::collective::WireBytes,
    pub source: RdmaEnd,
    pub window: RdmaEnd,
    /// This message's place in its send's run.
    pub place: flex_rs::scheduler::RunPlace,
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

/// The cards a sharded session spans: the one this process drives and its peer's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardPair {
    pub own: PciAddress,
    pub peer: PciAddress,
}

/// The ids a rank's collective control blocks name: per landing slot, its own sid (its message
/// into that slot signals on it) and the peer's (its wait for the peer's message into that slot
/// counts on it, and this card's `XSEG` entry for it points at the slot); and the peer's BAT index
/// (where its sends land).
///
/// The BAT index is the peer's RANK — the vendor's `dest_rank`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerRoute {
    me: TpIdentity,
    peer: TpIdentity,
    /// This rank's region of the peer's window — where its sends land.
    lands_in: PeerRegion,
    /// The peer's region of this card's window — where the peer's sends land.
    hears_in: PeerRegion,
}

impl PeerRoute {
    /// The route between `me` and its peer. `None` in a world of one.
    pub fn of(me: TpIdentity) -> Option<Self> {
        let peer = me.peer()?;
        Some(Self {
            me,
            peer,
            lands_in: peer.region_of(me)?,
            hears_in: me.region_of(peer)?,
        })
    }

    /// This rank's region of the peer's window: where its messages land on the peer's card.
    pub fn lands_in(self) -> PeerRegion {
        self.lands_in
    }

    /// The peer's region of this card's window: where the peer's messages land, and where this
    /// card's `XSEG` entries for the peer's slots point.
    pub fn hears_in(self) -> PeerRegion {
        self.hears_in
    }

    /// The sid this rank's message into `slot` signals on.
    pub fn send_sid(self, slot: LandingSlot) -> u8 {
        Sid::of(self.me, slot).get()
    }

    /// The sid this rank waits on for the peer's message into `slot` — and the `XSEG` entry this
    /// card points at the slot.
    pub fn wait_sid(self, slot: LandingSlot) -> u8 {
        Sid::of(self.peer, slot).get()
    }

    /// The peer's BAT index: its rank, which `TpWorld::of` bounds far below a byte.
    pub fn dest_tid(self) -> u8 {
        self.peer.rank() as u8
    }
}

/// Why a sharded session could not link to its peer. Each is a load refusal: a topology without
/// device p2p RDMA is refused, never served over the host.
#[derive(Debug)]
pub enum AttachRefusal {
    /// A world of one has no peer.
    NoPeer(TpIdentity),
    /// The peer card is this card.
    SelfLoop { card: PciAddress },
    /// The link names a card this process does not drive.
    NotDriven(CardTaken),
    /// The process is already linked, as another rank or for another window.
    AlreadyLinked {
        linked: (TpIdentity, RecvWindow),
        asked: (TpIdentity, RecvWindow),
    },
    /// senlib could not name the peer card's region-2 address.
    PeerRegion(String),
    /// Base-address-table entry `tid` could not be installed.
    Bat { tid: u8, why: String },
    /// The runtime this card's side of the link needs did not come up.
    BringUp(String),
    /// A landing slot's xseg word did not compose, or its `XSEG` entry could not be programmed.
    Xseg(String),
}

impl fmt::Display for AttachRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPeer(me) => write!(
                f,
                "rank {} of a {}-rank world has no peer to link",
                me.rank(),
                me.world().get()
            ),
            Self::SelfLoop { card } => write!(
                f,
                "the peer card is this card ({card}) — a rank cannot send to itself"
            ),
            Self::NotDriven(e) => write!(f, "the link's own card is not this process's: {e}"),
            Self::AlreadyLinked { linked, asked } => write!(
                f,
                "this process is linked as {linked:?} and cannot link again as {asked:?} — one \
                 card's XSEG entries point at one window's slots, so every bundle it runs shares \
                 one window"
            ),
            Self::PeerRegion(e) => write!(
                f,
                "the peer card's region-2 address could not be read: {e} (device p2p RDMA is the \
                 only path; there is no host fallback)"
            ),
            Self::Bat { tid, why } => {
                write!(f, "installing base-address-table entry {tid} failed: {why}")
            }
            Self::BringUp(e) => write!(f, "the card runtime did not come up: {e}"),
            Self::Xseg(e) => write!(f, "programming a landing slot's XSEG entry failed: {e}"),
        }
    }
}

impl std::error::Error for AttachRefusal {}

/// ⭐ THE P2P LINK A SHARDED SESSION SENDS OVER — tensor parallelism's one load-time fact: which
/// physical card is the peer. Minted only by [`link_peer`], which installs the peer card's
/// region-2 address in this card's base-address table (device p2p RDMA — never host DMA), so this
/// card's sends reach the peer. The other half — this card's `XSEG` entry for each of the peer's
/// landing slots, pointed at that slot of the receive window — is [`point_landing_slots`]'s, once
/// the session that owns the window has allocated it.
///
/// ONE PER PROCESS. The process drives one card and the card's `XSEG` entries point at one window's
/// slots, so every bundle the process runs (decode, prefill, every rung) lands its peer's sends in
/// the one window the link programmed — [`crate::superdsc_exec::Executor::load`] refuses a bundle
/// whose window is elsewhere.
#[derive(Debug)]
pub struct PeerLink {
    me: TpIdentity,
    route: PeerRoute,
    window: RecvWindow,
    /// The window's first byte as the allocator places it — set once, by [`point_landing_slots`].
    landing: OnceLock<u64>,
}

impl PeerLink {
    /// The rank this process runs as.
    pub fn identity(&self) -> TpIdentity {
        self.me
    }

    /// The ids this process's sends and waits name.
    pub fn route(&self) -> PeerRoute {
        self.route
    }

    /// Where the peer's sends land on this card.
    pub fn window(&self) -> RecvWindow {
        self.window
    }

    /// The window's first byte as the allocator placed it on this card — `None` until
    /// [`point_landing_slots`] has pointed the slots at it.
    pub fn landing(&self) -> Option<u64> {
        self.landing.get().copied()
    }
}

/// The ranks of a world place their receive windows at different device addresses. A send names
/// its own window's address for its peer's (the two cards allocate one sequence), so this world
/// would send to addresses its peers do not land at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowsApart {
    pub own: Option<u64>,
    pub peers: Vec<Option<u64>>,
}

impl fmt::Display for WindowsApart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "this rank's receive window lands at {:x?}, but its peers' land at {:x?} — every rank \
             of a world must place its window at one address, which each send names for its peer's",
            self.own, self.peers
        )
    }
}

impl std::error::Error for WindowsApart {}

/// ⭐ EVERY RANK'S WINDOW AT ONE ADDRESS — checked, not assumed. A send's xseg word names the
/// sender's own window address for the peer's (`flex_rs::scheduler::build_rdma_send_cb`), which is right
/// only while both cards place the window identically; the followers report theirs with their
/// ready frame, and the leader holds its first step until this passes.
///
/// # Errors
/// [`WindowsApart`]: any peer's landing address that is not this rank's, or a rank with none.
pub fn refuse_if_windows_apart(
    own: Option<u64>,
    peers: &[Option<u64>],
) -> Result<(), WindowsApart> {
    if own.is_some() && peers.iter().all(|p| *p == own) {
        Ok(())
    } else {
        Err(WindowsApart {
            own,
            peers: peers.to_vec(),
        })
    }
}

fn link_cell() -> &'static Mutex<Option<&'static PeerLink>> {
    static LINK: Mutex<Option<&'static PeerLink>> = Mutex::new(None);
    &LINK
}

/// The link this process made, when it runs one rank of a sharded world.
pub fn peer_link() -> Option<&'static PeerLink> {
    *link_cell().lock().unwrap_or_else(|e| e.into_inner())
}

/// Link this process — rank `me`, driving `cards.own` — to its peer on `cards.peer`, for sends
/// landing in `window`. Linking again as the same rank for the same window returns the link
/// already made.
///
/// # Errors
/// [`AttachRefusal`], by name.
pub fn link_peer(
    me: TpIdentity,
    cards: CardPair,
    window: RecvWindow,
) -> Result<&'static PeerLink, AttachRefusal> {
    let route = PeerRoute::of(me).ok_or(AttachRefusal::NoPeer(me))?;
    if cards.own == cards.peer {
        return Err(AttachRefusal::SelfLoop { card: cards.own });
    }
    let mut cell = link_cell().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(link) = *cell {
        return if (link.me, link.window) == (me, window) {
            Ok(link)
        } else {
            Err(AttachRefusal::AlreadyLinked {
                linked: (link.me, link.window),
                asked: (me, window),
            })
        };
    }
    drive_card(&cards.own).map_err(AttachRefusal::NotDriven)?;
    init_runtime().map_err(|e| AttachRefusal::BringUp(e.to_string()))?;
    // ⭐ THE SYMMETRIC BAT, AS THE tp37 CELL INSTALLS IT (#1008): entry `dest_tid` — the peer's
    // rank, the destination every send names — holds the peer card's region-2 PCI address, from
    // senlib's own region-address door (the vendor's `initialize_p2p_address_table` value
    // source), for the card the address names, never an enumeration order.
    let peer_region2 = flex_rs::senlib_ffi::pci_addr_for_region(cards.peer.as_str(), 2)
        .map_err(AttachRefusal::PeerRegion)?;
    flex_rs::senlib_ffi::set_base_addr_table(route.dest_tid(), peer_region2).map_err(|why| {
        AttachRefusal::Bat {
            tid: route.dest_tid(),
            why,
        }
    })?;
    let link: &'static PeerLink = Box::leak(Box::new(PeerLink {
        me,
        route,
        window,
        landing: OnceLock::new(),
    }));
    *cell = Some(link);
    Ok(link)
}

/// ⭐ POINT EVERY ONE OF THE PEER'S LANDING SLOTS AT ITS SLOT OF THIS CARD'S WINDOW — one `XSEG`
/// entry per slot, each programmed once, as the tp37 cell programs its one. An F-send lands where its
/// sid's entry points and nowhere else (#1009, E8a), so the peer's message into slot `s` signals
/// `wait_sid(s)` and that entry names the slot: `hal_sbf::rdma_xseg` of the slot's flat VA plus the
/// one-flit L4 compensation, the VA from [`flex_rs::scheduler::landing_va_bytes`] — the same compose
/// the send's own word takes.
///
/// ⛔ THE VA IS THE ALLOCATOR'S, SO THIS RUNS ONCE THE WINDOW IS ALLOCATED. The window is a placement
/// in the weight segment, and where that segment's bytes ARE — the region's segment and the
/// allocation's offset inside it — exists only once a session has allocated it. `window_region` is
/// that allocation: the weight segment of the session that OWNS it (every other session aliases
/// it). MEASURED, #1009 run 5: the entries pointed at `(program seg 1, layout offset)`, an address
/// the allocator gave to something else, and every collective ran clean over a window the peer's
/// bytes never reached — garbage from token 0.
///
/// # Errors
/// [`AttachRefusal`]: the runtime, a slot past the allocation, or a word that will not compose or
/// program.
pub fn point_landing_slots(link: &PeerLink, window_region: &DevAddr) -> Result<(), AttachRefusal> {
    let rt = get_or_init_runtime().map_err(|e| AttachRefusal::BringUp(e.to_string()))?;
    let resolver = FlexAllocatorAddressResolver::new(
        rt.get_allocator()
            .map_err(|e| AttachRefusal::BringUp(format!("{e:?}")))?,
    );
    let cap = crate::bundle_code::collective::RDMA_MESSAGE_CAP_BYTES;
    let landing = flex_rs::scheduler::landing_va_bytes(
        &resolver,
        &RdmaOperand {
            base: window_region.0.clone(),
            segment: RecvWindow::SEGMENT as u8,
            offset: link.window.offset(),
        },
        link.window.size(),
    )
    .map_err(|e| AttachRefusal::Xseg(e.to_string()))?;
    if link.landing.get().is_some_and(|l| *l != landing) {
        return Err(AttachRefusal::Xseg(format!(
            "the window is pointed at 0x{landing:x}, but this process already pointed it at {:x?}",
            link.landing.get()
        )));
    }
    let _ = link.landing.set(landing);
    let mut pointed: Vec<String> = Vec::new();
    for (slot, at) in link.window.slots(link.route.hears_in()) {
        // The slot's own extent: one message, or the region's remainder for its last.
        let len = (link.window.region().get() - u64::from(slot.message()) * cap).min(cap);
        let va = flex_rs::scheduler::landing_va_bytes(
            &resolver,
            &RdmaOperand {
                base: window_region.0.clone(),
                segment: RecvWindow::SEGMENT as u8,
                offset: at,
            },
            len,
        )
        .map_err(|e| AttachRefusal::Xseg(e.to_string()))?;
        let word = flex_rs::hal_sbf::rdma_xseg(va + flex_rs::control_block_wire::Flits::BYTES)
            .map_err(AttachRefusal::Xseg)?;
        flex_rs::senlib_ffi::write_xseg(link.route.wait_sid(slot), word)
            .map_err(AttachRefusal::Xseg)?;
        pointed.push(format!("XSEG[{}]=0x{va:x}", link.route.wait_sid(slot)));
    }
    // The load-time ordering made observable: this line precedes the rank's ready report, and so
    // every step's sends.
    tracing::info!(
        "[spyre-tp] rank {} of {}: {} landing slot(s) pointed at the receive window ({:?}): {}",
        link.me.rank(),
        link.me.world().get(),
        pointed.len(),
        link.window,
        pointed.join(", "),
    );
    Ok(())
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

    /// Per landing slot, each rank of a two-rank world signals on its own sid and waits on the
    /// sid its peer signals for that slot, and lands on the peer's BAT index; a world of one has
    /// no route.
    #[test]
    fn a_route_names_each_slots_sids_and_the_peers_index() {
        use crate::bundle_code::collective::{TpWorld, WindowCut};
        let ranks: Vec<TpIdentity> = TpWorld::of::<2>().ranks().collect();
        let routes = [ranks[0], ranks[1]].map(|r| PeerRoute::of(r).unwrap());
        assert_eq!(routes.map(|r| r.dest_tid()), [1, 0]);
        for k in 0..2 {
            for m in 0..2 {
                let slot = LandingSlot::of(WindowCut::of_collective(k), m).unwrap();
                // The two ends of one landing: what rank 0 sends on is what rank 1 waits on.
                assert_eq!(routes[0].send_sid(slot), routes[1].wait_sid(slot));
                assert_eq!(routes[1].send_sid(slot), routes[0].wait_sid(slot));
                assert_ne!(routes[0].send_sid(slot), routes[1].send_sid(slot));
            }
        }
        let alone = TpWorld::of::<1>().ranks().next().unwrap();
        assert_eq!(PeerRoute::of(alone), None);
    }

    /// A world runs only when every rank's window lands at this rank's address: one peer elsewhere,
    /// or a rank that pointed none, refuses by name with every address.
    #[test]
    fn a_world_whose_windows_land_apart_refuses() {
        let a = Some(0x4_c842_2800_u64);
        assert_eq!(refuse_if_windows_apart(a, &[a]), Ok(()));
        assert_eq!(refuse_if_windows_apart(a, &[a, a]), Ok(()));
        let apart = Some(0x4_c862_2800_u64);
        assert_eq!(
            refuse_if_windows_apart(a, &[apart]),
            Err(WindowsApart {
                own: a,
                peers: vec![apart]
            })
        );
        assert!(refuse_if_windows_apart(a, &[None]).is_err());
        assert!(refuse_if_windows_apart(None, &[None]).is_err());
    }

    /// A card is named by its full PCI address, in either case, kept lowercase; anything else is
    /// refused by name.
    #[test]
    fn a_card_is_named_by_its_full_pci_address() {
        let a = PciAddress::parse("0000:3B:00.0").unwrap();
        assert_eq!(a.as_str(), "0000:3b:00.0");
        assert_eq!(a, PciAddress::parse("0000:3b:00.0").unwrap());
        for bad in [
            "0",
            "3b:00.0",
            "0000:3b:00",
            "0000:3b:00.8",
            "0000:3b.00:0",
            "0000:3b:00:0",
            "000g:3b:00.0",
            "0000:3b:00.0 ",
            "",
        ] {
            assert_eq!(
                PciAddress::parse(bad),
                Err(NotAPciAddress(bad.to_string())),
                "{bad:?}"
            );
        }
    }

    /// The two refusals that need no card: a world of one has no peer, and a peer card that is
    /// this card is a rank sending to itself.
    #[test]
    fn a_link_refuses_no_peer_and_a_self_loop_before_touching_the_card() {
        use crate::bundle_code::collective::{Payload, TpWorld, WireBytes};
        let window = RecvWindow::new(
            1 << 20,
            Payload::from(WireBytes::CAP),
            crate::bundle_code::collective::TpWorld::of::<2>(),
        )
        .unwrap();
        let alone = TpWorld::of::<1>().ranks().next().unwrap();
        let card = |bus: &str| PciAddress::parse(&format!("0000:{bus}:00.0")).unwrap();
        let pair = |own, peer| CardPair {
            own: card(own),
            peer: card(peer),
        };
        assert!(matches!(
            link_peer(alone, pair("3b", "86"), window),
            Err(AttachRefusal::NoPeer(_))
        ));
        let rank0 = TpWorld::of::<2>().ranks().next().unwrap();
        assert!(matches!(
            link_peer(rank0, pair("3b", "3b"), window),
            Err(AttachRefusal::SelfLoop { card: c }) if c == card("3b")
        ));
        assert!(peer_link().is_none(), "a refused link links nothing");
    }

    /// A process drives one card, named by its address: choosing it again is a no-op, choosing
    /// another is refused by name, before any bring-up.
    #[test]
    fn a_process_drives_the_one_card_it_chose() {
        let first = PciAddress::parse("0000:3b:00.0").unwrap();
        let other = PciAddress::parse("0000:86:00.0").unwrap();
        drive_card(&first).expect("the first choice is the card");
        drive_card(&first).expect("the same card again");
        assert_eq!(driven_card(), DrivenCard::At(first.clone()));
        assert_eq!(
            drive_card(&other),
            Err(CardTaken {
                driving: DrivenCard::At(first),
                asked: other
            })
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
}
