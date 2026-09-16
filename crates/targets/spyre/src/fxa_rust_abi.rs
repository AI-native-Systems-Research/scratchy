// SPDX-License-Identifier: Apache-2.0
//! Real `fxa_*` C-ABI entrypoints, implemented natively in Rust over
//! `flex-rs` instead of crossing into `csrc/spyre_sdk_abi.cpp`'s C++ shim.
//!
//! `sdk_abi.rs` declares these exact symbols as `unsafe extern "C"`;
//! `spyre_sdk_abi.cpp` no longer defines any of them (it only still
//! provides the two deeptools functions out of flex-rs's scope), so these
//! `#[unsafe(no_mangle)]` definitions are the ONLY providers of `fxa_*` in the
//! final binary — same symbol names, same signatures, no C++ flex:: path.
//!
//! One process-wide `RuntimeContext`, matching `spyre_sdk_abi.cpp`'s own
//! `g_rt`/`call_once` pattern: `fxa_init_runtime` lazily builds it once,
//! backed by the real senlib scheduler (`Backend::Pf`, `SenlibHardwareSubmit`,
//! `SenlibQueueCapacity`) and a real `FlexAllocator` whose CB address
//! resolution goes through `FlexAllocatorAddressResolver` — the piece that
//! previously had no construction call site anywhere in scratchy.
#![cfg(feature = "spyre-hw")]

use std::ffi::{CStr, c_char, c_void};
use std::sync::{Arc, Mutex, OnceLock};

use flex_rs::address::{ByteSize, Chunk, CompositeAddress, LogicalAddress};
use flex_rs::allocator::{AllocationDirective, FlexAllocator, FlexAllocatorInit};
use flex_rs::compute::{BootstrapOffset, ComputeParams, KernelName};
use flex_rs::device_memory_allocator::DeviceMemoryAllocator;
use flex_rs::dma::{DmaDirection, DmaParams, DmaShaping, HostVirtualAddress};
use flex_rs::flex_config::DeviceKind;
use flex_rs::memory_region::{DomainId, MemoryType};
use flex_rs::runtime::RuntimeContext;
use flex_rs::scheduler::{
    DeviceAddressResolver, FlexAllocatorAddressResolver, HardwareSubmit, QueueCapacitySource,
    SchedulerBackendConfig, SenlibHardwareSubmit, SenlibQueueCapacity, create_scheduler,
};
use flex_rs::stream::{RuntimeStreamMode, RuntimeStreamPriority, StreamId};
use flex_rs::topology::populate_domain_topology;

type RuntimeInit = Result<Arc<RuntimeContext>, String>;

fn runtime_cell() -> &'static Mutex<Option<RuntimeInit>> {
    static CELL: OnceLock<Mutex<Option<RuntimeInit>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(None))
}

/// Idempotent bring-up: real senlib device init (`flex_shim_init_runtime`)
/// then a real `RuntimeContext` over a `Pf` backend + a real `FlexAllocator`
/// whose address resolution is `FlexAllocatorAddressResolver` (production,
/// no longer an unreachable island — see `flex_rs::scheduler` module doc).
///
/// Single on-card domain (`DomainId(0)`), matching `spyre_sdk_abi.cpp`'s own
/// `fxa_alloc`, which passes `domain_ids = {0}` unconditionally: neither
/// side has ever asked the topology which domain is real (see that
/// function's own `⛔⛔⛔` comment about the bimodal-latency investigation);
/// this bridge does not invent a multi-domain topology query that has no
/// call site anywhere else in this codebase either.
fn build_runtime() -> Result<Arc<RuntimeContext>, String> {
    let rc = unsafe { flex_rs::senlib_ffi::flex_shim_init_runtime(0) };
    if rc != 0 {
        return Err(format!("flex_shim_init_runtime failed: rc={rc}"));
    }

    // Real single on-card domain, sized off the card's actual HBM capacity
    // and RCU core count — `flex::populateDomainTopology`
    // (`device_memory_topology.cpp`), ported natively as
    // `topology::populate_domain_topology`. No longer the 32 GiB/32-core
    // placeholder this used to be (see git history): that shape caused a
    // real `FlexAllocator` OOM during hardware testing, traced to this
    // function undersizing/oversizing the domain relative to the real card.
    // SAFETY: `flex_shim_init_runtime` above already succeeded, so the
    // device handle this reads from is live.
    let device_memory_size = unsafe { flex_rs::senlib_ffi::flex_senlib_device_memory_size() };
    let topo = populate_domain_topology(device_memory_size);

    let init = FlexAllocatorInit::default();
    let dma = DeviceMemoryAllocator::new(flex_rs::allocator::DEVICE_ALIGNMENT);
    let allocator = Arc::new(Mutex::new(
        FlexAllocator::new(init, topo, dma).map_err(|e| e.to_string())?,
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
    .map_err(|e| format!("create_scheduler: {e:?}"))?;

    RuntimeContext::create(None, scheduler, allocator).map_err(|e| e.to_string())
}

fn get_or_init_runtime() -> Result<Arc<RuntimeContext>, String> {
    let mut guard = runtime_cell().lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Some(build_runtime());
    }
    guard.as_ref().unwrap().clone()
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
    // Reuse the ONE bring-up path: `build_runtime` is what the rest of this file goes through, and
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
/// let _budget = scratchy_target_spyre::fxa_rust_abi::CardMemory(8 * 1024 * 1024 * 1024);
/// ```
///
/// ```compile_fail,E0308
/// // ...and it cannot be smuggled in as a bare integer either.
/// fn wants_the_card(_: scratchy_target_spyre::fxa_rust_abi::CardMemory) {}
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

// ---------------------------------------------------------------------
// Runtime + streams
// ---------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn fxa_init_runtime() -> i32 {
    match get_or_init_runtime() {
        Ok(_) => 0,
        Err(e) => {
            tracing::error!("init_runtime: {e}");
            -2
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn fxa_prewarm() {
    std::thread::spawn(|| {
        if let Err(e) = get_or_init_runtime() {
            tracing::error!("prewarm failed (will retry in prepare): {e}");
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn fxa_stream_create() -> *mut c_void {
    let rt = match get_or_init_runtime() {
        Ok(rt) => rt,
        Err(_) => return std::ptr::null_mut(),
    };
    let id = match rt.create_stream(
        RuntimeStreamPriority::Normal,
        RuntimeStreamMode::StrictOrdering,
    ) {
        Ok(id) => id,
        Err(e) => {
            tracing::error!("stream_create: {e}");
            return std::ptr::null_mut();
        }
    };
    let boxed: Box<(Arc<RuntimeContext>, StreamId)> = Box::new((rt, id));
    Box::into_raw(boxed) as *mut c_void
}

/// # Safety: `stream` must be a pointer returned by `fxa_stream_create` and
/// not already destroyed.
unsafe fn stream_handle<'a>(stream: *mut c_void) -> Option<&'a (Arc<RuntimeContext>, StreamId)> {
    if stream.is_null() {
        return None;
    }
    Some(unsafe { &*(stream as *const (Arc<RuntimeContext>, StreamId)) })
}

#[unsafe(no_mangle)]
pub extern "C" fn fxa_stream_destroy(stream: *mut c_void) {
    if stream.is_null() {
        return;
    }
    // SAFETY: `stream` is a live box created by `fxa_stream_create`, per
    // this module's own contract (mirrors `spyre_sdk_abi.cpp`'s
    // `destroyStream` taking a created handle back).
    let boxed = unsafe { Box::from_raw(stream as *mut (Arc<RuntimeContext>, StreamId)) };
    let (rt, id) = *boxed;
    let _ = rt.destroy_stream(id);
}

/// # Safety
/// `stream` must be a pointer returned by `fxa_stream_create` and not already destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fxa_sync(stream: *mut c_void) -> i32 {
    let Some((rt, id)) = (unsafe { stream_handle(stream) }) else {
        return -1;
    };
    let Ok(Some(s)) = rt.get_stream(*id) else {
        return -1;
    };
    match s.synchronize() {
        Ok(()) => 0,
        Err(e) => {
            tracing::error!("sync: {e}");
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn fxa_runtime_reset() -> i32 {
    match RuntimeContext::reset() {
        Ok(()) => {
            *runtime_cell().lock().unwrap_or_else(|e| e.into_inner()) = None;
            0
        }
        Err(_) => -1,
    }
}

// ---------------------------------------------------------------------
// Device memory
// ---------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn fxa_alloc(nbytes: u64, program: i32) -> *mut c_void {
    let rt = match get_or_init_runtime() {
        Ok(rt) => rt,
        Err(_) => return std::ptr::null_mut(),
    };
    let directive = match AllocationDirective::new(
        flex_rs::domain::PlacementPolicy::Bind,
        vec![DomainId(0)],
        None,
        if program != 0 {
            MemoryType::Program
        } else {
            MemoryType::Tensor
        },
    ) {
        Ok(d) => d,
        Err(e) => {
            tracing::error!("alloc({nbytes}): {e}");
            return std::ptr::null_mut();
        }
    };
    let allocator = match rt.get_allocator() {
        Ok(a) => a,
        Err(e) => {
            tracing::error!("alloc({nbytes}): {e}");
            return std::ptr::null_mut();
        }
    };
    let mut guard = allocator.lock().unwrap_or_else(|e| e.into_inner());
    match guard.allocate(ByteSize(nbytes), &directive) {
        Ok(addr) => Box::into_raw(Box::new(addr)) as *mut c_void,
        Err(e) => {
            tracing::error!("alloc({nbytes}): {e}");
            std::ptr::null_mut()
        }
    }
}

/// # Safety: `addr` must be a pointer produced by `fxa_alloc`/`fxa_addr_from_chunk`.
unsafe fn addr_ref<'a>(addr: *const c_void) -> Option<&'a CompositeAddress> {
    if addr.is_null() {
        return None;
    }
    Some(unsafe { &*(addr as *const CompositeAddress) })
}

#[unsafe(no_mangle)]
pub extern "C" fn fxa_addr_free(addr: *mut c_void) {
    if addr.is_null() {
        return;
    }
    // SAFETY: `addr` is a live box created by `fxa_alloc`/`fxa_addr_from_chunk`.
    let boxed = unsafe { Box::from_raw(addr as *mut CompositeAddress) };
    // Freeing device memory (not just the descriptor) additionally needs a
    // live RuntimeContext's allocator; best-effort, matching the C++ shim's
    // `fxa_addr_free`, which only `delete`s the descriptor and never frees
    // the underlying region either (the executor never frees regions).
    drop(boxed);
}

/// # Safety
/// `addr` must be a pointer produced by `fxa_alloc`/`fxa_addr_from_chunk`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fxa_addr_total_size(addr: *const c_void) -> u64 {
    match unsafe { addr_ref(addr) } {
        Some(a) => a.total_size().as_u64(),
        None => 0,
    }
}

/// # Safety
/// `addr` must be a pointer produced by `fxa_alloc`/`fxa_addr_from_chunk`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fxa_addr_is_single_chunk(addr: *const c_void) -> i32 {
    match unsafe { addr_ref(addr) } {
        Some(a) => i32::from(a.is_single_chunk()),
        None => 0,
    }
}

/// # Safety
/// `addr` must be a pointer produced by `fxa_alloc`/`fxa_addr_from_chunk`, or null;
/// `region`/`offset`/`size`/`domain` must be valid, non-aliased out-params.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fxa_addr_chunk0(
    addr: *const c_void,
    region: *mut u64,
    offset: *mut u64,
    size: *mut u64,
    domain: *mut u64,
) -> i32 {
    let Some(a) = (unsafe { addr_ref(addr) }) else {
        return -2;
    };
    let Some(c0) = a.chunks().first() else {
        return -1;
    };
    // SAFETY: caller-supplied out-params, per this function's own contract
    // (mirrors `spyre_sdk_abi.cpp::fxa_addr_chunk0`).
    unsafe {
        *region = c0.addr.region_id.0;
        *offset = c0.addr.offset.0;
        *size = c0.size.0;
        *domain = c0.domain_id.0 as u64;
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn fxa_addr_from_chunk(
    region: u64,
    offset: u64,
    size: u64,
    domain: u64,
) -> *mut c_void {
    let addr = LogicalAddress::new(
        flex_rs::address::RegionId(region),
        flex_rs::address::ByteOffset(offset),
    );
    let chunk = Chunk {
        addr,
        size: ByteSize(size),
        domain_id: DomainId(domain as u32),
    };
    Box::into_raw(Box::new(CompositeAddress::from_chunk(chunk))) as *mut c_void
}

// ---------------------------------------------------------------------
// DMA + compute launches
// ---------------------------------------------------------------------

/// # Safety
/// `stream` must be live (per `fxa_sync`'s contract); `addr` must be a pointer produced
/// by `fxa_alloc`/`fxa_addr_from_chunk`; `host` must point to at least `size` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fxa_h2d(
    stream: *mut c_void,
    host: *const c_void,
    size: u64,
    addr: *const c_void,
) -> i32 {
    let Some((rt, id)) = (unsafe { stream_handle(stream) }) else {
        return -1;
    };
    let Some(a) = (unsafe { addr_ref(addr) }) else {
        return -1;
    };
    let Ok(Some(s)) = rt.get_stream(*id) else {
        return -1;
    };
    // `size` is the caller's own explicit DMA byte count — matches
    // `spyre_sdk_abi.cpp::fxa_h2d`'s `createDmaParams(host, size,
    // /*to_device=*/true, addr, nullptr)`, where the transfer size is
    // whatever the caller passed, independent of `addr`'s own (possibly
    // alignment-padded) `total_size()` — see `Stream::h2d`'s own doc
    // ("the whole-region form, where the device region's alignment pad
    // exceeds the logical byte count the slice would carry"). Using
    // `from_device_allocation` here would silently substitute
    // `addr.total_size()` for `size`, over-reading past the caller's host
    // buffer whenever the two differ.
    // `a` was already validated non-null by `addr_ref` above, so
    // `Some(...)` here can never trip `DmaParamsError::NullDeviceAddress`.
    let params = DmaParams::with_explicit_size(
        HostVirtualAddress(host as usize),
        ByteSize(size),
        DmaDirection::HostToDevice,
        Some(Arc::new(a.clone())),
        DmaShaping::default(),
    )
    .expect("device_address is non-null, validated by addr_ref above");
    match s.launch_h2d(params, None) {
        Ok(()) => 0,
        Err(e) => {
            tracing::error!("h2d({size}): {e}");
            -1
        }
    }
}

/// # Safety
/// `stream` must be live (per `fxa_sync`'s contract); `addr` must be a pointer produced
/// by `fxa_alloc`/`fxa_addr_from_chunk`; `host` must point to at least `size` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fxa_d2h(
    stream: *mut c_void,
    host: *mut c_void,
    size: u64,
    addr: *const c_void,
) -> i32 {
    let Some((rt, id)) = (unsafe { stream_handle(stream) }) else {
        return -1;
    };
    let Some(a) = (unsafe { addr_ref(addr) }) else {
        return -1;
    };
    let Ok(Some(s)) = rt.get_stream(*id) else {
        return -1;
    };
    // See `fxa_h2d`'s comment: `size` is the caller's own explicit transfer
    // size, matching `spyre_sdk_abi.cpp::fxa_d2h`'s `createDmaParams(host,
    // size, /*to_device=*/false, addr, nullptr)`. Substituting
    // `addr.total_size()` would risk writing past the caller's host buffer.
    // `a` was already validated non-null by `addr_ref` above, so
    // `Some(...)` here can never trip `DmaParamsError::NullDeviceAddress`.
    let params = DmaParams::with_explicit_size(
        HostVirtualAddress(host as usize),
        ByteSize(size),
        DmaDirection::DeviceToHost,
        Some(Arc::new(a.clone())),
        DmaShaping::default(),
    )
    .expect("device_address is non-null, validated by addr_ref above");
    match s.launch_d2h(params, None) {
        Ok(()) => 0,
        Err(e) => {
            tracing::error!("d2h({size}): {e}");
            -1
        }
    }
}

/// # Safety
/// `stream`/`prog` must be live per `fxa_sync`/`fxa_alloc`'s contracts;
/// `tensor_allocs` must point to `n` valid `CompositeAddress*` (as produced by
/// `fxa_alloc`/`fxa_addr_from_chunk`); `tensor_byte_offsets` must point to `n_tbo` valid `u64`s
/// (or be anything when `n_tbo == 0`); `kernel_name` must be a valid NUL-terminated C string or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fxa_compute(
    stream: *mut c_void,
    prog: *const c_void,
    tensor_allocs: *const *const c_void,
    n: u64,
    kernel_name: *const c_char,
    bootstrap_offset: u64,
    tensor_byte_offsets: *const u64,
    n_tbo: u64,
    pipeline_barrier: i32,
) -> i32 {
    let Some((rt, id)) = (unsafe { stream_handle(stream) }) else {
        return -1;
    };
    let Some(prog_addr) = (unsafe { addr_ref(prog) }) else {
        return -1;
    };
    let Ok(Some(s)) = rt.get_stream(*id) else {
        return -1;
    };

    let mut allocs = Vec::with_capacity(n as usize);
    for i in 0..n {
        // SAFETY: `tensor_allocs`/`n` describe a valid array of `n`
        // `CompositeAddress*`, per this function's own contract.
        let p = unsafe { *tensor_allocs.add(i as usize) };
        let Some(a) = (unsafe { addr_ref(p) }) else {
            return -1;
        };
        allocs.push(Arc::new(a.clone()));
    }
    let tbo: Vec<flex_rs::address::ByteOffset> = if n_tbo == 0 {
        Vec::new()
    } else {
        // SAFETY: `tensor_byte_offsets`/`n_tbo` describe a valid array, per contract.
        unsafe { std::slice::from_raw_parts(tensor_byte_offsets, n_tbo as usize) }
            .iter()
            .map(|o| flex_rs::address::ByteOffset(*o))
            .collect()
    };
    let name = if kernel_name.is_null() {
        String::new()
    } else {
        // SAFETY: `kernel_name` is a valid NUL-terminated C string, per contract.
        unsafe { CStr::from_ptr(kernel_name) }
            .to_string_lossy()
            .into_owned()
    };

    let params = ComputeParams::new(
        Arc::new(prog_addr.clone()),
        allocs,
        KernelName(name),
        BootstrapOffset(bootstrap_offset),
        tbo,
    )
    .with_pipeline_barrier(pipeline_barrier != 0);

    match s.launch_compute(params, None) {
        Ok(()) => 0,
        Err(e) => {
            tracing::error!("compute: {e}");
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn fxa_prog_offset_base() -> u64 {
    flex_rs::compute::PROG_OFFSET_BASE.0
}

// ---------------------------------------------------------------------
// deeptools host functions — out of flex-rs's scope: program correction
// (`fxa_process_compute_on_host`) and the fp16-bit-conversion pair are
// deeptools SDK calls, not flex ones. NOT redefined here:
// `csrc/spyre_sdk_abi.cpp` provides exactly these three symbols
// unconditionally now, and `sdk_abi.rs`'s existing `unsafe extern "C"`
// declarations already bind to them — no Rust-side change needed for this
// trio.
