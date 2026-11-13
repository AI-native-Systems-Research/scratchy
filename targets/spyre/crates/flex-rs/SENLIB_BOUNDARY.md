# senlib FFI boundary

This crate is a native Rust port of the flex-owned call-graph slice reachable
from scratchy's Spyre SDK ABI shim (CodeQL closure: 134 flex-owned types, 320
flex-owned functions, 17 ABI entrypoints — see `/tmp/flex_scope_final.json`
for the master scope). flex's own logic — region/block bookkeeping, best-fit
+ coalescing, allocation-directive resolution, stream/scheduler state
machines, in-flight counters, completion-callback lifecycle, control-block
fixup/translation math, env/JSON config parsing, fp8/fp16 bit-twiddling —
is ported natively into Rust. (The RDMA/HDMA multi-device slice was deleted
wholesale in issue #530 when its FFI bottom was mock-only, then re-added in
issue #550 against the real cross-port P2P surface — see its section
below.) Every point where that logic bottoms out into real hardware or
firmware programming crosses through an `extern "C"` declaration in
`src/senlib_ffi.rs`, and nowhere else: that module (plus a handful of
`Drop`-adjacent pointer/slice operations) is where every `unsafe` block in
this crate lives. The former mock-device `mmap`/`munmap` libc binding went
away with the senulator backend (34a3eb9a6) — the crate has no non-senlib
`extern "C"` block at all now.

Each parallel phase of this port wrote its own `senlib_ffi_<label>.rs` file
(to avoid symbol/name collisions while phases were being written
concurrently) plus a per-phase `SENLIB_BOUNDARY_<label>.md`. This document
is the Assembly phase's consolidation of all of those into one ordered
reference; the per-phase `.md` files have been folded in and removed. The
per-phase `.rs` files were folded the same way in issue #530: the former
re-export hub plus six `senlib_ffi_<label>.rs` modules are now ONE module,
`src/senlib_ffi.rs`, carrying every declaration, handle type, and safe
wrapper. Its module doc records the fold and the one cross-phase duplicate
that surfaced during Assembly (see below).

## Consolidation fix: duplicate "how many Spyre cards" declarations

Three phases independently reached `senlib::v2::SenPci::ncards()` — the same
real, zero-argument senlib call — from three different call sites, and each
declared its own `extern "C"` binding for it:

| Declared as | File | Reached from |
|---|---|---|
| `flex_senlib_pci_ncards` | `senlib_ffi_runtime.rs` | `RuntimeContext`/`getNumDevices()` (`runtime_context.cpp:44,109`) |
| `senlib_senpci_ncards` | `senlib_ffi_config_util.rs` | `FlexConfig::FlexSpyreDevices` (`flex_config.cpp:379,408`) |
| (part of) `flex_senlib_query_topology` | `senlib_ffi.rs` | topology-phase per-card HBM-layout query |

`flex_senlib_query_topology` no longer exists: it was never implemented in
`cxx-shim/senlib_ffi.cpp`, and real-hardware testing hit a `FlexAllocator`
OOM traced to runtime bring-up sizing its single domain off a 32 GiB
placeholder instead of the card's real HBM capacity. It was replaced by
`flex_senlib_device_memory_size()` + `flex_senlib_rcu_num_cores()` — the
two real calls `populateDomainTopology` actually makes (`DeviceHandle::
GetDmpaSize()` → `PfInterface::GetLPDDRSize()`, and
`SenPci::RCU_num_cores()`); this crate's 1p0 target has exactly one
domain, so a full multi-domain query was never actually needed. The other
two rows were an exact duplicate. Resolution: `flex_senlib_pci_ncards`
(declared in what was then `senlib_ffi_runtime.rs`, now `senlib_ffi.rs`'s
runtime-bring-up section) is kept as the one canonical declaration; the
config/util phase's `senlib_senpci_ncards` extern was deleted and its safe
wrapper `sen_pci_card_count()` now calls the canonical declaration
directly. No caller (`flex_config.rs`, `runtime.rs`) needed to change.

Also removed during consolidation: the pre-split hub's placeholder externs
`flex_senlib_stream_create`/`_destroy`/`_submit`/`_poll_completions` and
their `SenlibStreamHandle`/`RawControlBlock` types. These had zero callers
anywhere in the crate — they were superseded by the scheduler phase's
real, call-site-verified declarations (`senlib_cbi_*`/`senlib_rbi_*`, used
by `scheduler.rs`/`stream.rs`) before this file was consolidated, and were
dead code.

Also consolidated (not an FFI duplicate, but the same kind of parallel-agent
overlap): `util.rs` had re-ported `flex::FlexLogExtra` under the name
`UtilFlexLogExtra`, byte-for-byte behaviorally identical to
`telemetry::FlexLogExtra` (both port the same `util.hpp`/`util.cpp`
source). The `util.rs` copy had no external callers, so it was deleted;
`telemetry::FlexLogExtra` is the one kept.

## Per-subsystem boundary, in call-graph order

### Foundation: allocator (`allocator.rs`, `memory_region.rs`, `domain.rs`, `topology.rs`, `device_memory_allocator.rs`; FFI in `senlib_ffi.rs`)

Exactly two calls, both verified against `device_memory_allocator.cpp`,
unified across PF/VF backends since which backend is active is a
hardware-driver concern, not flex logic:

- **`flex_senlib_memory_allocate(num_bytes) -> SenlibAllocationHandle`** —
  PF: `senlib::MemoryAllocator::Allocate()` (`:212`); VF:
  `senlib::v2::VfWrapper::Allocate()` (`:176`). `handle == 0` is the failure
  sentinel for both. Takes only a byte count — neither real signature
  accepts a domain id (`TryAllocate`'s `domain_id` is logged-only in the
  C++, per its own TODO at `device_memory_allocator.cpp:153`). The VF
  retry-with-backoff loop (`:180-191`) is flex-side logic, not firmware
  behavior, and is NOT folded into this call's contract: it is ported
  natively (`device_memory_allocator.rs`'s `try_allocate`), which calls
  this FFI function in a loop.
- **`flex_senlib_memory_free(handle) -> i32`** — PF: `MemoryAllocator::Free()`
  (`:272`); VF: `VfWrapper::Deallocate()` (`:261`). Returns 0 on success,
  -1 if the free did not reach senlib — a failed deallocation must be
  distinguishable from a successful one (the same swallowed-failure shape
  the doorbell path fixed).

Everything else — `preAllocateRegions`, `allocateNewRegion`,
`allocateInDomain`/`allocateInRegion`, `alignSpyreAllocation`,
`resolveDirective`, region free-list/coalescing (BTreeSet-based),
`DeviceTopology`/`MemoryDomain`/`AllocationDirective`/`LogicalAddress`/
`Chunk`/`CompositeAddress` as pure value types — is native Rust, no
`unsafe`. `DeviceTopology`'s own *population* (querying real hardware for
domain/core counts) crosses the boundary as `flex_senlib_device_memory_size`
+ `flex_senlib_rcu_num_cores` (declared in `senlib_ffi.rs`'s
runtime-bring-up section) and IS wired to a native caller:
`topology::populate_domain_topology` → `fxa_rust_abi.rs::build_runtime`
(see the coverage cross-check below).

### Runtime core (`runtime.rs`, `stream.rs`; FFI in `senlib_ffi.rs`)

`RuntimeContext`'s and `RuntimeStream`'s own state-machine bookkeeping
(singleton lifecycle, stream table, in-flight counters, shutdown flag,
deferred-exception storage, completion-callback lifecycle) makes exactly one
direct senlib call:

- **`flex_senlib_pci_ncards() -> u64`** — `senlib::v2::SenPci::ncards()`,
  used by `getNumDevices()` (`runtime_context.cpp:44,109`) to learn the
  total card count. Irreducible: PCI/hardware topology enumeration, not
  something flex computes. Memoized natively via `OnceLock` in
  `num_physical_cards()`; `SPYRE_DEVICES`/`AIU_WORLD_SIZE` filtering on top
  of that count is `RuntimeConfig`/`FlexConfig` logic, ported separately.

`RuntimeStream::launchOperation*` never touches senlib directly in the C++ —
it validates parameters and calls `scheduler_->schedule(...)`; the actual
hardware submission lives in the scheduler backend (below). Device-interface
construction, the allocator, and the scheduler backend are each a different
subsystem's own boundary and are deliberately not re-declared here.

### Scheduler (`scheduler.rs`; FFI in `senlib_ffi.rs`)

`flex::RuntimeScheduler` (abstract base) + `PfRuntimeScheduler`/
`VfRuntimeScheduler`/`MockRuntimeScheduler`. Three calls:

1. **`senlib::ControlBlockInterface::capacity() const`**
   (`senlib_cbi_capacity`) — `RuntimeScheduler::getQueueCapacity`
   (`runtime_scheduler.cpp:1025`) via `device_handle_->ComputeCbi()`/
   `AsyncDmaICbi()`/`AsyncDmaOCbi()`. Hardware/firmware-configured queue
   depth; nothing on the host side can derive it.
2. **`senlib::ControlBlockInterface::QueueControlBlocksSBF(...)`**
   (`senlib_cbi_queue_control_blocks_sbf`) — the literal doorbell ring.
   Direct call site: `MockRuntimeScheduler::submitCompute`
   (`mock_runtime_scheduler.cpp:322`). PF/VF reach the same entry point
   indirectly through `ResponseWorker::QueueCbs()`
   (`response_worker.cpp:1214,1268`), which this crate treats as an opaque
   native-Rust dependency behind the `HardwareSubmit` trait rather than
   re-declaring bindings for it.
3. **`senlib::ResponseBlockInterface::ReceiveResponsesSBF(...)`**
   (`senlib_rbi_receive_responses_sbf`) — mock-only synchronous response
   drain, `mock_runtime_scheduler.cpp:329`.

Everything else — per-op fence/barrier bookkeeping, in-flight counters,
pipeline-switch detection, queue-capacity backpressure polling — is native.
Building the actual `ControlBlockSBF` bit layout, DMA-chunk splitting, IOMMU
mapping, and segment-table population is the control-blocks/device-interface
subsystems' own scope, plugged in through the `HardwareSubmit` trait rather
than re-implemented here.

**Formerly a known gap, now resolved for two of the three**:
`scheduler/runtime_scheduler_factory.cpp` (`createScheduler`) and
`scheduler/scheduler_stats.hpp` are now ported (`create_scheduler`,
`SchedulerStats`, both in `scheduler.rs` — see "Coverage cross-check"
below for exactly what didn't come along). `scheduler/
scheduler_compute_info.hpp` remains genuinely out of scope: every type it
declares owns a graph-compiler-phase object (`ControlBlockStream`,
`sendnn::SegmentTable`, a `moodycamel::ConcurrentQueue`) that this crate
does not have — not a disclaimer pointing at a phase that doesn't exist,
but a real dependency this crate has no module for.

### DMA/compute params + control blocks (`dma.rs`, `compute.rs`, `control_blocks.rs`)

Zero senlib calls. `DmaParams`/`ComputeParams`/`FillParams`
construction+validation, control-block fixup structs
(`XlatFixup`/`DmaFixup`/`TransferFixup`/`DdlFixup`/`ProgFixup`), DMVA→DMPA
translation, segment-offset bit arithmetic, and the mock/software
`DmaControlBlockHandler`/`ControlBlockHandlerBase` CB executors (selected
when `RuntimeConfig::FlexCompute()` is `"NULL"`/`"SENULATOR"`) are all plain
struct/arithmetic logic or `memcpy` into a flex-owned `DeviceMemory` mock
buffer — never a senlib call. `senlib_ffi.rs`'s module doc carries this
subsystem's "declares nothing, on purpose" note (formerly the entire
content of `senlib_ffi_controlblocks.rs`). The one real hardware-submission path this mock backend
stands in for as an alternative (`ResponseWorker::LaunchCbs`) belongs to the
scheduler subsystem above, not duplicated here.

### Multi-device / RDMA / HDMA — deleted (#530), re-added with real entrypoints (#550)

#530 deleted the former `multi_device.rs` (1928 lines) and its thirteen FFI
declarations on evidence that was, and remains, correct:

1. **No callers.** Nothing in the workspace referenced
   `flex_rs::multi_device` — only `lib.rs`'s module declaration and the
   FFI hub's re-export.
2. **No link targets.** `cxx-shim/senlib_ffi.cpp` implemented zero of the
   thirteen symbols; anything calling them would have failed at `ld` with
   undefined references. Eight were written against flex-cxx's mock-device
   shared-memory layer (`RdmaShm`), not even the real P2P surface.

That half-state was a trap for whoever wires P2P next, so #550 (TP prep)
re-added the surface with a REAL transport bottom rather than restoring
the trap:

- **Restored verbatim** (from `b4a34d57f`): the transport-AGNOSTIC native
  port — `BitmaskHelper`/`BitSetHelper` bit bookkeeping,
  `CollBufferAllocator_t`/`Hdma_Allocation_Manager` allocation algorithms,
  the `hdma_op_t`/SID/VSID write-done instruction framing +
  `InstructionQueue`, `RdmaRendezvousManager`'s 3-state POSIX-file
  barrier/broadcast, `CollectiveIPC` over Unix-domain sockets, and
  `AIUTopo`'s protocol-selection policy. None of these touches FFI; the
  two `#[allow(dead_code)]`s the deleted copy carried did not come back
  (the fields are read now — `world_size` bounds-checks peer ranks,
  `bytes_per_chunk` has a parity accessor).
- **Rebuilt**: `RdmaUnit`. The mock `RdmaShm` counters/Xseg registers are
  gone; per-peer state is the cross-port IOMMU mapping, `transfer_window`
  resolves the mapped address an ordinary scheduler-submitted H2D DMA is
  aimed at, `send_write_done` frames a `VsidInstruction` instead of poking
  a shm counter, and `wait_for_barrier` takes its counter reads through
  the `WdoneCounterSource` trait (see the ledger below). The mock-era
  `Xseg`/`write_xseg` register layer has no real counterpart — the mapping
  IS the translation — and was not restored.
- **Re-added with real shim implementations** (their original names): the
  five senlib-real declarations —
  `flex_senlib_pin_and_map`/`_pinned_memory_release`/`_pinned_memory_hptr`
  (`PfWrapper::allocate_pin_and_map`, hdma_shm.cpp:903-914) and
  `flex_senlib_direct_set_wdone`/`flex_senlib_ainc_wdone`
  (`PfInterface::direct_set_wdone`/`ainc_wdone`,
  host_wdone.cpp:413,417,1025).
- **Not restored**: the eight mock-`RdmaShm` declarations. Their roles are
  filled by the real surface: address translation (`rdma_xseg_*`) is
  `flex_senlib_p2p_create_mapping` (senlib
  `vfio_backend::create_iommu_mapping_for_cross_port_p2p_rdma`, pod header
  `/opt/ibm/spyre/senlib/include/vfio/1p0/vfio_backend.hpp:73`); the data
  movement (`rdma_copy_to_remote`) is an ordinary H2D DMA aimed at the
  mapped address; the write-done counters (`rdma_wdone_inc`/`_check`) are
  the framed VSID instructions plus the register pokes above.
  `flex_senlib_get_dmabuf` (`PfInterface::get_dmabuf`,
  pf_interface.hpp:111) is the BAR-export primitive the mock never had.
  Peer enumeration was already this crate's surface
  (`sen_pci_card_count`/`pcie_address_for_card`).

**Why senlib's mapping primitive and not spyre-comms.** The vendor's real
collectives library is spyre-comms (`libspyre_comms.so`: `Context`/
`WorkSchedule`, barrier/broadcast/send/recv/allgather/allreduce/reduce),
and #550's transport is nominally "spyre-comms/SenRDMA". But its entry
point cannot be scratchy's seam:
`spyre_comms::initialize_library(spyre::GlobalRuntime::get(),
spyre::getDefaultStreamRuntimeHandle())` (torch-spyre
`csrc/distributed/spyre_ccl.cpp:49-51`) takes a live C++
`flex::RuntimeContext*` and `flex::RuntimeStream*` (`module.h:41-42`,
`spyre_stream.h:137`) — exactly the objects #512's fxa collapse removed
when flex-rs replaced the C++ flex runtime. scratchy therefore binds the
senlib primitive BELOW spyre-comms (the IOMMU mapping every P2P DMA sits
on), and the restored native sections above it are scratchy's counterpart
of the comms layer. If a future scratchy ever hosts a real C++
`flex::RuntimeContext`, binding spyre-comms directly becomes possible and
this decision should be revisited.

**FIXME(pod-verify) ledger** — every signature fact this re-add could not
check against the on-pod SDK, each marked in place, each with a one-line
fix:

| # | Fact | Assumed | Fix if wrong |
|---|---|---|---|
| 1 | vfio_backend include path / namespace | `<senlib/vfio/1p0/vfio_backend.hpp>`, `senlib::v2` (the pod .so's mangled symbol still says `v2`) | adjust the include + namespace uses in `senlib_ffi.cpp` |
| 2 | vfio_backend reachability from the public `PfInterface` | `PfInterface_impl` IS-A `vfio_backend`; `PfInterface` holds its impl at +0x18 (pod disassembly) | replace `p2p_vfio_backend()`'s body with the public accessor, if one exists |
| 3 | `peer_dev_info` semantics | public `vector<pair<int,void*>>`; per-entry meaning (aux = fd? BAR? slot; addr = DMA-target? host-mapped?) UNKNOWN | none needed for the code; the first pod run LOGS the segments and this table gets the answers |
| 4 | `PfWrapper::allocate_pin_and_map` signature/return | `(size_t, const char* name)` → shared_ptr-like wrapper with `hptr()` | branch on `name` for the overload pair; swap `static_pointer_cast` for `shared_ptr<void>(std::move(pm))` |
| 5 | wdone register widths | `sid: u8`, `val: i32` (pre-#530 citation) | widen the extern + wrapper |
| 6 | `PfInterface::get_dmabuf` return | the DMABuf fd itself | unwrap the wrapper object in the shim |
| 7 | `SenPciShared::pf()`'s return type vs the `PciDevId` the mapping takes | same type (the shim's `senlib_senpci_pf_address` proves `pf()` exists) | construct the `PciDevId` explicitly |

**Remaining wiring legs** (this re-add is the machinery; these are #550's
closing criteria): the cross-card transfer experiment — create the mapping
at attach, log `peer_dev_info` raw, aim an ordinary H2D DMA at the
resulting address — verified on two cards (or the IOMMU-mapping leg on
one); and the `WdoneCounterSource` binding for `wait_for_barrier`, which
needs the real cross-card counter's read path (host_wdone monitor vs.
direct register read) established on the pod. The P2P DMA itself carries a
context id that must match (`SNT_DMA_DCR::DMO_CONFIG::P2P_TC`;
`DMA_INTERRUPT_STATE::DMD_COMPDMO/DMD_COMPDMI_P2P_UNEXPCTXID`) — that is
why the old hand-poked-window writes arrived but were refused — so the
experiment must configure the context id, not just the address.

### Config/util (`flex_config.rs`, `util.rs`; FFI in `senlib_ffi.rs`)

Almost entirely env-var/JSON-config parsing with process-wide caching.
Exactly two calls, both inside `FlexConfig` (not `RuntimeConfig`):

- **`senlib::v2::SenPci::ncards()`/`SenPciShared::ncards()`** —
  `FlexConfig::FlexSpyreDevices` (`flex_config.cpp:379,408`). Same real call
  as `flex_senlib_pci_ncards` above — see the consolidation fix at the top
  of this document. Wrapped as `sen_pci_card_count()`.
- **`senlib::v2::SenPci::pf(card_index).to_string()`/`SenPciShared::pf(...)`** —
  `FlexConfig::RdmaGetPCIeAddress` (`flex_config.cpp:442,471`). PCIe bus
  address for a card's physical function; bus/driver-assigned, flex cannot
  derive it. Wrapped as `pcie_address_for_card(card_index)`.

Everything else — the complete `FlexConfig` settings surface (~65 settings
across both header variants), `RuntimeConfig`/`RuntimeSetting<T>`,
`sendnn::Status`, `AsPercentOf`/`ceil_to`/`CheckIfNotAligned` — is native.

**Not ported** (declared, no definition found in the provided C++ sources):
`flex::OnStatusWarning(const std::string&, int&)` and
`flex::OnStatusInfo(const std::string&, int&)` are declared in
`util.hpp:157,164` but no body for either appears in `util.cpp` or anywhere
else searched. Only the unrelated single-arg `sendnn::OnStatusWarning(const
Status&)` (`status.cpp:43`) has a body, and that one is ported
(`util::on_status_warning`). If CodeQL's closure genuinely reaches the
two-arg `flex::` overloads, their definition lives in a `.cpp` this port was
never given.

### Device-interface misc (`host_interface.rs`)

Zero senlib calls anywhere in this slice. Two of the three modules this
section originally named have since been deleted: `compiler_interface.rs`
(b47bd7a5d — helpers for an unported path) and `device_types.rs`
(34a3eb9a6 — the senulator/mock-device backend and its config knobs went
with it, taking the local `mmap`/`munmap` libc binding described below;
the crate now has no non-senlib `extern "C"` block at all). The C++-side
accounting is kept for provenance:

- `af_device/af_device_memory.cpp` — PF/VF/COMPILE devices have no
  host-accessible device memory at all; every method is a constant, a
  `nullptr`, or an explicit "not implemented" error.
- `mock_device/{device_memory_mock,mmapped_storage}.*` — the mock backend was
  a local `mmap`/`shm`-backed file, an OS-syscall boundary, not senlib
  (deleted with the senulator backend).
- `compiler_interface/dee_graph_converter.cpp`'s five in-scope functions
  (`get_dtcompiler_export_dir`, `remove_dtcompiler_export_dir`,
  `maybe_wrap_dim`, `convert_shape_to_vecint64[_rev]`) — env/config lookups,
  filesystem cleanup, and plain arithmetic — were ported into
  `compiler_interface.rs`, itself later deleted (b47bd7a5d) as helpers for
  an unported path. The rest of that file (`dee::PBD`, DeepRT invocation)
  is deep compiler-only machinery outside the ABI-reachable closure.
- `host_interface/raii_buffer.cpp` — `posix_memalign`/`memset`/`memcpy`/`free`
  only; ported onto `std::alloc` instead, needing no FFI at all.

**Stubbed, with reason**: `MmappedStorage::FlexMonitor`'s crash-cleanup
child process (`fork()`+`execlp("flex_monitor", ...)`) — best-effort
process supervision with no bearing on the ABI-reachable allocate/free/
read/write path's correctness; `fork()` has no direct `std` equivalent.
`MmappedStorage`'s blocking retry-loop attach constructor is simplified to a
single non-retrying attempt (the multi-process race it guards against
doesn't arise in-process). `ShmFileTraits` (the `senulator_flex_cbh` sibling
of the ported `PlainFileTraits`) is out of scope — `MockDeviceMemory` only
instantiates `MmappedStorage<PlainFileTraits>`.

**Known gaps**: `compiler_interface/host_compute/{host_compute_interface,
fused_batchnorm_precompute}.hpp`, `compiler_interface/util/
{input_output_meta_data,super_node}.hpp` (`HostComputeInterface`,
`FusedBatchNormPrecompute`, `InputOutputMetaData`, `SuperNode`,
`ProgramDefinition`) are in the master scope, attributed to this
subsystem's source directory, but not ported — the former
`compiler_interface.rs` covered only the five `dee_graph_converter.cpp`
functions above, and was itself deleted.
`device_interface/iommu/{iommu_mapping.cpp,iommu_raii_buffer.hpp}`
and `device_types/pf_device/pf_device_handle.cpp` are likewise unported
(`device_types/pf_device/pf_iommu_mapper.{hpp,cpp}` IS ported, as
`iommu.rs`);
`scheduler.rs`'s doc comment refers to "`IommuMapperInterface`/`DeviceHandle`"
as belonging to "the device-interface phase's files," but no such module
exists in this crate.

## Coverage cross-check against the master scope (134 types / 320 functions — the ORIGINAL closure; the multi-device slice it counted was deleted in #530 and re-added per #550, rebuilt onto the real P2P surface)

**The graph-compiler/whole-graph-execution mode is out of scope, confirmed
by checking the real ABI shim, not assumed.** flex supports two distinct
ways of running work: (a) being handed a whole `sendnn::Graph` to compile
and schedule itself (`dee::`/DeepRT compiler, `flex::Scheduler`,
`runtime_graph/`'s executor/partitioner/super-node machinery), and (b) a
caller submitting pre-compiled DMA/compute ops directly through
`RuntimeStream`/`RuntimeScheduler` (`runtime_stream/`) — which is all
scratchy's `fxa_*` ABI ever does. Grepping the real pre-port ABI shim
(`spyre_sdk_abi.cpp`) for `FlexFileTransfer`/`runtime_graph`/
`FlexComputeInterface`/`Scheduler::`/`dee::` turns up zero references; the
only "graph" touch anywhere in the ABI-reachable `runtime_stream/` sources
is `#include "flex/runtime_graph/flex_config.hpp"` — just where
`FlexConfig` happens to live in the directory tree (already ported
natively), not a functional dependency. So every item below in that mode
is a **false positive in the master 320-function scope**, not a real
gap — confirmed by tracing actual reachability, not by matching names:

- `runtime_graph/graph/{graph_executor/graph_executor.cpp,
  graph_partitioner/graph_partitioner.cpp,
  graph_partitioner/graph_partitioner_config.hpp,
  super_node/{super_node_context.cpp,super_node_info.hpp}}`,
  `runtime_graph/{flex_frame.cpp,flex_compute_interface.cpp}` — the
  graph-executor/partitioner/super-node layer, unreachable from the real
  ABI.
- `device_interface/data_transfer/flex_file_transfer.hpp` — extends
  `FlexComputeInterface`; same mode, same unreachability.
- `compiler_interface/host_compute/*` (`HostComputeInterface`,
  `FusedBatchNormPrecompute`), `compiler_interface/util/
  {input_output_meta_data,super_node}.hpp` (`InputOutputMetaData`,
  `SuperNode`, `ProgramDefinition`) — all four live in
  `dee::`/`dee::host_compute`, DeepRT's offline graph-compiler namespace;
  same bucket this doc already flagged `dee::PBD`/DeepRT invocation into.

Genuinely missing or genuinely unportable, unrelated to the graph-compiler
mode above:

- `device_types/pf_device/pf_device_handle.cpp` — every method is a
  one-line delegation straight to senlib (`GetDmpaSize`, `AsyncDmaICbi`/
  `ComputeCbi`/`Rbi`, `IncWdoneValue`, ...), already covered piecemeal by
  the shim functions cited elsewhere in this doc. Not a native-Rust gap —
  it's FFI-boundary orchestration this crate correctly keeps shim-side.
- `scheduler/scheduler_compute_info.hpp` — every type it declares
  (`SchedulerComputeInfo`/`SchedulerNetworkInfo`/`SchedulerPartitionInfo`/
  `SchedulerFrameInfo`) owns a `ControlBlockStream`, a
  `sendnn::SegmentTable`, or a
  `moodycamel::ConcurrentQueue<DeviceMemoryAllocationPtr>` — graph-compiler
  object-graph types this crate does not have. Confirmed genuinely
  unportable, not just unattempted.

Closed since this cross-check was first written:
`control_blocks/control_block_handler/status_callback.cpp`
(`StatusPromiseCallback`, now in `control_blocks.rs`);
`host_interface/host_memory_buffer.hpp` (`HostMemoryBuffer`, now in
`host_interface.rs` — its `sendnn::Segment` field stays an opaque
placeholder; its only real call site, `flat_executor.cpp`, is in the
graph-compiler mode confirmed out of scope above, so nothing will ever
need a real one); `runtime_stream/1p0/device_memory_topology.cpp`
(`populateDomainTopology`, now `topology::populate_domain_topology`, wired
into `fxa_rust_abi.rs::build_runtime` in place of the 32 GiB placeholder
domain that was causing real hardware-test OOMs);
`scheduler/scheduler_stats.hpp` (`SchedulerStats`, now in `scheduler.rs` —
everything except `Count()`, which stays unported for the same reason
`scheduler_compute_info.hpp` does, see below); `scheduler/
runtime_scheduler_factory.{hpp,cpp}` (`create_scheduler`, now in
`scheduler.rs`, wired into `fxa_rust_abi.rs::build_runtime` in place of the
hardcoded `Backend::Pf(PfBackend::new(...))` — though the device-kind
argument is itself still hardcoded to `DeviceKind::Pf` there, since
`flex_shim_init_runtime` only ever brings up a PF device; VF/MOCK selection
awaits that shim itself gaining a device-kind branch).

Everything else in the master scope cross-checked against source-file
groupings above (allocator, runtime core, scheduler's non-factory pieces,
control-block fixup/DMA/compute params, config/util,
device-interface-misc's in-scope slice, telemetry) is accounted for either
as a native port, an FFI wrapper documented above, or one of the
reasoned/time-boxed stubs listed per subsystem — with the multi-device
slice re-added per #550, its transport-agnostic half restored verbatim and
its `RdmaUnit` rebuilt onto the real cross-port P2P surface (see its
section above). The gaps
above should be picked up as a follow-on pass; none of them are on the
allocator/runtime-core/scheduler-submission critical path this crate's
existing tests exercise, but they are real entries in the 320-function
closure and are not yet represented anywhere in this crate.

## Build/link seams (issue #531, verified 2026-10-16)

#531's original text had two halves. The `sdk_abi.rs` half was mooted by
PR #512, which deleted `sdk_abi.rs` outright — the tape player now speaks
flex-rs directly, so there is no C ABI left to keep honest. The build-seam
half, re-verified at this fold:

- No "adapter TU" language remains anywhere in the spyre tree — the stale
  talk was pruned by 9d5401888 ("build(spyre): drop stale sendnn_*/flex
  link flags; correct entrypoint count"), and #512 removed the rest.
- `flex-rs/build.rs` (the only build script doing SDK work) links ONLY
  senlib plus its own ldd-verified transitive deps — flightlog and the
  boost set (log, log_setup, thread, filesystem, atomic, regex, json,
  container, locale, chrono). No `libflex`, no `sendnn_*`.
- `scratchy-target-spyre`'s `builder/build.rs` is a doc-only no-op
  (`rerun-if-changed` only), and `ktir/ktir-emulator/build.rs` links
  Accelerate/objc2 only — no SDK seams.

What this cannot prove locally is the actual link against the real on-pod
SDK — `ld` resolving the shim's senlib symbols against
`/opt/ibm/spyre/senlib/lib`. That needs a pod build, which is user-gated.
