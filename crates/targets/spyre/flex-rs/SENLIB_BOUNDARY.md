# senlib FFI boundary

This crate is a native Rust port of the flex-owned call-graph slice reachable
from scratchy's Spyre SDK ABI shim (CodeQL closure: 134 flex-owned types, 320
flex-owned functions, 17 ABI entrypoints — see `/tmp/flex_scope_final.json`
for the master scope). flex's own logic — region/block bookkeeping, best-fit
+ coalescing, allocation-directive resolution, stream/scheduler state
machines, in-flight counters, completion-callback lifecycle, control-block
fixup/translation math, RDMA/HDMA rendezvous and bitmap bookkeeping,
env/JSON config parsing, fp8/fp16 bit-twiddling — is ported natively into
Rust. Every point where that logic bottoms out into real hardware or
firmware programming crosses through an `extern "C"` declaration in one of
this crate's `senlib_ffi*.rs` files, and nowhere else: those files (plus a
handful of `Drop`-adjacent pointer/slice operations and one `mmap`/`munmap`
libc binding for the *software* mock-device backend — an OS-syscall
boundary, not a senlib one) are where every `unsafe` block in this crate
lives.

Each parallel phase of this port wrote its own `senlib_ffi_<label>.rs` file
(to avoid symbol/name collisions while phases were being written
concurrently) plus a per-phase `SENLIB_BOUNDARY_<label>.md`. This document
is the Assembly phase's consolidation of all of those into one ordered
reference; the per-phase `.md` files have been folded in and removed.
`src/senlib_ffi.rs` is the corresponding code-level consolidation point — it
re-exports every phase's FFI module and documents the one cross-phase
duplicate that surfaced (see below).

## Consolidation fix: duplicate "how many Spyre cards" declarations

Three phases independently reached `senlib::v2::SenPci::ncards()` — the same
real, zero-argument senlib call — from three different call sites, and each
declared its own `extern "C"` binding for it:

| Declared as | File | Reached from |
|---|---|---|
| `flex_senlib_pci_ncards` | `senlib_ffi_runtime.rs` | `RuntimeContext`/`getNumDevices()` (`runtime_context.cpp:44,109`) |
| `senlib_senpci_ncards` | `senlib_ffi_config_util.rs` | `FlexConfig::FlexSpyreDevices` (`flex_config.cpp:379,408`) |
| (part of) `flex_senlib_query_topology` | `senlib_ffi.rs` | topology-phase per-card HBM-layout query |

`flex_senlib_query_topology` legitimately does more than count cards (it
also reports per-card HBM bank layout), so it stays a separate declaration.
The other two were an exact duplicate. Resolution: `flex_senlib_pci_ncards`
in `senlib_ffi_runtime.rs` is kept as the one canonical declaration;
`senlib_ffi_config_util.rs`'s `senlib_senpci_ncards` extern was deleted and
its safe wrapper `sen_pci_card_count()` now calls the canonical declaration
directly. No caller (`flex_config.rs`, `runtime.rs`) needed to change.

Also removed during consolidation: `senlib_ffi.rs`'s pre-split placeholder
externs `flex_senlib_stream_create`/`_destroy`/`_submit`/`_poll_completions`
and their `SenlibStreamHandle`/`RawControlBlock` types. These had zero
callers anywhere in the crate — they were superseded by the scheduler
phase's real, call-site-verified `senlib_ffi_scheduler.rs`
(`senlib_cbi_*`/`senlib_rbi_*`, used by `scheduler.rs`/`stream.rs`) before
this file was consolidated, and were dead code.

Also consolidated (not an FFI duplicate, but the same kind of parallel-agent
overlap): `util.rs` had re-ported `flex::FlexLogExtra` under the name
`UtilFlexLogExtra`, byte-for-byte behaviorally identical to
`telemetry::FlexLogExtra` (both port the same `util.hpp`/`util.cpp`
source). The `util.rs` copy had no external callers, so it was deleted;
`telemetry::FlexLogExtra` is the one kept.

## Per-subsystem boundary, in call-graph order

### Foundation: allocator (`allocator.rs`, `memory_region.rs`, `domain.rs`, `topology.rs`, `device_memory_allocator.rs`, `senlib_ffi_allocator.rs`)

Exactly two calls, both verified against `device_memory_allocator.cpp`,
unified across PF/VF backends since which backend is active is a
hardware-driver concern, not flex logic:

- **`flex_senlib_memory_allocate(num_bytes, domain_id_or_negative_one) -> SenlibAllocationHandle`** —
  PF: `senlib::MemoryAllocator::Allocate()` (`:212`); VF:
  `senlib::v2::VfWrapper::Allocate()` (`:176-191`, including the
  retry-with-backoff loop, folded into this call's contract since VF retry
  policy is firmware behavior). `handle == 0` is the failure sentinel for
  both.
- **`flex_senlib_memory_free(handle)`** — PF: `MemoryAllocator::Free()`
  (`:272`); VF: `VfWrapper::Deallocate()` (`:261`).

Everything else — `preAllocateRegions`, `allocateNewRegion`,
`allocateInDomain`/`allocateInRegion`, `alignSpyreAllocation`,
`resolveDirective`, region free-list/coalescing (BTreeSet-based),
`DeviceTopology`/`MemoryDomain`/`AllocationDirective`/`LogicalAddress`/
`Chunk`/`CompositeAddress` as pure value types — is native Rust, no
`unsafe`. `DeviceTopology`'s own *population* (querying real hardware for
domain/core counts) is `flex_senlib_query_topology`, a different
subsystem's boundary (declared in `senlib_ffi.rs`; not yet wired to a native
caller — see that file's doc comment).

### Runtime core (`runtime.rs`, `stream.rs`, `senlib_ffi_runtime.rs`)

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

### Scheduler (`scheduler.rs`, `senlib_ffi_scheduler.rs`)

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

### DMA/compute params + control blocks (`dma.rs`, `compute.rs`, `control_blocks.rs`, `senlib_ffi_controlblocks.rs`)

Zero senlib calls. `DmaParams`/`ComputeParams`/`FillParams`
construction+validation, control-block fixup structs
(`XlatFixup`/`DmaFixup`/`TransferFixup`/`DdlFixup`/`ProgFixup`), DMVA→DMPA
translation, segment-offset bit arithmetic, and the mock/software
`DmaControlBlockHandler`/`ControlBlockHandlerBase` CB executors (selected
when `RuntimeConfig::FlexCompute()` is `"NULL"`/`"SENULATOR"`) are all plain
struct/arithmetic logic or `memcpy` into a flex-owned `DeviceMemory` mock
buffer — never a senlib call. `senlib_ffi_controlblocks.rs` intentionally
declares nothing. The one real hardware-submission path this mock backend
stands in for as an alternative (`ResponseWorker::LaunchCbs`) belongs to the
scheduler subsystem above, not duplicated here.

### Multi-device / RDMA / HDMA (`multi_device.rs`, `senlib_ffi_multi_device.rs`)

Nine calls, all irreducible kernel-driver or hardware-register operations:

- **`flex_senlib_pin_and_map`/`_pinned_memory_release`/`_pinned_memory_hptr`** —
  `PfWrapper`/`VfWrapper::allocate_pin_and_map` (`hdma_shm.cpp:903` etc.):
  page-locking host memory and mapping it into the device's IOMMU/DMA
  address space.
- **`flex_senlib_direct_set_wdone`/`flex_senlib_ainc_wdone`** —
  `pfw_->interface()->direct_set_wdone`/`ainc_wdone` (`host_wdone.cpp:413,417,1025`):
  device-side hardware write-done register pokes.
- **`flex_senlib_rdma_shm_open`/`_close`** — `RdmaShm` construction/teardown
  (`rdma_unit.cpp:340`): the mock-device-memory-exchange layer standing in
  for real RDMA hardware.
- **`flex_senlib_rdma_copy_to_remote`** — the literal RDMA write
  (`rdma_unit.cpp:415-416`, `memcpy` into a peer's mapped device memory).
- **`flex_senlib_rdma_wdone_inc`/`_check`** — remote write-done counter
  increment/poll (`rdma_unit.cpp:499,457`).
- **`flex_senlib_rdma_xseg_bytes`/`_valid`/`_write`** — the address-translation
  (Xseg) register backing the mock RDMA region (`rdma_unit.cpp:402,409,524`).

Ported natively instead: `BitmaskHelper`/`BitSetHelper` bit bookkeeping,
`CollBufferAllocator_t`/`Hdma_Allocation_Manager` allocation algorithms,
`HostWdone::make_instruction`/ring-counter bookkeeping,
`RdmaRendezvousManager::Barrier`/`BarrierAndBcast` (re-implemented over a
plain POSIX rendezvous file rather than the C++'s `shm_open`/`mmap`, since
there is no senlib/PF/VF involvement in that protocol), `CollectiveIPC`
(Unix-domain sockets), and `AIUTopo`'s protocol-selection policy.

**Known gaps** (not ported, time-boxed out of this pass): JSON
topology-file ingestion (`AIUTopo::process_topo_json_*`) — `AiuTopo::new`
takes an already-resolved table instead; `HdmaShmMgmt`/`HcollShmManager_t`'s
R5/CPU monitor-mode instruction staging and HMLT table read/display;
`HostWdone`'s background monitor thread (the instruction *encoding* is
ported, the thread loop reading pinned memory is not); `RdmaUnit`
construction's PID-rendezvous handshake.

### Config/util (`flex_config.rs`, `util.rs`, `senlib_ffi_config_util.rs`)

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

### Device-interface misc (`device_types.rs`, `compiler_interface.rs`, `host_interface.rs`)

Zero senlib calls anywhere in this slice:

- `af_device/af_device_memory.cpp` — PF/VF/COMPILE devices have no
  host-accessible device memory at all; every method is a constant, a
  `nullptr`, or an explicit "not implemented" error.
- `mock_device/{device_memory_mock,mmapped_storage}.*` — the mock backend is
  a local `mmap`/`shm`-backed file, an OS-syscall boundary, not senlib. The
  `mmap`/`munmap` libc binding this requires is kept as a small local
  `extern "C"` block in `device_types.rs` rather than a `senlib_ffi_*.rs`
  file, specifically so it isn't miscounted as part of the senlib ledger.
- `compiler_interface/dee_graph_converter.cpp`'s five in-scope functions
  (`get_dtcompiler_export_dir`, `remove_dtcompiler_export_dir`,
  `maybe_wrap_dim`, `convert_shape_to_vecint64[_rev]`) — env/config lookups,
  filesystem cleanup, and plain arithmetic. The rest of that file (`dee::PBD`,
  DeepRT invocation) is deep compiler-only machinery outside the
  ABI-reachable closure.
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
subsystem's source directory, but not ported — `compiler_interface.rs`'s
own scope note only covers the five `dee_graph_converter.cpp` functions
above. `device_interface/iommu/{iommu_mapping.cpp,iommu_raii_buffer.hpp}`
and `device_types/pf_device/pf_device_handle.cpp` are likewise unported;
`scheduler.rs`'s doc comment refers to "`IommuMapperInterface`/`DeviceHandle`"
as belonging to "the device-interface phase's files," but no such module
exists in this crate.

## Coverage cross-check against the master scope (134 types / 320 functions)

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
control-block fixup/DMA/compute params, multi-device, config/util,
device-interface-misc's in-scope slice, telemetry) is accounted for either
as a native port, an FFI wrapper documented above, or one of the
reasoned/time-boxed stubs listed per subsystem. The gaps above
should be picked up as a follow-on pass; none of them are on the allocator/
runtime-core/scheduler-submission/multi-device-transfer critical path this
crate's existing tests exercise, but they are real entries in the
320-function closure and are not yet represented anywhere in this crate.
