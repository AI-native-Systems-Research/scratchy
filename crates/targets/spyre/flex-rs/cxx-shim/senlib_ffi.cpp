// SPDX-License-Identifier: Apache-2.0
//
// flex-rs's real senlib FFI binding. This is the ONLY C++ compiled into
// flex-rs, and it calls ONLY `senlib::` — never `flex::`. Device bring-up
// (opening the card, reading its real HBM size, obtaining CB/RB interface
// pointers) is done here directly against `senlib::v2::PfWrapper`/
// `PfInterface`, the same primitives flex's own `PfDeviceHandle` constructor
// uses internally (verified against
// flex/src/device_types/pf_device/pf_device_handle.cpp) — but the
// orchestration (which senlib calls to make, in what order, P2P/NTB setup,
// IOMMU mapping, etc.) is NOT replicated here beyond single-device bring-up;
// all of THAT logic that this crate needs lives in `src/device_handle.rs`.
//
// Every function here is a one-line marshal over a real, public senlib call.
// If a function in this file grows a branch that isn't "did the senlib call
// succeed", it's in the wrong file — move that logic to device_handle.rs.
//
// Every body is wrapped try/catch(...) so a C++ exception never unwinds
// across the C ABI into Rust (UB) — errors are logged to stderr and reported
// as a sentinel/null return instead.

#include <senlib/1p0/senpci.hpp>
#include <senlib/1p0/pf_wrapper.hpp>
#include <senlib/1p0/pf_interface.hpp>
#include <senlib/1p0/control_block_interface.hpp>
#include <senlib/1p0/response_block_interface.hpp>
#include <senlib/shared/memory_allocator.hpp>
#include <senlib/shared/senpci_shared.hpp>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

// Process-wide PF device handle, brought up once. Mirrors flex's own
// call_once/shared_ptr pattern for the same reason: opening the device is
// expensive (real hardware bring-up) and must happen exactly once per
// process, not once per FFI call.
std::shared_ptr<senlib::v2::PfWrapper> g_pfw;
std::once_flag g_pfw_once;
int g_pfw_init_rc = -1;

// Diagnostic channel for the exceptions this ABI boundary is obliged to catch.
//
// A C++ exception cannot cross an `extern "C"` frame into Rust, so every entry
// point below MUST catch. The failure mode that caused a real hardware hang was
// not the catching -- it was throwing the information away: the payload went to
// stderr and Rust received only a status code (or, worse, a value indicating
// success). A stderr line is also easy to misread as senlib reporting a
// hardware condition rather than the shim announcing it just ate an exception.
//
// So each catch block records `what()` here and the Rust wrapper attaches it to
// its `Err`, putting the RAS name and message into tracing where it belongs.
static thread_local std::string g_last_error;

static void shim_record_error(const char* fn, const char* what) {
    g_last_error = std::string(fn) + ": " + what;
    fprintf(stderr, "[senlib-ffi] %s\n", g_last_error.c_str());
}

// Returns the calling thread's most recent shim error, or "" if none. The
// pointer is owned by the shim and valid until this thread's next failed call.
//
// `extern "C"` explicitly: this definition sits ABOVE the file's big
// `extern "C" { ... }` block (it has to, since the entry points inside that
// block call `shim_record_error`), so without this it would get C++ linkage and
// the name Rust links against would not exist. `cargo check` does not catch
// that — it never links — so this only shows up as `undefined symbol` at the
// final binary link.
extern "C" const char* flex_shim_last_error(void) {
    return g_last_error.c_str();
}

void init_pfw_once(int32_t /*logical_device_id*/) {
    std::call_once(g_pfw_once, [] {
        try {
            // Real device identification + open: senlib::v2::SenPci::pf()
            // with no argument selects "first available device", matching
            // flex's own CreatePciId() fallback path
            // (pf_device_handle.cpp:145) for the single-device case this ABI
            // slice needs. senlib::v2::PfWrapper's constructor is the actual
            // VFIO device open.
            auto pci_id = senlib::v2::SenPci::pf();
            g_pfw = std::make_shared<senlib::v2::PfWrapper>(pci_id);
            g_pfw_init_rc = 0;
        } catch (const std::exception& e) {
            fprintf(stderr, "[senlib-ffi] init_pfw: %s\n", e.what());
            g_pfw_init_rc = -2;
        } catch (...) {
            fprintf(stderr, "[senlib-ffi] init_pfw: unknown exception\n");
            g_pfw_init_rc = -2;
        }
    });
}

// Process-wide memory allocator, constructed once GetLPDDRSize() is known.
// Mirrors flex::DeviceMemoryAllocator's own PF-mode construction
// (device_memory_allocator.cpp:122): base = reinterpret_cast<void*>(alignment)
// (a nonzero placeholder so device address 0 is never returned as valid, per
// senlib's own memory_allocator.hpp doc comment), size = real LPDDR size,
// page_size = DEVICE_ALIGNMENT (128, matching flex-rs's own
// allocator.rs::DEVICE_ALIGNMENT).
constexpr uint64_t kDeviceAlignment = 128;

std::shared_ptr<senlib::v2::MemoryAllocator> g_mem_allocator;
std::once_flag g_mem_allocator_once;

senlib::v2::MemoryAllocator* mem_allocator() {
    std::call_once(g_mem_allocator_once, [] {
        auto size = g_pfw->interface()->GetLPDDRSize();
        g_mem_allocator = std::make_shared<senlib::v2::MemoryAllocator>(
            reinterpret_cast<void*>(kDeviceAlignment), size, kDeviceAlignment);
    });
    return g_mem_allocator.get();
}

// Process-wide handle table for outstanding device-memory allocations.
// senlib::v2::MemoryAllocator::Allocate returns a raw void*; Rust's
// senlib_ffi_allocator::SenlibAllocationHandle is a bare u64, so this table
// is what turns "free this u64" back into "free this void*" (the pointer
// value itself, widened to u64, IS the handle — no separate ownership object
// to track, unlike the old flex::DeviceMemoryAllocationPtr shared_ptr table).
std::mutex g_alloc_table_mu;
std::unordered_map<uint64_t, void*> g_alloc_table;

}  // namespace

extern "C" {

// ---------------------------------------------------------------------
// Runtime bring-up (senlib_ffi.rs)
// ---------------------------------------------------------------------

int32_t flex_shim_init_runtime(int32_t logical_device_id) {
    try {
        init_pfw_once(logical_device_id);
        return (g_pfw_init_rc == 0 && g_pfw) ? 0 : (g_pfw_init_rc == 0 ? -1 : g_pfw_init_rc);
    } catch (const std::exception& e) {
        fprintf(stderr, "[senlib-ffi] flex_shim_init_runtime: %s\n", e.what());
        return -2;
    } catch (...) {
        fprintf(stderr, "[senlib-ffi] flex_shim_init_runtime: unknown exception\n");
        return -2;
    }
}

// Real device memory capacity, in bytes — senlib::v2::PfInterface::GetLPDDRSize(),
// the exact senlib call flex's own PfDeviceHandle constructor uses
// (pf_device_handle.cpp:153: `pfw_->interface()->GetLPDDRSize()`).
uint64_t flex_senlib_device_memory_size(void) {
    try {
        if (!g_pfw) {
            fprintf(stderr, "[senlib-ffi] flex_senlib_device_memory_size: runtime not initialized\n");
            return 0;
        }
        return static_cast<uint64_t>(g_pfw->interface()->GetLPDDRSize());
    } catch (const std::exception& e) {
        fprintf(stderr, "[senlib-ffi] flex_senlib_device_memory_size: %s\n", e.what());
        return 0;
    } catch (...) {
        fprintf(stderr, "[senlib-ffi] flex_senlib_device_memory_size: unknown exception\n");
        return 0;
    }
}

// Real RCU (compute-core) count on the open card —
// senlib::v2::SenPci::RCU_num_cores(), the exact senlib call
// flex::populateDomainTopology uses to build the 1p0 target's single-domain
// core-affinity vector (device_memory_topology.cpp:32). Static/no bring-up
// prerequisite, but declared alongside flex_senlib_device_memory_size since
// both feed the same DeviceTopology population call.
uint32_t flex_senlib_rcu_num_cores(void) {
    try {
        return static_cast<uint32_t>(senlib::v2::SenPci::RCU_num_cores());
    } catch (const std::exception& e) {
        fprintf(stderr, "[senlib-ffi] flex_senlib_rcu_num_cores: %s\n", e.what());
        return 0;
    } catch (...) {
        fprintf(stderr, "[senlib-ffi] flex_senlib_rcu_num_cores: unknown exception\n");
        return 0;
    }
}

// ---------------------------------------------------------------------
// senlib_ffi_allocator.rs — device memory allocate/free
//   senlib::v2::MemoryAllocator::Allocate(bytes) -> void* / Free(void*)
// ---------------------------------------------------------------------

uint64_t flex_senlib_memory_allocate(uint64_t num_bytes) {
    try {
        if (!g_pfw) {
            fprintf(stderr, "[senlib-ffi] flex_senlib_memory_allocate: runtime not initialized\n");
            return 0;
        }
        // This 1p0 target has exactly one memory domain (matches flex's own
        // single-domain PF topology). A domain_id parameter was removed from
        // this signature (and the Rust extern decl in
        // senlib_ffi_allocator.rs) because neither senlib::MemoryAllocator::
        // Allocate(bytes) (PF) nor VfWrapper::Allocate(flits) (VF) takes one —
        // see flex-cxx's own TODO at device_memory_allocator.cpp:153. Keeping
        // an unforwarded, unread param here was a real Rust/C++ ABI arity
        // mismatch (harmless on x86_64 System V since the extra arg just sat
        // in an ignored register, but exactly the kind of unlocked skeleton
        // this hardening pass exists to remove).
        void* ptr = mem_allocator()->Allocate(static_cast<size_t>(num_bytes));
        if (ptr == nullptr) {
            return 0;
        }
        uint64_t handle = reinterpret_cast<uint64_t>(ptr);
        if (handle == 0) {
            fprintf(stderr, "[senlib-ffi] flex_senlib_memory_allocate: pointer 0 collides with failure sentinel\n");
            return 0;
        }
        std::lock_guard<std::mutex> lock(g_alloc_table_mu);
        g_alloc_table[handle] = ptr;
        return handle;
    } catch (const std::exception& e) {
        fprintf(stderr, "[senlib-ffi] flex_senlib_memory_allocate: %s\n", e.what());
        return 0;
    } catch (...) {
        fprintf(stderr, "[senlib-ffi] flex_senlib_memory_allocate: unknown exception\n");
        return 0;
    }
}

int32_t flex_senlib_memory_free(uint64_t handle) {
    try {
        if (handle == 0) {
            return 0;
        }
        std::lock_guard<std::mutex> lock(g_alloc_table_mu);
        auto it = g_alloc_table.find(handle);
        if (it == g_alloc_table.end()) {
            shim_record_error("flex_senlib_memory_free", "unknown handle");
            return -1;
        }
        mem_allocator()->Free(it->second);
        g_alloc_table.erase(it);
        return 0;
    } catch (const std::exception& e) {
        shim_record_error("flex_senlib_memory_free", e.what());
        return -1;
    } catch (...) {
        shim_record_error("flex_senlib_memory_free", "unknown exception");
        return -1;
    }
}

// ---------------------------------------------------------------------
// senlib_ffi_scheduler.rs — control-block-interface capacity / doorbell,
// response-block receive.
//   senlib::v2::PfInterface::GetComputeCbi()/GetDmaiCbi()/GetDmaoCbi()
//   senlib::v2::PfInterface::GetRbi()
// (senlib/1p0/pf_interface.hpp)
// ---------------------------------------------------------------------

enum FlexShimPipeline : int32_t {
    kFlexShimPipelineComputeCbi = 0,
    kFlexShimPipelineAsyncDmaICbi = 1,
    kFlexShimPipelineAsyncDmaOCbi = 2,
};

static senlib::v2::ControlBlockInterface* cbi_for_pipeline(int32_t pipeline) {
    if (!g_pfw) {
        return nullptr;
    }
    auto* iface = g_pfw->interface();
    switch (pipeline) {
        case kFlexShimPipelineComputeCbi:
            return iface->GetCmptCbi();
        case kFlexShimPipelineAsyncDmaICbi:
            return iface->GetDmaiCbi();
        case kFlexShimPipelineAsyncDmaOCbi:
            return iface->GetDmaoCbi();
        default:
            return nullptr;
    }
}

void* flex_shim_cbi_for_pipeline(int32_t pipeline) {
    try {
        return static_cast<void*>(cbi_for_pipeline(pipeline));
    } catch (...) {
        fprintf(stderr, "[senlib-ffi] flex_shim_cbi_for_pipeline: exception\n");
        return nullptr;
    }
}

void* flex_shim_rbi_for_pipeline(int32_t /*pipeline*/) {
    try {
        if (!g_pfw) {
            return nullptr;
        }
        // senlib exposes exactly one Rbi() per device.
        //
        // `PfInterface::GetRbi()` is declared as returning
        // `ResponseBlockInterfaceImpl*` (senlib/1p0/pf_interface.hpp:88), but
        // the type every consumer works with -- and the type
        // `senlib_response_block_interface_receive_responses_sbf` below casts
        // this void* back to -- is the polymorphic base
        // `senlib::v2::ResponseBlockInterface`
        // (senlib/1p0/response_block_interface.hpp:31,52). The upcast must
        // therefore happen HERE, on the typed pointer, not implicitly through
        // void*: `static_cast<void*>(Impl*)` followed by
        // `static_cast<Base*>(void*)` reinterprets the derived address as a
        // base address, which is only accidentally correct when the base
        // subobject happens to sit at offset 0. Matches how
        // `cbi_for_pipeline` above already returns the base
        // `ControlBlockInterface*`.
        return static_cast<void*>(static_cast<senlib::v2::ResponseBlockInterface*>(g_pfw->interface()->GetRbi()));
    } catch (...) {
        fprintf(stderr, "[senlib-ffi] flex_shim_rbi_for_pipeline: exception\n");
        return nullptr;
    }
}

// Status + out-param rather than a bare `size_t`: `0` is a LEGAL return from
// `capacity()` that upper layers read as "capacity not reported", which makes
// `hasQueueCapacity` allow every submission. Encoding a throw as `0` therefore
// silently disabled the only backpressure in the system -- the same class of
// bug as the swallowed submit that wedged the card.
int32_t senlib_control_block_interface_capacity(void* cbi, size_t* out_capacity) {
    try {
        if (!cbi || !out_capacity) {
            shim_record_error("senlib_control_block_interface_capacity", "null cbi/out_capacity");
            return -1;
        }
        *out_capacity = static_cast<senlib::v2::ControlBlockInterface*>(cbi)->capacity();
        return 0;
    } catch (const std::exception& e) {
        shim_record_error("senlib_control_block_interface_capacity", e.what());
        return -1;
    } catch (...) {
        shim_record_error("senlib_control_block_interface_capacity", "unknown exception");
        return -1;
    }
}

// Returns 0 on success, -1 if the submission did not happen.
//
// The real call site (response_worker.cpp:1268,
// `cbi->QueueControlBlocksSBF(&cbr->GetCB(0), cbr->GetNumberOfCBs())`) is a
// bare `void` call inside no try/catch at all: if senlib throws, the exception
// propagates out of ResponseWorker and tears the request down. A C++ exception
// cannot cross this C ABI, so the closest faithful behavior is to report the
// failure to Rust and let the Rust wrapper raise it -- NOT to log and return
// as if the doorbell had been rung, which converts a hard submit failure into
// an eternally-outstanding request (the caller has already incremented its
// in-flight count by then). Likewise a null `cbi` is reported rather than
// treated as a successful no-op: the real code has no such path (it
// dereferences the CBI unconditionally).
int32_t senlib_control_block_interface_queue_control_blocks_sbf(void* cbi, const void* cbs, size_t count) {
    try {
        if (!cbi || !cbs) {
            fprintf(stderr, "[senlib-ffi] senlib_control_block_interface_queue_control_blocks_sbf: null cbi/cbs\n");
            return -1;
        }
        auto* iface = static_cast<senlib::v2::ControlBlockInterface*>(cbi);
        auto* src = static_cast<const SentientSoc::V1::ControlBlockSBF*>(cbs);
        // `count == 0` is forwarded, not short-circuited: the real code passes
        // `GetNumberOfCBs()` straight through and senlib owns the semantics.
        iface->QueueControlBlocksSBF(src, static_cast<uint64_t>(count));
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[senlib-ffi] senlib_control_block_interface_queue_control_blocks_sbf: %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr,
                "[senlib-ffi] senlib_control_block_interface_queue_control_blocks_sbf: unknown exception\n");
        return -1;
    }
}

// Port of `ControlBlockInterface::FreeResponses(uint64_t num_responses = 0)`
// (`senlib/1p0/control_block_interface.hpp:49`, pure virtual). Called by
// `ResponseWorker::ParseResponseBlocks` once per pipeline per drain
// (`response_worker.cpp:1141-1152`) to return consumed response-block slots to
// senlib. Until this symbol existed the port never returned that credit, so
// senlib's queue occupancy grew monotonically and `QueueControlBlocksSBF`
// eventually threw "Control block queue is full, and has not drained" at
// exactly `cb_queue_length` CBs -- the reproducible long-context batched-decode
// hang. Same error convention as the queue call above: report failure to Rust
// rather than swallowing it.
int32_t senlib_control_block_interface_free_responses(void* cbi, uint64_t num_responses) {
    try {
        if (!cbi) {
            fprintf(stderr, "[senlib-ffi] senlib_control_block_interface_free_responses: null cbi\n");
            return -1;
        }
        auto* iface = static_cast<senlib::v2::ControlBlockInterface*>(cbi);
        iface->FreeResponses(num_responses);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[senlib-ffi] senlib_control_block_interface_free_responses: %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "[senlib-ffi] senlib_control_block_interface_free_responses: unknown exception\n");
        return -1;
    }
}

// Status + out-param rather than a bare count: the real drain calls this with
// `min_count == 0`, so `0` means "nothing has arrived yet" and is the normal,
// expected answer. Reporting a throw as `0` made an RBI failure indistinguishable
// from an idle poll, so the drain loop would spin to its full timeout instead of
// surfacing the error.
int32_t senlib_response_block_interface_receive_responses_sbf(void* rbi, void* out, size_t max_count,
                                                              size_t min_count, uint64_t* out_received) {
    try {
        if (!rbi || !out || !out_received) {
            shim_record_error("senlib_response_block_interface_receive_responses_sbf", "null rbi/out/out_received");
            return -1;
        }
        auto* iface = static_cast<senlib::v2::ResponseBlockInterface*>(rbi);
        auto* dest = static_cast<SentientSoc::V1::ResponseBlockSBF*>(out);
        *out_received = iface->ReceiveResponsesSBF(dest, static_cast<uint64_t>(max_count),
                                                   static_cast<uint64_t>(min_count));
        return 0;
    } catch (const std::exception& e) {
        shim_record_error("senlib_response_block_interface_receive_responses_sbf", e.what());
        return -1;
    } catch (...) {
        shim_record_error("senlib_response_block_interface_receive_responses_sbf", "unknown exception");
        return -1;
    }
}

// ---------------------------------------------------------------------
// senlib_ffi_runtime.rs / senlib_ffi_config_util.rs — card count
//   senlib::v2::SenPci::ncards() — a trivial static query with no bring-up
//   prerequisite.
// ---------------------------------------------------------------------

uint64_t flex_senlib_pci_ncards(void) {
    try {
        return static_cast<uint64_t>(senlib::v2::SenPci::ncards());
    } catch (const std::exception& e) {
        fprintf(stderr, "[senlib-ffi] flex_senlib_pci_ncards: %s\n", e.what());
        return 0;
    } catch (...) {
        fprintf(stderr, "[senlib-ffi] flex_senlib_pci_ncards: unknown exception\n");
        return 0;
    }
}

// ---------------------------------------------------------------------
// senlib_ffi_config_util.rs — per-card PCIe bus address
//   senlib::v2::SenPciShared::pf(card_index).to_string()
//   (flex/src/util/flex_config.cpp:471, FlexConfig::RdmaGetPCIeAddress)
// ---------------------------------------------------------------------

const char* senlib_senpci_pf_address(uint32_t card_index) {
    try {
        // senlib_ffi_config_util.rs's doc comment promises "a senlib-owned,
        // null-terminated buffer valid for the duration of the call (no
        // ownership transfer)" — a thread_local std::string buffer backing
        // the returned c_str() matches that contract (stable until the next
        // call on this thread, no ownership transfer to Rust).
        static thread_local std::string addr;
        addr = senlib::v2::SenPciShared::pf(card_index).to_string();
        return addr.c_str();
    } catch (const std::exception& e) {
        fprintf(stderr, "[senlib-ffi] senlib_senpci_pf_address: %s\n", e.what());
        return nullptr;
    } catch (...) {
        fprintf(stderr, "[senlib-ffi] senlib_senpci_pf_address: unknown exception\n");
        return nullptr;
    }
}

// ---------------------------------------------------------------------
// senlib_ffi_iommu.rs — pf_iommu_mapper.cpp's Map/Unmap
// ---------------------------------------------------------------------

// `writable` matches the real `PfIommuMapper::Map(IommuMappingPtr*, void* hmva,
// size_t size_bytes, bool writable)` signature (pf_iommu_mapper.cpp:44). As in
// the real implementation, it is intentionally NOT forwarded to
// RegisterMemForIOMMU/pin_and_map below (neither takes a writable/permission
// argument there either, pf_iommu_mapper.cpp:48-49) -- real Map() only uses
// it to construct the IommuMapping bookkeeping object it returns to its
// caller, which on the Rust side is handled by iommu.rs::PfIommuMapper::map
// after this call returns. Accepted here purely so this shim's signature
// matches the real Map() it mirrors.
int32_t flex_senlib_iommu_register_and_map(void* hmva, uint64_t size_bytes, bool writable, void** out_iova, uint64_t* out_size) {
    (void)writable;
    try {
        if (!g_pfw) {
            fprintf(stderr, "[senlib-ffi] flex_senlib_iommu_register_and_map: runtime not initialized\n");
            return -1;
        }
        // Exact call pair, in this order, from pf_iommu_mapper.cpp:50-51.
        // Both take the RAW, un-rounded size_bytes there -- only Unmap
        // page-rounds (see below), and that asymmetry is deliberate in the
        // C++ (it carries its own `TODO(mschaal): Should we really try to fix
        // this here?`), so it is reproduced rather than "corrected".
        g_pfw->interface()->RegisterMemForIOMMU(hmva, size_bytes);
        auto pm = g_pfw->interface()->pin_and_map(hmva, size_bytes);
        if (pm.dev_shm_sglist.empty()) {
            // pf_iommu_mapper.cpp:53-56: RAS::DEVICE_PF::IommuNoMapping().Throw().
            fprintf(stderr, "[senlib-ffi] flex_senlib_iommu_register_and_map: IommuNoMapping (empty dev_shm_sglist)\n");
            return -1;
        }
        // Only dev_shm_sglist.front() is consulted, and its `length` (not the
        // requested size_bytes) becomes the mapping's SizeInBytes --
        // pf_iommu_mapper.cpp:57-59. Verified faithful; do not "fix" into a
        // multi-entry walk.
        *out_iova = pm.dev_shm_sglist.front().ptr;
        *out_size = pm.dev_shm_sglist.front().length;
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[senlib-ffi] flex_senlib_iommu_register_and_map: %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "[senlib-ffi] flex_senlib_iommu_register_and_map: unknown exception\n");
        return -1;
    }
}

int32_t flex_senlib_iommu_unmap_and_unregister(void* hmva, void* iova, uint64_t size_bytes) {
    try {
        if (!g_pfw) {
            fprintf(stderr, "[senlib-ffi] flex_senlib_iommu_unmap_and_unregister: runtime not initialized\n");
            return -1;
        }
        // pf_iommu_mapper.cpp:90-93: a fresh `pinned_memory` carrying exactly
        // one sgpair {iova, page-rounded size}, then unpin_and_unmap followed
        // by UnregisterMemForIOMMU with the SAME page-rounded size (the
        // rounding itself is done by the caller, iommu.rs::unmap, mirroring
        // `sendnn::ceil_to(mapping.SizeInBytes(), page_size_)`).
        // `senlib::v2::sgpair` is `{void* ptr; size_t length;}`
        // (senlib/shared/memory_allocator_util.hpp:31-34), so this
        // brace-initializer order matches the C++'s.
        senlib::v2::pinned_memory pm;
        pm.dev_shm_sglist.push_back({iova, static_cast<size_t>(size_bytes)});
        g_pfw->interface()->unpin_and_unmap(pm);
        g_pfw->interface()->UnregisterMemForIOMMU(hmva, static_cast<size_t>(size_bytes));
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[senlib-ffi] flex_senlib_iommu_unmap_and_unregister: %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "[senlib-ffi] flex_senlib_iommu_unmap_and_unregister: unknown exception\n");
        return -1;
    }
}

}  // extern "C"
