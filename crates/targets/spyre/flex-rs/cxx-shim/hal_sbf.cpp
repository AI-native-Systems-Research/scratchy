// SPDX-License-Identifier: Apache-2.0
//
// flex-rs's hal SBF wire-door shim: C-ABI builders and decoders over the
// Spyre SDK's own `hal/snt1p0` SBF headers — the R5 control-block section,
// the Rdma12 unicast-send DMO section, the Rdma10 wdone-wait DMI section,
// and the R5 appdata arm of the response block.
//
// THE LAW (the proprietary line): the hal headers are OCO-marked trade-secret
// source. Scratchy Rust source never includes them and never carries their
// field maps, bit ranges, enum bit values, or packing logic — Rust hands THIS
// file semantic scalars (modes by meaning, byte offsets, sids, masks) and
// reads back finished words. ALL bit packing lives on this side of the
// boundary, and even here it is not copied vendor source: every builder
// constructs the SDK's own section types and calls their own setters — the
// same calls the vendor's own scheduler makes
// (flex/src/runtime_stream/1p0/pf_runtime_scheduler.cpp, the unicast
// ten-setter loop and the wdone wait; citation-only, pod-verified
// 2026-10-30). The finished words are the section object's own bytes: the
// very bytes `ControlBlockSBF::set_r5`/`set_dmo`/`set_dmi` would copy into a
// control block.
//
// Every entry validates its semantic inputs and REFUSES by name (return -1,
// reason via `flex_hal_last_error()`) — it never clamps and never invents a
// value. The laws that belong to the tp builders (the never-zero chunk, the
// −128 peer-base compensation, the multicast/T-send refusals) stay in Rust
// and are NOT re-implemented here: this shim encodes exactly what it is
// handed.
//
// Every body is wrapped try/catch(...) so a C++ exception never unwinds
// across the C ABI into Rust — same rule as senlib_ffi.cpp.

#include <hal/snt1p0/r5_control_block_section_sbf.hpp>
#include <hal/snt1p0/rdma_control_block_section_sbf.hpp>
#include <hal/snt1p0/virtual_address_sbf.hpp>
#include <hal/snt1p0/app_data_sbf.hpp>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

// Diagnostic channel, mirroring senlib_ffi.cpp's: a refused input or a
// caught exception must not cross the C ABI silently. Rust's wrappers
// attach this to their Err via `flex_hal_last_error()`.
thread_local std::string g_hal_last_error;

void hal_record_error(const char* fn, const char* what) {
    g_hal_last_error = std::string(fn) + ": " + what;
    fprintf(stderr, "[hal-sbf] %s\n", g_hal_last_error.c_str());
}

// The finished-words copy. A section object IS the wire format — the control
// block assigns these very objects into its storage — so the object's bytes
// out are the bytes the device sees. Each door fixes the word count with a
// static_assert on the section's sizeof.
template <typename SectionT>
void section_words(const SectionT& section, uint64_t* out_words) {
    std::memcpy(out_words, &section, sizeof(SectionT));
}

}  // namespace

extern "C" {

// Returns the calling thread's most recent hal_sbf error, or "" if none.
// Owned by the shim and valid until this thread's next failed call.
const char* flex_hal_last_error(void) {
    return g_hal_last_error.c_str();
}

// The R5 control-block section (1 word). `mode` is OUR ordinal: 0 = skip
// (the section is skipped — the all-zero QGI word), 1 = r5_only, 2 = spawn.
// `core_mask` carries the header's own core-mask semantic, bit 0 =
// supervisor and bit N = worker N (r5_control_block_section_sbf.hpp:29-31),
// 9 bits total. The device address is (segment, offset_bytes); the header's
// own SegOff path converts bytes to flits (virtual_address_sbf.hpp:68-76).
// Skip mode carries no mask and no address — a skip that names either is a
// caller bug and refuses rather than silently discarding them.
int32_t flex_hal_r5_section_encode(uint32_t mode, uint64_t core_mask,
                                   uint64_t segment, uint64_t offset_bytes,
                                   uint64_t* out_word) {
    static_assert(sizeof(hal::snt1p0::R5ControlBlockSectionSBF) == 8,
                  "R5 section is one 8-byte word");
    try {
        if (out_word == nullptr) {
            hal_record_error("flex_hal_r5_section_encode", "null out_word");
            return -1;
        }
        if (mode > 2) {
            hal_record_error("flex_hal_r5_section_encode",
                             "mode must be 0 (skip), 1 (r5_only) or 2 (spawn)");
            return -1;
        }
        if ((core_mask >> hal::snt1p0::NUM_R5_CORES) != 0) {
            hal_record_error("flex_hal_r5_section_encode",
                             "core_mask exceeds the 9 core-mask bits");
            return -1;
        }
        if (segment > 7) {
            hal_record_error("flex_hal_r5_section_encode",
                             "segment exceeds the 3-bit segment id");
            return -1;
        }
        hal::snt1p0::R5ControlBlockSectionSBF r5;
        if (mode == 0) {
            if (core_mask != 0 || segment != 0 || offset_bytes != 0) {
                hal_record_error("flex_hal_r5_section_encode",
                                 "skip mode carries no core mask and no address");
                return -1;
            }
            // The default-constructed QGI word: all zero.
        } else {
            const auto type = mode == 1 ? hal::snt1p0::R5TypeEnum::R5ONLY
                                        : hal::snt1p0::R5TypeEnum::SPAWN;
            // The header's own (type, segment_id, offset_BYTES) constructor.
            r5 = hal::snt1p0::R5ControlBlockSectionSBF(type, segment, offset_bytes);
            r5.SetCoreMasks(hal::snt1p0::core_masks(core_mask));
        }
        *out_word = r5.GetValue();
        return 0;
    } catch (const std::exception& e) {
        hal_record_error("flex_hal_r5_section_encode", e.what());
        return -1;
    } catch (...) {
        hal_record_error("flex_hal_r5_section_encode", "unknown exception");
        return -1;
    }
}

// The Rdma12 RDMA_SINGLECAST send DMO section (2 words): the vendor
// scheduler's ten-setter unicast send shape. `sid` is the wdone signal
// index and `tid` the destination BAT index (the vendor's dest_rank).
// `length_bytes` is the chunk's byte length: the 15-bit flit field is
// zero-means-max, so the legal range is (0, 4 MiB] — 4 MiB stores the zero
// sentinel and reads back as the 32768-flit maximum (the vendor's
// P2P_RDMA_WINDOW_SIZE). The source device address is (segment,
// offset_bytes). `dest_remote_base_bytes` is the peer's remote base for
// this transfer in bytes, compensation already applied by the caller — the
// −128 law is the tp builders', not the shim's. The shim composes the xseg
// word exactly as the vendor loop does: flits, then the valid bit (30),
// then odd parity over the result (31). `ta` is hardwired 0 and the endian
// field keeps its constructor default, both per the vendor loop.
// first/last/sync/xsegv are the caller's protocol bits.
int32_t flex_hal_rdma12_unicast_send_encode(
    uint64_t sid, uint64_t tid, uint64_t length_bytes,
    uint64_t src_segment, uint64_t src_offset_bytes,
    uint64_t dest_remote_base_bytes,
    uint8_t first, uint8_t last, uint8_t sync, uint8_t xsegv,
    uint64_t* out_words) {
    static_assert(sizeof(hal::snt1p0::Rdma12ControlBlockSectionSBF) == 16,
                  "Rdma12 section is two 8-byte words");
    try {
        if (out_words == nullptr) {
            hal_record_error("flex_hal_rdma12_unicast_send_encode", "null out_words");
            return -1;
        }
        if (sid > 255) {
            hal_record_error("flex_hal_rdma12_unicast_send_encode",
                             "sid exceeds the 8-bit sid field");
            return -1;
        }
        if (tid > 63) {
            hal_record_error("flex_hal_rdma12_unicast_send_encode",
                             "tid exceeds the 6-bit tid field");
            return -1;
        }
        if (src_segment > 7) {
            hal_record_error("flex_hal_rdma12_unicast_send_encode",
                             "src_segment exceeds the 3-bit segment id");
            return -1;
        }
        // 4 MiB == the vendor's P2P_RDMA_WINDOW_SIZE == the zero-means-max
        // sentinel: every other length must fit the 15-bit flit field.
        constexpr uint64_t kWindowBytes = 1ULL << 22;
        if (length_bytes == 0 || length_bytes > kWindowBytes) {
            hal_record_error("flex_hal_rdma12_unicast_send_encode",
                             "length_bytes must be in (0, 4 MiB] (15-bit flit field, "
                             "zero-means-max)");
            return -1;
        }
        // The xseg word's flits must leave bits 30 (valid) and 31 (parity)
        // free.
        const uint64_t dest_flits =
            hal::snt1p0::VirtualAddressSBF::bytes_to_flits(dest_remote_base_bytes);
        if ((dest_flits >> 30) != 0) {
            hal_record_error("flex_hal_rdma12_unicast_send_encode",
                             "dest_remote_base_bytes overflows the xseg word");
            return -1;
        }

        hal::snt1p0::Rdma12ControlBlockSectionSBF rdma(
            hal::snt1p0::DmaTypeEnum::RDMA_SINGLECAST);
        rdma.SetTa(0);  // hardwired by the vendor loop
        rdma.SetTid(tid);
        rdma.SetSid(sid);
        rdma.SetFirst(first != 0);
        rdma.SetLast(last != 0);
        rdma.SetSync(sync != 0);
        rdma.SetXsegv(xsegv != 0);
        rdma.SetLengthBytes(length_bytes);
        uint64_t xseg = dest_flits;
        xseg |= 1ULL << 30;  // the valid bit
        xseg |= static_cast<uint64_t>(std::popcount(xseg) & 1ULL) << 31;  // odd parity
        rdma.SetXseg(xseg);
        // Source device address via the mixin's own (segment_id,
        // offset_flits) overload; bytes_to_flits is the header's conversion.
        rdma.SetDeviceAddress(
            src_segment,
            hal::snt1p0::VirtualAddressSBF::bytes_to_flits(src_offset_bytes));
        section_words(rdma, out_words);
        return 0;
    } catch (const std::exception& e) {
        hal_record_error("flex_hal_rdma12_unicast_send_encode", e.what());
        return -1;
    } catch (...) {
        hal_record_error("flex_hal_rdma12_unicast_send_encode", "unknown exception");
        return -1;
    }
}

// The Rdma10 RDMA_WDONE_BARRIER wait DMI section (2 words): waits for the
// write-done counters of every sid in `sids` — one list_bitmask bit per
// sid, composed here from the list. Which sids to wait on is the caller's
// law; an empty list refuses (a wait over nothing is not a wait this
// runtime ever means to issue).
int32_t flex_hal_rdma10_wdone_wait_encode(const uint64_t* sids, uint32_t sid_count,
                                          uint8_t sync, uint64_t* out_words) {
    static_assert(sizeof(hal::snt1p0::RdmaWdoneControlBlockSectionSBF) == 16,
                  "Rdma10 section is two 8-byte words");
    try {
        if (out_words == nullptr) {
            hal_record_error("flex_hal_rdma10_wdone_wait_encode", "null out_words");
            return -1;
        }
        if (sid_count == 0 || sids == nullptr) {
            hal_record_error("flex_hal_rdma10_wdone_wait_encode",
                             "empty wdone sid list");
            return -1;
        }
        uint64_t mask = 0;
        for (uint32_t i = 0; i < sid_count; ++i) {
            if (sids[i] > 63) {
                hal_record_error("flex_hal_rdma10_wdone_wait_encode",
                                 "sid exceeds the 64-bit list_bitmask");
                return -1;
            }
            mask |= 1ULL << sids[i];
        }
        hal::snt1p0::RdmaWdoneControlBlockSectionSBF wait;
        wait.SetListbitmask(mask);
        wait.SetSync(sync != 0);
        section_words(wait, out_words);
        return 0;
    } catch (const std::exception& e) {
        hal_record_error("flex_hal_rdma10_wdone_wait_encode", e.what());
        return -1;
    } catch (...) {
        hal_record_error("flex_hal_rdma10_wdone_wait_encode", "unknown exception");
        return -1;
    }
}

// The semantic mirror of the R5 appdata arm — plain uint64_t fields, no bit
// layout. The Rust side (senlib_ffi.rs's B.1 wrappers) mirrors this layout
// #[repr(C)]. `error_type` carries AppR5Section::ErrorType by value (BEU /
// EXCEPTION); the raw dwords ride along for diagnostics.
struct FlexHalR5AppData {
    // Header
    uint64_t cb_type;
    uint64_t wcore_compl;
    uint64_t worker_mask;
    uint64_t core_id;
    uint64_t completion_msg;
    uint64_t error_type;
    uint64_t job_cc;
    // Branch target data (kills and BEU)
    uint64_t branch_target_data;
    // BEU error
    uint64_t beu_value;
    uint64_t beu_cause;
    uint64_t beu_accrued;
    // R5 exception
    uint64_t interrupt_flag;
    uint64_t mcause_code;
    uint64_t mip;
    uint64_t mepc;
    uint64_t mtval;
    // Raw dwords
    uint64_t dw0;
    uint64_t dw1;
    uint64_t dw2;
    uint64_t dw3;
};

// Decodes the R5 appdata arm of a response block. `app_words` are the four
// 8-byte words of the APP section exactly as the response block carries
// them; the door sets them into the SDK's own AppR5Section and reads every
// field back through its own getters. The arm is only meaningful when the
// return section says RB_ERROR with a QGI/QGI_NO_DIAG locator and the CB
// type is R5 (1) or R5-with-spawn (2) — that gating stays with the caller
// (the return section is already decoded Rust-side); this door only
// translates the four words into semantics.
int32_t flex_hal_r5_appdata_decode(const uint64_t* app_words,
                                   FlexHalR5AppData* out) {
    static_assert(sizeof(hal::snt1p0::AppR5Section) == 4 * 8,
                  "R5 appdata arm is four 8-byte words");
    try {
        if (app_words == nullptr || out == nullptr) {
            hal_record_error("flex_hal_r5_appdata_decode", "null app_words or out");
            return -1;
        }
        hal::snt1p0::AppR5Section s;
        s.SetDw0(app_words[0]);
        s.SetDw1(app_words[1]);
        s.SetDw2(app_words[2]);
        s.SetDw3(app_words[3]);
        out->cb_type = s.GetCbType();
        out->wcore_compl = s.GetWcoreCompl();
        out->worker_mask = s.GetWorkerMask();
        out->core_id = s.GetCoreId();
        out->completion_msg = s.GetCompletionMsg();
        out->error_type = s.GetErrorType();
        out->job_cc = s.GetJobCc();
        out->branch_target_data = s.GetBranchTargetData();
        out->beu_value = s.GetBeuValue();
        out->beu_cause = s.GetBeuCause();
        out->beu_accrued = s.GetBeuAccrued();
        out->interrupt_flag = s.GetInterruptFlag();
        out->mcause_code = s.GetMcauseCode();
        out->mip = s.GetMip();
        out->mepc = s.GetMepc();
        out->mtval = s.GetMtval();
        out->dw0 = s.GetDw0();
        out->dw1 = s.GetDw1();
        out->dw2 = s.GetDw2();
        out->dw3 = s.GetDw3();
        return 0;
    } catch (const std::exception& e) {
        hal_record_error("flex_hal_r5_appdata_decode", e.what());
        return -1;
    } catch (...) {
        hal_record_error("flex_hal_r5_appdata_decode", "unknown exception");
        return -1;
    }
}

}  // extern "C"
