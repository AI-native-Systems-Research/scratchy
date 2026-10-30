//! The R5/RDMA wire-door builders — flex-rs's semantic face over the
//! cxx-shim `hal_sbf.cpp` doors (the r5-wire port).
//!
//! THE LAW (the proprietary line — `cxx-shim/hal_sbf.cpp`'s own header
//! carries it in full): the hal SBF headers are OCO-marked trade-secret
//! source; their field maps, bit ranges, enum bit values, and packing logic
//! live ONLY on the C++ side of the shim boundary. This module carries OUR
//! vocabulary — modes by meaning, byte offsets, sids, masks — and hands
//! those scalars to the shim's C-ABI doors, which return finished section
//! words. No type, function, or test in this file knows where any bit sits.
//!
//! These doors round out the crate's control-block surface:
//! `control_block_wire.rs` hand-ports the graph-free compute/DMA/fill
//! shapes (migrating those behind the shim is a NAMED follow-up, not this
//! port); THIS module is the shim-backed half — the R5 control-block
//! section, the Rdma12 unicast-send DMO section, the Rdma10 wdone-wait DMI
//! section, and the R5 appdata arm of the response block — the sections the
//! tensor-parallel runtime needs. Nothing in this crate emits them yet:
//! Phase 2's builders (tp-clean) are the consumer, and they consume these
//! doors rather than encoding bits themselves.
//!
//! The raw C-ABI-shaped wrappers live in `crate::senlib_ffi` (the crate's
//! one unsafe home); everything here is safe and semantic.

use crate::senlib_ffi::{self, HalR5AppData};

/// The R5 control-block section's mode, by meaning. The mapping from meaning
/// to the shim door's own ordinal protocol is this enum's private business
/// (`shim_ordinal`); no caller ever handles a number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R5SectionMode {
    /// Skip the section entirely — the all-zero QGI encoding. A skip carries
    /// no core mask and no address; the shim door refuses one that does.
    Skip,
    /// Run the R5 core (`r5_only`).
    R5Only,
    /// Spawn R5 work (`spawn`).
    Spawn,
}

impl R5SectionMode {
    /// This mode's ordinal in the shim door's own C-ABI protocol — the
    /// contract documented at the door (`cxx-shim/hal_sbf.cpp`), not a
    /// header value re-stated here.
    fn shim_ordinal(self) -> u32 {
        match self {
            R5SectionMode::Skip => 0,
            R5SectionMode::R5Only => 1,
            R5SectionMode::Spawn => 2,
        }
    }
}

/// The R5 control-block section: one finished word, packed by the shim from
/// semantic scalars.
///
/// `core_mask` is the shim door's own core-selection mask — its semantic
/// (which bit selects which core) is documented at the door and stays on the
/// C++ side; Rust composes and carries it as an opaque value.
/// `segment`/`offset_bytes` name the section's device address; the flit
/// conversion and every field placement happen inside the shim.
///
/// # Errors
/// Returns `Err` with the shim's message on any refused input — the shim
/// never clamps and never invents a value (a skip that names a mask or an
/// address, a mask or segment past the section's fields, a misaligned or
/// out-of-segment offset all refuse by name).
pub fn r5_section(
    mode: R5SectionMode,
    core_mask: u64,
    segment: u64,
    offset_bytes: u64,
) -> Result<[u64; 1], String> {
    let word = senlib_ffi::hal_r5_section_encode(
        mode.shim_ordinal(),
        core_mask,
        segment,
        offset_bytes,
    )?;
    Ok([word])
}

/// The Rdma12 RDMA_SINGLECAST send DMO section: two finished words, packed
/// by the shim from semantic scalars — the vendor scheduler's ten-setter
/// unicast shape (`submitP2PRdma.cpp`'s unicast loop), with `ta` hardwired
/// and the endian field left at its default, both per that loop.
///
/// - `sid` is the wdone signal index (the signal is from OUR rank);
/// - `dest_tid` names the destination peer by BAT index (the vendor's
///   `dest_ranks[0]` — the tp=2 ring is unicast, one peer);
/// - `length_bytes` is this chunk's byte length;
/// - `(src_segment, src_offset_bytes)` is the source device address;
/// - `dest_remote_base_bytes` is the peer's remote base for this transfer,
///   compensation already applied by the caller — the −128 landing law is
///   the tp builders' door, not this port's;
/// - `first`/`last`/`sync`/`xsegv` are the caller's protocol bits, forwarded
///   verbatim (the laws that govern them — the F-send law above all — are
///   Rust's, at this door).
///
/// The shim composes the xseg word exactly as the vendor loop does; the
/// finished words are the section object's own bytes — the very bytes
/// `set_dmo` would copy into a control block.
///
/// # Errors
/// Returns `Err` with the shim's message on any refused input (a length the
/// wire cannot encode, a misaligned length or address, a sid/tid/segment
/// past its field — all by name, never clamped).
pub fn rdma12_unicast_send(
    sid: u64,
    dest_tid: u64,
    length_bytes: u64,
    src_segment: u64,
    src_offset_bytes: u64,
    dest_remote_base_bytes: u64,
    first: bool,
    last: bool,
    sync: bool,
    xsegv: bool,
) -> Result<[u64; 2], String> {
    senlib_ffi::hal_rdma12_unicast_send_encode(
        sid,
        dest_tid,
        length_bytes,
        src_segment,
        src_offset_bytes,
        dest_remote_base_bytes,
        first,
        last,
        sync,
        xsegv,
    )
}

/// The Rdma10 RDMA_WDONE_BARRIER wait DMI section: two finished words,
/// packed by the shim from the wdone sid list (the wait bitmask is composed
/// inside the shim — which sids to wait on is the caller's law).
///
/// # Errors
/// Returns `Err` with the shim's message on any refused input (an empty
/// list, a sid past the wait mask — by name).
pub fn rdma10_wdone_wait(sids: &[u64], sync: bool) -> Result<[u64; 2], String> {
    senlib_ffi::hal_rdma10_wdone_wait_encode(sids, sync)
}

/// The R5 appdata arm of a response block: the four APP section words,
/// exactly as the response block carries them, decoded through the SDK's
/// own getters into the semantic mirror. The arm is only meaningful when
/// the return section says RB_ERROR with a QGI/QGI_NO_DIAG locator and the
/// CB type is R5 — that gating stays with the caller
/// (`control_block_wire::ResponseBlockWire` already decodes the return
/// section); this door only translates the four words into semantics.
///
/// # Errors
/// Returns `Err` with the shim's message if the decode refused its input.
pub fn r5_appdata(app_words: &[u64; 4]) -> Result<HalR5AppData, String> {
    senlib_ffi::hal_r5_appdata_decode(app_words)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The semantic mirror is `#[repr(C)]`-identical to the shim's own
    /// `FlexHalR5AppData` — twenty 8-byte fields, no padding. This is the
    /// ABI the decode door writes through; if either side drifts, this
    /// assert fires before any decoded field can be misread.
    #[test]
    fn hal_r5_appdata_mirror_matches_the_shim_struct_layout() {
        assert_eq!(std::mem::size_of::<HalR5AppData>(), 20 * 8);
        assert_eq!(std::mem::align_of::<HalR5AppData>(), 8);
    }

    /// The mode enum is by meaning, and its ordinal mapping is total: every
    /// variant maps to a distinct shim ordinal. (The ordinals themselves are
    /// the shim's own protocol; this test pins only that the mapping is a
    /// bijection over the enum, so a future variant cannot silently reuse a
    /// retired one.)
    #[test]
    fn r5_section_mode_ordinals_are_distinct() {
        let ordinals = [
            R5SectionMode::Skip.shim_ordinal(),
            R5SectionMode::R5Only.shim_ordinal(),
            R5SectionMode::Spawn.shim_ordinal(),
        ];
        for i in 0..ordinals.len() {
            for j in (i + 1)..ordinals.len() {
                assert_ne!(ordinals[i], ordinals[j]);
            }
        }
    }
}
