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

use crate::senlib_ffi::{self, HalR5AppData, Rdma12UnicastSendParams};

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
    let word =
        senlib_ffi::hal_r5_section_encode(mode.shim_ordinal(), core_mask, segment, offset_bytes)?;
    Ok([word])
}

/// The Rdma12 RDMA_SINGLECAST send DMO section: two finished words, packed
/// by the shim from the door's argument bundle
/// ([`Rdma12UnicastSendParams`], defined beside the FFI door — one home,
/// both consumers) — the vendor scheduler's ten-setter unicast shape
/// (`submitP2PRdma.cpp`'s unicast loop), with `ta` hardwired and the endian
/// field left at its default, both per that loop.
///
/// - `sid` is the wdone signal index (the signal is from OUR rank);
/// - `dest_tids` names the destination peer(s) by BAT index — the vendor's
///   own plural shape (`dest_ranks`), carried plural so the multicast
///   refusal below can name the fan-out it refused rather than silently
///   dropping peers;
/// - `length_bytes` is this chunk's byte length;
/// - `(src_segment, src_offset_bytes)` is the source device address;
/// - `dest_remote_base_bytes` is the peer's remote base for this transfer,
///   compensation already applied by the caller — the −128 landing law is
///   the tp builders' door, not this port's;
/// - `first`/`last`/`sync` are the caller's protocol bits, forwarded
///   verbatim;
/// - `xsegv` must be false at this door — the T-send law below.
///
/// The laws are ours, at this door (the shim packs, it never decides), and
/// they are written against the bundle's fields:
///
/// - **The unicast law**: the tp=2 ring is unicast. An empty `dest_tids`
///   refuses (a send names the peer it lands on), and more than one
///   destination refuses as un-ported multicast fan-out — the vendor's
///   multicast shape (`Rdma12` RDMA_MULTICAST behind per-peer XSEG CBs)
///   gets no encoder here, per the spec's A.3.
/// - **The F-send law** (fixture spec D1; failure-to-type catalog 6.2/6.4):
///   `xsegv=true` refuses. It asks the wire to deliver the CB's xseg word
///   into the peer's XSEG[SID] table as a T-send preamble — measured on
///   card as a preamble overwrite that fires +1 with no wait pending
///   (E8b/H10). The receiver host-programs the table instead:
///   [`senlib_ffi::write_xseg`] with the same composed word this door's
///   encoder produces.
///
/// The numeric capacity caps (a length the wire cannot encode, a
/// misaligned length or address, a sid/tid/segment past its field) are
/// refused at the shim door, by name, never clamped — their bounds live
/// with the field widths on the C++ side and are not re-stated here.
///
/// The shim composes the xseg word exactly as the vendor loop does; the
/// finished words are the section object's own bytes — the very bytes
/// `set_dmo` would copy into a control block.
///
/// # Errors
/// Returns `Err` naming the refused law (empty destination, multicast
/// fan-out, T-send preamble), or the shim's message on any refused input.
pub fn rdma12_unicast_send(params: &Rdma12UnicastSendParams<'_>) -> Result<[u64; 2], String> {
    if params.dest_tids.is_empty() {
        return Err(
            "the RDMA send has no destination: dest_tids is empty — a send names the peer it lands on by BAT index"
                .to_string(),
        );
    }
    if params.dest_tids.len() != 1 {
        return Err(format!(
            "multicast fan-out is un-ported: dest_tids names {} peers — the tp=2 ring is unicast, one destination per send",
            params.dest_tids.len()
        ));
    }
    if params.xsegv {
        return Err(
            "the T-send preamble is un-ported (the F-send law): xsegv=true asks the wire to deliver the CB's xseg word into the peer's XSEG table ahead of the payload — measured as a preamble overwrite that fires +1 with no wait pending (E8b/H10). The receiver host-programs the table instead: senlib_ffi::write_xseg with the same composed word the send encoder produces"
                .to_string(),
        );
    }
    senlib_ffi::hal_rdma12_unicast_send_encode(params)
}

/// The Rdma10 RDMA_WDONE_BARRIER wait DMI section: two finished words,
/// packed by the shim from the wdone sid list (the wait bitmask is composed
/// inside the shim — which sids to wait on is the caller's law).
///
/// The duplicate law is ours, at this door: the wait list is a SET of sids,
/// not a sequence — a repeated sid refuses rather than counting twice (the
/// vendor's own `submitP2PRdmaWait` throws `RDMASrcRankAlreadySet` on a
/// duplicate rank; we refuse by name instead of panicking past the wire).
/// The empty-list and per-sid range caps are refused at the shim door, by
/// name — their bounds live with the mask width on the C++ side and are
/// not re-stated here.
///
/// # Errors
/// Returns `Err` naming the duplicate sid, or the shim's message on any
/// refused input (an empty list, a sid past the wait mask — by name).
pub fn rdma10_wdone_wait(sids: &[u64], sync: bool) -> Result<[u64; 2], String> {
    for (i, &sid) in sids.iter().enumerate() {
        if sids[..i].contains(&sid) {
            return Err(format!(
                "duplicate sid {sid} in the wdone wait list — the wait is a set of sids, not a sequence"
            ));
        }
    }
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

    // The refusal doors (B.2's laws). Each test drives ONLY the refusal
    // path — the law fires before any shim call, so these are pure Rust
    // and assert our own named messages, never golden bytes.

    /// The send bundle the refusal tests drive: one call-site spelling of
    /// the door's argument bundle, with the field under test varied per
    /// test. Every other field's value is inert on the refusal paths (the
    /// laws fire before any shim call reads them).
    fn send_params(dest_tids: &[u64], xsegv: bool) -> Rdma12UnicastSendParams<'_> {
        Rdma12UnicastSendParams {
            sid: 0,
            dest_tids,
            length_bytes: 128,
            src_segment: 0,
            src_offset_bytes: 0,
            dest_remote_base_bytes: 0,
            first: true,
            last: true,
            sync: false,
            xsegv,
        }
    }

    /// A send with no destination refuses by name — a send names the peer
    /// it lands on by BAT index.
    #[test]
    fn rdma12_send_refuses_an_empty_destination_list() {
        let err = rdma12_unicast_send(&send_params(&[], false))
            .expect_err("an empty dest_tids must refuse");
        assert!(
            err.contains("no destination"),
            "the refusal must name the law, got: {err}"
        );
    }

    /// More than one destination refuses as un-ported multicast fan-out —
    /// the tp=2 ring is unicast, and the vendor's multicast shape gets no
    /// encoder here.
    #[test]
    fn rdma12_send_refuses_multicast_fan_out() {
        let err = rdma12_unicast_send(&send_params(&[3, 7], false))
            .expect_err("a two-peer dest_tids must refuse");
        assert!(
            err.contains("multicast fan-out is un-ported"),
            "the refusal must name the law, got: {err}"
        );
        assert!(
            err.contains("2 peers"),
            "the refusal must name the fan-out it refused, got: {err}"
        );
    }

    /// The F-send law: xsegv=true refuses. The T-send preamble is un-ported
    /// (measured as a preamble overwrite that fires +1 with no wait
    /// pending); the receiver host-programs the table via write_xseg.
    #[test]
    fn rdma12_send_refuses_the_t_send_preamble() {
        let err =
            rdma12_unicast_send(&send_params(&[3], true)).expect_err("xsegv=true must refuse");
        assert!(
            err.contains("T-send preamble is un-ported"),
            "the refusal must name the law, got: {err}"
        );
        assert!(
            err.contains("write_xseg"),
            "the refusal must name the replacement door, got: {err}"
        );
    }

    /// The wait list is a set of sids, not a sequence — a repeated sid
    /// refuses by name rather than counting twice.
    #[test]
    fn rdma10_wait_refuses_a_duplicate_sid() {
        let err = rdma10_wdone_wait(&[5, 1, 5], false).expect_err("a duplicate sid must refuse");
        assert!(
            err.contains("duplicate sid 5"),
            "the refusal must name the duplicate, got: {err}"
        );
    }
}
