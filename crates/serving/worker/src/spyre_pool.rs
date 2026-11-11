// SPDX-License-Identifier: Apache-2.0
//! THE CARD'S SESSIONS — recognising a SuperDSC bundle, picking a prefill rung, and the per-process
//! print flags the launch path reads.
//!
//! The KV pool itself has no host plumbing left here: it is a placed tensor of every bundle
//! (`PagedStep::card_placements`), the scheduler's block table names its pages, and each launch
//! stages the paged step's tables from that table (`scratchy_target_spyre::paged_step`).

#[cfg(feature = "spyre-hw")]
use crate::spyre_types::SuperDscSession;
#[cfg(feature = "spyre-hw")]
use scratchy_target_spyre::manifest::SengraphBundle;

/// EVERY PRINT-ONLY FLAG THIS FILE READS ON A PER-STEP OR PER-LAUNCH PATH, RESOLVED ONCE PER PROCESS.
///
/// ⛔⛔⛔ `std::env::var_os` LOCKS THE ENVIRONMENT AND WALKS IT, and these sit on the launch path — per
/// prefill chunk and per decode step. Host time paid every token to answer a question whose answer
/// cannot change: nothing mutates the environment mid-run.
///
/// ⭐ `superdsc_exec::Diag` ALREADY DID EXACTLY THIS (one `OnceLock`, ten flags). An instrument may
/// print; it must not cost.
#[cfg(feature = "spyre-hw")]
pub(crate) struct Flags {
    pub(crate) phase_time: bool,
}

#[cfg(feature = "spyre-hw")]
pub(crate) fn flags() -> &'static Flags {
    static F: std::sync::OnceLock<Flags> = std::sync::OnceLock::new();
    F.get_or_init(|| {
        let on = |k: &str| std::env::var_os(k).is_some();
        Flags {
            phase_time: on("SCRATCHY_SDSC_PHASE_TIME"),
        }
    })
}

/// Index of the SMALLEST prefill rung that can hold `real` query rows per request and `live`
/// requests. `rungs` is ascending by `(width, seqs)`, and the caller caps its chunk at the top
/// rung's width, so this only returns `None` for an EMPTY ladder. A packed rung holds any live
/// count at or below its `seqs` (an empty window is staged fully masked, page-0), so the first
/// fit over the ascending list is the narrowest width and — at a tie — the SOLO rung, keeping a
/// one-request launch byte-identical to the no-packing build.
#[cfg(feature = "spyre-hw")]
pub(crate) fn prefill_rung_for(
    rungs: &[(usize, usize, SuperDscSession)],
    real: usize,
    live: usize,
) -> Option<usize> {
    rungs
        .iter()
        .position(|(w, seqs, _)| *w >= real && *seqs >= live)
        .or_else(|| rungs.len().checked_sub(1))
}

/// The decode-slot sentinel prefix the SuperDSC codegen stamps into
/// `group.decode.graph_json` in place of a real sengraph JSON. The bytes after the
/// prefix are the dxp-bundle fingerprint `fp` (the `bundle_code` registry key).
#[cfg(feature = "spyre-hw")]
pub(crate) const SUPERDSC_SENTINEL: &str = "SUPERDSC_BUNDLE:";

/// True iff the baked bundle is the SUPERDSC dxp bundle. The SuperDSC emitter does
/// NOT produce a DeepTools-compilable sengraph JSON; instead the codegen stamps the
/// single group's decode-slot `graph_json` with the `SUPERDSC_BUNDLE:<fp>` sentinel
/// (the dxp bundle is a DIRECTORY of per-supernode dxp blobs + plan, recovered at
/// load via `bundle_code::bundle(fp)`).
#[cfg(feature = "spyre-hw")]
pub(crate) fn is_superdsc_bundle(bundle: &SengraphBundle) -> bool {
    bundle.groups.len() == 1
        && bundle
            .groups
            .first()
            .is_some_and(|g| g.decode.graph_json.starts_with(SUPERDSC_SENTINEL))
}
