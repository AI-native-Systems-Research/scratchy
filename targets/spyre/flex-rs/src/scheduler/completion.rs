//! The CB completion decision and the two capabilities that hang off it.
//!
//! ⛔ THIS IS A SEPARATE MODULE ON PURPOSE. [`CbRetired`] and
//! [`QuarantinedBuf`] have private fields, and `scheduler.rs` is one enormous
//! module — a private field is visible to every line of the module that
//! declares it, so a capability declared next to its users guards nothing. From
//! outside, `CbRetired(())` and `parked.0` do not compile, which is what makes
//! "put the buffer back without establishing that the card is done with its CB"
//! and "take a quarantined buffer back out" unwritable rather than merely
//! discouraged.

use super::{PendingRequest, PendingRequestState};
use crate::control_block_wire::{ResponseBlockWire, ResponseStatus};
use crate::stream::SchedulerError;

#[cfg(kani)]
mod kani_proofs;

/// The hardware-side failure of one CB, as seen by `onCbComplete`. Typed
/// (rather than a bare diagnostic string) because a timed-out CB and a CB the
/// hardware rejected demand different handling from the DMA completion path: a
/// rejected CB is RETIRED — its buffers are free to reuse — while a timed-out
/// CB may still be sitting in the hardware queue, so recycling (and thereby
/// remapping) its IOMMU-mapped buffer hands the card a dangling IOVA.
///
/// `timed_out` is PRIVATE: the DMA path does not read it, it asks
/// [`CbRetired::from_completion`] for the capability to reuse a buffer, and
/// only that function inspects this flag. A `bool` a caller must remember to
/// check is the shape of the original bug.
#[derive(Debug)]
pub struct CbCompleteError {
    timed_out: bool,
    detail: String,
}

impl std::fmt::Display for CbCompleteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for CbCompleteError {}

/// Which of the two things a completion means for the CB's BUFFER — the only
/// question the DMA path actually has. Produced by an EXHAUSTIVE match on
/// [`PendingRequestState`], so a state added to that enum cannot compile until
/// someone decides which of these it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HardwareOwnership {
    /// The card is done with this CB: its buffer may be remapped and reused.
    Released,
    /// The card may still be holding this CB, so its buffer's IOVA may still
    /// be mastered: the buffer must never be remapped or freed.
    MayStillHoldCb,
}

/// The capability to return a DMA buffer to the reuse pool: proof that the
/// card is finished with the control block that names it.
///
/// ⛔ The field is private and there are exactly two constructors, so
/// [`DmaBufPool::enqueue`] cannot be called at all without answering "is the
/// card done with this CB?". That is the class this kills: the original defect
/// was a buffer put back into the pool — and therefore remapped, i.e.
/// UNMAPPED — while its CB was still in the hardware queue, which the card
/// reports as a fatal `RAS::PCI::BusFence`. A comment or a remembered `if`
/// cannot enforce it; a missing token is a compile error.
///
/// Modelled on this file's [`TagLease`], the other capability here: a response
/// tag cannot be recycled while a CB using it may be in flight, because the
/// lease lives in the slot the tag names.
#[derive(Debug)]
pub struct CbRetired(());

impl CbRetired {
    /// From a completion result. `None` when the CB TIMED OUT — the one case
    /// where the hardware may still hold it. A rejected or cancelled CB IS
    /// retired, so it yields the token: its buffer is free to reuse.
    ///
    /// This is the ONLY place a completion result is inspected for this
    /// question. It reads the typed [`CbCompleteError`]; the diagnostic string
    /// is not an API.
    pub fn from_completion(result: &Result<(), SchedulerError>) -> Option<Self> {
        match result {
            Ok(()) => Some(Self(())),
            Err(e) => match e.downcast_ref::<CbCompleteError>() {
                Some(cb_err) if cb_err.timed_out => None,
                // A non-completion error (queue capacity, a failed submission,
                // a panic caught in the callback wrapper) means the CB never
                // reached, or never left, the hardware queue as a live entry.
                _ => Some(Self(())),
            },
        }
    }

    /// For a buffer that was never handed to hardware, so no CB can name it:
    /// a freshly allocated `DmaBuf`. Named rather than defaulted, so it shows
    /// up in a grep for every place the check is bypassed.
    pub fn never_submitted() -> Self {
        Self(())
    }
}

/// What one CB's completion was. `Retired` CARRIES the reuse capability, so
/// the decision and the permission are one value: a caller cannot obtain the
/// permission without going through the decision, and cannot make the decision
/// without receiving the permission or an error.
#[derive(Debug)]
pub(super) enum CbOutcome {
    Retired(CbRetired),
    Failed(CbCompleteError),
}

impl CbOutcome {
    /// The error, for the paths that only forward a diagnostic.
    pub(super) fn failure(self) -> Option<CbCompleteError> {
        match self {
            Self::Retired(_) => None,
            Self::Failed(e) => Some(e),
        }
    }
}

/// The return section a completion decision reads: the hardware's own, or —
/// for a slot no response ever reached — [`ResponseBlockWire::ZERO`], the
/// zero-initialised value the C++'s `PendingRequest` always holds.
///
/// ⛔ THIS TYPE EXISTS BECAUSE AN `Option` HERE IS A LOADED GUN. The C++'s
/// `response_block` is a VALUE, so its three-arm disjunction always evaluates
/// every arm; the port made it `Option` and then wrote
/// `let rb = pr.response_block?;`, which silently deleted the
/// `state == TIMED_OUT` arm for exactly the slots that arm was written for.
/// There is no absent case to short-circuit on here, so that mistake is not
/// expressible: absence is REPRESENTED (as the zero response), never a reason
/// to skip a check.
#[derive(Debug, Clone, Copy)]
struct EffectiveResponse(ResponseBlockWire);

impl EffectiveResponse {
    /// The response a decision must read: the hardware's, or the zero value for
    /// a slot that never received one.
    fn received_or_zero(rb: Option<ResponseBlockWire>) -> Self {
        Self(rb.unwrap_or(ResponseBlockWire::ZERO))
    }

    fn of(pr: &PendingRequest) -> Self {
        Self::received_or_zero(pr.response_block)
    }
}

/// What the C++'s three-arm disjunction says about ONE CB, as three cases
/// rather than a bool plus a remembered flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CbVerdict {
    /// All three arms false: the card answered and reported no fault.
    Clean,
    /// The card answered and the response says the CB failed (cancelled, or a
    /// status other than `RB_GOOD`). It is RETIRED — its buffer is free.
    Rejected,
    /// No response arrived within `FLEX_RESPONSE_WORKER_TIMEOUT_SECONDS`. The
    /// card may still be holding the CB, so its buffer's IOVA may still be
    /// mastered.
    TimedOut,
}

/// The decision itself: the port of
/// ```text
/// if(current_state == PendingRequestState::TIMED_OUT || rc.GetCancel() == 1 ||
///    rc.GetStatus() != RBStatusTypeEnum::RB_GOOD)
/// ```
/// and NOTHING else — no allocation, no formatting, no logging. Kept separate
/// from `eval_cb_complete_status` for two reasons: a pure function of its two
/// arguments is provable over the WHOLE input space (see the `kani_proofs`
/// child module — 7 states x every 64-bit return word), and a diagnostic
/// cannot influence a verdict that is built after it.
///
/// Neither argument can be missing or unclassified: the state arm is an
/// exhaustive match (a new [`PendingRequestState`] variant is a compile error
/// here until someone decides what it means for the card's grip on the CB),
/// and [`EffectiveResponse`] has no absent case to short-circuit on.
fn cb_verdict(state: PendingRequestState, rb: EffectiveResponse) -> CbVerdict {
    let ownership = match state {
        PendingRequestState::TimedOut => HardwareOwnership::MayStillHoldCb,
        PendingRequestState::Invalid
        | PendingRequestState::Available
        | PendingRequestState::Issued
        | PendingRequestState::Succeeded
        | PendingRequestState::Skipped
        | PendingRequestState::Failed => HardwareOwnership::Released,
    };
    match ownership {
        HardwareOwnership::MayStillHoldCb => CbVerdict::TimedOut,
        HardwareOwnership::Released => {
            if rb.0.cancelled() || !matches!(rb.0.status(), ResponseStatus::Good) {
                CbVerdict::Rejected
            } else {
                CbVerdict::Clean
            }
        }
    }
}

/// Port of the response-block error check inside
/// `PfRuntimeScheduler::onCbComplete` (`pf_runtime_scheduler_utils.cpp:437-447`):
///
/// ```text
/// if(current_state == PendingRequestState::TIMED_OUT || rc.GetCancel() == 1 ||
///    rc.GetStatus() != RBStatusTypeEnum::RB_GOOD)
/// ```
///
/// Every arm is evaluated, and none of them can be skipped by a missing value:
/// the state arm is an exhaustive match producing [`HardwareOwnership`] (a new
/// `PendingRequestState` variant is a compile error until it is classified),
/// and the response arms read an [`EffectiveResponse`], which has no absent
/// case. A previous version began `let rb = pr.response_block?;`, which
/// short-circuited every reaped slot to "no error": a timed-out DMA completed
/// its user callback with `Ok(())` and copied back a shadow buffer the
/// hardware had never filled.
///
/// Returns [`CbOutcome::Retired`] — which CARRIES the buffer-reuse capability
/// — only when all three arms are false, and otherwise the typed error holding
/// the `make_hw_error()` diagnostic the C++ builds.
pub(super) fn eval_cb_complete_status(pr: &PendingRequest) -> CbOutcome {
    let timed_out = match cb_verdict(pr.state, EffectiveResponse::of(pr)) {
        CbVerdict::Clean => return CbOutcome::Retired(CbRetired(())),
        CbVerdict::Rejected => false,
        CbVerdict::TimedOut => true,
    };
    let detail = match pr.response_block {
        None => format!(
            "CB state={:?} (no response block received) node_name={:?}",
            pr.state, pr.node_name,
        ),
        Some(rb) => format!(
            "CB tag={:?} state={:?} cancel={} status={:?} locator={:#x} mark={} edep={} flr={} node_name={:?}{}",
            rb.tag(),
            pr.state,
            rb.cancelled(),
            rb.status(),
            rb.locator_bytes(),
            rb.mark(),
            rb.edep(),
            rb.flr(),
            pr.node_name,
            qgi_detail(rb),
        ),
    };
    CbOutcome::Failed(CbCompleteError { timed_out, detail })
}

/// ⭐ THE FAULTING DEVICE ADDRESS, APPENDED TO THE DIAGNOSTIC — empty for every response that does not
/// carry one, so a non-compute failure reads exactly as it did before.
///
/// ⛔ THIS EXISTS BECAUSE `locator=0x2` ON ITS OWN COST A SESSION. A compute-side fault reported only
/// as `status=Error locator=0x2` says nothing about WHERE it faulted, and the response block was
/// carrying the address the whole time in its app section (see
/// [`ResponseBlockWire::app_qgi`]) — we decoded the first 8 bytes and threw the rest away. An
/// unmapped-address case (`riu_unmp_err`/`prep_unmp_err`) means the access fell outside the
/// `paddr..paddr+length` some translation declared, which is a statement about a SEGMENT'S EXTENT and
/// is not otherwise deducible from the failure at all.
fn qgi_detail(rb: ResponseBlockWire) -> String {
    let Some(qgi) = rb.app_qgi() else {
        return String::new();
    };
    let mut out = String::new();
    if qgi.is_hmi() {
        out.push_str(&format!(
            " HMI={} addr={:#x} ({} flits) syndrome={:#x} tag={:#x} snid={:#x}",
            if qgi.hmi_is_store() { "STORE" } else { "FETCH" },
            qgi.hmi_address_bytes(),
            qgi.hmi_address_flits(),
            qgi.hmi_syndrome(),
            qgi.hmi_tag(),
            qgi.hmi_snid(),
        ));
    }
    if qgi.is_qgi() {
        let cases: Vec<String> = qgi.qgi_error_cases().map(|c| format!("{c:?}")).collect();
        // ⭐ `job_count` IS THE HARDWARE'S OWN STATEMENT OF HOW FAR THE QG CHAIN WALK GOT, and it was
        // being decoded and thrown away. Every Prep case in the table is about parsing the job-header
        // chain (`PrepZeroFlitCnt` = a header whose flit count read as zero, `PrepSwVer` = one whose
        // SW version is not dip's `0xdd`), and a faulting ADDRESS alone cannot say whether the walk
        // failed on the FIRST header or the Nth: the same address is reached by "started here" and by
        // "walked here". `job_count == 0` means the device never validated a single job — which is a
        // statement about the bytes AT the bootstrap, not about the chain — while any non-zero value
        // means the walk consumed that many headers first and the defect is in the count the previous
        // header declared. That distinction is the whole difference between suspecting the upload and
        // suspecting the emission, and it costs one field.
        out.push_str(&format!(
            " QGI addr={:#x} ({} flits) syndrome={:#x} job_count={} cases=[{}]",
            qgi.qgi_address_bytes(),
            qgi.qgi_address_flits(),
            qgi.qgi_syndrome(),
            qgi.job_count(),
            cases.join(","),
        ));
        if qgi.qgi_error_cases().any(|c| c.is_unmapped_address()) {
            out.push_str(
                " — UNMAPPED ADDRESS: this access fell outside the paddr..paddr+length a \
                 translation declared, so suspect a segment's declared EXTENT, not its base",
            );
        }
    }
    if out.is_empty() && qgi.beat0_byte8() != 0 {
        out.push_str(&format!(
            " (app section is not a QGI/HMI record: beat0_byte8={:#x}, documented unused)",
            qgi.beat0_byte8()
        ));
    }
    out
}

/// A DMA buffer parked because the control block naming it may still be live
/// in the hardware queue.
///
/// ⛔ There is no accessor, no `Deref`, and no destructuring from outside this
/// module: a `DmaBuf` goes IN and nothing comes out. The mapping and the shadow
/// allocation it owns therefore stay alive for the process's lifetime, which is
/// the entire point — the alternative is handing the card a dangling IOVA. The
/// cost is bounded: one buffer per timed-out chunk.
pub(super) struct QuarantinedBuf(#[allow(dead_code)] super::DmaBuf);

impl QuarantinedBuf {
    pub(super) fn park(buf: super::DmaBuf) -> Self {
        Self(buf)
    }
}

#[cfg(test)]
mod cb_complete_status_tests {
    use super::*;
    use crate::control_block_wire::ResponseBlockWire;
    use crate::scheduler::{PipelineId, QueuingMode, mock_error};

    /// One `pending_requests_` slot as the completion path sees it. Only the
    /// four fields `eval_cb_complete_status` reads carry meaning here.
    fn slot(
        state: PendingRequestState,
        response_block: Option<ResponseBlockWire>,
    ) -> PendingRequest {
        PendingRequest {
            id: 7,
            batch_id: 0,
            cb_idx: 0,
            pipeline: Some(PipelineId::AsyncDmaI),
            queuing_mode: QueuingMode::Regular,
            state,
            issue_time_point: None,
            launch_time_point: None,
            completion_time_point: None,
            control_block: None,
            response_block,
            node_name: "dma-chunk".to_string(),
        }
    }

    /// An all-zero return section decodes to `status=Good, cancel=0`, i.e. the
    /// clean completion the hardware writes.
    fn good_rb() -> ResponseBlockWire {
        ResponseBlockWire::from_bytes(&[0; crate::control_block_wire::RB_NUM_BYTES])
    }

    /// `status` is bits 6:5 of the return word; `0b11` is `RB_ERROR`.
    fn rejected_rb() -> ResponseBlockWire {
        let mut bytes = [0u8; crate::control_block_wire::RB_NUM_BYTES];
        bytes[0] = 0b11 << 5;
        ResponseBlockWire::from_bytes(&bytes)
    }

    /// THE REGRESSION. A slot reaped by `check_pending_job_timeouts` has
    /// `response_block == None` precisely because no response ever arrived, so
    /// a function that starts `let rb = pr.response_block?` reports "no error"
    /// for it — which completed the user's DMA callback with `Ok(())` and
    /// copied back a shadow buffer the hardware had never filled. The state
    /// check is FIRST in the C++'s disjunction and does not depend on a
    /// response block existing.
    #[test]
    fn a_reaped_slot_with_no_response_block_is_a_timed_out_error() {
        let err = eval_cb_complete_status(&slot(PendingRequestState::TimedOut, None))
            .failure()
            .expect("a timed-out CB with no response is an ERROR, not a clean completion");
        assert!(err.timed_out, "and it must be classified as a TIMEOUT");
        assert!(
            err.to_string().contains("no response block received"),
            "the diagnostic must say why there is no response block: {err}"
        );
    }

    /// The only case that may report success.
    #[test]
    fn a_good_response_on_an_issued_slot_is_not_an_error() {
        assert!(
            matches!(
                eval_cb_complete_status(&slot(PendingRequestState::Issued, Some(good_rb()))),
                CbOutcome::Retired(_)
            ),
            "a GOOD, uncancelled response block on an ISSUED slot is a clean completion, and \
             carries the capability to reuse the buffer"
        );
    }

    /// A CB the hardware REJECTED is retired: its buffer is free to reuse, so
    /// it must not be classified as timed out (which would quarantine the
    /// buffer for the process's lifetime).
    #[test]
    fn a_rejected_cb_is_an_error_that_is_not_timed_out() {
        let err = eval_cb_complete_status(&slot(PendingRequestState::Issued, Some(rejected_rb())))
            .failure()
            .expect("RB_ERROR status is a hardware error");
        assert!(
            !err.timed_out,
            "a rejected CB is retired, not in flight — its buffer must be recycled, not quarantined"
        );
    }

    /// A late response does not un-time-out the slot: the buffer's IOVA was
    /// live in a CB the hardware held past the deadline, so this still reports
    /// a timeout to the DMA path.
    #[test]
    fn a_timed_out_slot_with_a_late_good_response_is_still_a_timeout() {
        let err = eval_cb_complete_status(&slot(PendingRequestState::TimedOut, Some(good_rb())))
            .failure()
            .expect("the state check alone makes this an error");
        assert!(err.timed_out);
    }

    /// The capability law, at the boundary the DMA path actually uses: only a
    /// CB the card has finished with yields the token that lets its buffer back
    /// into the pool. A rejected CB is finished; a timed-out one is not.
    #[test]
    fn only_a_finished_cb_yields_the_buffer_reuse_capability() {
        let timed_out = eval_cb_complete_status(&slot(PendingRequestState::TimedOut, None))
            .failure()
            .expect("timed out");
        assert!(
            CbRetired::from_completion(&Err(Box::new(timed_out))).is_none(),
            "a timed-out CB must NOT yield the token: the card may still hold it"
        );

        let rejected =
            eval_cb_complete_status(&slot(PendingRequestState::Issued, Some(rejected_rb())))
                .failure()
                .expect("rejected");
        assert!(
            CbRetired::from_completion(&Err(Box::new(rejected))).is_some(),
            "a REJECTED CB is retired — its buffer is free, so it must yield the token"
        );

        assert!(
            CbRetired::from_completion(&Ok(())).is_some(),
            "a clean completion yields the token"
        );
        assert!(
            CbRetired::from_completion(&Err(mock_error("queue capacity exceeded"))).is_some(),
            "a submission that never became a live CB yields the token"
        );
    }
}
