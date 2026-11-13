//! Machine-checked proofs for the CB completion decision — the function that
//! decides whether the card is finished with a control block, and therefore
//! whether the IOMMU-mapped buffer that CB names may be recycled.
//!
//! The types in the parent module make the mistakes UNWRITABLE; these prove the
//! decision those types carry is the RIGHT one, over the whole input space
//! rather than at the handful of points a unit test can name: 7 states × every
//! 64-bit return word × present/absent response block.
//!
//! The defect they exist for: the decision short-circuited on a MISSING
//! response block (`let rb = pr.response_block?;`), which silently deleted the
//! `state == TIMED_OUT` arm of the C++'s three-way disjunction
//! (`pf_runtime_scheduler_utils.cpp:437-447`) for exactly the case that arm was
//! written for — a slot the timeout scan reaped, which has no response block
//! precisely BECAUSE no response ever arrived. A timed-out DMA then reported
//! `Ok(())` and its buffer went back into the reuse pool under a CB the card may
//! still have been holding.
//!
//! ⭐ They target [`cb_verdict`], the allocation-free core, NOT
//! `eval_cb_complete_status`: the `format!` of the diagnostic drags the whole
//! `core::fmt` machinery into CBMC (a harness against it ran >15 min without a
//! verdict). Splitting the decision from its diagnostic is what makes the law
//! provable — and the wrapper cannot alter a verdict it is handed.

use super::{CbCompleteError, CbRetired, CbVerdict, EffectiveResponse, cb_verdict};
use crate::control_block_wire::{RB_NUM_BYTES, ResponseBlockWire, ResponseStatus};
use crate::scheduler::PendingRequestState;

/// A response block built from a nondeterministic return word. Only bytes 0..8
/// (`RET_SECTION`) are decoded by `ResponseBlockWire`, so nondet over that word
/// is nondet over everything any completion decision can read.
fn nondet_rb(word: u64) -> ResponseBlockWire {
    let mut bytes = [0u8; RB_NUM_BYTES];
    bytes[0..8].copy_from_slice(&word.to_le_bytes());
    ResponseBlockWire::from_bytes(&bytes)
}

/// THE LAW THE BUG BROKE. A CB the timeout scan reaped is never a clean
/// completion — for ANY response block, including the absent one that is the
/// only way a reaped slot can actually look.
#[kani::proof]
fn a_timed_out_cb_is_never_a_clean_completion() {
    let present: bool = kani::any();
    let word: u64 = kani::any();
    let rb = EffectiveResponse::received_or_zero(present.then(|| nondet_rb(word)));
    assert!(
        cb_verdict(PendingRequestState::TimedOut, rb) == CbVerdict::TimedOut,
        "a timed-out CB is a hardware error, never a clean completion"
    );
}

/// The decision is EXACTLY the C++'s disjunction, over every state and every
/// return word — no arm may be dropped, and none invented. An absent response
/// block is modelled the way the C++ models it: the zero-initialised value its
/// `PendingRequest` always holds (`status = RB_GOOD`, `cancel = 0`), so absence
/// contributes nothing to the verdict and cannot ERASE the state arm.
#[kani::proof]
fn the_decision_is_exactly_the_cxx_disjunction() {
    let state = PendingRequestState::from_u8(kani::any());
    let present: bool = kani::any();
    let word: u64 = kani::any();

    // `auto rc = pr.response_block.ret();` — on a slot that never received a
    // response, that reads the zero value.
    let effective = if present {
        nondet_rb(word)
    } else {
        ResponseBlockWire::ZERO
    };
    let cxx_bad = state == PendingRequestState::TimedOut
        || effective.cancelled()
        || !matches!(effective.status(), ResponseStatus::Good);

    let verdict = cb_verdict(
        state,
        EffectiveResponse::received_or_zero(present.then(|| nondet_rb(word))),
    );
    assert_eq!(
        verdict != CbVerdict::Clean,
        cxx_bad,
        "the Rust verdict diverged from the C++ predicate it ports"
    );
}

/// And the verdict's THREE cases are not two: a CB the card rejected must be
/// distinguishable from one it never answered, because only the second forbids
/// reusing the buffer. `Rejected` iff the response itself is bad and the state
/// is not TIMED_OUT.
#[kani::proof]
fn rejected_and_timed_out_are_never_confused() {
    let state = PendingRequestState::from_u8(kani::any());
    let word: u64 = kani::any();
    let rb = nondet_rb(word);
    let verdict = cb_verdict(state, EffectiveResponse::received_or_zero(Some(rb)));

    let response_is_bad = rb.cancelled() || !matches!(rb.status(), ResponseStatus::Good);
    assert_eq!(
        verdict == CbVerdict::TimedOut,
        state == PendingRequestState::TimedOut,
        "TimedOut must be exactly the state arm — the one case where the card may still hold the CB"
    );
    assert_eq!(
        verdict == CbVerdict::Rejected,
        state != PendingRequestState::TimedOut && response_is_bad,
        "Rejected must be exactly a bad response on a CB the card has finished with"
    );
}

/// The capability law: the permission to put a buffer back in the pool is
/// withheld for exactly the timed-out CBs. `CbRetired::from_completion` is the
/// only path from a completion result to that permission, and `DmaBufPool::
/// enqueue` cannot be called without it.
#[kani::proof]
fn only_a_finished_cb_yields_the_buffer_reuse_capability() {
    let timed_out: bool = kani::any();
    // Built directly rather than through `eval_cb_complete_status`, whose
    // `format!` is what makes that function unprovable; this is the same value
    // it produces, with an empty (non-allocating) diagnostic.
    let err = CbCompleteError {
        timed_out,
        detail: String::new(),
    };
    let result: Result<(), crate::stream::SchedulerError> = Err(Box::new(err));
    assert_eq!(
        CbRetired::from_completion(&result).is_some(),
        !timed_out,
        "the buffer-reuse capability must be withheld for exactly the timed-out CBs"
    );
}

/// A clean completion always yields it — the pool must not starve on success.
#[kani::proof]
fn a_clean_completion_yields_the_capability() {
    assert!(CbRetired::from_completion(&Ok(())).is_some());
}
