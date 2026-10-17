//! The cross-stream hazard tracker — the port of the vendor's
//! `track_hazards`/`skip_hazard` surface (#546, #500 TODO 7).
//!
//! Vendor surface (pod SDK headers, which postdate the local flex repo —
//! the open repo predates this subsystem entirely):
//!
//! - `RuntimeStream(..., bool track_hazards = false)`
//!   (`runtime_stream.hpp:88`): "If true, register this stream with the
//!   scheduler's hazard tracker so it participates in cross-stream
//!   RAW/WAR/WAW edge insertion. Unregistered streams are treated as
//!   user-owned and never edged."
//! - `RuntimeContext::createStream(..., bool track_hazards = false)`
//!   (`runtime_context.hpp:70-78`): "the scheduler mints cross-stream
//!   ordering edges for it; if false (default) the stream is user-owned and
//!   untracked. Same-stream ordering is unaffected."
//! - `RuntimeOperationParams::skip_hazard = false`
//!   (`runtime_submission_params.hpp:88`): "If true, hide this op from the
//!   cross-stream hazard tracker" — per `createDmaParams`'s parameter doc
//!   (`:662-664`), a skipped op is "neither checked against nor recorded as
//!   a producer/consumer". In the shipped surface only the DMA path threads
//!   it through (both `DmaParams` constructors take `skip_hazard_in =
//!   false`, and `createDmaParams` exposes it); this port mirrors that:
//!   [`DmaShaping`](crate::dma::DmaShaping) carries it, every other kind
//!   submits the base default `false`.
//! - `createEvent()` (`runtime_submission_params.hpp:805-812`): the event
//!   factory "used by the hazard tracker (producer completion events)" —
//!   the tracker's edge primitive is the very [`Event`] ported in
//!   `crate::event`.
//!
//! # How the vendor's edges become this port's waits
//!
//! The vendor inserts an edge as an `EventWait` SPLICE queued ahead of the
//! conflicting op, satisfied by an `EventSignal` spliced after the producer
//! (both counted in `queued_in_flight_` and drained by the scheduler
//! thread). This crate has no scheduler thread and no op queue — submission
//! is synchronous on the caller — so the same edge is realized as: the
//! conflicting op's caller parks in [`Event::wait`] on the producer's
//! completion event (minted here, one per tracked op) before the op is
//! submitted, and the producer's wrapped completion callback signals (or,
//! on failure, poisons) that event. `crate::stream`'s
//! `launch_event_signal`/`launch_event_wait` is the same translation for
//! the user-level event ops.
//!
//! # Recorded divergences (synchronous port)
//!
//! - **No queued accounting.** The vendor's splices are counted in
//!   `queued_in_flight_`; here the wait burns a caller thread, which is
//!   this port's uniform answer to queued ops (see `crate::stream`'s
//!   module doc).
//! - **No deadlock detector — and tracker-induced waits cannot need one.**
//!   The vendor arms RAS Deadlock detection at `synchronize()` for circular
//!   unsignaled `EventWait`s. Here, an op only ever waits on outstanding
//!   ops that registered BEFORE it: registration and conflict scan happen
//!   under one lock hold, so every wait-graph edge points from a higher to
//!   a lower registration sequence number — a DAG, acyclic by construction.
//!   A cycle through USER events (`launch_event_wait`) is still possible
//!   and still blocks its caller threads, exactly like the vendor without
//!   its detector; recorded rather than reimplemented.
//! - **Address sets are derived, not vendor-literal.** The vendor's tracker
//!   lives inside the SDK's scheduler (not in the shipped headers), so the
//!   read/write sets per op kind are DERIVED here from what each params
//!   struct addresses — see [`address_sets`]'s doc for the per-kind
//!   choices. Over-edging only serializes; under-edging corrupts; every
//!   ambiguous kind is chosen conservative.
//! - **Poison inheritance.** When a producer op fails, the vendor's
//!   spliced `EventWait` hits a poisoned event and `processEventWaitOp`'s
//!   poison arm records the producer's error on the WAITER's stream. Here
//!   the parked caller drives its own completion with
//!   [`PoisonedEventError`] wrapping the producer's error — the waiter
//!   stream's deferred error, surfaced by its `synchronize()`, never a
//!   synchronous throw. Same error object, same stickiness, same deferral
//!   point.

use std::any::Any;
use std::collections::HashSet;
use std::error::Error;
use std::sync::{Arc, Mutex};

use crate::address::{Chunk, CompositeAddress, RegionId};
use crate::compute::ComputeParams;
use crate::dma::{DmaDirection, DmaParams, FillParams};
use crate::event::{Event, PoisonedEventError};
use crate::memory_region::DomainId;
use crate::stream::{CompletionCallback, RuntimeOperationKind, SchedulerError, StreamId};

/// The Box-half of a shared scheduler error. When a tracked op fails, its
/// error must reach BOTH the stream's own completion (a `Box`, via
/// [`SchedulerError`]) and every waiter parked on the op's event (an
/// [`Arc`], via [`Event::poison`]) — one error object, two handles, which
/// is the Rust shape of the vendor sharing one `exception_ptr` between
/// producer and waiters. `Display`/`source` delegate, so the producer
/// stream's deferred error renders exactly as it would have unwrapped.
struct SharedSchedulerError(Arc<dyn Error + Send + Sync>);

impl std::fmt::Debug for SharedSchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedSchedulerError({})", self.0)
    }
}

impl std::fmt::Display for SharedSchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for SharedSchedulerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.0.as_ref() as &(dyn Error + 'static))
    }
}

/// One device-memory byte range a tracked op reads or writes: chunks are
/// the granularity `CompositeAddress` carries, so chunks are the
/// granularity edges are minted at. Two ranges conflict only within the
/// same domain AND region — different domains/regions are disjoint address
/// spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AddrRange {
    domain_id: DomainId,
    region_id: RegionId,
    /// Inclusive start byte offset within the region.
    start: u64,
    /// Exclusive end byte offset within the region.
    end: u64,
}

impl AddrRange {
    fn from_chunk(chunk: &Chunk) -> Self {
        let start = chunk.addr.offset.0;
        Self {
            domain_id: chunk.domain_id,
            region_id: chunk.addr.region_id,
            start,
            end: start + chunk.size.0,
        }
    }

    fn overlaps(&self, other: &Self) -> bool {
        self.domain_id == other.domain_id
            && self.region_id == other.region_id
            && self.start < other.end
            && other.start < self.end
    }
}

/// One registered, not-yet-completed tracked op: its stream, its
/// registration sequence number (the total order the wait DAG is built
/// on — see the module doc's acyclicity note), its address sets, and the
/// producer completion event every conflicting later op parks on.
struct OutstandingOp {
    stream: StreamId,
    seq: u64,
    reads: Vec<AddrRange>,
    writes: Vec<AddrRange>,
    done: Arc<Event>,
}

#[derive(Default)]
struct TrackerState {
    tracked_streams: HashSet<StreamId>,
    outstanding: Vec<OutstandingOp>,
    next_seq: u64,
}

/// The outcome of [`HazardTracker::on_submit`] for `schedule()`.
pub(crate) enum HazardDisposition {
    /// Proceed with submission, passing this — for a tracked op, hazard-
    /// wrapped — completion to the backend. For an untracked op (untracked
    /// stream, `skip_hazard`, or no device addresses) the completion is
    /// returned untouched and nothing was recorded.
    Proceed(CompletionCallback),
    /// A conflicting producer FAILED while this op was parked on its event:
    /// the completion has ALREADY been driven with the producer's error
    /// (wrapped in [`PoisonedEventError`]), so the op must NOT be submitted.
    /// `schedule()` returns `Ok(())` — the failure is the waiter stream's
    /// deferred error, the same reporting point as every other submission
    /// failure in this crate.
    ProducerFailed,
}

/// The scheduler's cross-stream hazard tracker. Owned by
/// [`RuntimeScheduler`](crate::scheduler::RuntimeScheduler); consulted once
/// per `schedule()` call. Interior state is one mutex, held only for
/// registration/scan/retire — NEVER while parked on an event (the
/// completion path must be able to retire through it).
pub(crate) struct HazardTracker {
    state: Arc<Mutex<TrackerState>>,
}

impl HazardTracker {
    pub(crate) fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(TrackerState::default())),
        }
    }

    /// Port of the registration side of `track_hazards`
    /// (`runtime_stream.hpp:82-88`): a stream registers as hazard-tracked
    /// at `register_stream` time; unregistered streams are user-owned and
    /// never edged. Registering with `false` (or re-registering) removes
    /// the stream from the tracked set.
    pub(crate) fn register_stream(&self, id: StreamId, track_hazards: bool) {
        let mut st = self.state.lock().unwrap();
        if track_hazards {
            st.tracked_streams.insert(id);
        } else {
            st.tracked_streams.remove(&id);
        }
    }

    /// Stream teardown: stop tracking it and drop its outstanding records.
    /// `RuntimeStream`'s `Drop` drains (`synchronize`) before releasing, so
    /// any outstanding entries' events are already terminal by the time
    /// this runs — there can be no waiter left parked on one.
    pub(crate) fn release_stream(&self, id: StreamId) {
        let mut st = self.state.lock().unwrap();
        st.tracked_streams.remove(&id);
        st.outstanding.retain(|o| o.stream != id);
    }

    /// The `schedule()`-time consult: the synchronous port of the vendor's
    /// edge minting. Computes the op's address sets, parks the caller on
    /// every conflicting still-outstanding op registered by ANOTHER tracked
    /// stream (RAW/WAR/WAW all resolve to later-waits-for-earlier), then
    /// hands back a completion wrapped to signal-or-poison the op's own
    /// event and retire its record.
    ///
    /// Registration happens under the same lock hold as the conflict scan
    /// and BEFORE any parking — that atomicity is what makes the wait graph
    /// a DAG (an op can only ever wait on lower sequence numbers) and what
    /// makes a single scan sufficient (anything registered later sees THIS
    /// op and waits on it).
    pub(crate) fn on_submit(
        &self,
        stream: StreamId,
        kind: RuntimeOperationKind,
        payload: &dyn Any,
        skip_hazard: bool,
        completion: CompletionCallback,
    ) -> HazardDisposition {
        // `skip_hazard`: "neither checked against nor recorded as a
        // producer/consumer" (`runtime_submission_params.hpp:662-664`).
        if skip_hazard {
            return HazardDisposition::Proceed(completion);
        }
        let (reads, writes) = address_sets(kind, payload);
        if reads.is_empty() && writes.is_empty() {
            // No device addresses: nothing to check against, nothing to
            // record as — the same disposition `skip_hazard` buys
            // explicitly (P2PRdmaWait, firmware ops, host callbacks).
            return HazardDisposition::Proceed(completion);
        }
        let (seq, done, conflicts) = {
            let mut st = self.state.lock().unwrap();
            if !st.tracked_streams.contains(&stream) {
                // User-owned stream, never edged (`runtime_stream.hpp:85-86`).
                return HazardDisposition::Proceed(completion);
            }
            let seq = st.next_seq;
            st.next_seq += 1;
            let done = Arc::new(Event::new());
            let conflicts: Vec<Arc<Event>> = st
                .outstanding
                .iter()
                // Same-stream ordering is the stream's own, never edged
                // (`runtime_context.hpp:73`).
                .filter(|o| o.stream != stream)
                .filter(|o| conflicts_with(&reads, &writes, o))
                .map(|o| o.done.clone())
                .collect();
            st.outstanding.push(OutstandingOp {
                stream,
                seq,
                reads,
                writes,
                done: done.clone(),
            });
            (seq, done, conflicts)
        }; // Lock released BEFORE parking: the completion path retires
        // through this same mutex and must never be blocked by a waiter.

        for ev in &conflicts {
            ev.wait();
            if ev.is_poisoned() {
                // The producer failed while we were parked on it — the
                // vendor's spliced EventWait hitting a poisoned event.
                // processEventWaitOp's poison arm records the producer's
                // error on the WAITER's stream; the port drives this op's
                // completion with that error (deferred + sticky shutdown
                // on the waiter stream, never a synchronous throw) and
                // tells the caller not to submit.
                self.retire(seq);
                let producer_error = ev
                    .get_error()
                    .expect("is_poisoned implies poison stored an error first");
                completion(Err(Box::new(PoisonedEventError::new(producer_error))));
                return HazardDisposition::ProducerFailed;
            }
        }

        let state = self.state.clone();
        let tracked_done = done;
        let tracked_seq = seq;
        let wrapped: CompletionCallback = Box::new(move |result| {
            let result = match result {
                Ok(()) => {
                    tracked_done.signal();
                    Ok(())
                }
                Err(e) => {
                    // One error object, two handles: the Arc half poisons
                    // every waiter parked on this op's event; the Box half
                    // (SharedSchedulerError, delegating Display/source)
                    // flows to the producer stream's own completion so its
                    // `synchronize()` re-throws the same failure.
                    let shared: Arc<dyn Error + Send + Sync> = Arc::from(e);
                    tracked_done.poison(shared.clone());
                    Err(Box::new(SharedSchedulerError(shared)) as SchedulerError)
                }
            };
            {
                let mut st = state.lock().unwrap();
                st.outstanding.retain(|o| o.seq != tracked_seq);
            }
            completion(result);
        });
        HazardDisposition::Proceed(wrapped)
    }

    fn retire(&self, seq: u64) {
        let mut st = self.state.lock().unwrap();
        st.outstanding.retain(|o| o.seq != seq);
    }

    /// Test introspection: how many tracked ops are currently registered
    /// as outstanding.
    #[cfg(test)]
    pub(crate) fn outstanding_len(&self) -> usize {
        self.state.lock().unwrap().outstanding.len()
    }

    /// Test introspection: is this stream registered as hazard-tracked?
    #[cfg(test)]
    pub(crate) fn is_tracked(&self, id: StreamId) -> bool {
        self.state.lock().unwrap().tracked_streams.contains(&id)
    }
}

/// Does an op with these address sets conflict with an earlier outstanding
/// op? All three hazard classes resolve to later-waits-for-earlier:
/// RAW (this op reads what the earlier op wrote), WAR (this op writes what
/// the earlier op read), WAW (this op writes what the earlier op wrote).
fn conflicts_with(reads: &[AddrRange], writes: &[AddrRange], other: &OutstandingOp) -> bool {
    reads
        .iter()
        .any(|r| other.writes.iter().any(|w| r.overlaps(w)))
        || writes
            .iter()
            .any(|w| other.reads.iter().any(|r| w.overlaps(r)))
        || writes
            .iter()
            .any(|w| other.writes.iter().any(|w2| w.overlaps(w2)))
}

/// The per-kind read/write device-address sets of one submitted op,
/// downcast from `SubmittedOperation`'s opaque payload. DERIVED, not
/// vendor-literal (the SDK's tracker is not in the shipped headers) — the
/// recorded choices, all biased toward over-edging (which serializes)
/// rather than under-edging (which corrupts):
///
/// - **H2D** writes its device address (host→device copy lands there);
///   **D2H** reads it.
/// - **Fill** writes its device address.
/// - **Compute** is conservative read+write on its program allocation AND
///   every tensor alloc: a kernel's true read/write split is not
///   expressible in `ComputeParams`, and the inputs/outputs distinction
///   lives in kernel-land, not the submission params.
/// - **P2P** kinds that carry a device address (legacy data ops,
///   `P2PRdmaSend`) are conservative read+write on it: the local side of a
///   cross-port transfer is producer or consumer depending on direction,
///   which the params do not pin down. `P2PRdmaWait` and the firmware ops
///   carry no device address at all.
/// - A payload whose type does not match its kind (impossible through the
///   typed `launch_*` paths; the scheduler's own arms report those) yields
///   empty sets — untracked.
fn address_sets(kind: RuntimeOperationKind, payload: &dyn Any) -> (Vec<AddrRange>, Vec<AddrRange>) {
    use RuntimeOperationKind as K;
    let chunks_of = |addr: &CompositeAddress| -> Vec<AddrRange> {
        addr.chunks().iter().map(AddrRange::from_chunk).collect()
    };
    match kind {
        K::H2D | K::D2H => {
            let Some(dma) = payload.downcast_ref::<DmaParams>() else {
                return (vec![], vec![]);
            };
            let ranges = chunks_of(dma.device_address());
            match dma.direction() {
                DmaDirection::HostToDevice => (vec![], ranges),
                DmaDirection::DeviceToHost => (ranges, vec![]),
            }
        }
        K::Fill => {
            let Some(fill) = payload.downcast_ref::<FillParams>() else {
                return (vec![], vec![]);
            };
            (vec![], chunks_of(fill.device_address()))
        }
        K::Compute => {
            let Some(compute) = payload.downcast_ref::<ComputeParams>() else {
                return (vec![], vec![]);
            };
            let mut ranges = chunks_of(compute.device_address());
            for alloc in compute.tensor_allocs() {
                ranges.extend(chunks_of(alloc));
            }
            (ranges.clone(), ranges)
        }
        K::LegacyP2PSendData | K::LegacyP2PRecvData => {
            let Some(p2p) = payload.downcast_ref::<crate::scheduler::P2PDataParams>() else {
                return (vec![], vec![]);
            };
            let ranges = chunks_of(&p2p.device_address);
            (ranges.clone(), ranges)
        }
        K::LegacyP2PDataExchange => {
            let Some(ops) = payload.downcast_ref::<Vec<crate::scheduler::P2PDataParams>>() else {
                return (vec![], vec![]);
            };
            let mut ranges = Vec::new();
            for p2p in ops {
                ranges.extend(chunks_of(&p2p.device_address));
            }
            (ranges.clone(), ranges)
        }
        K::P2PRdmaSend => {
            let Some(p2p) = payload.downcast_ref::<crate::scheduler::P2PRdmaSendParams>() else {
                return (vec![], vec![]);
            };
            let ranges = chunks_of(&p2p.device_address);
            (ranges.clone(), ranges)
        }
        // No device addresses in any of these params: neither checked
        // against nor recorded.
        K::P2PRdmaWait
        | K::P2PFirmwareSignal
        | K::P2PFirmwareWait
        | K::HostCallback
        | K::EventSignal
        | K::EventWait => (vec![], vec![]),
    }
}

impl Default for HazardTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Unit tests for the tracker itself, driven directly through `on_submit`
/// with real params payloads and test-invoked completions — no backend, no
/// `RuntimeStream`. The end-to-end wiring (through the real
/// `RuntimeScheduler::schedule` and a mock backend with manual
/// completions) is pinned in `scheduler.rs`'s `hazard_edge_tests`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::{ByteOffset, ByteSize, LogicalAddress};
    use crate::dma::{DmaShaping, HostVirtualAddress};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Duration;

    fn addr_at(region: u64, offset: u64, size: u64) -> Arc<CompositeAddress> {
        Arc::new(CompositeAddress::from_chunk(Chunk::new(
            LogicalAddress::new(RegionId(region), ByteOffset(offset)),
            ByteSize(size),
            DomainId(0),
        )))
    }

    fn h2d(region: u64) -> (RuntimeOperationKind, Box<DmaParams>) {
        (
            RuntimeOperationKind::H2D,
            Box::new(
                DmaParams::from_device_allocation(
                    HostVirtualAddress(0x1000),
                    DmaDirection::HostToDevice,
                    Some(addr_at(region, 0, 4096)),
                    DmaShaping::default(),
                )
                .unwrap(),
            ),
        )
    }

    fn d2h(region: u64) -> (RuntimeOperationKind, Box<DmaParams>) {
        (
            RuntimeOperationKind::D2H,
            Box::new(
                DmaParams::from_device_allocation(
                    HostVirtualAddress(0x1000),
                    DmaDirection::DeviceToHost,
                    Some(addr_at(region, 0, 4096)),
                    DmaShaping::default(),
                )
                .unwrap(),
            ),
        )
    }

    fn no_op_completion() -> (CompletionCallback, Arc<Mutex<Option<String>>>) {
        let seen = Arc::new(Mutex::new(None));
        let seen2 = seen.clone();
        (
            Box::new(move |result| {
                *seen2.lock().unwrap() = Some(match result {
                    Ok(()) => "ok".to_string(),
                    Err(e) => format!("err: {e}"),
                });
            }),
            seen,
        )
    }

    /// A tracker with two tracked streams, `a` and `b`.
    fn tracker_with_ab() -> (Arc<HazardTracker>, StreamId, StreamId) {
        let t = Arc::new(HazardTracker::new());
        let (a, b) = (StreamId::test_id(1), StreamId::test_id(2));
        t.register_stream(a, true);
        t.register_stream(b, true);
        (t, a, b)
    }

    fn proceed_or_panic(disposition: HazardDisposition) -> CompletionCallback {
        match disposition {
            HazardDisposition::Proceed(cb) => cb,
            HazardDisposition::ProducerFailed => {
                panic!("expected the submission to proceed, not inherit a failure")
            }
        }
    }

    // RAW: b's read of X waits for a's earlier write of X.
    #[test]
    fn raw_edge_waits_for_earlier_cross_stream_write() {
        let (t, a, b) = tracker_with_ab();
        let (kind_a, payload_a) = h2d(1);
        let (cb_a, _seen_a) = no_op_completion();
        let cb_a = proceed_or_panic(t.on_submit(a, kind_a, payload_a.as_ref(), false, cb_a));
        assert_eq!(t.outstanding_len(), 1, "a's op must be recorded");

        let (kind_b, payload_b) = d2h(1);
        let returned = Arc::new(AtomicBool::new(false));
        let returned2 = returned.clone();
        let t2 = t.clone();
        let waiter = thread::spawn(move || {
            let (cb_b, _seen_b) = no_op_completion();
            let disposition = t2.on_submit(b, kind_b, payload_b.as_ref(), false, cb_b);
            returned2.store(true, Ordering::Release);
            disposition
        });

        thread::sleep(Duration::from_millis(50));
        assert!(
            !returned.load(Ordering::Acquire),
            "b must stay parked while a's write is outstanding"
        );

        cb_a(Ok(()));
        assert!(matches!(
            waiter.join().unwrap(),
            HazardDisposition::Proceed(_)
        ));
        // a retired on its completion; b is the one outstanding record.
        assert_eq!(t.outstanding_len(), 1);
    }

    // WAW: b's write of X waits for a's earlier write of X.
    #[test]
    fn waw_edge_waits_for_earlier_cross_stream_write() {
        let (t, a, b) = tracker_with_ab();
        let (kind_a, payload_a) = h2d(1);
        let (cb_a, _seen_a) = no_op_completion();
        let cb_a = proceed_or_panic(t.on_submit(a, kind_a, payload_a.as_ref(), false, cb_a));

        let (kind_b, payload_b) = h2d(1);
        let waiter = {
            let t = t.clone();
            thread::spawn(move || {
                let (cb_b, _seen_b) = no_op_completion();
                t.on_submit(b, kind_b, payload_b.as_ref(), false, cb_b)
            })
        };
        thread::sleep(Duration::from_millis(50));
        cb_a(Ok(()));
        assert!(matches!(
            waiter.join().unwrap(),
            HazardDisposition::Proceed(_)
        ));
    }

    // WAR: b's write of X waits for a's earlier read of X.
    #[test]
    fn war_edge_waits_for_earlier_cross_stream_read() {
        let (t, a, b) = tracker_with_ab();
        let (kind_a, payload_a) = d2h(1);
        let (cb_a, _seen_a) = no_op_completion();
        let cb_a = proceed_or_panic(t.on_submit(a, kind_a, payload_a.as_ref(), false, cb_a));

        let (kind_b, payload_b) = h2d(1);
        let waiter = {
            let t = t.clone();
            thread::spawn(move || {
                let (cb_b, _seen_b) = no_op_completion();
                t.on_submit(b, kind_b, payload_b.as_ref(), false, cb_b)
            })
        };
        thread::sleep(Duration::from_millis(50));
        cb_a(Ok(()));
        assert!(matches!(
            waiter.join().unwrap(),
            HazardDisposition::Proceed(_)
        ));
    }

    // Same-stream ordering is the stream's own — never edged.
    #[test]
    fn same_stream_ops_never_edge() {
        let (t, a, _b) = tracker_with_ab();
        let (kind_a, payload_a) = h2d(1);
        let (cb_a, _seen_a) = no_op_completion();
        let _cb_a = proceed_or_panic(t.on_submit(a, kind_a, payload_a.as_ref(), false, cb_a));

        // A read of the SAME address on the SAME stream: returns
        // immediately, no park.
        let (kind_b, payload_b) = d2h(1);
        let (cb_b, _seen_b) = no_op_completion();
        let started = std::time::Instant::now();
        let disposition = t.on_submit(a, kind_b, payload_b.as_ref(), false, cb_b);
        assert!(matches!(disposition, HazardDisposition::Proceed(_)));
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "same-stream op must not park on its own stream's outstanding op"
        );
    }

    // Disjoint addresses (different regions) never edge.
    #[test]
    fn disjoint_addresses_do_not_edge() {
        let (t, a, b) = tracker_with_ab();
        let (kind_a, payload_a) = h2d(1);
        let (cb_a, _seen_a) = no_op_completion();
        let _cb_a = proceed_or_panic(t.on_submit(a, kind_a, payload_a.as_ref(), false, cb_a));

        let (kind_b, payload_b) = d2h(2);
        let (cb_b, _seen_b) = no_op_completion();
        let started = std::time::Instant::now();
        let disposition = t.on_submit(b, kind_b, payload_b.as_ref(), false, cb_b);
        assert!(matches!(disposition, HazardDisposition::Proceed(_)));
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "disjoint addresses must not park"
        );
    }

    // skip_hazard: "neither checked against nor recorded as a
    // producer/consumer".
    #[test]
    fn skip_hazard_is_neither_checked_nor_recorded() {
        let (t, a, b) = tracker_with_ab();
        // Recorded side: a skips → nothing outstanding.
        let (kind_a, payload_a) = h2d(1);
        let (cb_a, _seen_a) = no_op_completion();
        let _ = t.on_submit(a, kind_a, payload_a.as_ref(), true, cb_a);
        assert_eq!(t.outstanding_len(), 0, "a skipped op is not recorded");

        // Checked side: a recorded normally, b skips → b does not park.
        let (kind_a2, payload_a2) = h2d(1);
        let (cb_a2, _seen_a2) = no_op_completion();
        let _cb_a2 = proceed_or_panic(t.on_submit(a, kind_a2, payload_a2.as_ref(), false, cb_a2));
        let (kind_b, payload_b) = d2h(1);
        let (cb_b, _seen_b) = no_op_completion();
        let started = std::time::Instant::now();
        let disposition = t.on_submit(b, kind_b, payload_b.as_ref(), true, cb_b);
        assert!(matches!(disposition, HazardDisposition::Proceed(_)));
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "a skipped op must not park on a conflicting producer"
        );
    }

    // Untracked streams are user-owned: never edged, never recorded.
    #[test]
    fn untracked_stream_is_user_owned() {
        let t = Arc::new(HazardTracker::new());
        let a = StreamId::test_id(1);
        let b = StreamId::test_id(2);
        t.register_stream(a, true);
        t.register_stream(b, false);

        // b untracked → its op is not recorded.
        let (kind_b, payload_b) = h2d(1);
        let (cb_b, _seen_b) = no_op_completion();
        let _ = t.on_submit(b, kind_b, payload_b.as_ref(), false, cb_b);
        assert_eq!(t.outstanding_len(), 0, "untracked op is not recorded");

        // a tracked → does not park on b's (unrecorded) conflicting op.
        let (kind_a, payload_a) = d2h(1);
        let (cb_a, _seen_a) = no_op_completion();
        let started = std::time::Instant::now();
        let disposition = t.on_submit(a, kind_a, payload_a.as_ref(), false, cb_a);
        assert!(matches!(disposition, HazardDisposition::Proceed(_)));
        assert!(
            started.elapsed() < Duration::from_millis(50),
            "tracked op must not park on an untracked producer"
        );
    }

    // A failed producer's error is inherited by the waiter, through the
    // waiter's own completion, as a PoisonedEventError chain.
    #[test]
    fn waiter_inherits_failed_producer_error() {
        let (t, a, b) = tracker_with_ab();
        let (kind_a, payload_a) = h2d(1);
        let (cb_a, _seen_a) = no_op_completion();
        let cb_a = proceed_or_panic(t.on_submit(a, kind_a, payload_a.as_ref(), false, cb_a));

        let (kind_b, payload_b) = d2h(1);
        let (cb_b, seen_b) = no_op_completion();
        let waiter = {
            let t = t.clone();
            thread::spawn(move || t.on_submit(b, kind_b, payload_b.as_ref(), false, cb_b))
        };
        thread::sleep(Duration::from_millis(50));

        cb_a(Err("producer boom".to_string().into()));
        match waiter.join().unwrap() {
            HazardDisposition::ProducerFailed => {}
            HazardDisposition::Proceed(_) => {
                panic!("expected the waiter to inherit the producer's failure")
            }
        }
        let seen = seen_b.lock().unwrap().clone().unwrap();
        assert!(
            seen.contains("poisoned by a failed producer stream"),
            "waiter error must name the poison: {seen}"
        );
        assert!(
            seen.contains("producer boom"),
            "waiter error must carry the producer's own error: {seen}"
        );
    }

    // The wrapped completion retires the op's record on both success and
    // failure.
    #[test]
    fn completion_retires_the_record() {
        let (t, a, _b) = tracker_with_ab();
        let (kind_a, payload_a) = h2d(1);
        let (cb_a, _seen_a) = no_op_completion();
        let cb_a = proceed_or_panic(t.on_submit(a, kind_a, payload_a.as_ref(), false, cb_a));
        assert_eq!(t.outstanding_len(), 1);
        cb_a(Ok(()));
        assert_eq!(t.outstanding_len(), 0, "success must retire the record");

        let (kind_a2, payload_a2) = h2d(1);
        let (cb_a2, _seen_a2) = no_op_completion();
        let cb_a2 = proceed_or_panic(t.on_submit(a, kind_a2, payload_a2.as_ref(), false, cb_a2));
        cb_a2(Err("fail".to_string().into()));
        assert_eq!(t.outstanding_len(), 0, "failure must retire the record");
    }

    // release_stream drops tracking and the stream's records.
    #[test]
    fn release_stream_drops_tracking_and_records() {
        let (t, a, _b) = tracker_with_ab();
        assert!(t.is_tracked(a));
        let (kind_a, payload_a) = h2d(1);
        let (cb_a, _seen_a) = no_op_completion();
        let _cb_a = proceed_or_panic(t.on_submit(a, kind_a, payload_a.as_ref(), false, cb_a));
        t.release_stream(a);
        assert!(!t.is_tracked(a));
        assert_eq!(t.outstanding_len(), 0, "release must drop the records");
    }

    // The per-kind address-set derivation: H2D writes, D2H reads.
    #[test]
    fn address_sets_h2d_writes_d2h_reads() {
        let (_, h2d_payload) = h2d(1);
        let (reads, writes) = address_sets(RuntimeOperationKind::H2D, h2d_payload.as_ref());
        assert!(reads.is_empty());
        assert_eq!(writes.len(), 1);

        let (_, d2h_payload) = d2h(1);
        let (reads, writes) = address_sets(RuntimeOperationKind::D2H, d2h_payload.as_ref());
        assert!(writes.is_empty());
        assert_eq!(reads.len(), 1);
    }

    // Kinds with no device addresses are neither checked nor recorded.
    #[test]
    fn address_sets_empty_for_addressless_kinds() {
        let (reads, writes) = address_sets(RuntimeOperationKind::HostCallback, &());
        assert!(reads.is_empty() && writes.is_empty());
    }
}
