//! Port of `flex::Event` — the shared synchronization primitive for
//! inter-stream coordination.
//!
//! Source: `flex/src/runtime_stream/operations/runtime_operation.hpp:25-80`
//! (the local flex reference repo; the pod SDK's
//! `runtime_submission_params.hpp:38` forward-declares the same class with the
//! note "Event's full definition lives in
//! runtime_stream/operations/runtime_operation.hpp" — that header is not
//! shipped in the SDK, so the open repo is the authority for the definition).
//!
//! `EventSignalOp`/`EventWaitOp` (`runtime_operation_event.hpp`) do not get
//! Rust op-struct counterparts. In the vendor, operations reach the scheduler
//! as `RuntimeOperation` subclasses and the scheduler thread routes events to
//! `processEventSignalOp`/`processEventWaitOp`
//! (`runtime_scheduler_internal.cpp:339-350`) — never to `schedule()`. In this
//! crate operations cross the [`crate::stream::SchedulerHandle`] boundary as
//! [`crate::stream::SubmittedOperation`]s, but events still never cross it:
//! [`crate::stream::RuntimeStream::launch_event_signal`] /
//! [`crate::stream::RuntimeStream::launch_event_wait`] execute the pair
//! directly, and the ops' own forwarded queries (`EventWaitOp::isReady` /
//! `isPoisoned` / `getError`) are `Event`'s methods here. This is why the
//! event pair was previously recorded as unported in `stream.rs`'s module
//! doc — "no `Event` type exists anywhere in this crate — writing one would
//! be invented behavior rather than a port" — and why this module exists now.
//!
//! Contract tests: ports of the vendor's
//! `tests/runtime_stream/stream/event_primitives_test.cpp` (8) and
//! `event_wait_test.cpp` (8), adapted where noted (the C++ measures thread
//! CPU time via `CLOCK_THREAD_CPUTIME_ID` to distinguish a park from a spin;
//! Rust's std has no such API, so those tests keep the behavioral assertions
//! — park, wake, promptness — and record the dropped metric).

use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Port of `flex::Event` (`runtime_operation.hpp:25-80`): a one-shot
/// signal/poison latch shared across streams.
///
/// `signal()` and `poison()` are independent and sticky — setting one never
/// clears or blocks the other, whichever order they land in. `poison` is
/// first-caller-wins. `wait()` blocks until either is set and rethrows
/// nothing; callers decide what a poison means (a waiter stream turns it into
/// its own error state — see `launch_event_wait`).
///
/// Deliberate type deviation: the C++ stores a `std::exception_ptr` and
/// `getError()` hands out copies — a refcounted handle, so a producer stream
/// and every waiter rethrow the *same* exception. The Rust port stores
/// `Arc<dyn Error>` and [`Event::get_error`] returns a clone of that handle,
/// which is exactly that sharing. For the same reason the stream's deferred
/// slot holds `Arc<RuntimeStreamError>`
/// ([`crate::stream::RuntimeStream::peek_deferred_error`]): a poison hands
/// the producer's real error to a waiter without moving it out from under
/// the producer's own `synchronize()`.
pub struct Event {
    signaled: AtomicBool,
    poisoned: AtomicBool,
    claimed: AtomicBool,
    error: Mutex<Option<Arc<dyn Error + Send + Sync>>>,
    mtx: Mutex<()>,
    cv: Condvar,
}

impl Event {
    pub fn new() -> Self {
        Self {
            signaled: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            claimed: AtomicBool::new(false),
            error: Mutex::new(None),
            mtx: Mutex::new(()),
            cv: Condvar::new(),
        }
    }

    /// Port of `Event::signal`: set `signaled_` (release) and wake every
    /// waiter. Notify inside the lock scope, matching the C++ `lock_guard`
    /// block exactly.
    pub fn signal(&self) {
        {
            let _lk = self.mtx.lock().unwrap();
            self.signaled.store(true, Ordering::Release);
            self.cv.notify_all();
        }
    }

    /// Port of `Event::poison`: first caller wins (`claimed_` CAS), store the
    /// error, then publish `poisoned_` LAST (under `mtx_`) so a reader that
    /// observes `is_poisoned()` can never see it before `error_` is written.
    /// Notify outside the lock, matching the C++.
    pub fn poison(&self, err: Arc<dyn Error + Send + Sync>) {
        // Port of the C++ `claimed_.compare_exchange_strong(expected, true,
        // memory_order_relaxed)` — the failure order defaults to the success
        // order in the single-order overload.
        if self
            .claimed
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        *self.error.lock().unwrap() = Some(err);
        {
            let _lk = self.mtx.lock().unwrap();
            self.poisoned.store(true, Ordering::Release);
        }
        self.cv.notify_all();
    }

    /// Port of `Event::isSignaled` (`memory_order_acquire`).
    #[must_use]
    pub fn is_signaled(&self) -> bool {
        self.signaled.load(Ordering::Acquire)
    }

    /// Port of `Event::isPoisoned` (`memory_order_acquire`).
    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Port of `Event::getError` — only meaningful after `is_poisoned()`
    /// returns true (that ordering is what `poison`'s publish-last sequence
    /// guarantees). Returns a clone of the shared handle, the way the C++
    /// returns a copy of the `exception_ptr`.
    #[must_use]
    pub fn get_error(&self) -> Option<Arc<dyn Error + Send + Sync>> {
        self.error.lock().unwrap().clone()
    }

    /// Port of `Event::wait`: block until `signal()` or `poison()` is called;
    /// return immediately if either already has (atomic fast path). Does NOT
    /// rethrow on poison — callers check `is_poisoned()`/`get_error()`
    /// themselves, exactly like the C++.
    pub fn wait(&self) {
        if self.is_signaled() || self.is_poisoned() {
            return;
        }
        let lk = self.mtx.lock().unwrap();
        let _guard = self
            .cv
            .wait_while(lk, |()| !self.is_signaled() && !self.is_poisoned())
            .unwrap();
    }
}

impl Default for Event {
    fn default() -> Self {
        Self::new()
    }
}

/// The error a waited [`Event`] was poisoned with, wrapped so it can enter a
/// waiter stream's deferred slot while preserving the producer's original
/// error as `source()`. The Rust shape of `processEventWaitOp`'s poison arm
/// (`runtime_scheduler_internal.cpp:419-437`: `hooks.setError(op->getError())`
/// — the waiter stream records the producer's real error and flips sticky
/// shutdown).
pub struct PoisonedEventError {
    source: Arc<dyn Error + Send + Sync>,
}

impl PoisonedEventError {
    pub fn new(source: Arc<dyn Error + Send + Sync>) -> Self {
        Self { source }
    }
}

impl std::fmt::Debug for PoisonedEventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PoisonedEventError({})", self.source)
    }
}

impl std::fmt::Display for PoisonedEventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "waited event was poisoned by a failed producer stream: {}",
            self.source
        )
    }
}

impl Error for PoisonedEventError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.as_ref() as &(dyn Error + 'static))
    }
}

/// Ports of the vendor's `tests/runtime_stream/stream/event_primitives_test.cpp`
/// and `event_wait_test.cpp`.
///
/// The C++ suite exercises `Event` directly and through `EventWaitOp` /
/// `EventSignalOp` wrappers; this crate has no op wrappers (see the module
/// doc), so the wrapper-level assertions (`wait_op.isReady()` after
/// `signal_op.getEvent()->signal()`) become the same assertions against the
/// shared `Event` — the wrappers only forwarded these calls.
///
/// The C++ `WaiterParksAndWakesOnSignal` / fast-path tests assert on
/// `CLOCK_THREAD_CPUTIME_ID` deltas to prove the wait parks instead of
/// spinning. Rust's std has no thread-CPU-time clock, so those ports keep the
/// behavioral assertions (the waiter must still be parked 20 ms in, must wake
/// promptly, and the fast paths must return without blocking) and drop the
/// CPU-time metric — recorded here rather than silently.
///
/// The vendor's `event_primitives_test.cpp` also carries a TODO
/// ("verify that operations queued after EventWaitOp on a stream don't
/// execute until the event is signaled after Scheduler integration"): in this
/// port submission is synchronous, so a caller blocked inside
/// `launch_event_wait` cannot submit anything further on that stream by
/// construction — program order is the ordering. The stream-level tests in
/// `stream.rs` pin that shape.
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    /// A minimal concrete error so tests can assert handle identity with
    /// `Arc::ptr_eq` (the C++ asserts `EXPECT_EQ(getError(), err)` on
    /// exception_ptr identity).
    #[derive(Debug)]
    struct TestError(&'static str);

    impl std::fmt::Display for TestError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl Error for TestError {}

    fn test_error(msg: &'static str) -> Arc<dyn Error + Send + Sync> {
        Arc::new(TestError(msg))
    }

    // event_primitives_test.cpp: SignalUnblocksWait — signaling makes the
    // waited event ready.
    #[test]
    fn signal_unblocks_wait() {
        let event = Event::new();
        assert!(!event.is_signaled());

        event.signal();

        assert!(event.is_signaled());
    }

    // event_primitives_test.cpp: PoisonedEventFailsWaitOp — poison stores the
    // error and is_poisoned()/get_error() reflect it (handle identity).
    #[test]
    fn poisoned_event_fails_wait_op() {
        let event = Event::new();
        assert!(!event.is_poisoned());

        let err = test_error("stream failed");
        event.poison(err.clone());

        assert!(event.is_poisoned());
        let stored = event.get_error().expect("poison must store the error");
        assert!(
            Arc::ptr_eq(&stored, &err),
            "get_error must return the same shared handle poison stored"
        );
    }

    // event_primitives_test.cpp: IndependentEventsDoNotInterfere.
    #[test]
    fn independent_events_do_not_interfere() {
        let event_a = Event::new();
        let event_b = Event::new();

        event_a.signal();

        assert!(event_a.is_signaled());
        assert!(!event_b.is_signaled());
    }

    // event_primitives_test.cpp: SharedEventLinksBetweenSignalAndWaitOp — one
    // Event shared by both sides; a signal on the shared object is visible to
    // the waiter.
    #[test]
    fn shared_event_links_between_signal_and_wait_op() {
        let event = Arc::new(Event::new());
        let waiter_side = event.clone();

        assert!(!waiter_side.is_signaled());

        event.signal();

        assert!(waiter_side.is_signaled());
    }

    // event_primitives_test.cpp: MultipleWaitOpsShareOneEvent — one signal()
    // satisfies every waiter on the same Event.
    #[test]
    fn multiple_wait_ops_share_one_event() {
        let event = Arc::new(Event::new());
        let waiters: Vec<_> = (0..3).map(|_| event.clone()).collect();

        for w in &waiters {
            assert!(!w.is_signaled());
        }

        event.signal();

        for w in &waiters {
            assert!(w.is_signaled());
        }
    }

    // event_primitives_test.cpp: PoisonIsVisibleAcrossThreads — producer
    // poisons on another thread; a consumer that observes is_poisoned() can
    // then read get_error() and see the same handle.
    #[test]
    fn poison_is_visible_across_threads() {
        let event = Arc::new(Event::new());
        let err = test_error("stream failed");

        let producer_event = event.clone();
        let producer_err = err.clone();
        let producer = thread::spawn(move || producer_event.poison(producer_err));

        // Consumer waits for the poisoned flag before reading the error —
        // same shape as the C++ spin loop.
        while !event.is_poisoned() {}
        let stored = event.get_error().expect("poison must store the error");
        assert!(Arc::ptr_eq(&stored, &err));

        producer.join().unwrap();
    }

    // event_primitives_test.cpp: DoubleSignalIsStickyNoOp.
    #[test]
    fn double_signal_is_sticky_no_op() {
        let event = Event::new();
        event.signal();
        assert!(event.is_signaled());

        event.signal();
        assert!(event.is_signaled());
        assert!(!event.is_poisoned());
    }

    // event_primitives_test.cpp: SignalAndPoisonAreIndependent — in both
    // orders, setting one never clears the other.
    #[test]
    fn signal_and_poison_are_independent() {
        let event = Event::new();
        let err = test_error("stream failed");

        event.signal();
        event.poison(err.clone());
        assert!(event.is_signaled());
        assert!(event.is_poisoned());
        assert!(Arc::ptr_eq(&event.get_error().unwrap(), &err));

        let reversed = Event::new();
        reversed.poison(err);
        reversed.signal();
        assert!(reversed.is_poisoned());
        assert!(reversed.is_signaled());
        assert!(reversed.get_error().is_some());
    }

    // event_wait_test.cpp: WaiterParksAndWakesOnSignal — a waiter parked in
    // wait() must still be there 20 ms later and must wake on signal().
    // (The C++ also asserts the wait burned <5 ms of thread CPU time; Rust's
    // std has no CLOCK_THREAD_CPUTIME_ID, so that metric is dropped — see the
    // module doc.)
    #[test]
    fn waiter_parks_and_wakes_on_signal() {
        let event = Arc::new(Event::new());
        let waiter_returned = Arc::new(AtomicBool::new(false));

        let waiter_event = event.clone();
        let waiter_flag = waiter_returned.clone();
        let waiter = thread::spawn(move || {
            waiter_event.wait();
            waiter_flag.store(true, Ordering::Release);
        });

        // Give the waiter thread time to park in the CV.
        thread::sleep(Duration::from_millis(20));
        assert!(
            !waiter_returned.load(Ordering::Acquire),
            "waiter should not have returned before signal()"
        );

        event.signal();

        // Waiter must wake promptly (join is unbounded, matching the C++).
        waiter.join().unwrap();
        assert!(waiter_returned.load(Ordering::Acquire));
    }

    // event_wait_test.cpp: WaiterWakesOnPoison.
    #[test]
    fn waiter_wakes_on_poison() {
        let event = Arc::new(Event::new());
        let waiter_returned = Arc::new(AtomicBool::new(false));

        let waiter_event = event.clone();
        let waiter_flag = waiter_returned.clone();
        let waiter = thread::spawn(move || {
            waiter_event.wait();
            waiter_flag.store(true, Ordering::Release);
        });

        thread::sleep(Duration::from_millis(20));
        assert!(!waiter_returned.load(Ordering::Acquire));

        let err = test_error("stream failed");
        event.poison(err.clone());

        waiter.join().unwrap();

        assert!(waiter_returned.load(Ordering::Acquire));
        assert!(event.is_poisoned());
        assert!(Arc::ptr_eq(&event.get_error().unwrap(), &err));
    }

    // event_wait_test.cpp: PoisonErrorRethrowAfterWait — after wait() returns
    // on a poisoned event, the stored error is retrievable (the C++ rethrows
    // it; the Rust shape is the shared handle + PoisonedEventError's
    // source() chain).
    #[test]
    fn poison_error_rethrow_after_wait() {
        let event = Event::new();
        let err = test_error("poison test");
        event.poison(err.clone());

        // wait() returns (fast path or CV); caller then decides to rethrow.
        event.wait();

        assert!(event.is_poisoned());
        let stored = event.get_error().unwrap();
        assert!(Arc::ptr_eq(&stored, &err));
        assert_eq!(stored.to_string(), "poison test");
    }

    // event_wait_test.cpp: AlreadySignaledFastPath — wait() on an
    // already-signaled event returns immediately without blocking. (The C++
    // asserts <1 ms of thread CPU; the Rust port asserts the return, with
    // the metric dropped — see the module doc.)
    #[test]
    fn already_signaled_fast_path() {
        let event = Event::new();
        event.signal();

        event.wait();

        assert!(event.is_signaled());
    }

    // event_wait_test.cpp: AlreadyPoisonedFastPath.
    #[test]
    fn already_poisoned_fast_path() {
        let event = Event::new();
        event.poison(test_error("already poisoned"));

        event.wait();

        assert!(event.is_poisoned());
    }

    // event_wait_test.cpp: ConcurrentPoisonFirstCallerWins — 8 threads poison
    // concurrently; exactly one error is stored and it is one of theirs.
    #[test]
    fn concurrent_poison_first_caller_wins() {
        const THREADS: usize = 8;
        let event = Arc::new(Event::new());
        let errors: Vec<_> = (0..THREADS)
            .map(|i| test_error(Box::leak(format!("error {i}").into_boxed_str())))
            .collect();

        let mut threads = Vec::with_capacity(THREADS);
        for err in &errors {
            let event = event.clone();
            let err = err.clone();
            threads.push(thread::spawn(move || event.poison(err)));
        }
        for t in threads {
            t.join().unwrap();
        }

        assert!(event.is_poisoned());
        let stored = event.get_error().expect("exactly one poison must win");
        assert!(
            errors.iter().any(|e| Arc::ptr_eq(e, &stored)),
            "stored error is not one of the injected errors"
        );
    }

    // event_wait_test.cpp: SignalThenPoisonNoLostWakeup — a parked waiter
    // wakes regardless of which flag lands first.
    #[test]
    fn signal_then_poison_no_lost_wakeup() {
        let event = Arc::new(Event::new());
        let waiter_returned = Arc::new(AtomicBool::new(false));

        let waiter_event = event.clone();
        let waiter_flag = waiter_returned.clone();
        let waiter = thread::spawn(move || {
            waiter_event.wait();
            waiter_flag.store(true, Ordering::Release);
        });

        thread::sleep(Duration::from_millis(20));
        assert!(
            !waiter_returned.load(Ordering::Acquire),
            "waiter should not have returned before signal()"
        );

        event.signal();
        event.poison(test_error("late poison"));

        waiter.join().unwrap();
        assert!(waiter_returned.load(Ordering::Acquire));
        assert!(event.is_signaled());
    }

    // event_wait_test.cpp: PoisonThenSignalNoLostWakeup.
    #[test]
    fn poison_then_signal_no_lost_wakeup() {
        let event = Arc::new(Event::new());
        let waiter_returned = Arc::new(AtomicBool::new(false));

        let waiter_event = event.clone();
        let waiter_flag = waiter_returned.clone();
        let waiter = thread::spawn(move || {
            waiter_event.wait();
            waiter_flag.store(true, Ordering::Release);
        });

        thread::sleep(Duration::from_millis(20));
        assert!(
            !waiter_returned.load(Ordering::Acquire),
            "waiter should not have returned before poison()"
        );

        event.poison(test_error("early poison"));
        event.signal();

        waiter.join().unwrap();
        assert!(waiter_returned.load(Ordering::Acquire));
        assert!(event.is_poisoned());
        assert!(event.is_signaled());
    }
}
