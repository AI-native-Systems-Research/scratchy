//! The #611/#653 on-card confirmation leg: ONE cancelled marker through the
//! REAL senlib link path, asserting the verified vendor contract on hardware.
//!
//! `#[ignore]`d because it needs a Spyre card — `cargo test` skips it
//! everywhere cardless (CI, laptops). Run it BY NAME on a card host (the
//! exact invocation is in [`cancelled_marker_on_card`]'s doc). It is the
//! single RBI consumer while it runs: the marker is queued directly through
//! `SenlibHardwareSubmit` — the same three senlib calls `ResponseWorker`
//! itself makes — mirroring how the VENDOR queues its own shutdown marker
//! (outside the normal queuing path, `response_worker.cpp:326`), so the
//! port's `ResponseCompletionThread` stays parked on its wakeup channel
//! (`num_pending_requests` is 0; nothing was submitted through the
//! scheduler) and cannot steal the shared RBI cursor from this drain.
//!
//! What this proves on card, and nothing more: the marker (a zero-work
//! compute CB with the CTRL cancel byte) is accepted by the real ring, its
//! ring slot is consumed, and an RB carrying the cancel bit comes back. It
//! does NOT prove a work-carrying cancelled CB would skip its work — that
//! is unverified even in the vendor's own sources (see
//! `ControlBlockWire::cancelled`'s doc for both pinned caveats).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use flex_rs::allocator::{DEVICE_ALIGNMENT, FlexAllocator, FlexAllocatorInit};
use flex_rs::control_block_wire::{ControlBlockWire, RB_NUM_BYTES, ResponseBlockWire, ResponseTag};
use flex_rs::device_memory_allocator::DeviceMemoryAllocator;
use flex_rs::flex_config::DeviceKind;
use flex_rs::iommu::PfIommuMapper;
use flex_rs::runtime::RuntimeContext;
use flex_rs::scheduler::{
    FlexAllocatorAddressResolver, HardwareSubmit, PipelineId, SchedulerBackendConfig,
    SenlibHardwareSubmit, SenlibQueueCapacity, create_scheduler,
};
use flex_rs::senlib_ffi::{
    flex_senlib_device_memory_size, flex_shim_init_runtime, sen_pci_card_count,
};
use flex_rs::stream::{RuntimeStreamMode, RuntimeStreamPriority};
use flex_rs::topology::populate_domain_topology;

/// Drain budget. The vendor waits 5 s on its own shutdown marker
/// (`response_worker.cpp`, `kShutdownTimeout`); this doubles that for a
/// first-ever on-card run, stays bounded, and FAILS with the observation
/// rather than hanging when the RB never lands.
const RB_WAIT: Duration = Duration::from_secs(10);
/// Poll gap for the non-blocking drain (`min_count == 0`, matching
/// `ResponseWorker::FetchResponseBlocks`' real call shape).
const POLL_GAP: Duration = Duration::from_millis(25);
/// How many RBs one drain call reads at once (only the marker is expected).
const DRAIN_BATCH: usize = 16;

/// Queue EXACTLY ONE `ControlBlockWire::cancelled` marker on real hardware
/// and assert the verified contract: (a) the RB that comes back carries the
/// cancel bit, (b) the ring slot was consumed — an RB exists for the
/// submitted CB and its response-slot credit round-trips back to senlib
/// through `FreeResponses`.
///
/// Refuses loudly when cardless: on a host with no Spyre card (or against
/// the cardless link stub) this FAILS with a message saying so — never a
/// silent pass or vacuous skip.
///
/// Run by name on a card host, from the repo root, WITHOUT
/// `SCRATCHY_SKIP_SENDNN_CXX` (the shim must really compile and link):
///
/// ```text
/// FLEX_RS_SPYRE_ROOT=<real spyre root, e.g. /opt/ibm/spyre> \
/// BOOST_LIB=<dir holding libboost_*.so, e.g. /usr/lib64> \
/// cargo test -p flex-rs --test on_card_cancelled_marker -- --ignored --nocapture
/// ```
#[ignore = "on-card leg (#611/#653): needs a real Spyre card — run by name, see the test's doc for the invocation"]
#[test]
fn cancelled_marker_on_card() {
    // ── Refusal guard 1: no card visible ⇒ fail loudly, by name. ──
    let ncards = sen_pci_card_count();
    assert!(
        ncards > 0,
        "REFUSED (not silently passed): sen_pci_card_count() == {ncards} — no Spyre card is \
         visible on this host. `cancelled_marker_on_card` is the #611/#653 on-card confirmation \
         leg and must run on a Spyre host against the REAL senlib link (no \
         SCRATCHY_SKIP_SENDNN_CXX, real FLEX_RS_SPYRE_ROOT/BOOST_LIB)."
    );

    // ── Refusal guard 2: real senlib bring-up must succeed. ──
    // The cardless link stub makes this fail (init returns nonzero); real
    // hardware returns 0. Device 0: the same single-PF bring-up the card
    // surface uses (`flex_shim_init_runtime` only ever opens a PF).
    let rc = unsafe { flex_shim_init_runtime(0) };
    assert!(
        rc == 0,
        "REFUSED (not silently passed): flex_shim_init_runtime(0) returned {rc} — the real \
         senlib device did not come up, so the on-card contract cannot be observed here."
    );

    // ── The real bring-up, exactly the card surface's own sequence (all
    // public API): topology from the card's actual HBM/core counts, a real
    // FlexAllocator, the Real scheduler backend, a RuntimeContext + stream. ──
    let device_memory_size = unsafe { flex_senlib_device_memory_size() };
    assert!(
        device_memory_size > 0,
        "REFUSED: the card reports 0 device memory — bring-up did not yield a usable device"
    );
    let topo = populate_domain_topology(device_memory_size);
    let allocator = Arc::new(Mutex::new(
        FlexAllocator::new(
            FlexAllocatorInit::default(),
            topo,
            DeviceMemoryAllocator::new(DEVICE_ALIGNMENT),
        )
        .expect("FlexAllocator::new must succeed on a real card"),
    ));
    let addr_resolver: Arc<dyn flex_rs::scheduler::DeviceAddressResolver> =
        Arc::new(FlexAllocatorAddressResolver::new(allocator.clone()));
    let hw = Arc::new(SenlibHardwareSubmit::new());
    let capacity = Arc::new(SenlibQueueCapacity);
    let iommu_mapper = PfIommuMapper::new();
    let scheduler = create_scheduler(
        &DeviceKind::Pf,
        SchedulerBackendConfig::Real(hw.clone(), capacity, addr_resolver, iommu_mapper),
    )
    .expect("create_scheduler(Real) must succeed on a real card");
    let context = RuntimeContext::create(None, scheduler, allocator)
        .expect("RuntimeContext::create must succeed on a real card");
    let _stream_id = context
        .create_stream(
            RuntimeStreamPriority::Normal,
            RuntimeStreamMode::StrictOrdering,
            true,
        )
        .expect("create_stream must succeed on a real card");

    // ── Queue EXACTLY ONE cancelled marker, the vendor's own way: directly
    // through the doorbell, outside the scheduler's normal queuing path
    // (the vendor queues its shutdown marker the same way, and its runtime
    // treats a cancel RB as an error on the normal path — boundary (b) in
    // `ControlBlockWire::cancelled`'s doc). The scheduler's completion
    // thread is parked (nothing pending), so this test is the one RBI
    // consumer. Tag 0x611: distinctive, echoes the issue; nothing else was
    // submitted, so no collision is possible. ──
    let marker_tag = ResponseTag(0x611);
    let marker = ControlBlockWire::cancelled(marker_tag);
    assert!(
        marker.ctrl_cancelled(),
        "the marker must carry the CTRL cancel byte"
    );
    hw.queue_control_blocks(PipelineId::Compute, &[marker])
        .expect("queueing the one cancelled marker through the real doorbell must succeed");

    // ── Bounded diagnostic drain: poll the shared RBI for the marker's RB,
    // freeing every received slot's credit back to senlib as we go (the
    // same receive → free pairing `ParseResponseBlocks` does). ──
    let deadline = Instant::now() + RB_WAIT;
    let mut marker_rb: Option<ResponseBlockWire> = None;
    let mut marker_matches = 0usize;
    let mut total_drained = 0u64;
    while marker_rb.is_none() {
        assert!(
            Instant::now() < deadline,
            "NO RB for the cancelled marker (tag {marker_tag:?}) within {RB_WAIT:?}: the marker \
             consumed a ring slot but no response came back. Drained {total_drained} RBs total, \
             none matching — this is the observation, not a pass."
        );
        let mut rbs = [ResponseBlockWire::from_bytes(&[0u8; RB_NUM_BYTES]); DRAIN_BATCH];
        let (n, mut credit) = hw.receive_responses(&mut rbs, 0);
        total_drained += n;
        for rb in &rbs[..n as usize] {
            // Diagnostic observation for the leg log (--nocapture): the real
            // card's status byte for a cancelled CB is NOT part of the
            // asserted contract (unverified even in vendor sources).
            eprintln!(
                "[on-card] RB: tag={:?} cancelled={} status={:?}",
                rb.tag(),
                rb.cancelled(),
                rb.status()
            );
            if rb.tag() == marker_tag {
                marker_rb = Some(*rb);
                marker_matches += 1;
            }
        }
        if n > 0 {
            let share = credit
                .take(n)
                .expect("the credit must cover exactly what was received");
            hw.free_responses(PipelineId::Compute, share);
        }
        credit.settle();
        if n == 0 {
            std::thread::sleep(POLL_GAP);
        }
    }

    // (a) The RB carries the cancel bit — hardware echoed the CTRL cancel
    // into the ReturnSection, the verified vendor contract.
    let rb = marker_rb.unwrap();
    assert!(
        rb.cancelled(),
        "the marker's RB (tag {marker_tag:?}) came back WITHOUT the cancel bit — the on-card \
         behavior diverges from the verified vendor contract"
    );
    // Exactly one RB for the marker: one submitted CB, one response.
    assert_eq!(
        marker_matches, 1,
        "exactly one RB must carry the marker's tag"
    );
    // (b) The ring slot was consumed: an RB exists for the submitted CB
    // (the drain above found it), and every received response slot was
    // returned to senlib through the real FreeResponses path — the credit
    // types make it impossible to free more than was received, and the
    // drain loop freed everything it received (total_drained slots).
    assert!(
        total_drained >= 1,
        "the drain must have received at least the marker's own RB"
    );
    eprintln!(
        "[on-card] PASS: cancelled marker (tag {marker_tag:?}) consumed its ring slot and its RB \
         came back with the cancel bit; {total_drained} response slot(s) freed back to senlib"
    );
}
