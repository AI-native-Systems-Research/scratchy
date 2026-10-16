//! Per-CB device timing, harvested from the response blocks the completion thread already decodes.
//!
//! Every 64-byte response block carries a timestamp section — five 32-bit cycle counts plus a power
//! word — written by hardware for every completed CB
//! ([`ResponseBlockWire::timestamps`](crate::control_block_wire::ResponseBlockWire::timestamps)).
//! The vendor's own runtime consumed exactly this off the same completion path: its
//! `PendingRequest` carried telemetry members filled by `DecodeTimestamps`/`TimestampDecoder` in
//! the telemetry phase, and `ResponseBlockWrapperManager` summarized them
//! (`total_elapsed_cycles()`, `elapsed_power_stamp()`). This module is the scratchy-shaped
//! equivalent: the response worker records one [`RbTiming`] per completed CB into a bounded
//! process-wide ring, and whoever wants the numbers drains it.
//!
//! ALWAYS ON, BY DESIGN: one record is a copy of the decoded response block plus a mutex push on
//! the completion thread, per CB — negligible against the 4–5µs a CB costs to submit, and there is
//! no configuration to get wrong. The consumer decides what the numbers mean; the player drains
//! after the step's `synchronize` and attributes per op with no drain beyond the one the step
//! already performs — the point being that drain-based attribution costs ~11% of decode to
//! estimate what this carries for free.
//!
//! The ring is bounded (drop-oldest at [`CAP`] entries): a consumer that never drains costs a
//! fixed few MB, not unbounded growth. Draining returns entries in completion order;
//! [`RbTiming::id`] is assigned in submission order under the submission mutex, so sorting by it
//! reproduces submission order.

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::control_block_wire::ResponseBlockWire;
use crate::scheduler::PipelineId;

/// One completed CB's timing facts, as hardware wrote them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RbTiming {
    /// The pending request's global id — assigned in submission order under the submission mutex
    /// (`ProcessTags`'s `global_pr_id`), so sorting drained entries by `id` reproduces submission
    /// order regardless of the order the pipelines completed in.
    pub id: u64,
    /// Which pipeline the CB ran on. Compute CBs are the ones a caller can attribute to ops; the
    /// async-DMA pipelines carry the preamble-H2D / logits-D2H crossings, which are their own
    /// per-step cost.
    pub pipeline: PipelineId,
    /// The decoded response block itself — the vendor's `PendingRequest` carried the RB too, and
    /// this way the card-measured phase accessors
    /// ([`ResponseBlockWire::exec_cmpt_cycles`] and siblings) have exactly one definition. This
    /// record stays raw: it names nothing and sums nothing.
    pub rb: ResponseBlockWire,
}

/// The production ring's capacity. A decode step is ~162 CBs and a prefill chunk a few hundred, so
/// this holds hundreds of steps of history — far more than the one step a consumer that drains
/// after every `synchronize` needs, and a fixed ceiling for one that never drains.
const CAP: usize = 1 << 16;

/// The bounded drop-oldest ring itself. Production runs one process-wide instance (below); tests
/// own theirs — the ring is shared state BY DESIGN in production (every completed CB's record
/// funnels into the one instance, including fake CBs any scheduler test completes), which is
/// exactly why two tests reaching for the process-wide ring race each other.
struct RbRing {
    deque: VecDeque<RbTiming>,
    cap: usize,
}

impl RbRing {
    fn with_cap(cap: usize) -> Self {
        Self {
            deque: VecDeque::new(),
            cap,
        }
    }

    fn record(&mut self, timing: RbTiming) {
        if self.deque.len() >= self.cap {
            self.deque.pop_front();
        }
        self.deque.push_back(timing);
    }

    fn drain(&mut self) -> Vec<RbTiming> {
        self.deque.drain(..).collect()
    }
}

fn ring() -> &'static Mutex<RbRing> {
    static RING: std::sync::OnceLock<Mutex<RbRing>> = std::sync::OnceLock::new();
    RING.get_or_init(|| Mutex::new(RbRing::with_cap(CAP)))
}

/// Record one completed CB's timing. Called only from the response worker's completion path, so
/// the lock is uncontended in practice.
pub(crate) fn record(timing: RbTiming) {
    ring()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .record(timing);
}

/// Take every recorded timing, leaving the ring empty. Entries are in completion order; sort by
/// [`RbTiming::id`] for submission order.
pub fn drain() -> Vec<RbTiming> {
    ring().lock().unwrap_or_else(|e| e.into_inner()).drain()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_block_wire::{PW_SECTION, RB_NUM_BYTES, TS_SECTION};

    /// An RB whose TS section and power word decode to exactly `ts`/`power`, built through the
    /// real `from_bytes` so the tests exercise the decode path a completed CB takes.
    fn rb_with(ts: [u32; 5], power: u32) -> ResponseBlockWire {
        let mut b = [0u8; RB_NUM_BYTES];
        for (i, w) in ts.iter().enumerate() {
            b[TS_SECTION + i * 4..TS_SECTION + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        b[PW_SECTION..PW_SECTION + 4].copy_from_slice(&power.to_le_bytes());
        ResponseBlockWire::from_bytes(&b)
    }

    #[test]
    fn record_then_drain_returns_entries_and_empties() {
        let mut ring = RbRing::with_cap(8);
        ring.record(RbTiming {
            id: 1,
            pipeline: PipelineId::Compute,
            rb: rb_with([1, 2, 3, 4, 5], 6),
        });
        ring.record(RbTiming {
            id: 2,
            pipeline: PipelineId::AsyncDmaI,
            rb: rb_with([0; 5], 0),
        });
        let drained = ring.drain();
        assert_eq!(
            drained,
            vec![
                RbTiming {
                    id: 1,
                    pipeline: PipelineId::Compute,
                    rb: rb_with([1, 2, 3, 4, 5], 6),
                },
                RbTiming {
                    id: 2,
                    pipeline: PipelineId::AsyncDmaI,
                    rb: rb_with([0; 5], 0),
                },
            ]
        );
        assert!(ring.drain().is_empty(), "drain must leave the ring empty");
    }

    #[test]
    fn ring_is_bounded_and_drops_oldest() {
        let mut ring = RbRing::with_cap(16);
        for i in 0..24u64 {
            ring.record(RbTiming {
                id: i,
                pipeline: PipelineId::Compute,
                rb: rb_with([0; 5], 0),
            });
        }
        let drained = ring.drain();
        assert_eq!(drained.len(), 16);
        // The OLDEST 8 were evicted: the surviving ids are 8..24.
        assert_eq!(drained.first().unwrap().id, 8);
        assert_eq!(drained.last().unwrap().id, 23);
    }
}
