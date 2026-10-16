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
//! ALWAYS ON, BY DESIGN: one record is a copy of five u32s and a power word plus a mutex push on
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
    /// The five TS words, raw and unnamed
    /// ([`ResponseBlockWire::timestamps`](crate::control_block_wire::ResponseBlockWire::timestamps)
    /// — the slot-to-name mapping is a card measurement, not an assumption).
    pub timestamps: [u32; 5],
    /// The power word.
    pub power: u32,
}

/// The ring's capacity. A decode step is ~162 CBs and a prefill chunk a few hundred, so this holds
/// hundreds of steps of history — far more than the one step a consumer that drains after every
/// `synchronize` needs, and a fixed ceiling for one that never drains.
const CAP: usize = 1 << 16;

fn ring() -> &'static Mutex<VecDeque<RbTiming>> {
    static RING: std::sync::OnceLock<Mutex<VecDeque<RbTiming>>> = std::sync::OnceLock::new();
    RING.get_or_init(|| Mutex::new(VecDeque::new()))
}

/// Record one completed CB's timing. Called only from the response worker's completion path, so
/// the lock is uncontended in practice.
pub(crate) fn record(timing: RbTiming) {
    let mut ring = ring().lock().unwrap_or_else(|e| e.into_inner());
    if ring.len() >= CAP {
        ring.pop_front();
    }
    ring.push_back(timing);
}

/// Take every recorded timing, leaving the ring empty. Entries are in completion order; sort by
/// [`RbTiming::id`] for submission order.
pub fn drain() -> Vec<RbTiming> {
    let mut ring = ring().lock().unwrap_or_else(|e| e.into_inner());
    ring.drain(..).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_then_drain_returns_entries_and_empties() {
        let first = drain();
        assert!(first.is_empty(), "test requires an empty ring to start");
        record(RbTiming {
            id: 1,
            pipeline: PipelineId::Compute,
            timestamps: [1, 2, 3, 4, 5],
            power: 6,
        });
        record(RbTiming {
            id: 2,
            pipeline: PipelineId::AsyncDmaI,
            timestamps: [0; 5],
            power: 0,
        });
        let drained = drain();
        assert_eq!(
            drained,
            vec![
                RbTiming {
                    id: 1,
                    pipeline: PipelineId::Compute,
                    timestamps: [1, 2, 3, 4, 5],
                    power: 6
                },
                RbTiming {
                    id: 2,
                    pipeline: PipelineId::AsyncDmaI,
                    timestamps: [0; 5],
                    power: 0
                },
            ]
        );
        assert!(drain().is_empty(), "drain must leave the ring empty");
    }

    #[test]
    fn ring_is_bounded_and_drops_oldest() {
        assert!(drain().is_empty(), "test requires an empty ring to start");
        for i in 0..(CAP as u64 + 8) {
            record(RbTiming {
                id: i,
                pipeline: PipelineId::Compute,
                timestamps: [0; 5],
                power: 0,
            });
        }
        let drained = drain();
        assert_eq!(drained.len(), CAP);
        // The OLDEST 8 were evicted: the surviving ids are 8..CAP+8.
        assert_eq!(drained.first().unwrap().id, 8);
        assert_eq!(drained.last().unwrap().id, CAP as u64 + 7);
    }
}
