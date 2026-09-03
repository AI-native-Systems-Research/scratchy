// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Live per-token sampler tape — the "watch the model think" seam.
//!
//! For each generated token the backend publishes a [`SamplerRecord`]: the
//! sampled token, the top-k candidates it was choosing between (with their raw
//! softmax probabilities), the winning probability, and the full-vocab entropy
//! of the distribution. A UI pulls the latest on its own cadence.
//!
//! Same discipline as [`crate::forward_telemetry`]: recording is gated by a
//! single relaxed atomic ([`is_enabled`](SamplerTelemetry::is_enabled)) that a
//! consumer flips on only while watching, so the sampler kernel skips the extra
//! entropy pass and the candidate spill when nobody is looking. Publish and read
//! both `try_lock`, so neither side ever blocks the other — a gauge, not a
//! stream.
//!
//! Only the non-greedy sampler produces a distribution; on the greedy/argmax
//! path there are no probabilities at all, so [`SamplerRecord::greedy`] is set
//! and the distribution fields are left empty/NaN rather than fabricated.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// One candidate token the model considered, with its raw softmax probability
/// (pre-top-p-cutoff — "what the model thought", so a shown top-k need not
/// sum to 1).
#[derive(Clone, Copy, Debug)]
pub struct SamplerCandidate {
    pub token_id: u32,
    pub prob: f32,
}

/// The sampling distribution for a single generated token.
#[derive(Clone, Debug, Default)]
pub struct SamplerRecord {
    /// The token actually emitted (not necessarily the top candidate — top-p /
    /// temperature can pick a lower-ranked one).
    pub sampled_token_id: u32,
    /// Greedy/argmax path: no distribution exists, so the fields below are unset
    /// (`max_prob`/`entropy_nats` are NaN, `top_k` is empty).
    pub greedy: bool,
    /// Probability of the top candidate — the model's confidence this step.
    pub max_prob: f32,
    /// Full-vocab Shannon entropy in nats (UI converts to bits for display).
    pub entropy_nats: f32,
    /// Candidates sorted by probability, descending.
    pub top_k: Vec<SamplerCandidate>,
}

/// Process-global publish point for the live sampler tape.
pub struct SamplerTelemetry {
    enabled: AtomicBool,
    seq: AtomicU64,
    latest: Mutex<Option<Arc<SamplerRecord>>>,
}

impl SamplerTelemetry {
    /// The process-wide instance. Writers and readers in any crate share it.
    pub fn global() -> &'static SamplerTelemetry {
        static G: OnceLock<SamplerTelemetry> = OnceLock::new();
        G.get_or_init(|| SamplerTelemetry {
            enabled: AtomicBool::new(false),
            seq: AtomicU64::new(0),
            latest: Mutex::new(None),
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
    }

    /// Publish the distribution for the token that was just sampled.
    ///
    /// Never blocks: if the reader momentarily holds the slot the publish is
    /// skipped and the next token publishes instead.
    pub fn publish(&self, record: SamplerRecord) {
        if let Ok(mut slot) = self.latest.try_lock() {
            *slot = Some(Arc::new(record));
            self.seq.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The most recently published token, if any. Never blocks a publisher.
    pub fn latest(&self) -> Option<Arc<SamplerRecord>> {
        self.latest.try_lock().ok()?.clone()
    }

    /// Monotonic count of published tokens — lets a UI detect a fresh step.
    pub fn seq(&self) -> u64 {
        self.seq.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_by_default_and_toggles() {
        let t = SamplerTelemetry::global();
        assert!(!t.is_enabled());
        t.set_enabled(true);
        assert!(t.is_enabled());
        t.set_enabled(false);
        assert!(!t.is_enabled());
    }

    #[test]
    fn publish_then_read_latest() {
        let t = SamplerTelemetry::global();
        let before = t.seq();
        t.publish(SamplerRecord {
            sampled_token_id: 42,
            greedy: false,
            max_prob: 0.61,
            entropy_nats: 1.47,
            top_k: vec![
                SamplerCandidate {
                    token_id: 42,
                    prob: 0.61,
                },
                SamplerCandidate {
                    token_id: 7,
                    prob: 0.18,
                },
            ],
        });
        assert_eq!(t.seq(), before + 1);
        let rec = t.latest().expect("published record");
        assert_eq!(rec.sampled_token_id, 42);
        assert_eq!(rec.top_k.len(), 2);
        assert!((rec.max_prob - 0.61).abs() < 1e-6);
    }
}
