// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Live forward-pass kernel tape — a lock-free, opt-in observability seam.
//!
//! A backend (e.g. the Metal interpreter) records the *sequence of GPU
//! kernels it dispatches* for one full model forward into a [`ForwardRecord`]
//! and [`publish`](ForwardTelemetry::publish)es it once per token. A UI pulls
//! the latest record on its own cadence via [`latest`](ForwardTelemetry::latest).
//!
//! The forward pass is sacred: recording is gated by a single relaxed atomic
//! ([`is_enabled`](ForwardTelemetry::is_enabled)) that a consumer flips on only
//! while it is actually watching. When disabled the cost is one atomic load per
//! forward. The publish and read both use `try_lock`, so neither side ever
//! blocks the other — a missed publish just means the reader sees the previous
//! forward for one more tick (this is a gauge, not a stream).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Coarse family a dispatched kernel belongs to, used to color the live tape.
///
/// Deliberately arch-agnostic: each backend maps its own kernel identifiers
/// onto these families, so the UI never depends on backend internals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelKind {
    Embed,
    Norm,
    Rope,
    Attention,
    GatedDeltaNet,
    Gemm,
    Moe,
    Mlp,
    Sample,
    Vision,
    Elementwise,
    Other,
}

impl KernelKind {
    /// All families, in a stable order suitable for a legend.
    pub const ALL: [KernelKind; 12] = [
        KernelKind::Embed,
        KernelKind::Norm,
        KernelKind::Rope,
        KernelKind::Attention,
        KernelKind::GatedDeltaNet,
        KernelKind::Gemm,
        KernelKind::Moe,
        KernelKind::Mlp,
        KernelKind::Sample,
        KernelKind::Vision,
        KernelKind::Elementwise,
        KernelKind::Other,
    ];

    /// Short human label for legends.
    pub fn label(self) -> &'static str {
        match self {
            KernelKind::Embed => "embed",
            KernelKind::Norm => "norm",
            KernelKind::Rope => "rope",
            KernelKind::Attention => "attn",
            KernelKind::GatedDeltaNet => "Δnet",
            KernelKind::Gemm => "gemm",
            KernelKind::Moe => "moe",
            KernelKind::Mlp => "mlp",
            KernelKind::Sample => "head",
            KernelKind::Vision => "vis",
            KernelKind::Elementwise => "elem",
            KernelKind::Other => "other",
        }
    }
}

/// One dispatched kernel on the tape.
///
/// Beyond its [`KernelKind`] it carries two forward-compiler decisions: whether
/// a `Dispatch → Dispatch` barrier was emitted before it (the compiler's hazard
/// analysis), and whether it is a fused kernel (the compiler folded several ops
/// into one dispatch).
#[derive(Clone, Copy, Debug)]
pub struct TapeEntry {
    pub kind: KernelKind,
    pub barrier: bool,
    pub fused: bool,
}

/// The kernel-dispatch tape of a single forward pass.
///
/// `tape[i]` is the i-th kernel the backend actually dispatched (gated/skipped
/// dispatches are not recorded), so its length is the real dispatch count and
/// its shape reveals the per-layer rhythm of the forward.
#[derive(Clone, Debug, Default)]
pub struct ForwardRecord {
    /// Padded token width of the bucket this forward ran (1 for decode).
    pub num_tokens: u32,
    /// Sequences in the batch.
    pub num_seqs: u32,
    /// One entry per dispatched kernel, in dispatch order.
    pub tape: Vec<TapeEntry>,
}

/// Process-global publish point for the live forward tape.
pub struct ForwardTelemetry {
    enabled: AtomicBool,
    seq: AtomicU64,
    latest: Mutex<Option<Arc<ForwardRecord>>>,
}

impl ForwardTelemetry {
    /// The process-wide instance. Writers and readers in any crate share it.
    pub fn global() -> &'static ForwardTelemetry {
        static G: OnceLock<ForwardTelemetry> = OnceLock::new();
        G.get_or_init(|| ForwardTelemetry {
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

    /// Publish the tape for the forward that just finished encoding.
    ///
    /// Never blocks the caller: if the reader momentarily holds the slot the
    /// publish is skipped and the next token's forward publishes instead.
    pub fn publish(&self, record: ForwardRecord) {
        if let Ok(mut slot) = self.latest.try_lock() {
            *slot = Some(Arc::new(record));
            self.seq.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The most recently published forward, if any. Never blocks a publisher.
    pub fn latest(&self) -> Option<Arc<ForwardRecord>> {
        self.latest.try_lock().ok()?.clone()
    }

    /// Monotonic count of published forwards — lets a UI detect a fresh tape.
    pub fn seq(&self) -> u64 {
        self.seq.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_by_default_and_toggles() {
        let t = ForwardTelemetry::global();
        assert!(!t.is_enabled());
        t.set_enabled(true);
        assert!(t.is_enabled());
        t.set_enabled(false);
        assert!(!t.is_enabled());
    }

    #[test]
    fn publish_then_read_latest() {
        let t = ForwardTelemetry::global();
        let before = t.seq();
        let entry = |kind| TapeEntry {
            kind,
            barrier: false,
            fused: false,
        };
        t.publish(ForwardRecord {
            num_tokens: 1,
            num_seqs: 1,
            tape: vec![
                entry(KernelKind::Norm),
                entry(KernelKind::Attention),
                entry(KernelKind::Gemm),
            ],
        });
        assert_eq!(t.seq(), before + 1);
        let rec = t.latest().expect("published record");
        assert_eq!(rec.tape.len(), 3);
        assert_eq!(rec.tape[1].kind, KernelKind::Attention);
    }

    #[test]
    fn all_labels_present() {
        for k in KernelKind::ALL {
            assert!(!k.label().is_empty());
        }
    }
}
