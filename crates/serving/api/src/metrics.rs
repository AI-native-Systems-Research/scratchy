// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Prometheus metrics for vLLM serving layer.
//!
//! Provides a global `VllmMetrics` singleton with counters and histograms
//! for request tracking, latency, and cache usage.

use std::sync::OnceLock;

use prometheus::{
    Gauge, Histogram, HistogramOpts, IntCounter, IntGauge, Registry, register_gauge_with_registry,
    register_histogram_with_registry, register_int_counter_with_registry,
    register_int_gauge_with_registry,
};

static METRICS: OnceLock<VllmMetrics> = OnceLock::new();

/// Global metrics for the vLLM serving layer.
pub struct VllmMetrics {
    /// Prometheus registry holding all metrics.
    pub registry: Registry,

    // -- Request counters --
    /// Total number of requests received.
    pub requests_total: IntCounter,
    /// Total number of successful requests.
    pub requests_success_total: IntCounter,
    /// Total number of failed requests.
    pub requests_failed_total: IntCounter,
    /// Number of currently active (in-flight) requests.
    pub requests_active: IntGauge,

    // -- Latency histograms --
    /// End-to-end request latency in seconds.
    pub request_latency_seconds: Histogram,
    /// Time to first token in seconds.
    pub time_to_first_token_seconds: Histogram,
    /// Inter-token latency in seconds.
    pub inter_token_latency_seconds: Histogram,

    // -- Token counters --
    /// Total number of output tokens generated.
    pub output_tokens_total: IntCounter,
    /// Total number of prompt tokens processed.
    pub prompt_tokens_total: IntCounter,

    // -- Scheduler gauges --
    /// Number of requests currently running.
    pub num_requests_running: Gauge,
    /// Number of requests waiting to be scheduled.
    pub num_requests_waiting: Gauge,

    // -- Cache gauges --
    /// KV cache usage as a fraction (0.0 - 1.0).
    pub kv_cache_usage_perc: Gauge,
    /// Number of GPU KV cache blocks in use.
    pub gpu_cache_blocks_used: IntGauge,
    /// Total number of GPU KV cache blocks.
    pub gpu_cache_blocks_total: IntGauge,
    /// Number of blocks retained in the prefix cache.
    pub prefix_cache_blocks: IntGauge,

    // -- Speculative decoding counters --
    /// Total number of spec decode draft rounds.
    pub spec_decode_num_drafts: IntCounter,
    /// Total number of draft tokens proposed.
    pub spec_decode_num_draft_tokens: IntCounter,
    /// Total number of draft tokens accepted.
    pub spec_decode_num_accepted_tokens: IntCounter,
}

impl VllmMetrics {
    /// Get or initialize the global metrics singleton.
    pub fn global() -> &'static VllmMetrics {
        METRICS.get_or_init(|| {
            let registry = Registry::new_custom(Some("vllm".to_string()), None)
                .expect("failed to create prometheus registry");

            let requests_total = register_int_counter_with_registry!(
                "requests_total",
                "Total number of requests received",
                registry
            )
            .unwrap();

            let requests_success_total = register_int_counter_with_registry!(
                "requests_success_total",
                "Total number of successful requests",
                registry
            )
            .unwrap();

            let requests_failed_total = register_int_counter_with_registry!(
                "requests_failed_total",
                "Total number of failed requests",
                registry
            )
            .unwrap();

            let requests_active = register_int_gauge_with_registry!(
                "requests_active",
                "Number of currently active requests",
                registry
            )
            .unwrap();

            let request_latency_seconds = register_histogram_with_registry!(
                HistogramOpts::new(
                    "request_latency_seconds",
                    "End-to-end request latency in seconds",
                )
                .buckets(vec![
                    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
                ]),
                registry
            )
            .unwrap();

            let time_to_first_token_seconds = register_histogram_with_registry!(
                HistogramOpts::new(
                    "time_to_first_token_seconds",
                    "Time to first token in seconds",
                )
                .buckets(vec![
                    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
                ]),
                registry
            )
            .unwrap();

            let inter_token_latency_seconds = register_histogram_with_registry!(
                HistogramOpts::new(
                    "inter_token_latency_seconds",
                    "Inter-token latency in seconds",
                )
                .buckets(vec![
                    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
                ]),
                registry
            )
            .unwrap();

            let output_tokens_total = register_int_counter_with_registry!(
                "output_tokens_total",
                "Total number of output tokens generated",
                registry
            )
            .unwrap();

            let prompt_tokens_total = register_int_counter_with_registry!(
                "prompt_tokens_total",
                "Total number of prompt tokens processed",
                registry
            )
            .unwrap();

            let num_requests_running = register_gauge_with_registry!(
                "num_requests_running",
                "Number of requests currently running",
                registry
            )
            .unwrap();

            let num_requests_waiting = register_gauge_with_registry!(
                "num_requests_waiting",
                "Number of requests waiting to be scheduled",
                registry
            )
            .unwrap();

            let kv_cache_usage_perc = register_gauge_with_registry!(
                "gpu_cache_usage_perc",
                "KV cache usage as a fraction (0.0 - 1.0)",
                registry
            )
            .unwrap();

            let gpu_cache_blocks_used = register_int_gauge_with_registry!(
                "gpu_cache_blocks_used",
                "Number of GPU KV cache blocks in use",
                registry
            )
            .unwrap();

            let gpu_cache_blocks_total = register_int_gauge_with_registry!(
                "gpu_cache_blocks_total",
                "Total number of GPU KV cache blocks",
                registry
            )
            .unwrap();

            let prefix_cache_blocks = register_int_gauge_with_registry!(
                "prefix_cache_blocks",
                "Number of blocks retained in the prefix cache",
                registry
            )
            .unwrap();

            let spec_decode_num_drafts = register_int_counter_with_registry!(
                "spec_decode_num_drafts",
                "Total number of spec decode draft rounds",
                registry
            )
            .unwrap();

            let spec_decode_num_draft_tokens = register_int_counter_with_registry!(
                "spec_decode_num_draft_tokens",
                "Total number of draft tokens proposed",
                registry
            )
            .unwrap();

            let spec_decode_num_accepted_tokens = register_int_counter_with_registry!(
                "spec_decode_num_accepted_tokens",
                "Total number of draft tokens accepted",
                registry
            )
            .unwrap();

            VllmMetrics {
                registry,
                requests_total,
                requests_success_total,
                requests_failed_total,
                requests_active,
                request_latency_seconds,
                time_to_first_token_seconds,
                inter_token_latency_seconds,
                output_tokens_total,
                prompt_tokens_total,
                num_requests_running,
                num_requests_waiting,
                kv_cache_usage_perc,
                gpu_cache_blocks_used,
                gpu_cache_blocks_total,
                prefix_cache_blocks,
                spec_decode_num_drafts,
                spec_decode_num_draft_tokens,
                spec_decode_num_accepted_tokens,
            }
        })
    }

    /// Encode all metrics in Prometheus text format.
    pub fn encode(&self) -> String {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_singleton() {
        let m1 = VllmMetrics::global();
        let m2 = VllmMetrics::global();
        // Same pointer.
        assert!(std::ptr::eq(m1, m2));
    }

    #[test]
    fn test_counter_increment() {
        let m = VllmMetrics::global();
        let before = m.requests_total.get();
        m.requests_total.inc();
        assert_eq!(m.requests_total.get(), before + 1);
    }

    #[test]
    fn test_gauge_set() {
        let m = VllmMetrics::global();
        m.requests_active.set(5);
        assert_eq!(m.requests_active.get(), 5);
        m.requests_active.set(0);
    }

    #[test]
    fn test_encode_produces_text() {
        let m = VllmMetrics::global();
        m.requests_total.inc();
        let text = m.encode();
        assert!(text.contains("vllm_requests_total"));
    }
}
