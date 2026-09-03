// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! ORCA (Open Request Cost Aggregation) load-reporting headers.
//!
//! Smart HTTP load balancers (e.g., Envoy) use ORCA to route traffic away
//! from overloaded backends. When a client sends an
//! `endpoint-load-metrics-format` request header, the server attaches an
//! `endpoint-load-metrics` response header with current load stats.

use crate::metrics::VllmMetrics;

/// The request header that clients send to opt-in to ORCA reporting.
pub const ORCA_REQUEST_HEADER: &str = "endpoint-load-metrics-format";

/// The response header attached when the client opts in.
pub const ORCA_RESPONSE_HEADER: &str = "endpoint-load-metrics";

/// Build the ORCA response header value for the given format.
///
/// Returns `Some((header_name, header_value))` for "TEXT" or "JSON" formats,
/// or `None` for unsupported formats.
pub fn orca_header(format: &str) -> Option<(String, String)> {
    let m = VllmMetrics::global();
    format_orca(
        format,
        m.kv_cache_usage_perc.get(),
        m.num_requests_waiting.get(),
    )
}

/// Format ORCA header from raw metric values (pure function, no global state).
fn format_orca(format: &str, kv: f64, waiting: f64) -> Option<(String, String)> {
    let value = match format.to_ascii_uppercase().as_str() {
        "TEXT" => format!(
            "named_metrics.kv_cache_usage_perc={kv},named_metrics.num_requests_waiting={waiting}"
        ),
        "JSON" => format!(
            "{{\"named_metrics\":{{\"kv_cache_usage_perc\":{kv},\"num_requests_waiting\":{waiting}}}}}"
        ),
        _ => return None,
    };

    Some((ORCA_RESPONSE_HEADER.to_string(), value))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orca_text_format() {
        let result = format_orca("TEXT", 0.0, 0.0);
        assert!(result.is_some());
        let (name, value) = result.unwrap();
        assert_eq!(name, ORCA_RESPONSE_HEADER);
        assert!(value.contains("kv_cache_usage_perc="));
        assert!(value.contains("num_requests_waiting="));
    }

    #[test]
    fn test_orca_json_format() {
        let (_, value) = format_orca("JSON", 0.0, 0.0).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&value).unwrap();
        assert!(parsed["named_metrics"]["kv_cache_usage_perc"].is_number());
        assert!(parsed["named_metrics"]["num_requests_waiting"].is_number());
    }

    #[test]
    fn test_orca_case_insensitive() {
        assert!(format_orca("text", 0.0, 0.0).is_some());
        assert!(format_orca("json", 0.0, 0.0).is_some());
        assert!(format_orca("Json", 0.0, 0.0).is_some());
    }

    #[test]
    fn test_orca_unsupported_format() {
        assert!(format_orca("BINARY", 0.0, 0.0).is_none());
        assert!(format_orca("", 0.0, 0.0).is_none());
    }

    #[test]
    fn test_orca_with_nonzero_values() {
        let (_, value) = format_orca("TEXT", 0.75, 3.0).unwrap();
        assert!(value.contains("kv_cache_usage_perc=0.75"));
        assert!(value.contains("num_requests_waiting=3"));

        let (_, value) = format_orca("JSON", 0.75, 3.0).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&value).unwrap();
        assert_eq!(parsed["named_metrics"]["kv_cache_usage_perc"], 0.75);
        assert_eq!(parsed["named_metrics"]["num_requests_waiting"], 3.0);
    }

    #[test]
    fn test_orca_header_returns_some() {
        // Integration test: orca_header reads from global metrics and produces output.
        // Does not assert exact values since other tests may mutate shared gauges.
        assert!(orca_header("TEXT").is_some());
        assert!(orca_header("JSON").is_some());
        assert!(orca_header("BINARY").is_none());
    }
}
