// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Tracing subscriber initialization.
//!
//! Provides a one-shot `init_tracing()` that sets up `tracing_subscriber::fmt`
//! with an `EnvFilter` for controlling log levels.
//!
//! With the `otel` feature enabled, `init_tracing_with_otel()` adds an
//! OpenTelemetry layer that exports spans via OTLP (gRPC or HTTP).

use std::io::IsTerminal;
use std::sync::Once;

use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();

/// Whether the fmt subscriber should emit ANSI color codes.
///
/// `tracing_subscriber::fmt` writes to stdout by default and turns ANSI on
/// unconditionally — it does not check for a terminal. Left alone it litters
/// redirected logs with escape sequences, so gate coloring on stdout being a
/// TTY (matching the banner's stderr check in the serve command).
fn use_ansi() -> bool {
    std::io::stdout().is_terminal()
}

/// Initialize the global tracing subscriber (console logging only).
///
/// Uses `RUST_LOG` env var if set, otherwise falls back to `log_level`.
/// Safe to call multiple times — only the first call takes effect.
pub fn init_tracing(log_level: &str) {
    INIT.call_once(|| {
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));

        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_ansi(use_ansi())
            .compact()
            .with_target(false)
            .init();
    });
}

/// OpenTelemetry configuration.
#[cfg(feature = "otel")]
#[derive(Debug, Clone)]
pub struct OtelConfig {
    /// OTLP endpoint (e.g. "http://localhost:4317" for gRPC).
    pub endpoint: String,
}

/// Build an OTLP span exporter for the given endpoint.
///
/// Separated from `init_tracing_with_otel` for testability.
#[cfg(feature = "otel")]
pub fn build_otlp_exporter(
    endpoint: &str,
) -> Result<opentelemetry_otlp::SpanExporter, opentelemetry_otlp::ExporterBuildError> {
    use opentelemetry_otlp::WithExportConfig;

    opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
}

/// Build a tracer provider with a batch exporter.
///
/// Separated from `init_tracing_with_otel` for testability.
#[cfg(feature = "otel")]
pub fn build_tracer_provider(
    exporter: opentelemetry_otlp::SpanExporter,
) -> opentelemetry_sdk::trace::SdkTracerProvider {
    opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            opentelemetry_sdk::Resource::builder()
                .with_service_name("vllm-rust")
                .build(),
        )
        .build()
}

/// Initialize tracing with both console output and OpenTelemetry export.
///
/// Spans are sent to the OTLP collector at `otel_config.endpoint` via gRPC.
/// Returns a [`OtelGuard`] that flushes pending spans on drop.
///
/// Safe to call multiple times — only the first call takes effect. If called
/// after `init_tracing()`, this is a no-op (the plain subscriber wins).
#[cfg(feature = "otel")]
pub fn init_tracing_with_otel(log_level: &str, otel_config: &OtelConfig) -> Option<OtelGuard> {
    use opentelemetry::trace::TracerProvider as _;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let mut guard = None;

    INIT.call_once(|| {
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));

        // Build the OTLP exporter.
        let exporter = match build_otlp_exporter(&otel_config.endpoint) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("WARNING: failed to create OTLP exporter: {err}");
                // Fall back to plain tracing.
                tracing_subscriber::fmt()
                    .with_env_filter(filter)
                    .with_ansi(use_ansi())
                    .compact()
                    .with_target(false)
                    .init();
                return;
            }
        };

        let provider = build_tracer_provider(exporter);
        let tracer = provider.tracer("vllm");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

        tracing_subscriber::registry()
            .with(filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .with_ansi(use_ansi())
                    .compact()
                    .with_target(false),
            )
            .with(otel_layer)
            .init();

        guard = Some(OtelGuard { provider });
    });

    guard
}

/// RAII guard that shuts down the OpenTelemetry tracer provider on drop,
/// flushing any pending spans.
#[cfg(feature = "otel")]
pub struct OtelGuard {
    provider: opentelemetry_sdk::trace::SdkTracerProvider,
}

#[cfg(feature = "otel")]
impl std::fmt::Debug for OtelGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OtelGuard").finish()
    }
}

#[cfg(feature = "otel")]
impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Err(err) = self.provider.shutdown() {
            eprintln!("WARNING: OpenTelemetry shutdown error: {err}");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_tracing_idempotent() {
        // Multiple calls should not panic.
        init_tracing("warn");
        init_tracing("debug");
    }

    #[cfg(feature = "otel")]
    mod otel_tests {
        use super::*;

        #[test]
        fn test_otel_config_debug() {
            let config = OtelConfig {
                endpoint: "http://localhost:4317".to_string(),
            };
            let debug_str = format!("{config:?}");
            assert!(debug_str.contains("localhost:4317"));
        }

        #[test]
        fn test_otel_config_clone() {
            let config = OtelConfig {
                endpoint: "http://collector:4317".to_string(),
            };
            let cloned = config.clone();
            assert_eq!(config.endpoint, cloned.endpoint);
        }

        #[tokio::test]
        async fn test_build_otlp_exporter_valid_endpoint() {
            // Building an exporter with a valid endpoint should succeed.
            // The exporter is lazy — it doesn't actually connect until spans are sent.
            let result = build_otlp_exporter("http://localhost:4317");
            assert!(result.is_ok(), "expected Ok, got: {result:?}");
        }

        #[tokio::test]
        async fn test_build_tracer_provider() {
            // Build a provider and verify it can create a tracer without panicking.
            use opentelemetry::trace::TracerProvider as _;

            let exporter = build_otlp_exporter("http://localhost:4317").unwrap();
            let provider = build_tracer_provider(exporter);
            let _tracer = provider.tracer("test");

            // Shutdown should succeed even though no spans were sent.
            assert!(provider.shutdown().is_ok());
        }

        #[tokio::test]
        async fn test_otel_guard_drop_calls_shutdown() {
            let exporter = build_otlp_exporter("http://localhost:4317").unwrap();
            let provider = build_tracer_provider(exporter);
            let guard = OtelGuard { provider };
            let debug_str = format!("{guard:?}");
            assert!(debug_str.contains("OtelGuard"));
            // Guard drop will call shutdown — should not panic.
            drop(guard);
        }
    }
}
