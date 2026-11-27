//! The tree's one client-TLS door.
//!
//! Every ureq agent in the workspace takes its TLS config from here, so the
//! client side of the tree — model downloads, `chat --url`, bench dataset
//! fetches, the RAG embedding POST, the weight prober, the Hub-registry build
//! script — runs rustls on the pure-Rust RustCrypto provider and compiles no
//! C. The workspace's `ureq` is built `rustls-no-provider`: without this
//! config, an agent panics on its first HTTPS call rather than silently
//! falling back to `ring`, so a new door must come through [`tls_config`] or
//! [`builder`] — there is no default provider to forget to disable.
//!
//! The trade, accepted for these non-critical paths (user order 2026-11-27):
//! `rustls-rustcrypto` is a pre-release, unaudited provider — "DO NOT USE IN
//! PRODUCTION" upstream — in exchange for a `cc`-free build. The *server*
//! side is a different door and keeps `ring`: `scratchy-serving-api`'s `tls`
//! feature (server.rs) terminates production TLS and is not covered by that
//! acceptance.
//!
//! ureq's provider setter is documented UNSTABLE API (it wraps not-yet-1.x
//! rustls); isolating it here means an upstream rename breaks this one file,
//! not every door.

/// A `TlsConfig` with the RustCrypto provider installed — what every ureq
/// agent in the tree builds on.
pub fn tls_config() -> ureq::tls::TlsConfig {
    builder().build()
}

/// A `TlsConfigBuilder` with the RustCrypto provider installed, for callers
/// that adjust the rest of the config (e.g. `disable_verification` for the
/// bench harness's loopback test server).
pub fn builder() -> ureq::tls::TlsConfigBuilder {
    ureq::tls::TlsConfig::builder()
        .unversioned_rustls_crypto_provider(rustls_rustcrypto::provider().into())
}
