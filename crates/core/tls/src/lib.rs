//! The tree's one client-TLS door, and the only place a ureq agent is born.
//!
//! Every HTTPS client in the workspace — model downloads, `chat --url`, bench
//! dataset fetches, the RAG embedding POST, the weight prober, the Hub-registry
//! build script — builds its agent from [`agent_config`], so the client side of
//! the tree runs rustls on the pure-Rust RustCrypto provider and compiles no
//! C. The workspace's `ureq` is built `rustls-no-provider`: without this
//! config an agent would panic on its first HTTPS call, so construction itself
//! is owned here — root `clippy.toml` bans `ureq::Agent::config_builder`, the
//! one-shot `ureq::get`/`post`/… verbs, `ureq::agent`/`run` and
//! `TlsConfig::builder` outside this crate (this crate carries its own
//! `clippy.toml` so its one construction site is exempt without an
//! `#[allow]`), and CI runs clippy with `-D warnings`: a new door that skips
//! this crate is a build failure, not a field panic.
//!
//! The trade, accepted for these non-critical paths (user order 2026-11-27):
//! `rustls-rustcrypto` is a pre-release, unaudited provider — "DO NOT USE IN
//! PRODUCTION" upstream — in exchange for a `cc`-free build. What crosses it,
//! stated plainly: bearer credentials — the Hub downloader's and the registry
//! build script's `HF_TOKEN`, and the RAG embedding client's
//! `VLLM_EMBEDDING_API_KEY` — plus the weights and dataset bytes themselves.
//! The *server* side is a different door and keeps `ring`:
//! `scratchy-serving-api`'s `tls` feature (server.rs) terminates production
//! TLS and is not covered by that acceptance.
//!
//! ureq's provider setter is documented UNSTABLE API (it wraps not-yet-1.x
//! rustls); isolating it here means an upstream rename breaks this one file,
//! not every door.

/// An agent config with the tree's TLS door already applied. Callers chain
/// their own timeouts, redirect limits and pooling onto this — nothing else.
pub fn agent_config() -> ureq::config::ConfigBuilder<ureq::typestate::AgentScope> {
    ureq::Agent::config_builder().tls_config(tls_config())
}

/// The same config with certificate verification disabled, for the bench
/// harness's loopback test server only. The handshake still runs on the
/// RustCrypto provider — `rustls-no-provider` leaves no default to fall back
/// to.
pub fn insecure_agent_config() -> ureq::config::ConfigBuilder<ureq::typestate::AgentScope> {
    ureq::Agent::config_builder().tls_config(builder().disable_verification(true).build())
}

fn tls_config() -> ureq::tls::TlsConfig {
    builder().build()
}

fn builder() -> ureq::tls::TlsConfigBuilder {
    ureq::tls::TlsConfig::builder()
        .unversioned_rustls_crypto_provider(rustls_rustcrypto::provider().into())
}
