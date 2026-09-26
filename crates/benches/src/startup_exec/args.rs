// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Arguments for `scr bench startup --exec`.
//!
//! Their own module so the names need no `Startup` prefix to disambiguate: the
//! module path already says which command they belong to, and nothing here is
//! meant to be referred to bare from elsewhere in the crate.

use clap::Parser;

/// How the first token is observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    /// One-shot CLI: the clock stops on the first content byte of stdout.
    Cli,
    /// HTTP server: the clock stops on the first streamed token.
    Server,
}

/// Which rung of the cache ladder a repetition runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Scenario {
    /// Page cache evicted and derived on-disk caches removed.
    Frozen,
    /// Fresh process, caches warm — the honest everyday case.
    Cold,
    /// Process resident and past its first request.
    Warm,
}

/// Which framework the child is, for the one thing that is framework-specific:
/// classifying its stdout banner in CLI mode.
///
/// A closed set rather than a free string, because the failure mode of a typo is
/// silent and wrong rather than loud: `--backend mlx_lm` would fall through to
/// scratchy's rules, the `==========` separator mlx-lm prints would count as
/// content, and the clock would stop on the banner — an impossibly fast TTFT
/// with no error. The child *command* stays free-form (`--child-cmd`); only the
/// banner grammar is enumerated, and adding a framework makes the compiler point
/// at every match that needs an arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Backend {
    /// `Using model: ...` plus ISO-8601 tracing lines.
    Scratchy,
    /// Output delimited by a rule of `=` characters.
    MlxLm,
    /// Server mode only — see `Backend::rejects_cli_mode`.
    Vllm,
}

impl Backend {
    /// Why this backend cannot be measured in CLI mode, if it cannot.
    ///
    /// vLLM has no self-contained one-shot generate to time: its CLI is a client
    /// that needs a server already up, so timing it would measure a warm request
    /// and label it a cold start. Refusing is the same call the harness this
    /// replaced made for ollama — better no number than a number that looks like
    /// a cold start and is not one. vLLM is measured in `--mode server`, where
    /// `--backend` is not consulted at all.
    pub fn rejects_cli_mode(self) -> Option<&'static str> {
        match self {
            Self::Vllm => Some(
                "vLLM has no one-shot CLI generate to time — its CLI is a client for an \
                 already-running server, so timing it would measure a warm request and call \
                 it a cold start. Use `--mode server` (where --backend is not used).",
            ),
            Self::Scratchy | Self::MlxLm => None,
        }
    }
}

/// How to evict the OS page cache for a FROZEN repetition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Evict {
    /// macOS `purge`: drops the whole unified buffer cache, symmetrically for
    /// every framework under test.
    Purge,
    /// Linux `posix_fadvise(DONTNEED)` over `--evict-path`. Deliberately not
    /// `drop_caches`, which is not namespaced and would evict a shared node's
    /// entire page cache.
    Fadvise,
    /// Leave the page cache alone (FROZEN then collapses toward COLD).
    None,
}

/// Exec-boundary mode for `scr bench startup`.
///
/// Without `--exec`, `bench startup` times `LLMBuilder::build()` in-process.
/// With it, the clock starts before `fork`/`exec` and stops on the first token
/// a *child* process emits — which is the only way to see process init, dynamic
/// linking and first-touch page faults, and the only way to measure a framework
/// that is not this binary.
#[derive(Parser, Debug)]
#[command(next_help_heading = "Exec-boundary mode (--exec)")]
pub struct ExecArgs {
    /// Measure exec -> first token across a process boundary instead of timing
    /// in-process engine construction.
    ///
    /// `requires` rather than a runtime check: there is no such thing as this
    /// mode without a child to launch, so clap refuses it up front.
    #[arg(long, requires = "child_cmd")]
    pub exec: bool,

    /// The command to launch, as shell words — same convention as
    /// `scr sweep --serve-cmd`. Any framework works without code changes:
    /// "scr serve MODEL --port 8731" or
    /// "python -m mlx_lm.server --model MODEL --port 8731".
    ///
    /// In CLI mode it must carry `{prompt}` (and usually `{output_len}`),
    /// because each framework spells its prompt flag differently.
    #[arg(long, value_parser = ChildCommand::parse)]
    pub child_cmd: Option<ChildCommand>,

    /// Which framework the child is, for classifying its stdout banner in CLI
    /// mode. Not consulted in server mode.
    #[arg(long, value_enum, default_value_t = Backend::Scratchy)]
    pub backend: Backend,

    /// How the first token is observed.
    #[arg(long, value_enum, default_value_t = Mode::Server)]
    pub mode: Mode,

    /// Rungs to run, in order. Comma-separated: `--scenarios frozen,cold,warm`.
    #[arg(long, value_enum, value_delimiter = ',', default_value = "cold,warm")]
    pub scenarios: Vec<Scenario>,

    /// Repetitions per scenario (warm is always 1 — it is one resident server).
    #[arg(long, default_value_t = 3)]
    pub reps: usize,

    /// Page-cache eviction for FROZEN. Defaults to `purge` on macOS and
    /// `fadvise` elsewhere.
    #[arg(long, value_enum, default_value_t = default_evict())]
    pub evict: Evict,

    /// Files or directories to evict for FROZEN with `--evict fadvise` (the
    /// weight shards and the binary). Directories recurse.
    #[arg(long)]
    pub evict_path: Vec<std::path::PathBuf>,

    /// Directories to delete before a FROZEN repetition (derived caches such as
    /// an aligned-weights sidecar or a `__pycache__`).
    #[arg(long)]
    pub remove_path: Vec<std::path::PathBuf>,

    /// Port the child server listens on.
    #[arg(long, default_value_t = 8731)]
    pub port: u16,

    /// Approximate prompt length in tokens.
    #[arg(long, default_value_t = 64)]
    pub input_len: usize,

    /// Tokens to generate per request.
    #[arg(long, default_value_t = 32)]
    pub output_len: usize,

    /// Base seed. Prompts are unique per seed so no prefix cache can serve a
    /// repeat and collapse TTFT.
    #[arg(long, default_value_t = 1000)]
    pub seed: u64,

    /// Requests to sample for the WARM rung, each with a unique prompt.
    #[arg(long, default_value_t = 20)]
    pub warm_requests: usize,

    /// Seconds to wait after ready before sampling WARM, so lazy
    /// post-ready initialization does not land inside the measurement.
    #[arg(long, default_value_t = 8.0)]
    pub settle_s: f64,

    /// Readiness poll interval. This is the only quantization in `t_ready`.
    #[arg(long, default_value_t = 20)]
    pub poll_interval_ms: u64,

    /// How long to wait for `/v1/models` to answer 200.
    #[arg(long, default_value_t = 600)]
    pub ready_timeout_s: u64,

    /// A second `--child-cmd` to run the blocking parity gate against before
    /// any timing. A broken dequant path can be fast, so timing a wrong
    /// computation is worse than not timing at all.
    #[arg(long, value_parser = ChildCommand::parse, requires = "parity_backend")]
    pub parity_cmd: Option<ChildCommand>,

    /// Which framework `--parity-cmd` is, for its banner rules.
    #[arg(long, value_enum, default_value_t = Backend::MlxLm)]
    pub parity_backend: Backend,
}

/// A child command, already split into argv.
///
/// A newtype rather than a `String` so the shell-words split and the
/// "not empty" invariant are established once, at argument-parse time, instead
/// of being re-derived and re-checked wherever the command is used. The
/// placeholder requirement stays a cross-field check because it depends on
/// `--mode`, which clap cannot express — but `has_placeholder` keeps the
/// question on the type.
#[derive(Clone, Debug)]
pub struct ChildCommand {
    /// As typed, for provenance output.
    pub raw: String,
    /// Split on shell words: `{prompt}` survives as one argv element even when
    /// the prompt contains spaces, because splitting happens before expansion.
    pub argv: Vec<String>,
}

impl ChildCommand {
    fn parse(s: &str) -> Result<Self, String> {
        let argv = shell_words::split(s).map_err(|e| format!("not valid shell words: {e}"))?;
        if argv.is_empty() {
            return Err("command is empty".to_string());
        }
        Ok(Self {
            raw: s.to_string(),
            argv,
        })
    }

    /// Whether the command says where the prompt goes. Required in CLI mode.
    pub fn has_placeholder(&self) -> bool {
        self.raw.contains("{prompt}")
    }

    /// The port the child was told to listen on, when its own argv says.
    ///
    /// Asked so a disagreement with `--port` can be refused. It is only ever a
    /// check, never adopted: a child that takes its port from a config file or
    /// an environment variable has none on its command line, so inferring the
    /// port from argv would silently do nothing in exactly the cases where the
    /// caller most needs to be told.
    pub fn port(&self) -> Option<u16> {
        let mut rest = self.argv.iter();
        while let Some(arg) = rest.next() {
            if let Some(v) = arg.strip_prefix("--port=") {
                return v.parse().ok();
            }
            if arg == "--port" {
                return rest.next()?.parse().ok();
            }
        }
        None
    }
}

/// `purge` is macOS-only and `posix_fadvise` does not exist there, so the
/// right default is platform-dependent.
fn default_evict() -> Evict {
    if cfg!(target_os = "macos") {
        Evict::Purge
    } else {
        Evict::Fadvise
    }
}
