// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench startup --exec` — startup latency measured across a process
//! boundary, for any backend.
//!
//! The default (in-process) mode of `bench startup` times
//! `LLMBuilder::build()` inside this process. That cannot answer "how long from
//! launch until the user sees a word": it never execs, so it misses process
//! init, dynamic linking and first-touch page faults, and it never generates,
//! so it has no first token to stop a clock on.
//!
//! This mode holds the clock itself. It starts immediately before `fork`/`exec`
//! and stops on the first *content* byte of the first token — one span, not a
//! sum of two separately-measured ones. There is exactly ONE implementation of
//! that stopwatch and every backend goes through it, which is the fairness
//! guarantee: nothing is self-reported. The backend is named by
//! `--child-cmd "<command>"`, the same shell-words convention `scr sweep` uses
//! for `--serve-cmd`/`--bench-cmd`, so comparing against mlx-lm, vLLM or
//! anything else that speaks OpenAI-compatible HTTP needs no code here.
//!
//! THE CACHE LADDER, which is the whole point of the scenario flag:
//!
//! | surface                     | FROZEN   | COLD    | WARM              |
//! |-----------------------------|----------|---------|-------------------|
//! | OS page cache (weights/bin) | evicted  | warm    | warm              |
//! | derived on-disk caches      | removed  | present | present           |
//! | process                     | fresh    | fresh   | resident, ≥1 req  |
//!
//! FROZEN must fault far more than COLD or the eviction did not take effect and
//! the run is void — `major_faults` is the evidence, and the validity check at
//! the end of a run asserts it rather than leaving it to a reader.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::args::{BenchStartupArgs, EvictStrategy, StartupMode, StartupScenario};

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------
// A seeded word salad. Two properties matter and neither is "realistic text":
//   1. DETERMINISTIC given the seed, so a rerun measures the same work.
//   2. UNIQUE per seed, so no prefix cache can serve a later request and
//      collapse TTFT to ~0. That failure mode is not hypothetical: a run of
//      `bench serve` against a shared-prefix dataset reported a 98% prefix
//      cache hit rate, which inflated throughput 1.84x and understated TTFT
//      11x before it was caught.
const WORDS: &str = "harbor lantern gravel meadow cinder quartz plateau bramble thicket ember \
     sparrow willow basalt cobalt drifting shallow ridge canyon tundra fjord \
     marble copper silent hollow amber jasper cedar frost pebble current \
     glacier summit orchard beacon compass anchor rudder mariner tempest \
     monsoon zephyr equinox solstice meridian latitude sextant almanac";

fn build_prompt(seed: u64, approx_tokens: usize) -> String {
    let words: Vec<&str> = WORDS.split_whitespace().collect();
    let mut rng = StdRng::seed_from_u64(seed);
    // ~2.2 tokens per word for this salad. Approximate by design: fairness
    // needs both backends to see byte-identical text, not an exact length.
    let n = ((approx_tokens as f64 / 2.2).round() as usize).max(4);
    (0..n)
        .map(|_| words[rng.random_range(0..words.len())])
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// CLI-mode stdout classification
// ---------------------------------------------------------------------------
// In CLI mode the first content byte is the first token, but every CLI prints
// a banner first. This decides "is this line still banner?" so the clock stops
// on a token rather than a header.
fn is_prelude(backend: &str, line: &str) -> bool {
    let s = line.trim();
    if s.is_empty() {
        return true;
    }
    match backend {
        // mlx_lm.generate delimits its output with a rule of '=' characters.
        "mlx-lm" => s.chars().all(|c| c == '='),
        // scratchy prints "Using model: ..." and, if RUST_LOG was not
        // silenced, ISO-8601 tracing lines.
        _ => {
            s.starts_with("Using model:")
                || (s.len() > 20
                    && s.as_bytes()[..4].iter().all(u8::is_ascii_digit)
                    && s.as_bytes()[4] == b'-')
        }
    }
}

// ---------------------------------------------------------------------------
// Child resource accounting
// ---------------------------------------------------------------------------
/// One child's own resource usage, as reported when it was reaped.
pub(crate) struct ChildUsage {
    /// Peak resident set of THAT child.
    pub peak_rss_mib: f64,
    /// Major (disk-backed) faults that child took. The evidence an eviction
    /// actually happened.
    pub major_faults: i64,
    /// Whether it exited 0.
    pub ok: bool,
}

/// Reap `child` with `wait4(2)` so the usage belongs to *that* child.
///
/// `getrusage(RUSAGE_CHILDREN)` cannot do this, and the difference is not
/// academic: its `ru_maxrss` is a high-water mark over every child the process
/// has ever reaped, so a per-repetition delta silently breaks after the first
/// rep — one earlier, larger child hides every later one. A Metal run reported
/// a WARM server at 1 MiB for exactly that reason, because an earlier COLD rep
/// had already pushed the mark to ~270 MiB. `wait4` returns the usage of the
/// single child that exited, which is what the report claims to show, and it
/// makes the fault count per-rep rather than a difference of running totals.
///
/// `std::process::Child`'s `Drop` does not reap on Unix, so reaping here does
/// not collide with it — but nothing may call `Child::kill` afterwards, since
/// the pid is free to be recycled. Kill first, then reap.
fn wait4_child(child: &mut Child) -> Result<ChildUsage> {
    let pid = child.id() as libc::pid_t;
    let mut status: libc::c_int = 0;
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `pid` is a live (possibly already-exited but unreaped) child of
    // this process; `status` and `ru` are initialized locals that wait4 only
    // writes through and does not retain.
    let rc = unsafe { libc::wait4(pid, &mut status, 0, &mut ru) };
    anyhow::ensure!(rc == pid, "wait4({pid}) returned {rc}");
    let ok = libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0;
    Ok(ChildUsage {
        peak_rss_mib: rss_to_mib(ru.ru_maxrss as i64),
        major_faults: ru.ru_majflt as i64,
        ok,
    })
}

// ---------------------------------------------------------------------------
// Cache-state control
// ---------------------------------------------------------------------------
/// Evict the page cache so a FROZEN launch faults its weights back in.
///
/// Two strategies, because the honest mechanism differs by platform:
///
/// * `purge` (macOS) drops the whole unified buffer cache. It is symmetric —
///   it evicts CPython and a framework's dylibs exactly as it evicts the
///   scratchy binary — which is what makes a cross-framework FROZEN fair.
/// * `fadvise` (Linux) calls `posix_fadvise(POSIX_FADV_DONTNEED)` on the named
///   paths only. Deliberately NOT `drop_caches`: that file is not namespaced,
///   so writing it from a container evicts the *host's* entire page cache and
///   would perturb every other workload on a shared node. fadvise is
///   unprivileged and surgical, and read-only mmap'd weight shards are exactly
///   the clean-page case where DONTNEED is reliable.
fn evict(strategy: EvictStrategy, paths: &[PathBuf]) -> Result<()> {
    match strategy {
        EvictStrategy::None => Ok(()),
        EvictStrategy::Purge => {
            // `purge` is root-only ("Unable to purge disk buffers: Operation not
            // permitted" otherwise), so it goes through sudo — and through
            // `-n`, so a missing credential fails immediately instead of
            // blocking a benchmark on a password prompt. `preflight_evict`
            // below checks for the credential before any measurement starts.
            let st = Command::new("sudo")
                .args(["-n", "purge"])
                .status()
                .context("failed to run `sudo -n purge`")?;
            anyhow::ensure!(
                st.success(),
                "`sudo -n purge` exited with {st}. purge needs root and macOS has no \
                 unprivileged equivalent — run `sudo -v` first to cache the credential, \
                 or pass `--evict none` (FROZEN then collapses toward COLD)"
            );
            Ok(())
        }
        EvictStrategy::Fadvise => {
            anyhow::ensure!(
                cfg!(target_os = "linux"),
                "--evict fadvise needs Linux (macOS has no posix_fadvise); use --evict purge"
            );
            anyhow::ensure!(
                !paths.is_empty(),
                "--evict fadvise needs --evict-path (the weight shards and the binary); \
                 without paths it would silently evict nothing and FROZEN would be a lie"
            );
            for p in paths {
                fadvise_dontneed(p)
                    .with_context(|| format!("fadvise DONTNEED failed for {}", p.display()))?;
            }
            Ok(())
        }
    }
}

/// Check the eviction mechanism will work *before* anything is measured.
///
/// A FROZEN run that discovers its eviction is unusable has already thrown away
/// the cache state it needed, so the failure has to come first. This mirrors the
/// "authorizing once up front" step in the shell harness this replaces.
fn preflight_evict(strategy: EvictStrategy, paths: &[PathBuf]) -> Result<()> {
    match strategy {
        EvictStrategy::None => Ok(()),
        EvictStrategy::Purge => {
            let ok = Command::new("sudo")
                .args(["-n", "true"])
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            anyhow::ensure!(
                ok,
                "FROZEN with `--evict purge` needs a cached sudo credential: `purge` requires \
                 root and macOS has no unprivileged equivalent.\n  \
                 Run `sudo -v`, then re-run this command.\n  \
                 Or pass `--evict none` to skip eviction — FROZEN then collapses toward COLD \
                 and the frozen-vs-cold validity checks will not mean anything."
            );
            Ok(())
        }
        EvictStrategy::Fadvise => {
            anyhow::ensure!(
                cfg!(target_os = "linux"),
                "`--evict fadvise` needs Linux (macOS has no posix_fadvise); use `--evict purge`"
            );
            anyhow::ensure!(
                !paths.is_empty(),
                "`--evict fadvise` needs `--evict-path` (the weight shards and the binary)"
            );
            for p in paths {
                anyhow::ensure!(
                    p.exists(),
                    "--evict-path {} does not exist; it would evict nothing and FROZEN \
                     would silently be a COLD run",
                    p.display()
                );
            }
            Ok(())
        }
    }
}

/// Drop `path`'s pages from the page cache. Recurses into directories so
/// `--evict-path <snapshot-dir>` covers every shard without listing them.
fn fadvise_dontneed(path: &PathBuf) -> Result<()> {
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            fadvise_dontneed(&entry?.path())?;
        }
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        let f = std::fs::File::open(path)?;
        // SAFETY: `f` owns a live fd for the duration of the call; len 0 means
        // "to end of file". POSIX_FADV_DONTNEED only drops clean pages.
        let rc = unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        anyhow::ensure!(rc == 0, "posix_fadvise returned {rc}");
    }
    #[cfg(not(target_os = "linux"))]
    let _ = path;
    Ok(())
}

// ---------------------------------------------------------------------------
// Readiness
// ---------------------------------------------------------------------------
/// Wait until `GET /v1/models` answers 200, and return the instant it did.
///
/// Deliberately not `sweep::wait_for_server`, which polls with a TCP connect:
/// accept() succeeds as soon as the listener binds, which can precede the
/// model being resident. `t_ready` is defined as "model loaded and ready to
/// serve", so it has to be an HTTP 200 on a real endpoint. Poll granularity is
/// the only quantization in `t_ready` and is reported next to the number.
fn wait_ready(
    agent: &crate::http::Agent,
    base_url: &str,
    child: &mut Child,
    timeout: Duration,
    poll: Duration,
) -> Result<Instant> {
    let deadline = Instant::now() + timeout;
    let url = format!("{base_url}/v1/models");
    while Instant::now() < deadline {
        if let Ok(resp) = agent.get(&url).call()
            && resp.status().is_success()
        {
            return Ok(Instant::now());
        }
        if let Some(st) = child.try_wait()? {
            anyhow::bail!("server exited before becoming ready ({st})");
        }
        std::thread::sleep(poll);
    }
    anyhow::bail!("server not ready within {timeout:?}")
}

// ---------------------------------------------------------------------------
// One measured repetition
// ---------------------------------------------------------------------------
#[derive(Debug, Default, Clone, serde::Serialize)]
pub(crate) struct Rep {
    pub scenario: String,
    pub mode: String,
    pub rep: usize,
    /// exec -> first content byte of the first token. The headline.
    pub ttft_exec_s: Option<f64>,
    /// exec -> `/v1/models` answers 200 (server mode only).
    pub t_ready_s: Option<f64>,
    /// Request send -> first token, i.e. TTFT in the usual sense.
    pub ttft_from_send_s: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub output_tokens: usize,
    pub peak_rss_mib: f64,
    pub major_faults: i64,
    pub text: String,
}

/// Spawn a one-shot CLI and stop the clock on its first content byte.
///
/// A line cannot be classified as banner or content until it ends, but the
/// arrival of its FIRST byte can be remembered — so every line's first byte is
/// a candidate and the candidate is committed once the line resolves as
/// content. Stopping on the newline instead would overstate TTFT by a whole
/// line of tokens.
fn run_cli(argv: &[String], backend: &str) -> Result<Rep> {
    let t_zero = Instant::now();
    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("failed to exec {}", argv[0]))?;

    let mut stdout = child.stdout.take().expect("piped");
    let mut t_first: Option<Instant> = None;
    let mut line = String::new();
    let mut line_start: Option<Instant> = None;
    let mut text = String::new();
    let mut byte = [0u8; 1];

    loop {
        match stdout.read(&mut byte) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let now = Instant::now();
        let ch = byte[0] as char;
        text.push(ch);
        if line.is_empty() {
            line_start = Some(now);
        }
        if ch == '\n' {
            if t_first.is_none() && !is_prelude(backend, &line) {
                t_first = line_start;
            }
            line.clear();
            line_start = None;
            continue;
        }
        line.push(ch);
    }
    // Output may end without a trailing newline.
    if t_first.is_none() && !is_prelude(backend, &line) {
        t_first = line_start;
    }

    let usage = wait4_child(&mut child)?;
    anyhow::ensure!(
        usage.ok || t_first.is_some(),
        "child exited non-zero without emitting a token"
    );

    Ok(Rep {
        ttft_exec_s: t_first.map(|t| t.duration_since(t_zero).as_secs_f64()),
        peak_rss_mib: usage.peak_rss_mib,
        major_faults: usage.major_faults,
        text,
        ..Default::default()
    })
}

/// Everything a repetition needs that does not vary between repetitions.
struct Ctx<'a> {
    agent: &'a crate::http::Agent,
    base_url: String,
    model: &'a str,
    input_len: usize,
    output_len: usize,
    ready_timeout: Duration,
    poll: Duration,
    settle: Duration,
}

/// Substitute `{prompt}` and `{output_len}` in a child command.
///
/// CLI mode has to pass the prompt on the child's command line, and every
/// framework spells that flag differently (`-q` for `scr chat`, `--prompt` for
/// `mlx_lm.generate`). Rather than carry a per-backend command table — which is
/// precisely what forced a match arm per framework in the script this replaces —
/// the caller writes whatever flags it wants and marks where the values go.
fn expand(argv: &[String], prompt: &str, output_len: usize) -> Vec<String> {
    let n = output_len.to_string();
    argv.iter()
        .map(|a| a.replace("{prompt}", prompt).replace("{output_len}", &n))
        .collect()
}

/// Spawn a server, wait for ready, then stream one completion whose clock
/// started before `exec`.
///
/// Request 0's `t_zero` is the pre-exec instant, so `ttft_exec` spans process
/// init, weight load, pipeline compile, KV allocation, warmup, prefill and the
/// first sample as ONE measured number.
fn run_server(
    ctx: &Ctx<'_>,
    argv: &[String],
    prompt: &str,
    warm_requests: usize,
    seed: u64,
) -> Result<Rep> {
    let (agent, base_url, model) = (ctx.agent, ctx.base_url.as_str(), ctx.model);
    let output_len = ctx.output_len;
    let t_zero = Instant::now();
    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("failed to exec {}", argv[0]))?;

    // Always reap the child, even on the error paths below: `ru_maxrss` for
    // children is only accounted once the child has been waited on.
    let result = (|| -> Result<Rep> {
        let t_ready = wait_ready(agent, base_url, &mut child, ctx.ready_timeout, ctx.poll)?;
        let first = stream_completion(agent, base_url, model, prompt, output_len, t_zero)?;

        let mut rep = Rep {
            ttft_exec_s: first.ttft_abs_s,
            t_ready_s: Some(t_ready.duration_since(t_zero).as_secs_f64()),
            ttft_from_send_s: first.ttft_from_send_s,
            tpot_ms: first.tpot_ms,
            output_tokens: first.output_tokens,
            text: first.text,
            ..Default::default()
        };

        if warm_requests > 0 {
            // Let post-ready background work drain before sampling steady
            // state; some backends finish lazy initialization after they
            // start answering.
            std::thread::sleep(ctx.settle);
            let mut ttfts = Vec::new();
            let mut tpots = Vec::new();
            for i in 0..warm_requests {
                // A UNIQUE prompt per request, or a prefix cache serves the
                // repeat and the number is meaningless.
                let p = build_prompt(seed + 1000 + i as u64, ctx.input_len);
                let t = Instant::now();
                let r = stream_completion(agent, base_url, model, &p, output_len, t)?;
                if let Some(v) = r.ttft_from_send_s {
                    ttfts.push(v * 1000.0);
                }
                if let Some(v) = r.tpot_ms {
                    tpots.push(v);
                }
            }
            ttfts.sort_by(f64::total_cmp);
            tpots.sort_by(f64::total_cmp);
            rep.ttft_from_send_s = median(&ttfts).map(|v| v / 1000.0);
            rep.tpot_ms = median(&tpots);
        }
        Ok(rep)
    })();

    // Kill first, then reap: wait4 frees the pid, after which kill() could
    // signal an unrelated recycled process.
    let _ = child.kill();
    let usage = wait4_child(&mut child)?;

    let mut rep = result?;
    rep.peak_rss_mib = usage.peak_rss_mib;
    rep.major_faults = usage.major_faults;
    Ok(rep)
}

/// `ru_maxrss` units differ by platform: Linux reports KiB, macOS bytes.
fn rss_to_mib(raw: i64) -> f64 {
    if cfg!(target_os = "macos") {
        raw as f64 / 1_048_576.0
    } else {
        raw as f64 / 1024.0
    }
}

struct Streamed {
    /// First token, measured from the caller's `t_zero`.
    ttft_abs_s: Option<f64>,
    /// First token, measured from when the request was sent.
    ttft_from_send_s: Option<f64>,
    tpot_ms: Option<f64>,
    output_tokens: usize,
    text: String,
}

/// POST /v1/completions with `stream=true` and timestamp the first content
/// byte. Mirrors the SSE handling in `serve.rs` so both clients agree on what
/// "first token" means.
fn stream_completion(
    agent: &crate::http::Agent,
    base_url: &str,
    model: &str,
    prompt: &str,
    output_len: usize,
    t_zero: Instant,
) -> Result<Streamed> {
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        // Greedy, explicitly. Leaving temperature unset lets each SERVER apply
        // its own default, so two backends get timed on different sampling
        // paths — measured at 44.3 vs 71.2 ms TTFT, pure artifact. Greedy also
        // matches what the parity gate verifies, so what is timed is what was
        // checked.
        "temperature": 0.0,
        "stream": true,
        "ignore_eos": true,
    });

    let t_send = Instant::now();
    let resp = agent
        .post(format!("{base_url}/v1/completions"))
        .header("content-type", "application/json")
        .send(body.to_string().as_str())
        .context("completion request failed")?;
    anyhow::ensure!(
        resp.status().is_success(),
        "HTTP {} from /v1/completions: {}",
        resp.status(),
        resp.into_body().read_to_string().unwrap_or_default()
    );

    let mut reader = resp.into_body().into_reader();
    let mut chunk = [0u8; 8192];
    let mut buf = String::new();
    let mut t_first: Option<Instant> = None;
    let mut t_last = t_send;
    let mut n_tokens = 0usize;
    let mut text = String::new();

    loop {
        let n = match reader.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        buf.push_str(&String::from_utf8_lossy(&chunk[..n]));
        while let Some(pos) = buf.find('\n') {
            let line = buf[..pos].trim().to_string();
            buf = buf[pos + 1..].to_string();
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                continue;
            }
            let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };
            let piece = parsed["choices"][0]["text"].as_str().unwrap_or("");
            if piece.is_empty() {
                continue;
            }
            let now = Instant::now();
            if t_first.is_none() {
                t_first = Some(now);
            }
            t_last = now;
            n_tokens += 1;
            text.push_str(piece);
        }
    }

    // TPOT over N-1 intervals, matching serve.rs.
    let tpot_ms = match (t_first, n_tokens) {
        (Some(f), n) if n > 1 => {
            Some(t_last.duration_since(f).as_secs_f64() * 1000.0 / (n - 1) as f64)
        }
        _ => None,
    };
    Ok(Streamed {
        ttft_abs_s: t_first.map(|t| t.duration_since(t_zero).as_secs_f64()),
        ttft_from_send_s: t_first.map(|t| t.duration_since(t_send).as_secs_f64()),
        tpot_ms,
        output_tokens: n_tokens,
        text,
    })
}

// ---------------------------------------------------------------------------
// Statistics and reporting
// ---------------------------------------------------------------------------
fn median(sorted: &[f64]) -> Option<f64> {
    (!sorted.is_empty()).then(|| crate::serve::percentile(sorted, 50.0))
}

/// `median (p10-p90) xN` — the spread and the rep count travel with the number.
///
/// Never a bare mean: a mean hid a bimodal ITL distribution in this repo for a
/// week (see `crates/cli/scr/src/commands/chat.rs`).
fn cell(vals: &mut [f64]) -> String {
    if vals.is_empty() {
        return "—".to_string();
    }
    vals.sort_by(f64::total_cmp);
    if vals.len() == 1 {
        return format!("{:.3} x1", vals[0]);
    }
    format!(
        "{:.3} ({:.3}-{:.3}) x{}",
        crate::serve::percentile(vals, 50.0),
        crate::serve::percentile(vals, 10.0),
        crate::serve::percentile(vals, 90.0),
        vals.len()
    )
}

fn report(reps: &[Rep], poll_ms: u64) -> Vec<String> {
    let scenarios = ["frozen", "cold", "warm"];
    println!();
    println!("============================================================");
    println!("EXEC-BOUNDARY STARTUP BENCHMARK");
    println!("============================================================");
    println!(
        "ttft_exec = exec -> first token byte, one external stopwatch. \
         Cells are median (p10-p90) xreps. t_ready poll granularity {poll_ms} ms."
    );
    println!();
    println!(
        "{:<9} {:<7} {:>26} {:>26}",
        "scenario", "mode", "ttft_exec (s)", "t_ready (s)"
    );
    for sc in scenarios {
        for mode in ["cli", "server"] {
            let mut ttft: Vec<f64> = reps
                .iter()
                .filter(|r| r.scenario == sc && r.mode == mode)
                .filter_map(|r| r.ttft_exec_s)
                .collect();
            if ttft.is_empty() {
                continue;
            }
            let mut ready: Vec<f64> = reps
                .iter()
                .filter(|r| r.scenario == sc && r.mode == mode)
                .filter_map(|r| r.t_ready_s)
                .collect();
            println!(
                "{:<9} {:<7} {:>26} {:>26}",
                sc.to_uppercase(),
                mode,
                cell(&mut ttft),
                cell(&mut ready)
            );
        }
    }

    println!();
    println!(
        "{:<9} {:<7} {:>16} {:>14}",
        "scenario", "mode", "peak_rss (MiB)", "major_faults"
    );
    for sc in scenarios {
        for mode in ["cli", "server"] {
            let group: Vec<&Rep> = reps
                .iter()
                .filter(|r| r.scenario == sc && r.mode == mode)
                .collect();
            if group.is_empty() {
                continue;
            }
            let mut rss: Vec<f64> = group.iter().map(|r| r.peak_rss_mib).collect();
            let mut flt: Vec<f64> = group.iter().map(|r| r.major_faults as f64).collect();
            rss.sort_by(f64::total_cmp);
            flt.sort_by(f64::total_cmp);
            println!(
                "{:<9} {:<7} {:>16.0} {:>14.0}",
                sc.to_uppercase(),
                mode,
                median(&rss).unwrap_or(0.0),
                median(&flt).unwrap_or(0.0)
            );
        }
    }

    // ---- validity checks ---------------------------------------------------
    // A run that fails one of these is not a slow result, it is a void one.
    println!();
    println!("validity checks");
    let mut failures = Vec::new();
    let med_for = |sc: &str, mode: &str, f: &dyn Fn(&Rep) -> Option<f64>| -> Option<f64> {
        let mut v: Vec<f64> = reps
            .iter()
            .filter(|r| r.scenario == sc && r.mode == mode)
            .filter_map(f)
            .collect();
        v.sort_by(f64::total_cmp);
        median(&v)
    };
    let mut any = false;
    for mode in ["cli", "server"] {
        let faults_f = |r: &Rep| Some(r.major_faults as f64);
        let ttft_f = |r: &Rep| r.ttft_exec_s;
        if let (Some(ff), Some(fc)) = (
            med_for("frozen", mode, &faults_f),
            med_for("cold", mode, &faults_f),
        ) {
            any = true;
            let ok = ff > fc;
            println!(
                "- {} — {mode}: FROZEN major faults {ff:.0} vs COLD {fc:.0} ({})",
                if ok { "PASS" } else { "**FAIL**" },
                if ok {
                    "eviction took effect"
                } else {
                    "eviction did NOT take effect"
                }
            );
            if !ok {
                failures.push(format!("{mode} page-cache control"));
            }
        }
        if let (Some(tf), Some(tc)) = (
            med_for("frozen", mode, &ttft_f),
            med_for("cold", mode, &ttft_f),
        ) {
            any = true;
            let ok = tf > tc;
            println!(
                "- {} — {mode}: FROZEN ttft_exec {tf:.3}s vs COLD {tc:.3}s{}",
                if ok { "PASS" } else { "**FAIL**" },
                if ok {
                    ""
                } else {
                    "  (a frozen start should never be faster)"
                }
            );
            if !ok {
                failures.push(format!("{mode} frozen<cold"));
            }
        }
    }
    if !any {
        println!("- n/a — these compare FROZEN against COLD and this run has only one of them.");
    }
    if !failures.is_empty() {
        println!();
        println!(
            "{} check(s) failed: {}. Treat the affected rows as invalid.",
            failures.len(),
            failures.join(", ")
        );
    }
    println!("============================================================");
    failures
}

// ---------------------------------------------------------------------------
// Parity gate
// ---------------------------------------------------------------------------
/// Refuse to benchmark two backends that do not agree on output.
///
/// A broken dequant path can be *fast*, so timing a wrong computation is worse
/// than not timing at all. Enforced on short high-confidence prompts only:
/// greedy decoding follows argmax, so two different kernel stacks legitimately
/// split at near-ties deep inside a long open-ended generation, and gating on
/// that would block on ordinary floating-point noise.
const PARITY_PROMPTS: &[(&str, &str, usize)] = &[
    (
        "capital",
        "What is the capital of France? Answer in one word.",
        12,
    ),
    (
        "arith",
        "What is 17 plus 25? Reply with just the number.",
        12,
    ),
    ("count", "Count from 1 to 10, separated by commas.", 40),
];

fn normalize(s: &str, backend: &str) -> String {
    let body = if backend == "mlx-lm" {
        s.split("==========").nth(1).unwrap_or(s)
    } else {
        s
    };
    body.lines()
        .filter(|l| !l.starts_with("Using model:"))
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn parity_gate(a_cmd: &[String], a_backend: &str, b_cmd: &[String], b_backend: &str) -> Result<()> {
    println!("=== parity gate: same model, greedy, high-confidence prompts ===");
    let mut failures = Vec::new();
    for (name, prompt, ntok) in PARITY_PROMPTS {
        let a = normalize(
            &run_cli(&expand(a_cmd, prompt, *ntok), a_backend)?.text,
            a_backend,
        );
        let b = normalize(
            &run_cli(&expand(b_cmd, prompt, *ntok), b_backend)?.text,
            b_backend,
        );
        let agree = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
        let common = a.chars().count().min(b.chars().count());
        let exact = common > 0 && agree == common;
        println!(
            "  {name:<9} {:<5} agree {agree}/{common} chars",
            if exact { "OK" } else { "FAIL" }
        );
        if !exact {
            println!("      a: {a}");
            println!("      b: {b}");
            failures.push(*name);
        }
    }
    anyhow::ensure!(
        failures.is_empty(),
        "parity gate failed on {}: these have deterministic answers, so a mismatch points at the \
         load path (wrong quant preset, group size, or dequant), not floating-point noise. \
         Every timing below would be measuring a different computation. Refusing to benchmark.",
        failures.join(", ")
    );
    println!("  PARITY OK — all high-confidence prompts match exactly.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
pub(crate) fn run(args: &BenchStartupArgs) -> Result<()> {
    let ex = &args.exec_opts;
    let child_cmd = ex
        .child_cmd
        .as_deref()
        .context("--exec needs --child-cmd, e.g. --child-cmd \"scr serve MODEL --port 8731\"")?;
    let argv = shell_words::split(child_cmd).context("--child-cmd is not valid shell words")?;
    anyhow::ensure!(!argv.is_empty(), "--child-cmd is empty");

    // CLI mode puts the prompt on the child's command line, so the caller has
    // to say where it goes — the flag differs per framework and this harness
    // deliberately knows nothing about any framework's flags.
    anyhow::ensure!(
        ex.mode != StartupMode::Cli || child_cmd.contains("{prompt}"),
        "--mode cli needs a {{prompt}} placeholder in --child-cmd, e.g.\n  \
         --child-cmd \"target/release/scr chat -m M --device metal -q {{prompt}} \
         --max-tokens {{output_len}}\""
    );
    if let Some(ref p) = ex.parity_cmd {
        anyhow::ensure!(
            p.contains("{prompt}"),
            "--parity-cmd needs a {{prompt}} placeholder too (the gate runs both children in CLI mode)"
        );
        anyhow::ensure!(
            child_cmd.contains("{prompt}"),
            "--parity-cmd requires --child-cmd to carry a {{prompt}} placeholder as well"
        );
    }

    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;
    let agent = crate::http::agent(false);
    let ctx = Ctx {
        agent: &agent,
        base_url: format!("http://127.0.0.1:{}", ex.port),
        model: &model,
        input_len: ex.input_len,
        output_len: ex.output_len,
        ready_timeout: Duration::from_secs(ex.ready_timeout_s),
        poll: Duration::from_millis(ex.poll_interval_ms),
        settle: Duration::from_secs_f64(ex.settle_s),
    };

    if let Some(ref other) = ex.parity_cmd {
        let other_argv = shell_words::split(other)?;
        parity_gate(&argv, &ex.backend, &other_argv, &ex.parity_backend)?;
    }

    // Provenance: a number without its machine state is not a result.
    println!("model    : {model}");
    println!("child    : {child_cmd}");
    println!("backend  : {}  mode: {:?}", ex.backend, ex.mode);
    println!("evict    : {:?}  scenarios: {}", ex.evict, ex.scenarios);
    println!(
        "os       : {} / {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    let scenarios: Vec<StartupScenario> = ex
        .scenarios
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<StartupScenario>()
                .map_err(|e| anyhow::anyhow!(e))
        })
        .collect::<Result<_>>()?;

    // Fail before measuring, not after the first eviction attempt has already
    // destroyed the cache state a FROZEN rep needed.
    if scenarios.contains(&StartupScenario::Frozen) {
        preflight_evict(ex.evict, &ex.evict_path)?;
    }

    let mut reps: Vec<Rep> = Vec::new();
    // Has any child been launched yet in this run? COLD and WARM both mean
    // "the caches are populated", which is only true once something has run.
    let mut launched = false;
    for sc in scenarios {
        let n = if sc == StartupScenario::Warm {
            1
        } else {
            ex.reps
        };
        for rep in 0..n {
            // FROZEN: remove derived on-disk caches first, then evict the page
            // cache, so the removals above cannot repopulate it.
            if sc == StartupScenario::Frozen {
                for p in &ex.remove_path {
                    let _ = std::fs::remove_dir_all(p);
                }
                evict(ex.evict, &ex.evict_path)?;
            }

            // PRIME. A COLD rung means "you ran this before" — but the first
            // launch in a fresh run has never touched the binary or the
            // weights, so without this it silently measures FROZEN instead.
            // Observed on Metal before this existed: COLD rep 0 took 7.935 s
            // with ~31,570 major faults while rep 1 took 1.218 s with ~0, so
            // the rung's own median averaged a frozen start with a cold one.
            // One throwaway launch, discarded, and only when nothing has run
            // yet: a FROZEN rep earlier in the list has already populated the
            // caches, so priming after one would be wasted work.
            if !launched && sc != StartupScenario::Frozen {
                eprintln!(
                    "--- priming ({sc:?} needs populated caches; this launch is discarded) ---"
                );
                let warmup_prompt = build_prompt(ex.seed, ex.input_len);
                match ex.mode {
                    StartupMode::Cli => {
                        run_cli(&expand(&argv, &warmup_prompt, ex.output_len), &ex.backend)?;
                    }
                    StartupMode::Server => {
                        run_server(&ctx, &argv, &warmup_prompt, 0, ex.seed)?;
                    }
                }
                // `launched` is set below once the measured rep completes; no
                // need to set it here, and doing so reads as dead.
            }

            let seed = ex.seed + rep as u64 * 17;
            let prompt = build_prompt(seed, ex.input_len);
            eprintln!("--- {sc:?}/{:?} rep {rep} ---", ex.mode);

            let mut r = match ex.mode {
                StartupMode::Cli => run_cli(&expand(&argv, &prompt, ex.output_len), &ex.backend)?,
                StartupMode::Server => run_server(
                    &ctx,
                    &argv,
                    &prompt,
                    if sc == StartupScenario::Warm {
                        ex.warm_requests
                    } else {
                        0
                    },
                    seed,
                )?,
            };
            r.scenario = format!("{sc:?}").to_lowercase();
            r.mode = format!("{:?}", ex.mode).to_lowercase();
            r.rep = rep;
            if let Some(t) = r.ttft_exec_s {
                eprintln!("    ttft_exec = {t:.3} s");
            }
            reps.push(r);
            launched = true;
        }
    }

    let failures = report(&reps, ex.poll_interval_ms);

    if let Some(ref path) = args.output_json {
        std::fs::write(path, serde_json::to_string_pretty(&reps)?)?;
        eprintln!("results written to {path}");
    }
    // A run that observed no first token measured nothing, and must not look
    // like success to a script. The report above is already empty in that case,
    // but an exit code of 0 would say otherwise — and the whole point of this
    // harness is that a measurement which did not happen fails loudly.
    anyhow::ensure!(
        reps.iter().any(|r| r.ttft_exec_s.is_some()),
        "no repetition observed a first token, so nothing was measured. Likely causes: \
         the child never wrote generated text to stdout (CLI mode reads stdout only), \
         every line it did write was classified as banner (see `--backend`), or the \
         server answered /v1/models but returned no streamed content."
    );
    anyhow::ensure!(
        failures.is_empty(),
        "validity checks failed; the run is void"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prelude_skips_banners_not_tokens() {
        assert!(is_prelude("scratchy", "Using model: foo/bar"));
        assert!(is_prelude("scratchy", "   "));
        assert!(is_prelude(
            "scratchy",
            "2026-09-23T12:00:00.000000Z  INFO thing happened"
        ));
        assert!(!is_prelude("scratchy", "Speculative decoding is"));
        assert!(is_prelude("mlx-lm", "=========="));
        assert!(!is_prelude("mlx-lm", "The capital of France"));
        // A short numeric-looking token must not be mistaken for a tracing line.
        assert!(!is_prelude("scratchy", "42"));
    }

    #[test]
    fn prompts_are_deterministic_and_unique_per_seed() {
        assert_eq!(build_prompt(7, 64), build_prompt(7, 64));
        assert_ne!(build_prompt(7, 64), build_prompt(8, 64));
    }

    #[test]
    fn cell_carries_spread_and_rep_count() {
        assert_eq!(cell(&mut []), "—");
        assert_eq!(cell(&mut [1.5]), "1.500 x1");
        let s = cell(&mut [3.0, 1.0, 2.0]);
        assert!(s.starts_with("2.000 ("), "got {s}");
        assert!(s.ends_with("x3"), "got {s}");
    }

    #[test]
    fn percentile_is_the_one_in_serve_rs() {
        // Guards against a second percentile implementation drifting in here:
        // numpy.percentile([1,2,3,4], 50, method='linear') == 2.5
        assert_eq!(crate::serve::percentile(&[1.0, 2.0, 3.0, 4.0], 50.0), 2.5);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
    }

    #[test]
    fn exec_flags_parse_and_default_sanely() {
        use clap::Parser;
        let a = BenchStartupArgs::try_parse_from([
            "startup",
            "-m",
            "org/model",
            "--exec",
            "--child-cmd",
            "scr serve org/model --port 8731",
            "--scenarios",
            "frozen,cold,warm",
            "--reps",
            "5",
        ])
        .expect("exec flags should parse");
        assert!(a.exec_opts.exec);
        assert_eq!(a.exec_opts.reps, 5);
        assert_eq!(a.exec_opts.mode, StartupMode::Server);
        assert_eq!(a.exec_opts.port, 8731);
        let scenarios: Vec<StartupScenario> = a
            .exec_opts
            .scenarios
            .split(',')
            .map(|s| s.trim().parse().unwrap())
            .collect();
        assert_eq!(
            scenarios,
            vec![
                StartupScenario::Frozen,
                StartupScenario::Cold,
                StartupScenario::Warm
            ]
        );
        // The in-process path must keep working untouched when --exec is absent.
        let plain = BenchStartupArgs::try_parse_from(["startup", "-m", "org/model"]).unwrap();
        assert!(!plain.exec_opts.exec);
        assert_eq!(plain.num_iters_cold, 3);
    }

    #[test]
    fn placeholders_expand_for_any_framework_flag_spelling() {
        // scratchy spells it -q; mlx_lm.generate spells it --prompt. Neither is
        // known to this module, which is the point.
        let scr = vec![
            "scr".into(),
            "-q".into(),
            "{prompt}".into(),
            "-n".into(),
            "{output_len}".into(),
        ];
        assert_eq!(
            expand(&scr, "hello world", 32),
            vec!["scr", "-q", "hello world", "-n", "32"]
        );
        let mlx = vec!["python".into(), "--prompt".into(), "{prompt}".into()];
        assert_eq!(expand(&mlx, "hi", 8), vec!["python", "--prompt", "hi"]);
    }

    #[test]
    fn scenario_parse_rejects_typos() {
        assert!("frozen".parse::<StartupScenario>().is_ok());
        assert!("COLD".parse::<StartupScenario>().is_ok());
        assert!("lukewarm".parse::<StartupScenario>().is_err());
    }

    /// Banner-only output must yield NO measurement, not a fast one.
    ///
    /// This is the other half of the first-content-byte contract: a child that
    /// prints only a prelude has not produced a token, so `ttft_exec` must stay
    /// `None` and the caller must be able to tell. Verified against a real child
    /// rather than a string, because the byte loop and the classifier have to
    /// agree — and a run of only such reps exits non-zero (see `run`).
    #[test]
    fn banner_only_child_measures_nothing() {
        let rep = run_cli(
            &[
                "/bin/sh".into(),
                "-c".into(),
                "echo 'Using model: fake'; echo".into(),
            ],
            "scratchy",
        )
        .expect("child runs");
        assert!(
            rep.ttft_exec_s.is_none(),
            "a banner-only child reported ttft_exec = {:?}; the clock stopped on the banner",
            rep.ttft_exec_s
        );
    }

    /// A real child, reaped with wait4, must report ITS OWN usage.
    ///
    /// This is the regression guard for the bug a Metal run exposed: with
    /// `getrusage(RUSAGE_CHILDREN)` the second child's "peak RSS" was the amount
    /// by which it exceeded the first child's high-water mark, which reported a
    /// live server at 1 MiB. Run a big child then a small one; the small one
    /// must not inherit the big one's figure, and the fault count must not be a
    /// difference of running totals.
    #[test]
    fn wait4_attributes_usage_to_the_child_that_exited() {
        let big = run_cli(
            &[
                "/bin/sh".into(),
                "-c".into(),
                // Touch ~64 MiB so this child's peak RSS is unmistakable.
                "s=$(head -c 67108864 /dev/zero | tr '\\0' 'x'); echo ${#s}".into(),
            ],
            "scratchy",
        )
        .expect("big child runs");
        let small =
            run_cli(&["/bin/echo".into(), "hi".into()], "scratchy").expect("small child runs");

        assert!(
            big.peak_rss_mib > small.peak_rss_mib,
            "the 64 MiB child ({:.1} MiB) should out-measure /bin/echo ({:.1} MiB); \
             if these are equal or inverted, usage is being read from the process-wide \
             high-water mark again",
            big.peak_rss_mib,
            small.peak_rss_mib
        );
        // The earlier, larger child must not leak into the later one.
        assert!(
            small.peak_rss_mib < 32.0,
            "/bin/echo reported {:.1} MiB",
            small.peak_rss_mib
        );
        // Per-child counters can be zero, never negative — a negative value is
        // the signature of subtracting two running totals.
        assert!(small.major_faults >= 0 && big.major_faults >= 0);
    }

    #[test]
    fn rss_units_differ_by_platform() {
        // Linux getrusage reports KiB, macOS bytes.
        let expect = if cfg!(target_os = "macos") {
            1.0
        } else {
            1024.0
        };
        assert_eq!(rss_to_mib(1_048_576), expect);
    }
}
