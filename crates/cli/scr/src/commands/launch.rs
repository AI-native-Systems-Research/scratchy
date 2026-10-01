// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr launch claude` — serve a model locally and start Claude Code pointed at
//! it, mirroring `ollama launch claude --model <m>`.
//!
//! Claude Code speaks the Anthropic Messages API (`/v1/messages`), which the
//! scratchy server already exposes. This command starts that server as a child
//! process (unless `--server-url` reuses a running one), waits for `/health`,
//! then runs `claude` with the `ANTHROPIC_*` environment configured to route
//! every model tier at the locally-served model, tearing the server down when
//! Claude Code exits.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::args::LaunchClaudeArgs;

/// How long to wait for the spawned server to become healthy. Generous because
/// the first run may download weights from HuggingFace before loading.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(600);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(200);
/// How long to let the spawned server drain on SIGTERM (release GPU residency)
/// before falling back to SIGKILL. Generous so a large model always drains.
const SERVER_SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

/// How launch resolved the Claude Code session id and what to add to the
/// `claude` command line for it (the store is keyed by the id in every case).
#[derive(Debug, PartialEq, Eq)]
enum SessionInject {
    /// `--resume <id>` (first-class) → resume the session + key by id.
    Resume(String),
    /// `--session-id <id>` (first-class, or a freshly minted id) → pin + key.
    New(String),
    /// The id was found inside the forwarded `-- …` args, so it's already on the
    /// claude command line — don't re-add it (just key the store by it).
    AlreadyInArgs,
}

/// Resolve which session id to key the KV store by, and how to tell `claude`
/// about it, from the explicit flags + the forwarded claude args. `None` = no id
/// anywhere → the caller mints a fresh one (kept out of here so this stays pure /
/// unit-testable). Priority: first-class `--resume` > first-class `--session-id`
/// > an id inside the passed-through `-- …` args.
fn resolve_session_inject(
    resume: Option<String>,
    session_id: Option<String>,
    claude_args: &[String],
) -> Option<(String, SessionInject)> {
    if let Some(id) = resume {
        Some((id.clone(), SessionInject::Resume(id)))
    } else if let Some(id) = session_id {
        Some((id.clone(), SessionInject::New(id)))
    } else {
        session_id_from_args(claude_args).map(|id| (id, SessionInject::AlreadyInArgs))
    }
}

pub async fn run_launch_claude(args: LaunchClaudeArgs) -> Result<()> {
    let model = args
        .model_tag
        .clone()
        .or_else(|| args.model.clone())
        .context("a model is required: `scr launch claude <MODEL>`")?;

    // Warn early if this build has no GPU backend: the server will run on CPU,
    // and a multi-billion-parameter model prefilling Claude Code's large prompt
    // (full system prompt + tool schemas, easily 10k+ tokens) can take tens of
    // seconds per turn — long enough to hit the engine's 60s no-progress abort
    // before the first token.
    // A build with no compiled GPU backend (neither `cuda` nor `metal`) runs on
    // CPU. macOS no longer auto-enables metal — you build `--features metal`
    // explicitly, same as cuda on Linux — so the CLI crate's own `metal` feature
    // is an accurate signal here on every platform.
    let no_gpu_backend = !cfg!(feature = "cuda") && !cfg!(feature = "metal");
    if args.server_url.is_none()
        && (args.device == "cpu" || (args.device == "auto" && no_gpu_backend))
    {
        eprintln!(
            "Warning: no GPU backend is compiled into this binary, so the server will run on \
             CPU and may be very slow. Rebuild with `--features metal` (Mac) or `--features \
             cuda` and run that binary, or pass --device <metal|cuda:N> if your build supports it."
        );
    }

    // Only forward an explicit --tool-call-parser; when unset the served engine
    // auto-selects one from the model family in `initialize_stack`
    // (scratchy_serving_api::tool_parser::detect_tool_parser).
    let tool_parser = args.tool_call_parser.clone();

    // Resolve the Claude Code session id and decide what to forward to claude.
    // Priority:
    //   1. first-class `--resume <id>`     → resume that session.
    //   2. first-class `--session-id <id>` → pin a (new/known) session id.
    //   3. `--resume`/`--session-id` inside the forwarded `-- …` claude args
    //      (already in claude_args — don't re-add).
    //   4. none → mint a fresh id, pin it via `--session-id`, and print the
    //      resume command on exit so start→Ctrl-C→resume is copy-paste.
    let (_session_id, inject) = resolve_session_inject(
        args.resume.clone(),
        args.session_id.clone(),
        &args.claude_args,
    )
    .unwrap_or_else(|| {
        let id = gen_uuid_v4();
        (id.clone(), SessionInject::New(id))
    });

    // Either reuse a running server, or spawn one we own (and tear down on exit).
    let (base_url, server) = match args.server_url.clone() {
        Some(url) => {
            let url = url.trim_end_matches('/').to_string();
            eprintln!("Using existing scratchy server at {url}");
            let ignored = ignored_with_server_url(&args);
            if !ignored.is_empty() {
                eprintln!(
                    "Warning: --server-url reuses a server launch did not start, so {} {} no \
                     effect here. Pass {} to that server's own `scr serve` command (or drop \
                     --server-url) before reading a measurement that depends on them.",
                    ignored.join(", "),
                    if ignored.len() == 1 { "has" } else { "have" },
                    if ignored.len() == 1 { "it" } else { "them" },
                );
            }
            (url, None)
        }
        None => {
            // Spawning a local server means running `<this binary> serve`, which
            // only exists when the `serve` engine is compiled in. A
            // `claude-remote`-only build must be pointed at an existing server.
            if !cfg!(feature = "serve") {
                bail!(
                    "this build has no local server (the `serve`/`claude` feature is \
                     disabled); pass --server-url <url> to point `launch claude` at a \
                     running scratchy server"
                );
            }
            let server = ServerProcess::spawn(&model, &args, tool_parser.as_deref())?;
            let url = format!("http://127.0.0.1:{}", server.port);
            (url, Some(server))
        }
    };

    // Point Claude Code at our server and fan the single model out to every
    // tier (opus/sonnet/haiku/subagent), matching `ollama launch`. The haiku
    // tier matters: Claude Code makes separate background calls under it.
    let mut cmd = std::process::Command::new(&args.claude_bin);
    cmd.env("ANTHROPIC_BASE_URL", &base_url)
        .env("ANTHROPIC_AUTH_TOKEN", &args.auth_token)
        // Blank the API key so Claude Code can't fall back to a real Anthropic
        // account when our base URL is set.
        .env("ANTHROPIC_API_KEY", "")
        .env("ANTHROPIC_MODEL", &model)
        .env("ANTHROPIC_DEFAULT_OPUS_MODEL", &model)
        .env("ANTHROPIC_DEFAULT_SONNET_MODEL", &model)
        .env("ANTHROPIC_DEFAULT_HAIKU_MODEL", &model)
        .env("CLAUDE_CODE_SUBAGENT_MODEL", &model);
    // Note: we deliberately leave prompt caching ON. The server ignores the
    // `cache_control` markers Claude Code sends (serde drops the unknown field),
    // and its own automatic KV prefix caching speeds up requests regardless —
    // so forcing DISABLE_PROMPT_CACHING would only trigger Claude Code's
    // "slower and cost more" warning without any benefit.
    // Forward the session flag to claude — except when it's already in the
    // passed-through `-- …` args (case 3), where re-adding would duplicate it.
    match &inject {
        SessionInject::Resume(id) => {
            cmd.arg("--resume").arg(id);
        }
        SessionInject::New(id) => {
            cmd.arg("--session-id").arg(id);
        }
        SessionInject::AlreadyInArgs => {}
    }
    cmd.args(&args.claude_args);

    eprintln!(
        "Launching `{}` against {base_url} (model: {model})",
        args.claude_bin
    );

    let status = cmd.status().with_context(|| {
        format!(
            "failed to launch `{}` — is Claude Code installed and on PATH?",
            args.claude_bin
        )
    })?;

    // Tell the user how to resume THIS session. We print the WRAPPED command (so
    // they don't copy claude's bare `claude --resume <id>`, which would bypass the
    // local server). Only for a startable (non-resume) session; printed even on
    // Ctrl-C (non-zero status).
    if let SessionInject::New(id) = &inject {
        eprintln!(
            "\nscratchy: resume this session with:\n  \
             scr launch claude {model} --resume {id}\n"
        );
    }

    // Tear the server down (kills the child we spawned), releasing GPU residency.
    drop(server);

    if !status.success() {
        if let Some(code) = status.code() {
            bail!("`{}` exited with status {code}", args.claude_bin);
        }
        bail!("`{}` terminated by signal", args.claude_bin);
    }
    Ok(())
}

/// Context window the spawned server gets when `--max-model-len` is unset. Big
/// enough for Claude Code's own system prompt + tool schemas plus a real session,
/// and under the ceiling of every model we launch.
const DEFAULT_MAX_MODEL_LEN: usize = 65536;

/// The `serve` argv for the server we spawn — everything from the subcommand name
/// on. Pure, so the whole forwarding surface is unit-testable: a serving knob is
/// reachable from `launch claude` exactly when it shows up in this vector.
///
/// Anything `--serve-arg` names, launch does not also push. clap rejects a
/// repeated argument rather than letting the last one win (`args_override_self`
/// is off), so suppressing our own occurrence is what makes the escape hatch
/// *total* — it can replace a knob launch has an opinion about, not just add one
/// launch has never heard of.
fn serve_argv(
    model: &str,
    port: u16,
    args: &LaunchClaudeArgs,
    tool_parser: Option<&str>,
) -> Result<Vec<String>> {
    /// Launch's own occurrence, unless the passthrough already named this flag.
    fn pair(argv: &mut Vec<String>, taken: &[&str], flag: &str, value: impl std::fmt::Display) {
        if !taken.contains(&flag) {
            argv.push(flag.to_string());
            argv.push(value.to_string());
        }
    }

    // Flag names the escape hatch mentions, `--flag=value` reduced to `--flag`.
    let taken: Vec<&str> = args
        .serve_arg
        .iter()
        .filter(|a| a.starts_with('-'))
        .map(|a| a.split('=').next().unwrap_or(a.as_str()))
        .collect();

    // The two launch genuinely owns: it reserves the port and health-checks that
    // exact address, so a server told to listen elsewhere would never look
    // healthy — a 600 s timeout instead of an error. Say so now.
    for owned in ["--host", "--port"] {
        if taken.contains(&owned) {
            bail!(
                "`--serve-arg {owned}` is not supported: launch reserves a free port and \
                 health-checks 127.0.0.1 on it, so a server listening elsewhere never becomes \
                 healthy. Start that server yourself and point launch at it with --server-url."
            );
        }
    }

    let mut argv = vec!["serve".to_string(), model.to_string()];
    pair(&mut argv, &taken, "--host", "127.0.0.1");
    pair(&mut argv, &taken, "--port", port);
    pair(&mut argv, &taken, "--device", &args.device);
    pair(&mut argv, &taken, "--dtype", &args.dtype);
    // Defaults, not hard-codes: each carries a launch-specific opinion `scr serve`
    // does not share (see `LaunchClaudeArgs::max_num_seqs` for the batch-1 memory
    // rationale), and each stays overridable from its own flag. The batch-1 one has
    // already been lost once to a refactor (added in d555fd53, dropped, restored) —
    // it OOMs a 32 GiB box when it goes missing, so keep it emitted here.
    pair(&mut argv, &taken, "--max-num-seqs", args.max_num_seqs);
    pair(
        &mut argv,
        &taken,
        "--max-model-len",
        args.max_model_len.unwrap_or(DEFAULT_MAX_MODEL_LEN),
    );
    if let Some(parser) = tool_parser {
        pair(&mut argv, &taken, "--tool-call-parser", parser);
    }
    if let Some(dtype) = &args.kv_cache_dtype {
        pair(&mut argv, &taken, "--kv-cache-dtype", dtype);
    }
    if let Some(spec) = &args.speculative_model {
        pair(&mut argv, &taken, "--speculative-model", spec);
    }
    if let Some(k) = args.num_speculative_tokens {
        pair(&mut argv, &taken, "--num-speculative-tokens", k);
    }
    if args.no_prefix_caching && !taken.contains(&"--no-prefix-caching") {
        argv.push("--no-prefix-caching".to_string());
    }
    argv.extend(args.serve_arg.iter().cloned());
    Ok(argv)
}

/// The serving flags that only reach a server `launch` starts itself, named so
/// `--server-url` can say out loud which ones it is dropping. A silently ignored
/// ablation flag is worse than a missing one: the run still produces numbers, and
/// they look like the ablation.
fn ignored_with_server_url(args: &LaunchClaudeArgs) -> Vec<&'static str> {
    [
        (args.max_num_seqs != 1, "--max-num-seqs"),
        (args.max_model_len.is_some(), "--max-model-len"),
        (args.tool_call_parser.is_some(), "--tool-call-parser"),
        (args.kv_cache_dtype.is_some(), "--kv-cache-dtype"),
        (args.speculative_model.is_some(), "--speculative-model"),
        (
            args.num_speculative_tokens.is_some(),
            "--num-speculative-tokens",
        ),
        (args.no_prefix_caching, "--no-prefix-caching"),
        (!args.serve_arg.is_empty(), "--serve-arg"),
    ]
    .into_iter()
    .filter_map(|(present, flag)| present.then_some(flag))
    .collect()
}

/// A scratchy server we spawned and own. Killed + reaped on drop.
struct ServerProcess {
    child: std::process::Child,
    port: u16,
}

impl ServerProcess {
    fn spawn(model: &str, args: &LaunchClaudeArgs, tool_parser: Option<&str>) -> Result<Self> {
        // Bind :0 to grab a free port, then hand it to the child.
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")
                .context("failed to reserve a local port for the server")?;
            listener.local_addr()?.port()
        };

        // The server is this same binary's `serve` subcommand.
        let exe = std::env::current_exe().context("cannot resolve the current executable")?;

        // Route the server's logs to a file: keeps Claude Code's TUI clean while
        // leaving the device choice, prompt sizes, and prefill/decode timings
        // available for debugging.
        let log_path = std::env::temp_dir().join(format!("scr-launch-{port}.log"));
        let log_file = std::fs::File::create(&log_path)
            .with_context(|| format!("failed to create server log file {}", log_path.display()))?;
        let log_file2 = log_file
            .try_clone()
            .context("failed to clone the server log file handle")?;

        let mut cmd = std::process::Command::new(&exe);
        cmd.args(serve_argv(model, port, args, tool_parser)?)
            .stdout(std::process::Stdio::from(log_file))
            .stderr(std::process::Stdio::from(log_file2));

        eprintln!(
            "Starting scratchy server on 127.0.0.1:{port} (model: {model}); logs: {}",
            log_path.display()
        );
        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to spawn server via {}", exe.display()))?;

        // Model load + first prime can take a long time; the server's own progress
        // is redirected to the log file, so without this the launch looks frozen.
        // Surface it on the launch terminal (before claude's TUI takes over): the
        // elapsed time plus the server's latest log line. Only when stderr is a TTY
        // (never spam a pipe/redirect).
        use std::io::{IsTerminal, Write};
        let tty = std::io::stderr().is_terminal();
        let start = Instant::now();
        let mut last_tick = start;
        loop {
            if let Some(status) = child.try_wait().context("failed to poll server status")? {
                if tty {
                    eprint!("\r\x1b[K");
                }
                bail!("scratchy server exited before becoming healthy ({status})");
            }
            if start.elapsed() > HEALTH_TIMEOUT {
                let _ = child.kill();
                let _ = child.wait();
                if tty {
                    eprint!("\r\x1b[K");
                }
                bail!(
                    "scratchy server did not become healthy within {}s",
                    HEALTH_TIMEOUT.as_secs()
                );
            }
            if health_check(port) {
                break;
            }
            if tty && last_tick.elapsed() >= Duration::from_millis(700) {
                last_tick = Instant::now();
                let msg = last_log_line(&log_path).unwrap_or_else(|| "starting…".to_string());
                let msg: String = msg.chars().take(96).collect();
                // \r to redraw in place, \x1b[K to clear any longer previous line.
                eprint!(
                    "\r  scratchy: loading… {}s — {msg}\x1b[K",
                    start.elapsed().as_secs()
                );
                let _ = std::io::stderr().flush();
            }
            std::thread::sleep(HEALTH_POLL_INTERVAL);
        }
        if tty {
            eprint!("\r\x1b[K");
        }
        eprintln!(
            "Server healthy on port {port} ({}s).",
            start.elapsed().as_secs()
        );
        Ok(Self { child, port })
    }
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        eprintln!("Shutting down scratchy server (graceful)...");
        // SIGTERM (not SIGKILL) so the server runs its graceful shutdown and
        // releases GPU residency — a SIGKILL here strands system-wired GPU memory
        // (`Child::kill` sends SIGKILL, which the server can't catch). Fall back to
        // SIGKILL only if it doesn't exit within the drain window.
        let pid = self.child.id() as libc::pid_t;
        // SAFETY: `pid` is our own live (un-reaped) child; SIGTERM is always valid.
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
        let start = Instant::now();
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => return, // exited gracefully (GPU residency released)
                Ok(None) => {}
                Err(_) => break, // can't poll → force-kill below
            }
            if start.elapsed() > SERVER_SHUTDOWN_GRACE {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        // Last resort: it ignored SIGTERM past the grace window.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Best-effort last non-empty line of the server log, with ANSI escape sequences
/// stripped, for surfacing the server's own load progress on the launch terminal.
fn last_log_line(path: &std::path::Path) -> Option<String> {
    let data = std::fs::read_to_string(path).ok()?;
    let line = data.lines().rev().find(|l| !l.trim().is_empty())?;
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // Skip a CSI escape (ESC [ … final-byte), e.g. color codes.
            for d in chars.by_ref() {
                if d.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    Some(out.trim().to_string())
}

/// Raw HTTP/1.1 `GET /health` over std::net — a 200 means the server is ready.
fn health_check(port: u16) -> bool {
    use std::io::{Read, Write};
    let Ok(mut stream) = std::net::TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_secs(1),
    ) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    if stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut buf = [0u8; 64];
    match stream.read(&mut buf) {
        Ok(n) => String::from_utf8_lossy(&buf[..n]).contains("200"),
        Err(_) => false,
    }
}

/// Extract an explicit Claude Code session id from the forwarded `claude` args:
/// `--session-id <id>` / `--session-id=<id>`, or `--resume <id>` / `--resume=<id>`
/// when followed by a concrete id (a bare `--resume` opens the interactive picker
/// and carries no id, so we can't key by it). Returns the first match.
fn session_id_from_args(args: &[String]) -> Option<String> {
    let is_id = |s: &str| !s.is_empty() && !s.starts_with('-');
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        for key in ["--session-id=", "--resume="] {
            if let Some(v) = a.strip_prefix(key)
                && is_id(v)
            {
                return Some(v.to_string());
            }
        }
        if (a == "--session-id" || a == "--resume")
            && let Some(v) = args.get(i + 1)
            && is_id(v)
        {
            return Some(v.clone());
        }
        i += 1;
    }
    None
}

/// A RFC-4122 v4-shaped UUID for a fresh session id. Not cryptographically
/// random — derived from the wall clock + pid via splitmix64 — but unique enough
/// to key a per-session KV store and valid-shaped for `claude --session-id`.
fn gen_uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        ^ ((std::process::id() as u64).wrapping_shl(32))
        ^ 0x9E37_79B9_7F4A_7C15;
    let mut z = seed;
    let mut next = || {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&next().to_le_bytes());
    b[8..].copy_from_slice(&next().to_le_bytes());
    b[6] = (b[6] & 0x0F) | 0x40; // version 4
    b[8] = (b[8] & 0x3F) | 0x80; // variant 1
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0],
        b[1],
        b[2],
        b[3],
        b[4],
        b[5],
        b[6],
        b[7],
        b[8],
        b[9],
        b[10],
        b[11],
        b[12],
        b[13],
        b[14],
        b[15]
    )
}

#[cfg(test)]
mod tests {
    use super::{
        LaunchClaudeArgs, SessionInject, gen_uuid_v4, ignored_with_server_url,
        resolve_session_inject, serve_argv, session_id_from_args,
    };

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    /// Parse a `launch claude` command line the way clap will at runtime, so the
    /// forwarding tests exercise the real flag surface rather than a hand-built
    /// struct that can drift from it.
    fn launch_args(extra: &[&str]) -> LaunchClaudeArgs {
        use clap::Parser as _;
        let mut argv = vec!["claude", "my-model"];
        argv.extend_from_slice(extra);
        LaunchClaudeArgs::try_parse_from(argv).expect("launch args should parse")
    }

    /// The value after the LAST occurrence of `flag` — clap's own
    /// override-on-repeat semantics, so an assertion reads as what the spawned
    /// `serve` will actually resolve.
    fn last_val<'a>(argv: &'a [String], flag: &str) -> Option<&'a str> {
        let i = argv.iter().rposition(|a| a == flag)?;
        argv.get(i + 1).map(String::as_str)
    }

    #[test]
    fn serve_argv_defaults_keep_batch_one_and_restate_nothing_else() {
        let argv = serve_argv("my-model", 1234, &launch_args(&[]), None).unwrap();
        assert_eq!(&argv[..2], &s(&["serve", "my-model"])[..]);
        assert_eq!(last_val(&argv, "--port"), Some("1234"));
        // The batch-1 memory rationale survives as the default…
        assert_eq!(last_val(&argv, "--max-num-seqs"), Some("1"));
        assert_eq!(last_val(&argv, "--max-model-len"), Some("65536"));
        // …while an unset knob is left to `serve`'s own default instead of being
        // restated (and skewed) here.
        for flag in [
            "--kv-cache-dtype",
            "--speculative-model",
            "--num-speculative-tokens",
            "--no-prefix-caching",
            "--tool-call-parser",
        ] {
            assert!(!argv.iter().any(|a| a == flag), "{flag} should be absent");
        }
    }

    /// Every ablation switch the bench matrix names has to be expressible as
    /// `scr launch claude` flags. This is that list, as flags.
    #[test]
    fn serve_argv_forwards_every_ablation_knob() {
        let args = launch_args(&[
            "--max-num-seqs",
            "4",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--speculative-model",
            "ngram",
            "--num-speculative-tokens",
            "3",
            "--no-prefix-caching",
            "--max-model-len",
            "32768",
        ]);
        let argv = serve_argv("my-model", 1, &args, Some("gemma4")).unwrap();
        assert_eq!(last_val(&argv, "--max-num-seqs"), Some("4"));
        assert_eq!(last_val(&argv, "--kv-cache-dtype"), Some("fp8_e4m3"));
        assert_eq!(last_val(&argv, "--speculative-model"), Some("ngram"));
        assert_eq!(last_val(&argv, "--num-speculative-tokens"), Some("3"));
        assert_eq!(last_val(&argv, "--max-model-len"), Some("32768"));
        assert_eq!(last_val(&argv, "--tool-call-parser"), Some("gemma4"));
        assert!(argv.iter().any(|a| a == "--no-prefix-caching"));
    }

    #[test]
    fn serve_arg_passthrough_reaches_serve_and_wins_last() {
        // A `serve` flag launch has never heard of, a `=`-joined one, and one
        // launch does have an opinion about: the passthrough must reach the child
        // AND replace launch's own occurrence, since clap rejects a repeat.
        let args = launch_args(&[
            "--serve-arg",
            "--no-tool-spans",
            "--serve-arg",
            "--block-size=32",
            "--serve-arg",
            "--max-num-seqs",
            "--serve-arg",
            "8",
        ]);
        let argv = serve_argv("my-model", 1, &args, None).unwrap();
        assert!(argv.iter().any(|a| a == "--no-tool-spans"));
        assert!(argv.iter().any(|a| a == "--block-size=32"));
        // launch's own `--max-num-seqs 1` is gone, not merely outranked: a second
        // occurrence is an ArgumentConflict in the child, not a last-one-wins.
        assert_eq!(argv.iter().filter(|a| *a == "--max-num-seqs").count(), 1);
        assert_eq!(last_val(&argv, "--max-num-seqs"), Some("8"));
    }

    /// The parity check that matters: `serve`'s own parser has to accept
    /// everything launch emits and resolve it to the values launch meant. A knob
    /// that only *looks* forwarded — renamed upstream, or value-taking where
    /// launch passes a switch — would otherwise yield a server that parsed fine
    /// and ran un-ablated.
    #[test]
    fn serve_accepts_and_resolves_the_forwarded_argv() {
        use clap::Parser as _;
        let args = launch_args(&[
            "--max-num-seqs",
            "4",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--speculative-model",
            "ngram",
            "--num-speculative-tokens",
            "3",
            "--no-prefix-caching",
            "--serve-arg",
            "--enable-metrics",
        ]);
        // argv[0] ("serve") doubles as the binary name clap expects to skip.
        let argv = serve_argv("my-model", 9999, &args, Some("gemma4")).unwrap();
        let serve = crate::args::ServeArgs::try_parse_from(&argv)
            .expect("serve must accept every argument launch forwards");
        assert_eq!(serve.resolved_model().unwrap(), "my-model");
        assert_eq!(serve.host, "127.0.0.1");
        assert_eq!(serve.port, 9999);
        assert_eq!(serve.max_num_seqs, Some(4));
        assert_eq!(serve.max_model_len, Some(super::DEFAULT_MAX_MODEL_LEN));
        assert_eq!(serve.kv_cache_dtype, "fp8_e4m3");
        assert_eq!(serve.speculative_model.as_deref(), Some("ngram"));
        assert_eq!(serve.num_speculative_tokens, 3);
        assert!(serve.no_prefix_caching);
        assert_eq!(serve.tool_call_parser.as_deref(), Some("gemma4"));
        // …including the flag launch itself knows nothing about.
        assert!(serve.enable_metrics);
    }

    /// The override has to hold where it counts: in the child's parser. clap
    /// rejects a repeated argument instead of taking the last one, so this fails
    /// the moment launch pushes its own occurrence alongside the passthrough.
    #[test]
    fn passthrough_override_parses_in_the_child() {
        use clap::Parser as _;
        let args = launch_args(&["--serve-arg", "--max-num-seqs", "--serve-arg", "8"]);
        let argv = serve_argv("my-model", 1, &args, None).unwrap();
        let serve = crate::args::ServeArgs::try_parse_from(&argv)
            .expect("an override must not collide with launch's own default");
        assert_eq!(serve.max_num_seqs, Some(8));
    }

    /// `--host`/`--port` are launch's own: it reserves the port and health-checks
    /// that address. Overriding them would time out after 600 s instead of
    /// failing, so they are refused up front.
    #[test]
    fn serve_arg_refuses_the_two_flags_launch_owns() {
        for flag in ["--port", "--host", "--port=9000"] {
            let args = launch_args(&["--serve-arg", flag]);
            let err =
                serve_argv("my-model", 1, &args, None).expect_err("launch owns the listen address");
            assert!(
                err.to_string().contains("--server-url"),
                "the error should point at the supported way: {err}"
            );
        }
    }

    #[test]
    fn serve_arg_does_not_swallow_the_claude_args() {
        let args = launch_args(&["--serve-arg", "--enable-metrics", "--", "-p", "hi"]);
        assert_eq!(args.serve_arg, s(&["--enable-metrics"]));
        assert_eq!(args.claude_args, s(&["-p", "hi"]));
    }

    #[test]
    fn server_url_names_the_knobs_it_cannot_apply() {
        assert!(ignored_with_server_url(&launch_args(&[])).is_empty());
        assert_eq!(
            ignored_with_server_url(&launch_args(&[
                "--no-prefix-caching",
                "--max-num-seqs",
                "4"
            ])),
            vec!["--max-num-seqs", "--no-prefix-caching"]
        );
        assert_eq!(
            ignored_with_server_url(&launch_args(&["--serve-arg", "--enable-metrics"])),
            vec!["--serve-arg"]
        );
    }

    #[test]
    fn last_log_line_strips_ansi_and_takes_tail() {
        use super::last_log_line;
        let dir = std::env::temp_dir().join(format!("scratchy-xl-log-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("s.log");
        std::fs::write(
            &p,
            "\u{1b}[2m2026\u{1b}[0m INFO first line\n\u{1b}[32m INFO\u{1b}[0m Application startup complete\n\n",
        )
        .unwrap();
        assert_eq!(
            last_log_line(&p).as_deref(),
            Some("INFO Application startup complete")
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn session_inject_priority() {
        // first-class --resume wins over everything (and resumes, not pins).
        assert_eq!(
            resolve_session_inject(Some("R".into()), Some("S".into()), &s(&["--resume", "A"])),
            Some(("R".into(), SessionInject::Resume("R".into())))
        );
        // then first-class --session-id (pins a new/known id).
        assert_eq!(
            resolve_session_inject(None, Some("S".into()), &s(&["--resume", "A"])),
            Some(("S".into(), SessionInject::New("S".into())))
        );
        // then an id inside the forwarded `-- …` args — keyed but NOT re-added.
        assert_eq!(
            resolve_session_inject(None, None, &s(&["--resume", "A"])),
            Some(("A".into(), SessionInject::AlreadyInArgs))
        );
        // nothing anywhere → None (caller mints a fresh id).
        assert_eq!(resolve_session_inject(None, None, &s(&["-p", "hi"])), None);
    }

    #[test]
    fn session_id_parsing() {
        assert_eq!(
            session_id_from_args(&s(&["--resume", "abc-123"])),
            Some("abc-123".into())
        );
        assert_eq!(
            session_id_from_args(&s(&["--session-id", "xyz"])),
            Some("xyz".into())
        );
        assert_eq!(
            session_id_from_args(&s(&["--session-id=deadbeef"])),
            Some("deadbeef".into())
        );
        assert_eq!(
            session_id_from_args(&s(&["--resume=foo"])),
            Some("foo".into())
        );
        // bare --resume (interactive picker) carries no id
        assert_eq!(session_id_from_args(&s(&["--resume"])), None);
        assert_eq!(session_id_from_args(&s(&["--resume", "--verbose"])), None);
        assert_eq!(session_id_from_args(&s(&["-p", "hello"])), None);
    }

    #[test]
    fn uuid_is_v4_shaped_and_unique() {
        let u = gen_uuid_v4();
        assert_eq!(u.len(), 36);
        let parts: Vec<&str> = u.split('-').collect();
        assert_eq!(
            parts.iter().map(|p| p.len()).collect::<Vec<_>>(),
            vec![8, 4, 4, 4, 12]
        );
        assert!(u.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
        assert_eq!(&u[14..15], "4", "version nibble");
        assert!(
            matches!(&u[19..20], "8" | "9" | "a" | "b"),
            "variant nibble"
        );
        assert_ne!(gen_uuid_v4(), gen_uuid_v4(), "two mints must differ");
    }
}
