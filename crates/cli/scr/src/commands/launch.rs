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
        cmd.arg("serve")
            .arg(model)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--device")
            .arg(&args.device)
            .arg("--dtype")
            .arg(&args.dtype)
            .stdout(std::process::Stdio::from(log_file))
            .stderr(std::process::Stdio::from(log_file2));
        if let Some(parser) = tool_parser {
            cmd.arg("--tool-call-parser").arg(parser);
        }
        // A single interactive Claude Code session needs only one in-flight
        // sequence, and batch 1 keeps the GDN recurrent-state pool tiny: it is
        // sized `--max-num-seqs × per-slot` (61 MiB/slot on Qwen3.5-MoE-35B), so
        // the serve default of 256 reserves 15.7 GiB and OOMs a 32 GiB box before
        // the first token. Restores the launch serving config dropped in a later
        // refactor (originally added in d555fd53).
        cmd.arg("--max-num-seqs").arg("1");
        cmd.arg("--max-model-len")
            .arg(args.max_model_len.unwrap_or(65536).to_string());

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
    use super::{SessionInject, gen_uuid_v4, resolve_session_inject, session_id_from_args};

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
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
