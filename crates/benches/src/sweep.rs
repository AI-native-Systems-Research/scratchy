// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench sweep` — parameter sweep orchestrator.
//!
//! Runs a benchmark command under multiple parameter combinations (loaded from
//! JSON files) and collects results into a timestamped output directory.
//! Mirrors Python's `vllm bench sweep serve` and `vllm bench sweep startup`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};

use crate::args::{SweepCommand, SweepCommands, SweepServeArgs, SweepStartupArgs};

// ---------------------------------------------------------------------------
// Parameter sweep types
// ---------------------------------------------------------------------------

/// A single parameter combination: key → value (CLI flag name → value string).
type ParamItem = BTreeMap<String, serde_json::Value>;

/// Read parameter combinations from a JSON file.
/// Accepts either a JSON array of objects or a single object whose values are
/// objects (keys become `_benchmark_name`).
fn read_params(path: &str) -> Result<Vec<ParamItem>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read params file: {path}"))?;
    let val: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("Invalid JSON in {path}"))?;

    match val {
        serde_json::Value::Array(arr) => arr
            .into_iter()
            .map(|v| {
                v.as_object()
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .ok_or_else(|| anyhow::anyhow!("Expected JSON object in array"))
            })
            .collect(),
        serde_json::Value::Object(map) => {
            let mut out = Vec::new();
            for (name, v) in map {
                let obj = v
                    .as_object()
                    .ok_or_else(|| anyhow::anyhow!("Expected JSON object for key {name}"))?;
                let mut item: ParamItem = obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                item.insert("_benchmark_name".to_string(), serde_json::json!(name));
                out.push(item);
            }
            Ok(out)
        }
        _ => anyhow::bail!("Params file must be a JSON array or object"),
    }
}

/// Apply a parameter combination to a command, appending --key value pairs.
fn apply_overrides(cmd: &[String], overrides: &ParamItem) -> Vec<String> {
    let mut out = cmd.to_vec();
    for (key, val) in overrides {
        if key.starts_with('_') {
            continue; // skip metadata keys like _benchmark_name
        }
        let flag = if key.starts_with("--") {
            key.clone()
        } else {
            format!("--{}", key.replace('_', "-"))
        };
        match val {
            serde_json::Value::Bool(true) => out.push(flag),
            serde_json::Value::Bool(false) => {}
            _ => {
                out.push(flag);
                out.push(val_to_string(val));
            }
        }
    }
    out
}

fn val_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

/// Produce a filesystem-safe name for a parameter combination.
fn sanitize_name(item: &ParamItem) -> String {
    let parts: Vec<String> = item
        .iter()
        .filter(|(k, _)| !k.starts_with('_'))
        .map(|(k, v)| format!("{k}={}", val_to_string(v)))
        .collect();
    if parts.is_empty() {
        "default".to_string()
    } else {
        parts
            .join("-")
            .replace(['/', '\\', ' ', ':', '"', '\''], "_")
    }
}

/// Strip an existing --output-json / --output_json flag from a command.
fn strip_output_json(cmd: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut skip_next = false;
    for arg in cmd {
        if skip_next {
            skip_next = false;
            continue;
        }
        if arg == "--output-json" || arg == "--output_json" {
            skip_next = true;
            continue;
        }
        if arg.starts_with("--output-json=") || arg.starts_with("--output_json=") {
            continue;
        }
        out.push(arg.clone());
    }
    out
}

// ---------------------------------------------------------------------------
// Run a single benchmark invocation
// ---------------------------------------------------------------------------

fn run_one(
    cmd: &[String],
    output_path: &Path,
    show_stdout: bool,
    dry_run: bool,
) -> Result<Option<serde_json::Value>> {
    // If results already exist, load and return them (resume support).
    if output_path.exists() {
        eprintln!("[SKIP] Results exist: {}", output_path.display());
        let text = std::fs::read_to_string(output_path)?;
        return Ok(Some(serde_json::from_str(&text)?));
    }

    eprintln!("[RUN] {}", shell_words::join(cmd));
    eprintln!("  -> {}", output_path.display());

    if dry_run {
        return Ok(None);
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let status = Command::new(&cmd[0])
        .args(&cmd[1..])
        .stdout(if show_stdout {
            std::process::Stdio::inherit()
        } else {
            std::process::Stdio::null()
        })
        .stderr(std::process::Stdio::inherit())
        .status()
        .with_context(|| format!("Failed to execute: {}", cmd[0]))?;

    if !status.success() {
        anyhow::bail!("Command exited with status {status}");
    }

    let text = std::fs::read_to_string(output_path)
        .with_context(|| format!("Expected output file: {}", output_path.display()))?;
    Ok(Some(serde_json::from_str(&text)?))
}

// ---------------------------------------------------------------------------
// Sweep: serve
// ---------------------------------------------------------------------------

/// Pull `host` and `port` out of a base URL.
///
/// This is the only thing the sweep ever wanted from a URL parser, and the
/// `url` crate costs 26 crates to provide it — `url` plus the ICU/idna subtree
/// it pulls for internationalised domain names, which a `--base-url` pointing
/// at a benchmark server does not need.
///
/// Bare `host:port` (no scheme) is accepted, as are IPv6 literals in brackets.
/// A URL with no explicit port gets 8000, the vLLM default.
fn split_host_port(base_url: &str) -> Result<(String, u16)> {
    let rest = base_url
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(base_url);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    if authority.is_empty() {
        anyhow::bail!("no host in URL");
    }

    // IPv6 literals are bracketed, and their colons are not port separators.
    if let Some(close) = authority.strip_prefix('[').and_then(|a| a.find(']')) {
        let host = &authority[1..=close];
        let port = match authority[close + 2..].strip_prefix(':') {
            Some(p) => p.parse()?,
            None => 8000,
        };
        return Ok((host.to_string(), port));
    }

    match authority.rsplit_once(':') {
        Some((host, port)) if !host.is_empty() => Ok((host.to_string(), port.parse()?)),
        _ => Ok((authority.to_string(), 8000)),
    }
}

/// Wait for a server at `base_url` to become ready by attempting TCP
/// connections to its host:port. We avoid pulling in an HTTP client —
/// once the TCP socket accepts we assume the server is ready.
fn wait_for_server(base_url: &str, timeout_secs: u64) -> Result<()> {
    let (host, port) =
        split_host_port(base_url).with_context(|| format!("Invalid base URL: {base_url}"))?;
    let addr = format!("{host}:{port}");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    eprintln!("Waiting for server at {addr} ...");
    loop {
        if std::time::Instant::now() > deadline {
            anyhow::bail!("Server did not become ready within {timeout_secs}s");
        }
        // Resolve every attempt rather than once up front: `connect_timeout`
        // takes a `SocketAddr`, which parses ONLY numeric literals, so going
        // straight from the string rejected every hostname — `localhost`
        // included. Resolution also belongs inside the loop because a name
        // may not be answering yet either, which is the condition we are
        // here to wait out.
        if let Ok(addrs) = std::net::ToSocketAddrs::to_socket_addrs(&addr)
            && addrs.into_iter().any(|a| {
                std::net::TcpStream::connect_timeout(&a, std::time::Duration::from_secs(2)).is_ok()
            })
        {
            eprintln!("Server ready.");
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}

fn run_sweep_serve(args: SweepServeArgs) -> Result<()> {
    let serve_cmd: Vec<String> = shell_words::split(&args.serve_cmd)?;
    let bench_cmd: Vec<String> = shell_words::split(&args.bench_cmd)?;

    let serve_params = match &args.serve_params {
        Some(p) => read_params(p)?,
        None => vec![BTreeMap::new()],
    };
    let bench_params = match &args.bench_params {
        Some(p) => read_params(p)?,
        None => vec![BTreeMap::new()],
    };

    let timestamp = args
        .resume
        .clone()
        .unwrap_or_else(|| crate::timestamp_filename_tag("_"));
    let output_dir = PathBuf::from(&args.output_dir).join(&timestamp);

    if let Some(ref _resume) = args.resume {
        if !output_dir.exists() {
            anyhow::bail!(
                "Cannot resume from non-existent directory: {}",
                output_dir.display()
            );
        }
        eprintln!("Resuming from {}", output_dir.display());
    }

    for serve_comb in &serve_params {
        let server_cmd = apply_overrides(&serve_cmd, serve_comb);

        // Start the server as a child process.
        eprintln!("\n[BEGIN SERVER] {}", shell_words::join(&server_cmd));
        if args.dry_run {
            for bench_comb in &bench_params {
                for run in 0..args.num_runs {
                    let dir_name = format!(
                        "SERVE-{}-BENCH-{}",
                        sanitize_name(serve_comb),
                        sanitize_name(bench_comb)
                    );
                    let out_path = output_dir.join(&dir_name).join(format!("run={run}.json"));
                    let mut full_cmd = apply_overrides(&bench_cmd, bench_comb);
                    full_cmd = strip_output_json(&full_cmd);
                    full_cmd.extend(["--output-json".to_string(), out_path.display().to_string()]);
                    eprintln!("[DRY-RUN] {}", shell_words::join(&full_cmd));
                }
            }
            eprintln!("[END SERVER]");
            continue;
        }

        let mut server_proc = Command::new(&server_cmd[0])
            .args(&server_cmd[1..])
            .stdout(if args.show_stdout {
                std::process::Stdio::inherit()
            } else {
                std::process::Stdio::null()
            })
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .with_context(|| format!("Failed to start server: {}", server_cmd[0]))?;

        // Extract base URL from bench_cmd or default.
        let base_url = bench_cmd
            .windows(2)
            .find_map(|w| {
                if w[0] == "--base-url" || w[0] == "--base_url" {
                    Some(w[1].clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "http://127.0.0.1:8000".to_string());

        let server_result = (|| -> Result<()> {
            wait_for_server(&base_url, args.server_ready_timeout as u64)?;

            for bench_comb in &bench_params {
                let dir_name = format!(
                    "SERVE-{}-BENCH-{}",
                    sanitize_name(serve_comb),
                    sanitize_name(bench_comb)
                );
                for run in 0..args.num_runs {
                    let out_path = output_dir.join(&dir_name).join(format!("run={run}.json"));
                    let mut full_cmd = apply_overrides(&bench_cmd, bench_comb);
                    full_cmd = strip_output_json(&full_cmd);
                    full_cmd.extend(["--output-json".to_string(), out_path.display().to_string()]);
                    run_one(&full_cmd, &out_path, args.show_stdout, false)?;
                }
            }
            Ok(())
        })();

        // Always kill server.
        let _ = server_proc.kill();
        let _ = server_proc.wait();
        eprintln!("[END SERVER]");

        server_result?;
    }

    if !args.dry_run {
        eprintln!("\nSweep complete. Results in {}", output_dir.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Sweep: startup
// ---------------------------------------------------------------------------

fn run_sweep_startup(args: SweepStartupArgs) -> Result<()> {
    let startup_cmd: Vec<String> = shell_words::split(&args.startup_cmd)?;

    let serve_params = match &args.serve_params {
        Some(p) => read_params(p)?,
        None => vec![BTreeMap::new()],
    };
    let startup_params = match &args.startup_params {
        Some(p) => read_params(p)?,
        None => vec![BTreeMap::new()],
    };

    let timestamp = args
        .resume
        .clone()
        .unwrap_or_else(|| crate::timestamp_filename_tag("_"));
    let output_dir = PathBuf::from(&args.output_dir).join(&timestamp);

    if let Some(ref _resume) = args.resume {
        if !output_dir.exists() {
            anyhow::bail!(
                "Cannot resume from non-existent directory: {}",
                output_dir.display()
            );
        }
        eprintln!("Resuming from {}", output_dir.display());
    }

    for serve_comb in &serve_params {
        for startup_comb in &startup_params {
            let dir_name = format!(
                "SERVE-{}-STARTUP-{}",
                sanitize_name(serve_comb),
                sanitize_name(startup_comb)
            );

            for run in 0..args.num_runs {
                let out_path = output_dir.join(&dir_name).join(format!("run={run}.json"));

                let mut full_cmd = apply_overrides(&startup_cmd, serve_comb);
                full_cmd = apply_overrides(&full_cmd, startup_comb);
                full_cmd = strip_output_json(&full_cmd);
                full_cmd.extend(["--output-json".to_string(), out_path.display().to_string()]);

                run_one(&full_cmd, &out_path, args.show_stdout, args.dry_run)?;
            }
        }
    }

    if !args.dry_run {
        eprintln!("\nSweep complete. Results in {}", output_dir.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub(crate) fn run_bench_sweep(cmd: SweepCommand) -> Result<()> {
    match cmd.command {
        SweepCommands::Serve(args) => run_sweep_serve(args),
        SweepCommands::Startup(args) => run_sweep_startup(args),
    }
}

#[cfg(test)]
mod tests {
    use super::split_host_port;

    #[test]
    fn parses_scheme_host_port() {
        assert_eq!(
            split_host_port("http://localhost:8137").unwrap(),
            ("localhost".into(), 8137)
        );
        assert_eq!(
            split_host_port("https://example.com:443/v1").unwrap(),
            ("example.com".into(), 443)
        );
    }

    /// The path must not be mistaken for part of the authority.
    #[test]
    fn ignores_path_query_and_fragment() {
        for u in [
            "http://host:9000/v1/completions",
            "http://host:9000?a=b",
            "http://host:9000#frag",
        ] {
            assert_eq!(split_host_port(u).unwrap(), ("host".into(), 9000), "{u}");
        }
    }

    /// vLLM's default, matching the previous `url.port().unwrap_or(8000)`.
    #[test]
    fn defaults_to_8000_without_a_port() {
        assert_eq!(
            split_host_port("http://localhost/v1").unwrap(),
            ("localhost".into(), 8000)
        );
    }

    /// A bare host:port with no scheme is a reasonable thing to pass.
    #[test]
    fn accepts_a_bare_authority() {
        assert_eq!(
            split_host_port("127.0.0.1:8000").unwrap(),
            ("127.0.0.1".into(), 8000)
        );
    }

    /// IPv6 colons are not port separators — a naive rsplit would take `:1`
    /// out of `::1` and try to connect to a host named `[`.
    #[test]
    fn handles_ipv6_literals() {
        assert_eq!(
            split_host_port("http://[::1]:8137/v1").unwrap(),
            ("::1".into(), 8137)
        );
        assert_eq!(
            split_host_port("http://[::1]/v1").unwrap(),
            ("::1".into(), 8000)
        );
    }

    #[test]
    fn rejects_garbage() {
        assert!(split_host_port("http://host:notaport").is_err());
        assert!(split_host_port("").is_err());
    }
}

#[cfg(test)]
mod wait_tests {
    use super::wait_for_server;

    /// The regression this guards: `connect_timeout` takes a `SocketAddr`,
    /// which parses only numeric literals, so the old code rejected every
    /// hostname — `localhost` included — and `bench sweep serve` could never
    /// wait for a server addressed by name.
    #[test]
    fn waits_for_a_server_addressed_by_hostname() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        wait_for_server(&format!("http://localhost:{port}"), 5)
            .expect("should connect to a listening server by hostname");
    }

    #[test]
    fn waits_for_a_server_addressed_by_ip() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        wait_for_server(&format!("http://127.0.0.1:{port}"), 5)
            .expect("should connect to a listening server by IP");
    }

    /// Nothing listening: it must time out rather than hang or succeed.
    #[test]
    fn times_out_when_nothing_is_listening() {
        // Bind then drop, so the port is almost certainly free.
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let started = std::time::Instant::now();
        assert!(wait_for_server(&format!("http://127.0.0.1:{port}"), 1).is_err());
        assert!(started.elapsed() < std::time::Duration::from_secs(20));
    }
}
