// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Test server helper — starts the vLLM server in-process or as a child process.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::task::JoinHandle;

/// Default timeout waiting for the server to become healthy.
const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

/// Poll interval when waiting for /health.
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Whether to spawn a child process by default.
fn should_spawn() -> bool {
    if std::env::var("VLLM_TEST_SPAWN").as_deref() == Ok("1") {
        return true;
    }
    cfg!(feature = "cuda")
}

/// Internal mode: how the server is running.
enum TestServerMode {
    InProcess {
        step_handle: JoinHandle<()>,
        server_handle: JoinHandle<()>,
    },
    ChildProcess {
        child: std::process::Child,
    },
}

/// A running vLLM server for E2E testing.
///
/// The server runs either in-process on a background tokio task, or as a
/// spawned child process (for CUDA tests, to reclaim GPU memory between tests).
/// On drop, the server is stopped and resources are released.
pub struct TestServer {
    mode: TestServerMode,
    port: u16,
    base_url: String,
}

impl TestServer {
    /// Create a builder for configuring a test server with the given model.
    pub fn builder(model: &str) -> TestServerBuilder {
        TestServerBuilder {
            model: model.to_string(),
            extra_args: Vec::new(),
            startup_timeout: DEFAULT_STARTUP_TIMEOUT,
            port: None,
            tool_call_parser: None,
            reasoning_parser: None,
            lora_adapter: None,
            pooling_strategy: None,
            disable_async_scheduling: false,
            runner: "generate".to_string(),
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            dtype: None,
            device: None,
            spawn: should_spawn(),
            enforce_eager: None,
        }
    }

    /// The base URL (e.g. `http://127.0.0.1:12345`).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The port the server is listening on.
    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        match &mut self.mode {
            TestServerMode::InProcess {
                step_handle,
                server_handle,
            } => {
                step_handle.abort();
                server_handle.abort();
            }
            TestServerMode::ChildProcess { child } => {
                let _ = child.kill();
                let _ = child.wait();
                // Give the CUDA driver time to reclaim GPU memory after process exit.
                std::thread::sleep(Duration::from_secs(2));
            }
        }
    }
}

/// Builder for configuring and starting a [`TestServer`].
pub struct TestServerBuilder {
    model: String,
    extra_args: Vec<String>,
    startup_timeout: Duration,
    port: Option<u16>,
    tool_call_parser: Option<String>,
    reasoning_parser: Option<String>,
    lora_adapter: Option<String>,
    pooling_strategy: Option<String>,
    disable_async_scheduling: bool,
    runner: String,
    dtype: Option<String>,
    device: Option<String>,
    tensor_parallel_size: usize,
    pipeline_parallel_size: usize,
    spawn: bool,
    enforce_eager: Option<bool>,
}

impl TestServerBuilder {
    /// Add extra CLI arguments (e.g. `--tool-call-parser hermes`).
    pub fn with_args(mut self, args: &[&str]) -> Self {
        self.extra_args.extend(args.iter().map(|s| s.to_string()));
        self
    }

    /// Override the port (default: random free port).
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Override the startup timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.startup_timeout = timeout;
        self
    }

    /// Set the tool call parser (e.g. "hermes", "llama3_json", "kimi_k2").
    pub fn with_tool_call_parser(mut self, parser: &str) -> Self {
        self.tool_call_parser = Some(parser.to_string());
        self
    }

    /// Set the reasoning parser (e.g. "deepseek_r1", "qwen3").
    pub fn with_reasoning_parser(mut self, parser: &str) -> Self {
        self.reasoning_parser = Some(parser.to_string());
        self
    }

    /// Set a LoRA adapter to load (local path or HF repo ID).
    pub fn with_lora_adapter(mut self, adapter: &str) -> Self {
        self.lora_adapter = Some(adapter.to_string());
        self
    }

    /// Set the pooling strategy for embeddings (e.g. "mean", "cls", "last").
    pub fn with_pooling_strategy(mut self, strategy: &str) -> Self {
        self.pooling_strategy = Some(strategy.to_string());
        self
    }

    /// Disable async scheduling (use synchronous step loop instead).
    pub fn with_sync_scheduling(mut self) -> Self {
        self.disable_async_scheduling = true;
        self
    }

    /// Set the runner type: "generate" (default) or "pooling".
    /// In pooling mode, embedding requests go through the scheduler and
    /// generation endpoints are rejected.
    pub fn with_runner(mut self, runner: &str) -> Self {
        self.runner = runner.to_string();
        self
    }

    /// Set the tensor parallel size (number of GPUs).
    pub fn with_tensor_parallel_size(mut self, tp: usize) -> Self {
        self.tensor_parallel_size = tp;
        self
    }

    /// Set the pipeline parallel size (number of pipeline stages).
    pub fn with_pipeline_parallel_size(mut self, pp: usize) -> Self {
        self.pipeline_parallel_size = pp;
        self
    }

    /// Override the weight dtype (e.g. "f32", "f16", "bf16").
    pub fn with_dtype(mut self, dtype: &str) -> Self {
        self.dtype = Some(dtype.to_string());
        self
    }

    /// Override the device (e.g. "cpu", "cuda:0", "metal").
    pub fn with_device(mut self, device: &str) -> Self {
        self.device = Some(device.to_string());
        self
    }

    /// Enable or disable child-process spawning mode.
    /// When enabled, the server runs as a separate OS process so GPU memory
    /// is fully reclaimed on drop.
    pub fn with_spawn(mut self, spawn: bool) -> Self {
        self.spawn = spawn;
        self
    }

    /// Override the enforce-eager setting.
    /// When `false`, CUDA graphs are enabled (default for CLI).
    /// When `true`, CUDA graphs are disabled (default for in-process tests).
    pub fn with_enforce_eager(mut self, eager: bool) -> Self {
        self.enforce_eager = Some(eager);
        self
    }

    /// Start the server and wait for it to become healthy.
    pub async fn start(self) -> Result<TestServer> {
        if self.spawn {
            self.start_child_process().await
        } else {
            self.start_in_process().await
        }
    }

    /// Start the server as a child process.
    async fn start_child_process(self) -> Result<TestServer> {
        let port = self.port.unwrap_or_else(|| {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        });
        let base_url = format!("http://127.0.0.1:{port}");

        let binary = resolve_binary_path()?;

        let mut cmd = std::process::Command::new(&binary);
        cmd.arg("serve")
            .arg(&self.model)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string());

        let dtype = self.dtype.as_deref().unwrap_or("auto");
        cmd.arg("--dtype").arg(dtype);

        let device = self.device.as_deref().unwrap_or("auto");
        cmd.arg("--device").arg(device);

        if self.tensor_parallel_size > 1 {
            cmd.arg("--tensor-parallel-size")
                .arg(self.tensor_parallel_size.to_string());
        }

        if self.pipeline_parallel_size > 1 {
            cmd.arg("--pipeline-parallel-size")
                .arg(self.pipeline_parallel_size.to_string());
        }

        if let Some(ref parser) = self.tool_call_parser {
            cmd.arg("--tool-call-parser").arg(parser);
        }

        if let Some(ref parser) = self.reasoning_parser {
            cmd.arg("--reasoning-parser").arg(parser);
        }

        if let Some(ref adapter) = self.lora_adapter {
            cmd.arg("--lora-adapter").arg(adapter);
        }

        if let Some(ref strategy) = self.pooling_strategy {
            cmd.arg("--pooling-strategy").arg(strategy);
        }

        if self.runner != "generate" {
            cmd.arg("--runner").arg(&self.runner);
        }

        if self.disable_async_scheduling {
            cmd.arg("--disable-async-scheduling");
        }

        if self.enforce_eager == Some(true) {
            cmd.arg("--enforce-eager");
        }

        for arg in &self.extra_args {
            cmd.arg(arg);
        }

        // Inherit stderr so child server logs are visible in real-time.
        // Stdout is still piped (unused).
        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit());

        eprintln!("[E2E] Spawning: {:?}", cmd);

        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to spawn scr binary at {}", binary.display()))?;

        // Wait for the server to become healthy, checking for early child exit.
        let health_result =
            wait_for_health_with_child(&base_url, &mut child, self.startup_timeout).await;

        if let Err(e) = health_result {
            eprintln!("[E2E] Health check failed: {e}");
            let _ = child.kill();
            let _ = child.wait(); // reap
            return Err(e);
        }

        let test_server = TestServer {
            mode: TestServerMode::ChildProcess { child },
            port,
            base_url,
        };

        eprintln!("[E2E] Server healthy on port {port}");
        Ok(test_server)
    }

    /// Start the server in-process and wait for it to become healthy.
    async fn start_in_process(self) -> Result<TestServer> {
        // Initialize tracing. Uses RUST_LOG env if set, otherwise "info" so
        // initialization progress is visible and hangs are diagnosable.
        let log_level = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
        scratchy_core_common::telemetry::init_tracing(&log_level);

        let port = self.port.unwrap_or_else(|| {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        });

        let base_url = format!("http://127.0.0.1:{port}");
        let bind_address = format!("127.0.0.1:{port}");
        let model = self.model.clone();

        // Build VllmConfig for the requested configuration.
        let pooling_strategy = self
            .pooling_strategy
            .clone()
            .unwrap_or_else(|| "auto".to_string());
        let runner = self.runner.clone();
        let dtype = self.dtype.clone().unwrap_or_else(|| "auto".to_string());
        let device = self.device.clone().unwrap_or_else(|| "auto".to_string());
        let mut config = scratchy_serving_api::init::VllmConfig {
            model: model.clone(),
            device,
            dtype,
            max_num_seqs: Some(256),
            block_size: 16,
            gpu_memory_utilization: 0.9,
            lora_adapter: self.lora_adapter.clone(),
            pooling_strategy,
            tensor_parallel_size: self.tensor_parallel_size,
            pipeline_parallel_size: self.pipeline_parallel_size,
            disable_async_scheduling: self.disable_async_scheduling,
            runner: runner.clone(),
            ..Default::default()
        };
        if let Some(eager) = self.enforce_eager {
            config.enforce_eager = eager;
        }

        // Initialize the full stack (blocking — downloads model, loads weights).
        // 5 min timeout: first run may need to download multi-GB model weights.
        let init_timeout = Duration::from_secs(300);
        eprintln!("[E2E] starting initialize_stack for {model}...");
        let mut stack = tokio::time::timeout(
            init_timeout,
            tokio::task::spawn_blocking(move || {
                eprintln!("[E2E] spawn_blocking: calling initialize_stack");
                let result = scratchy_serving_api::init::initialize_stack(&config, None);
                eprintln!("[E2E] initialize_stack returned: {}", result.is_ok());
                result
            }),
        )
        .await
        .map_err(|_| anyhow::anyhow!("initialize_stack timed out after {init_timeout:?}"))?
        .context("initialize_stack panicked")?
        .context("failed to initialize stack")?;
        eprintln!("[E2E] stack initialized successfully");

        // Configure tool call parser if requested.
        if let Some(ref parser_name) = self.tool_call_parser {
            let parser = scratchy_serving_api::tool_parser::get_tool_parser(parser_name)
                .map_err(|e| anyhow::anyhow!(e))?;
            Arc::get_mut(&mut stack.engine)
                .expect("engine should not be shared yet")
                .set_tool_parser(parser);
            tracing::info!("Tool call parser configured: {}", parser_name);
        }

        // Configure reasoning parser if requested.
        if let Some(ref parser_name) = self.reasoning_parser {
            let vocab = stack
                .engine
                .tokenizer()
                .ok_or_else(|| anyhow::anyhow!("reasoning parser requires a tokenizer"))?
                .get_vocab();
            let parser =
                scratchy_serving_api::reasoning_parser::get_reasoning_parser(parser_name, &vocab)
                    .map_err(|e| anyhow::anyhow!(e))?;
            Arc::get_mut(&mut stack.engine)
                .expect("engine should not be shared yet")
                .set_reasoning_parser(parser);
            tracing::info!("Reasoning parser configured: {}", parser_name);
        }

        // Spawn the engine step loop.
        let step_handle = stack.engine.spawn_step_loop();

        // Build server config and app state.
        let server_config = scratchy_serving_api::server::ServerConfig {
            bind_address,
            version: format!("0.1.0-rust-test ({})", stack.model_name),
            cors_enabled: true,
            metrics_enabled: false,
            ssl_keyfile: None,
            ssl_certfile: None,
            ssl_ca_certs: None,
            startup_instant: None,
        };

        let is_pooling = runner == "pooling";
        let app_state = Arc::new(scratchy_serving_api::server::AppState {
            engine: stack.engine,
            config: server_config,
            is_pooling,
            scratchy_core_config: None,
        });

        // Spawn the HTTP server on a background task.
        let server_handle = tokio::spawn(async move {
            if let Err(e) = scratchy_serving_api::server::serve(app_state).await {
                tracing::error!("Test server error: {e}");
            }
        });

        let test_server = TestServer {
            mode: TestServerMode::InProcess {
                step_handle,
                server_handle,
            },
            port,
            base_url,
        };

        // Wait for the server to become healthy.
        wait_for_health(&test_server.base_url, self.startup_timeout).await?;

        tracing::info!("Test server healthy on port {port}");
        Ok(test_server)
    }
}

/// Resolve the path to the `scr` binary for child-process mode.
///
/// Builds the CLI binary if it doesn't exist or is stale relative to the
/// test binary. This ensures E2E tests always run against current code.
fn resolve_binary_path() -> Result<std::path::PathBuf> {
    // 1. Explicit env var override — skip auto-build.
    if let Ok(p) = std::env::var("VLLM_TEST_BINARY") {
        let path = std::path::PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
        anyhow::bail!(
            "VLLM_TEST_BINARY set to {} but file not found",
            path.display()
        );
    }

    // 2. Locate the binary next to the test binary.
    let (profile_dir, binary_path) = if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|p| p.to_path_buf()))
    {
        let profile_dir = if dir.ends_with("deps") {
            dir.parent().unwrap_or(&dir).to_path_buf()
        } else {
            dir
        };
        let candidate = profile_dir.join("scr");
        (Some(profile_dir), Some(candidate))
    } else {
        (None, None)
    };

    // 3. Check if the binary exists and is up-to-date.
    //    If the test binary is newer than the scr binary, rebuild.
    let needs_build = match &binary_path {
        Some(path) if path.exists() => {
            // Check if test binary is newer (meaning source changed).
            let test_mtime = std::env::current_exe()
                .ok()
                .and_then(|p| p.metadata().ok())
                .and_then(|m| m.modified().ok());
            let bin_mtime = path.metadata().ok().and_then(|m| m.modified().ok());
            match (test_mtime, bin_mtime) {
                (Some(t), Some(b)) => t > b,
                _ => false,
            }
        }
        _ => true,
    };

    if needs_build {
        // Determine profile from the path (release vs debug).
        let is_release = profile_dir
            .as_ref()
            .and_then(|p| p.file_name())
            .is_some_and(|n| n == "release");

        let mut cmd = std::process::Command::new("cargo");
        // Clear CARGO_PKG_* and CARGO_MANIFEST_DIR env vars that the test
        // harness sets for vllm-e2e. If these leak into the nested cargo
        // build, ring's build.rs records them via rerun-if-env-changed and
        // on the next build (where they're absent) cargo sees them as
        // changed, marking ring dirty and recompiling the entire TLS stack.
        for (key, _) in std::env::vars() {
            if key.starts_with("CARGO_PKG_") || key == "CARGO_MANIFEST_DIR" {
                cmd.env_remove(&key);
            }
        }
        cmd.arg("build").arg("-p").arg("scratchy-cli");
        if is_release {
            cmd.arg("--release");
        }

        // Forward feature flags from env if set by the test build.
        // The E2E test Cargo invocation sets these features on vllm-e2e,
        // but we need them on scratchy-cli too.
        let mut features = Vec::new();
        if cfg!(feature = "cuda") {
            features.push("cuda");
        }
        if cfg!(feature = "nccl") {
            features.push("nccl");
        }
        if cfg!(feature = "metal") {
            features.push("metal");
        }
        if !features.is_empty() {
            cmd.arg("--features").arg(features.join(","));
        }

        eprintln!("[E2E] Building scr binary: {:?}", cmd);
        let status = cmd
            .status()
            .context("failed to run cargo build for scratchy-cli")?;
        if !status.success() {
            anyhow::bail!("cargo build -p scratchy-cli failed with {status}");
        }
    }

    // Return the binary path.
    if let Some(path) = binary_path
        && path.exists()
    {
        return Ok(path);
    }

    // Fallback.
    for candidate in &["target/release/scr", "target/debug/scr"] {
        let path = std::path::PathBuf::from(candidate);
        if path.exists() {
            return Ok(path);
        }
    }

    anyhow::bail!(
        "Could not find the scr binary after build. Set VLLM_TEST_BINARY or build with `cargo build -p scratchy-cli`."
    )
}

/// Poll the /health endpoint until it returns 200 or we time out.
async fn wait_for_health(base_url: &str, timeout: Duration) -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let client = reqwest::Client::new();
    let health_url = format!("{base_url}/health");
    let start = std::time::Instant::now();

    loop {
        if start.elapsed() > timeout {
            anyhow::bail!(
                "Server did not become healthy within {}s",
                timeout.as_secs()
            );
        }

        match client.get(&health_url).send().await {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            _ => {}
        }

        tokio::time::sleep(HEALTH_POLL_INTERVAL).await;
    }
}

/// Like `wait_for_health` but also checks if the child process has exited
/// (e.g. panic, assertion failure). Fails fast instead of waiting the full
/// timeout.
async fn wait_for_health_with_child(
    base_url: &str,
    child: &mut std::process::Child,
    timeout: Duration,
) -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let client = reqwest::Client::new();
    let health_url = format!("{base_url}/health");
    let start = std::time::Instant::now();

    loop {
        // Fail fast: check if child exited.
        if let Some(status) = child.try_wait().context("failed to check child status")? {
            anyhow::bail!(
                "Server child process exited before becoming healthy ({})",
                status
            );
        }

        if start.elapsed() > timeout {
            anyhow::bail!(
                "Server did not become healthy within {}s",
                timeout.as_secs()
            );
        }

        match client.get(&health_url).send().await {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            _ => {}
        }

        tokio::time::sleep(HEALTH_POLL_INTERVAL).await;
    }
}
