// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `cuda` — the scratchy CLI on NVIDIA H100: the shared commands (`scratchy_cli`), served through
//! the CUDA worker.
//!
//! Usage:
//!   cuda serve --model <path-or-hf-id>
//!   cuda chat -m <path-or-hf-id> -q "..."

// Anchor the CUDA worker crate into the link: the engine serves through the `WorkerFactory`
// registry and never names a concrete worker, so without this reference the linker may drop the
// factory's `inventory::submit!`.
extern crate scratchy_target_cuda_worker as _;

use clap::Parser;

/// scratchy on NVIDIA H100.
#[derive(Parser, Debug)]
#[command(name = "cuda", version, about, long_about = None)]
struct CudaCli {
    #[command(subcommand)]
    command: scratchy_cli::args::Commands,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file if present (before clap parses, so env vars are visible).
    dotenvy::dotenv().ok();
    let r = scratchy_cli::run::<CudaCli>(CudaCli::parse().command).await;
    // A run a termination signal ended exits as the signal's default action would have
    // (128 + SIGINT), not as a failure.
    if scratchy_cli::interrupted(&r) {
        eprintln!("interrupted");
        std::process::exit(130);
    }
    r
}
