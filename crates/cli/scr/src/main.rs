// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr` — the scratchy CLI, serving on metal.
//!
//! Usage:
//!   scr serve --model <path-or-hf-id>
//!   scr bench --model <path-or-hf-id>
//!   scr model convert --input <dir> --output <dir>

// Anchor the target's worker crate into the link: the engine serves through the `WorkerFactory`
// registry and never names a concrete worker, so without this reference the linker may drop the
// factory's `inventory::submit!`.
#[cfg(feature = "metal-base")]
extern crate scratchy_target_metal_worker as _;

use clap::Parser;
use scratchy_cli::args::Cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file if present (before clap parses, so env vars are visible).
    dotenvy::dotenv().ok();
    let cli = Cli::parse();
    let r = scratchy_cli::run::<Cli>(cli.command).await;
    // A run a termination signal ended exits as the signal's default action would have
    // (128 + SIGINT), not as a failure.
    if scratchy_cli::interrupted(&r) {
        eprintln!("interrupted");
        std::process::exit(130);
    }
    r
}
