// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr-tui` — standalone binary entry point for the interactive terminal
//! UI. Separate from the `scr` CLI/workspace because goose (this crate's
//! agent brain) is a git dependency not published to crates.io.

use clap::Parser;

/// Interactive terminal UI: run a model in-process behind goose's agent.
#[derive(Parser, Debug)]
#[command(name = "scr-tui", version, override_usage = "scr-tui [MODEL] [PROMPT]")]
struct Args {
    /// Model to run: local path or HuggingFace model ID.
    #[arg(default_value = "mlx-community/Llama-3.1-8B-Instruct-4bit")]
    model: String,
    /// Optional one-shot prompt: run a single headless turn and exit.
    prompt: Option<String>,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    scratchy_tui::run(args.model, args.prompt).await
}
