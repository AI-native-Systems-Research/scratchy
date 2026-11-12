// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! The scratchy CLI's commands, shared by every target's binary: `scr` (metal) and a target's own
//! (`spyre`) each parse their command line, link their target's worker, and run the shared
//! subcommands through [`run`]. Nothing here names a worker: the engine serves through the one
//! the binary links (`linked_factory`).

pub mod args;
pub mod commands;
mod http;

use args::Commands;

/// Whether a termination signal ended the run through its normal teardown (`Interrupted`).
#[cfg(feature = "chat")]
pub fn interrupted(r: &anyhow::Result<()>) -> bool {
    r.as_ref()
        .is_err_and(|e| e.is::<scratchy_serving_api::llm::Interrupted>())
}

/// Without the in-process engine, nothing ends a run on a signal but the signal itself.
#[cfg(not(feature = "chat"))]
pub fn interrupted(_: &anyhow::Result<()>) -> bool {
    false
}

/// Run one shared subcommand. `C` is the binary's own top-level command — what its completion
/// scripts are generated from, under its own name.
pub async fn run<C: clap::CommandFactory>(command: Commands) -> anyhow::Result<()> {
    match command {
        #[cfg(feature = "serve")]
        Commands::Serve(args) => commands::serve::run_serve(*args).await,
        #[cfg(feature = "claude")]
        Commands::Launch(cmd) => {
            use crate::args::LaunchSubcommand;
            match cmd.command {
                LaunchSubcommand::Claude(args) => commands::launch::run_launch_claude(args).await,
            }
        }
        #[cfg(feature = "bench")]
        Commands::Bench(cmd) => commands::bench::run_bench(*cmd).await,
        #[cfg(any(feature = "chat", feature = "serve"))]
        Commands::Batch(args) | Commands::RunBatch(args) => commands::batch::run_batch(args).await,
        Commands::Chat(args) => commands::chat::run_chat(args).await,
        Commands::CollectEnv(args) => commands::collect_env::run_collect_env(args).await,
        Commands::Complete(args) => commands::chat::run_complete(args).await,
        Commands::Model(cmd) => {
            use crate::args::{CacheSubcommand, ModelSubcommand};
            match cmd.command {
                ModelSubcommand::Convert(args) => commands::convert::run_convert(args).await,
                ModelSubcommand::Pull(args) => commands::pull::run_pull(args).await,
                ModelSubcommand::Ls(args) | ModelSubcommand::List(args) => {
                    commands::model::run_model_list(args).await
                }
                ModelSubcommand::Rm(args) => commands::model::run_model_rm(args).await,
                ModelSubcommand::Cache(cache_cmd) => match cache_cmd.command {
                    CacheSubcommand::Clean(args) => commands::cache::run_cache_clean(args).await,
                    CacheSubcommand::Inspect(args) => {
                        commands::cache::run_cache_inspect(args).await
                    }
                },
                #[cfg(feature = "metal")]
                ModelSubcommand::Info(args) => commands::model_info::run_info(args).await,
                #[cfg(feature = "model")]
                ModelSubcommand::Names(args) => commands::model::run_model_names(args).await,
            }
        }
        Commands::Completions(args) => {
            commands::completions::run_completions(C::command(), args.shell, args.install)
        }
    }
}
