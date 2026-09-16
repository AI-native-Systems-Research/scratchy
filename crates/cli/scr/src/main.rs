// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! vLLM CLI binary — standalone Rust inference engine.
//!
//! Usage:
//!   scr serve --model <path-or-hf-id>
//!   scr bench --model <path-or-hf-id>
//!   scr model convert --input <dir> --output <dir>

mod args;
mod commands;
mod http;

use clap::Parser;

use crate::args::{Cli, Commands};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let r = run().await;
    // ⛔⛔⛔ SENLIB SEGFAULTS IN ITS OWN `atexit` HANDLER, SO THIS PROCESS DOES NOT RUN ONE.
    //
    // MEASURED 2026-08-14 under `MALLOC_PERTURB_=165` (which fills freed memory, turning a use-after-free
    // into an immediate fault instead of a corrupted chunk header that trips much later):
    // ```
    // Batch complete: 4 requests (4 succeeded, 0 failed) in 21.01s
    // UniProcExecutor: shutting down
    // Signal Received: 11 (Segmentation fault)
    // *** BACKTRACE ***
    //   libsenlib-dd2.so(+0x44c8df)
    //   libc(+0x411c0)
    //   libsenlib-dd2.so(senlib::v2::CleanupRegistry::ExecuteAll+0xbd)
    //   libc(__cxa_finalize+0x161)
    //   libsenlib-dd2.so(+0x44c7c7)
    // ```
    // `__cxa_finalize` → senlib's static destructor → its own cleanup registry → fault. Nothing of ours is
    // on that stack, and every frame is in a library we do not build. Without `MALLOC_PERTURB_` the same
    // corruption surfaces later as the glibc heap aborts this run has ended with for weeks
    // (`corrupted size vs. prev_size`, `malloc(): mismatching next->prev_size`) — one cause, three faces.
    //
    // ⭐ IT WAS NOT COSMETIC. Those aborts land in a SIGABRT handler that calls malloc to print a backtrace,
    // on the heap it is reporting as corrupt, so it deadlocks: 23 threads on a futex, holding `/dev/vfio`.
    // One run held the card for over an hour, and every run after it died `flex runtime init failed (rc=-2)`
    // — which reads exactly like a code fault and is not one. SIGINT and SIGQUIT are ignored by that process
    // and SIGTERM is caught-and-blocked, so nothing short of SIGKILL recovers it.
    //
    // `_exit` skips `atexit`/`__cxa_finalize` and nothing else: OUR destructors have already run (every
    // value the command owned was dropped when it returned), the output file is written and closed, and both
    // streams are flushed below. The kernel then closes the vfio fd, which is what resets the device — the
    // same thing that happened when this process crashed here, except deterministic, with a truthful exit
    // code and no signal handler to hang in.
    //
    // ⛔ GATED ON THE FEATURE THAT LOADS SENLIB. metal and cuda have no such library and their `atexit` is
    // fine; skipping it there would silence a real bug rather than route around someone else's.
    #[cfg(feature = "spyre-hw")]
    {
        use std::io::Write;
        let code = match &r {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("Error: {e:?}");
                1
            }
        };
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        // SAFETY: `_exit` is async-signal-safe and takes no locks. Everything this process still owns is
        // memory and file descriptors, both reclaimed by the kernel.
        unsafe { libc::_exit(code) };
    }
    #[cfg(not(feature = "spyre-hw"))]
    r
}

async fn run() -> anyhow::Result<()> {
    // Load .env file if present (before clap parses, so env vars are visible).
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    match cli.command {
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
        Commands::Bench(cmd) => commands::bench::run_bench(cmd).await,
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
                #[cfg(any(feature = "cuda", feature = "metal"))]
                ModelSubcommand::Info(args) => commands::model_info::run_info(args).await,
                #[cfg(feature = "model")]
                ModelSubcommand::Names(args) => commands::model::run_model_names(args).await,
            }
        }
        Commands::Completions(args) => {
            commands::completions::run_completions(args.shell, args.install)
        }
    }
}
