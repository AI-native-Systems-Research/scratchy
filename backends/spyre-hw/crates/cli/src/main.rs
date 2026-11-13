// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `spyre` — the scratchy CLI on IBM Spyre: the shared commands (`scratchy_cli`), served through
//! the spyre worker, plus the card's own `spyre bundle run`.
//!
//! Usage:
//!   spyre serve --model <path-or-hf-id>
//!   spyre chat -m <path-or-hf-id> -q "..."

#[cfg(feature = "spyre-hw")]
mod bundle;

// Anchor the spyre worker crate into the link: the engine serves through the `WorkerFactory`
// registry and never names a concrete worker, so without this reference the linker may drop the
// factory's `inventory::submit!`.
#[cfg(any(feature = "spyre-hw", feature = "spyre-emu"))]
extern crate scratchy_target_spyre_worker as _;

// Anchor the tp alias into the build: `tp/<W>` is how the CLI names the sharded worlds
// the bake emits, and resolver v2 drops the features of an optional dependency nothing
// references — without this anchor every TP build silently bakes one card.
#[cfg(feature = "tp")]
extern crate tp as _;

use clap::{Parser, Subcommand};

/// scratchy on IBM Spyre.
#[derive(Parser, Debug)]
#[command(name = "spyre", version, about, long_about = None)]
struct SpyreCli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    #[command(flatten)]
    Shared(scratchy_cli::args::Commands),
    /// Run a Spyre bundle this binary did not bake — a `dxp_standalone` output
    /// directory plus the producer's launch map. Every other launch path here needs
    /// the bundle baked into the binary at build time, which leaves an externally
    /// produced one unrunnable.
    #[cfg(feature = "spyre-hw")]
    Bundle(bundle::BundleCommand),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file if present (before clap parses, so env vars are visible).
    dotenvy::dotenv().ok();
    let cli = SpyreCli::parse();
    let r = match cli.command {
        Command::Shared(command) => scratchy_cli::run::<SpyreCli>(command).await,
        #[cfg(feature = "spyre-hw")]
        Command::Bundle(command) => bundle::run(command),
    };
    // A run a termination signal ended exits as the signal's default action would have
    // (128 + SIGINT), not as a failure.
    let failed = if scratchy_cli::interrupted(&r) {
        130
    } else {
        1
    };
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
    // ⛔ GATED ON THE TIER THAT LOADS SENLIB. The emulator has no such library and its `atexit` is
    // fine; skipping it there would silence a real bug rather than route around someone else's.
    #[cfg(feature = "spyre-hw")]
    {
        use std::io::Write;
        let code = match &r {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("Error: {e:?}");
                failed
            }
        };
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        // SAFETY: `_exit` is async-signal-safe and takes no locks. Everything this process still owns is
        // memory and file descriptors, both reclaimed by the kernel.
        unsafe { libc::_exit(code) };
    }
    #[cfg(not(feature = "spyre-hw"))]
    if scratchy_cli::interrupted(&r) {
        eprintln!("interrupted");
        std::process::exit(failed);
    }
    #[cfg(not(feature = "spyre-hw"))]
    r
}
