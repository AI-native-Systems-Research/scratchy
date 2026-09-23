// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr bench` — benchmarking tools for the scratchy inference engine.
//!
//! This crate provides latency, serving, and throughput benchmarks that can be
//! driven from any CLI frontend. The canonical entry point is [`run_bench`].

mod args;
pub(crate) mod datasets;
#[cfg(feature = "datasets")]
mod hotpotqa;
pub(crate) mod http;
mod latency;
mod longbench;
#[cfg(feature = "datasets")]
mod msmarco;
#[cfg(feature = "datasets")]
mod multihop;
mod musique;
mod niah;
mod ragcsv;
mod ruler;
mod serve;
mod spans;
mod startup;
mod startup_exec;
mod sweep;
mod throughput;

pub use args::{
    BenchCommand, BenchCommands, BenchHotpotqaArgs, BenchLatencyArgs, BenchLongbenchArgs,
    BenchMsmarcoArgs, BenchMultihopArgs, BenchMusiqueArgs, BenchNiahArgs, BenchRagcsvArgs,
    BenchRulerArgs, BenchServeArgs, BenchSpansArgs, BenchStartupArgs, BenchThroughputArgs,
    SweepCommand, SweepCommands, SweepServeArgs, SweepStartupArgs,
};

/// UTC `YYYYMMDD<sep>HHMMSS` tag for auto-generated result filenames.
pub(crate) fn timestamp_filename_tag(sep: &str) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = now / 86_400;
    let secs_of_day = now % 86_400;

    // Civil calendar algorithm (simplified Euclidean); mirrors
    // scratchy-serving-api's `days_to_ymd` (kept separate — this crate
    // doesn't depend on that one).
    let is_leap = |y: u64| y.is_multiple_of(4) && (!y.is_multiple_of(100) || y.is_multiple_of(400));
    let mut year = 1970u64;
    let mut days_left = days;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days_left < days_in_year {
            break;
        }
        days_left -= days_in_year;
        year += 1;
    }
    let month_days: [u64; 12] = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1u64;
    for (i, &md) in month_days.iter().enumerate() {
        if days_left < md {
            month = i as u64 + 1;
            break;
        }
        days_left -= md;
    }
    let day = days_left + 1;
    let (hour, minute, second) = (
        secs_of_day / 3600,
        (secs_of_day / 60) % 60,
        secs_of_day % 60,
    );

    format!("{year:04}{month:02}{day:02}{sep}{hour:02}{minute:02}{second:02}")
}

/// Format a rate as `it/s` (fast) or `s/it` (slow), matching Python tqdm style.
pub(crate) fn fmt_tqdm_rate(state: &indicatif::ProgressState, w: &mut dyn std::fmt::Write) {
    let per_sec = state.per_sec();
    if per_sec >= 1.0 {
        write!(w, "{per_sec:.2}it/s").unwrap();
    } else if per_sec > 0.0 {
        write!(w, "{:.2}s/it", 1.0 / per_sec).unwrap();
    } else {
        write!(w, "?it/s").unwrap();
    }
}

/// Dispatch bench subcommands.
pub async fn run_bench(cmd: BenchCommand) -> anyhow::Result<()> {
    match cmd.command {
        BenchCommands::Latency(args) => {
            // LLM creates its own tokio runtime, so we must exit the
            // current one before calling run_bench_latency.
            tokio::task::spawn_blocking(move || latency::run_bench_latency(*args)).await??;
            Ok(())
        }
        BenchCommands::Serve(args) => serve::run_bench_serve(args).await,
        BenchCommands::Startup(args) => {
            tokio::task::spawn_blocking(move || startup::run_bench_startup(args)).await??;
            Ok(())
        }
        BenchCommands::Sweep(cmd) => {
            tokio::task::spawn_blocking(move || sweep::run_bench_sweep(cmd)).await??;
            Ok(())
        }
        BenchCommands::Throughput(args) => {
            tokio::task::spawn_blocking(move || throughput::run_bench_throughput(args)).await??;
            Ok(())
        }
        BenchCommands::Spans(args) => {
            tokio::task::spawn_blocking(move || spans::run_bench_spans(args)).await??;
            Ok(())
        }
        BenchCommands::Niah(args) => {
            tokio::task::spawn_blocking(move || niah::run_bench_niah(args)).await??;
            Ok(())
        }
        BenchCommands::Ruler(args) => {
            tokio::task::spawn_blocking(move || ruler::run_bench_ruler(args)).await??;
            Ok(())
        }
        BenchCommands::Ragcsv(args) => {
            tokio::task::spawn_blocking(move || ragcsv::run_bench_ragcsv(args)).await??;
            Ok(())
        }
        #[cfg(feature = "datasets")]
        BenchCommands::Multihop(args) => {
            tokio::task::spawn_blocking(move || multihop::run_bench_multihop(args)).await??;
            Ok(())
        }
        BenchCommands::Musique(args) => {
            tokio::task::spawn_blocking(move || musique::run_bench_musique(args)).await??;
            Ok(())
        }
        #[cfg(feature = "datasets")]
        BenchCommands::Hotpotqa(args) => {
            tokio::task::spawn_blocking(move || hotpotqa::run_bench_hotpotqa(args)).await??;
            Ok(())
        }
        #[cfg(feature = "datasets")]
        BenchCommands::Msmarco(args) => {
            tokio::task::spawn_blocking(move || msmarco::run_bench_msmarco(args)).await??;
            Ok(())
        }
        // These three subcommands parse parquet datasets; the arrow/parquet
        // subtree is only compiled under `--features datasets`. Without it the
        // CLI still accepts the subcommand but fails fast with a rebuild hint.
        #[cfg(not(feature = "datasets"))]
        BenchCommands::Multihop(_) | BenchCommands::Hotpotqa(_) | BenchCommands::Msmarco(_) => {
            anyhow::bail!(
                "this `scr bench` subcommand reads parquet datasets; rebuild with `--features datasets`"
            )
        }
        BenchCommands::Longbench(args) => {
            tokio::task::spawn_blocking(move || longbench::run_bench_longbench(args)).await??;
            Ok(())
        }
    }
}
