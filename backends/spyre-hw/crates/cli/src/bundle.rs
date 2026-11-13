// SPDX-License-Identifier: Apache-2.0

//! `spyre bundle run` — the CLI face of [`scratchy_target_spyre_card::bundle_run`].
//!
//! The work lives in the spyre target crate, because the executor, the bundle types and the device
//! libraries are there. This is argument plumbing and nothing else.

use anyhow::Result;
use clap::{Parser, Subcommand};

/// `spyre bundle <subcommand>`.
#[derive(Parser, Debug)]
pub struct BundleCommand {
    #[command(subcommand)]
    pub command: BundleSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum BundleSubcommand {
    /// Load, prepare and run one bundle directory on the card.
    Run(BundleRunArgs),
}

/// Arguments for `spyre bundle run`.
#[derive(Parser, Debug)]
pub struct BundleRunArgs {
    /// The bundle directory: `sdsc_*.json` + `placements.json` + `spyreCodeDir/`.
    pub dir: String,

    /// Fill a source tensor from a file: `--input t0=q.bin`. The file's length must equal that
    /// tensor's placement size — a short buffer would leave the tail whatever the device had there,
    /// which is a silently wrong answer rather than a failure.
    ///
    /// Three markers ride on this same list, so no argument plumbing changes:
    /// `stick:t<id>=<rows>` states the row count the descriptors address a tensor with (the device
    /// layout is stick-major and a row-major bind is a plausible wrong answer);
    /// `raw:t<id>` stages that tensor's bytes verbatim, with no IEEE-to-SEN fp16 re-encoding, which
    /// an int32 index buffer requires because the fp16 pass rewrites each 4-byte entry as two fp16
    /// mantissas; and `const:t<id>=<value>` binds a one-element fp16 scalar stated here rather than
    /// in a two-byte file, e.g. `const:t4294967275=12.0` for a ScalarMul scale.
    #[arg(long = "input")]
    pub input: Vec<String>,

    /// Read a tensor back after the run: `--output t2=o.bin`.
    #[arg(long = "output")]
    pub output: Vec<String>,
}

pub fn run(command: BundleCommand) -> Result<()> {
    match command.command {
        BundleSubcommand::Run(args) => scratchy_target_spyre_card::bundle_run::run(
            std::path::Path::new(&args.dir),
            &args.input,
            &args.output,
        ),
    }
}
