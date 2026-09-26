// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! `scr bundle run` — the CLI face of [`scratchy_target_spyre::bundle_run`].
//!
//! The work lives in the spyre target crate, because the executor, the bundle types and the device
//! libraries are there. This is argument plumbing and nothing else.

use anyhow::Result;

use crate::args::BundleRunArgs;

pub fn run(args: BundleRunArgs) -> Result<()> {
    scratchy_target_spyre::bundle_run::run(
        std::path::Path::new(&args.dir),
        &args.input,
        &args.output,
    )
}
