// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `scr model convert` subcommand — weight format conversion (stub).

use anyhow::Result;
use tracing::info;

use crate::args::ConvertArgs;

/// Run the convert subcommand.
pub async fn run_convert(args: ConvertArgs) -> Result<()> {
    scratchy_core_common::telemetry::init_tracing("info");

    info!(
        "Convert: {} -> {} (dtype={})",
        args.input, args.output, args.dtype
    );
    eprintln!(
        "Weight conversion is not yet implemented.\n\
         Input:  {}\n\
         Output: {}\n\
         DType:  {}",
        args.input, args.output, args.dtype,
    );

    Ok(())
}
