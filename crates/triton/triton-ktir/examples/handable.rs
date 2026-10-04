// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! CAN WE HAND THIS CONFIGURATION TO THEIR LOWERING? One line per configuration.
//!
//! ```bash
//! cargo run --offline --example handable -- <config> <grid-csv>
//! ```
//!
//! The check is CONSTRUCTION, not comparison. The first version of this file diffed our
//! variant set against theirs, which answers a weaker question: two vocabularies can agree
//! name-for-name while the value still cannot be built, because a region's shape, a result
//! count or an attribute payload disagrees. So this runs the real boundary,
//! [`triton_ktir::handoff::lower`], and reports either the built function or the refusal
//! that stopped it, naming the stage.
//!
//! Every refusal is by name, so a REFUSED line is a work item rather than a mystery.

use std::path::PathBuf;

use ktir_core::arena::Arena;
use triton_ktir::text::parse;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("use: handable <config> <grid-csv>");
        std::process::exit(2);
    }
    let config = &args[0];
    let grid: Vec<i64> = args[1]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().expect("grid is a comma-separated integer list"))
        .collect();

    let ttir_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate sits one level under crates/triton")
        .join("test-experiment1/ktir")
        .join(format!("{config}.ttir.mlir"));
    let ttir = std::fs::read_to_string(&ttir_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", ttir_path.display()));

    macro_rules! refused {
        ($stage:expr, $err:expr) => {{
            println!("REFUSED  {config:34} {:<11} {}", $stage, $err);
            return;
        }};
    }

    let mut m = match parse::parse(&ttir) {
        Ok(m) => m,
        Err(e) => refused!("ttir-parse", e),
    };
    if let Err(e) = triton_ktir::make_ktir(&mut m, &grid) {
        refused!("make_ktir", e);
    }
    if let Err(e) = triton_ktir::passes::to_ktir::run(&mut m, &grid) {
        refused!("to_ktir", e);
    }

    // THE BOUNDARY. Past here every value is theirs.
    let arena = Arena::new();
    match triton_ktir::passes::to_ktir_emit::lower(&m, &arena) {
        Err(e) => refused!("handoff", e),
        Ok(f) => println!(
            "HANDED   {config:34} {:<11} name=@{} args={} grid={:?} ops={} deep={}",
            "",
            f.name,
            f.arguments.len(),
            f.grid,
            f.operations.len(),
            f.ops_deep().len()
        ),
    }
}
