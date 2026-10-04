//! Run bridge two on a fixture's ttir golden and write the resulting KTIR as TEXT.
//!
//! ```bash
//! cargo run --offline --example emit_ktir -- attention_flash_noncausal /tmp/out.ktir.mlir
//! ```
//!
//! THE TEXT IS A TEST INSTRUMENT, and this example is part of that instrument rather
//! than an exception to it. The compile path is value-in/value-out; this exists so the
//! END-TO-END check can hand the port's own KTIR to `triton-superdsc-lower` (which
//! still takes text -- see the crate docs' note on the remaining work) and then to
//! `dxp_standalone`, which is the only way to prove the port's output is not merely
//! structurally equal to the C++'s but downstream-VALID.

use std::path::PathBuf;

use triton_ktir::text::{parse, print};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("use: emit_ktir <config> <out.ktir.mlir>");
        std::process::exit(2);
    }
    let goldens = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate sits one level under crates/triton")
        .join("test-goldens/ktir")
        .join(&args[0]);

    let ttir = std::fs::read_to_string(goldens.join("0_ttir.mlir"))
        .unwrap_or_else(|e| panic!("cannot read {}'s ttir: {e}", args[0]));
    let grid: Vec<i64> = std::fs::read_to_string(goldens.join("grid.txt"))
        .unwrap_or_else(|e| panic!("cannot read {}'s grid: {e}", args[0]))
        .trim()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("grid.txt is a comma-separated integer list"))
        .collect();

    let mut m = parse::parse(&ttir).unwrap_or_else(|e| panic!("ttir: {e}"));
    triton_ktir::make_ktir(&mut m, &grid).unwrap_or_else(|e| panic!("make_ktir refused: {e}"));

    // The census, so the number of ops that went out the far end is on the record
    // beside whatever the next stage reports.
    eprintln!("{}: grid={grid:?}", args[0]);
    for (name, count) in m.census() {
        eprintln!("  {count:>4}  {name}");
    }
    std::fs::write(&args[1], print::print(&m))
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", args[1]));
    eprintln!("wrote {}", args[1]);
}
