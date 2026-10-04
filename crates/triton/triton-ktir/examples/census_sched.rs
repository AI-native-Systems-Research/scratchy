//! Census of the full `to_ktir::run` pipeline on a fixture — the instrument the
//! golden tests compute `ours()` from, printed so a scaling question can be answered
//! from numbers rather than inference.
//!
//! ```bash
//! cargo run --offline --example census_sched -- attention_flash_noncausal
//! ```

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("use: census_sched <config> [out.ktir.mlir]");
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

    let mut m = triton_ktir::text::parse::parse(&ttir).unwrap_or_else(|e| panic!("ttir: {e}"));
    triton_ktir::make_ktir(&mut m, &grid).unwrap_or_else(|e| panic!("make_ktir refused: {e}"));
    triton_ktir::passes::to_ktir::run(&mut m, &grid)
        .unwrap_or_else(|e| panic!("to_ktir refused: {e}"));
    for (name, count) in m.census() {
        println!("{count:>5}  {name}");
    }
    // The text too, when a second argument names a path -- the same instrument
    // `emit_ktir` is for bridge two, for the full pipeline.
    if let Some(path) = std::env::args().nth(2) {
        std::fs::write(&path, triton_ktir::text::print::print(&m))
            .unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
        eprintln!("wrote {path}");
    }
}
