// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! Bake one fixture's `ktir-superdsc` descriptors into a `dxp_standalone` input directory.
//!
//! ```bash
//! cargo run --offline --example bake -- <config> <grid-csv> <program> <out-dir>
//! ```
//!
//! Same reading as `drive` — the ttir golden, `make_ktir`, `to_ktir`, the program-derived
//! `BundleLayout`, the chosen body — and then [`triton_ktir_superdsc::bake::write_dir`] puts the
//! descriptors where the validation gate reads them:
//!
//! ```bash
//! dxp_standalone -d <out-dir> -b sentient
//! ```
//!
//! ⛔ COPY THE DIRECTORY BEFORE EVERY RUN. `dxp_standalone` WRITES `spyreCodeDir/` into its input
//! directory, so a second run is not a rerun of the first. `spyreCodeDir/` appearing is also the
//! success signal — success is otherwise silent, with no output lines at all.
//!
//! ⛔ AND `-b sentient`, NEVER `-b senulator`: the latter aborts on the tool's OWN shipped fixtures,
//! so a failure under it says nothing about the input.
//!
//! `bake` deliberately does NOT print a verdict about the descriptors. It reports what it wrote; the
//! only thing that can judge them is the scheduler.

use std::path::PathBuf;

use ktir_superdsc::ktir_node::{Elementwise, Program};

fn program_from(s: &str) -> Option<Program> {
    Some(match s {
        "rmsnorm" => Program::RmsNorm,
        "matmul" => Program::Matmul,
        "silumul" => Program::SiluMul,
        "scalarmul" => Program::ScalarMul,
        "lmlast" => Program::LmLast,
        "transpose" => Program::Transpose,
        other => {
            let kind = other.strip_prefix("ew:")?;
            Program::Elementwise(match kind {
                "add" => Elementwise::Add,
                "mul" => Elementwise::Mul,
                "sub" => Elementwise::Sub,
                "silu" => Elementwise::Silu,
                "gelu" => Elementwise::Gelu,
                "exp" => Elementwise::Exp,
                "rsqrt" => Elementwise::Rsqrt,
                "sqrt" => Elementwise::Sqrt,
                "abs" => Elementwise::Abs,
                "reciprocal" => Elementwise::Reciprocal,
                "sigmoid" => Elementwise::Sigmoid,
                "tanh" => Elementwise::Tanh,
                "mish" => Elementwise::Mish,
                "realdiv" => Elementwise::RealDiv,
                "maximum" => Elementwise::Maximum,
                "minimum" => Elementwise::Minimum,
                _ => return None,
            })
        }
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        eprintln!("use: bake <config> <grid-csv> <program> <out-dir>");
        std::process::exit(2);
    }
    let (config, program_s, out_dir) = (&args[0], &args[2], PathBuf::from(&args[3]));
    let grid: Vec<i64> = args[1]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().expect("grid is a comma-separated integer list"))
        .collect();
    let Some(program) = program_from(program_s) else {
        eprintln!("unknown program `{program_s}`");
        std::process::exit(2);
    };

    // ⛔ EXIT NON-ZERO ON EVERY REFUSAL. `drive` prints one line and returns 0 because it is a
    // survey; this writes a directory something else will validate, and a bake that refused must not
    // look like a bake that wrote nothing yet succeeded.
    macro_rules! refused {
        ($stage:expr, $err:expr) => {{
            println!("REFUSED  {config:32} {program_s:10} {:<12} {}", $stage, $err);
            std::process::exit(1);
        }};
    }

    let ttir_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate sits one level under crates/triton")
        .join("test-experiment1/ktir")
        .join(format!("{config}.ttir.mlir"));
    let ttir = std::fs::read_to_string(&ttir_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", ttir_path.display()));

    let mut m = match triton_ktir::text::parse::parse(&ttir) {
        Ok(m) => m,
        Err(e) => refused!("ttir-parse", e),
    };
    if let Err(e) = triton_ktir::make_ktir(&mut m, &grid) {
        refused!("make_ktir", e);
    }
    if let Err(e) = triton_ktir::passes::to_ktir::run(&mut m, &grid) {
        refused!("to_ktir", e);
    }
    let node = match triton_ktir_superdsc::node_for(&m, program) {
        Ok(n) => n,
        Err(e) => refused!(e.stage, e.message),
    };
    let ops = match triton_ktir_superdsc::emit_node(&node) {
        Ok(ops) => ops,
        Err(e) => refused!(e.stage, e.message),
    };
    let written = match triton_ktir_superdsc::bake::write_dir(&out_dir, &ops) {
        Ok(w) => w,
        Err(e) => refused!(e.stage, e.message),
    };
    // The op names in execute order, because that is the join between `bundle.mlir`'s Nth execute,
    // `sdsc_N.json` and the `{N}_{op_name}` key inside it — the one thing a reader of a dxp failure
    // needs in order to know which descriptor it is about. A `time > 1` op is PRE-UNROLLED into
    // `time` consecutive files (see `bake::trips`), so the index advances by that, not by one.
    let mut i = 0usize;
    for e in &ops {
        let n = e.time.max(1);
        for t in 0..n {
            println!(
                "sdsc_{i}.json  {}{}",
                e.op_name,
                if n > 1 {
                    format!("  (trip {t} of {n})")
                } else {
                    String::new()
                }
            );
            i += 1;
        }
    }
    println!(
        "BAKED    {config:32} {program_s:10} {} file(s) -> {}",
        written.len(),
        out_dir.display()
    );
}
