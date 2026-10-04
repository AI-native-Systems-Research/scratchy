// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! Drive one fixture through `ktir-superdsc`'s lowering.
//!
//! ```bash
//! cargo run --offline --example drive -- <config> <grid-csv> <program>
//! ```
//!
//! `<program>` is stated, not inferred — see the crate docs on why the node kind is an argument
//! during bring-up. Accepted: `rmsnorm`, `matmul`, `silumul`, `scalarmul`, `lmlast`, `transpose`,
//! `ew:add|mul|sub|silu|gelu|exp|rsqrt|sqrt|abs|reciprocal|sigmoid|tanh|mish|realdiv|maximum|minimum`,
//! and the two that cross the model-geometry const door, `rope` and `attn`.
//!
//! `rope` and `attn` need `HEAD_DIM` / `H` / `GQA` / `N_TOK`, and this driver READS them out of
//! `test/experiment1/index.json`'s `constexprs` for the named configuration — the fixture's own
//! `tl.constexpr` values, recorded there by `run_experiment1.py` from each fixture's `constexprs()`.
//! A missing key is a REFUSAL naming the config and the key: a defaulted head count is a wrong GQA
//! grouping with nothing downstream to catch it.
//!
//! Environment:
//! * `KTIR_DUMP=1` — the handed-off `IRFunction`, one op per line.
//! * `KTIR_LAYOUT_DUMP=1` — the derived `BundleLayout` (placements + scale registry).
//! * `KTIR_ADDR_DUMP=1` — each emitted descriptor's per-core HBM start addresses.
//! * `KTIR_NO_LAYOUT=1` — drive with `layout: None`, `ktir-superdsc`'s unit-test addressing. The
//!   CONTROL for "did the layout change this verdict?".
//! * `KTIR_ROPE_ROWS=token-major` — state that the roped plane is token-major. See
//!   [`triton_ktir_superdsc::RopeRows`]: the fixtures are HEAD-major and the two are
//!   indistinguishable by shape, so `rope` refuses by default and this override exists only to
//!   measure the door BEHIND that refusal. It is a lie about our fixtures; a result obtained under
//!   it is not a bake.

use std::path::PathBuf;

use ktir_superdsc::ktir_node::{Elementwise, Program};
use triton_ktir_superdsc::RopeRows;

fn program_from(s: &str) -> Option<Program> {
    Some(match s {
        "rmsnorm" => Program::RmsNorm,
        "matmul" => Program::Matmul,
        "silumul" => Program::SiluMul,
        "scalarmul" => Program::ScalarMul,
        "lmlast" => Program::LmLast,
        "transpose" => Program::Transpose,
        "rope" => Program::Rope,
        "attn" => Program::Attn,
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
    if args.len() != 3 {
        eprintln!("use: drive <config> <grid-csv> <program>");
        std::process::exit(2);
    }
    let (config, program_s) = (&args[0], &args[2]);
    let grid: Vec<i64> = args[1]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().expect("grid is a comma-separated integer list"))
        .collect();
    let Some(program) = program_from(program_s) else {
        eprintln!("unknown program `{program_s}`");
        std::process::exit(2);
    };

    let ttir_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate sits one level under crates/triton")
        .join("test-experiment1/ktir")
        .join(format!("{config}.ttir.mlir"));
    let ttir = std::fs::read_to_string(&ttir_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", ttir_path.display()));

    macro_rules! refused {
        ($stage:expr, $err:expr) => {{
            println!("REFUSED  {config:32} {program_s:10} {:<12} {}", $stage, $err);
            return;
        }};
    }

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
    // `KTIR_DUMP=1` prints the handed-off function as their structural readings see it. The
    // instrument, not the path — every refusal below is about the SHAPE of this list.
    if std::env::var("KTIR_DUMP").is_ok() {
        print!("{}", triton_ktir_superdsc::dump(&node));
    }
    // THE LAYOUT IS DERIVED FROM THE PROGRAM (`triton_ktir_superdsc::layout`), because two bodies
    // cannot emit without one: `rmsnorm`/`scalarmul` resolve their constant to a slot in
    // `BundleLayout::scalarmul_scales` through `scale_slot`, which is `layout.and_then(..)` and so
    // always answers "absent" for `None`.
    //
    // `KTIR_NO_LAYOUT=1` selects the old `None` arm — `ktir-superdsc`'s unit-test addressing
    // (per-op `segment_base(arg_index)`, no registry). It is the CONTROL: it makes "did the layout
    // change this fixture's verdict?" a question this one binary answers.
    let layout = if std::env::var("KTIR_NO_LAYOUT").is_ok() {
        None
    } else {
        match triton_ktir_superdsc::layout::for_node(&node) {
            Ok(l) => Some(l),
            Err(e) => refused!(e.stage, e.message),
        }
    };
    // `KTIR_LAYOUT_DUMP=1` prints the plan the addresses come out of: every placement's
    // `(segment, offset, size)` and the scale registry in slot order. The instrument for
    // "which segment did t1 get, and is the epsilon in slot 0" — the two facts a wrong
    // address here would come from.
    if std::env::var("KTIR_LAYOUT_DUMP").is_ok() {
        match &layout {
            None => println!("layout: none (KTIR_NO_LAYOUT)"),
            Some(l) => {
                for (tid, p) in &l.placements {
                    println!(
                        "place t{tid:<12} seg{} off={:<10} size={:<10} role={:?}",
                        p.segment, p.offset, p.size, p.role
                    );
                }
                for (i, s) in l.scalarmul_scales.iter().enumerate() {
                    println!("scale[{i}] = {s:e}  -> const t{}", u32::MAX - 20 - i as u32);
                }
                println!("synth.next = {}", l.synth.borrow().next);
            }
        }
    }
    // THE GEOMETRY DOOR. `rope` cannot be dispatched from a runtime `Program`, so it gets its own
    // entry point and the geometry is READ from the fixture's recorded constexprs. Note
    // `rows_are_requests: false`, and it is a reading, not a default: the fixtures sweep
    // `N_TOK` CONSECUTIVE POSITIONS of one sequence (`rope.py`'s `offs_m = off_h * N_TOK +
    // start_m * BLOCK_M`), never one row per request.
    //
    // ⛔ ATTENTION HAS NO ARM HERE, ON PURPOSE. The old `Program::Attn` arm routed through
    // `drive_attn` → `attn_at::<NQH, NKVH, HD>`, scratchy's prebuilt fragment-assembly lowering
    // selected by constexpr-geometry match — pattern matching instead of lowering the given
    // code. Removed; attention is `emit_node_with`'s problem like every other program (the
    // whole-function door in `bake_py` is the from-source path that runs on card).
    let geometry_driven = match program {
        Program::Rope => {
            let g = |k: &str| match constexpr_int(config, k) {
                Ok(v) => Ok(v),
                Err(e) => Err(e),
            };
            let (hd, h, n_tok) = match (g("HEAD_DIM"), g("H"), g("N_TOK")) {
                (Ok(a), Ok(b), Ok(c)) => (a, b, c),
                (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => refused!("constexpr", e),
            };
            let rows = match std::env::var("KTIR_ROPE_ROWS").as_deref() {
                Ok("token-major") => RopeRows::TokenMajor,
                // The fixture's own layout. `rope.py` builds `x` as `[H * N_TOK, HEAD_DIM]` and
                // indexes row `off_h * N_TOK + start_m * BLOCK_M` — head outermost.
                _ => RopeRows::HeadMajor,
            };
            Some(triton_ktir_superdsc::drive_rope(
                &node,
                hd,
                n_tok,
                h * hd,
                rows,
                false,
                layout.as_ref(),
            ))
        }
        _ => None,
    };
    let emitted =
        geometry_driven.unwrap_or_else(|| triton_ktir_superdsc::emit_node_with(&node, layout.as_ref()));
    match emitted {
        Err(e) => refused!(e.stage, e.message),
        Ok(ops) => {
            // `KTIR_ADDR_DUMP=1` prints each descriptor's per-core START ADDRESSES
            // (`startAddressCoreCorelet_`), which is the ONE observable that proves the layout
            // reached the emission: every one must be `SEGMENT_OFFSETS[seg] + intra` for the
            // segment the placement above states. Reading the plan and reading the addresses are
            // two different checks — a plan can be right and still not be the one used.
            if std::env::var("KTIR_ADDR_DUMP").is_ok() {
                for e in &ops {
                    let Some(op) = &e.op else { continue };
                    for m in &op.dscs_ {
                        for dsc in m.values() {
                            for n in &dsc.scheduleTree_ {
                                let a: Vec<&str> = n
                                    .startAddressCoreCorelet_
                                    .data_
                                    .values()
                                    .map(|s| s.as_str())
                                    .collect();
                                println!(
                                    "addr {:<22} {:<28} {:<4} {a:?}",
                                    e.op_name, n.name_, n.component_
                                );
                            }
                        }
                    }
                }
            }
            println!(
                "EMITTED  {config:32} {program_s:10} {:<12} ops={}",
                "", ops.len()
            )
        }
    }
}

/// ONE INTEGER `tl.constexpr` OF ONE CONFIGURATION, read out of `test/experiment1/index.json`.
///
/// ⛔ THIS IS A READING, NOT A TABLE. The alternative was a `match config { "rope_kv8" => 128, .. }`
/// in this driver, which would be a second statement of the head geometry, free to drift from the
/// kernel that is compiled with it. `run_experiment1.py` writes `constexprs` straight from each
/// fixture's own `constexprs()` (`cases()`, `constexprs=ce`; serialized at
/// `"constexprs": {k: repr(v) if float else v}`), so this file IS the fixture's constexpr set.
///
/// ⛔ AND IT IS A LINE SCAN BECAUSE `serde_json` IS NOT A DEPENDENCY OF THIS CRATE, and its
/// `Cargo.toml` was outside this change. It is fail-closed rather than lenient: `index.json` is
/// machine-written with `json.dump(indent=2, sort_keys=True)`, so a configuration's `constexprs`
/// block always precedes its `"name"` key; this walk collects the most recent block and requires one
/// to have been seen INSIDE the same entry. A reordering, a reformat, a missing key or a non-integer
/// value all produce an `Err` naming the config and the key — never a number.
fn constexpr_int(config: &str, key: &str) -> Result<u32, String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate sits one level under crates/triton")
        .join("test-experiment1/index.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut in_block = false;
    // The constexprs of the entry currently being scanned — cleared at each entry boundary so a
    // block can never be attributed to a later configuration.
    let mut here: Vec<(String, String)> = Vec::new();
    for line in text.lines() {
        let t = line.trim_end();
        if t == "  {" {
            here.clear();
            in_block = false;
        } else if t == "    \"constexprs\": {" {
            here.clear();
            in_block = true;
        } else if in_block && (t == "    }" || t == "    },") {
            // WITH OR WITHOUT THE TRAILING COMMA. `constexprs` sorts before `fixture`, so its close
            // is `    },`; only a last key would close bare. Matching only the bare form left
            // `in_block` true for the rest of the file and every lookup answered "no configuration
            // named X" — a refusal, which is why this was a wasted run and not a wrong number.
            in_block = false;
        } else if in_block {
            let s = t.trim().trim_end_matches(',');
            if let Some((k, v)) = s.split_once("\": ") {
                here.push((k.trim_start_matches('"').to_string(), v.to_string()));
            }
        } else if t == format!("    \"name\": \"{config}\",") {
            if here.is_empty() {
                return Err(format!(
                    "{}: configuration `{config}` states no `constexprs` block before its `name` \
                     key — this walk relies on `json.dump(sort_keys=True)`'s ordering and will not \
                     guess. Read the geometry from the fixture directly.",
                    path.display()
                ));
            }
            let Some((_, v)) = here.iter().find(|(k, _)| k == key) else {
                return Err(format!(
                    "{config}: `constexprs` states no `{key}`. It has {:?}. The head geometry is the \
                     fixture's own `tl.constexpr` and this driver will not default it — a wrong head \
                     count is a wrong GQA grouping with nothing downstream to catch it.",
                    here.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>()
                ));
            };
            return v.parse::<u32>().map_err(|_| {
                format!(
                    "{config}: `constexprs.{key}` is `{v}`, not a non-negative integer — the const \
                     door takes a `u32`, so a float or a string here is a refusal, not a cast"
                )
            });
        }
    }
    Err(format!(
        "{}: no configuration named `{config}`",
        path.display()
    ))
}
