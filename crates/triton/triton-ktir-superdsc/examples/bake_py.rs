// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! THE WHOLE CHAIN, FROM TRITON SOURCE: `.py` -> ttir -> KTIR -> SuperDSC -> a dxp input dir.
//!
//! ```bash
//! cargo run --offline --example bake_py -- rmsnorm_granite /tmp/out
//! ```
//!
//! # WHY THIS EXISTS WHEN `bake` ALREADY DID
//!
//! `bake` and `drive` start from `test/experiment1/ktir/<config>.ttir.mlir` -- a CHECKED-IN
//! ttir golden. That skips the first bridge entirely and then reports the chain as having run.
//! Every dxp_standalone result this crate has produced so far was `ttir -> ... -> bake`, not
//! `triton -> ... -> bake`, and the difference is a whole stage: the Python front end, the
//! constexpr resolution, and `make_ttir`'s six passes.
//!
//! Nothing here is new capability -- `triton_frontend::codegen::compile` and
//! `triton_frontend::opt::make_ttir` have existed and are covered by
//! `triton-ktir/tests/pure_rust_ktir.rs`, which starts every one of its cases from source. This
//! only puts them in front of the bake, so the claim "it bakes" covers the leg it names.
//!
//! No file is read but the `.py`. The ttir goldens are not consulted, so a stale golden cannot
//! make this pass.

use std::path::PathBuf;

use ktir_superdsc::ktir_node::Program;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;
use triton_frontend::{codegen, opt};

// ⭐ THE CASE TABLE MOVED TO `triton_ktir_superdsc::cases`, so `triton-numeric`'s emulator check
// reads the SAME statement of what a configuration is. It used to live here, and a second
// statement of `rope_q32`'s grid in another consumer is exactly the defect recorded in that
// module. Nothing about the configurations changed in the move.
use triton_ktir_superdsc::cases;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("use: bake_py <config> <out-dir>");
        eprintln!("configs: {}", cases::ALL.join(" "));
        std::process::exit(2);
    }
    let (config, out_dir) = (&args[0], PathBuf::from(&args[1]));
    let Some((kspec, grid, program)) = cases::case(config) else {
        eprintln!("no case for `{config}`");
        std::process::exit(2);
    };

    macro_rules! refused {
        ($stage:expr, $err:expr) => {{
            println!("REFUSED  {config:24} {:<12} {}", $stage, $err);
            std::process::exit(1);
        }};
    }

    // LEG 1 — Triton .py -> raw ttir. The leg every earlier run in this crate skipped.
    let src = std::fs::read_to_string(&kspec.file)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", kspec.file));
    let mut tt = match codegen::compile(&src, &kspec, Target::spyre()) {
        Ok(m) => m,
        Err(e) => refused!("codegen", e),
    };
    if let Err(e) = opt::make_ttir(&mut tt) {
        refused!("make_ttir", e);
    }
    // TTIR_DUMP=1 prints the post-make_ttir ttir THIS PATH ACTUALLY BUILT — the same
    // instrument as KTIR_DUMP below, one leg earlier, for refusals that fire at
    // `from_ttir` and so can only be diagnosed from the artifact the refusal names.
    if std::env::var("TTIR_DUMP").as_deref() == Ok("1") {
        println!("{}", triton_frontend::ttir::print::print_module(&tt));
    }

    // LEG 2 — ttir value -> our KTIR value. No text in between.
    let mut m = match triton_ktir::from_ttir::convert(&tt) {
        Ok(m) => m,
        Err(e) => refused!("from_ttir", e),
    };
    if let Err(e) = triton_ktir::make_ktir(&mut m, &grid) {
        refused!("make_ktir-ktir", e);
    }
    if let Err(e) = triton_ktir::passes::to_ktir::run(&mut m, &grid) {
        refused!("to_ktir", e);
    }

    // LEG 3 — our KTIR -> their request type -> their lowering.
    let node = match triton_ktir_superdsc::node_for(&m, program) {
        Ok(n) => n,
        Err(e) => refused!(e.stage, e.message),
    };
    // KTIR_DUMP=1 prints the KTIR THIS PATH ACTUALLY BUILT, before any lowering.
    //
    // ⛔ THE INSTRUMENT THAT WAS MISSING. Every KTIR I inspected while chasing the matmul shape
    // came from `emit_ktir`, which reads a checked-in ttir GOLDEN -- so a change to the .py fixture
    // could not appear in it, and I read "the view is unchanged" off a dump that could never have
    // shown the change. Four fix attempts were reverted on the strength of that.
    if std::env::var("KTIR_DUMP").as_deref() == Ok("1") {
        println!("{}", triton_ktir_superdsc::dump(&node));
    }

    // THE MODEL-GEOMETRY CONST DOOR. `emit_node` refuses `Program::{Rope, Attn}` ON PURPOSE: their
    // bodies are monomorphised on the head geometry (`rope_at::<HD>`, `attn_at::<NQH, NKVH, HD>`),
    // which is a CONST parameter, so no runtime `Program` can reach them and defaulting the geometry
    // would make a wrong head count a silently wrong grouping. `drive_rope` takes it explicitly.
    //
    // The values come from the fixture's OWN `tl.constexpr`s -- the same numbers the kernel was
    // compiled against on the leg above, not a second statement of them that could drift.
    if program == Program::Rope {
        let ce = |k: &str| match kspec.constexprs.get(k) {
            Some(Val::Int(v)) => u32::try_from(*v)
                .map_err(|_| format!("`{k}` = {v} does not fit the geometry's u32")),
            _ => Err(format!("`{k}` is not an integer tl.constexpr of this configuration")),
        };
        let (hd, h, n_tok) = match (ce("HEAD_DIM"), ce("H"), ce("N_TOK")) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => refused!("constexpr", e),
        };
        // ⛔⛔⛔ THE ROW ORDER IS STATED, NOT INFERRED, AND IT CANNOT BE INFERRED. A roped plane is
        // `[mq·heads, hd]` under BOTH orders -- same shape, same element count, same footprint -- so
        // no shape check, no arrangement check and no footprint guard separates them, and the wrong
        // one places every (row, head) block at another block's address and reports success. See
        // `RopeRows`, which carries the five lines of `KtirFunc::rope` that settle it.
        //
        // TOKEN-major is what `rope_at` addresses. `KTIR_ROPE_ROWS=head-major` states the other one,
        // so "would the wrong order have been caught?" stays a question THIS binary answers rather
        // than a claim about a guard nobody fired.
        let rows = match std::env::var("KTIR_ROPE_ROWS").as_deref() {
            Ok("head-major") => triton_ktir_superdsc::RopeRows::HeadMajor,
            _ => triton_ktir_superdsc::RopeRows::TokenMajor,
        };
        let layout = match triton_ktir_superdsc::layout::for_node(&node) {
            Ok(l) => l,
            Err(e) => refused!(e.stage, e.message),
        };
        // `rows_are_requests` is false: these are prefill token rows, not one row per decode request.
        match triton_ktir_superdsc::drive_rope(&node, hd, n_tok, h * hd, rows, false, Some(&layout)) {
            Err(e) => refused!(e.stage, e.message),
            Ok(ops) => match triton_ktir_superdsc::bake::write_dir(&out_dir, &ops) {
                Err(e) => refused!("bake", e),
                Ok(w) => {
                    // ⛔ THE LAUNCH MAP TOO, or this bake cannot be run. The other two arms write it
                    // and this one did not, so a rope baked 35 descriptors and then had no
                    // `placements.json` for the launcher to bind against — the bake looked like a
                    // success and was unrunnable. Same call, same reason as the whole-function arm.
                    if let Err(e) = triton_ktir_superdsc::bake::write_placements(&out_dir, &layout) {
                        refused!("bake", e);
                    }
                    println!(
                        "BAKED    {config:24} {:<12} drive_rope: ops={} files={}",
                        "",
                        ops.len(),
                        w.len()
                    );
                    return;
                }
            },
        }
    }

    // ⛔ ATTENTION HAS NO MODEL-GEOMETRY ARM HERE, ON PURPOSE. There used to be a `KTIR_ATTN=1`
    // door that recognised `Program::Attn` and slotted in `attn_at::<NQH, NKVH, HD>`'s fragment
    // assembly (scratchy's prebuilt attention lowering) selected by constexpr-geometry match.
    // That is pattern matching, not lowering: it substitutes a hand-rolled implementation for the
    // given Triton code. Removed. Attention goes through the whole-function door below
    // (`KTIR_WHOLE=1`) like every other kernel — its ops, from the compiled fixture, one at a
    // time. (The old comment claiming the whole door "computes the wrong numbers on card" was
    // wrong — that defect was host staging, fixed by `attn_card_stage.py`'s layout law; the
    // per-trip math was verified by stage read-backs.)

    // KTIR_WHOLE=1 takes the whole-function door instead of the per-`Program` one: it walks the
    // function's ops rather than reading its parameter list as one op's operands. That is the door a
    // whole-kernel producer needs, so it is how a multi-matmul kernel gets driven.
    if std::env::var("KTIR_WHOLE").as_deref() == Ok("1") {
        let layout = match triton_ktir_superdsc::layout::for_node(&node) {
            Ok(l) => l,
            Err(e) => refused!(e.stage, e.message),
        };
        match triton_ktir_superdsc::emit_whole(&node, Some(&layout)) {
            Err(e) => refused!(e.stage, e.message),
            Ok(ops) => {
                // The launch map, from the SAME layout this emission used — see
                // `bake::write_placements` for why the directory cannot yield it on its own.
                let written = triton_ktir_superdsc::bake::write_placements(&out_dir, &layout)
                    .and_then(|()| triton_ktir_superdsc::bake::write_dir(&out_dir, &ops));
                match written {
                Err(e) => refused!("bake", e),
                Ok(w) => {
                    println!(
                        "BAKED    {config:24} {:<12} whole-function: ops={} files={}",
                        "",
                        ops.len(),
                        w.len()
                    );
                    return;
                }
                }
            }
        }
    }

    // ⭐ THE LAYOUT IS BUILT HERE, ONCE, AND USED TWICE — for the emission and for
    // `placements.json`. `emit_node_with` exists for exactly this: `emit_node` would derive its own,
    // and two derivations of a device memory plan is how the host and the descriptors come to
    // disagree about where a tensor lives.
    let layout = match triton_ktir_superdsc::layout::for_node(&node) {
        Ok(l) => l,
        Err(e) => refused!(e.stage, e.message),
    };
    let ops = match triton_ktir_superdsc::emit_node_with(&node, Some(&layout)) {
        Ok(o) => o,
        Err(e) => refused!(e.stage, e.message),
    };

    // LEG 4 — the descriptors as the files dxp_standalone reads, plus the one file a LAUNCH needs
    // and dxp does not produce: each tensor's (segment, offset, size).
    if let Err(e) = triton_ktir_superdsc::bake::write_placements(&out_dir, &layout) {
        refused!(e.stage, e.message);
    }
    match triton_ktir_superdsc::bake::write_dir(&out_dir, &ops) {
        Err(e) => refused!("bake", e),
        Ok(written) => {
            println!(
                "BAKED    {config:24} {:<12} from .py: ops={} files={}",
                "",
                ops.len(),
                written.len()
            );
            for f in &written {
                println!("           {f}");
            }
        }
    }
}
