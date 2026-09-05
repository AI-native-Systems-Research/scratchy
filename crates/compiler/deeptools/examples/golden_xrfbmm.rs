//! RECONSTRUCT IBM'S OWN int8 BMM THROUGH THE ISLAND AND DIFF IT AGAINST THEIRS.
//!
//! The reference is `~/tmp/dt_src/dcc/test/PT/xrfbmm_int8_fwd.mlir` — a complete int8 batched matmul,
//! and the smallest complete DataflowIR program IBM ships. If the island can hold it and the printer
//! can write it, the island is not missing anything a real program needs.
//!
//! # ⛔⛔ IT USED TO PRINT AND STOP, AND "STILL BYTE-MATCHES IBM'S FILE" WAS SAID OF IT FOR MONTHS
//!
//! Nothing here compared anything, so it exited 0 whatever it printed. When the comparison was
//! finally written the reconstruction was **36 ops against IBM's 59** — missing six of the ten loop
//! bounds, three of the six units, the L0 memory view, SIX of the nine `affine.for` levels, and the
//! entire second half of the `x2` unroll. Every one of those is a different program.
//!
//! A byte match was never the right claim either: the file is a **lit test**, and its first fifty
//! lines are `CHECK-SENT-IR` assertions about SentientIR — island TWO. Only what follows
//! `func.func @dataflowProgram` is DataflowIR. So the comparison is the OPERATION SEQUENCE of the
//! program body: which ops, in which order, ignoring SSA names, whitespace and the module framing
//! (see [`body`]).
//!
//! # ⛔⛔ AND THE OPERATION SEQUENCE IS BLIND TO THE ADDRESSING, WHICH IS WHERE A REAL BUG LIVED
//!
//! [`ops`] keeps only the mnemonic on each line, so `%w[%i * 64, %j, 0]` and `%w[%i, %j, 0]` are one
//! string to it. The emitter shipped a nest reading a SIXTY-FOURTH of its weight tile while this
//! example reported 56 against 56 and agreed. So [`addressing`] now compares every affine map and
//! every access subscript as well — canonicalised, because the two files spell one map two ways
//! (`(i, j) -> (128 * i + j)` against `(d0, d1) -> (d0 * 128 + d1)`), and α-renamed, so `%idx0`
//! against `%21` is forgiven while `%idx0` against `%idx2` is not.
//!
//! ⭐ MEASURED THAT IT SEES WHAT THE SEQUENCE CANNOT: changing the reconstruction's `%idx0 + 2` to
//! `+ 0` — deleting the unroll's row advance — leaves "56 program ops, the sequences agree" and is
//! caught by "addressing term 13 diverges: IBM's is `map(1:d0*1+2)`, ours is `map(1:d0*1+0)`".
//!
//! ⛔ AND IT PROVES SHAPE, NOT CORRECTNESS. It says the emitter can express what the backend expects,
//! not that a program built this way computes the right thing — that is the pod's verdict, and
//! nothing here substitutes for it.
//!
//! Runs under `cargo test -p deeptools` (the manifest marks this example `test = true`) and prints
//! the module under `cargo run -p deeptools --example golden_xrfbmm`.

use deeptools::arch::Dd2;
use deeptools::islands::dataflow_ir::op::{Bound, Index, LocalUnit, Op, Precision, Val};
use deeptools::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, MemRef, Vector};
use deeptools::islands::dataflow_ir::{Grid, KernelName, Program, ProgramName, Run, Values, print};
use deeptools::units::{DfirUnit, Row};

/// 🎯 THE RECONSTRUCTION MATCHES IBM'S PROGRAM, OP FOR OP — run by `cargo test`, not only by hand.
#[test]
fn the_island_holds_ibms_own_program() {
    main();
}

fn main() {
    let mut v = Values::default();

    // The extents, as `arith.constant`s the loops count to.
    //
    // ⛔ ALL TEN OF THEM. The five outermost are 1 and it is tempting to leave them out, but a loop
    // that runs ONCE is still a region, a barrier and an induction variable every enclosed transfer
    // is strided by — dropping them is a different program, not a simplification of the same one.
    let dx_cx = v.mint();
    let dy_cy = v.mint();
    let dmb_cmb = v.mint();
    let dout_cout = v.mint();
    let din_cin = v.mint();
    let cx = v.mint();
    let cout_sout = v.mint();
    let cin_sin = v.mint();
    let cy_mb = v.mint();
    let zero = v.mint();

    // The units. `C0-CL0-PT-0` is PT row 0 of core 0, corelet 0.
    let sfp = v.mint();
    let l0_lu = v.mint();
    let pt_0 = v.mint();
    let pt_1 = v.mint();
    let pt_2 = v.mint();
    let l0_unit = v.mint();
    let l0_memory = v.mint();

    // Inside the PT's program: its two register files, viewed as memrefs.
    let lrf_unit = v.mint();
    let lrf = v.mint();
    let xrf_unit = v.mint();
    let xrf = v.mint();

    let i7 = v.mint();
    let i1 = v.mint();
    let i2 = v.mint();
    let i3 = v.mint();
    let i4 = v.mint();
    let i5 = v.mint();
    let i6 = v.mint();
    let idx00 = v.mint();
    let i8_iv = v.mint();
    let idx0 = v.mint();
    let idx2 = v.mint();
    let i9 = v.mint();

    let int8 = ElemType::Int(8);
    let lrf_ty = MemRef {
        shape: vec![6, 128],
        elem: int8,
    };
    let xrf_ty = MemRef {
        shape: vec![64, 128],
        elem: int8,
    };
    let stick = Vector {
        len: 16,
        elem: int8,
    };
    let wide = Vector {
        len: 256,
        elem: int8,
    };
    let acc = Vector {
        len: 64,
        elem: int8,
    };

    // `#map4 = affine_map<(d0) -> (d0 mod 16)>` — the select's lane map.
    let selection_map = AffineMap::unary(AffineExpr::dim(0).modulo(16));
    // `#map5 = affine_map<(d0) -> ((d0 mod 128) floordiv 2)>` — the IMA8 reduction on RCUDD1A
    // (`SNComputeLowering.cpp:164-171`). The inner parentheses are load-bearing.
    let reduction_map = AffineMap::unary(AffineExpr::dim(0).modulo(128).floordiv(2));

    // One MAC pair of the inner loop: load the kernel stick, receive the input, widen it, multiply.
    let mac = |v: &mut Values, out_row: i64| -> Vec<Op> {
        let ker = v.mint();
        let inp = v.mint();
        let widened = v.mint();
        let out = v.mint();
        vec![
            Op::VectorLoad {
                result: ker,
                view: xrf,
                indices: vec![Index::Val(idx0), Index::Const(0)],
                view_ty: xrf_ty.clone(),
                ty: wide,
            },
            Op::Receive {
                result: inp,
                from: l0_lu,
                ty: stick,
            },
            Op::Select {
                result: widened,
                input: inp,
                selection_map: selection_map.clone(),
                input_ty: stick,
                ty: wide,
            },
            Op::Multiply {
                result: out,
                a: widened,
                b: ker,
                reduction_map: reduction_map.clone(),
                operand_ty: wide,
                ty: acc,
            },
            Op::VectorStore {
                value: out,
                view: lrf,
                indices: vec![Index::Const(out_row), Index::Const(0)],
                view_ty: lrf_ty.clone(),
                ty: acc,
            },
        ]
    };

    let mut inner = mac(&mut v, 4);
    inner.extend(mac(&mut v, 5));

    // ⛔⛔ `%idx2 = %idx0 + 2` — THE SECOND HALF OF THE `x2` UNROLL READS A DIFFERENT KERNEL ROW, and
    // this apply is what says so. An accumulating half that re-read `%idx0` would multiply the same
    // weights twice and accumulate the wrong product into the right buffer.
    inner.push(Op::Apply {
        result: idx2,
        map: AffineMap::unary(AffineExpr::dim(0).plus(AffineExpr::Const(2))),
        args: vec![idx0],
    });

    // The accumulating half: the same, reading the partial sum back and sending it south.
    //
    // ⛔⛔ IT RUNS TWICE, AND THE RECONSTRUCTION USED TO RUN IT ONCE. `unrollFactor =
    // #sentient<unroll_factor x2>` on both `sentient.vector_mac`s of the CHECK block is the same x2
    // the two multiplying halves above already show; the accumulating side unrolls with them. One
    // copy is half a MAC pair, which is exactly the class of gap `Op::UNROLL` exists to close — and
    // nothing caught it, because this example printed and never compared.
    let accumulate = |v: &mut Values, psum_row: i64| -> Vec<Op> {
        let ker = v.mint();
        let inp = v.mint();
        let widened = v.mint();
        let psum = v.mint();
        let out = v.mint();
        vec![
            Op::VectorLoad {
                result: ker,
                view: xrf,
                indices: vec![Index::Val(idx2), Index::Const(0)],
                view_ty: xrf_ty.clone(),
                ty: wide,
            },
            Op::Receive {
                result: inp,
                from: l0_lu,
                ty: stick,
            },
            Op::Select {
                result: widened,
                input: inp,
                selection_map: selection_map.clone(),
                input_ty: stick,
                ty: wide,
            },
            Op::VectorLoad {
                result: psum,
                view: lrf,
                indices: vec![Index::Const(psum_row), Index::Const(0)],
                view_ty: lrf_ty.clone(),
                ty: acc,
            },
            Op::MultiplyAccumulate {
                result: out,
                a: widened,
                b: ker,
                acc: psum,
                reduction_map: reduction_map.clone(),
                operand_ty: wide,
                ty: acc,
            },
            Op::Send {
                to: pt_1,
                data: out,
                ty: acc,
            },
        ]
    };
    inner.extend(accumulate(&mut v, 4));
    inner.extend(accumulate(&mut v, 5));

    // ⭐ THE BOUNDS IN IBM'S OWN ORDER AND WITH ITS OWN VALUES (`xrfbmm_int8_fwd.mlir:40-49`).
    let body = vec![
        Op::Constant {
            result: dx_cx,
            value: 1,
        },
        Op::Constant {
            result: dy_cy,
            value: 1,
        },
        Op::Constant {
            result: dmb_cmb,
            value: 1,
        },
        Op::Constant {
            result: dout_cout,
            value: 1,
        },
        Op::Constant {
            result: din_cin,
            value: 1,
        },
        Op::Constant {
            result: cx,
            value: 1,
        },
        Op::Constant {
            result: cout_sout,
            value: 2,
        },
        Op::Constant {
            result: cin_sin,
            value: 2,
        },
        Op::Constant {
            result: cy_mb,
            value: 2,
        },
        Op::Constant {
            result: zero,
            value: 0,
        },
        // ⛔ THE SFP AND PT-2 ARE NAMED THOUGH THIS PROGRAM NEVER SENDS TO THEM. A `get_unit` is a
        // claim on the unit; the set a program names is part of what the backend schedules against,
        // so dropping the ones it does not send to is a different program.
        Op::GetUnit {
            result: sfp,
            core: 0,
            corelet: Some(0),
            unit: DfirUnit::Sfp,
        },
        Op::GetUnit {
            result: l0_lu,
            core: 0,
            corelet: Some(0),
            unit: DfirUnit::L0lu,
        },
        Op::GetUnit {
            result: pt_0,
            core: 0,
            corelet: Some(0),
            unit: DfirUnit::PtRow(Row::checked(0).expect("every arch has row 0")),
        },
        Op::GetUnit {
            result: pt_1,
            core: 0,
            corelet: Some(0),
            unit: DfirUnit::PtRow(Row::checked(1).expect("every arch has row 1")),
        },
        Op::GetUnit {
            result: pt_2,
            core: 0,
            corelet: Some(0),
            unit: DfirUnit::PtRow(Row::checked(2).expect("DD2 has eight PT rows")),
        },
        Op::GetUnit {
            result: l0_unit,
            core: 0,
            corelet: Some(0),
            unit: DfirUnit::L0,
        },
        // L0 as a flat 1024-byte view — the module-level memory the LU streams out of.
        Op::GetLogicalMemoryView {
            result: l0_memory,
            from: l0_unit,
            start: zero,
            layout: AffineMap::linear(&[1]),
            ty: MemRef {
                shape: vec![1024],
                elem: int8,
            },
        },
        Op::ProgramUnit {
            units: vec![pt_0],
            precision: Some(Precision::Int8),
            body: vec![
                Op::GetLocalUnit {
                    result: lrf_unit,
                    of: pt_0,
                    which: LocalUnit::PtLrf,
                },
                Op::GetLogicalMemoryView {
                    result: lrf,
                    from: lrf_unit,
                    start: zero,
                    layout: AffineMap::linear(&[128, 1]),
                    ty: lrf_ty.clone(),
                },
                Op::GetLocalUnit {
                    result: xrf_unit,
                    of: pt_0,
                    which: LocalUnit::PtXrf,
                },
                Op::GetLogicalMemoryView {
                    result: xrf,
                    from: xrf_unit,
                    start: zero,
                    layout: AffineMap::linear(&[128, 1]),
                    ty: xrf_ty.clone(),
                },
                // ⛔⛔ SIX OUTER LEVELS, EVERY ONE OF THEM RUNNING ONCE. `Dx_Cx`, `Dy_Cy`,
                // `Dmb_Cmb`, `Dout_Cout`, `Din_Cin` and `Cx` are all 1 in this program, which is
                // exactly why they were missing: a nest that runs once looks like no nest at all in
                // the output. It is not — each is a region and a barrier, and `FITS_LX` REMOVING a
                // level rather than setting its bound to 1 is the same distinction one layer up.
                nest(
                    &[
                        (i1, dx_cx),
                        (i2, dy_cy),
                        (i3, dmb_cmb),
                        (i4, dout_cout),
                        (i5, din_cin),
                        (i6, cx),
                    ],
                    Op::For {
                        iv: i7,
                        lo: Bound::Const(0),
                        hi: Bound::Val(cout_sout),
                        body: vec![
                            Op::Apply {
                                result: idx00,
                                map: AffineMap::unary(AffineExpr::dim(0).times(8)),
                                args: vec![i7],
                            },
                            Op::For {
                                iv: i8_iv,
                                lo: Bound::Const(0),
                                hi: Bound::Val(cin_sin),
                                body: vec![
                                    Op::Apply {
                                        result: idx0,
                                        map: AffineMap {
                                            dims: 2,
                                            results: vec![
                                                AffineExpr::dim(0)
                                                    .times(4)
                                                    .plus(AffineExpr::dim(1)),
                                            ],
                                        },
                                        args: vec![i8_iv, idx00],
                                    },
                                    Op::For {
                                        iv: i9,
                                        lo: Bound::Const(0),
                                        hi: Bound::Val(cy_mb),
                                        body: inner,
                                    },
                                ],
                            },
                        ],
                    },
                ),
            ],
        },
    ];

    // ⭐ THE ARCH IS PART OF THE GOLDEN. `dcc/test/PT/xrfbmm_int8_fwd.mlir` names `ptrow0..ptrow7`,
    // which is DD2's eight-row grid — a `Sen1p5` annotation here would not compile, and that is the
    // point of `Program` carrying it.
    let run: Run<Dd2> = Run {
        kernel: KernelName::Golden("xrfbmm_int8_fwd"),
        programs: vec![Program {
            name: ProgramName::Golden("dataflowProgram"),
            grid: Grid::single(),
            arch: core::marker::PhantomData,
            body,
        }],
    };

    let ours = print::run(&run);
    print!("{ours}");
    let _ = (Val(0), v.issued());
    compare(&ours);
}

/// 🛑 DIFF THE OPERATION SEQUENCE AGAINST IBM'S OWN FILE, AND FAIL ON A DIVERGENCE.
///
/// ⛔⛔ THIS EXAMPLE USED TO PRINT AND STOP, and "still byte-matches IBM's file" was said of it for a
/// long time. It never compared anything, so it exited 0 whatever it printed, and the claim had
/// nothing behind it. Measured when the comparison was finally written: 36 ops against IBM's 59.
///
/// ⛔ AND A BYTE MATCH WAS NEVER THE RIGHT CLAIM ANYWAY. `xrfbmm_int8_fwd.mlir` is a **lit test**: its
/// first fifty lines are `CHECK-SENT-IR` assertions about SentientIR — island TWO — and the
/// DataflowIR input only begins at `func.func @dataflowProgram`. Comparing whole files compares a
/// FileCheck script against a module.
///
/// ⭐ SO THE COMPARISON IS THE OPERATION SEQUENCE: which ops, in which order, ignoring SSA names and
/// whitespace. That is what "the island can hold this program" means — a missing `affine.for` level
/// or a dropped unroll half is a different program, and neither shows up in a line count.
fn compare(ours: &str) {
    let Some(home) = std::env::var_os("HOME") else {
        eprintln!("\n[golden] no HOME, so IBM's reference cannot be found — nothing compared");
        return;
    };
    let reference =
        std::path::PathBuf::from(home).join("tmp/dt_src/dcc/test/PT/xrfbmm_int8_fwd.mlir");
    let Ok(theirs) = std::fs::read_to_string(&reference) else {
        eprintln!(
            "\n[golden] {} is absent — nothing compared",
            reference.display()
        );
        return;
    };
    // The DataflowIR input starts where the FileCheck block ends.
    //
    // ⛔⛔ THE **LAST** OCCURRENCE, NOT THE FIRST. `CHECK-SENT-IR-LABEL:` names
    // `func.func @dataflowProgram` in a COMMENT at the top of the file, so a `find` starts the slice
    // 27 lines early and sweeps in the `#map4`/`#map5` declarations. The op comparison never noticed
    // because it strips comments and a `#map` line names no dialect op — but the addressing
    // comparison reads those lines, and they are island TWO's assertions, not this program.
    let Some(at) = theirs.rfind("func.func @dataflowProgram") else {
        panic!(
            "{} no longer contains a `func.func @dataflowProgram`",
            reference.display()
        );
    };

    // ⛔ IBM'S MAPS ARE DECLARED ABOVE THE PROGRAM AND USED BY ALIAS. `#map5` is bound at the top of
    // the file and referenced as `reduction_map = #map5` inside; ours are inlined. Substituting the
    // definition in is what makes the two comparable — slicing them off instead would compare our
    // maps against nothing.
    let their_text = resolve_map_aliases(&theirs, &theirs[at..]);

    let their_ops = body(&ops(&their_text, true));
    let our_ops = body(&ops(ours, false));
    eprintln!(
        "\n[golden] IBM {} program ops, ours {}",
        their_ops.len(),
        our_ops.len()
    );
    if let Some((at, (want, got))) = their_ops
        .iter()
        .zip(our_ops.iter())
        .enumerate()
        .find(|(_, (want, got))| want != got)
    {
        panic!(
            "op {at} diverges: IBM's is `{want}`, ours is `{got}`\n  IBM : {}\n  ours: {}",
            their_ops.join(" "),
            our_ops.join(" ")
        );
    }
    assert_eq!(
        their_ops.len(),
        our_ops.len(),
        "the sequences agree as far as the shorter runs, and then one stops:\n  IBM : {}\n  ours: {}",
        their_ops.join(" "),
        our_ops.join(" ")
    );
    eprintln!("[golden] the operation sequences agree");

    // 🛑🛑 AND NOW THE ADDRESSING, which the sequence above cannot see.
    let their_addressing = addressing(&their_text);
    let our_addressing = addressing(ours);
    eprintln!(
        "[golden] IBM {} addressing terms, ours {}",
        their_addressing.len(),
        our_addressing.len()
    );
    if let Some((at, (want, got))) = their_addressing
        .iter()
        .zip(our_addressing.iter())
        .enumerate()
        .find(|(_, (want, got))| want != got)
    {
        panic!(
            "addressing term {at} diverges: IBM's is `{want}`, ours is `{got}`\n  IBM : {}\n  ours: {}",
            their_addressing.join(" "),
            our_addressing.join(" ")
        );
    }
    assert_eq!(
        their_addressing.len(),
        our_addressing.len(),
        "the addressing agrees as far as the shorter runs, and then one stops:\n  IBM : {}\n  ours: {}",
        their_addressing.join(" "),
        our_addressing.join(" ")
    );
    eprintln!("[golden] the addressing agrees");
}

/// THE DIMENSION NAMES OF A MAP, in order — `(i, j)` gives `["i", "j"]`.
fn dim_names(dims: &str) -> Vec<String> {
    let Some(open) = dims.find('(') else {
        return Vec::new();
    };
    let after = &dims[open + 1..];
    let Some(close) = after.find(')') else {
        return Vec::new();
    };
    after[..close]
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Rewrite each dimension name to `dN` BY POSITION, matching whole identifiers only.
///
/// ⛔ WHOLE IDENTIFIERS. A textual replace of `i` rewrites the `i` inside `floordiv`, and the map
/// that comes out is not a map.
fn rename_dims(expr: &str, names: &[String]) -> String {
    let mut out = String::with_capacity(expr.len());
    let mut token = String::new();
    let flush = |token: &mut String, out: &mut String| {
        if !token.is_empty() {
            match names.iter().position(|name| name == token.as_str()) {
                Some(at) => out.push_str(&format!("d{at}")),
                None => out.push_str(token),
            }
            token.clear();
        }
    };
    for c in expr.chars() {
        if c.is_alphanumeric() || c == '_' {
            token.push(c);
        } else {
            flush(&mut token, &mut out);
            out.push(c);
        }
    }
    flush(&mut token, &mut out);
    out
}

/// Strip ONE matched outer parenthesis pair, and only if it really is matched.
///
/// ⛔ `trim_start_matches('(')` IS NOT THIS. `(d0 mod 128) floordiv 2` opens with a paren whose
/// partner is in the middle, so trimming produced `d0 mod 128) floordiv 2` — an expression with an
/// unbalanced bracket that then compared unequal to the identical map spelled by the other file.
fn strip_outer_parens(expr: &str) -> &str {
    let expr = expr.trim();
    if !expr.starts_with('(') {
        return expr;
    }
    let mut depth = 0usize;
    for (at, c) in expr.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return if at == expr.len() - 1 {
                        expr[1..at].trim()
                    } else {
                        expr
                    };
                }
            }
            _ => {}
        }
    }
    expr
}

/// SUBSTITUTE `#mapN = affine_map<..>` DEFINITIONS INTO A BODY THAT REFERS TO THEM BY ALIAS.
fn resolve_map_aliases(whole: &str, body: &str) -> String {
    let mut resolved = body.to_owned();
    for line in whole.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix('#') else {
            continue;
        };
        let Some((name, definition)) = rest.split_once('=') else {
            continue;
        };
        let (name, definition) = (name.trim(), definition.trim());
        if definition.starts_with("affine_map<") {
            resolved = resolved.replace(&format!("#{name}"), definition);
        }
    }
    resolved
}

/// THE PROGRAM'S OWN OPS — everything between its first constant and its last non-`func` op.
///
/// ⛔⛔ THE FRAMING DIFFERS BY DESIGN AND IS THE ONE THING NOT COMPARED. IBM's file is a single-program
/// lit fixture: `func.func @dataflowProgram` at top level with a bare `func.call` after it. Ours is
/// what `dbo-opt --from-dfir` takes for a launch group — an outer module, a DECLARATION module whose
/// entry calls each program in trip order, and one module per program. Both name the program once
/// and call it once; only the shape of the wrapper differs, and that shape is the backend's
/// requirement, not a divergence.
///
/// ⛔ SO THE EXCLUSION IS EXACTLY TWO `func.*` RUNS, one at each end, and NOTHING inside. A window
/// that skipped anything else would be a comparison tuned until it passed.
fn body(ops: &[String]) -> Vec<String> {
    let from = ops
        .iter()
        .position(|op| op == "arith.constant")
        .unwrap_or(0);
    let to = ops
        .iter()
        .rposition(|op| !op.starts_with("func."))
        .map_or(ops.len(), |at| at + 1);
    ops[from..to.max(from)].to_vec()
}

/// WRAP `innermost` IN ONE `affine.for` PER `(induction variable, bound)`, outermost first.
///
/// ⛔ WRITTEN AS A FOLD RATHER THAN AS SIX NESTED LITERALS, because six hand-written levels is six
/// chances to pair an induction variable with the wrong bound, and that is a program that runs.
fn nest(levels: &[(Val, Val)], innermost: Op) -> Op {
    levels
        .iter()
        .rev()
        .fold(innermost, |body, (iv, hi)| Op::For {
            iv: *iv,
            lo: Bound::Const(0),
            hi: Bound::Val(*hi),
            body: vec![body],
        })
}

/// THE PROGRAM'S ADDRESSING, IN ORDER — every affine map and every access subscript.
///
/// ⛔⛔ THE OPERATION SEQUENCE CANNOT SEE AN ADDRESSING BUG, AND ONE LIVED BEHIND IT. [`ops`] keeps
/// only the mnemonic on each line, so a `vector_load %w[%i * 64, %j, 0]` and a
/// `vector_load %w[%i, %j, 0]` are the same string to it. The emitter shipped a nest reading a
/// SIXTY-FOURTH of its weight tile while this example reported 56 ops against 56 and agreed.
///
/// ⭐ CANONICALISED, BECAUSE THE TWO FILES SPELL ONE MAP TWO WAYS. IBM writes `affine_map<(d0)[]->
/// (8*d0)>`; our printer writes `affine_map<(d0) -> (d0 * 8)>`. Both are the same map, so a term
/// that is a product of a dim and a literal is rendered `dN*c` with the terms sorted — anything
/// richer (a `mod`, a `floordiv`) keeps its text with the whitespace squeezed out, since those the
/// two files do spell alike.
///
/// ⭐ AND SSA NAMES ARE α-RENAMED, NOT ERASED. `%idx0` against `%21` is not a divergence; `%idx0`
/// against `%idx2` IS. Numbering each name in order of first appearance IN THE ADDRESSING STREAM
/// keeps the second while forgiving the first — erasing them instead is what let the subscripts go
/// uncompared in the first place.
fn addressing(text: &str) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let mut out = Vec::new();
    let mut alpha = |token: &str| -> String {
        match names.iter().position(|seen| seen == token) {
            Some(at) => format!("%v{at}"),
            None => {
                names.push(token.to_owned());
                format!("%v{}", names.len() - 1)
            }
        }
    };

    for line in text.lines() {
        let line = line.split("//").next().unwrap_or("");
        // Every `affine_map<..>`, balanced to its own closing angle bracket.
        let mut rest = line;
        while let Some(at) = rest.find("affine_map<") {
            let body = &rest[at + "affine_map<".len()..];
            // ⛔ THE `>` OF `->` IS NOT A CLOSING BRACKET. Counting it ended every capture at
            // `(d0) -`, so every map canonicalised to the same string and the comparison was
            // blind in exactly the way it exists to prevent.
            let mut depth = 1usize;
            let mut end = body.len();
            let mut previous = ' ';
            for (i, c) in body.char_indices() {
                match c {
                    '<' => depth += 1,
                    '>' if previous != '-' => {
                        depth -= 1;
                        if depth == 0 {
                            end = i;
                            break;
                        }
                    }
                    _ => {}
                }
                previous = c;
            }
            out.push(format!("map({})", canonical_map(&body[..end])));
            rest = &body[end..];
        }
        // Every access subscript: a `%name[..]` on a load or a store.
        //
        // ⛔ ONLY WHERE A `%value` IMMEDIATELY PRECEDES THE BRACKET. Taking every `[..]` swept in
        // attribute lists and dense literals, which put terms in the stream that address nothing.
        let mut rest = line;
        while let Some(open) = rest.find('[') {
            let after = &rest[open + 1..];
            let Some(close) = after.find(']') else { break };
            let inside = &after[..close];
            let accessed = rest[..open]
                .rsplit(|c: char| c.is_whitespace() || c == ',')
                .next()
                .is_some_and(|token| token.starts_with('%'));
            // `[]` is IBM's empty symbol list, and a `memref<..>` type carries none of these.
            if accessed && !inside.trim().is_empty() {
                let subscript: Vec<String> = inside
                    .split(',')
                    .map(|term| {
                        let term = term.trim();
                        match term.strip_prefix('%') {
                            Some(_) => alpha(term),
                            None => term.replace(' ', ""),
                        }
                    })
                    .collect();
                out.push(format!("at[{}]", subscript.join(",")));
            }
            rest = &after[close..];
        }
    }
    out
}

/// ONE AFFINE MAP AS A SORTED SUM OF `dN*c` TERMS, or its squeezed text where it is richer.
///
/// ⛔ THE DIMS ARE RENAMED BY POSITION, BECAUSE THE TWO FILES NAME THEM DIFFERENTLY. IBM writes
/// `affine_map<(i, j) -> (128 * i + j)>` where our printer writes `(d0, d1) -> (d0 * 128 + d1)`.
/// Those are one map; comparing the spellings makes every layout map a false divergence.
fn canonical_map(body: &str) -> String {
    let Some((dims, result)) = body.split_once("->") else {
        return body.replace([' ', '\t'], "");
    };
    let names = dim_names(dims);
    let arity = names.len();
    let result = rename_dims(strip_outer_parens(result.trim()), &names);
    if result.contains("mod") || result.contains("floordiv") || result.contains("ceildiv") {
        return format!("{arity}:{}", result.replace([' ', '\t'], ""));
    }
    let result = result.as_str();
    let mut terms: Vec<(u64, i64)> = Vec::new();
    for term in result.split('+') {
        let mut dim = None;
        let mut coefficient: i64 = 1;
        for factor in term.split('*') {
            let factor = factor.trim();
            match factor.strip_prefix('d').and_then(|n| n.parse::<u64>().ok()) {
                Some(at) => dim = Some(at),
                None => match factor.parse::<i64>() {
                    Ok(literal) => coefficient *= literal,
                    // Not a plain sum of products — fall back rather than mis-canonicalise.
                    Err(_) => return format!("{arity}:{}", result.replace([' ', '\t'], "")),
                },
            }
        }
        terms.push((dim.unwrap_or(u64::MAX), coefficient));
    }
    terms.sort_unstable();
    let rendered: Vec<String> = terms
        .iter()
        .map(|(dim, coefficient)| match *dim {
            u64::MAX => format!("{coefficient}"),
            at => format!("d{at}*{coefficient}"),
        })
        .collect();
    format!("{arity}:{}", rendered.join("+"))
}

/// The dialect operations one module names, in order — `dataflow.send`, `affine.for`, and so on.
fn ops(text: &str, strip_comments: bool) -> Vec<String> {
    const DIALECTS: [&str; 6] = ["dataflow", "vectorchain", "affine", "arith", "func", "agen"];
    let mut out = Vec::new();
    for line in text.lines() {
        let line = match strip_comments {
            true => line.split("//").next().unwrap_or(""),
            false => line,
        };
        for word in line.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.')) {
            if let Some((dialect, op)) = word.split_once('.')
                && DIALECTS.contains(&dialect)
                && !op.is_empty()
            {
                out.push(word.to_owned());
                break;
            }
        }
    }
    out
}
