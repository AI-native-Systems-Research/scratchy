//! THE ISLAND AS MLIR TEXT — the one place DataflowIR becomes characters.
//!
//! ⭐ DETERMINISTIC BY CONSTRUCTION. SSA names come from the builder's counter and attributes are
//! written in a fixed order, so two emissions of one program are byte-identical. That is what makes
//! the memoization in the bake queue sound: a group whose input recurs must hash the same.
//!
//! ⭐⭐ THE MODULE STRUCTURE AND THE SHARED SPELLINGS LIVE HERE; ONE OP'S SYNTAX LIVES WITH THE OP.
//! [`emit`] reads the dialect off the op and hands it to that dialect's own printer, so adding an
//! operation is one arm in one file rather than a choice of where in a single match to put it.

use std::fmt::Write as _;

use crate::islands::dataflow_ir::dialects::dataflow::Precision;
use crate::islands::dataflow_ir::dialects::{
    Index, Op, Val, affine, agen, arith, dataflow, scf, symbol, uniform, vector, vectorchain,
};
use crate::islands::dataflow_ir::ty::{
    AffineExpr, AffineMap, ElemType, IntegerSet, MemRef, Vector,
};
use crate::islands::dataflow_ir::{Grid, Program, Run};

/// A WHOLE RUN AS ONE MLIR MODULE — the shape `dbo-adapt-scheduler-dfir` consumes.
///
/// The top module holds an UNNAMED declaration module, whose function calls each program in run
/// order and forward-declares them, then one NAMED module per program
/// (`dbo/test/adapt-scheduler-dfir-multi.mlir`).
#[must_use]
pub fn run<A: crate::arch::Arch>(run: &Run<A>) -> String {
    let mut out = String::new();
    out.push_str("module {\n");

    // The declaration module: unnamed, as the scheduler's is. It records which kernel each schedule
    // belongs to, and nothing downstream can reconstruct that.
    out.push_str("  module {\n");
    let grid = run
        .programs
        .first()
        .map(|program| grid_attr(program.grid.extents()))
        .unwrap_or_else(|| grid_attr(Grid::single().extents()));
    let _ = writeln!(
        out,
        "    func.func @{}() attributes {{grid = {grid}}} {{",
        run.kernel
    );
    for program in &run.programs {
        let _ = writeln!(out, "      call @{}() : () -> ()", program.name);
    }
    out.push_str("      return\n    }\n");
    for program in &run.programs {
        let _ = writeln!(out, "    func.func private @{}()", program.name);
    }
    out.push_str("  }\n");

    for program in &run.programs {
        program_module(&mut out, program);
    }

    out.push_str("}\n");
    out
}

/// One named module: the program's function, holding its DataflowIR.
///
/// ⛔ THE FUNCTION IS `private`, AS THE SCHEDULER EMITS IT. `dbo-adapt-scheduler-dfir` is what makes
/// it public, because the DCC pipeline runs symbol-dce and the callers are in a symbol table of
/// their own (`dbo/test/adapt-scheduler-dfir-multi.mlir:29-32`). Emitting it public would be doing
/// that pass's job for it, and differently.
fn program_module<A: crate::arch::Arch>(out: &mut String, program: &Program<A>) {
    let _ = writeln!(out, "  module @{} {{", program.name);
    let _ = writeln!(
        out,
        "    func.func private @{}() attributes {{grid = {}}} {{",
        program.name,
        grid_attr(program.grid.extents())
    );
    // The preamble binds the units and views; the program units then run on them.
    for op in &program.preamble {
        emit(out, op, 3);
    }
    // ⭐⭐ AT LEAST ONE `dataflow.program_unit`, BY THE TYPE. `AdaptSchedulerDfir.cpp:63-78` walks
    // each child module for a `func.func` holding one and fails the compile with "found no program
    // to compile" when none does — which is exactly what a flat body produced.
    for unit in program.units.iter() {
        let precision = match unit.precision {
            Some(p) => format!(" {{precision = \"{}\"}}", Precision::spelling(p)),
            None => String::new(),
        };
        indent(out, 3);
        let _ = writeln!(
            out,
            "dataflow.program_unit {}{precision} : {{",
            vals(&unit.on.vals())
        );
        for op in &unit.body {
            emit(out, op, 4);
        }
        indent(out, 3);
        out.push_str("}\n");
    }
    out.push_str("      return\n    }\n  }\n");
}

fn grid_attr(grid: &[u32]) -> String {
    format!(
        "[{}]",
        grid.iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub(crate) fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

pub(crate) fn val(v: Val) -> String {
    format!("%{}", v.0)
}

pub(crate) fn vals(items: &[Val]) -> String {
    items
        .iter()
        .copied()
        .map(val)
        .collect::<Vec<_>>()
        .join(", ")
}

/// ONE OP, INDENTED, BY THE DIALECT THAT DECLARES IT.
pub(crate) fn emit(out: &mut String, op: &Op, depth: usize) {
    indent(out, depth);
    match op {
        Op::Arith(op) => arith::emit(out, op),
        Op::Scf(op) => scf::emit(out, op, depth),
        Op::Affine(op) => affine::emit(out, op, depth),
        Op::Vector(op) => vector::emit(out, op),
        Op::Dataflow(op) => dataflow::emit(out, op, depth),
        Op::Agen(op) => agen::emit(out, op, depth),
        Op::VectorChain(op) => vectorchain::emit(out, op),
        Op::Symbol(op) => symbol::emit(out, op),
        Op::Uniform(op) => uniform::emit(out, op, depth),
    }
}

pub(crate) fn index_list(indices: &[Index]) -> String {
    indices
        .iter()
        .map(|index| match index {
            Index::Val(v) => val(*v),
            Index::Const(n) => n.to_string(),
            // `%arg9 + %arg8 * 8` — the terms in the order the walk collected them, outermost loop
            // first, with a unit stride written bare.
            Index::Strided(terms, offset) => {
                let mut parts: Vec<String> = terms
                    .iter()
                    .map(|(v, stride)| match stride {
                        1 => val(*v),
                        n => format!("{} * {n}", val(*v)),
                    })
                    .collect();
                // ⭐ THE ADDEND LAST AND ONLY WHEN IT MOVES SOMETHING — `+ 0` is the same address
                // written longer, and the vendored files never write it.
                if *offset != 0 {
                    parts.push(offset.to_string());
                }
                parts.join(" + ")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn memref(ty: &MemRef) -> String {
    let shape = ty
        .shape
        .iter()
        .map(|extent| format!("{extent}x"))
        .collect::<String>();
    format!("memref<{shape}{}>", elem(ty.elem))
}

pub(crate) fn vector(ty: Vector) -> String {
    format!("vector<{}x{}>", ty.len, elem(ty.elem))
}

fn elem(ty: ElemType) -> String {
    match ty {
        ElemType::Int(bits) => format!("i{bits}"),
        ElemType::F16 => "f16".to_owned(),
        ElemType::F32 => "f32".to_owned(),
        ElemType::Bf16 => "bf16".to_owned(),
        ElemType::F8E4M3Fn => "f8E4M3FN".to_owned(),
        ElemType::F8E8M0Fnu => "f8E8M0FNU".to_owned(),
        ElemType::F8E5M2 => "f8E5M2".to_owned(),
        ElemType::F4E2M1Fn => "f4E2M1FN".to_owned(),
        ElemType::MxFloat(bits) => format!("!dataflow.mxfloat<{bits}>"),
    }
}

pub(crate) fn affine_map(map: &AffineMap) -> String {
    let dims = (0..map.dims)
        .map(|d| format!("d{d}"))
        .collect::<Vec<_>>()
        .join(", ");
    // ⛔ THE SYMBOL GROUP IS OMITTED WHEN THERE ARE NONE, NOT PRINTED EMPTY. MLIR writes
    // `affine_map<(d0) -> (d0)>` and `affine_map<()[s0] -> (s0)>`; `affine_map<(d0)[] -> (d0)>`
    // does not round-trip. So a map with `syms: 0` prints exactly what it printed before symbols
    // existed, which is every map this bridge emits into a program.
    let syms = if map.syms == 0 {
        String::new()
    } else {
        format!(
            "[{}]",
            (0..map.syms)
                .map(|s| format!("s{s}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let results = map
        .results
        .iter()
        .map(|expr| affine_expr(expr, 0, false))
        .collect::<Vec<_>>()
        .join(", ");
    format!("affine_map<({dims}){syms} -> ({results})>")
}

/// `affine_set<(d0, ..)[s0, ..] : (c, ..)>` — the constraints in the order they were built.
///
/// ⭐ THE SYMBOL LIST IS WRITTEN ONLY WHEN THERE IS ONE, which is how MLIR writes it: the static
/// masks print `affine_set<(d0) : (d0 - 48 >= 0, -d0 + 63 >= 0)>` and the dynamic one
/// `affine_set<(d0)[s0] : (d0 + s0 * 8 - 64 >= 0, -d0 + 63 >= 0)>`
/// (`dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:5-7`). An always-printed
/// `[]` would make every static set a diff against a vendored file.
pub(crate) fn integer_set(set: &IntegerSet) -> String {
    let dims = (0..set.dims)
        .map(|d| format!("d{d}"))
        .collect::<Vec<_>>()
        .join(", ");
    let symbols = if set.symbols == 0 {
        String::new()
    } else {
        format!(
            "[{}]",
            (0..set.symbols)
                .map(|s| format!("s{s}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let constraints = set
        .constraints
        .iter()
        .map(|c| {
            let relation = if c.is_equality { "== 0" } else { ">= 0" };
            format!("{} {relation}", affine_expr(&c.expr, 0, false))
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("affine_set<({dims}){symbols} : ({constraints})>")
}

/// BINDING POWER, so the printed map is the one MLIR would print.
///
/// `+` is loosest; `*`, `mod` and `floordiv` bind tighter; a dimension or a literal is atomic.
const fn precedence(expr: &AffineExpr) -> u8 {
    match expr {
        AffineExpr::Dim(_) | AffineExpr::Sym(_) | AffineExpr::Const(_) => 3,
        AffineExpr::Mul(..) | AffineExpr::Mod(..) | AffineExpr::FloorDiv(..) => 2,
        AffineExpr::Add(..) => 1,
    }
}

/// AN EXPRESSION, PARENTHESISED EXACTLY WHERE IT HAS TO BE.
///
/// Two rules, and both are taken from what the vendored maps look like:
///
/// * a child that binds LOOSER than its parent needs parentheses — `(d0 + 1) * 4`;
/// * a `mod` or `floordiv` parenthesises any non-atomic operand, which is how
///   `(d0 mod 128) floordiv 2` — the int8 reduction map
///   (`dcc/test/PT/xrfbmm_int8_fwd.mlir:41`) — is written.
///
/// ⛔ AND `d0 * 128 + d1` IS LEFT ALONE. The first version parenthesised every non-atomic operand,
/// which produced `((d0 * 128) + d1)` for the layout map the reference writes as `128 * i + j`. Not
/// wrong — MLIR reads both the same — but it makes every diff against a vendored file noise, which
/// is the only way this printer can be checked without a pod.
fn affine_expr(expr: &AffineExpr, parent: u8, parent_is_divlike: bool) -> String {
    let own = precedence(expr);
    let text = match expr {
        AffineExpr::Dim(d) => format!("d{d}"),
        AffineExpr::Sym(sym) => format!("s{sym}"),
        AffineExpr::Const(n) => n.to_string(),
        // ⛔ A NEGATIVE ADDEND IS A SUBTRACTION, NOT A SUM WITH A NEGATIVE. MLIR prints `d1 - 2 >= 0`
        // and never `d1 + -2 >= 0`, and that is the lower half of every spanning constraint whose
        // page does NOT start at zero — `#set1` of `dcc/test/Dialect/Dataflow/paged_mem_view.mlir:11`
        // is `(d0 >= 0, -d0 + 63 >= 0, d1 - 2 >= 0, -d1 + 3 >= 0, d2 == 0)`. Same reason as the
        // negation below: printing the literal shape makes every such set a spurious diff.
        AffineExpr::Add(a, b) => match **b {
            AffineExpr::Const(n) if n < 0 => {
                format!("{} - {}", affine_expr(a, own, false), n.unsigned_abs())
            }
            _ => format!(
                "{} + {}",
                affine_expr(a, own, false),
                affine_expr(b, own, false)
            ),
        },
        // ⛔ TIMES MINUS ONE IS A NEGATION, NOT A PRODUCT. MLIR prints `-d2 + 63`, never
        // `d2 * -1 + 63`, and the upper half of every spanning constraint an `affine_set` carries
        // is exactly that shape (`#set`, `#set1`, `#set3` of the reference DataflowIR). Printing
        // the product makes every one of them a spurious diff against a vendored file.
        AffineExpr::Mul(a, b) if matches!(**b, AffineExpr::Const(-1)) => {
            format!("-{}", affine_expr(a, 2, false))
        }
        AffineExpr::Mul(a, b) => format!(
            "{} * {}",
            affine_expr(a, own, false),
            affine_expr(b, own, false)
        ),
        AffineExpr::Mod(a, b) => format!(
            "{} mod {}",
            affine_expr(a, own, true),
            affine_expr(b, own, true)
        ),
        AffineExpr::FloorDiv(a, b) => format!(
            "{} floordiv {}",
            affine_expr(a, own, true),
            affine_expr(b, own, true)
        ),
    };
    let atomic = own == 3;
    if !atomic && (own < parent || parent_is_divlike) {
        format!("({text})")
    } else {
        text
    }
}
