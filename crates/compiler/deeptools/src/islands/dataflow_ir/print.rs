//! THE ISLAND AS MLIR TEXT — the one place DataflowIR becomes characters.
//!
//! ⭐ DETERMINISTIC BY CONSTRUCTION. SSA names come from the builder's counter and attributes are
//! written in a fixed order, so two emissions of one program are byte-identical. That is what makes
//! the memoization in the bake queue sound: a group whose input recurs must hash the same.

use std::fmt::Write as _;

use crate::generated::{ParamKey, ParamValue, RegName};
use crate::islands::dataflow_ir::op::{Bound, Index, LogicKind, Op, Precision, RegAddr, Val};
use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, MemRef, Vector};
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
    for op in &program.body {
        emit(out, op, 3);
    }
    out.push_str("      return\n    }\n  }\n");
}

/// `affine_map<(d0, .., dn) -> (d0, .., dn)>` — the identity over `rank` dims.
///
/// ⭐ `load_order`/`store_order` SAY WHICH AXIS MOVES FASTEST, and the scheduler writes the identity
/// for every access in its own output (`#map4`, `#map8`). Row-major order is what the view's own
/// `layout_map` already states, so ordering it again differently would be two answers to one
/// question.
fn identity_map(rank: usize) -> String {
    let dims: Vec<String> = (0..rank).map(|d| format!("d{d}")).collect();
    format!("affine_map<({}) -> ({})>", dims.join(", "), dims.join(", "))
}

/// `affine_set<(d0, .., dn) : (d0 == 0, .., dn >= 0, -dn + LANES-1 >= 0)>` — which lanes are live.
///
/// ⭐⭐ THE INNERMOST AXIS IS THE LANE AXIS, and it is the only one that spans: every outer dim is
/// pinned to 0 and the last runs `0 .. lanes-1`. That is verbatim the shape the scheduler emits
/// (`#set1`, `#set3`), and it is the same "continuous prefix of live lanes" a
/// `create_affine_mask` carries — stated over the memref's dims instead of the vector's.
///
/// ⛔⛔ THE SPAN IS THE VECTOR'S, THE RANK IS THE MEMREF'S. This took `lanes` from the memref's
/// innermost dim, which is only the same number while every access reads a whole stick. A narrower
/// load — the single activation the PT's west port takes — is `vector<1xf16>` out of a
/// `memref<8x2x64xf16>`, and the backend refuses the mismatch outright: "Number of elements in
/// return type not matching with load_set/store_set elements".
fn lane_set(ty: &MemRef, lanes: u64) -> String {
    let rank = ty.shape.len();
    let dims: Vec<String> = (0..rank).map(|d| format!("d{d}")).collect();
    let last = rank.saturating_sub(1);
    let mut constraints: Vec<String> = (0..last).map(|d| format!("d{d} == 0")).collect();
    constraints.push(format!("d{last} >= 0"));
    constraints.push(format!("-d{last} + {} >= 0", lanes.saturating_sub(1)));
    format!(
        "affine_set<({}) : ({})>",
        dims.join(", "),
        constraints.join(", ")
    )
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

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

fn val(v: Val) -> String {
    format!("%{}", v.0)
}

fn vals(items: &[Val]) -> String {
    items
        .iter()
        .copied()
        .map(val)
        .collect::<Vec<_>>()
        .join(", ")
}

/// ONE OP.
#[expect(
    clippy::too_many_lines,
    reason = "one arm per DataflowIR op, each three lines: splitting it would put a dialect's \
              syntax in two files and make the next op's arm a choice of where to add it"
)]
fn emit(out: &mut String, op: &Op, depth: usize) {
    indent(out, depth);
    match op {
        Op::Constant { result, value } => {
            let _ = writeln!(out, "{} = arith.constant {value} : index", val(*result));
        }
        Op::GetUnit {
            result,
            residency,
            unit,
        } => {
            use crate::units::Residency;

            // Attributes are written in MLIR's own key order, which is alphabetical: core, corelet,
            // name, type. Which of `core`/`corelet` appear is the residency's, and a missing
            // `corelet` is a fact rather than a hole — `ExtendUnitNameToCorelet` errors on one that
            // is absent where it needs it, and reads `0` as the name suffix `"0"`
            // (`DataflowToSentient.cpp:104-117`).
            let spelling = unit.spelling();
            let (attrs, name) = match residency {
                Residency::Global => (String::new(), spelling.to_owned()),
                Residency::Scratchpad { core } => (
                    format!("core = {} : i32, ", core.get()),
                    format!("C{}-{spelling}", core.get()),
                ),
                Residency::CoreWide { core } => (
                    format!("core = {} : i32, corelet = 0 : i32, ", core.get()),
                    format!("C{}-{spelling}", core.get()),
                ),
                Residency::Corelet { core, corelet } => (
                    format!(
                        "core = {} : i32, corelet = {} : i32, ",
                        core.get(),
                        corelet.get()
                    ),
                    format!("C{}-{spelling}-CL{}", core.get(), corelet.get()),
                ),
            };
            let _ = writeln!(
                out,
                "{} = dataflow.get_unit {{{attrs}name = \"{name}\", type = \"{spelling}\"}} : index",
                val(*result),
            );
        }
        Op::GetLocalUnit { result, of, which } => {
            let _ = writeln!(
                out,
                "{} = dataflow.get_local_unit {} {{name = \"{}\"}} : index",
                val(*result),
                val(*of),
                which.spelling()
            );
        }
        Op::GetLogicalMemoryView {
            result,
            from,
            start,
            layout,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = dataflow.get_logical_memory_view {}, {} {{layout_map = {}}} : index, index, {}",
                val(*result),
                val(*from),
                val(*start),
                affine_map(layout),
                memref(ty)
            );
        }
        Op::ProgramUnit {
            units,
            precision,
            body,
        } => {
            let precision = match precision {
                Some(p) => format!(" {{precision = \"{}\"}}", Precision::spelling(*p)),
                None => String::new(),
            };
            let _ = writeln!(out, "dataflow.program_unit {}{precision} : {{", vals(units));
            for inner in body {
                emit(out, inner, depth + 1);
            }
            indent(out, depth);
            out.push_str("}\n");
        }
        Op::Estimate {
            result,
            input,
            kind,
            version,
            input_ty,
            ty,
        } => {
            let version = match version {
                Some(v) => format!(
                    " {{version = #vectorchain<{} {}>}}",
                    kind.version_mnemonic(),
                    v.spelling()
                ),
                None => String::new(),
            };
            let _ = writeln!(
                out,
                "{} = vectorchain.{} {}{version} : {}, {}",
                val(*result),
                kind.spelling(),
                val(*input),
                vector(*input_ty),
                vector(*ty)
            );
        }
        Op::ScanWithGap {
            result,
            input,
            reduction_op,
            input_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.scan_with_gap {} {{reduction_op = #vectorchain<binary_operator {}>, \
                 gap = 8 : index, eval_order = #vectorchain<eval_order left_to_right>}} : {}, {}",
                val(*result),
                val(*input),
                reduction_op.spelling(),
                vector(*input_ty),
                vector(*ty)
            );
        }
        Op::True { result } => {
            let _ = writeln!(out, "{} = arith.constant true", val(*result));
        }
        Op::Compare {
            result,
            iv,
            against,
        } => {
            let _ = writeln!(
                out,
                "{} = arith.cmpi eq, {}, {} : index",
                val(*result),
                val(*iv),
                val(*against)
            );
        }
        Op::Logic {
            result,
            kind,
            operands,
        } => {
            let mnemonic = match kind {
                LogicKind::And => "arith.andi",
                LogicKind::Or => "arith.ori",
                LogicKind::Not => "arith.xori",
            };
            let _ = writeln!(out, "{} = {mnemonic} {} : i1", val(*result), vals(operands));
        }
        Op::If {
            cond,
            body,
            else_body,
        } => {
            let _ = writeln!(out, "scf.if {} {{", val(*cond));
            for inner in body {
                emit(out, inner, depth + 1);
            }
            indent(out, depth);
            if else_body.is_empty() {
                out.push_str("}\n");
            } else {
                out.push_str("} else {\n");
                for inner in else_body {
                    emit(out, inner, depth + 1);
                }
                indent(out, depth);
                out.push_str("}\n");
            }
        }
        Op::For { iv, lo, hi, body } => {
            let _ = writeln!(
                out,
                "affine.for {} = {} to {} {{",
                val(*iv),
                bound(*lo),
                bound(*hi)
            );
            for inner in body {
                emit(out, inner, depth + 1);
            }
            indent(out, depth);
            out.push_str("}\n");
        }
        Op::Apply { result, map, args } => {
            let _ = writeln!(
                out,
                "{} = affine.apply {}({})",
                val(*result),
                affine_map(map),
                vals(args)
            );
        }
        Op::AgenVectorLoad {
            result,
            view,
            indices,
            view_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = agen.vector_load {}[{}] {{load_order = {}, load_set = {}}} : {}, {}",
                val(*result),
                val(*view),
                index_list(indices),
                identity_map(view_ty.shape.len()),
                lane_set(view_ty, ty.len),
                memref(view_ty),
                vector(*ty)
            );
        }
        Op::AgenVectorStore {
            value,
            view,
            indices,
            view_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "agen.vector_store {}, {}[{}] {{store_order = {}, store_set = {}}} : {}, {}",
                val(*value),
                val(*view),
                index_list(indices),
                identity_map(view_ty.shape.len()),
                lane_set(view_ty, ty.len),
                memref(view_ty),
                vector(*ty)
            );
        }
        Op::VectorLoad {
            result,
            view,
            indices,
            view_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = affine.vector_load {}[{}] : {}, {}",
                val(*result),
                val(*view),
                index_list(indices),
                memref(view_ty),
                vector(*ty)
            );
        }
        Op::VectorStore {
            value,
            view,
            indices,
            view_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "affine.vector_store {}, {}[{}] : {}, {}",
                val(*value),
                val(*view),
                index_list(indices),
                memref(view_ty),
                vector(*ty)
            );
        }
        Op::Send { to, data, ty } => {
            let _ = writeln!(
                out,
                "dataflow.send {}, {} : {}",
                val(*to),
                val(*data),
                vector(*ty)
            );
        }
        Op::Receive { result, from, ty } => {
            let _ = writeln!(
                out,
                "{} = dataflow.receive {} : {}",
                val(*result),
                val(*from),
                vector(*ty)
            );
        }
        Op::SyncSend { to, signal } => {
            let _ = writeln!(
                out,
                // ⛔⛔ THE WAIT MODE IS OPTIONAL IN THE DIALECT AND MANDATORY TO LOWER.
                // `Dataflow.td:200` declares it `OptionalAttr<BoolAttr>`, but
                // `DataflowToSentient.cpp:219-222` fails outright when a send has none: "Unknown
                // async transfers modes for sentient".
                //
                // ⭐ `true` IS THE ONLY SAFE ANSWER HERE, and it is what the scheduler writes for
                // every send in its own output. It "controls whether the sender waits for
                // outstanding asynchronous transfers before signalling" (`:191-192`) — and a
                // `ddl.sync` exists to say the data has ARRIVED, so a signal that does not wait can
                // be observed before the transfer it is announcing lands. `false` is an
                // optimisation that needs something else to order the transfers, and no vendored
                // template states one: `ddl.sync` carries `units`, `signal_name` and `receive`, and
                // nothing about async waits.
                "dataflow.sync_send {} {{dbgName = \"{}\", \
                 wait_immediately_for_async_transfers = true}} : index",
                val(*to),
                signal.spelling()
            );
        }
        Op::SyncRecv { from, signal } => {
            let _ = writeln!(
                out,
                "dataflow.sync_recv {} {{dbgName = \"{}\"}} : index",
                val(*from),
                signal.spelling()
            );
        }
        Op::ImplicitSync {
            view,
            dst,
            size,
            view_ty,
        } => {
            let _ = writeln!(
                out,
                "dataflow.implicit_sync_on_streaming_buffer {}, {}, {} : {}, index, index",
                val(*view),
                val(*dst),
                val(*size),
                memref(view_ty)
            );
        }
        Op::Opaque {
            func,
            read_write,
            read_only,
            params,
        } => {
            let _ = writeln!(
                out,
                "dataflow.opaque {{func_name = \"{}\", parameter_dictionary = {}, \
                 read_only_register_dictionary = {}, read_write_register_dictionary = {}}}",
                func.spelling().to_lowercase(),
                dictionary(params, ParamKey::spelling, |v| ParamValue::spelling(v)
                    .to_owned()),
                // ⛔⛔ THE `R` IS LOAD-BEARING, NOT DECORATION. `insertReg` writes
                // `"R" + std::to_string(startAddress)` (`ddc/ddcv1.cpp:3350`) and the consumer
                // takes it back off by POSITION: `port_str = "lrf" + port_str.erase(0, 1)`
                // (`dcc/src/Dialect/Sentient/Utils.cpp:157`). A bare `"20"` becomes `lrf0` — the
                // wrong register, silently — and a bare `"8"` becomes `"lrf"`, which is no port at
                // all and aborts with "Invalid port".
                dictionary(read_only, RegName::spelling, |a: RegAddr| format!(
                    "R{}",
                    a.0
                )),
                dictionary(read_write, RegName::spelling, |a: RegAddr| format!(
                    "R{}",
                    a.0
                ))
            );
        }
        Op::Select {
            result,
            input,
            selection_map,
            input_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.select {} {{selection_map = {}}} : {}, {}",
                val(*result),
                val(*input),
                affine_map(selection_map),
                vector(*input_ty),
                vector(*ty)
            );
        }
        Op::Multiply {
            result,
            a,
            b,
            reduction_map,
            operand_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.multiply {}, {} {{reduction_map = {}}} : {}, {}, {}",
                val(*result),
                val(*a),
                val(*b),
                affine_map(reduction_map),
                vector(*operand_ty),
                vector(*operand_ty),
                vector(*ty)
            );
        }
        Op::MultiplyAccumulate {
            result,
            a,
            b,
            acc,
            reduction_map,
            operand_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.multiply_and_accumulate {}, {}, {} {{reduction_map = {}}} : {}, {}, {}, {}",
                val(*result),
                val(*a),
                val(*b),
                val(*acc),
                affine_map(reduction_map),
                vector(*operand_ty),
                vector(*operand_ty),
                vector(*ty),
                vector(*ty)
            );
        }
        Op::ElementWiseCompare {
            result,
            op1,
            op2,
            mask,
            compare_op,
            operand_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.element_wise_compare {}, {}[{} : {}] {{compare_op = #vectorchain<element_wise_compare_operator {}>}} : {}, {}, {}",
                val(*result),
                val(*op1),
                val(*op2),
                val(*mask),
                vector(*ty),
                compare_op.spelling(),
                vector(*operand_ty),
                vector(*operand_ty),
                vector(*ty)
            );
        }
        Op::ElementWiseSelection {
            result,
            cond,
            lhs,
            rhs,
            mask,
            cond_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.element_wise_selection {} ? {} : {} [{} : {}] : {}, {}, {}, {}",
                val(*result),
                val(*cond),
                val(*lhs),
                val(*rhs),
                val(*mask),
                vector(*cond_ty),
                vector(*cond_ty),
                vector(*ty),
                vector(*ty),
                vector(*ty)
            );
        }
        Op::Binary {
            result,
            op1,
            op2,
            mask,
            binary_op,
            op_specific_map,
            operand_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.binary {}, {}[{} : {}] {{binary_op = #vectorchain<binary_operator {}>, op_specific_map = {}}} : {}, {}, {}",
                val(*result),
                val(*op1),
                val(*op2),
                val(*mask),
                mask_ty(*ty),
                binary_op.spelling(),
                affine_map(op_specific_map),
                vector(*operand_ty),
                vector(*operand_ty),
                vector(*ty)
            );
        }
        Op::DenseConstant { result, one, ty } => {
            // MLIR prints a float splat in scientific form, which is what the vendored IR shows:
            // `arith.constant dense<0.000000e+00> : vector<64xf16>`.
            let literal = match (one, ty.elem) {
                (false, ElemType::Int(_)) => "0".to_owned(),
                (true, ElemType::Int(_)) => "1".to_owned(),
                (false, _) => "0.000000e+00".to_owned(),
                (true, _) => "1.000000e+00".to_owned(),
            };
            let _ = writeln!(
                out,
                "{} = arith.constant dense<{literal}> : {}",
                val(*result),
                vector(*ty)
            );
        }
        Op::ConstantBitstream { result, value, ty } => {
            let values = value
                .iter()
                .map(|bits| format!("{bits:#x}"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                out,
                "{} = vectorchain.constant_bitstream {{value = [{values}]}} : {}",
                val(*result),
                vector(*ty)
            );
        }
        Op::Shuffle {
            result,
            input,
            indices,
            repetition,
            input_ty,
            ty,
        } => {
            let indices = indices
                .iter()
                .map(|index| format!("{index} : i32"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                out,
                "{} = vectorchain.shuffle input({}) {{indices = [{indices}], repetition = {repetition} : i32}} : {}, {}",
                val(*result),
                val(*input),
                vector(*input_ty),
                vector(*ty)
            );
        }
        Op::Cast {
            result,
            input,
            input_ty,
            ty,
        } => {
            let _ = writeln!(
                out,
                "{} = vectorchain.cast {} : {}, {}",
                val(*result),
                val(*input),
                vector(*input_ty),
                vector(*ty)
            );
        }
        Op::CreateAffineMask { result, lanes, ty } => {
            // A continuous prefix of live lanes, as `getStaticContinuousMaskValue` builds.
            let _ = writeln!(
                out,
                "{} = vectorchain.create_affine_mask {{mask_set = affine_set<(d0) : (d0 >= 0, -d0 + {} >= 0)>}} : {}",
                val(*result),
                lanes.saturating_sub(1),
                vector(*ty)
            );
        }
    }
}

fn bound(b: Bound) -> String {
    match b {
        Bound::Const(n) => n.to_string(),
        Bound::Val(v) => val(v),
    }
}

fn index_list(indices: &[Index]) -> String {
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

/// A `{key = "value", ..}` attribute dictionary, KEY-SORTED.
///
/// ⛔ SORTED BECAUSE MLIR SORTS. A `DictionaryAttr` is stored key-ordered, so a round trip through
/// the parser reorders anything else — and a printer whose output does not survive a round trip
/// cannot be checked against the vendored files.
/// ONE MLIR DICTIONARY ATTRIBUTE, sorted by key so the emission is byte-reproducible.
///
/// ⭐ THE KEYS AND VALUES ARRIVE AS TOKENS and are spelled here, at the one place text is the point.
/// Everything upstream holds enums, so a key that does not exist is a variant that does not exist.
fn dictionary<K: Copy, V: Copy>(
    entries: &[(K, V)],
    key: impl Fn(K) -> &'static str,
    value: impl Fn(V) -> String,
) -> String {
    let mut sorted: Vec<(&'static str, String)> =
        entries.iter().map(|(k, v)| (key(*k), value(*v))).collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    format!(
        "{{{}}}",
        sorted
            .iter()
            .map(|(key, value)| format!("{key} = \"{value}\""))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn memref(ty: &MemRef) -> String {
    let shape = ty
        .shape
        .iter()
        .map(|extent| format!("{extent}x"))
        .collect::<String>();
    format!("memref<{shape}{}>", elem(ty.elem))
}

/// The i1 mask type that matches a vector's lane count.
fn mask_ty(ty: Vector) -> String {
    format!("vector<{}xi1>", ty.len)
}

fn vector(ty: Vector) -> String {
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
        ElemType::F4E2M1Fn => "f4E2M1FN".to_owned(),
        ElemType::MxFloat(bits) => format!("!dataflow.mxfloat<{bits}>"),
    }
}

fn affine_map(map: &AffineMap) -> String {
    let dims = (0..map.dims)
        .map(|d| format!("d{d}"))
        .collect::<Vec<_>>()
        .join(", ");
    let results = map
        .results
        .iter()
        .map(|expr| affine_expr(expr, 0, false))
        .collect::<Vec<_>>()
        .join(", ");
    format!("affine_map<({dims}) -> ({results})>")
}

/// BINDING POWER, so the printed map is the one MLIR would print.
///
/// `+` is loosest; `*`, `mod` and `floordiv` bind tighter; a dimension or a literal is atomic.
const fn precedence(expr: &AffineExpr) -> u8 {
    match expr {
        AffineExpr::Dim(_) | AffineExpr::Const(_) => 3,
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
        AffineExpr::Const(n) => n.to_string(),
        AffineExpr::Add(a, b) => format!(
            "{} + {}",
            affine_expr(a, own, false),
            affine_expr(b, own, false)
        ),
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

#[cfg(test)]
mod tests {
    use super::{Op, Val, emit};
    use crate::units::{Core, Corelet, DfirUnit, Residency};

    /// A core this build's arch has.
    fn core(index: u32) -> Core {
        Core::checked(index).expect("cores 0 and 1 exist on every arch this crate builds for")
    }

    /// A corelet this build's arch has.
    fn corelet(index: u32) -> Corelet {
        Corelet::checked(index).expect("corelets 0 and 1 exist on every arch this crate builds for")
    }

    /// ⭐⭐ IBM'S OWN UNIT BLOCK, REPRODUCED — all four residency classes, carried as TEXT rather
    /// than as a shape.
    ///
    /// The nineteen lines are `/tmp/ktir_ref/export/debug/dfir.mlir:45-63`, the DataflowIR the
    /// stock KTIR pathway produced on a two-core two-corelet run. They are the whole point of
    /// [`Residency`]: `C0-l3lu` carries `corelet = 0`, `C0-lx` carries NO corelet, and `hbm`
    /// carries NEITHER attribute — three distinct attribute shapes, of which the previous
    /// `core: u32, corelet: Option<u32>` pair could spell two.
    ///
    /// ⛔ AN ASSERTION ON THE SHAPE WOULD NOT HAVE CAUGHT THE BUG THIS REPLACES. "The HBM line has
    /// no `core`" passes just as well on an emitter that also drops `core` from the LX. This
    /// compares IBM's bytes in IBM's order, so a wrong attribute set on any one of the four classes
    /// is a diff rather than a silent agreement.
    #[test]
    fn reproduces_ibms_unit_block() {
        let mut ops = Vec::new();
        let mut next = 0u32;
        let mut unit = |residency, unit| {
            let op = Op::GetUnit {
                result: Val(next),
                residency,
                unit,
            };
            next += 1;
            ops.push(op);
        };

        // Non-parallel compute, declared in the core group: `core` AND `corelet = 0`.
        for which in [DfirUnit::L3lu, DfirUnit::L3su] {
            for c in 0..2 {
                unit(Residency::CoreWide { core: core(c) }, which);
            }
        }
        // Parallel compute, declared in the corelet group. Corelet-major, as IBM emits it.
        for which in [DfirUnit::Lxlu, DfirUnit::Sfp, DfirUnit::Lxsu] {
            for cl in 0..2 {
                for c in 0..2 {
                    unit(
                        Residency::Corelet {
                            core: core(c),
                            corelet: corelet(cl),
                        },
                        which,
                    );
                }
            }
        }
        // Then the memories: the global root first, then one scratchpad per core.
        unit(Residency::Global, DfirUnit::Hbm);
        for c in 0..2 {
            unit(Residency::Scratchpad { core: core(c) }, DfirUnit::Lx);
        }

        let mut got = String::new();
        for op in &ops {
            emit(&mut got, op, 0);
        }

        let want = r#"%0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3lu", type = "l3lu"} : index
%1 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l3lu", type = "l3lu"} : index
%2 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l3su", type = "l3su"} : index
%3 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l3su", type = "l3su"} : index
%4 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxlu-CL0", type = "lxlu"} : index
%5 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-lxlu-CL0", type = "lxlu"} : index
%6 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxlu-CL1", type = "lxlu"} : index
%7 = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-lxlu-CL1", type = "lxlu"} : index
%8 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfp-CL0", type = "sfp"} : index
%9 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-sfp-CL0", type = "sfp"} : index
%10 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-sfp-CL1", type = "sfp"} : index
%11 = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-sfp-CL1", type = "sfp"} : index
%12 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-lxsu-CL0", type = "lxsu"} : index
%13 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-lxsu-CL0", type = "lxsu"} : index
%14 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-lxsu-CL1", type = "lxsu"} : index
%15 = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-lxsu-CL1", type = "lxsu"} : index
%16 = dataflow.get_unit {name = "hbm", type = "hbm"} : index
%17 = dataflow.get_unit {core = 0 : i32, name = "C0-lx", type = "lx"} : index
%18 = dataflow.get_unit {core = 1 : i32, name = "C1-lx", type = "lx"} : index
"#;

        for (at, (want_line, got_line)) in want.lines().zip(got.lines()).enumerate() {
            assert_eq!(
                want_line, got_line,
                "unit line {at} diverges from IBM's own DataflowIR"
            );
        }
        assert_eq!(
            want.lines().count(),
            got.lines().count(),
            "emitted a different number of unit lines than IBM's block has"
        );
    }
}
