// SPDX-License-Identifier: Apache-2.0
//! REGENERATES `minted.rs`, the sampler fixture of ktir-emulator's `sampler_on_emulator` test.
//!
//! The sampler is minted outside this crate (scratchy-subtile mints the `Sample` node, the Triton
//! splice compiles `crates/targets/spyre/kernels/sample_fwd.py` into its KTIR), so this is NOT a
//! ktir-emulator target (the crate states its tests explicitly; nothing here is built with it).
//! It is a scratchy-target-spyre integration test, run by copying it in. From the repo root:
//!
//! ```text
//! cp crates/targets/spyre/ktir/ktir-emulator/tests/fixtures/sampler_granite/regenerate.rs \
//!    crates/targets/spyre/tests/regenerate_sampler_fixture.rs
//! cargo test -p scratchy-target-spyre --features spyre-emu \
//!    --test regenerate_sampler_fixture -- --ignored
//! rm crates/targets/spyre/tests/regenerate_sampler_fixture.rs
//! ```
//!
//! It runs the production mint (the same calls the bake makes) and writes the program as
//! ktir-core const data, with the facts the emulator needs to run it: which tensor each argument
//! is bound to, and which load-time table each tensor holds, in which arrangement. Table BYTES
//! are not written: the emulator test builds them.
#![cfg(feature = "spyre-emu")]

use std::fmt::Write as _;

use ktir_core::affine::{AffineExpr, AffineMap, AffineSet};
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;
use ktir_superdsc::element_arrangement::ElementArrangement;
use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{SampleRows, SourceShape, SubOp, lower_region, mint_sample};
use scratchy_target_spyre::wiring::LoadConst;

/// granite-3.1-2b-instruct's vocabulary: the row the draw runs over.
const VOCAB: u32 = 49155;

fn git(args: &[&str]) -> String {
    std::process::Command::new("git")
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Which of a function's arguments it WRITES: a `ktdp.store`'s tile names a
/// `construct_access_tile`, whose view names a `construct_memory_view`, whose first operand is
/// the argument (the rule `scratchy_target_spyre::runner` builds its session spec by).
fn written_args(f: &IRFunction<'static>) -> std::collections::HashSet<Ssa> {
    let producer = |v: Ssa| f.operations.iter().find(|o| o.result == Some(v));
    let mut out = std::collections::HashSet::new();
    for op in f.operations {
        if op.op_type != OpKind::KtdpStore {
            continue;
        }
        let view = op
            .operands
            .get(1)
            .and_then(|&t| producer(t))
            .and_then(|t| t.operands.first().copied())
            .and_then(producer);
        if let Some(arg) = view.and_then(|v| v.operands.first().copied()) {
            out.insert(arg);
        }
    }
    out
}

/// Renders ktir-core values as const Rust, noting which types the text names.
#[derive(Default)]
struct Render {
    used: std::collections::BTreeSet<&'static str>,
}

impl Render {
    fn ssa(&mut self, s: Ssa) -> String {
        self.used.insert("Ssa");
        format!("Ssa({})", s.0)
    }

    fn dtype(&mut self, d: ktir_core::dtypes::DType) -> String {
        self.used.insert("DType");
        format!("DType::{d:?}")
    }

    fn ty(&mut self, t: IrType<'_>) -> String {
        self.used.insert("IrType");
        match t {
            IrType::Tensor { dims, elem } => {
                format!(
                    "IrType::Tensor {{ dims: &{dims:?}, elem: {} }}",
                    self.dtype(elem)
                )
            }
            IrType::MemRef { dims, elem } => {
                format!(
                    "IrType::MemRef {{ dims: &{dims:?}, elem: {} }}",
                    self.dtype(elem)
                )
            }
            IrType::AccessTile { dims } => format!("IrType::AccessTile {{ dims: &{dims:?} }}"),
            IrType::Index => "IrType::Index".to_string(),
            IrType::Scalar(d) => format!("IrType::Scalar({})", self.dtype(d)),
        }
    }

    fn float(f: f64) -> String {
        if f.is_nan() {
            "f64::NAN".to_string()
        } else if f == f64::INFINITY {
            "f64::INFINITY".to_string()
        } else if f == f64::NEG_INFINITY {
            "f64::NEG_INFINITY".to_string()
        } else {
            format!("{f:?}")
        }
    }

    fn expr(&mut self, e: &AffineExpr<'_>) -> String {
        self.used.insert("AffineExpr");
        let two = |r: &mut Self, v: &str, a: &AffineExpr<'_>, b: &AffineExpr<'_>| {
            format!("AffineExpr::{v}(&{}, &{})", r.expr(a), r.expr(b))
        };
        match e {
            AffineExpr::Dim(i) => format!("AffineExpr::Dim({i})"),
            AffineExpr::Sym(i) => format!("AffineExpr::Sym({i})"),
            AffineExpr::Const(c) => format!("AffineExpr::Const({c})"),
            AffineExpr::Ref(s) => format!("AffineExpr::Ref({s:?})"),
            AffineExpr::Neg(a) => format!("AffineExpr::Neg(&{})", self.expr(a)),
            AffineExpr::Add(a, b) => two(self, "Add", a, b),
            AffineExpr::Sub(a, b) => two(self, "Sub", a, b),
            AffineExpr::Mul(a, b) => two(self, "Mul", a, b),
            AffineExpr::FloorDiv(a, b) => two(self, "FloorDiv", a, b),
            AffineExpr::Mod(a, b) => two(self, "Mod", a, b),
            AffineExpr::Max(a, b) => two(self, "Max", a, b),
            AffineExpr::Min(a, b) => two(self, "Min", a, b),
        }
    }

    fn map(&mut self, m: &AffineMap<'_>) -> String {
        self.used.insert("AffineMap");
        let exprs: Vec<String> = m.exprs.iter().map(|e| self.expr(e)).collect();
        format!(
            "AffineMap {{ num_dims: {}, num_syms: {}, exprs: &[{}] }}",
            m.num_dims,
            m.num_syms,
            exprs.join(", ")
        )
    }

    fn set(&mut self, s: &AffineSet<'_>) -> String {
        self.used.insert("AffineSet");
        self.used.insert("Constraint");
        self.used.insert("ConstraintKind");
        let cs: Vec<String> = s
            .constraints
            .iter()
            .map(|c| {
                format!(
                    "Constraint {{ expr: {}, kind: ConstraintKind::{:?} }}",
                    self.expr(&c.expr),
                    c.kind
                )
            })
            .collect();
        format!(
            "AffineSet {{ num_dims: {}, num_syms: {}, constraints: &[{}] }}",
            s.num_dims,
            s.num_syms,
            cs.join(", ")
        )
    }

    fn attr(&mut self, a: &Attr<'_>) -> String {
        self.used.insert("Attr");
        match a {
            Attr::Int(i) => format!("Attr::Int({i})"),
            Attr::IntList(v) => format!("Attr::IntList(&{v:?})"),
            Attr::Float(f) => format!("Attr::Float({})", Self::float(*f)),
            Attr::FloatList(v) => {
                let fs: Vec<String> = v.iter().map(|f| Self::float(*f)).collect();
                format!("Attr::FloatList(&[{}])", fs.join(", "))
            }
            Attr::Str(s) => format!("Attr::Str({s:?})"),
            Attr::StrList(v) => format!("Attr::StrList(&{v:?})"),
            Attr::Bool(b) => format!("Attr::Bool({b})"),
            Attr::Dtype(d) => format!("Attr::Dtype({})", self.dtype(*d)),
            Attr::Op(k) => {
                self.used.insert("OpKind");
                format!("Attr::Op(OpKind::{k:?})")
            }
            Attr::Ssas(v) => {
                let ss: Vec<String> = v.iter().map(|s| self.ssa(*s)).collect();
                format!("Attr::Ssas(&[{}])", ss.join(", "))
            }
            Attr::AffineMap(m) => format!("Attr::AffineMap({})", self.map(m)),
            Attr::AffineMapList(ms) => {
                let ms: Vec<String> = ms.iter().map(|m| self.map(m)).collect();
                format!("Attr::AffineMapList(&[{}])", ms.join(", "))
            }
            Attr::AffineSet(s) => format!("Attr::AffineSet({})", self.set(s)),
        }
    }

    fn op(&mut self, o: &Operation<'_>, indent: usize) -> String {
        self.used.insert("Operation");
        self.used.insert("OpKind");
        self.used.insert("AttrKey");
        let result = match o.result {
            Some(s) => format!("Some({})", self.ssa(s)),
            None => "None".to_string(),
        };
        let operands: Vec<String> = o.operands.iter().map(|s| self.ssa(*s)).collect();
        let attrs: Vec<String> = o
            .attributes
            .iter()
            .map(|(k, v)| format!("(AttrKey::{k:?}, {})", self.attr(v)))
            .collect();
        let result_type = match o.result_type {
            Some(t) => format!("Some({})", self.ty(t)),
            None => "None".to_string(),
        };
        let pad = " ".repeat(indent);
        let regions: Vec<String> = o
            .regions
            .iter()
            .map(|r| {
                let ops: Vec<String> = r
                    .iter()
                    .map(|o| format!("{pad}        {},\n", self.op(o, indent + 8)))
                    .collect();
                format!("&[\n{}{pad}    ]", ops.concat())
            })
            .collect();
        format!(
            "Operation {{ result: {result}, op_type: OpKind::{:?}, operands: &[{}], attributes: \
             &[{}], result_type: {result_type}, regions: &[{}] }}",
            o.op_type,
            operands.join(", "),
            attrs.join(", "),
            regions.join(", ")
        )
    }
}

#[test]
#[ignore = "regenerates ktir-emulator's sampler fixture; run by hand, see the module doc"]
fn regenerate_sampler_fixture() {
    // The one-request lm-head shape the mint admits: `[1, 1] x [1, vocab]`.
    let input = LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: 1, cols: 1 },
            SourceShape {
                rows: 1,
                cols: VOCAB,
            },
        ],
        ops: vec![OpDesc {
            op: SubOp::MatmulTile {
                n: VOCAB,
                weight: GemmWeight::Dense,
            },
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
        }],
        result: 0,
    };
    let mut ir = lower_region(&input, std::num::NonZeroU32::new(64).unwrap());
    let mint = mint_sample(&mut ir, SampleRows::All);
    let node = ir
        .nodes
        .iter()
        .find(|n| matches!(n.op, SubOp::Sample { .. }))
        .expect("the mint appends a Sample node");
    let spliced = scratchy_triton_splice::lower(node, &ir, false).expect("the sampler splices");
    let k = spliced
        .ktir
        .as_ref()
        .expect("the spliced op carries its program");
    let func = k.func;

    // Every argument's tensor, and whether the program writes it.
    let writes = written_args(&func);
    let mut r = Render::default();
    let bindings: Vec<String> = func
        .arguments
        .iter()
        .map(|(ssa, _)| {
            let tid = u64::from(k.bindings[ssa.slot()].get());
            format!("    ({}, {tid}, {}),\n", r.ssa(*ssa), writes.contains(ssa))
        })
        .collect();

    // The load-time tables, by the bake's own derivations: which tensor, which of the
    // sampler's operand tables, its dtype and shape, and its arrangement.
    let staggered = scratchy_target_spyre::lower_subtile_tape_to_ktir::staggered_of_ir(&ir)
        .expect("arrangements");
    let consts =
        scratchy_target_spyre::lower_subtile_tape_to_ktir::load_consts_of_ir(&ir, &staggered)
            .expect("the load-const list derives");
    let tables: Vec<String> = consts
        .iter()
        .map(|&(tid, kind)| {
            let LoadConst::SampleTable {
                operand,
                arrangement,
                ..
            } = kind
            else {
                panic!("the sampler binds only its own tables, got {kind:?}");
            };
            let (_, dt, shape) = kind.build();
            let dt = match dt {
                scratchy_tensors::DType::F32 => ktir_core::dtypes::DType::F32,
                scratchy_tensors::DType::I32 => ktir_core::dtypes::DType::I32,
                scratchy_tensors::DType::F16 => ktir_core::dtypes::DType::F16,
                other => panic!("operand {operand} stages as {other:?}"),
            };
            let stag = arrangement == ElementArrangement::Dl16ToFp32;
            format!(
                "    ({tid}, {operand}, {}, {shape:?}, {stag}),\n",
                r.dtype(dt)
            )
        })
        .collect();

    let args: Vec<String> = func
        .arguments
        .iter()
        .map(|(s, t)| format!("({}, {})", r.ssa(*s), r.ty(*t)))
        .collect();
    let ops: Vec<String> = func
        .operations
        .iter()
        .map(|o| format!("        {},\n", r.op(o, 8)))
        .collect();
    let return_type = match func.return_type {
        Some(s) => format!("Some({s:?})"),
        None => "None".to_string(),
    };

    let row_tid = node.inputs[0].tensor.index();
    let pad_vocab = ir.tensors[row_tid].cols;
    // The function itself is always named; every other type only when the text uses it.
    r.used.insert("IRFunction");
    let mut uses = String::new();
    for (ty, path) in [
        ("AffineExpr", "affine::AffineExpr"),
        ("AffineMap", "affine::AffineMap"),
        ("AffineSet", "affine::AffineSet"),
        ("Constraint", "affine::Constraint"),
        ("ConstraintKind", "affine::ConstraintKind"),
        ("AttrKey", "attrkey::AttrKey"),
        ("DType", "dtypes::DType"),
        ("Attr", "ir::Attr"),
        ("IRFunction", "ir::IRFunction"),
        ("Operation", "ir::Operation"),
        ("Ssa", "ir::Ssa"),
        ("IrType", "irtype::IrType"),
        ("OpKind", "opkind::OpKind"),
    ] {
        if r.used.contains(ty) {
            writeln!(uses, "use ktir_emulator::{path};").unwrap();
        }
    }

    let head = git(&["rev-parse", "HEAD"]);
    let sampler_sha = git(&["log", "-1", "--format=%H", "--", "kernels/sample_fwd.py"]);
    let splice_sha = git(&["log", "-1", "--format=%H", "--", "../../triton/splice"]);
    let text = format!(
        "// SPDX-License-Identifier: Apache-2.0\n\
         // GENERATED by `regenerate.rs` beside this file; do not edit by hand.\n\
         //\n\
         // THE MINTED SAMPLER, PINNED. The program the spyre target mints for granite-3.1-2b-instruct's\n\
         // {VOCAB}-wide logits row (padded to {pad_vocab}): scratchy-subtile's `mint_sample(SampleRows::All)`\n\
         // over the one-request lm-head `[1, 1] x [1, {VOCAB}]`, spliced by `scratchy_triton_splice::lower`\n\
         // from `crates/targets/spyre/kernels/sample_fwd.py`.\n\
         //\n\
         // Dumped at commit {head}, where\n\
         //   crates/targets/spyre/kernels/sample_fwd.py was last changed in {sampler_sha}, and\n\
         //   crates/triton/splice was last changed in {splice_sha}.\n\
         // This fixture pins THAT sampler: a later change to either does not reach the emulator test\n\
         // until this file is regenerated. Regenerate from the repo root with:\n\
         //   cp crates/targets/spyre/ktir/ktir-emulator/tests/fixtures/sampler_granite/regenerate.rs \\\n\
         //      crates/targets/spyre/tests/regenerate_sampler_fixture.rs\n\
         //   cargo test -p scratchy-target-spyre --features spyre-emu \\\n\
         //      --test regenerate_sampler_fixture -- --ignored\n\
         //   rm crates/targets/spyre/tests/regenerate_sampler_fixture.rs\n\
         \n\
         use ktir_superdsc::ktir_node::Program;\n\
         {uses}\n\
         /// The row's real width, and the padded width the program and its tables run over.\n\
         pub const VOCAB: usize = {VOCAB};\n\
         pub const PAD_VOCAB: usize = {pad_vocab};\n\
         /// The logits row the draw reads, and the index stick it writes.\n\
         pub const ROW_TID: u64 = {row_tid};\n\
         pub const IDX_TID: u64 = {idx_tid};\n\
         /// The seed quadruple's four `[1, 1]` i32 words, and the temperature's reciprocal (f32).\n\
         pub const SEED_TIDS: [u64; 4] = {seed_tids:?};\n\
         pub const INVT_TID: u64 = {invt_tid};\n\
         \n\
         /// (argument, the tensor it is bound to, whether the program writes it)\n\
         pub const BINDINGS: &[(Ssa, u64, bool)] = &[\n{bindings}];\n\
         \n\
         /// The load-time tables: (tensor, the sampler's operand table number, dtype, [rows, cols],\n\
         /// staged in the `Dl16ToFp32` arrangement). Their bytes are built by the test.\n\
         pub const TABLES: &[(u64, u8, DType, [usize; 2], bool)] = &[\n{tables}];\n\
         \n\
         /// What the node computes, as the mint states it.\n\
         pub const PROGRAM: Program = Program::{program:?};\n\
         \n\
         pub static SAMPLER: IRFunction<'static> = IRFunction {{\n\
         \x20   name: {name:?},\n\
         \x20   arguments: &[{args}],\n\
         \x20   operations: &[\n{ops}    ],\n\
         \x20   grid: {grid:?},\n\
         \x20   return_type: {return_type},\n\
         }};\n",
        idx_tid = node.output.tensor.index(),
        seed_tids = mint.seed_words.map(|t| t.index() as u64),
        invt_tid = mint.invt.index(),
        bindings = bindings.concat(),
        tables = tables.concat(),
        program = k.program,
        name = func.name,
        args = args.join(", "),
        ops = ops.concat(),
        grid = func.grid,
    );
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("ktir/ktir-emulator/tests/fixtures/sampler_granite/minted.rs");
    std::fs::write(&out, text).expect("write the fixture");
    eprintln!("wrote {}", out.display());
}
