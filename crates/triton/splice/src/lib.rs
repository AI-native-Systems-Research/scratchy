// SPDX-License-Identifier: Apache-2.0
//! THE SPLICE — where the tape's node and the kernel meet.
//!
//! Two producers can hand [`ktir_superdsc::ktir_node::KtirNode`] to the one KTIR→SuperDSC
//! lowering, and this crate is the second one's door:
//!
//! * scratchy's own builder (`lower_subtile_tape_to_ktir::KtirFunc`) — one node, one
//!   hand-written KTIR program, the card-proven status quo;
//! * the TRITON LADDER (`triton-frontend` → TTIR → `triton-ktir` → KTIR), re-hosted at
//!   `crates/triton/` — a `.py` kernel compiled at `#[forward]` expansion time, in-process,
//!   with no Python executing (the same discipline the torch carriers landed under #112).
//!
//! # WHAT THE TAPE STATES THAT A KERNEL CANNOT
//!
//! A Triton kernel is a function over its parameters. It has no words for "which graph
//! buffer is parameter 0" or "this is an rmsnorm". The tape states both:
//!
//! * **The operand binding.** The registry row declares the kernel's parameter order to be
//!   the node's operand order — parameter `i` is `node.inputs[i]`'s tensor, and the last
//!   parameter is the node's output. [`lower`] then re-states the adapter's positional
//!   `BufferId`s with the tape's real tensor indices, because `regions()` resolves a
//!   parameter's `Region.tid` — the operand name every descriptor addresses by, and the
//!   slot `scalarmul_scales` is looked up through — from `bindings[i]`. A splice that
//!   kept the positional ids would emit descriptors over buffers `t0/t1/t2` while the
//!   tape's buffers are `t37/t41/t42`: well-formed, and wrong.
//! * **The kind.** The row states the [`Program`] classification, the same value the
//!   builder path's `finish_shaped` states. The standalone triton-spyre position had to
//!   INFER this from the op soup; scratchy has the tape, so it STATES it, and a mis-stated
//!   row is caught by the byte-identity golden rather than by a classifier that could
//!   mis-recognize.
//!
//! # THE GATE, AND WHY IT IS AT THE DESCRIPTOR LEVEL
//!
//! Every registry row lands WITH its byte-identity golden: the descriptors the spliced path
//! emits must be byte-identical to the builder path's before the builder arm for that op is
//! deleted. The comparison is at the **EmittedOp/descriptor level, not the KTIR level** —
//! the two producers legitimately spell the program differently (the builder writes
//! `math.sqrt(mean + eps)` with an f32 island and a divisor; the kernel writes
//! `rsqrt((mean + eps).to(f32)).to(f16)` with a folded reciprocal), and the consumer
//! (`lower_ktir_to_superdsc`) assembles its descriptors from `regions()` + the program's
//! stated constants, not from the op soup. Descriptor identity is therefore the strongest
//! gate that is not also a false one.
//!
//! # ⛔ WHAT THIS CRATE DELIBERATELY DOES NOT SPLICE
//!
//! Attention and rope: their `EmittedOp`s carry consumer bake-plan facts (`kv_page_fold`,
//! `kv_request`, fold roles, const-generic geometry) that no Triton kernel states and no
//! registry row can carry. The registry simply has no row for them — a request to splice
//! one returns `Ok(None)` and the caller falls through to the builder arm, which keeps its
//! own refusals. They land when their facts sidecars land.
//!
//! fp8 `MatmulTile`: the builder threads a cross-node activation-quantize dedup
//! (`quantized`) that is a bundle-level fact, not a node-level one. Not a row.
//!
//! `RmsNorm { gain: OnePlusScale }`: refused by name, exactly as the builder arm refuses
//! it — the kernel does not exist for the (1 + w) convention and a silent fallback to the
//! Scale kernel is the quietly-wrong-model failure the builder's refusal documents.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ktir_core::arena::Arena;
use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::ktir_node::{BufferId, KtirNode, Program};
use scratchy_subtile::subtile_ir::{GainConvention, SubOp, SubtileIR, SubtileNode};

use triton_frontend::semantic::Val;
use triton_frontend::target::Target;
use triton_frontend::codegen::{ArgSpec, KernelSpec};

/// One registry row: the kernel's file and entry, and the STATED classification — declared
/// data, not logic. Adding a kernel is a row; nothing else in this crate changes.
pub struct TritonKernelRow {
    /// The kernel's file, under `crates/targets/spyre/kernels/`.
    pub kernel: &'static str,
    /// The `@triton.jit` function's name inside that file.
    pub entry: &'static str,
    /// The stated classification — what `lower_ktir_to_superdsc` dispatches on, and the
    /// same value the builder path's `finish_shaped` states for this op.
    pub program: Program,
}

/// THE REGISTRY — every `SubOp` kind with a Triton kernel, in migration order.
///
/// ⭐ ONE KERNEL PER OP KIND, AND THE OPERAND ORDER IS THE NODE'S. The builder's
/// `KtirFunc::rmsnorm(x, gamma, out)` and the kernel's `(x, gamma, out)` parameters must
/// agree parameter-for-parameter; the registry does not permute.
pub fn registry<F: scratchy_subtile::subtile_ir::RopeForm>(op: &SubOp<F>) -> Option<TritonKernelRow> {
    match op {
        // THE FIRST SPLICE. `Program::RmsNorm`'s consumer body already reads the epsilon
        // off EITHER producer spelling (`math.sqrt` chains — the builder's — or
        // `math.rsqrt`, the Triton fixture's), so the card-proven assembly is reached
        // unchanged.
        SubOp::RmsNorm {
            gain: GainConvention::Scale,
            ..
        } => Some(TritonKernelRow {
            kernel: "rmsnorm.py",
            entry: "rmsnorm_fwd",
            program: Program::RmsNorm,
        }),
        // THE SECOND SPLICE, and the pattern for every `Program::Elementwise` kind to
        // come: the consumer dispatches on the STATED kind (not the op soup), so the
        // spliced descriptors reach the same assembly the builder's program does.
        SubOp::SiluMul => Some(TritonKernelRow {
            kernel: "silumul.py",
            entry: "silumul_fwd",
            program: Program::SiluMul,
        }),
        // ⛔ NO ROW FOR attention/rope (consumer bake-plan facts), fp8 matmul (bundle-level
        // quantize dedup), or (1 + w) gains (no kernel exists). See the module header.
        _ => None,
    }
}

/// THE SPLICE. Compile the kernel the registry names for this node and hand back the same
/// [`EmittedOp`] the builder arm produces — `EmittedOp::bare(name)` with `ktir` set.
///
/// `Ok(None)` when the registry has no row (the caller falls through to the builder). A
/// row that FAILS to compile or lower is an `Err` — a loud expansion-time failure, never a
/// silent fallthrough, because a registry row that quietly stops working would make the
/// registry a lie.
///
/// The name is the builder's own law — `rmsnorm_s{node.id}` — so the spliced op's
/// `op_name` and the emulator's function key are IDENTICAL between the two paths. That is
/// the byte-identity golden's requirement.
pub fn lower<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
) -> Result<Option<EmittedOp>, String> {
    let Some(row) = registry(&node.op) else {
        return Ok(None);
    };
    // ⛔ THE ARITY IS THE NODE'S OWN CONTRACT, stated once per op kind so the splice and
    // the builder cannot disagree about it. The builder arm's own check is identical.
    let arity = match &node.op {
        SubOp::RmsNorm { .. } => 2,
        SubOp::SiluMul => 2,
        // ⛔ NO `_` ARM. A spliced kind is a row above, and a row without an arity here is
        // an unreachable — the same discipline `lower_one_node`'s match holds.
        _ => return Err(format!(
            "triton splice: {} has a registry row but no arity — the row is incomplete",
            row.kernel
        )),
    };
    if node.inputs.len() != arity {
        return Err(format!(
            "triton splice: {} t{} expects {} operand(s), found {}",
            row.kernel,
            node.output.tensor.index(),
            arity,
            node.inputs.len()
        ));
    }
    let src = read_kernel(row.kernel)?;
    let spec = kernel_spec(node, ir, &row)?;
    let m = compile_kernel(&src, &spec, &grid(node, ir)?)?;
    let k = mint(node, m, &row)?;
    let name = format!("{}_s{}", program_stem(&row), node.id.index());
    let mut e = EmittedOp::bare(name);
    e.ktir = Some(k);
    Ok(Some(e))
}

/// Read a kernel file from `crates/targets/spyre/kernels/`.
fn read_kernel(kernel: &str) -> Result<String, String> {
    let path = kernels_dir().join(kernel);
    std::fs::read_to_string(&path)
        .map_err(|e| format!("triton splice: cannot read {}: {e}", path.display()))
}

/// `crates/targets/spyre/kernels/` — resolved from THIS crate's manifest dir.
fn kernels_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../targets/spyre/kernels")
        .canonicalize()
        .unwrap_or_else(|_| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../targets/spyre/kernels")
        })
}

/// THE KERNEL'S LAUNCH CONTRACT, stated from the node's own shapes — the signature (one
/// descriptor per operand plus the output), the constexprs (the node's facts as
/// `tl.constexpr`s, the monomorphisation key), all in the case-table's own shape
/// (`cases::spec`).
fn kernel_spec<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    _ir: &SubtileIR<F>,
    row: &TritonKernelRow,
) -> Result<KernelSpec, String> {
    let out = &node.output;
    let (m, c) = (out.region.rows.len, out.region.cols.len);
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    match (&node.op, row.entry) {
        (SubOp::RmsNorm { eps, .. }, "rmsnorm_fwd") => {
            // The fixture's own parameter spellings: desc_x, desc_w, desc_o, then the
            // constexprs M / D_MODEL / BLOCK_M / EPS / INV_D. Every constexpr is BOTH a
            // signature entry (`ArgSpec::Constexpr`) and a binding, exactly as the case
            // table's `spec` helper states a configuration.
            for p in ["desc_x", "desc_w", "desc_o"] {
                signature.insert(p.to_string(), ArgSpec::parse("*fp16").map_err(|e| e.to_string())?);
            }
            let mut ce = |k: &str, v: Val| -> Result<(), String> {
                signature.insert(k.to_string(), ArgSpec::Constexpr);
                constexprs.insert(k.to_string(), v);
                Ok(())
            };
            ce("M", Val::Int(i128::from(m)))?;
            ce("D_MODEL", Val::Int(i128::from(c)))?;
            ce("BLOCK_M", Val::Int(i128::from(m)))?;
            // ⛔ THE EPSILON AND ITS RECIPROCAL ARE BIT-PINNED, AND THE DIRECTION
            // MATTERS. The consumer's eps lookup (`scale_slot`) matches the registry by
            // BITS, and the registry holds the tape's f32 eps. The Triton ladder's
            // `LegalizeTypes` collapses the f32 island around `rsqrt` and
            // `step_2b_island_constants` ROUNDS THE VALUE to f16 (`f.to_f16()`), so
            // `1e-5f32` becomes `0.00001001358` in the program — measured in
            // triton-ktir-superdsc's own layout docs. The consumer's lookup therefore
            // takes an f16-image fallback (builder-exact bits first, then the registry
            // entry whose f16 image equals the program's eps), which is physically exact:
            // the worker binds ONE fp16 per registry slot (`SCALE_BYTES = 2`), so the
            // value the descriptor reads is the f16 image either way.
            ce("EPS", Val::Float(f64::from(*eps)))?;
            // `1.0 / D_MODEL` folded on the host, exactly as the fixture's own
            // `constexprs()` helper does — one multiply instead of a divide, and NOT a
            // registry scale (the consumer binds `1/cols` at the reserved
            // `RMS_INVCOLS_TID`, never through `scalarmul_scales`).
            ce("INV_D", Val::Float(1.0 / f64::from(c)))?;
        }
        (SubOp::SiluMul, "silumul_fwd") => {
            // The kernel's own parameter spellings: desc_g, desc_u, desc_o, then the
            // constexprs M / N / BLOCK_M / BLOCK_N.
            for p in ["desc_g", "desc_u", "desc_o"] {
                signature.insert(p.to_string(), ArgSpec::parse("*fp16").map_err(|e| e.to_string())?);
            }
            let mut ce = |k: &str, v: Val| -> Result<(), String> {
                signature.insert(k.to_string(), ArgSpec::Constexpr);
                constexprs.insert(k.to_string(), v);
                Ok(())
            };
            // The whole region, one tile: `BLOCK_M = M` rows and `BLOCK_N = N` columns,
            // the same no-row-blocking law `KtirFunc::silu_mul` states for itself.
            ce("M", Val::Int(i128::from(m)))?;
            ce("N", Val::Int(i128::from(c)))?;
            ce("BLOCK_M", Val::Int(i128::from(m)))?;
            ce("BLOCK_N", Val::Int(i128::from(c)))?;
        }
        (op, entry) => {
            return Err(format!(
                "triton splice: no kernel signature for {op:?} at entry `{entry}` — the row is \
                 incomplete"
            ))
        }
    }
    Ok(KernelSpec {
        kernel: row.entry.to_string(),
        signature,
        constexprs,
        file: kernels_dir()
            .join(row.kernel)
            .to_string_lossy()
            .into_owned(),
    })
}

/// The launch grid. The builder's programs are `(gx, gy) = (1, 1)` for every pointwise
/// kind; the kernel is compiled at the same grid the case table states (`vec![1]`).
fn grid<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    _ir: &SubtileIR<F>,
) -> Result<Vec<i64>, String> {
    match &node.op {
        SubOp::RmsNorm { .. } => Ok(vec![1]),
        SubOp::SiluMul => Ok(vec![1]),
        _ => Err("triton splice: no grid for this op kind — the row is incomplete".to_string()),
    }
}

/// THE FULL LADDER for one kernel: parse → TTIR → `make_ttir` → `from_ttir` → `make_ktir`
/// → `to_ktir` — the exact sequence `bake_py.rs` drives, stated once.
fn compile_kernel(
    src: &str,
    spec: &KernelSpec,
    grid: &[i64],
) -> Result<triton_ktir::ir::Module, String> {
    let mut tt = triton_frontend::codegen::compile(src, spec, Target::spyre())
        .map_err(|e| format!("triton splice: {}: {e}", spec.file))?;
    triton_frontend::opt::make_ttir(&mut tt)
        .map_err(|e| format!("triton splice: make_ttir: {e}"))?;
    let mut m = triton_ktir::from_ttir::convert(&tt)
        .map_err(|e| format!("triton splice: from_ttir: {e}"))?;
    triton_ktir::make_ktir(&mut m, grid)
        .map_err(|e| format!("triton splice: make_ktir: {e}"))?;
    triton_ktir::passes::to_ktir::run(&mut m, grid)
        .map_err(|e| format!("triton splice: to_ktir: {e}"))?;
    Ok(m)
}

/// Mint the [`KtirNode`]: the adapter's `node_for` compilation, then RE-STATE the fields
/// the tape owns — the name (the builder's naming law), the bindings (the tape's tensor
/// indices in the node's operand order), `mask` (none for rmsnorm), and `node_out_tid`
/// (None — the program writes its own node's output).
///
/// ⛔ THE RE-STATE IS THE WHOLE POINT. `node_for` mints positional `BufferId`s
/// (`0, 1, 2`); the tape's rmsnorm node reads `t37`/`t41` and writes `t42`. The
/// bindings are ours to state, exactly as `KtirFunc::finish_shaped` states them, because
/// only the splice knows which graph tensor each kernel parameter addresses.
fn mint<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    module: triton_ktir::ir::Module,
    row: &TritonKernelRow,
) -> Result<KtirNode, String> {
    // The adapter's mint: `to_ktir_emit::lower` over the module, positional buffer ids.
    let positional = triton_ktir_superdsc::node_for(&module, row.program)
        .map_err(|e| format!("triton splice: node_for: {e}"))?;
    // ⛔ BINDINGS ARE THE TAPE'S TENSOR INDICES, in the node's operand order, with the
    // output LAST — the exact law `KtirFunc::finish_shaped` states. `regions()` reads
    // `bindings[i]` for `arguments[i]`, so the kernel's parameter order must be the
    // node's operand order (the registry row's contract, checked at the signature above).
    let mut bindings: Vec<BufferId> = node
        .inputs
        .iter()
        .map(|tr| BufferId::new(tr.tensor.index() as u32))
        .collect();
    bindings.push(BufferId::new(node.output.tensor.index() as u32));
    // ⭐ THE NAME IS THE BUILDER'S LAW, so the op_name and the emulator's function key are
    // identical between the two paths — the byte-identity golden's requirement.
    let name = Arena::global().str(format!(
        "{}_s{}",
        program_stem(row),
        node.id.index()
    ));
    let KtirNode { func, program, .. } = positional;
    if func.arguments.len() != bindings.len() {
        return Err(format!(
            "triton splice: {} compiled to {} parameters but the node states {} bindings — \
             the row's operand-order contract is violated",
            func.name,
            func.arguments.len(),
            bindings.len()
        ));
    }
    Ok(KtirNode {
        func: ktir_core::ir::IRFunction { name, ..func },
        program,
        bindings,
        mask: None,
        node_out_tid: None,
    })
}

/// The program stem the builder's naming law uses for this row (`rmsnorm_s{id}`).
fn program_stem(row: &TritonKernelRow) -> &'static str {
    match row.program {
        Program::RmsNorm => "rmsnorm",
        Program::SiluMul => "silumul",
        _ => "triton",
    }
}
