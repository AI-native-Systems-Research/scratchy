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
//! Attention: its `EmittedOp`s carry consumer bake-plan facts (`kv_page_fold`,
//! `kv_request`, fold roles, const-generic geometry) that no Triton kernel states and no
//! registry row can carry. The registry simply has no row for it — a request to splice
//! one returns `Ok(None)` and the caller falls through to the builder arm, which keeps its
//! own refusals. It lands when its facts sidecar lands.
//!
//! ⭐ ROPE, BY CONTRAST, SPLICES — and the module-header claim that it could not was
//! OVERSTATED, audited against the door: `rope_at` derives every fact it needs (`mq`,
//! `total`, `hd`) from the PROGRAM's own views and access tiles, and the one bundle fact
//! (`rows_are_requests`) is re-read by the door off `BundleAttnParams` AFTER the splice
//! returns. The kernel states the builder's own view extents (`[mq·heads, hd]`) and takes
//! one `[heads, half]` access tile per position, which is what the door's first-tile read
//! needs. See the rope row in [`registry`].
//!
//! fp8 `MatmulTile` — a row after all: the module header's old claim that the
//! activation-quantize dedup (`quantized`) is "a bundle-level fact, not a
//! node-level one" described the PRE-door splice design and was OVERSTATED for
//! the current architecture. The dedup set is threaded BUNDLE-WIDE by the door's
//! callers (`lower_graph_to_ktir`'s walk creates one per bundle and hands it to
//! `ktir_superdsc_door::lower`), which runs DOWNSTREAM of the splice: both
//! producers' `EmittedOp`s flow through the SAME door call, and
//! `matmul_fp8_descriptors` dedups there (`quantized.insert(a_name)` — q/k/v
//! share one quant, gate/up share another). The splice only has to mint a
//! `Program::Matmul` `KtirNode` whose weight view is fp8 and whose bindings are
//! arity-3; fp8-ness is recognized from the weight view's `is_fp8`, cross-checked
//! against arity, exactly as the builder's program is. See the fp8 row in
//! [`registry`].
//!
//! `RmsNorm { gain: OnePlusScale }`: refused by name, exactly as the builder arm refuses
//! it — the kernel does not exist for the (1 + w) convention and a silent fallback to the
//! Scale kernel is the quietly-wrong-model failure the builder's refusal documents.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ktir_core::arena::Arena;
use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::ktir_node::{BufferId, Elementwise, KtirNode, Program};
use scratchy_subtile::subtile_ir::{EwKind, GainConvention, SubOp, SubtileIR, SubtileNode};

use triton_frontend::codegen::{ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

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
pub fn registry<F: scratchy_subtile::subtile_ir::RopeForm>(
    op: &SubOp<F>,
) -> Option<TritonKernelRow> {
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
        // THE THIRD SPLICE — the biggest family by op count (every projection in every
        // layer). DENSE fp16 (arity-2, `matmul.py`) and fp8 W8A8 (arity-3,
        // `matmul_fp8.py`) are separate rows: `GemmWeight` on the node states which is
        // which, and the fp8 kernel's spelled `* w_scale` epilogue is what the ladder's
        // `verify_canonical_fp8_matmul_kernel` requires — see the module header for why
        // the door-side activation-quantize dedup is not this row's concern.
        SubOp::MatmulTile {
            weight: scratchy_subtile::lower::GemmWeight::Dense,
            ..
        } => Some(TritonKernelRow {
            kernel: "matmul.py",
            entry: "matmul_fwd",
            program: Program::Matmul,
        }),
        // THE SIXTH SPLICE — fp8 W8A8, the delivery target (granite 8b fp8). Same
        // `Program::Matmul` classification as dense: the DOOR discriminates fp8 from
        // the weight view's `is_fp8` + arity-3 bindings, never from the program kind,
        // so both rows reach the same `matmul_proven` door arm.
        SubOp::MatmulTile {
            weight: scratchy_subtile::lower::GemmWeight::Fp8Dynamic,
            ..
        } => Some(TritonKernelRow {
            kernel: "matmul_fp8.py",
            entry: "matmul_fp8_fwd",
            program: Program::Matmul,
        }),
        // THE FOURTH SPLICE — the split elementwise kinds, one entry per `EwKind` the
        // builder's own arm lowers (`lower_elementwise_node`): Add AND BiasAdd share
        // `add_fwd` (the builder maps both to `Elementwise::Add` and the same
        // `add_s{id}` name), Mul, Sub, and the standalone Silu. The kinds the builder
        // REFUSES (Gelu/QuickGelu/GeluErf) have no row: the splice never widens the
        // lowering's reach beyond the builder arm it replaces — that is the
        // byte-identity golden's precondition.
        SubOp::Elementwise(EwKind::Add | EwKind::BiasAdd) => Some(TritonKernelRow {
            kernel: "elementwise.py",
            entry: "add_fwd",
            program: Program::Elementwise(Elementwise::Add),
        }),
        SubOp::Elementwise(EwKind::Mul) => Some(TritonKernelRow {
            kernel: "elementwise.py",
            entry: "mul_fwd",
            program: Program::Elementwise(Elementwise::Mul),
        }),
        SubOp::Elementwise(EwKind::Sub) => Some(TritonKernelRow {
            kernel: "elementwise.py",
            entry: "sub_fwd",
            program: Program::Elementwise(Elementwise::Sub),
        }),
        SubOp::Elementwise(EwKind::Silu) => Some(TritonKernelRow {
            kernel: "elementwise.py",
            entry: "silu_fwd",
            program: Program::Elementwise(Elementwise::Silu),
        }),
        // THE FIFTH SPLICE — rope, the first op whose consumer (`rope_at`) runs through
        // the const-generic head-dim door. The node's own `head_dim` states the kernel's
        // HEAD_DIM/HALF; `rows_are_requests` already arrives as `lower`'s parameter and
        // the DOOR re-reads it off `BundleAttnParams` after the splice, so the only
        // facts this row needs are the node's. `RopeAppend` carries 6 inputs but only
        // the first three are the rotation (V and the KV-cache destinations flow through
        // GRAPH edges — the builder's own `lower_rope_node` reads `inputs[0..3]`), so
        // the kernel consumes x/cos/sin/out and the row covers BOTH `RopeAppend` and
        // the standalone `RopeRotate`.
        SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. } => Some(TritonKernelRow {
            kernel: "rope.py",
            entry: "rope_fwd",
            program: Program::Rope,
        }),
        // ⛔ NO ROW FOR attention (consumer bake-plan facts), Affine-int4 weights (no
        // kernel exists), or (1 + w) gains (no kernel exists). See the module header.
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
    // See `lower_one_node`: whether this bundle's rows are separate requests. Only the
    // prefill lm-head-tail fallthrough reads it — the builder's own fold guard, mirrored.
    rows_are_requests: bool,
) -> Result<Option<EmittedOp>, String> {
    let Some(row) = registry(&node.op) else {
        return Ok(None);
    };
    // ⛔ THE LM-HEAD IS NOT A SPLICE TARGET, for two separate reasons, both detected by
    // the builder's own vocab-width test (the lm_head matmul and the logits ScalarMul are
    // the ONLY ops whose output spans the result cols — every intermediate is hidden or
    // intermediate width):
    //
    // 1. THE PREFILL FOLD. At m>1 the builder REWRITES the node (last-row extraction at
    //    m=1 over a synthetic buffer), which no registry row can state — fall through so
    //    the builder takes its own `is_prefill_lm_head_tail` arm (`!rows_are_requests`).
    // 2. THE ODD VOCAB. A decode lm_head at vocab 49155 (granite) has N odd, and the
    //    ladder's `PlanCorelets` partitions a matmul across exactly 2 corelets —
    //    `recover_matmul_n` RED-stops an N that N/2 cannot tile, a FAITHFUL port of the
    //    C++ oracle (fixing it here would make the port disagree with the field). So an
    //    odd-N result-width matmul stays on the builder path until the ladder learns a
    //    single-corelet matmul plan. ⛔ AT ANY ROW COUNT: the batched-decode lm_head
    //    (rows > 1, `rows_are_requests`) carries the SAME odd N, and the first cut of
    //    this guard tested `rows == 1` only — so a batched-decode lm_head slipped past
    //    BOTH lm-head fallthroughs into `compile_kernel`, whose `PlanCorelets` refusal
    //    turned this guard's `Ok(None)` design into a loud `Err` bake failure on granite
    //    batched decode. Parity does not depend on the row count; the guard does not
    //    either.
    if matches!(node.op, SubOp::MatmulTile { .. }) {
        let result_cols = ir.tensors[ir.result.index()].cols;
        if node.output.region.cols.len == result_cols {
            let rows = node.output.region.rows.len;
            if rows > 1 && !rows_are_requests {
                return Ok(None); // 1. the prefill fold
            }
            if node.output.region.cols.len % 2 == 1 {
                return Ok(None); // 2. the odd vocab, at any row count
            }
            // An EVEN result-width matmul at m>1 with `rows_are_requests` (the
            // batched-decode lm_head) is spliced below like any other matmul.
        }
    }
    // ⛔ AN ELEMENTWISE NODE THE BUILDER WOULD ROW-BLOCK IS NOT A SPLICE TARGET. The
    // builder's own arm (`lower_elementwise_node`) blocks a whole-region lowering whose
    // live set — `rows × cols × live_tiles`, the same `EW_LX_ELEMS = 1M`-element budget
    // — does not fit a core's 2 MB LX, emitting MULTIPLE row blocks inside ONE program.
    // A one-tile kernel cannot spell that shape, so the splice mirrors the builder's
    // own guard and falls through: the two paths never disagree about which of them
    // takes the node. (`m == 1` is never blocked, so every decode node splices.)
    if let SubOp::Elementwise(kind) = &node.op {
        let (rows, cols) = (node.output.region.rows.len, node.output.region.cols.len);
        let live: u32 = match kind {
            EwKind::Silu => 6,
            _ => 3,
        };
        if u64::from(rows) * u64::from(cols) * u64::from(live) > 1024 * 1024 {
            return Ok(None);
        }
    }
    // ⛔ AND SILU-MUL IS THE SAME LAW AT EIGHT LIVE TILES. The builder's
    // `KtirFunc::silu_mul` row-blocks a whole `[mq, intermediate]` region that does not
    // fit (gate, up, neg, exp, the splat, denom, silu, y — EIGHT tiles, the widest live
    // set in the model; granite 8b's `[31, 12800]` prefill silu-mul is exactly the
    // region that overflows). MEASURED: without this guard the spliced one-tile program
    // declares a whole-region view the emulator's allocation bounds-check refuses
    // (`view [31, 12800] ... spans 793600 bytes but the tensor ... holds only 507904`),
    // while the builder's blocked program runs. The splice mirrors the builder's own
    // budget and falls through.
    if matches!(node.op, SubOp::SiluMul) {
        let (rows, cols) = (node.output.region.rows.len, node.output.region.cols.len);
        if u64::from(rows) * u64::from(cols) * 8 > 1024 * 1024 {
            return Ok(None);
        }
    }
    // ⛔⛔⛔ A COLUMN-CHUNKED NODE IS NOT A SPLICE TARGET — the whole pointwise family
    // (silumul and elementwise alike). The front end tiles a wide op into COLUMN CHUNKS
    // of `nb` (subtile_ir.rs's `n_blocks(out_cols, nb)`; production `nb = 8192`), and the
    // builder's program states each chunk's ACCESS-TILE CORNER (`load_region` honors
    // `region.cols.start`), which the door turns into the operand's column offset
    // (`pointwise_chunk_out_offset` → the 16384 B stick-group step at column 8192). The
    // kernels here state ONE whole-tensor tile at corner 0 — they cannot name a window.
    // MEASURED, granite-3.1-8b fp8 on card: the 12800-wide MLP intermediate is TWO chunks
    // (0..8192, 8192..12800), and without this guard the second chunk's silu/mulsilu read
    // and wrote the FIRST chunk's columns (the decode bundle's two differing descriptors
    // were exactly the second chunk's gate binding, 16384 B low) — fluent garbage out, on
    // a divergence the whole-region golden could not see because every fixture is
    // whole-region. 2b passed only because its intermediate is 8192 = exactly one block.
    // A windowed kernel is a follow-on row; until then a node whose regions are not the
    // whole tensors falls through to the builder, which states the corner itself.
    if matches!(node.op, SubOp::SiluMul | SubOp::Elementwise(_)) {
        let whole = |tr: &scratchy_subtile::subtile_ir::TensorRegion, ir: &SubtileIR<F>| {
            let s = &ir.tensors[tr.tensor.index()];
            tr.region.rows.start == 0
                && tr.region.rows.len == s.rows
                && tr.region.cols.start == 0
                && tr.region.cols.len == s.cols
        };
        if !whole(&node.output, ir) || node.inputs.iter().any(|tr| !whole(tr, ir)) {
            return Ok(None);
        }
    }

    // ⛔ THE ARITY IS THE NODE'S OWN CONTRACT, stated once per op kind so the splice and
    // the builder cannot disagree about it. The builder arm's own check is identical.
    // ⭐ ROPE'S ARITY IS 3, NOT THE NODE'S INPUT COUNT: `lower_rope_node` reads only
    // `inputs[0..3]` (x, cos, sin) — a `RopeAppend` carries 6 inputs but its V and
    // KV-cache destinations flow through GRAPH edges, not through the op — so the
    // splice binds the same first three operands the builder's program does, and the
    // check below is `>= 3` exactly as the builder's own `inputs.len() < 3` refusal is.
    let arity = match &node.op {
        SubOp::RmsNorm { .. } => 2,
        SubOp::SiluMul => 2,
        // ⭐ THE MATMUL'S ARITY IS THE WEIGHT SCHEME'S OWN: Dense is arity-2
        // `[act, weight]`, Fp8Dynamic is arity-3 `[act, weight_fp8, weight_scale]`
        // (the same routing `lower_matmul_node` does on `node.inputs.get(2)`). The
        // Affine scheme stays arity-2 but has NO row — it cannot reach this match,
        // and the registry's `None` returns before it.
        SubOp::MatmulTile { weight, .. } => match weight {
            scratchy_subtile::lower::GemmWeight::Dense => 2,
            scratchy_subtile::lower::GemmWeight::Fp8Dynamic => 3,
            scratchy_subtile::lower::GemmWeight::Affine { .. } => 2,
        },
        SubOp::Elementwise(EwKind::Silu) => 1,
        SubOp::Elementwise(_) => 2,
        SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. } => 3,
        // ⛔ NO `_` ARM. A spliced kind is a row above, and a row without an arity here is
        // an unreachable — the same discipline `lower_one_node`'s match holds.
        _ => {
            return Err(format!(
                "triton splice: {} has a registry row but no arity — the row is incomplete",
                row.kernel
            ));
        }
    };
    if node.inputs.len() < arity {
        return Err(format!(
            "triton splice: {} t{} expects at least {} operand(s), found {}",
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
    let name = format!("{}_s{}", program_stem(node, &row), node.id.index());
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
                signature.insert(
                    p.to_string(),
                    ArgSpec::parse("*fp16").map_err(|e| e.to_string())?,
                );
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
        (
            SubOp::MatmulTile {
                n,
                weight: scratchy_subtile::lower::GemmWeight::Dense,
            },
            "matmul_fwd",
        ) => {
            // A is [M, K] (m from the node's output rows, k from A's own columns); W is
            // the FUF convention's [K, N] region, n from the Linear's own stated width —
            // the tile keeps its Linear's `n` even when col-tiling split the output.
            let k = node.inputs[0].region.cols.len;
            for p in ["desc_a", "desc_w", "desc_o"] {
                signature.insert(
                    p.to_string(),
                    ArgSpec::parse("*fp16").map_err(|e| e.to_string())?,
                );
            }
            let mut ce = |k: &str, v: Val| -> Result<(), String> {
                signature.insert(k.to_string(), ArgSpec::Constexpr);
                constexprs.insert(k.to_string(), v);
                Ok(())
            };
            // ONE tile, the whole region — `KtirFunc::matmul`'s own whole-region law
            // (one linalg.matmul, no K loop).
            ce("M", Val::Int(i128::from(m)))?;
            ce("K", Val::Int(i128::from(k)))?;
            ce("N", Val::Int(i128::from(*n)))?;
            ce("BLOCK_M", Val::Int(i128::from(m)))?;
            ce("BLOCK_K", Val::Int(i128::from(k)))?;
            ce("BLOCK_N", Val::Int(i128::from(*n)))?;
        }
        (
            SubOp::MatmulTile {
                n,
                weight: scratchy_subtile::lower::GemmWeight::Fp8Dynamic,
            },
            "matmul_fp8_fwd",
        ) => {
            // The arity-3 twin of the dense arm: A [M, K], W fp8-packed [N, K] (the
            // checkpoint's own on-disk layout, 1 byte per element — the descriptor's
            // elem says fp8, the load widens on read), ws the [1, N] per-channel scale
            // row. K from A's own columns, n from the Linear's stated width — the same
            // derivations the dense arm states, and the ones `KtirFunc::matmul_fp8`
            // states for the builder path.
            let k = node.inputs[0].region.cols.len;
            signature.insert(
                "desc_x".to_string(),
                ArgSpec::parse("*fp16").map_err(|e| e.to_string())?,
            );
            signature.insert(
                "desc_w".to_string(),
                ArgSpec::parse("*fp8e4nv").map_err(|e| e.to_string())?,
            );
            signature.insert(
                "desc_ws".to_string(),
                ArgSpec::parse("*fp16").map_err(|e| e.to_string())?,
            );
            signature.insert(
                "desc_o".to_string(),
                ArgSpec::parse("*fp16").map_err(|e| e.to_string())?,
            );
            let mut ce = |k: &str, v: Val| -> Result<(), String> {
                signature.insert(k.to_string(), ArgSpec::Constexpr);
                constexprs.insert(k.to_string(), v);
                Ok(())
            };
            // ONE tile, the whole region — and the fp8 contract REFUSES anything else
            // (`verify_canonical_fp8_matmul_kernel`: `BLOCK_K < K` and `BLOCK_N < N`
            // are refused by name; a K-looped fp8 form is a follow-on, not this row).
            ce("M", Val::Int(i128::from(m)))?;
            ce("K", Val::Int(i128::from(k)))?;
            ce("N", Val::Int(i128::from(*n)))?;
            ce("BLOCK_M", Val::Int(i128::from(m)))?;
            ce("BLOCK_K", Val::Int(i128::from(k)))?;
            ce("BLOCK_N", Val::Int(i128::from(*n)))?;
        }
        (SubOp::SiluMul, "silumul_fwd") => {
            // The kernel's own parameter spellings: desc_g, desc_u, desc_o, then the
            // constexprs M / N / BLOCK_M / BLOCK_N.
            for p in ["desc_g", "desc_u", "desc_o"] {
                signature.insert(
                    p.to_string(),
                    ArgSpec::parse("*fp16").map_err(|e| e.to_string())?,
                );
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
        (SubOp::Elementwise(EwKind::Silu), "silu_fwd") => {
            for p in ["desc_x", "desc_o"] {
                signature.insert(
                    p.to_string(),
                    ArgSpec::parse("*fp16").map_err(|e| e.to_string())?,
                );
            }
            let mut ce = |k: &str, v: Val| -> Result<(), String> {
                signature.insert(k.to_string(), ArgSpec::Constexpr);
                constexprs.insert(k.to_string(), v);
                Ok(())
            };
            ce("M", Val::Int(i128::from(m)))?;
            ce("N", Val::Int(i128::from(c)))?;
            ce("BLOCK_M", Val::Int(i128::from(m)))?;
            ce("BLOCK_N", Val::Int(i128::from(c)))?;
        }
        (SubOp::Elementwise(_), "add_fwd" | "mul_fwd" | "sub_fwd") => {
            // The kernel's own parameter spellings: desc_a, desc_b, desc_o for every
            // binary entry, then the constexprs M / N / BLOCK_M / BLOCK_N — the same
            // whole-region single-tile law `lower_elementwise_node` states when the
            // region fits (the splice refused the node otherwise, above).
            for p in ["desc_a", "desc_b", "desc_o"] {
                signature.insert(
                    p.to_string(),
                    ArgSpec::parse("*fp16").map_err(|e| e.to_string())?,
                );
            }
            let mut ce = |k: &str, v: Val| -> Result<(), String> {
                signature.insert(k.to_string(), ArgSpec::Constexpr);
                constexprs.insert(k.to_string(), v);
                Ok(())
            };
            ce("M", Val::Int(i128::from(m)))?;
            ce("N", Val::Int(i128::from(c)))?;
            ce("BLOCK_M", Val::Int(i128::from(m)))?;
            ce("BLOCK_N", Val::Int(i128::from(c)))?;
        }
        (SubOp::RopeRotate { head_dim, .. } | SubOp::RopeAppend { head_dim, .. }, "rope_fwd") => {
            // The door's contract, stated from the node's own facts: total = the output's
            // declared width (`heads * hd`), heads = total / hd, mq = the output's rows —
            // the same derivation `KtirFunc::rope`'s views state. The builder's own
            // refusal (`total` not a whole number of `hd`-wide heads) is mirrored here
            // as an `Err`, not a fallthrough: the node is malformed, and the builder arm
            // would refuse it identically.
            let hd = head_dim.get();
            let total = c;
            if hd == 0 || total % hd != 0 {
                return Err(format!(
                    "triton splice: rope t{}: {total} cols is not a whole number of \
                     {hd}-wide heads",
                    node.output.tensor.index()
                ));
            }
            let heads = total / hd;
            for p in ["desc_x", "desc_cos", "desc_sin", "desc_o"] {
                signature.insert(
                    p.to_string(),
                    ArgSpec::parse("*fp16").map_err(|e| e.to_string())?,
                );
            }
            let mut ce = |k: &str, v: Val| -> Result<(), String> {
                signature.insert(k.to_string(), ArgSpec::Constexpr);
                constexprs.insert(k.to_string(), v);
                Ok(())
            };
            ce("H", Val::Int(i128::from(heads)))?;
            ce("MQ", Val::Int(i128::from(m)))?;
            ce("HEAD_DIM", Val::Int(i128::from(hd)))?;
            ce("HALF", Val::Int(i128::from(hd / 2)))?;
        }
        (op, entry) => {
            return Err(format!(
                "triton splice: no kernel signature for {op:?} at entry `{entry}` — the row is \
                 incomplete"
            ));
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
        SubOp::MatmulTile { .. } => Ok(vec![1]),
        SubOp::Elementwise(_) => Ok(vec![1]),
        // ONE WORK ITEM: the position loop is a constant-trip `tl.range` inside the
        // kernel, unrolled by the ladder (`to_ktir::unroll_constant_trip_loops`), so the
        // spliced program is straight-line like the builder's — no grid axis at all.
        SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. } => Ok(vec![1]),
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
    triton_ktir::make_ktir(&mut m, grid).map_err(|e| format!("triton splice: make_ktir: {e}"))?;
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
    // ⭐ ROPE BINDS THREE, not the node's whole input list: the kernel consumes x, cos,
    // sin (the rotation), while a `RopeAppend`'s V and KV-cache inputs flow through
    // GRAPH edges — `lower_rope_node` binds exactly these three plus the output, and so
    // does the splice. The arity match above pinned `inputs.len() >= 3`.
    let rope = matches!(node.op, SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. });
    let n_bound = if rope { 3 } else { node.inputs.len() };
    let mut bindings: Vec<BufferId> = node
        .inputs
        .iter()
        .take(n_bound)
        .map(|tr| BufferId::new(tr.tensor.index() as u32))
        .collect();
    bindings.push(BufferId::new(node.output.tensor.index() as u32));
    // ⭐ THE NAME IS THE BUILDER'S LAW, so the op_name and the emulator's function key are
    // identical between the two paths — the byte-identity golden's requirement.
    let name = Arena::global().str(format!("{}_s{}", program_stem(node, row), node.id.index()));
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

/// The program stem the builder's naming law uses for this node (`rmsnorm_s{id}`,
/// `add_s{id}`, …). ⛔ READ OFF THE NODE, NOT THE ROW: the elementwise rows carry the
/// CONSUMER's kind (`Elementwise::Add`), and the builder's name comes from the
/// PRODUCER's `EwKind` (`ew_kind_stem` — BiasAdd is named `add`, not `biasadd`). One
/// row may therefore mint several stems; the node states which.
fn program_stem<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    row: &TritonKernelRow,
) -> &'static str {
    if let SubOp::Elementwise(kind) = &node.op {
        return match kind {
            EwKind::Add | EwKind::BiasAdd => "add",
            EwKind::Mul => "mul",
            EwKind::Sub => "sub",
            EwKind::Silu => "silu",
            // ⛔ NO `_` ARM. A spliced elementwise kind has a row above; reaching here
            // with a kind that has none is an unreachable — the same discipline the
            // arity match holds.
            other => unreachable!("elementwise kind {other:?} has no program stem"),
        };
    }
    match row.program {
        Program::RmsNorm => "rmsnorm",
        Program::SiluMul => "silumul",
        Program::Matmul => "matmul",
        Program::Rope => "rope",
        _ => "triton",
    }
}
