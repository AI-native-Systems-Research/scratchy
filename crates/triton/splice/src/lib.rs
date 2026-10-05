// SPDX-License-Identifier: Apache-2.0
//! THE SPLICE — where the tape's node and the kernel meet.
//!
//! Two producers can hand [`ktir_superdsc::ktir_node::KtirNode`] to the one KTIR→SuperDSC
//! lowering, and this crate is the second one's door:
//!
//! * the TRITON LADDER (`triton-frontend` → TTIR → `triton-ktir` → KTIR), re-hosted at
//!   `crates/triton/` — a `.py` kernel compiled at `#[forward]` expansion time, in-process,
//!   with no Python executing (the same discipline the torch carriers landed under #112).
//!   This crate is the ONLY producer for the ops it covers: `lower_one_node` routes
//!   every spliced kind here unconditionally, and there is no builder arm to fall
//!   through to for them.
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
//! # ⛔⭐ TOTALITY BY CONSTRUCTION — NO LOOKUP, NO FALLTHROUGH, NO REFUSAL
//!
//! **We are a proc-macro compiler and everything is known at compile time.** The registry
//! is not a lookup table an op kind can miss at runtime: it is a set of TOTAL per-family
//! functions ([`rmsnorm_row`], [`matmul_row`], [`elementwise_row`], plus the silumul,
//! scalarmul and rope rows in [`row`]), each an EXHAUSTIVE match over its family's
//! vocabulary with NO `_` arm and no `Option`. A family member without a kernel is an
//! E0004 non-exhaustive-match error at compile time IN THIS CRATE — the kernel lands
//! with the match arm or the tree is red, and no build can ever reach a runtime
//! "no kernel for {:?}".
//!
//! The members whose device realization does not exist yet are likewise DECLARED, never
//! discovered at runtime:
//!
//! * **`GemmWeight::Affine`** — the wavefront lowering skips affine-quantized presets
//!   before a tape is ever lowered (`codegen`'s `no superdsc bundle` gate), so no Affine
//!   node reaches this crate on any build today. [`matmul_row`] still enumerates it:
//!   wiring one in is a compile error here until the kernel lands with the arm.
//! * **`EwKind::QuickGelu` / `EwKind::GeluErf`** — the DDL has NO primitive for either
//!   (the consumer's `elementwise_op_func` refuses them by name, and substituting
//!   `"gelu"` would run a different function and report success). [`elementwise_row`]
//!   enumerates them so the same law holds.
//! * **`GainConvention::OnePlusScale`** — the door's rmsnorm body assembles `xn·gamma`
//!   from the LOADED gain tensor and has no statement for a `+1` offset; the arm is the
//!   compile-time enumeration, never a silently-wrong Scale splice.
//!
//! # THE GATE, AND WHY IT IS AT THE DESCRIPTOR LEVEL
//!
//! Every registry row lands WITH its byte-identity golden: the descriptors the spliced
//! path emits must be byte-identical to the builder path's (`tests/triton_splice_golden.rs`
//! keeps the builder bodies as the control for exactly this). The comparison is at the
//! **EmittedOp/descriptor level, not the KTIR level** — the two producers legitimately
//! spell the program differently (the builder writes `math.sqrt(mean + eps)` with an f32
//! island and a divisor; the kernel writes `rsqrt((mean + eps).to(f32)).to(f16)` with a
//! folded reciprocal), and the consumer (`lower_ktir_to_superdsc`) assembles its
//! descriptors from `regions()` + the program's stated constants, not from the op soup.
//! Descriptor identity is therefore the strongest gate that is not also a false one.
//!
//! # ⛔ THE SHAPE WORK LIST, AS NAMED ERRORS
//!
//! The kernel families here state ONE whole-tensor tile at corner 0. A node whose shape
//! needs more than that is refused LOUDLY, naming the kernel capability that must land:
//!
//! * **windowed regions** (the front end's column chunking, production `nb = 8192`) — a
//!   kernel that states the chunk's access-tile corner;
//! * **LX row-blocking** (a whole region whose live set exceeds `EW_LX_ELEMS`) — a kernel
//!   that states multiple row blocks;
//! * **odd-N result-width matmuls** (granite's 49155 vocab) — a ladder PlanCorelets shape;
//! * **the prefill lm-head fold** (a vocab-wide matmul at m>1, `!rows_are_requests`).
//!
//! Each dies when its kernel lands; none is a fallthrough to a second producer, because
//! there is no second producer for a spliced kind.
//!
//! ⭐ ROPE SPLICES — the module-header claim that it could not was OVERSTATED, audited
//! against the door: `rope_at` derives every fact it needs (`mq`, `total`, `hd`) from the
//! PROGRAM's own views and access tiles, and the one bundle fact (`rows_are_requests`)
//! is re-read by the door off `BundleAttnParams` AFTER the splice returns. The kernel
//! states the builder's own view extents (`[mq·heads, hd]`) and takes one `[heads, half]`
//! access tile per position, which is what the door's first-tile read needs.
//!
//! fp8 `MatmulTile` — a row after all: the activation-quantize dedup (`quantized`) is
//! threaded BUNDLE-WIDE by the door's callers, which run DOWNSTREAM of the splice: both
//! producers' `EmittedOp`s flow through the SAME door call, and `matmul_fp8_descriptors`
//! dedups there. The splice only has to mint a `Program::Matmul` `KtirNode` whose weight
//! view is fp8 and whose bindings are arity-3; fp8-ness is recognized from the weight
//! view's `is_fp8`, cross-checked against arity, exactly as the builder's program is.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ktir_core::arena::Arena;
use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::ktir_node::{BufferId, Elementwise, KtirNode, Program};
use scratchy_subtile::lower::GemmWeight;
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

// ── THE TOTAL REGISTRY, FAMILY BY FAMILY ─────────────────────────────────────────
//
// ⛔⭐ Each function is an EXHAUSTIVE match over one family's vocabulary with no `_`
// arm and no `Option` — a family member without a kernel is E0004 at compile time
// here, which is the whole design: we are a procmacro compiler, everything is known
// at compile time, and no runtime lookup can miss.

/// `SubOp::RmsNorm`'s two gain conventions. The door's rmsnorm body assembles `xn·gamma`
/// from the loaded gain; the `(1 + w)` class (gemma) needs the door to state the offset,
/// which it does not yet — the arm exists so wiring one in is a compile error until the
/// kernel + door support land together.
pub fn rmsnorm_row(gain: GainConvention) -> TritonKernelRow {
    match gain {
        GainConvention::Scale => TritonKernelRow {
            kernel: "rmsnorm.py",
            entry: "rmsnorm_fwd",
            program: Program::RmsNorm,
        },
        // ⛔ NO DEVICE STATEMENT: the door multiplies by the LOADED gain with no offset
        // term, and the wavefront lowering CARRIES the convention rather than folding
        // it into the weights (folding would make the loaded gain disagree with the
        // checkpoint). Splicing the Scale kernel here would scale every gemma
        // activation by roughly nothing — a model that loads, runs, and is quietly
        // wrong. The arm is the compile-time enumeration; the realization lands here
        // when the door states the offset (metal's own `ScalarOffsetRmsNorm`, whose
        // kernel applies `weight + offset`, is the precedent shape).
        GainConvention::OnePlusScale => TritonKernelRow {
            kernel: "rmsnorm.py",
            entry: "rmsnorm_one_plus_scale_fwd",
            program: Program::RmsNorm,
        },
    }
}

/// `SubOp::MatmulTile`'s weight schemes. Affine-int4 is skipped preset-side by the
/// wavefront lowering (no superdsc bundle is emitted for an affine preset), so no
/// Affine node reaches [`lower`] on any build today; the arm exists so wiring one in
/// is a compile error here until the kernel lands.
pub fn matmul_row(weight: &GemmWeight) -> TritonKernelRow {
    match weight {
        GemmWeight::Dense => TritonKernelRow {
            kernel: "matmul.py",
            entry: "matmul_fwd",
            program: Program::Matmul,
        },
        // THE fp8 W8A8 ROW — the delivery target (granite 8b fp8). Same
        // `Program::Matmul` classification as dense: the DOOR discriminates fp8 from
        // the weight view's `is_fp8` + arity-3 bindings, never from the program kind,
        // so both rows reach the same `matmul` door arm. The kernel's spelled
        // `* w_scale` epilogue is what the ladder's
        // `verify_canonical_fp8_matmul_kernel` requires.
        GemmWeight::Fp8Dynamic => TritonKernelRow {
            kernel: "matmul_fp8.py",
            entry: "matmul_fp8_fwd",
            program: Program::Matmul,
        },
        // ⛔ NO KERNEL: the wavefront lowering skips affine presets before a tape is
        // lowered (`codegen`'s `no superdsc bundle` gate) and no consumer realizes an
        // affine contraction on this path, so nothing constructs this node on the spyre
        // path. The arm is the compile-time enumeration — a realization lands here
        // WITH its kernel (metal's qmv family is the precedent shape).
        GemmWeight::Affine { .. } => TritonKernelRow {
            kernel: "matmul_affine.py",
            entry: "matmul_affine_fwd",
            program: Program::Matmul,
        },
    }
}

/// `SubOp::Elementwise`'s kinds. The kinds the DDL has no primitive for are enumerated
/// with the same names the consumer's own refusal uses, so wiring one in fails at E0004
/// until a producer-side decomposition lands.
pub fn elementwise_row(kind: EwKind) -> TritonKernelRow {
    match kind {
        // Add AND BiasAdd share `add_fwd` — both map to `Elementwise::Add` and the same
        // `add_s{id}` name (the builder's own law).
        EwKind::Add | EwKind::BiasAdd => TritonKernelRow {
            kernel: "elementwise.py",
            entry: "add_fwd",
            program: Program::Elementwise(Elementwise::Add),
        },
        EwKind::Mul => TritonKernelRow {
            kernel: "elementwise.py",
            entry: "mul_fwd",
            program: Program::Elementwise(Elementwise::Mul),
        },
        EwKind::Sub => TritonKernelRow {
            kernel: "elementwise.py",
            entry: "sub_fwd",
            program: Program::Elementwise(Elementwise::Sub),
        },
        EwKind::Silu => TritonKernelRow {
            kernel: "elementwise.py",
            entry: "silu_fwd",
            program: Program::Elementwise(Elementwise::Silu),
        },
        // A REAL DDL primitive (`OpFunc::Gelu`): the SFP constant table ships gelu's
        // tanh polynomial, so this is one pointwise op. The kernel spells the tanh
        // form through the exp island (the frontend has no `tanh`), which is the same
        // approximation the `"gelu"` primitive itself makes — see `elementwise.py`'s
        // `gelu_fwd`.
        EwKind::Gelu => TritonKernelRow {
            kernel: "elementwise.py",
            entry: "gelu_fwd",
            program: Program::Elementwise(Elementwise::Gelu),
        },
        // ⛔ NO DDL PRIMITIVE, and substituting the nearest one is the bug: quick-gelu
        // is x·σ(1.702x), a DIFFERENT function from OpFunc::Gelu's tanh polynomial —
        // emitting "gelu" for it would run the wrong model and report success (the
        // consumer's own `elementwise_op_func` refusal names exactly this). A
        // realization must DECOMPOSE producer-side (its building blocks — sigmoid,
        // mul — ARE primitives), which is a kernel family of its own.
        EwKind::QuickGelu => TritonKernelRow {
            kernel: "elementwise.py",
            entry: "quickgelu_fwd",
            program: Program::Elementwise(Elementwise::QuickGelu),
        },
        // ⛔ SAME LAW: exact-erf gelu is the erf form, not the tanh polynomial, and the
        // consumer has no `erf` op either, so a decomposition cannot lower today.
        EwKind::GeluErf => TritonKernelRow {
            kernel: "elementwise.py",
            entry: "gelu_erf_fwd",
            program: Program::Elementwise(Elementwise::GeluErf),
        },
    }
}

/// The prefill lm-head fold's extraction row. The fold is TWO ops from ONE
/// `MatmulTile` node (see [`lower_all`]), and the extraction half is its own kernel
/// family — the row is declared here so the registry's own law holds ("adding a
/// kernel is a row; nothing else in this crate changes").
pub fn lmlast_row() -> TritonKernelRow {
    TritonKernelRow {
        kernel: "lmlast.py",
        entry: "lmlast_fwd",
        program: Program::LmLast,
    }
}

/// THE ROW FOR A NODE — the family functions composed. Every `SubOp` that can reach the
/// splice is one of the five families below; the ops the spyre target has no kernel AT
/// ALL for (attention, the expansion ops, reshape, …) never reach this crate —
/// `lower_one_node`'s own arms own those refusals by name.
pub fn row<F: scratchy_subtile::subtile_ir::RopeForm>(op: &SubOp<F>) -> TritonKernelRow {
    match op {
        SubOp::RmsNorm { gain, .. } => rmsnorm_row(*gain),
        SubOp::MatmulTile { weight, .. } => matmul_row(weight),
        SubOp::Elementwise(kind) => elementwise_row(*kind),
        SubOp::SiluMul => TritonKernelRow {
            kernel: "silumul.py",
            entry: "silumul_fwd",
            program: Program::SiluMul,
        },
        SubOp::ScalarMul { .. } => TritonKernelRow {
            kernel: "scalarmul.py",
            entry: "scalarmul_fwd",
            program: Program::ScalarMul,
        },
        // `RopeAppend` carries 6 inputs but only the first three are the rotation (V
        // and the KV-cache destinations flow through GRAPH edges — the builder's own
        // `lower_rope_node` reads `inputs[0..3]`), so the kernel consumes x/cos/sin/out
        // and the row covers BOTH `RopeAppend` and the standalone `RopeRotate`.
        SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. } => TritonKernelRow {
            kernel: "rope.py",
            entry: "rope_fwd",
            program: Program::Rope,
        },
        // ⛔⭐ THE OPS `lower_one_node` NEVER ROUTES HERE — enumerated by NAME, never
        // `_`, so adding a SubOp is an E0004 in this crate too. `lower_one_node`'s own
        // match is the gate: these kinds have their own arms there (attention's
        // const-generic geometry door, the host-routed rmsnorm pair, the by-name
        // refusals), and a kind reaching THIS match means the routing changed — the
        // message names what must land, and no `_` arm can swallow it.
        SubOp::AttnDecode { .. }
        | SubOp::RmsNormReduce { .. }
        | SubOp::RmsNormApply { .. }
        | SubOp::Reshape { .. }
        | SubOp::SumReduce { .. }
        | SubOp::TanhSoftCap
        | SubOp::RmsNormUnit { .. }
        | SubOp::ScalarWeightMul
        | SubOp::GateSplit { .. }
        | SubOp::GateApply
        | SubOp::GateScale
        | SubOp::LoadPixels { .. }
        | SubOp::LoadPosEmbeds { .. }
        | SubOp::EmbeddingGather { .. }
        | SubOp::VisionRope
        | SubOp::VarlenAttention { .. }
        | SubOp::EncoderAttn { .. }
        | SubOp::GatedDeltaNet
        | SubOp::Mean
        | scratchy_subtile::expansion_ops!() => TritonKernelRow {
            kernel: "ROUTED-ELSEWHERE",
            entry: "ROUTED-ELSEWHERE",
            program: Program::LmLast,
        },
    }
}

/// THE SPLICE — the ONLY producer for a spliced kind. Compile the kernel the registry
/// names for this node and hand back the [`EmittedOp`]: `EmittedOp::bare(name)` with
/// `ktir` set.
///
/// There is no `Ok(None)` and no builder to fall through to: every spliced kind either
/// emits or the bake stops, loudly, naming the kernel capability that must land. The
/// shape guards below are the work list — each one dies when its kernel states that
/// shape, and the end state has none.
///
/// The name is the builder's own law — `rmsnorm_s{node.id}` — so the spliced op's
/// `op_name` and the emulator's function key are IDENTICAL to the builder path's. That
/// is the byte-identity golden's requirement.
///
/// ⭐ ONE NODE MAY LOWER TO MORE THAN ONE OP, and only the splice knows which kinds:
/// [`lower_all`] is the entry the walk calls, and it routes every one-op kind here.
pub fn lower<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    // Whether this bundle's rows are separate requests (the lm-head fold's own fact).
    rows_are_requests: bool,
) -> Result<EmittedOp, String> {
    lower_one(node, ir, rows_are_requests)
}

/// EVERY OP THE SPLICE EMITS FOR ONE NODE. Every kind is one op except the prefill
/// lm-head tail, whose fold is TWO — the last-row extraction (its own program, the
/// reserved `LAST_HIDDEN_TID` staging) plus the re-lowered m=1 matmul — exactly the
/// shape of main's `lower_prefill_lm_head_at_m1`, which the fold kernel must
/// reproduce.
pub fn lower_all<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    rows_are_requests: bool,
) -> Result<Vec<EmittedOp>, String> {
    if let SubOp::MatmulTile { .. } = &node.op {
        let result_cols = ir.tensors[ir.result.index()].cols;
        if node.output.region.cols.len == result_cols && !rows_are_requests {
            let rows = node.output.region.rows.len;
            if rows > 1 {
                return lower_prefill_lm_head_fold(node, ir);
            }
        }
    }
    lower_one(node, ir, rows_are_requests).map(|e| vec![e])
}

fn lower_one<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    rows_are_requests: bool,
) -> Result<EmittedOp, String> {
    let row = row(&node.op);    // ⛔ THE ROUTING IS `lower_one_node`'S EXHAUSTIVE MATCH, and this guard is its echo:
    // the kinds that never route here carry the ROUTED-ELSEWHERE sentinel row, and a
    // node that reaches this check means the routing changed without adding a family
    // function — the message names the owner. It is unreachable through
    // `lower_one_node` by construction (its match is exhaustive over the same enum),
    // and this is a public fn, so the echo exists for the direct caller.
    if row.kernel == "ROUTED-ELSEWHERE" {
        return Err(format!(
            "triton splice: {:?} is not a spliced kind — `lower_one_node` owns its lowering \
             (attention's geometry door, the host-routed pair, or its own by-name refusal). \
             Routing it here means adding a family function and a kernel",
            node.op
        ));
    }
    // ⛔ THE LM-HEAD TAIL'S FOLD IS `lower_all`'s now — the vocab-wide m>1 matmul is
    // rewritten THERE (last-row extraction + the re-lowered m=1 matmul), so a
    // MatmulTile reaching THIS one-op body with m>1 over the result cols means the
    // one-op `lower` was called where the walk's `lower_all` belongs. The vocab-width
    // test is the builder's own (the lm_head matmul and the logits ScalarMul are the
    // ONLY ops whose output spans the result cols — every intermediate is hidden or
    // intermediate width).
    //
    // ⭐ THE ODD VOCAB SPLICES. A decode lm_head at vocab 49155 (granite) has N odd,
    //    and the ladder's `PlanCorelets` now re-patterns an odd N to `single_corelet`
    //    (the same re-patterning the C++ itself made for `split` at one stick) — the
    //    builder path has emitted exactly this matmul for this repo's whole life, so
    //    the device runs it, and the two-corelet plan was a LADDER artifact, not a
    //    device fact.
    if matches!(node.op, SubOp::MatmulTile { .. }) {
        let result_cols = ir.tensors[ir.result.index()].cols;
        if node.output.region.cols.len == result_cols {
            let rows = node.output.region.rows.len;
            if rows > 1 && !rows_are_requests {
                return Err(format!(
                    "triton splice: matmul_s{} is the prefill lm-head tail (vocab-wide, m={rows}, \
                     rows not requests) — the fold is `lower_all`'s; call it, not the one-op \
                     `lower`",
                    node.id.index()
                ));
            }
        }
    }
    // ⛔⛔⛔ A NODE WHOSE REGIONS ARE NOT WHOLE TENSORS IS A SHAPE ONLY A KERNEL THAT
    // STATES ITS CORNER CAN SPELL. The measured instance was the front end's COLUMN
    // CHUNKING of a wide pointwise op (`n_blocks(out_cols, nb)`; production `nb =
    // 8192`): the builder's program states each chunk's access-tile corner
    // (`load_region` honors `region.cols.start`), which the door turns into the
    // operand's column offset. MEASURED, granite-3.1-8b fp8 on card: the 12800-wide
    // MLP intermediate is TWO chunks (0..8192, 8192..12800), and a kernel stating
    // corner 0 read and wrote the FIRST chunk's columns — fluent garbage out, on a
    // divergence the whole-region golden could not see because every fixture is
    // whole-region. 2b passed only because its intermediate is 8192 = exactly one
    // block.
    //
    // ⭐ THE POINTWISE FAMILY NOW SPELLS IT: `elementwise.py` / `silumul.py` /
    // `scalarmul.py` state `N_TOTAL` (the tensor's storage width, named by the
    // descriptor's shape/strides) and `C_START` (the region's column corner, named
    // by the load/store offsets) — the same facts the builder's `load_region` /
    // `store_region` state, so the door reads the SAME region off the spliced
    // program. AND THE MATMUL FAMILY SPELLS ITS OWN ROW-0 WINDOW: `matmul.py` /
    // `matmul_fp8.py` state `M_TOTAL` (the output tensor's row extent) with the
    // `[M, N]` tile at row 0 — the prefill lm-head fold's m=1 tail, whose activation
    // is the LAST_HIDDEN synthetic and whose output is row 0 of the `[mq, vocab]`
    // logits storage. Every OTHER row states ONE whole-tensor tile at corner 0 and
    // cannot name a window, so the refusal stands for it (rope regions are whole in
    // production today), and this guard reads the REGION, not the op kind, so a
    // front-end change that windows any other op's regions reproduces the same
    // refusal with the kernel row named — a windowed kernel is the fix shape.
    {
        let states_the_corner = matches!(
            &node.op,
            SubOp::SiluMul
                | SubOp::ScalarMul { .. }
                | SubOp::MatmulTile { .. }
                | SubOp::Elementwise(
                    EwKind::Silu | EwKind::Gelu | EwKind::Add | EwKind::Mul | EwKind::Sub,
                )
        );
        if !states_the_corner {
            let whole = |tr: &scratchy_subtile::subtile_ir::TensorRegion, ir: &SubtileIR<F>| {
                let s = &ir.tensors[tr.tensor.index()];
                tr.region.rows.start == 0
                    && tr.region.rows.len == s.rows
                    && tr.region.cols.start == 0
                    && tr.region.cols.len == s.cols
            };
            if !whole(&node.output, ir) || node.inputs.iter().any(|tr| !whole(tr, ir)) {
                return Err(format!(
                    "triton splice: {} t{} has a windowed region (not the whole tensor) — the \
                     one-tile kernels state corner 0 only; the windowed-kernel family (access-tile \
                     corners stated from the region) has not landed",
                    program_stem(node, &row),
                    node.output.tensor.index()
                ));
            }
        } else {
            // ⛔ THE COLUMN CORNER IS STATED; THE ROW CORNER IS NOT. The pointwise
            // kernels load at `start_m * BLOCK_M` with a `[1]` grid — row 0 — so a
            // region whose ROW corner is nonzero is still a shape this family cannot
            // spell (the prefill lm-head fold's row extraction is exactly that, and
            // it is the fold worklist item). Production chunking never moves the row
            // corner (`lower_region` tiles columns only), so this is the loud edge.
            let row0 = |tr: &scratchy_subtile::subtile_ir::TensorRegion| {
                tr.region.rows.start == 0
            };
            if !row0(&node.output) || node.inputs.iter().any(|tr| !row0(tr)) {
                return Err(format!(
                    "triton splice: {} t{} has a nonzero ROW corner — the pointwise kernels \
                     load row 0 (`start_m * BLOCK_M` at a [1] grid); a row-windowed kernel is the \
                     prefill-fold worklist item",
                    program_stem(node, &row),
                    node.output.tensor.index()
                ));
            }
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
        // Affine scheme stays arity-2 (its scales/biases resolve from the SAME weight
        // source under tensor roles, not as separate IR operands).
        SubOp::MatmulTile { weight, .. } => match weight {
            GemmWeight::Dense => 2,
            GemmWeight::Fp8Dynamic => 3,
            GemmWeight::Affine { .. } => 2,
        },
        SubOp::Elementwise(EwKind::Silu | EwKind::Gelu) => 1,
        SubOp::Elementwise(_) => 2,
        SubOp::ScalarMul { .. } => 1,
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
    Ok(e)
}

/// ⭐ THE PREFILL LM-HEAD FOLD — main's `lower_prefill_lm_head_at_m1` shape, with the
/// Triton `lmlast.py` kernel as the extraction's body. The vocab-wide lm_head cannot
/// run at m>1 (it time-tiles; per-row time-tiling is unimplemented), and only the LAST
/// prompt token's logits are ever read, so the tail is TWO ops:
///
/// 1. THE EXTRACTION — `lmlast_fwd` copies row `selector_lastrow_col(mq) = mq - 1` of
///    the `[mq, hidden]` activation into the reserved `[1, hidden]` `LAST_HIDDEN_TID`
///    staging, one single-stick tile per stick-group (the only representable form: the
///    source buffer is stick-major, so a wider window over the row is not expressible).
///    Its KtirNode carries `node_out_tid` = the tail's OWN output tid — the door's
///    `lmlast` arm names its copies `lmlast{j}_o{tid}` after it, the same suffix the
///    m=1 matmul's `matmul_o{tid}` uses, so the whole tail reads as one node's.
/// 2. THE MATMUL — the SAME node at m=1, reading the staging at row 0, through the
///    ordinary matmul row (`lower_one`). At m=1 the tail is exactly the shape the
///    PROVEN decode path lowers, fp8 chain included.
///
/// ⛔ THE ROW IS `selector_lastrow_col(mq)`, THE SSOT THE KANI PROOF PINS — not
/// `rows.start + mq - 1`: the extraction addresses the buffer the tape takes to be
/// exactly `[mq, hidden]` at row 0, which is the only shape that reaches the fold (a
/// non-whole activation region is refused by the whole-region guard before this).
fn lower_prefill_lm_head_fold<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
) -> Result<Vec<EmittedOp>, String> {
    use scratchy_subtile::subtile_ir::Region as SubRegion;

    let a = &node.inputs[0];
    let (mq, hidden) = (a.region.rows.len, a.region.cols.len);
    let stk = 64u32; // Fp16::ELEMS_PER_STICK — the stick the door's lmlast arm checks
    if hidden % stk != 0 {
        return Err(format!(
            "triton splice: matmul_s{} (prefill lm-head tail): hidden={hidden} is not a whole \
             {stk}-fp16 stick, so the last prompt row is not a run of whole stick-groups — the \
             per-stick extraction cannot address it",
            node.id.index()
        ));
    }
    let row = scratchy_subtile::sdsc_abstract::selector_lastrow_col(mq as usize) as u32;
    let last_hidden = ktir_superdsc::reserved_tids::LAST_HIDDEN_TID;

    // ── Half 1: the extraction, through the lmlast kernel. ──
    let src = read_kernel("lmlast.py")?;
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    for p in ["desc_src", "desc_dst"] {
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
    ce("MQ", Val::Int(i128::from(mq)))?;
    ce("HIDDEN", Val::Int(i128::from(hidden)))?;
    ce("ROW", Val::Int(i128::from(row)))?;
    ce("N_STICKS", Val::Int(i128::from(hidden / stk)))?;
    let spec = KernelSpec {
        file: "lmlast.py".to_string(),
        kernel: "lmlast_fwd".to_string(),
        signature: signature.clone(),
        constexprs: constexprs.clone(),
    };
    let module = compile_kernel(&src, &spec, &[1])?;
    let extraction_node = SubtileNode {
        id: node.id,
        op: node.op,
        inputs: vec![*a],
        output: scratchy_subtile::subtile_ir::TensorRegion {
            tensor: scratchy_subtile::subtile_ir::TensorId::from_index(
                last_hidden as usize,
            ),
            region: SubRegion {
                rows: scratchy_subtile::subtile_ir::Range::new(0, 1),
                cols: scratchy_subtile::subtile_ir::Range::new(0, hidden),
            },
        },
    };
    let mut k = mint(&extraction_node, module, &lmlast_row())?;
    // The KtirNode's own fields: `node_out_tid` names the TAIL'S output (the door
    // names its copies `lmlast{j}_o{tid}` after it), and the name is the builder's
    // `lmlast_s{id}` law.
    k.node_out_tid = Some(BufferId::new(node.output.tensor.index() as u32));
    let xname = format!("lmlast_s{}", node.id.index());
    k.func.name = Arena::global().str(xname.clone());
    let mut extract = EmittedOp::bare(xname);
    extract.ktir = Some(k);

    // ── Half 2: the SAME node at m=1, reading the staging at row 0. ──
    let mut at_m1 = node.clone();
    let one_row = |tr: &scratchy_subtile::subtile_ir::TensorRegion| {
        scratchy_subtile::subtile_ir::TensorRegion {
            tensor: tr.tensor,
            region: SubRegion {
                rows: scratchy_subtile::subtile_ir::Range::new(tr.region.rows.start, 1),
                cols: tr.region.cols,
            },
        }
    };
    at_m1.inputs[0] = scratchy_subtile::subtile_ir::TensorRegion {
        tensor: scratchy_subtile::subtile_ir::TensorId::from_index(last_hidden as usize),
        region: SubRegion {
            rows: scratchy_subtile::subtile_ir::Range::new(0, 1),
            cols: scratchy_subtile::subtile_ir::Range::new(0, hidden),
        },
    };
    if let Some(w) = at_m1.inputs.get_mut(1) {
        *w = one_row(w);
    }
    at_m1.output = one_row(&at_m1.output);
    let matmul = lower_one(&at_m1, ir, false)?;

    Ok(vec![extract, matmul])
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
    ir: &SubtileIR<F>,
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
                weight: GemmWeight::Dense,
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
            // (one linalg.matmul, no K loop). `M_TOTAL` names the OUTPUT TENSOR's row
            // extent (the storage the descriptor addresses) while the store takes the
            // `[M, N]` tile at row 0 — the prefill lm-head fold's m=1 tail, whose
            // output is row 0 of the `[mq, vocab]` logits storage.
            let (m_total, _a_total) = matmul_window_of(node, ir)?;
            ce("M", Val::Int(i128::from(m)))?;
            ce("K", Val::Int(i128::from(k)))?;
            ce("N", Val::Int(i128::from(*n)))?;
            ce("BLOCK_M", Val::Int(i128::from(m)))?;
            ce("BLOCK_K", Val::Int(i128::from(k)))?;
            ce("BLOCK_N", Val::Int(i128::from(*n)))?;
            ce("M_TOTAL", Val::Int(i128::from(m_total)))?;
        }
        (
            SubOp::MatmulTile {
                n,
                weight: GemmWeight::Fp8Dynamic,
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
            // `M_TOTAL` is the dense arm's window law (see there).
            let (m_total, _) = matmul_window_of(node, ir)?;
            ce("M", Val::Int(i128::from(m)))?;
            ce("K", Val::Int(i128::from(k)))?;
            ce("N", Val::Int(i128::from(*n)))?;
            ce("BLOCK_M", Val::Int(i128::from(m)))?;
            ce("BLOCK_K", Val::Int(i128::from(k)))?;
            ce("BLOCK_N", Val::Int(i128::from(*n)))?;
            ce("M_TOTAL", Val::Int(i128::from(m_total)))?;
        }
        (SubOp::SiluMul, "silumul_fwd") => {
            // The kernel's own parameter spellings: desc_g, desc_u, desc_o, then the
            // constexprs M / N / BLOCK_M / BLOCK_N / N_TOTAL / C_START.
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
            // the same no-row-blocking law `KtirFunc::silu_mul` states for itself —
            // PLUS the STORAGE the region windows: the descriptor names the TENSOR
            // (`[M, N_TOTAL]`, strides `[N_TOTAL, 1]` — `KtirFunc::view`'s own shape
            // read) and the load/store names the CORNER (`C_START`), exactly as the
            // builder's `load_region`/`store_region` state it. A whole-region node
            // states `N_TOTAL = N`, `C_START = 0`.
            let (n_total, c_start) = window_of(node, ir)?;
            let (block_m, n_blocks, tail_h) =
                blocks_of(m, c, ktir_superdsc::superdsc_opspec::SILU_MUL_LIVE_TILES);
            ce("M", Val::Int(i128::from(m)))?;
            ce("N", Val::Int(i128::from(c)))?;
            ce("BLOCK_M", Val::Int(i128::from(block_m)))?;
            ce("BLOCK_N", Val::Int(i128::from(c)))?;
            ce("N_TOTAL", Val::Int(i128::from(n_total)))?;
            ce("C_START", Val::Int(i128::from(c_start)))?;
            ce("N_BLOCKS", Val::Int(i128::from(n_blocks)))?;
            ce("TAIL_H", Val::Int(i128::from(tail_h)))?;
        }
        (SubOp::Elementwise(EwKind::Silu | EwKind::Gelu), "silu_fwd" | "gelu_fwd") => {
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
            let (n_total, c_start) = window_of(node, ir)?;
            let (block_m, n_blocks, tail_h) =
                blocks_of(m, c, ktir_superdsc::superdsc_opspec::EW_SILU_LIVE_TILES);
            ce("M", Val::Int(i128::from(m)))?;
            ce("N", Val::Int(i128::from(c)))?;
            ce("BLOCK_M", Val::Int(i128::from(block_m)))?;
            ce("BLOCK_N", Val::Int(i128::from(c)))?;
            ce("N_TOTAL", Val::Int(i128::from(n_total)))?;
            ce("C_START", Val::Int(i128::from(c_start)))?;
            ce("N_BLOCKS", Val::Int(i128::from(n_blocks)))?;
            ce("TAIL_H", Val::Int(i128::from(tail_h)))?;
        }
        (SubOp::Elementwise(_), "add_fwd" | "mul_fwd" | "sub_fwd") => {
            // The kernel's own parameter spellings: desc_a, desc_b, desc_o for every
            // binary entry, then the constexprs M / N / BLOCK_M / BLOCK_N / N_TOTAL /
            // C_START — the same whole-region single-tile law `lower_elementwise_node`
            // states when the region fits, PLUS the storage-window facts (see
            // `window_of`).
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
            let (n_total, c_start) = window_of(node, ir)?;
            let (block_m, n_blocks, tail_h) =
                blocks_of(m, c, ktir_superdsc::superdsc_opspec::EW_BINARY_LIVE_TILES);
            ce("M", Val::Int(i128::from(m)))?;
            ce("N", Val::Int(i128::from(c)))?;
            ce("BLOCK_M", Val::Int(i128::from(block_m)))?;
            ce("BLOCK_N", Val::Int(i128::from(c)))?;
            ce("N_TOTAL", Val::Int(i128::from(n_total)))?;
            ce("C_START", Val::Int(i128::from(c_start)))?;
            ce("N_BLOCKS", Val::Int(i128::from(n_blocks)))?;
            ce("TAIL_H", Val::Int(i128::from(tail_h)))?;
        }
        (SubOp::ScalarMul { scale }, "scalarmul_fwd") => {
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
            let (n_total, c_start) = window_of(node, ir)?;
            // ⛔ THE WINDOW IS THE DEVICE WIDTH, NOT THE LOGICAL ONE — the door's own
            // law for this family. `scalarmul_scaled` pads the window through
            // `DeviceWidth::for_pointwise` (a ScalarMul on the padded logits must use
            // the width its producer matmul emitted), and the Triton front end refuses
            // a block whose last dim is under 16 bytes (`semantic.py:1863`), so
            // granite's 3-wide logits tail chunk is not spellable at its logical
            // width. Stating the DEVICE width clears that floor AND states the window
            // the descriptor actually computes. `for_pointwise` is IDEMPOTENT (every
            // branch of `bump_sticks_to_splittable` reproduces its input: a
            // full-occupancy pad is 32-divisible and stays; an 8-stick pad is
            // core-split ≥8 and stays), so the door re-derives the SAME width from
            // this program as from the builder's logical one — including through the
            // `pointwise_width_the_output_holds` cap, which sees identical `cols` on
            // both paths. The other pointwise families (elementwise, silumul) state
            // the LOGICAL width because their door arms do — `check_pointwise_cols`
            // refuses a non-stick width rather than padding it.
            let c_dev = ktir_superdsc::work::DeviceWidth::for_pointwise(c).get();
            let (block_m, n_blocks, tail_h) = blocks_of(m, c_dev, 3);
            ce("M", Val::Int(i128::from(m)))?;
            ce("N", Val::Int(i128::from(c_dev)))?;
            ce("BLOCK_M", Val::Int(i128::from(block_m)))?;
            ce("BLOCK_N", Val::Int(i128::from(c_dev)))?;
            ce("N_TOTAL", Val::Int(i128::from(n_total)))?;
            ce("C_START", Val::Int(i128::from(c_start)))?;
            ce("N_BLOCKS", Val::Int(i128::from(n_blocks)))?;
            ce("TAIL_H", Val::Int(i128::from(tail_h)))?;
            // THE SCALE, from the node's own payload. The consumer reads it OFF THE
            // PROGRAM (`program_scalarmul_scale`: one splat feeding every `arith.mulf`)
            // and looks the value up in `BundleLayout::scalarmul_scales`, the registry
            // the tape-side layout pass fills FROM THIS SAME PAYLOAD — so stating it
            // here as the kernel's constexpr keeps the program and the registry slot in
            // agreement by construction.
            ce("SCALE", Val::Float(f64::from(*scale)))?;
        }
        (SubOp::RopeRotate { head_dim, .. } | SubOp::RopeAppend { head_dim, .. }, "rope_fwd") => {
            // The door's contract, stated from the node's own facts: total = the output's
            // declared width (`heads * hd`), heads = total / hd, mq = the output's rows —
            // the same derivation `KtirFunc::rope`'s views state. The builder's own
            // refusal (`total` not a whole number of `hd`-wide heads) is mirrored here
            // as an `Err`: the node is malformed, and no kernel can state it.
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

/// The STORAGE-WINDOW FACTS a pointwise kernel states: the tensor's full column
/// extent (`N_TOTAL` — the descriptor's own shape/strides name the STORAGE, exactly
/// as `KtirFunc::view` reads it from the graph) and the region's column corner
/// (`C_START` — the load/store offsets name the WINDOW, exactly as the builder's
/// `load_region`/`store_region` state it from `region.cols.start`).
///
/// ⛔ READ OFF THE OUTPUT'S TENSOR. The front end's chunking gives every operand of a
/// chunked node the SAME column window (`lower_region` tiles all of a node's regions
/// by the output's block), so any operand would answer the same — but the OUTPUT is
/// the tensor the descriptor's store addresses, and a node whose operands DISAGREE
/// about the window is malformed in a way no kernel can spell: it is refused by name
/// here rather than silently strided wrong. A whole-region node answers
/// `(cols, 0)`, the constants the one-tile form always implied.
fn window_of<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
) -> Result<(u32, u32), String> {
    let out = &node.output;
    let n_total = ir.tensors[out.tensor.index()].cols;
    let (r, c_start) = (out.region.rows.start, out.region.cols.start);
    // The operands must window the SAME slice of the SAME storage — the builder's
    // own law (`load_region` per operand, one chunk per node).
    for tr in &node.inputs {
        let in_total = ir.tensors[tr.tensor.index()].cols;
        if tr.region.rows.start != r || tr.region.cols.start != c_start || in_total != n_total {
            return Err(format!(
                "triton splice: t{}'s operands disagree about the window (output [{r}, \
                 {c_start}] of a {n_total}-wide tensor, input [{}, {}] of a {in_total}-wide one) \
                 — a chunked node windows ALL its regions by the output's block \
                 (`lower_region`), so a disagreement is a malformed node no kernel can spell",
                out.tensor.index(),
                tr.region.rows.start,
                tr.region.cols.start,
            ));
        }
    }
    Ok((n_total, c_start))
}

/// THE MATMUL'S WINDOW FACTS — the row-0 window form the prefill lm-head fold's m=1
/// tail states: the out descriptor names the OUTPUT TENSOR's row extent (`M_TOTAL`,
/// the storage the layout reserved) while the store takes the `[M, N]` tile at row 0,
/// and the activation names its own `[M, K]` window. A whole-region node answers
/// `(out_rows, a_rows)` with `M_TOTAL = M`.
///
/// ⛔ THE ROW CORNER MUST BE 0 — the door's `base_addressed` refuses a nonzero row
/// corner on both the activation and the output, and no matmul kernel in this family
/// spells one. The fold's m=1 tail is row 0 of the `[mq, vocab]` logits storage, so
/// it passes; any other row-windowed matmul is a shape to spell with a kernel, not to
/// silently mis-address.
fn matmul_window_of<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
) -> Result<(u32, u32), String> {
    let out = &node.output;
    let a = &node.inputs[0];
    let out_total = ir.tensors[out.tensor.index()].rows;
    let a_total = ir.tensors.get(a.tensor.index()).map(|s| s.rows);
    if out.region.rows.start != 0 || a.region.rows.start != 0 {
        return Err(format!(
            "triton splice: matmul t{} has a nonzero ROW corner (output row {}, activation \
             row {}) — the door's `base_addressed` refuses a nonzero row corner on a matmul \
             operand, and the kernels state row 0; the prefill lm-head fold is the row-0 \
             window form",
            out.tensor.index(),
            out.region.rows.start,
            a.region.rows.start,
        ));
    }
    if let Some(a_rows) = a_total
        && a.region.rows.len != a_rows
    {
        return Err(format!(
            "triton splice: matmul t{}: the activation region is [1, {}] of a [{}, {}] tensor — \
             only the lm-head fold's LAST_HIDDEN synthetic (beyond the graph) windows a matmul's \
             activation; a graph activation must be read whole",
            out.tensor.index(),
            a.region.cols.len,
            a_rows,
            ir.tensors[a.tensor.index()].cols,
        ));
    }
    Ok((out_total, a_total.unwrap_or(a.region.rows.len)))
}

/// THE ROW-BLOCK CONSTANTS the pointwise kernels take — the builder's own blocking
/// law, stated from the node's own region: `BLOCK_M` is the SHARED `rows_per_block`
/// (`EW_LX_ELEMS / live / cols` — one function both paths read, so the builder's
/// programs and the spliced ones cannot disagree about a block height and emit
/// windows that overlap or leave a gap), `N_BLOCKS` full blocks follow, and `TAIL_H`
/// is the shorter last tile when the region does not divide evenly (the builder's
/// `h = blk.min(rows - off)`).
///
/// A region that FITS the budget answers `(rows, 1, 0)` — ONE whole-region tile, the
/// constants the one-tile form always implied, so nothing changes for decode (m=1)
/// or any prefill rung inside the LX.
///
/// ⛔ THE `live` COUNT IS THE FAMILY'S OWN — `EW_SILU_LIVE_TILES` /
/// `EW_BINARY_LIVE_TILES` / `SILU_MUL_LIVE_TILES` / 3 for scalarmul — read by the
/// CALLER, because it is the same fact the builder's own `by_row` condition reads and
/// a wrong count here would block at a different height than the builder and diverge
/// the door's windows.
fn blocks_of(rows: u32, cols: u32, live: u32) -> (u32, u32, u32) {
    let blk = ktir_superdsc::superdsc_opspec::rows_per_block(cols, live);
    if blk >= rows {
        // The region FITS: one whole-region tile, the constants the one-tile form
        // always stated (and the builder's un-blocked arm still states).
        return (rows, 1, 0);
    }
    let n_blocks = rows / blk;
    let tail = rows % blk;
    (blk, n_blocks, tail)
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
        SubOp::ScalarMul { .. } => Ok(vec![1]),
        // ONE WORK ITEM: the position loop is a constant-trip `tl.range` inside the
        // kernel, unrolled by the ladder (`to_ktir::unroll_constant_trip_loops`), so the
        // spliced program is straight-line like the builder's — no grid axis at all.
        SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. } => Ok(vec![1]),
        // ⛔ NO `_` ARM. A spliced kind is a row above, and a row without a grid here is
        // an unreachable — the same discipline the arity match holds.
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
            EwKind::Gelu => "gelu",
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
        Program::ScalarMul => "scalarmul",
        _ => "triton",
    }
}
