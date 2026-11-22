// SPDX-License-Identifier: Apache-2.0
//! THE SPLICE — where the tape's node and the kernel meet.
//!
//! Each tape node a Triton kernel covers is compiled here — `triton-frontend` → TTIR →
//! `triton-ktir` → KTIR, at `#[forward]` expansion time, in-process, with no Python executing —
//! and handed on as a [`scratchy_spyre_bundle::program::KtirNode`] to the one KTIR→SuperDSC
//! lowering (`scratchy-spyre-superdsc`). This crate is the ONLY producer for the ops it covers.
//!
//! # WHAT THE TAPE STATES THAT A KERNEL CANNOT
//!
//! A Triton kernel is a function over its parameters. It has no words for "which graph
//! buffer is parameter 0" or "this is an rmsnorm". The tape states both:
//!
//! * **The operand binding.** The registry row declares the kernel's parameter order to be
//!   the node's operand order — parameter `i` is `node.inputs[i]`'s tensor, and the last
//!   parameter is the node's output. [`lower`] then re-states the adapter's positional
//!   `BufferId`s with the tape's real tensor indices, the operand names every descriptor
//!   addresses by. A splice that kept the positional ids would emit descriptors over buffers
//!   `t0/t1/t2` while the tape's buffers are `t37/t41/t42`: well-formed, and wrong.
//! * **The kind.** The row states the [`Program`] classification, so nothing downstream
//!   infers it from the op soup.
//!
//! # ⛔⭐ TOTALITY BY CONSTRUCTION — NO LOOKUP, NO FALLTHROUGH
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
//! * **`GemmWeight::Affine`** — affine-quantized presets are skipped before a tape is ever
//!   lowered, so no Affine node reaches this crate. [`matmul_row`] still enumerates it:
//!   wiring one in is a compile error here until the kernel lands with the arm.
//! * **`EwKind::QuickGelu` / `EwKind::GeluErf`** — the DDL has no primitive for either, and
//!   substituting `gelu` would run a different function. [`elementwise_row`] enumerates
//!   them so the same law holds.
//!
//! # THE GATES
//!
//! The golden (`tests/triton_splice_golden.rs`) checks the splice's `EmittedOp`s and, for the
//! families the emulator runs, an emulator `max_abs` comparison; the bake's own emulator gate
//! runs every rung's programs against a host reference.
//!
//! # ⛔ THE SHAPE WORK LIST, AS NAMED ERRORS
//!
//! A node whose shape a kernel cannot state is refused at the bake, naming the kernel
//! capability that must land; the current guards are the work list.

pub mod compile;

/// One node's program as the splice mints it: its name in the bake log, its KTIR, and the
/// collective consume it belongs to — set by the caller that plans the collective.
#[derive(Clone)]
pub struct EmittedOp<T: scratchy_spyre_bundle::collective::Tp> {
    pub op_name: String,
    pub ktir: KtirNode,
    pub tp_consume: Option<scratchy_spyre_bundle::collective::TpConsume<T>>,
}

impl<T: scratchy_spyre_bundle::collective::Tp> EmittedOp<T> {
    /// A program outside any collective.
    pub fn new(op_name: String, ktir: KtirNode) -> EmittedOp<T> {
        EmittedOp {
            op_name,
            ktir,
            tp_consume: None,
        }
    }
}

use std::collections::HashMap;

use ktir_core::arena::Arena;
use ktir_core::program::{Elementwise, Program};
use scratchy_spyre_bundle::program::{BufferId, KtirNode};
use scratchy_subtile::lower::GemmWeight;
use scratchy_subtile::subtile_ir::{
    AttnScaleSite, EwKind, GainConvention, SubOp, SubtileIR, SubtileNode,
};
use spyre_superdsc::format::DataFormat as Df;

use crate::compile::CompiledKernel;
use triton_frontend::codegen::{ArgSpec, KernelSpec};
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

/// One registry row: the kernel's SOURCE TEXT and entry, and the STATED classification —
/// declared data, not logic. Adding a kernel is a row; nothing else in this crate changes.
///
/// ⭐ THE SOURCE RIDES THE ROW AS `include_str!` — a missing kernel file is a COMPILE
/// error in this crate, not a runtime file-read failure at `#[forward]` expansion. The
/// include is also cargo's own dependency edge: editing a `.py` rebuilds every caller,
/// so the tracking block the macros crate used to carry (reading this directory with
/// `std::fs` per expansion) is gone.
pub enum TritonKernelRow {
    /// A kernel this crate can compile: its full source, its entry, its classification.
    Spliced {
        src: &'static str,
        file: &'static str,
        entry: &'static str,
        program: Program,
    },
    /// A family member whose device realization does not exist yet — DECLARED, never
    /// discovered at expansion. The message is the refusal the splice states if the
    /// routing ever hands it here, and the arm keeps the family match exhaustive (E0004
    /// stays the failure mode for a new family member, not a runtime miss).
    Refused(&'static str),
}

// ── THE TOTAL REGISTRY, FAMILY BY FAMILY ─────────────────────────────────────────
//
// ⛔⭐ Each function is an EXHAUSTIVE match over one family's vocabulary with no `_`
// arm and no `Option` — a family member without a kernel is E0004 at compile time
// here, which is the whole design: we are a procmacro compiler, everything is known
// at compile time, and no runtime lookup can miss.

/// `SubOp::RmsNorm`'s two gain conventions — two entries of one kernel file. The
/// Scale form multiplies by the loaded gain; the `(1 + w)` class (gemma) adds the
/// offset over the load first, a splat the lowering stages as a stick constant, as it
/// stages the epsilon.
pub fn rmsnorm_row(gain: GainConvention) -> TritonKernelRow {
    match gain {
        GainConvention::Scale => TritonKernelRow::Spliced {
            src: include_str!("../../../../kernels/rmsnorm.py"),
            file: "rmsnorm.py",
            entry: "rmsnorm_fwd",
            program: Program::RmsNorm,
        },
        // ⭐ THE (1 + w) ROW — the offset is CARRIED, never folded into the weights
        // (folding would make the loaded gain disagree with the checkpoint, so every
        // consumer that re-reads it — dumps, the numeric reference, a second target —
        // would see a different tensor). The kernel states the addf over the load, so
        // the descriptor computes the formula the program runs.
        GainConvention::OnePlusScale => TritonKernelRow::Spliced {
            src: include_str!("../../../../kernels/rmsnorm.py"),
            file: "rmsnorm.py",
            entry: "rmsnorm_oneplus_fwd",
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
        GemmWeight::Dense => TritonKernelRow::Spliced {
            src: include_str!("../../../../kernels/matmul.py"),
            file: "matmul.py",
            entry: "matmul_fwd",
            program: Program::Matmul,
        },
        // THE fp8 W8A8 ROW. Same `Program::Matmul` classification as dense: fp8-ness is
        // the weight view's own format and the arity-3 bindings, never the program
        // kind. The kernel's spelled
        // `* w_scale` epilogue is what the ladder's
        // `verify_canonical_fp8_matmul_kernel` requires.
        GemmWeight::Fp8Dynamic => TritonKernelRow::Spliced {
            src: include_str!("../../../../kernels/matmul_fp8.py"),
            file: "matmul_fp8.py",
            entry: "matmul_fp8_fwd",
            program: Program::Matmul,
        },
        // ⛔ NO KERNEL: the wavefront lowering skips affine presets before a tape is
        // lowered (`codegen`'s `no superdsc bundle` gate) and no consumer realizes an
        // affine contraction on this path, so nothing constructs this node on the spyre
        // path. The arm is the compile-time enumeration — a realization lands here
        // WITH its kernel (the GPU target's qmv family is the precedent shape).
        GemmWeight::Affine { .. } => TritonKernelRow::Refused(
            "the affine weight scheme: the wavefront lowering skips affine presets before \
             a tape is lowered, so no node of this shape is constructed on the spyre path. \
             A realization lands here WITH its kernel",
        ),
    }
}

/// `SubOp::Elementwise`'s kinds. The kinds the DDL has no primitive for are enumerated
/// with the same names the consumer's own refusal uses, so wiring one in fails at E0004
/// until a producer-side decomposition lands.
pub fn elementwise_row(kind: EwKind) -> TritonKernelRow {
    let ew_src = || include_str!("../../../../kernels/elementwise.py");
    match kind {
        // Add AND BiasAdd share `add_fwd` — both map to `Elementwise::Add` and the same
        // `add_s{id}` name.
        EwKind::Add | EwKind::BiasAdd => TritonKernelRow::Spliced {
            src: ew_src(),
            file: "elementwise.py",
            entry: "add_fwd",
            program: Program::Elementwise(Elementwise::Add),
        },
        EwKind::Mul => TritonKernelRow::Spliced {
            src: ew_src(),
            file: "elementwise.py",
            entry: "mul_fwd",
            program: Program::Elementwise(Elementwise::Mul),
        },
        EwKind::Sub => TritonKernelRow::Spliced {
            src: ew_src(),
            file: "elementwise.py",
            entry: "sub_fwd",
            program: Program::Elementwise(Elementwise::Sub),
        },
        EwKind::Silu => TritonKernelRow::Spliced {
            src: ew_src(),
            file: "elementwise.py",
            entry: "silu_fwd",
            program: Program::Elementwise(Elementwise::Silu),
        },
        // A REAL DDL primitive (`OpFunc::Gelu`): the SFP constant table ships gelu's
        // tanh polynomial, so this is one pointwise op. The kernel spells the tanh
        // form through the exp island (the frontend has no `tanh`), which is the same
        // approximation the `"gelu"` primitive itself makes — see `elementwise.py`'s
        // `gelu_fwd`.
        EwKind::Gelu => TritonKernelRow::Spliced {
            src: ew_src(),
            file: "elementwise.py",
            entry: "gelu_fwd",
            program: Program::Elementwise(Elementwise::Gelu),
        },
        // ⛔ NO DDL PRIMITIVE, and substituting the nearest one is the bug: quick-gelu
        // is x·σ(1.702x), a DIFFERENT function from OpFunc::Gelu's tanh polynomial —
        // emitting "gelu" for it would run the wrong model and report success. A
        // realization must DECOMPOSE producer-side (its building blocks — sigmoid,
        // mul — ARE primitives), which is a kernel family of its own.
        EwKind::QuickGelu => TritonKernelRow::Refused(
            "quick-gelu: x·σ(1.702x) is a different function from the DDL's tanh-polynomial \
             gelu, and the DDL has no quick-gelu primitive — substituting `gelu` would run \
             the wrong model and report success. A realization must decompose \
             producer-side (sigmoid and mul ARE primitives)",
        ),
        // ⛔ SAME LAW: exact-erf gelu is the erf form, not the tanh polynomial, and the
        // consumer has no `erf` op either, so a decomposition cannot lower today.
        EwKind::GeluErf => TritonKernelRow::Refused(
            "exact-erf gelu: the erf form, not the tanh polynomial the DDL primitive makes, \
             and there is no `erf` op to decompose with — the realization is a kernel \
             family of its own",
        ),
        // No elementwise kernel for squared ReLU in `elementwise.py` — an arch using it
        // (Nemotron-H) fails the build at this row, by name.
        EwKind::Relu2 => TritonKernelRow::Refused(
            "relu2: no squared-ReLU kernel in the elementwise family — a realization is a \
             new `elementwise.py` entry",
        ),
    }
}

/// The prefill lm-head fold's extraction row. The fold is TWO ops from ONE
/// `MatmulTile` node (see [`lower_all`]), and the extraction half is its own kernel
/// family — the row is declared here so the registry's own law holds ("adding a
/// kernel is a row; nothing else in this crate changes").
pub fn lmlast_row() -> TritonKernelRow {
    TritonKernelRow::Spliced {
        src: include_str!("../../../../kernels/lmlast.py"),
        file: "lmlast.py",
        entry: "lmlast_fwd",
        program: Program::LmLast,
    }
}

/// THE BLOCK COPY — an all-gather consume's leg (`blockcopy.py`), placed by
/// [`lower_block_copy`], which the spyre emitter's collective arm calls; no `SubOp` reaches it
/// through [`row`].
pub fn blockcopy_row() -> TritonKernelRow {
    // The row's `entry` is the whole-block copy; a copy whose rows leave a tail is
    // `BLOCKCOPY_TAIL_ENTRY`, the same file's other entry (`blockcopy.py` says why there are two).
    TritonKernelRow::Spliced {
        src: include_str!("../../../../kernels/blockcopy.py"),
        file: "blockcopy.py",
        entry: "blockcopy_fwd",
        program: Program::BlockCopy,
    }
}

/// `blockcopy.py`'s second entry: the whole-block copy followed by its one shorter tail.
const BLOCKCOPY_TAIL_ENTRY: &str = "blockcopy_tail_fwd";

/// THE ROW FOR A NODE — the family functions composed. Every `SubOp` that can reach the
/// splice has a family below (attention rides its own `lower_attn` entry); the ops the
/// spyre target has no kernel FOR (the 2b expansion kinds, the host-routed pairs, …)
/// never reach this crate — `lower_one_node`'s own arms own those refusals by name.
pub fn row<F: scratchy_subtile::subtile_ir::RopeForm>(op: &SubOp<F>) -> TritonKernelRow {
    match op {
        SubOp::RmsNorm { gain, .. } => rmsnorm_row(*gain),
        SubOp::MatmulTile { weight, .. } => matmul_row(weight),
        SubOp::Elementwise(kind) => elementwise_row(*kind),
        SubOp::SiluMul => TritonKernelRow::Spliced {
            src: include_str!("../../../../kernels/silumul.py"),
            file: "silumul.py",
            entry: "silumul_fwd",
            program: Program::SiluMul,
        },
        SubOp::ScalarMul { .. } => TritonKernelRow::Spliced {
            src: include_str!("../../../../kernels/scalarmul.py"),
            file: "scalarmul.py",
            entry: "scalarmul_fwd",
            program: Program::ScalarMul,
        },
        // `RopeAppend` carries 6 inputs but only the first three are the rotation (V
        // and the KV-cache destinations flow through GRAPH edges), so the kernel
        // consumes x/cos/sin/out
        // and the row covers BOTH `RopeAppend` and the standalone `RopeRotate`.
        SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. } => TritonKernelRow::Spliced {
            src: include_str!("../../../../kernels/rope.py"),
            file: "rope.py",
            entry: "rope_fwd",
            program: Program::Rope,
        },
        // ATTENTION SPLICES — `lower_attn` is the entry (the paged step's tables and
        // pools are bindings of their own). This row exists so `row()` stays
        // exhaustive; the walk never routes attention through `lower_one`.
        SubOp::AttnDecode { .. } => attn_row(),
        // ⛔ THE MoE SUB-OPS ARE NOT SPLICED KINDS ANY MORE — the MoE lowers
        // only from `kernels/moe.py` through KTIR, and
        // `lower_subtile_tape_to_ktir` refuses every MoE SubOp before the
        // splice (its one Unhandled arm). A node reaching this row means the
        // routing changed; the fix is a kernel plus a KTIR path, never a
        // per-op row.
        SubOp::RouterNorm { .. }
        | SubOp::RouterLogits { .. }
        | SubOp::RouteScale { .. }
        | SubOp::RouteSoftmax
        | SubOp::RouteRenorm
        | SubOp::RouteArgsort
        | SubOp::RouteTopK { .. }
        | SubOp::RouteGatherScores
        | SubOp::RouteExpertScale { .. }
        | SubOp::ExpertGatedAct { .. }
        | SubOp::ExpertSort { .. }
        | SubOp::ExpertUnsort
        | SubOp::ExpertCombine { .. }
        | SubOp::ExpertMatmul { .. } => TritonKernelRow::Refused(
            "the MoE lowers only from kernels/moe.py through KTIR — no per-op MoE row exists",
        ),
        // ⛔ A COLLECTIVE IS NOT ONE KERNEL. It is a control-block sequence in the compute
        // stream — p2p sends to the peers' receive windows, a write-done wait, then the
        // consume — and its only kernel (the consume's add) is the `Add` row's. The spyre
        // emitter's collective arm owns it; no Triton program states a card-to-card transfer.
        SubOp::AllReduce | SubOp::AllGather { .. } => TritonKernelRow::Refused(
            "a collective lowers as a CB sequence by the spyre emitter's collective arm, \
             never as one spliced kernel",
        ),
        SubOp::TanhSoftCap { .. } => TritonKernelRow::Spliced {
            src: include_str!("../../../../kernels/tanhsoftcap.py"),
            file: "tanhsoftcap.py",
            entry: "tanhsoftcap_fwd",
            program: Program::TanhSoftCap,
        },
        SubOp::RmsNormUnit { .. } => TritonKernelRow::Spliced {
            src: include_str!("../../../../kernels/rmsnormunit.py"),
            file: "rmsnormunit.py",
            entry: "rmsnormunit_fwd",
            program: Program::RmsNormUnit,
        },
        SubOp::ScalarWeightMul => TritonKernelRow::Spliced {
            src: include_str!("../../../../kernels/scalarwmul.py"),
            file: "scalarwmul.py",
            entry: "scalarwmul_fwd",
            program: Program::ScalarWeightMul,
        },
        SubOp::Reshape { .. } => TritonKernelRow::Spliced {
            src: include_str!("../../../../kernels/reshape.py"),
            file: "reshape.py",
            entry: "reshape_fwd",
            program: Program::Reshape,
        },
        // ⛔⭐ THE OPS `lower_one_node` NEVER ROUTES HERE — enumerated by NAME, never
        // `_`, so adding a SubOp is an E0004 in this crate too. `lower_one_node`'s own
        // match is the gate: these kinds have their own arms there, and a kind reaching
        // THIS match means the routing changed — the
        // message names what must land, and no `_` arm can swallow it.
        SubOp::RmsNormReduce { .. }
        | SubOp::RmsNormApply { .. }
        | SubOp::SumReduce { .. }
        | SubOp::GateSplit { .. }
        | SubOp::GateApply
        | SubOp::Concat { .. }
        | SubOp::GateScale
        | SubOp::LoadRows { .. }
        | SubOp::EmbeddingGather { .. }
        | SubOp::VisionRope
        | SubOp::VarlenAttention { .. }
        | SubOp::EncoderAttn { .. }
        | SubOp::GatedDeltaNet
        | SubOp::Mamba2 { .. }
        | SubOp::KvAppend { .. }
        | SubOp::Mean
        // ⛔ AND THE 2b EXPANSION KINDS, named here instead of `expansion_ops!()`: the
        // 2a routing above made the macro's patterns for THOSE kinds unreachable
        // (unreachable_patterns is `-D warnings`), and the remaining members are a
        // fixed list the registry's `@expansion` section states — a new expansion op
        // is E0004 here until it gets its own row or refusal, exactly as before.
        // The remaining router rows are the biased routers (the Phase-2b
        // worklist); the rest have their own `lower_one_node` arms.
        | SubOp::KvEncode { .. }
        | SubOp::KvStage { .. }
        | SubOp::RotateRows { .. }
        | SubOp::AttnPackedKv
        | SubOp::RouteSigmoidBias { .. }
        | SubOp::RouteBias { .. }
        | SubOp::ExpertAct { .. }
        | SubOp::SampleRowsGather
        | SubOp::SampleRowsScatter
        | SubOp::AllRowsMatmul => TritonKernelRow::Refused(
            "not a spliced kind — `lower_one_node` owns this op's lowering (its own arm, a \
             host routing, or its own by-name refusal); reaching the splice means the \
             routing changed, and the fix is a family function plus a kernel",
        ),
        // A DFlash2 drafter's ops and its target's capture: Splash's kernels, ported as
        // the GPU target's opcode lowerings alone.
        SubOp::CaptureHidden { .. }
        | SubOp::DraftConv { .. }
        | SubOp::DraftAttention { .. }
        | SubOp::DraftContextKv { .. }
        | SubOp::DraftWindowStore
        | SubOp::DraftWindowLoad
        | SubOp::DraftSelect => TritonKernelRow::Refused(
            "a DFlash2 drafter's op (or its target's capture): its kernels are Splash's, \
             lowered on the GPU target alone — a spyre drafter needs its kernels first",
        ),
    }
}

/// THE SPLICE — the ONLY producer for a spliced kind. Compile the kernel the registry
/// names for this node and hand back the [`EmittedOp`]: `EmittedOp::bare(name)` with
/// `ktir` set.
///
/// There is no `Ok(None)` and nothing to fall through to: every spliced kind either
/// emits or the bake stops, loudly, naming the kernel capability that must land. The
/// shape guards below are the work list — each one dies when its kernel states that
/// shape, and the end state has none.
///
/// The name is `rmsnorm_s{node.id}`-shaped, so the spliced op's `op_name` and the
/// emulator's function key follow one naming law. That is what the golden's `op_name`
/// pins.
///
/// ⭐ ONE NODE MAY LOWER TO MORE THAN ONE OP, and only the splice knows which kinds:
/// [`lower_all`] is the entry the walk calls, and it routes every one-op kind here.
pub fn lower<
    T: scratchy_spyre_bundle::collective::Tp,
    F: scratchy_subtile::subtile_ir::RopeForm,
>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    // Whether this bundle's rows are separate requests (the lm-head fold's own fact).
    rows_are_requests: bool,
) -> Result<EmittedOp<T>, String> {
    lower_one(node, ir, rows_are_requests)
}

/// The prefill lm-head tail predicate, stated ONCE — every consumer of the fold
/// (the `lower_all` rewrite, the one-op `lower`'s routing echo below, and the
/// bake-time numeric gate's tail compare, which states the fold's last-row law
/// against the golden) reads
/// the SAME five conjuncts, because the width test alone was falsified by the
/// gemma-4 parity fixture (vocab 512), whose GLOBAL-layer q projection is ALSO
/// 512 cols (4 heads × hd 128) — the fold fired on every global layer's Q, and
/// the model's first sampled token was garbage while every width-128 tensor
/// stayed fp16-clean. A width COLLISION is not an identity: only the tail's own
/// rows are dead, and only the result's ancestors are the tail. The walk is
/// [`tensor_feeds_result_through_row_local_ops`] (main's own law). The fifth
/// conjunct names the fold's own product: a matmul whose activation is the
/// beyond-the-graph `LAST_HIDDEN` staging is the fold's re-lowered half-2
/// ALREADY — never a tail to fold (at m=1 the rows conjunct excluded it; at
/// m=S the activation must).
pub fn is_prefill_lm_head_tail<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    rows_are_requests: bool,
) -> bool {
    // ⭐ THE TAIL'S PRODUCT IS THE RESULT'S WIDTH — the node's output region is as wide as the
    // result, or (a vocabulary projection split across ranks, one rank's block) it reaches the
    // result through the all-gather that restores the whole vocabulary. One predicate, read by
    // every caller — the `lower_all` rewrite and the one-op routing echo — so both state the SAME
    // fold. (The width alone was never sufficient — the gemma-4 q-projection collision is what the
    // feeds-walk conjunct below kills.)
    let result_cols = ir.tensors[ir.result.index()].cols;
    let feeds_the_tail_product = node.output.region.cols.len == result_cols
        || matches!(node.op, SubOp::MatmulTile { .. })
            && gathers_to_width(ir, node.output.tensor, result_cols);
    // A projection's rows are its ACTIVATION's: every tail op has the same rows in and out, and a
    // projection's are the prompt rows it folds over.
    let rows = match &node.op {
        SubOp::MatmulTile { .. } => node.inputs.first().map_or(0, |a| a.region.rows.len),
        _ => node.output.region.rows.len,
    };
    feeds_the_tail_product
        && rows > 1
        && !rows_are_requests
        // ⭐ THE FOLD'S HALF-2 IS NEVER A TAIL TO FOLD — its activation is the
        // reserved `LAST_HIDDEN` staging, a synthetic BEYOND the graph (an id
        // past `ir.tensors`); a graph matmul's activation is always a graph
        // tensor. At m=1 the rows conjunct alone excluded the half-2; at m=S
        // this conjunct does, and no graph node changes side.
        && node
            .inputs
            .first()
            .is_some_and(|a| a.tensor.index() < ir.tensors.len())
        && tensor_feeds_result_through_row_local_ops(ir, node.output.tensor)
}

/// Whether `t` reaches an all-gather restoring `width` columns through single-input logit
/// epilogues — the path a split vocabulary projection takes to the logits.
fn gathers_to_width<F: scratchy_subtile::subtile_ir::RopeForm>(
    ir: &SubtileIR<F>,
    t: scratchy_subtile::subtile_ir::TensorId,
    width: u32,
) -> bool {
    ir.nodes
        .iter()
        .filter(|n| n.inputs.iter().any(|i| i.tensor == t))
        .any(|n| match &n.op {
            SubOp::AllGather { cols } => *cols == width,
            SubOp::ScalarMul { .. } | SubOp::TanhSoftCap { .. } if n.inputs.len() == 1 => {
                gathers_to_width(ir, n.output.tensor, width)
            }
            _ => false,
        })
}

/// EVERY OP THE SPLICE EMITS FOR ONE NODE. Every kind is one op except the prefill
/// lm-head tail, whose fold is TWO — the answer-row extraction (its own program, the
/// reserved `LAST_HIDDEN_TID` staging) plus the re-lowered m=S matmul — main's
/// `lower_prefill_lm_head_at_m1` shape, stated at the rung's own request count
/// (exactly m=1 there, byte-identical).
pub fn lower_all<
    T: scratchy_spyre_bundle::collective::Tp,
    F: scratchy_subtile::subtile_ir::RopeForm,
>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    rows_are_requests: bool,
    // ⭐ THE QUERY'S EXTENT PAIR — the rung's own `(width, capacity)`, the same mint
    // `lower_attn` takes. The lm-head tail's fold is the reader here: it mints the tail's
    // [`AnswerRows`] from the pair and refuses a mint that does not describe the graph
    // being lowered. Every other interception ignores it, exactly as the decode and batch
    // arms of `lower_attn` read the graph's rows and never the pair's split.
    query_rows: scratchy_spyre_bundle::rung::QueryExtent,
) -> Result<Vec<EmittedOp<T>>, String> {
    // ── THE HEAD-NORM SANDWICH'S THREE SEATS — the split (head) is where the fused
    // program is emitted; the flatten-back (tail) and the middle (norm) self-identify
    // as fused-away and emit NOTHING. The match is dataflow-based
    // ([`sandwich_at_split`]), so it is a pure function of the graph — every walk
    // reaches the same answer at the same nodes, and any failed check falls back to
    // the byte-identical un-fused lowering (the ordinary reshape/norm rows below).
    // ⛔ THE MATCHER ACCEPTS GAINED (`Scale`/`OnePlusScale`) AND `RmsNormUnit`
    // MIDDLES. The two gained conventions are two ENTRIES of the same windowed
    // body (`windowedrmsnorm_gained_fwd` / `windowedrmsnorm_oneplus_fwd`), the oneplus
    // form's offset its program's own addf — so neither convention needs to stay
    // un-fused: the matcher is a pure function of the graph either way.
    if let SubOp::Reshape { .. } = &node.op {
        if is_sandwich_flatten(node, ir) {
            return Ok(Vec::new());
        }
        if let Some(sandwich) = sandwich_at_split(node, ir) {
            return lower_sandwich_node(&sandwich);
        }
    }
    if matches!(&node.op, SubOp::RmsNorm { .. } | SubOp::RmsNormUnit { .. })
        && is_sandwich_norm(node, ir)
    {
        return Ok(Vec::new());
    }
    let is_prefill_lm_head_tail = is_prefill_lm_head_tail(node, ir, rows_are_requests);
    if let SubOp::MatmulTile { .. } = &node.op
        && is_prefill_lm_head_tail
    {
        return lower_prefill_lm_head_fold(node, ir, query_rows);
    }
    // ⭐ THE LOGITS SCALARMUL IS THE TAIL'S SECOND HALF — main's own law
    // (`lower_scalarmul_node(&node_at_one_row(node), …)` under the same
    // `is_prefill_lm_head_tail`), stated at the rung's own request count: the
    // m>1 matmul above it is folded to the m=S tail that writes S logits rows,
    // so the scale must run over those S rows too. Left un-folded it scales
    // mq × vocab logits of which mq−S rows were never written — row 0 is still
    // right (which is why text A/Bs pass), but the device pays mq−S rows of
    // reads of unwritten logits for nothing.
    if let SubOp::ScalarMul { .. } = &node.op
        && is_prefill_lm_head_tail
    {
        let at_tail = node_at_tail_rows(node, tail_requests(query_rows));
        return lower_one(&at_tail, ir, rows_are_requests).map(|e| vec![e]);
    }
    // ⭐ AND THE LOGITS TANH-SOFTCAP IS THE TAIL'S THIRD KIND — the same
    // `node_at_tail_rows` law (main gives TanhSoftCap the fold ScalarMul has):
    // the softcap sits ON the logits, so the folded tail caps the S written rows
    // of them instead of mq rows of which mq−S were never written.
    if let SubOp::TanhSoftCap { .. } = &node.op
        && is_prefill_lm_head_tail
    {
        let at_tail = node_at_tail_rows(node, tail_requests(query_rows));
        return lower_one(&at_tail, ir, rows_are_requests).map(|e| vec![e]);
    }
    // ⭐ A KV WRITER IS TWO PROGRAMS: the rotation (`rope.py`, the node's own output) and the paged
    // write that scatters the rotated K and the V into the pools (`reshape_and_cache_head_major.py`) — spyre-
    // inference's `do_kv_cache_update`, run before the step's attention reads the pages.
    if let SubOp::RopeAppend { .. } = &node.op {
        return Ok(vec![
            lower_one(node, ir, rows_are_requests)?,
            lower_kv_write(node, ir)?,
        ]);
    }
    lower_one(node, ir, rows_are_requests).map(|e| vec![e])
}

/// The folded lm-head tail's own row window — the FIRST `s` rows (the region's own
/// `rows.start`, length `s`) of the input and the output, where `s` is the rung's
/// request CAPACITY: the folded tail writes the `[S, vocab]` prefix of the `[mq, vocab]`
/// logits storage (request r's row is row r), so its consumers run over exactly those
/// rows. At `s = 1` this is main's own one-row law, byte-identical.
fn node_at_tail_rows<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    s: u32,
) -> SubtileNode<F> {
    let s_rows = |tr: &scratchy_subtile::subtile_ir::TensorRegion| {
        scratchy_subtile::subtile_ir::TensorRegion {
            tensor: tr.tensor,
            region: scratchy_subtile::subtile_ir::Region {
                rows: scratchy_subtile::subtile_ir::Range::new(tr.region.rows.start, s),
                cols: tr.region.cols,
            },
        }
    };
    let mut inputs = node.inputs.clone();
    if let Some(a) = inputs.first_mut() {
        *a = s_rows(a);
    }
    SubtileNode {
        id: node.id,
        op: node.op,
        inputs,
        output: s_rows(&node.output),
    }
}

/// The folded tail's own row count — the rung's request capacity `S` off the extent
/// pair the walk hands every lowering. The tail writes `S` rows on every launch of the
/// rung, including the empty windows a launch seated fewer requests leaves (their
/// logits rows are pads, never read); the readback takes the live count off the
/// launch's own slot list.
fn tail_requests(query_rows: scratchy_spyre_bundle::rung::QueryExtent) -> u32 {
    query_rows.requests().get()
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE HEAD-NORM SANDWICH — `Reshape{Times(H)} → RmsNorm/RmsNormUnit → Reshape{Times(1)}`,
//  gemma-4's per-head q/k norms (87% of its descriptors when un-fused), fused to ONE
//  windowed program over the ORIGINAL tensor with both reshapes deleted.
//
//  The flatten-back proof the match demands: the split states `Times(H)` (a per-head view
//  multiplies rows), the flatten states `Times(1)`, both preserve the element count and
//  row-major order (the Reshape law), and the flatten's `[rows, cols]` must equal the
//  SPLIT'S INPUT's `[rows, cols]` — the same extents, not merely the same count, so the
//  flatten restores exactly the arrangement the split left and head window `h` of the
//  output is the split input's columns `h·D..(h+1)·D`.
//
//  ⛔ EVERY CHECK IS A REFUSAL BACK TO THE UN-FUSED PATH. Any extra consumer of the
//  split's output (a skip connection reading the flat q), any op between norm and
//  flatten, an in-place alias (split output == flatten output), a windowed region (the
//  kernel addresses the TENSOR BASES, so it cannot state a region's corner) — the matcher
//  returns `None` and all three nodes lower through their ordinary rows.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The matched sandwich, carried from the split (the one position the fused program
/// is emitted from) to the mint.
struct HeadNormSandwich<'a, F: scratchy_subtile::subtile_ir::RopeForm> {
    /// The split `Reshape{Times(H)}` node.
    split: &'a SubtileNode<F>,
    /// The per-head norm node (gained or unit).
    norm: &'a SubtileNode<F>,
    /// The flatten-back `Reshape{Times(1)}` node.
    flatten: &'a SubtileNode<F>,
    /// The per-head width `D` (the split's output columns). The head COUNT is not
    /// carried: it is `full / D` of the split input's own extents, so no second spelling
    /// can disagree.
    head_dim: u32,
    /// The norm's epsilon, read once here so the mint states one value.
    eps: f32,
    /// Whether the norm carries a gamma row (`RmsNorm`) or not (`RmsNormUnit`).
    gained: bool,
    /// Whether the gained form is the `(1 + w)` convention — a THIRD entry of the
    /// same windowed body whose gamma leg adds `GAIN_OFFSET` over the load. Kept
    /// separate from `gained`: the unit form is neither, and the two gained
    /// conventions differ in the ENTRY and one constexpr only, never in bindings.
    oneplus: bool,
}

/// The nodes reading `t` (any region of it) — the consumers a single-use match demands.
fn consumers_of<F: scratchy_subtile::subtile_ir::RopeForm>(
    ir: &SubtileIR<F>,
    t: scratchy_subtile::subtile_ir::TensorId,
) -> Vec<&SubtileNode<F>> {
    ir.nodes
        .iter()
        .filter(|n| n.inputs.iter().any(|r| r.tensor == t))
        .collect()
}

/// Whether `n`'s input region is the WHOLE of `t` at the norm's own output extents —
/// the fused norm must read every element of the re-laid view (a partial read means a
/// window this fusion cannot state).
fn reads_whole<F: scratchy_subtile::subtile_ir::RopeForm>(
    n: &SubtileNode<F>,
    t: scratchy_subtile::subtile_ir::TensorId,
) -> bool {
    let Some(r) = n.inputs.iter().find(|r| r.tensor == t) else {
        return false;
    };
    r.region.rows.start == 0
        && r.region.cols.start == 0
        && r.region.rows.len == n.output.region.rows.len
        && r.region.cols.len == n.output.region.cols.len
}

/// The node whose OUTPUT is `t`, if exactly one — the producer a chain-walk follows
/// upward. `None` for a source (no producer) or a tensor two nodes write (not a chain).
fn producer_of<F: scratchy_subtile::subtile_ir::RopeForm>(
    ir: &SubtileIR<F>,
    t: scratchy_subtile::subtile_ir::TensorId,
) -> Option<&SubtileNode<F>> {
    let mut it = ir.nodes.iter().filter(|n| n.output.tensor == t);
    let p = it.next()?;
    it.next().is_none().then_some(p)
}

/// Match the sandwich AT THE SPLIT — the one position the fused program is emitted
/// from. Both gained conventions and the unit form match: the gained forms are two
/// entries of the same windowed body, the OnePlusScale form's offset its program's own
/// addf, so the matcher stays a pure function of the graph for every convention.
fn sandwich_at_split<'a, F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &'a SubtileNode<F>,
    ir: &'a SubtileIR<F>,
) -> Option<HeadNormSandwich<'a, F>> {
    use scratchy_subtile::subtile_ir::RowScale;
    let SubOp::Reshape {
        rows: RowScale::Times(heads),
        cols: head_dim,
    } = &node.op
    else {
        return None;
    };
    // A per-head view MULTIPLIES rows; `Times(1)` is a shape-preserving op's non-view.
    if heads.get() == 1 {
        return None;
    }
    if node.inputs.len() != 1 {
        return None;
    }
    let src = &node.inputs[0];
    let split_out = node.output.tensor;
    // The split's single consumer must be the per-head norm, reading the whole view.
    let [norm] = consumers_of(ir, split_out)[..] else {
        return None;
    };
    let (eps, gained, oneplus) = match &norm.op {
        SubOp::RmsNorm {
            eps,
            gain: GainConvention::Scale,
        } => (*eps, true, false),
        SubOp::RmsNorm {
            eps,
            gain: GainConvention::OnePlusScale,
        } => (*eps, true, true),
        SubOp::RmsNormUnit { eps } => (*eps, false, false),
        _ => return None,
    };
    if gained && norm.inputs.len() != 2 || !gained && norm.inputs.len() != 1 {
        return None;
    }
    if !reads_whole(norm, split_out) {
        return None;
    }
    // The norm's single consumer must be the flatten-back, restoring the split's input.
    let norm_out = norm.output.tensor;
    let [flatten] = consumers_of(ir, norm_out)[..] else {
        return None;
    };
    let SubOp::Reshape {
        rows: RowScale::Times(one),
        cols: flat_cols,
    } = &flatten.op
    else {
        return None;
    };
    if one.get() != 1 {
        return None;
    }
    // THE FLATTEN-BACK PROOF: same extents as the split's input, so the flat
    // arrangement — and therefore each head window — is restored exactly. A
    // same-count/different-shape flatten (e.g. `[m, H·D] → [m·D, H]`) permutes
    // elements and is NOT this fusion.
    if flatten.inputs.len() != 1
        || flatten.inputs[0].tensor != norm_out
        || flatten.output.region.rows.len != src.region.rows.len
        || *flat_cols != src.region.cols.len
        || flatten.output.tensor == split_out
    {
        return None;
    }
    // ⭐ THE KERNEL ADDRESSES THE TENSOR BASES (its descriptors bind the split's source
    // and the flatten's output directly), so it cannot state a region's corner. A
    // windowed sandwich stays un-fused — the ordinary reshape + norm + reshape rows
    // spell it.
    let whole = |tr: &scratchy_subtile::subtile_ir::TensorRegion| {
        let s = &ir.tensors[tr.tensor.index()];
        tr.region.rows.start == 0
            && tr.region.rows.len == s.rows
            && tr.region.cols.start == 0
            && tr.region.cols.len == s.cols
    };
    if !whole(src) || !whole(&flatten.output) || gained && !whole(&norm.inputs[1]) {
        return None;
    }
    Some(HeadNormSandwich {
        split: node,
        norm,
        flatten,
        head_dim: *head_dim,
        eps,
        gained,
        oneplus,
    })
}

/// Whether `node` is the MIDDLE (norm) of a sandwich matched at its own split — the
/// node whose program is emitted there and whose own lowering must therefore be NOTHING.
fn is_sandwich_norm<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
) -> bool {
    let Some(x) = node.inputs.first() else {
        return false;
    };
    producer_of(ir, x.tensor)
        .is_some_and(|split| sandwich_at_split(split, ir).is_some_and(|s| s.norm.id == node.id))
}

/// Whether `node` is the TAIL (flatten-back) of a sandwich — same derivation, one hop up.
fn is_sandwich_flatten<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
) -> bool {
    let Some(x) = node.inputs.first() else {
        return false;
    };
    producer_of(ir, x.tensor).is_some_and(|norm| is_sandwich_norm(norm, ir))
}

/// Lower the per-head norm SANDWICH to ONE program — the norm applied to head windows
/// of the ORIGINAL tensor, both reshapes deleted. [`HeadNormSandwich`]'s docs carry
/// the proof; this mints the kernel: `windowedrmsnorm.py`'s gained, oneplus, or unit
/// entry, compiled at the sandwich's own geometry, bound to the split's SOURCE `x`, the
/// gamma row (gained form only) and the flatten's OUTPUT — the same
/// synthetic-node-and-mint shape the prefill lm-head fold uses for `lmlast.py`.
fn lower_sandwich_node<
    T: scratchy_spyre_bundle::collective::Tp,
    F: scratchy_subtile::subtile_ir::RopeForm,
>(
    sandwich: &HeadNormSandwich<'_, F>,
) -> Result<Vec<EmittedOp<T>>, String> {
    let HeadNormSandwich {
        split,
        norm,
        flatten,
        head_dim,
        eps,
        gained,
        oneplus,
    } = sandwich;
    let x_r = &split.inputs[0];
    let out_r = &flatten.output;
    let (m, full) = (out_r.region.rows.len, out_r.region.cols.len);
    if *head_dim == 0 || full % head_dim != 0 {
        return Err(format!(
            "triton splice: rmsnorm_s{} (head-norm sandwich): the output t{} is [{m}, {full}] \
             over head dim {head_dim}, which does not divide — a per-head window must tile the \
             width exactly",
            split.id.index(),
            out_r.tensor.index()
        ));
    }
    let heads = full / head_dim;
    let src = include_str!("../../../../kernels/windowedrmsnorm.py");
    // THREE entries, one body: the matcher's own convention triple, stated as the
    // entry name only — the bindings are identical across the gained forms, and the
    // oneplus entry is the only one that takes `GAIN_OFFSET`.
    let entry = match (*gained, *oneplus) {
        (true, true) => "windowedrmsnorm_oneplus_fwd",
        (true, false) => "windowedrmsnorm_gained_fwd",
        (false, _) => "windowedrmsnorm_unit_fwd",
    };
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    let params: &[&str] = if *gained {
        &["desc_x", "desc_gamma", "desc_o"]
    } else {
        &["desc_x", "desc_o"]
    };
    for p in params {
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
    ce("D", Val::Int(i128::from(*head_dim)))?;
    ce("HEADS", Val::Int(i128::from(heads)))?;
    ce("EPS", Val::Float(f64::from(*eps)))?;
    ce("INV_D", Val::Float(1.0 / f64::from(*head_dim)))?;
    if *oneplus {
        // The convention's own `offset()` (1.0), staged by the lowering as a stick
        // constant like the epsilon.
        ce(
            "GAIN_OFFSET",
            Val::Float(f64::from(GainConvention::OnePlusScale.offset())),
        )?;
    }
    let spec = KernelSpec {
        file: "windowedrmsnorm.py".to_string(),
        kernel: entry.to_string(),
        signature,
        constexprs,
    };
    let module = compile_kernel(src, &spec, &[1])?;
    // The synthetic node: the NORM's op (so `mint`'s `program_stem` answers the
    // sandwich's own `rmsnorm` stem), the SPLIT's id (the naming law), and the
    // bindings in the kernel's parameter order — the split's source, the gamma row
    // (gained form only), the flatten's output.
    let mut inputs = vec![*x_r];
    if *gained {
        inputs.push(norm.inputs[1]);
    }
    let synth = SubtileNode {
        id: split.id,
        op: norm.op,
        inputs,
        output: *out_r,
    };
    let program = Program::WindowedRmsNorm { unit: !*gained };
    let k = mint(&synth, &module, &program, default_bindings(&synth))?;
    // `mint` already named the node `rmsnorm_s{split.id}` (the program stem + the
    // split's id, main's own naming law for the fused form).
    let name = k.func.name.to_string();
    let e = EmittedOp::new(name, k);
    Ok(vec![e])
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE PREFILL TAIL'S DATAFLOW WALK — main's own law, ported verbatim.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The ROW-LOCAL op set: ops whose output row `r` depends only on input row `r` (plus
/// weights and per-row tables). The prefill lm-head fold is sound only inside this set —
/// see [`tensor_feeds_result_through_row_local_ops`].
///
/// Pub for the spyre numeric gate, which closes its fold-law set with this same
/// classifier — every compared tid whose producer is row-local, consumes a
/// fold-family tensor and feeds the result rides the fold's one-row compare —
/// so the gate states the fold the program carries, never a re-derived set.
pub fn row_local_tail_op<F: scratchy_subtile::subtile_ir::RopeForm>(op: &SubOp<F>) -> bool {
    matches!(
        op,
        SubOp::MatmulTile { .. }
            | SubOp::SumReduce { .. }
            | SubOp::Elementwise(_)
            | SubOp::SiluMul
            | SubOp::ScalarMul { .. }
            | SubOp::RmsNorm { .. }
            | SubOp::RmsNormReduce { .. }
            | SubOp::RmsNormApply { .. }
            | SubOp::RmsNormUnit { .. }
            | SubOp::TanhSoftCap { .. }
            | SubOp::ScalarWeightMul
            // ⭐ AN ALL-GATHER IS ROW-LOCAL: row `r` of its output is row `r` of every rank's
            // block, side by side — a split vocabulary projection's tail passes through it.
            | SubOp::AllGather { .. }
    )
}

/// Does tensor `t` feed `ir.result` through ROW-LOCAL consumers only (or is it the result)?
///
/// Pub for the spyre numeric gate, which keys its result-riding fold law on this
/// same walk — the admission predicate itself, not a re-derived guess: the result
/// takes the fold's one-row compare iff a FOLDED node's output reaches it through
/// this walk, so nothing un-folded can ride in.
///
/// The prefill lm-head fold must fire ONLY on the result's own tail. The width test alone
/// (`cols == result_cols`) was falsified by the gemma-4 parity fixture (vocab 512): its
/// GLOBAL-layer q projection is also 512 cols (4 heads × hd 128), so the fold fired there
/// too — the q matmul ran at m=1, only row 0 of the roped Q was ever real, attention saw
/// zero rows past it, and the first sampled token was garbage while every width-128
/// tensor stayed fp16-clean. A width COLLISION is not an identity.
///
/// What separates the real tail: its rows die at the result (the host reads row 0 only),
/// and everything between it and the result is ROW-LOCAL — so computing one row and
/// placing it at row 0 loses nothing. The q projection fails that: its output passes
/// through a norm, a rope, and an ATTENTION (row-mixing — output row `i` reads K/V rows
/// `0..=i`), so every one of its rows is live and folding to one row corrupts the layer.
/// Rope is excluded from the row-local set deliberately: `RopeAppend` also writes the
/// paged KV cache, a consumer the prefill graph cannot see.
pub fn tensor_feeds_result_through_row_local_ops<F: scratchy_subtile::subtile_ir::RopeForm>(
    ir: &SubtileIR<F>,
    t: scratchy_subtile::subtile_ir::TensorId,
) -> bool {
    if t == ir.result {
        return true;
    }
    // Every consumer of `t` must be a row-local op whose own output keeps the property —
    // one row-mixing reader anywhere in the chain makes `t`'s other rows live and the
    // fold unsound.
    let mut any_reader = false;
    for n in &ir.nodes {
        if !n.inputs.iter().any(|i| i.tensor == t) {
            continue;
        }
        any_reader = true;
        if !row_local_tail_op(&n.op)
            || !tensor_feeds_result_through_row_local_ops(ir, n.output.tensor)
        {
            return false;
        }
    }
    any_reader
}

fn lower_one<
    T: scratchy_spyre_bundle::collective::Tp,
    F: scratchy_subtile::subtile_ir::RopeForm,
>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    rows_are_requests: bool,
) -> Result<EmittedOp<T>, String> {
    // ⛔ THE ROUTING IS `lower_one_node`'s EXHAUSTIVE MATCH, and a `Refused` row here
    // is its echo: the kinds that never route here carry one, and a node that reaches
    // this check means the routing changed without adding a family function — the
    // message names the owner. Unreachable through `lower_one_node` by construction
    // (its match is exhaustive over the same enum); this is a public fn, so the echo
    // exists for the direct caller.
    let row = row(&node.op);
    let TritonKernelRow::Spliced {
        src,
        file,
        entry,
        program,
    } = &row
    else {
        let TritonKernelRow::Refused(why) = &row else {
            unreachable!()
        };
        return Err(format!(
            "triton splice: {:?} — {why}. Routing it here means adding a family \
             function and a kernel",
            node.op
        ));
    };
    // ⛔ THE LM-HEAD TAIL'S FOLD IS `lower_all`'s now — the vocab-wide m>1 matmul is
    // rewritten THERE (answer-row extraction + the re-lowered m=S matmul), so a
    // MatmulTile reaching THIS one-op body as the tail means the one-op `lower`
    // was called where the walk's `lower_all` belongs. The test is the FULL
    // [`is_prefill_lm_head_tail`] — width, m>1, rows-not-requests, AND the
    // row-local dataflow walk — because the echo must not fire on a width
    // COLLISION the walk already refused (the q-projection law: a collision
    // matmul that feeds the result through rope lowers through here normally,
    // un-folded, every row kept).
    //
    if matches!(node.op, SubOp::MatmulTile { .. })
        && is_prefill_lm_head_tail(node, ir, rows_are_requests)
    {
        let rows = node.output.region.rows.len;
        return Err(format!(
            "triton splice: matmul_s{} is the prefill lm-head tail (vocab-wide, m={rows}, \
             rows not requests) — the fold is `lower_all`'s; call it, not the one-op \
             `lower`",
            node.id.index()
        ));
    }
    // ⛔⛔⛔ A NODE WHOSE REGIONS ARE NOT WHOLE TENSORS IS A SHAPE ONLY A KERNEL THAT
    // STATES ITS CORNER CAN SPELL. The measured instance was the front end's COLUMN
    // CHUNKING of a wide pointwise op (`n_blocks(out_cols, nb)`; production `nb =
    // 8192`): each chunk is a column window of one tensor, at its own corner.
    // MEASURED, granite-3.1-8b fp8 on card: the 12800-wide
    // MLP intermediate is TWO chunks (0..8192, 8192..12800), and a kernel stating
    // corner 0 read and wrote the FIRST chunk's columns — fluent garbage out, on a
    // divergence the whole-region golden could not see because every fixture is
    // whole-region. 2b passed only because its intermediate is 8192 = exactly one
    // block.
    //
    // ⭐ THE POINTWISE FAMILY NOW SPELLS IT: `elementwise.py` / `silumul.py` /
    // `scalarmul.py` state `N_TOTAL` (the tensor's storage width, named by the
    // descriptor's shape/strides) and `C_START` (the region's column corner, named
    // by the load/store offsets). AND THE MATMUL FAMILY SPELLS ITS OWN ROW-0 WINDOW: `matmul.py` /
    // `matmul_fp8.py` state `M_TOTAL` (the output tensor's row extent) with the
    // `[M, N]` tile at row 0 — the prefill lm-head fold's m=S tail, whose activation
    // is the LAST_HIDDEN synthetic and whose output is the `[S, vocab]` prefix of
    // the `[mq, vocab]` logits storage. Every OTHER row states ONE whole-tensor tile at corner 0 and
    // cannot name a window, so the refusal stands for it (rope regions are whole in
    // production today), and this guard reads the REGION, not the op kind, so a
    // front-end change that windows any other op's regions reproduces the same
    // refusal with the kernel row named — a windowed kernel is the fix shape.
    // ⭐ ROPE STATES ITS OWN ROW WINDOW — `R_START` bakes the region's row corner into
    // every access tile and `Y_TOTAL` the tensor's full re-tiled height, so the
    // per-position decomposition's windowed regions ARE spellable. The law (whole
    // columns on all four regions, ONE shared row window, equal tensor heights) is
    // stated where the facts are read, in the rope arm of `kernel_spec` below — which
    // is stricter than either branch here, so the generic guard stands down for it.
    if !matches!(
        &node.op,
        SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. }
    ) {
        let states_the_corner = matches!(
            &node.op,
            SubOp::SiluMul
                | SubOp::ScalarMul { .. }
                | SubOp::MatmulTile { .. }
                | SubOp::Elementwise(
                    EwKind::Silu | EwKind::Gelu | EwKind::Add | EwKind::Mul | EwKind::Sub,
                )
        );
        // ⭐ THE TANH-SOFTCAP STATES NEITHER CORNER — the softcap kernel's view IS the
        // region (`[M, N]`, tiles from row 0 / column 0), so BOTH corners must be 0 while
        // the extents are free: the prefill lm-head fold re-lowers the vocab-wide softcap
        // at m=S (the `[S, vocab]` prefix of the `[mq, vocab]` logits storage, a legal
        // prefix view), and the front end's chunker never tiles this op (its chunk set is
        // the elementwise/scalarmul/silumul family), so a column window never arises.
        let states_neither_corner = matches!(&node.op, SubOp::TanhSoftCap { .. });
        if states_neither_corner {
            let corner0 = |tr: &scratchy_subtile::subtile_ir::TensorRegion| {
                tr.region.rows.start == 0 && tr.region.cols.start == 0
            };
            if !corner0(&node.output) || node.inputs.iter().any(|tr| !corner0(tr)) {
                return Err(format!(
                    "triton splice: {} t{} has a nonzero region corner (row {} / column {}) — \
                     the softcap kernel's view IS the region with tiles from the origin, and no \
                     constant carries a corner; a windowed softcap is a kernel-family worklist \
                     item",
                    program_stem(node, program),
                    node.output.tensor.index(),
                    node.output.region.rows.start,
                    node.output.region.cols.start,
                ));
            }
        } else if !states_the_corner {
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
                    program_stem(node, program),
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
            let row0 = |tr: &scratchy_subtile::subtile_ir::TensorRegion| tr.region.rows.start == 0;
            if !row0(&node.output) || node.inputs.iter().any(|tr| !row0(tr)) {
                return Err(format!(
                    "triton splice: {} t{} has a nonzero ROW corner — the pointwise kernels \
                     load row 0 (`start_m * BLOCK_M` at a [1] grid); a row-windowed kernel is the \
                     prefill-fold worklist item",
                    program_stem(node, program),
                    node.output.tensor.index()
                ));
            }
        }
    }

    // ⛔ THE ARITY IS THE NODE'S OWN CONTRACT, stated once per op kind.
    // ⭐ ROPE'S ARITY IS 3, NOT THE NODE'S INPUT COUNT: the rotation reads only
    // `inputs[0..3]` (x, cos, sin) — a `RopeAppend` carries 6 inputs but its V and
    // KV-cache destinations flow through GRAPH edges, not through the op — so the
    // splice binds the first three operands, and the check below is `>= 3`.
    let arity = match &node.op {
        SubOp::RmsNorm { .. } => 2,
        SubOp::SiluMul => 2,
        // ⭐ THE MATMUL'S ARITY IS THE WEIGHT SCHEME'S OWN: Dense is arity-2
        // `[act, weight]`, Fp8Dynamic is arity-3 `[act, weight_fp8, weight_scale]`. The
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
        // ⭐ SCALAR-WEIGHT MUL IS ARITY-2 (`[act, weight]`). The MoE sub-ops have NO arity here any more: the registry
        // refuses them before the splice — the MoE lowers only from
        // `kernels/moe.py` through KTIR.
        SubOp::ScalarWeightMul => 2,
        SubOp::TanhSoftCap { .. } | SubOp::RmsNormUnit { .. } | SubOp::Reshape { .. } => 1,
        // ⛔ NO `_` ARM. A spliced kind is a row above, and a row without an arity here is
        // an unreachable — the same discipline `lower_one_node`'s match holds.
        _ => {
            return Err(format!(
                "triton splice: {file} has a registry row but no arity — the row is incomplete"
            ));
        }
    };
    if node.inputs.len() < arity {
        return Err(format!(
            "triton splice: {file} t{} expects at least {} operand(s), found {}",
            node.output.tensor.index(),
            arity,
            node.inputs.len()
        ));
    }
    let spec = kernel_spec(node, ir, file, entry)?;
    let m = compile_kernel(src, &spec, &grid(node, ir)?)?;
    let k = mint(node, &m, program, default_bindings(node))?;
    let name = format!("{}_s{}", program_stem(node, program), node.id.index());
    let e = EmittedOp::new(name, k);
    Ok(e)
}

/// ⭐ THE PREFILL LM-HEAD FOLD, with the Triton `lmlast.py` kernel as the extraction's
/// body. The vocab-wide lm_head cannot
/// run at m>1 (it time-tiles; per-row time-tiling is unimplemented), and only each
/// request's LAST prompt token's logits are ever read, so the tail is TWO ops:
///
/// 1. THE EXTRACTION — `lmlast_fwd` copies each request window's answer row (its
///    `r·w + selector_lastrow_col(w)`, the [`AnswerRows`] law minted from the rung's
///    own extent pair) of the `[mq, hidden]` activation into the reserved `[S, hidden]`
///    `LAST_HIDDEN_TID` staging, one single-stick tile per `(request, stick-group)`
///    (the only representable form: the source buffer is stick-major, so a wider
///    window over the row is not expressible).
/// 2. THE MATMUL — the SAME node at m=S, reading the staging's whole `[S, hidden]`
///    and writing the `[S, vocab]` prefix of the `[mq, vocab]` logits storage
///    (request r's logits row is row r), through the ordinary matmul row
///    (`lower_one`, `M_TOTAL` naming the storage while the store takes the `[S, n]`
///    tile at row 0). At `S = 1` the tail is exactly the shape the PROVEN decode
///    path lowers, fp8 chain included — byte-identical, the same one-row window.
///
/// ⛔ THE ROWS ARE THE EXTENT PAIR'S OWN, STATED ONCE — not `rows.start + mq - 1`:
/// the extraction addresses the buffer the tape takes to be exactly `[mq, hidden]`
/// at row 0, which is the only shape that reaches the fold (a non-whole activation
/// region is refused by the whole-region guard before this), and the pair's held
/// product must BE that `mq` — the same agreement check `lower_attn` runs, so a mint
/// that does not describe the graph being lowered is a refusal by name, never a
/// silently-wrong row. At `S = 1` (`w = mq`) request 0's row is
/// `selector_lastrow_col(mq)`, exactly the single-row law, and every tile, corner and
/// constexpr the fold states is byte-identical to it.
///
/// ⛔ THE COUNT IS THE RUNG'S CAPACITY, NOT THE LAUNCH'S LIVE REQUESTS. The splice
/// extracts `S` rows on every launch of the rung — including the empty windows a
/// launch seated fewer requests than the rung packs leaves. An empty window's hidden
/// is a PAD row's: the copy is harmless, and the row is never read (the readback
/// takes `rows().take(live)` with the live count off the launch's own slot list).
fn lower_prefill_lm_head_fold<
    T: scratchy_spyre_bundle::collective::Tp,
    F: scratchy_subtile::subtile_ir::RopeForm,
>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    query_rows: scratchy_spyre_bundle::rung::QueryExtent,
) -> Result<Vec<EmittedOp<T>>, String> {
    use scratchy_subtile::subtile_ir::Region as SubRegion;

    let a = &node.inputs[0];
    let (mq, hidden) = (a.region.rows.len, a.region.cols.len);
    let stk = scratchy_spyre_bundle::rung::FP16_LANES; // the tail copies whole fp16 sticks
    if hidden % stk != 0 {
        return Err(format!(
            "triton splice: matmul_s{} (prefill lm-head tail): hidden={hidden} is not a whole \
             {stk}-fp16 stick, so the last prompt row is not a run of whole stick-groups — the \
             per-stick extraction cannot address it",
            node.id.index()
        ));
    }
    // ⭐ THE PAIR MEETS THE GRAPH HERE, ONCE — the same check `lower_attn` runs: the
    // mint's held product must be this graph's own row count, or the mint describes a
    // different rung than the one being lowered. A disagreement is a REFUSAL by name,
    // never a silently-wrong answer row.
    if query_rows.graph_rows().get() != mq {
        return Err(format!(
            "triton splice: matmul_s{} (prefill lm-head tail): the graph's {} row(s) are not the \
             extent pair's {} (per-request {} × {} request(s)) — the mint does not describe the \
             graph it lowers",
            node.id.index(),
            mq,
            query_rows.graph_rows().get(),
            query_rows.per_request().get(),
            query_rows.requests().get(),
        ));
    }
    let answers = scratchy_spyre_bundle::rung::AnswerRows::of_extent(query_rows);
    let (w, s) = (
        query_rows.per_request().get(),
        u32::try_from(answers.count())
            .map_err(|_| "triton splice: the rung's request capacity overflows u32".to_string())?,
    );
    let last_hidden = scratchy_spyre_bundle::reserved_tids::LAST_HIDDEN_TID;
    // The bake-log proof the fold fired: every prefill rung's bake names its
    // own LAST_HIDDEN extraction — the op, the rows it takes, and the staging
    // it lands in. A rung whose bake lacks this line folded nothing. The count
    // stated is the rung's CAPACITY: a launch seated fewer requests still
    // extracts S rows, and the empty windows' rows are pad rows, never read.
    eprintln!(
        "[spyre-superdsc] prefill lm-head tail folded — lmlast_s{} extracts {} answer row(s) \
         (request r's at r·{} + {}) of {} (hidden {}) into LAST_HIDDEN t{}",
        node.id.index(),
        s,
        w,
        scratchy_spyre_bundle::rung::selector_lastrow_col(w as usize) as u32,
        mq,
        hidden,
        last_hidden
    );

    // ── Half 1: the extraction, through the lmlast kernel. ──
    let src = match lmlast_row() {
        TritonKernelRow::Spliced { src, .. } => src,
        TritonKernelRow::Refused(why) => return Err(why.to_string()),
    };
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
    ce("W", Val::Int(i128::from(w)))?;
    ce("S", Val::Int(i128::from(s)))?;
    ce("N_STICKS", Val::Int(i128::from(hidden / stk)))?;
    let spec = KernelSpec {
        file: "lmlast.py".to_string(),
        kernel: "lmlast_fwd".to_string(),
        signature: signature.clone(),
        constexprs: constexprs.clone(),
    };
    let module = compile_kernel(src, &spec, &[1])?;
    let extraction_node = SubtileNode {
        id: node.id,
        op: node.op,
        inputs: vec![*a],
        output: scratchy_subtile::subtile_ir::TensorRegion {
            tensor: scratchy_subtile::subtile_ir::TensorId::from_index(last_hidden as usize),
            region: SubRegion {
                rows: scratchy_subtile::subtile_ir::Range::new(0, s),
                cols: scratchy_subtile::subtile_ir::Range::new(0, hidden),
            },
        },
    };
    let lmlast_program = match lmlast_row() {
        TritonKernelRow::Spliced { program, .. } => program,
        TritonKernelRow::Refused(why) => return Err(why.to_string()),
    };
    let mut k = mint(
        &extraction_node,
        &module,
        &lmlast_program,
        default_bindings(&extraction_node),
    )?;
    let xname = format!("lmlast_s{}", node.id.index());
    k.func.name = Arena::global().str(xname.clone());
    let extract = EmittedOp::new(xname, k);

    // ── Half 2: the SAME node at m=S, reading the staging's whole [S, hidden] ──
    let mut at_ms = node.clone();
    let s_rows = |tr: &scratchy_subtile::subtile_ir::TensorRegion| {
        scratchy_subtile::subtile_ir::TensorRegion {
            tensor: tr.tensor,
            region: SubRegion {
                rows: scratchy_subtile::subtile_ir::Range::new(tr.region.rows.start, s),
                cols: tr.region.cols,
            },
        }
    };
    at_ms.inputs[0] = scratchy_subtile::subtile_ir::TensorRegion {
        tensor: scratchy_subtile::subtile_ir::TensorId::from_index(last_hidden as usize),
        region: SubRegion {
            rows: scratchy_subtile::subtile_ir::Range::new(0, s),
            cols: scratchy_subtile::subtile_ir::Range::new(0, hidden),
        },
    };
    if let Some(w) = at_ms.inputs.get_mut(1) {
        *w = s_rows(w);
    }
    at_ms.output = s_rows(&at_ms.output);
    let matmul = lower_one(&at_ms, ir, false)?;

    Ok(vec![extract, matmul])
}

/// THE BLOCK COPY'S PRODUCER — `dst = src` for a whole source (a graph tensor, or the receive
/// window beyond the graph) and a destination region that is a column window `[0, R) × [off,
/// off + C_SRC)` of a `dst_cols`-wide tensor: one spliced `blockcopy.py` program named `name`.
/// `node` is the collective the copy realizes — its id and op key the program like every other.
pub fn lower_block_copy<
    T: scratchy_spyre_bundle::collective::Tp,
    F: scratchy_subtile::subtile_ir::RopeForm,
>(
    node: &SubtileNode<F>,
    src: scratchy_subtile::subtile_ir::TensorRegion,
    dst: scratchy_subtile::subtile_ir::TensorRegion,
    dst_cols: u32,
    name: &str,
) -> Result<EmittedOp<T>, String> {
    let rows = src.region.rows.len;
    let c_src = src.region.cols.len;
    let c_dst = dst_cols;
    let off = dst.region.cols.start;
    let (src_py, file, whole_entry) = match blockcopy_row() {
        TritonKernelRow::Spliced {
            src, file, entry, ..
        } => (src, file, entry),
        TritonKernelRow::Refused(why) => return Err(why.to_string()),
    };
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    for p in ["desc_src", "desc_dst"] {
        signature.insert(
            p.to_string(),
            ArgSpec::parse("*fp16").map_err(|e| e.to_string())?,
        );
    }
    // The row blocks: the elementwise LX law over the copy's one live tile, so the program is
    // `O(rows / BLOCK_M)` straight-line copies at any width (`blockcopy.py`'s own doc says why).
    // The row count is not a constexpr: the kernel states it as `N_BLOCKS · BLOCK_M (+ TAIL_H)`.
    // A copy with no tail takes the whole-block entry, which has no `TAIL_H` to state.
    let (block_m, n_blocks, tail_h) = blocks_of(rows, c_src, BLOCK_COPY_LIVE_TILES);
    let tail = std::num::NonZeroU32::new(tail_h);
    let mut stated = vec![
        ("C_SRC", c_src),
        ("C_DST", c_dst),
        ("COL_OFF", off),
        ("BLOCK_M", block_m),
        ("N_BLOCKS", n_blocks),
    ];
    let entry = match tail {
        None => whole_entry,
        Some(t) => {
            stated.push(("TAIL_H", t.get()));
            BLOCKCOPY_TAIL_ENTRY
        }
    };
    for (k, v) in stated {
        signature.insert(k.to_string(), ArgSpec::Constexpr);
        constexprs.insert(k.to_string(), Val::Int(i128::from(v)));
    }
    let spec = KernelSpec {
        file: file.to_string(),
        kernel: entry.to_string(),
        signature,
        constexprs,
    };
    let module = compile_kernel(src_py, &spec, &[1])?;
    let copy = SubtileNode {
        id: node.id,
        op: node.op,
        inputs: vec![src],
        output: dst,
    };
    let mut k = mint(&copy, &module, &Program::BlockCopy, default_bindings(&copy))?;
    k.func.name = Arena::global().str(name.to_string());
    let e = EmittedOp::new(name.to_string(), k);
    Ok(e)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  PAGED ATTENTION — spyre-inference's serving step (`SpyreHeadMajorAttentionImpl`, the decoder
//  backend its default `SPYRE_ATTN_KV_LAYOUT=head_major` selects): the KV cache write
//  (`reshape_and_cache_head_major`) and the online-softmax page walk
//  (`page_attn_head_major_decode` / `_prefill`), as `kernels/reshape_and_cache_head_major.py`,
//  `kernels/page_attn_head_major_decode.py` and `kernels/page_attn_head_major_prefill.py` —
//  their kernels, operand for operand.
//
//  * the K and V CACHE TENSORS the layout names ARE THE POOLS, read head-major as
//    `[pages, nkvh, SchedulerBlockTokens::REFERENCE, hd]` — the same bytes as the cache's `[cap, nkvh·hd]`, so the
//    graph's own tensors carry the pages and nothing is re-allocated;
//  * a PAGE is the scheduler's KV block ([`SchedulerBlockTokens::REFERENCE`] slots, the block size the worker hands
//    the engine), so a block id the scheduler allocates IS a page of these pools;
//  * the PAGE INDEX TABLE, the MASK STACK, the QUERY ROW INDEX, `kv_row_pool` and the KV WRITE
//    INDEX are the step's host sources — tensors at RESERVED tids ([`PagedStepTids`]), filled from
//    the request's blocks (spyre-inference's `SpyreAttentionMetadataBuilder.build` and
//    `kv_write_index`);
//  * the write runs BEFORE the attention — the `RopeAppend` precedes its `AttnDecode` on the tape,
//    as `do_kv_cache_update` precedes the page walk — so a step's own keys are in the pool when its
//    pages are gathered, and the attention reads nothing but the pools.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Token slots per KV page — the scheduler's block, the block the page walk tiles (`BLOCK_SIZE`) and
/// the modulus the write's rows step by.
///
/// ⭐ spyre-inference's OWN block size, 128: `_DEFAULT_BLOCK_SIZE` (`spyre_inference/platform.py:146`),
/// which `_align_block_size` only ever aligns a user's size up to, and which every torch-spyre serve
/// runs (its compiled decode walk steps the page table and mask by 128 a trip, `affine_map<(d0)[s0]
/// -> (s0 + 128*d0)>`). Half the trips of a 64-slot page for the same context.
///
/// A whole page gathered as ONE index entry (the prompt kernel's `k_pages.index_select(0,
/// page_idx)`, every kv head) — 128 KB at 8 kv heads × 64 head dims, 256 KB at 128 — is gathered in
/// one op, as torch-spyre's compiled prefill gathers it (`index_select` in its
/// `codegen/compute_ops.py`).
///
/// ⭐ THE TWO PAGE QUANTITIES, AS TYPES — re-exported from their one home in
/// the bundle crate's `rung` (the law that relates them, [`PoolPageSlots::of_scheduler_blocks`],
/// needs both types in one scope, and both this crate and the bake depend on the bundle crate,
/// so that is the one home both can see):
/// [`SchedulerBlockTokens`] is the block size of the serving step we port — a TYPED
/// REFERENCE INPUT, NOT A DERIVATION, minted ONCE at
/// [`SchedulerBlockTokens::REFERENCE`]. The reference's own law for the value — a
/// multiple of 64 (`_BLOCK_SIZE_MULTIPLE`, the 128-byte fp16 stick), measured-throughput
/// argmax — is cited, not re-derived: we port the step, we do not redesign its block.
/// Every reader (the splice's walk and write, the worker, `required_block_size`) reads
/// that one value. The pool's page ([`PagedKvPool::POOL_PAGE_SLOTS`], a
/// [`PoolPageSlots`]) is a DIFFERENT quantity: before they were types, a plain integer
/// accepted either. Integer boundaries call `.get()`.
pub use scratchy_spyre_bundle::rung::{HeadWidth, PoolPageSlots, SchedulerBlockTokens};

/// An int32 stick: the generation's stick bytes over int32's four (spyre-inference's
/// `INT32_ELEMS_PER_STICK`).
pub const INT32_STICK: u32 = Df::UInt32.lanes::<ktir_core::ddl_binds::Baked>().get() as u32;

/// ⭐ WHAT A PAGED BUNDLE'S ROWS ARE — the fact spyre-inference's dispatch reads besides the query
/// width: one sequence's query rows, or one decode row of each sequence of a batch
/// (`SPYRE_BATCHED_DECODE=1`, their default). The bake knows it (a decode rung baked wider than one
/// row carries one token of each running request); the graph does not, so it is handed in, typed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PagedRows {
    /// One sequence's rows: `page_attn_head_major_decode` at one row, `_prefill` past it.
    Sequence,
    /// One decode row per sequence: `batched_decode_head_major`.
    Batch,
}

impl PagedRows {
    /// The bake's `rows_are_requests`: a decode rung whose rows are separate requests is a batch.
    pub const fn of_requests(rows_are_requests: bool) -> PagedRows {
        if rows_are_requests {
            PagedRows::Batch
        } else {
            PagedRows::Sequence
        }
    }
}

/// A decode batch's walk: its sequences, the blocks each walks, and the pages the pool holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeWalk {
    pub seqs: u32,
    pub blocks: u32,
    pub pages: u32,
}

/// A decode batch's heads: the kv heads a page holds, the query row's width (`num_heads ·
/// head_size`), and one head's width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeHeads {
    pub kv_heads: u32,
    pub q_width: u32,
    pub head_size: u32,
}

/// ⭐ A DECODE BATCH'S CHUNKING — spyre-inference's `batched_decode_chunking`
/// (`spyre_attn_bucketer.py`, 297fc48) over a batch of `seqs` sequences walking `blocks` blocks:
/// `blocks_per_chunk` block slots per chunk, so a chunk gathers `entries = blocks_per_chunk · seqs`
/// pages (their target: one per core), and `num_chunks` chunks cover the walk.
///
/// Triton's gather takes at least 8 indices (`semantic.py`'s `descriptor_gather`). A chunk of 8
/// entries or more gathers its pages whole, one index a page, as theirs does. A chunk of fewer
/// cannot, and gathers each page's kv heads as the folded cache's rows instead (`rows_per_page =
/// nkvh`, the per-sequence decode kernel's `kv_row_pool` form) — OURS, not theirs: their batched
/// kernel never folds, and measured folding "roughly 1.5-2x slower" (its docstring). Its gather is
/// `entries · nkvh` rows, and only a chunk under 8 of those takes more block slots than theirs: at
/// least `ceil(8 / (seqs · nkvh))`, the slots past a sequence's blocks gathering page 0 under a fully
/// masked tile, their own padding law ("padding columns gather page 0 under an all -inf mask and
/// contribute zero").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchChunking {
    pub seqs: u32,
    pub blocks_per_chunk: u32,
    pub num_chunks: u32,
    /// The index rows one page's gather takes: 1 (the whole page), or the kv heads (each head's
    /// block, the index tables folded by kv head).
    pub rows_per_page: u32,
}

impl BatchChunking {
    /// The cores a chunk's entries are sized to — the device's own (spyre-inference states the
    /// same count as `spyre_attn_bucketer._SPYRE_CORE_COUNT`).
    pub const CORES: u32 = <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::CORES;
    /// Triton's gather floor.
    pub const GATHER_FLOOR: u32 = 8;

    /// The most chunks one walk is cut into ([`MAX_WALK_TRIPS`]): a 32-sequence batch over a
    /// 2048-page pool is 64 chunks of 32 block slots, not 2048 chunks of one.
    pub const MAX_CHUNKS: u32 = MAX_WALK_TRIPS;

    /// One stick of fp16 lanes — a gathered row of the folded cache or the folded query is whole
    /// sticks of it, or the gather has no addressing for it.
    pub const STICK: u32 = scratchy_spyre_bundle::rung::FP16_LANES;

    /// The fold's page-side law, stated rather than relied on: the cut is `kv_heads · 2^i` rows a
    /// page with `2^i < GATHER_FLOOR`, so a page's share is `page_slots · head_size / 2^i`, whole
    /// rows of a one-stick head exactly when the floor divides the page's slots. The scheduler's
    /// block is a const, so this is a build fact.
    const PAGE_TAKES_THE_CUT: () = assert!(
        SchedulerBlockTokens::REFERENCE
            .get()
            .is_multiple_of(Self::GATHER_FLOOR),
        "a page's slots must divide by the gather floor, or a fold cut of a page is not whole rows"
    );

    /// ⛔ `seqs` AND `nkvh` ARE COUNTS, ONE OR MORE. Both divide here, so a zero is a panic at BAKE
    /// and not a wrong answer — `seqs` has divided since this function existed (`CORES / seqs`) and
    /// `nkvh` since the fold was added (`GATHER_FLOOR.div_ceil(seqs · rows_per_page)`). Every
    /// producer already has them non-zero (`nkvh` is the attention facts' class head count, minted
    /// at the bridge off `ModelFacts` — a non-zero `kv_heads` is a law of the facts' own mint —
    /// and a bundle's rows are its baked width), so the clamps below are not a tolerance for zero:
    /// they keep this `const fn` TOTAL, so a malformed step is a wrong chunking a test can catch
    /// rather than a build that dies inside a `const` evaluation. The honest end state is a
    /// `NonZeroU32` pair in the signature; that waits on `PagedStep`'s fields, which the bake's
    /// codegen quotes as `u32`.
    ///
    /// The fold also reads the query's own row width, `q_width = num_heads · head_size`, and the
    /// head's: a page and a query row split into `rows_per_page` rows each, and every one of those
    /// rows must be whole sticks of what the landing reads back.
    ///
    /// Under the gather floor, the fold cuts each page (and each query row) into as many rows as
    /// reach the floor with THEIR block slots, so a chunk under the floor takes no masked block
    /// slots past theirs: the rows are `kv_heads · 2^i`, the smallest that reaches the floor and
    /// keeps a query row's share `q_width / rows` whole sticks. A page's share needs no test of its
    /// own: it is `block_size · head_size / 2^i` for a cut of `2^i` rows a kv head, and the cut
    /// never passes the floor's 8, so the share is at least 16 whole sticks. Where no such cut
    /// reaches the floor, the widest whole-stick cut is taken and the block slots make up the rest
    /// as before — the law stays total. At `kv_heads` rows a page this is the kv-head fold exactly;
    /// its cut is a cut of the same bytes into finer equal rows.
    ///
    /// A walk past [`Self::MAX_CHUNKS`] chunks takes more block slots a chunk, but never as many
    /// pages a chunk as the `pages`-page pool holds (spyre-inference's eligibility: a gather of the
    /// whole pool faults their device, torch-spyre#4033) — past that it takes more chunks.
    ///
    /// Nor more pages a chunk than one access reaches ([`DeviceGen::MUTABLE_REACH_BYTES`]): a
    /// chunk's pages land as one fp16 gather, and a gather runs on one core as one access, so a
    /// chunk whose landed pages pass the reach is one dxp refuses ("EAR overflow detected";
    /// gemma-4-12b's 32-sequence decode landed 1024 pages of 512 KiB a chunk, twice the reach) — it
    /// takes more chunks instead, past [`Self::MAX_CHUNKS`] if it must. The gather floor's few pages
    /// stay under the reach at any page a model has.
    ///
    /// [`DeviceGen::MUTABLE_REACH_BYTES`]: ktir_core::ddl_binds::DeviceGen::MUTABLE_REACH_BYTES
    pub const fn of(walk: DecodeWalk, heads: DecodeHeads) -> BatchChunking {
        let () = Self::PAGE_TAKES_THE_CUT;
        let DecodeWalk {
            seqs,
            blocks,
            pages,
        } = walk;
        let DecodeHeads {
            kv_heads: nkvh,
            q_width,
            head_size,
        } = heads;
        let seqs = if seqs < 1 { 1 } else { seqs };
        let theirs = {
            let j = Self::CORES / seqs;
            let j = if j < blocks { j } else { blocks };
            if j < 1 { 1 } else { j }
        };
        let rows_per_page = if theirs * seqs < Self::GATHER_FLOOR {
            let mut rows = if nkvh < 1 { 1 } else { nkvh };
            // Double the cut while it is short of the floor at their block slots and the doubled
            // query share is still whole sticks — and only for a head one stick wide: a kv head's
            // page is then its slots' rows one after another, so a finer cut is whole rows of it,
            // and the landing reads those rows back to back as the page (the gather's one-stick
            // frame law). Two sticks or wider, a page's sticks interleave its rows, and a sub-page
            // entry is no block of it.
            // `rows < GATHER_FLOOR` bounds it: every doubling stays below twice the floor.
            while head_size == Self::STICK
                && rows < Self::GATHER_FLOOR
                && theirs * seqs * rows < Self::GATHER_FLOOR
                && q_width.is_multiple_of(rows * 2)
                && (q_width / (rows * 2)).is_multiple_of(Self::STICK)
                && q_width / (rows * 2) > 0
            {
                rows *= 2;
            }
            rows
        } else {
            1
        };
        let floor = Self::GATHER_FLOOR.div_ceil(seqs * rows_per_page);
        let bounded = blocks.div_ceil(Self::MAX_CHUNKS);
        let under_pool = if pages > seqs { (pages - 1) / seqs } else { 1 };
        let bounded = if bounded < under_pool {
            bounded
        } else {
            under_pool
        };
        let blocks_per_chunk = if theirs < floor { floor } else { theirs };
        let blocks_per_chunk = if blocks_per_chunk < bounded {
            bounded
        } else {
            blocks_per_chunk
        };
        let landed = SchedulerBlockTokens::REFERENCE.get() as u64
            * (if nkvh < 1 { 1 } else { nkvh }) as u64
            * head_size as u64
            * Df::Fp16.word_bytes().get()
            * seqs as u64;
        let reach =
            <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::MUTABLE_REACH_BYTES
                / landed;
        let blocks_per_chunk = if (blocks_per_chunk as u64) > reach {
            reach as u32
        } else {
            blocks_per_chunk
        };
        let blocks_per_chunk = if blocks_per_chunk < floor {
            floor
        } else {
            blocks_per_chunk
        };
        BatchChunking {
            seqs,
            blocks_per_chunk,
            num_chunks: blocks.div_ceil(blocks_per_chunk),
            rows_per_page,
        }
    }

    /// Pages one chunk gathers.
    pub const fn entries(self) -> u32 {
        self.blocks_per_chunk * self.seqs
    }

    /// Index rows one chunk gathers — `entries · rows_per_page`, the words of each chunk's page ids
    /// and of `rep_row_ids`.
    pub const fn index_rows(self) -> u32 {
        self.entries() * self.rows_per_page
    }
}

#[cfg(test)]
mod chunking {
    use super::{BatchChunking, DecodeHeads, DecodeWalk, SchedulerBlockTokens};
    use ktir_core::ddl_binds::{Baked, DeviceGen};

    /// Every chunk lands within one access's reach and the chunks still cover the walk, at every
    /// batch width, at head dims and kv heads past any model's, up to a 4096-block walk.
    #[test]
    fn a_chunk_lands_within_one_access_and_the_chunks_cover_the_walk() {
        for seqs in 1..=32u32 {
            for (nkvh, hd) in [
                (1u32, 64u32),
                (8, 64),
                (8, 128),
                (8, 256),
                (1, 512),
                (8, 512),
                (16, 512),
            ] {
                for blocks in [1u32, 7, 64, 1024, 2048, 4096] {
                    let c = BatchChunking::of(
                        DecodeWalk {
                            seqs,
                            blocks,
                            pages: 8192,
                        },
                        DecodeHeads {
                            kv_heads: nkvh,
                            q_width: 32 * hd,
                            head_size: hd,
                        },
                    );
                    let landed = u64::from(c.entries())
                        * u64::from(SchedulerBlockTokens::REFERENCE.get() * nkvh * hd)
                        * 2;
                    assert!(
                        landed <= <Baked as DeviceGen>::MUTABLE_REACH_BYTES,
                        "{seqs} seqs, {nkvh}x{hd}, {blocks} blocks: {c:?} lands {landed} B"
                    );
                    assert!(
                        c.blocks_per_chunk * c.num_chunks >= blocks,
                        "{c:?} over {blocks}"
                    );
                }
            }
        }
        // The decode batch dxp refused: 32 sequences of 512 KiB pages over a 2048-block walk.
        let c = BatchChunking::of(
            DecodeWalk {
                seqs: 32,
                blocks: 2048,
                pages: 8192,
            },
            DecodeHeads {
                kv_heads: 8,
                q_width: 16 * 256,
                head_size: 256,
            },
        );
        assert_eq!((c.blocks_per_chunk, c.num_chunks), (16, 128));
    }
}

/// ⭐⭐⭐ THE PAGED STEP'S HOST SOURCES — the buffers a paged forward's attention and KV write read
/// besides the pools, at their RESERVED tids (`scratchy_spyre_bundle::reserved_tids::PAGED_STEP_BASE`): one
/// session holds every bundle's programs, so a source numbered past ONE graph can be a real tensor of
/// another, and the reserved region names the same ids for every bundle. The splice that binds them
/// and the wiring that exposes them read them here. Each is spyre-inference's own operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PagedStepTids {
    /// `mask_stack` `[num_blocks, padded_query_len, block_size]`: one additive tile per walked block.
    pub mask: u32,
    /// `page_index_table` `[num_blocks, INT32_ELEMS_PER_STICK]` int32: row i holds the i-th block's
    /// page id at column 0.
    pub page_table: u32,
    /// `kv_write_index`: one `[T]` int32 index per kv head, one after another — the cache row of
    /// each token of each head.
    pub write_rows: u32,
    /// `query_row_index` `[padded_query_len]` int32: the sequence's rows of the query.
    pub query_rows: u32,
    /// A decode batch's `rep_row_ids` `[entries · rows_per_page]` int32 ([`BatchChunking`]): entry
    /// `j · seqs + s`'s row `a` holds `row · rows_per_page + a` of sequence `s`'s query row.
    pub rep_rows: u32,
    /// A decode batch's `chunk_page_ids` `[num_chunks, entries · rows_per_page]` int32: chunk `c`'s
    /// entry `j · seqs + s`'s row `a` holds `page · rows_per_page + a` of block slot `j` of sequence
    /// `s` (at `rows_per_page = num_kv_heads`, our fold under Triton's gather floor: [`BatchChunking`]).
    pub chunk_pages: u32,
    /// A decode batch's `mask_by_chunk` `[num_chunks, blocks_per_chunk, seqs, 1, 1, block_size]`:
    /// one row per block slot and sequence, broadcast over the kv heads in the kernel.
    pub chunk_masks: u32,
}

impl PagedStepTids {
    /// ⭐ ATTENTION CLASS `class`'s TABLES — each class's step its own, since its kv heads, head dim
    /// and mask are its own: the full class's at the `paged_step` sentinels, a hybrid model's sliding
    /// class's at its own (`reserved_tids::paged_class_tid`).
    pub const fn of(class: scratchy_ir::model_facts::AttnClass) -> PagedStepTids {
        use scratchy_spyre_bundle::reserved_tids::paged_class_tid;
        let c = paged_class(class);
        PagedStepTids {
            mask: paged_class_tid(c, 0),
            page_table: paged_class_tid(c, 1),
            write_rows: paged_class_tid(c, 2),
            query_rows: paged_class_tid(c, 3),
            rep_rows: paged_class_tid(c, 5),
            chunk_pages: paged_class_tid(c, 6),
            chunk_masks: paged_class_tid(c, 7),
        }
    }
}

/// The reserved-tid index of attention class `class`'s paged step — the full class first, so a
/// uniform model's step keeps the `paged_step` sentinels.
pub const fn paged_class(class: scratchy_ir::model_facts::AttnClass) -> u32 {
    match class {
        scratchy_ir::model_facts::AttnClass::Full => 0,
        scratchy_ir::model_facts::AttnClass::Sliding => 1,
    }
}

/// THE ATTENTION FAMILY's prefill kernel — `page_attn_head_major_prefill.py`, spyre-inference's
/// `page_attn_head_major_prefill_kernel`. [`lower_attn`] dispatches as spyre-inference does: this
/// kernel past one query row, [`attn_decode_row`]'s at one.
pub fn attn_row() -> TritonKernelRow {
    TritonKernelRow::Spliced {
        src: KvPool::BUILD.pick(
            include_str!("../../../../kernels/page_attn_head_major_prefill.py"),
            include_str!("../../../../kernels/page_attn_head_major_prefill_fp8.py"),
        ),
        file: KvPool::BUILD.pick(
            "page_attn_head_major_prefill.py",
            "page_attn_head_major_prefill_fp8.py",
        ),
        entry: KvPool::BUILD.pick(
            "page_attn_head_major_prefill_kernel",
            "page_attn_head_major_prefill_fp8_kernel",
        ),
        program: Program::PagedAttn,
    }
}

/// ⭐ THE KV POOL'S ELEMENT FORMAT — one per build (`kv-fp8`), the format every paged kernel writes
/// and reads its pages in. The fp8 pool halves the cache: its writer narrows each row to E4M3, and
/// every walk widens its pages back to f16 before their first use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvPool {
    Fp16,
    Fp8,
}

impl KvPool {
    /// This build's pool.
    pub const BUILD: KvPool = if cfg!(feature = "kv-fp8") {
        KvPool::Fp8
    } else {
        KvPool::Fp16
    };

    /// This pool's half of a paged kernel's pair — spyre-inference's own kernel, or its fp8-pool
    /// twin: the source, the file or the entry.
    pub const fn pick(self, fp16: &'static str, fp8: &'static str) -> &'static str {
        match self {
            KvPool::Fp16 => fp16,
            KvPool::Fp8 => fp8,
        }
    }

    /// The pool's pointer type in a paged kernel's signature.
    pub const fn pointer(self) -> &'static str {
        match self {
            KvPool::Fp16 => "*fp16",
            KvPool::Fp8 => "*fp8e4nv",
        }
    }

    /// Bytes one pool element holds.
    pub const fn element_bytes(self) -> u32 {
        match self {
            KvPool::Fp16 => 2,
            KvPool::Fp8 => 1,
        }
    }

    /// Pool elements one device stick holds.
    pub const fn stick_lanes(self) -> u32 {
        <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::FLIT_BYTES as u32
            / self.element_bytes()
    }

    /// How this pool's pages are laid out.
    pub const fn page_order(self) -> PageOrder {
        match self {
            KvPool::Fp16 => PageOrder::HeadMajor,
            KvPool::Fp8 => PageOrder::TokenMajor,
        }
    }
}

/// ⭐ HOW A PAGE IS LAID OUT. spyre-inference's fp16 page is head-major, `[kv_heads, block, head]`:
/// a head's keys at one token are one stick at head size 64. An fp8 page is token-major,
/// `[block, kv_heads, head]`: one token's keys are `kv_heads · head` bytes, whole sticks at every
/// head size, so the write lands whole sticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageOrder {
    HeadMajor,
    TokenMajor,
}

impl PageOrder {
    /// The KV write's index entries per written token: one per kv head on a head-major page, one
    /// on a token-major page.
    pub const fn write_entries(self, nkvh: u32) -> u32 {
        match self {
            PageOrder::HeadMajor => nkvh,
            PageOrder::TokenMajor => 1,
        }
    }

    /// ⭐ THE KV WRITE'S INDEX LAW — the cache row `cell`'s write entry lands at: head `cell.head`'s
    /// row `(page · kv_heads + head) · page_slots + slot` on a head-major page, the token's own row
    /// `page · page_slots + slot` on a token-major one. Every stager of the write's index reads it.
    pub const fn write_index(self, pool: PageGeometry, cell: PageCell) -> u32 {
        match self {
            PageOrder::HeadMajor => {
                (cell.page * pool.kv_heads + cell.head) * pool.page_slots + cell.slot
            }
            PageOrder::TokenMajor => cell.page * pool.page_slots + cell.slot,
        }
    }

    /// The pool element where `cell`'s head's `head_size` keys begin.
    pub const fn head_element(self, pool: PageGeometry, head_size: u32, cell: PageCell) -> u64 {
        let (kv_heads, slots, width) = (
            pool.kv_heads as u64,
            pool.page_slots as u64,
            head_size as u64,
        );
        let (page, slot, head) = (cell.page as u64, cell.slot as u64, cell.head as u64);
        match self {
            PageOrder::HeadMajor => ((page * kv_heads + head) * slots + slot) * width,
            PageOrder::TokenMajor => ((page * slots + slot) * kv_heads + head) * width,
        }
    }
}

/// A paged pool's extents — what every address law over its pages reads: the kv heads a page
/// holds and the token slots a page holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageGeometry {
    pub kv_heads: u32,
    pub page_slots: u32,
}

/// One kv head of one cached token: its page, its slot in the page, and the head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageCell {
    pub page: u32,
    pub slot: u32,
    pub head: u32,
}

/// THE ATTENTION FAMILY's decode kernel — `page_attn_head_major_decode.py`, spyre-inference's
/// `page_attn_head_major_decode_kernel`, at one query row.
pub fn attn_decode_row() -> TritonKernelRow {
    TritonKernelRow::Spliced {
        src: KvPool::BUILD.pick(
            include_str!("../../../../kernels/page_attn_head_major_decode.py"),
            include_str!("../../../../kernels/page_attn_head_major_decode_fp8.py"),
        ),
        file: KvPool::BUILD.pick(
            "page_attn_head_major_decode.py",
            "page_attn_head_major_decode_fp8.py",
        ),
        entry: KvPool::BUILD.pick(
            "page_attn_head_major_decode_kernel",
            "page_attn_head_major_decode_fp8_kernel",
        ),
        program: Program::PagedAttn,
    }
}

/// THE ATTENTION FAMILY's batched decode kernel — `batched_decode_head_major.py`, spyre-inference's
/// `batched_decode_head_major_kernel`, for a decode rung whose rows are separate sequences.
pub fn batched_decode_row() -> TritonKernelRow {
    TritonKernelRow::Spliced {
        src: KvPool::BUILD.pick(
            include_str!("../../../../kernels/batched_decode_head_major.py"),
            include_str!("../../../../kernels/batched_decode_head_major_fp8.py"),
        ),
        file: KvPool::BUILD.pick(
            "batched_decode_head_major.py",
            "batched_decode_head_major_fp8.py",
        ),
        entry: KvPool::BUILD.pick(
            "batched_decode_head_major_kernel",
            "batched_decode_head_major_fp8_kernel",
        ),
        program: Program::PagedAttn,
    }
}

/// The paged KV write's row — `reshape_and_cache_head_major.py`, spyre-inference's
/// `reshape_and_cache_head_major_kernel`: the scatter half of a KV writer (`RopeAppend`), its own
/// program.
pub fn kv_write_row() -> TritonKernelRow {
    TritonKernelRow::Spliced {
        src: KvPool::BUILD.pick(
            include_str!("../../../../kernels/reshape_and_cache_head_major.py"),
            include_str!("../../../../kernels/reshape_and_cache_head_major_fp8.py"),
        ),
        file: KvPool::BUILD.pick(
            "reshape_and_cache_head_major.py",
            "reshape_and_cache_head_major_fp8.py",
        ),
        entry: KvPool::BUILD.pick(
            "reshape_and_cache_head_major_kernel",
            "reshape_and_cache_head_major_fp8_kernel",
        ),
        program: Program::PagedKvWrite,
    }
}

/// The pool a cache tensor is: `pages` of `SchedulerBlockTokens::REFERENCE` slots at `kv_width = nkvh · hd` per slot.
/// Refused unless the cache is exactly that — a whole number of pages of the stated width.
fn pool_pages<F: scratchy_subtile::subtile_ir::RopeForm>(
    ir: &SubtileIR<F>,
    cache: &scratchy_subtile::subtile_ir::TensorRegion,
    kv_width: u32,
    what: &str,
) -> Result<u32, String> {
    let t = &ir.tensors[cache.tensor.index()];
    let block = SchedulerBlockTokens::REFERENCE.get();
    if t.cols != kv_width || t.rows == 0 || !t.rows.is_multiple_of(block) {
        return Err(format!(
            "triton splice: the {what} cache t{} is [{}, {}], which is not a whole number of \
             {block}-slot pages at {kv_width} columns — the pool it is read as is \
             [pages, nkvh, {block}, hd] over exactly those bytes",
            cache.tensor.index(),
            t.rows,
            t.cols
        ));
    }
    Ok(t.rows / SchedulerBlockTokens::REFERENCE.get())
}

/// A region that is its whole tensor, `rows × cols` — what a kernel viewing the tensor from corner
/// `[0, 0]` requires of its operand.
fn whole<F: scratchy_subtile::subtile_ir::RopeForm>(
    ir: &SubtileIR<F>,
    r: &scratchy_subtile::subtile_ir::TensorRegion,
    rows: u32,
    cols: u32,
    what: &str,
) -> Result<(), String> {
    let t = &ir.tensors[r.tensor.index()];
    let (rr, rc) = (&r.region.rows, &r.region.cols);
    if t.rows != rows
        || t.cols != cols
        || rr.start != 0
        || rc.start != 0
        || rr.len != rows
        || rc.len != cols
    {
        return Err(format!(
            "triton splice: the {what} t{} is [{}, {}] read at rows {}+{} cols {}+{}; the paged \
             kernels view it whole as [{rows}, {cols}]",
            r.tensor.index(),
            t.rows,
            t.cols,
            rr.start,
            rr.len,
            rc.start,
            rc.len
        ));
    }
    Ok(())
}

/// The most trips one page walk unrolls. Every trip is its own straight-line copy of the walk's
/// body in the baked program, so a longer walk takes more pages a trip instead of more trips.
pub const MAX_WALK_TRIPS: u32 = 64;

/// ⭐ A PROMPT WALK'S PAGES A TRIP — one (spyre-inference's walk, page by page) while `walk` fits
/// [`MAX_WALK_TRIPS`] trips, else the fewest that keeps it there: a power of two, and at least
/// Triton's gather floor, since a trip's pages are one gather.
pub fn trip_pages(walk: u32) -> u32 {
    if walk <= MAX_WALK_TRIPS {
        1
    } else {
        walk.div_ceil(MAX_WALK_TRIPS)
            .next_power_of_two()
            .max(BatchChunking::GATHER_FLOOR)
    }
}

/// ⭐ THE WALK — how many blocks of the page table a rung's attention reads (`NUM_BLOCKS`): the rung's
/// swept extent in pages. Every page of the pool at `ActiveCap::FULL` (a decode step's keys can sit
/// anywhere below the capacity), and at `ActiveCap::NONE` (a prompt chunk baked with no resident
/// prefix) the pages its own `mq` rows fill.
///
/// Blocks past a sequence's length carry a fully-masked tile, so a walk wider than the context costs
/// time, never correctness; a walk NARROWER than the context drops keys. So the walk is also the reach
/// of the rung: a step whose positions end at or below `walk · SchedulerBlockTokens::REFERENCE` can run on it, whatever its
/// start — the mask stack, not the rung, says which of the walked keys each row attends. The splice
/// bakes it and the wiring states it (`PagedStep::walk`) from this one function.
pub fn walk_pages(
    pages: u32,
    mq: u32,
    active_cap: scratchy_spyre_bundle::program::ActiveCap,
) -> u32 {
    let swept = active_cap.resolve(
        pages * SchedulerBlockTokens::REFERENCE.get(),
        scratchy_spyre_bundle::rung::POOL_STICK,
    );
    let walked = if swept == 0 { mq } else { swept };
    whole_trips(
        walked
            .div_ceil(SchedulerBlockTokens::REFERENCE.get())
            .min(pages),
    )
}

/// `pages` rounded up to whole trips of [`trip_pages`]: the pages past the pool's own are page 0
/// under a fully masked tile.
pub fn whole_trips(pages: u32) -> u32 {
    pages.next_multiple_of(trip_pages(pages))
}

/// ⭐ THE WALK RUNGS — spyre-inference's `SpyreAttnBucketer` (`spyre_attn_bucketer.py`, 297fc48): its
/// kv buckets are the powers of two from one block up to `max_model_len`, plus `max_model_len`
/// itself, and the kernel is specialized on `num_blocks = ceil(kv / block)` of each. Over a pool of
/// `pages` pages that is every power of two below `pages`, then `pages`. A step runs on the smallest
/// rung whose walk reaches its context, so a walk is at most about twice the context it serves —
/// "the extra padding each bucket costs is absorbed by the mask".
pub fn walk_buckets(pages: u32) -> Vec<u32> {
    let mut buckets = Vec::new();
    let mut b = 1u32;
    while b < pages {
        buckets.push(b);
        b *= 2;
    }
    buckets.push(pages);
    buckets
}

/// THE ATTENTION SPLICE — compile spyre-inference's kernel for this node's rows (one sequence's:
/// the decode kernel at one row, the prefill kernel past it; a decode batch's: the batched decode
/// kernel) at its geometry and the rung's walk ([`walk_pages`]), and mint the `Program::PagedAttn`
/// node over their operands, in their order: `[query, query_row_index, k_pages, v_pages,
/// page_index_table, (kv_row_pool,) mask_stack, out]`, or the batch's `[query, rep_row_ids, k_pages,
/// v_pages, chunk_page_ids, mask_by_chunk, out]`.
///
/// ⭐ THE PREFILL FAMILY — a packed prefill rung's pair (`per_request w`, `requests S > 1`) lowers
/// to `S` rebinds of the ONE compiled module: every constexpr of a window's program
/// (`NUM_TOKENS = w·S`, `PADDED_QUERY_LEN = w`, geometry, `SCALE`, `NUM_BLOCKS`) is identical
/// across windows, so the module is one object and a window differs only in which buffers its
/// `query_row_index`, `page_index_table` and `mask_stack` name — window `r`'s are
/// [`scratchy_spyre_bundle::reserved_tids::paged_window_tid`], window 0's being the step's own sentinels
/// (so a solo rung's family is the one op it has always baked, byte for byte). The windows are
/// `S` ordinary ops over the graph's absolute rows — the outputs are disjoint by construction
/// (window `r` scatters rows `r·w .. r·w+w` through its own query row index), the kvwrite stays
/// the one whole-graph program, and nothing below the splice knows a window exists.
pub fn lower_attn<
    T: scratchy_spyre_bundle::collective::Tp,
    F: scratchy_subtile::subtile_ir::RopeForm,
>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    scale_site: AttnScaleSite,
    active_cap: scratchy_spyre_bundle::program::ActiveCap,
    // ⭐ THE QUERY'S EXTENT PAIR — the rung's own `(width, capacity)`, minted once at the bake and
    // never divided back out of the graph's rows. The PREFILL arm spends it as the kernel's two
    // extents (`NUM_TOKENS` = the graph's rows, `PADDED_QUERY_LEN` = one request's width) and as
    // the walk's extent; the DECODE and BATCH arms read the graph's rows exactly as they did
    // before (one row per request, so the pair's product IS the row count by construction). The
    // agreement check below refuses a mint that does not describe the graph being lowered.
    query_rows: scratchy_spyre_bundle::rung::QueryExtent,
    paged_rows: PagedRows,
) -> Result<Vec<EmittedOp<T>>, String> {
    let batch = paged_rows == PagedRows::Batch;
    let decode = !batch && node.output.region.rows.len == 1;
    let row = if batch {
        batched_decode_row()
    } else if decode {
        attn_decode_row()
    } else {
        attn_row()
    };
    let (src, file, entry, program) = match &row {
        TritonKernelRow::Spliced {
            src,
            file,
            entry,
            program,
        } => (*src, *file, *entry, *program),
        TritonKernelRow::Refused(why) => return Err(why.to_string()),
    };
    // ⛔ EVERY FIELD NAMED, NO `..`: spyre-inference's kernels take a soft cap, a float8 cache and the
    // query's dtype, and this port takes none of them (the state and dots are f16, the cache is in
    // the model's 16-bit dtype). The op states none of them either, so none reaches here today; a
    // field added for one is a compile error at this line rather than an attention run without it.
    let SubOp::AttnDecode {
        valid_len: _,
        layout: _,
        producer,
        softmax_state: _,
        mask,
        attention,
    } = &node.op
    else {
        return Err(
            "triton splice: lower_attn called on a node that is not an AttnDecode".to_string(),
        );
    };
    if node.inputs.len() < 5 {
        return Err(format!(
            "triton splice: AttnDecode t{} expects 5 inputs [q, k cache, v cache, new k, new v], \
             found {}",
            node.output.tensor.index(),
            node.inputs.len()
        ));
    }
    // The walk reads the step's own keys out of the pools, and the only thing that puts them there
    // is this forward's KV writer — a `RopeAppend`, whose lowering scatters them before the walk.
    if !matches!(
        producer,
        scratchy_subtile::subtile_ir::KvCacheProducer::SameForwardRopeAppend { .. }
    ) {
        return Err(format!(
            "triton splice: AttnDecode t{}: its cache is not written by this forward's RopeAppend, \
             so the paged walk would read pools that never received this step's keys",
            node.output.tensor.index()
        ));
    }
    // The class's geometry off the op's own attention facts (minted at the bridge), and the
    // softmax scale at the site the weight fold decided: `InKernel` is the model's own scale,
    // `FoldedIntoWeights` the 1.0 the node runs at once √scale rides its W_q/W_k.
    let g = attention.class(mask.class());
    let (nqh, nkvh) = (g.q_heads().get(), g.kv_heads().get());
    // ⭐ THE HEAD WIDTH IS A TYPE — whole sticks by construction, so the extents below spend a
    // proven quantity and the one-stick/multi-stick question is `sticks()`, never a division.
    let hd = HeadWidth::of_whole_sticks(g.head_dim().get());
    let scale = scale_site.scale(*attention);
    let mq = node.output.region.rows.len;
    // ⭐ THE PAIR MEETS THE GRAPH HERE, ONCE — the mint's held product must be this graph's own row
    // count, or the mint describes a different rung than the one being lowered. A disagreement is a
    // REFUSAL by name (a bake bug, caught at the splice), never a silently-wrong extent: nothing
    // downstream may divide `mq` back to a per-request width, so this is the one place the two are
    // proven to agree.
    if query_rows.graph_rows().get() != mq {
        return Err(format!(
            "triton splice: AttnDecode t{}: the graph's {} row(s) are not the extent pair's \
             {} (per-request {} × {} request(s)) — the mint does not describe the graph it lowers",
            node.output.tensor.index(),
            mq,
            query_rows.graph_rows().get(),
            query_rows.per_request().get(),
            query_rows.requests().get(),
        ));
    }
    let kv_width = nkvh * hd.get();
    let pages = pool_pages(ir, &node.inputs[1], kv_width, "K")?;
    if pool_pages(ir, &node.inputs[2], kv_width, "V")? != pages {
        return Err(format!(
            "triton splice: AttnDecode t{}: the K and V pools differ in pages",
            node.output.tensor.index()
        ));
    }
    whole(ir, &node.inputs[0], mq, nqh * hd.get(), "query")?;
    whole(ir, &node.output, mq, nqh * hd.get(), "attention output")?;
    // ⭐ THE WALK'S EXTENT IS PER-REQUEST FOR A PREFILL CHUNK, THE GRAPH'S ROWS FOR THE BATCHED
    // KERNELS. The prefill kernel walks ONE request's pages — at one request the two are the same
    // number, which is why one `mq` served until S > 1; a packed rung's walk must cover one
    // request's `width` rows of its own record, not `width · S` rows of both records fused. The
    // decode kernel (one query row) and the batched kernel (the request axis carried inside one
    // launch) keep the graph's row count, exactly as before.
    let blocks = if batch || decode {
        walk_pages(pages, mq, active_cap)
    } else {
        walk_pages(pages, query_rows.per_request().get(), active_cap)
    };
    let tids = PagedStepTids::of(mask.class());
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    let t =
        |tr: &scratchy_subtile::subtile_ir::TensorRegion| BufferId::new(tr.tensor.index() as u32);
    if batch {
        // ⛔ spyre-inference's ELIGIBILITY (`_batched_decode_preconditions_met`, `spyre_attn.py`
        // 297fc48): a chunk gathering as many pages as the pool holds faults their device
        // (torch-spyre#4033), and their per-sequence LOOP — one launch a sequence — serves that step
        // instead. We have no per-sequence loop: one rung is one launch, so the only honest answer
        // is this refusal, by name. It needs a pool of at most `CORES` pages to fire, i.e. a context
        // of at most `CORES · SchedulerBlockTokens::REFERENCE` positions, and the one config in scope that small
        // (tinyllama-1.1b, 2048) bakes no spyre decode at all (its head is untied). PORTING their
        // loop is what admits such a pool.
        let chunking = BatchChunking::of(
            DecodeWalk {
                seqs: mq,
                blocks,
                pages,
            },
            DecodeHeads {
                kv_heads: nkvh,
                q_width: g.q_width(),
                head_size: g.head_dim().get(),
            },
        );
        if chunking.entries() >= pages {
            return Err(format!(
                "triton splice: AttnDecode t{}: a {mq}-sequence decode walking {blocks} block(s) \
                 gathers {} page(s) a chunk, and the pool holds {pages} — a gather of the whole \
                 pool faults the card, so spyre-inference serves this step with its per-sequence \
                 loop, one launch a sequence, which this port does not have",
                node.output.tensor.index(),
                chunking.entries()
            ));
        }
        for (p, ty) in [
            ("query", "*fp16"),
            ("rep_row_ids", "*i32"),
            ("k_pages", KvPool::BUILD.pointer()),
            ("v_pages", KvPool::BUILD.pointer()),
            ("chunk_page_ids", "*i32"),
            ("mask_by_chunk", "*fp16"),
            ("out", "*fp16"),
        ] {
            signature.insert(
                p.to_string(),
                ArgSpec::parse(ty).map_err(|e| e.to_string())?,
            );
        }
        for (k, v) in [
            ("NUM_TOKENS", mq),
            ("NUM_SEQS", chunking.seqs),
            ("BLOCKS_PER_CHUNK", chunking.blocks_per_chunk),
            ("NUM_CHUNKS", chunking.num_chunks),
            ("NUM_KV_HEADS", nkvh),
            ("NUM_QUERIES_PER_KV", nqh / nkvh),
            ("BLOCK_SIZE", SchedulerBlockTokens::REFERENCE.get()),
            ("HEAD_SIZE", hd.get()),
            ("NUM_PAGES", pages),
            ("ROWS_PER_PAGE", chunking.rows_per_page),
        ] {
            signature.insert(k.to_string(), ArgSpec::Constexpr);
            constexprs.insert(k.to_string(), Val::Int(i128::from(v)));
        }
        signature.insert("SCALE".to_string(), ArgSpec::Constexpr);
        constexprs.insert("SCALE".to_string(), Val::Float(f64::from(scale)));
        let spec = KernelSpec {
            kernel: entry.to_string(),
            signature,
            constexprs,
            file: file.to_string(),
        };
        let module = compile_kernel(src, &spec, &[1])?;
        let positional = crate::compile::node_for_compiled(&module, program)
            .map_err(|e| format!("triton splice: node_for: {e}"))?;
        let k = rebind(
            positional,
            format!("attn_s{}", node.id.index()),
            vec![
                t(&node.inputs[0]),
                BufferId::new(tids.rep_rows),
                t(&node.inputs[1]),
                t(&node.inputs[2]),
                BufferId::new(tids.chunk_pages),
                BufferId::new(tids.chunk_masks),
                t(&node.output),
            ],
        )?;
        let e = EmittedOp::new(format!("attn_s{}", node.id.index()), k);
        return Ok(vec![e]);
    }
    let pointers = [
        ("query", "*fp16"),
        ("query_row_index", "*i32"),
        ("k_pages", KvPool::BUILD.pointer()),
        ("v_pages", KvPool::BUILD.pointer()),
        ("page_index_table", "*i32"),
        ("mask_stack", "*fp16"),
        ("out", "*fp16"),
    ];
    for (p, ty) in pointers {
        signature.insert(
            p.to_string(),
            ArgSpec::parse(ty).map_err(|e| e.to_string())?,
        );
    }
    let mut sizes = vec![
        ("NUM_TOKENS", mq),
        ("NUM_HEADS", nqh),
        ("NUM_KV_HEADS", nkvh),
        ("HEAD_SIZE", hd.get()),
        ("BLOCK_SIZE", SchedulerBlockTokens::REFERENCE.get()),
        ("NUM_PAGES", pages),
        ("NUM_BLOCKS", blocks),
    ];
    if !decode {
        // ⭐ ONE REQUEST'S QUERY WIDTH, NOT THE GRAPH'S ROWS — the kernel gathers `P` rows through
        // `query_row_index` (a per-request table) and scatters them back through the same rows, so
        // its padded query length is the rung's per-request width. At one request the two agree,
        // which is why one `mq` served until S > 1.
        sizes.push(("PADDED_QUERY_LEN", query_rows.per_request().get()));
        sizes.push(("BLOCKS_PER_TRIP", trip_pages(blocks)));
    }
    for (k, v) in sizes {
        signature.insert(k.to_string(), ArgSpec::Constexpr);
        constexprs.insert(k.to_string(), Val::Int(i128::from(v)));
    }
    signature.insert("SCALE".to_string(), ArgSpec::Constexpr);
    constexprs.insert("SCALE".to_string(), Val::Float(f64::from(scale)));
    let spec = KernelSpec {
        kernel: entry.to_string(),
        signature,
        constexprs,
        file: file.to_string(),
    };
    let module = compile_kernel(src, &spec, &[1])?;
    let positional = crate::compile::node_for_compiled(&module, program)
        .map_err(|e| format!("triton splice: node_for: {e}"))?;
    // ⭐ THE WINDOW FAMILY — a prefill pair's capacity is how many one-sequence programs the one
    // compiled module is rebind to. Every OTHER arm is one window by its own pair: a batch's rows
    // are one request's one token each (`paged_rows == Batch`), so its capacity names QUERY ROWS a
    // rows==1 graph cannot split and the batch carries inside one launch; a one-row decode is the
    // same pair at capacity 1. Only the SEQUENCE arm with rows > 1 packs. A capacity past the
    // reserved tid space is REFUSED BY NAME — the bake's packed table outgrew what the region
    // holds, a bake bug, never a wrap onto another region's tids.
    let windows = if batch || decode {
        1
    } else {
        let s = query_rows.requests().get();
        if s > scratchy_spyre_bundle::reserved_tids::PAGED_MAX_WINDOWS {
            return Err(format!(
                "triton splice: AttnDecode t{}: a packed prefill of {s} window(s) — the reserved \
                 tid space holds {} (PAGED_MAX_WINDOWS); the bake's packed table outgrew it",
                node.output.tensor.index(),
                scratchy_spyre_bundle::reserved_tids::PAGED_MAX_WINDOWS,
            ));
        }
        s
    };
    // ⭐ THE BINDINGS ARE THE WINDOW'S WHOLE REACH. A window's activation-segment operands are
    // EXACTLY its three per-request tables (`query_row_index`, `page_index_table`, `mask_stack`)
    // at its own window tid, and nothing else — the list below is that law stated literally:
    // `query`/`out` are the graph's own rows (gathered and scattered through `query_row_index` —
    // absolute rows, window `r`'s being `r·w .. r·w+w`, never shifted), the pools are append-only
    // KV, and the decode-only `kv_row_pool` stays the step's own (decode never packs). A new
    // activation-segment operand added here is a new table the host must stage per window — it
    // does not ride for free.
    let KtirNode { func, program, .. } = positional;
    let mut out = Vec::with_capacity(windows as usize);
    for r in 0..windows {
        let window = |table: u32| {
            BufferId::new(scratchy_spyre_bundle::reserved_tids::paged_window_tid(
                paged_class(mask.class()),
                r,
                table,
            ))
        };
        let bindings = vec![
            t(&node.inputs[0]),
            window(0),
            t(&node.inputs[1]),
            t(&node.inputs[2]),
            window(1),
            window(2),
            t(&node.output),
        ];
        let name = if r == 0 {
            format!("attn_s{}", node.id.index())
        } else {
            format!("attn_s{}_w{}", node.id.index(), r)
        };
        if func.arguments.len() != bindings.len() {
            return Err(format!(
                "triton splice: {name} compiled to {} parameters but the node states {} \
                 bindings — the kernel's parameter list diverged from the row's contract",
                func.arguments.len(),
                bindings.len()
            ));
        }
        let k = KtirNode {
            func: ktir_core::ir::IRFunction {
                name: Arena::global().str(name.clone()),
                ..func
            },
            program,
            bindings,
        };
        let e = EmittedOp::new(name, k);
        out.push(e);
    }
    Ok(out)
}

/// THE KV WRITE'S SPLICE — the scatter half of a `RopeAppend`: compile spyre-inference's
/// `reshape_and_cache_head_major_kernel` for the node's rows and pools, and mint the
/// `Program::PagedKvWrite` node over their operands: `[key, value, k_rows, v_rows, row_index]`, where
/// `key` is the RopeAppend's own output (the rotated K) and `value` its V input.
pub fn lower_kv_write<
    T: scratchy_spyre_bundle::collective::Tp,
    F: scratchy_subtile::subtile_ir::RopeForm,
>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
) -> Result<EmittedOp<T>, String> {
    let row = kv_write_row();
    let (src, file, entry, program) = match &row {
        TritonKernelRow::Spliced {
            src,
            file,
            entry,
            program,
        } => (*src, *file, *entry, *program),
        TritonKernelRow::Refused(why) => return Err(why.to_string()),
    };
    let SubOp::RopeAppend {
        attention,
        rope,
        attn,
        ..
    } = &node.op
    else {
        return Err(
            "triton splice: lower_kv_write called on a node that is not a RopeAppend".to_string(),
        );
    };
    if node.inputs.len() < 6 {
        return Err(format!(
            "triton splice: RopeAppend t{} expects 6 inputs [k, cos, sin, v, k cache, v cache], \
             found {}",
            node.output.tensor.index(),
            node.inputs.len()
        ));
    }
    // The class's facts off the op's own attention value (minted at the bridge): the head
    // dim is the rope class's own, the head count the mask class's — never a division of
    // the region's width, which is CHECKED against them below.
    let g = attention.class(attn.class());
    let (hd, nkvh) = (
        HeadWidth::of_whole_sticks(rope.head_dim(*attention).get()),
        g.kv_heads().get(),
    );
    let (rows, kv_width) = (node.output.region.rows.len, node.output.region.cols.len);
    if kv_width != nkvh * hd.get() {
        return Err(format!(
            "triton splice: RopeAppend t{}: its {kv_width}-column K is not the class's \
             {nkvh} whole heads of {hd_wide}",
            node.output.tensor.index(),
            hd_wide = hd.get()
        ));
    }
    whole(ir, &node.output, rows, kv_width, "rotated K")?;
    whole(ir, &node.inputs[3], rows, kv_width, "V")?;
    let pages = pool_pages(ir, &node.inputs[4], kv_width, "K")?;
    if pool_pages(ir, &node.inputs[5], kv_width, "V")? != pages {
        return Err(format!(
            "triton splice: RopeAppend t{}: the K and V pools differ in pages",
            node.output.tensor.index()
        ));
    }
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    for (p, ty) in [
        ("key", "*fp16"),
        ("value", "*fp16"),
        ("k_rows", KvPool::BUILD.pointer()),
        ("v_rows", KvPool::BUILD.pointer()),
        ("row_index", "*i32"),
    ] {
        signature.insert(
            p.to_string(),
            ArgSpec::parse(ty).map_err(|e| e.to_string())?,
        );
    }
    let mut ces = vec![
        ("T", rows),
        ("NUM_KV_HEADS", nkvh),
        ("HEAD_SIZE", hd.get()),
        (
            "NUM_ROWS",
            pages * nkvh * SchedulerBlockTokens::REFERENCE.get(),
        ),
    ];
    // A token-major page's write lands whole sticks, and takes the stick's width.
    if let PageOrder::TokenMajor = KvPool::BUILD.page_order() {
        ces.push(("STICK", KvPool::BUILD.stick_lanes()));
    }
    for (k, v) in ces {
        signature.insert(k.to_string(), ArgSpec::Constexpr);
        constexprs.insert(k.to_string(), Val::Int(i128::from(v)));
    }
    let spec = KernelSpec {
        kernel: entry.to_string(),
        signature,
        constexprs,
        file: file.to_string(),
    };
    let module = compile_kernel(src, &spec, &[1])?;
    let positional = crate::compile::node_for_compiled(&module, program)
        .map_err(|e| format!("triton splice: node_for: {e}"))?;
    let t =
        |tr: &scratchy_subtile::subtile_ir::TensorRegion| BufferId::new(tr.tensor.index() as u32);
    let bindings = vec![
        t(&node.output),
        t(&node.inputs[3]),
        t(&node.inputs[4]),
        t(&node.inputs[5]),
        BufferId::new(PagedStepTids::of(attn.class()).write_rows),
    ];
    let name = format!("kvwrite_s{}", node.id.index());
    let k = rebind(positional, name.clone(), bindings)?;
    let e = EmittedOp::new(name, k);
    Ok(e)
}

/// A compiled program re-stated with this node's own name and bindings (`node_for` mints positional
/// buffer ids), refused if the kernel's parameter list and the bindings disagree in length.
fn rebind(positional: KtirNode, name: String, bindings: Vec<BufferId>) -> Result<KtirNode, String> {
    let KtirNode { func, program, .. } = positional;
    if func.arguments.len() != bindings.len() {
        return Err(format!(
            "triton splice: {name} compiled to {} parameters but the node states {} bindings — \
             the kernel's parameter list diverged from the row's contract",
            func.arguments.len(),
            bindings.len()
        ));
    }
    Ok(KtirNode {
        func: ktir_core::ir::IRFunction {
            name: Arena::global().str(name),
            ..func
        },
        program,
        bindings,
    })
}

/// THE KERNEL'S LAUNCH CONTRACT, stated from the node's own shapes — the signature (one
/// descriptor per operand plus the output), the constexprs (the node's facts as
/// `tl.constexpr`s, the monomorphisation key), all in the case-table's own shape
/// (`cases::spec`).
/// The `rope_fwd` kernel's contract, stated from the node's own facts — the one body both rope
/// ops' arms in `kernel_spec` run, each reading its head dim off its own fields: the rotate's
/// own, the append's rope class off its attention.
fn rope_fwd<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    signature: &mut HashMap<String, ArgSpec>,
    constexprs: &mut HashMap<String, Val>,
    hd: HeadWidth,
) -> Result<(), String> {
    // total = the output's declared width (`heads * hd`), heads = total / hd, mq = the
    // output's rows. A `total` that is not a whole number of `hd`-wide heads is an `Err`:
    // the node is malformed, and no kernel can state it.
    let out = &node.output;
    let total = out.region.cols.len;
    if !total.is_multiple_of(hd.get()) {
        return Err(format!(
            "triton splice: rope t{}: {total} cols is not a whole number of \
             {hd}-wide heads",
            out.tensor.index(),
            hd = hd.get()
        ));
    }
    let heads = total / hd.get();
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
    // ⭐ THE ROW WINDOW, THE PER-ROW ROPE LAW. The real front end's
    // per-position decomposition mints rope nodes whose regions are ONE
    // row's slice of the shared q plane — a row slice expressed by the
    // ACCESS TILE's row corner (`R_START`), never by a taller view. The
    // kernel therefore states `Y_TOTAL` (the x TENSOR's full re-tiled
    // height `rows·H`, so the tile corners stay in bounds) and `R_START`
    // (the window's row corner `ri·H`, baked into every load/store row);
    // `MQ` is then the REGION's row count. A whole-region node states
    // `R_START = 0`, `Y_TOTAL = MQ·H` — byte-identical to the one form
    // this kernel used to state. ⛔ THE WINDOWING LAW, stated here because
    // only this body reads the facts: all four regions share ONE row
    // window, take WHOLE columns of `total`, and sit over tensors of ONE
    // height (the single `Y_TOTAL` names all four views, and the
    // head-replicated cos/sin reinterpretation is same-bytes only at that
    // width).
    let x = &node.inputs[0];
    let window = (x.region.rows.start, x.region.rows.len);
    let y_rows = ir.tensors[x.tensor.index()].rows;
    for (tr, role) in [
        (x, "the `x` view"),
        (&node.inputs[1], "the `cos` view"),
        (&node.inputs[2], "the `sin` view"),
        (out, "the `out` view"),
    ] {
        let t_rows = ir.tensors[tr.tensor.index()].rows;
        let shared = (tr.region.rows.start, tr.region.rows.len) == window;
        let cols_whole = tr.region.cols.start == 0 && tr.region.cols.len == total;
        if !shared || !cols_whole || t_rows != y_rows {
            return Err(format!(
                "triton splice: rope t{}: {role} states rows [{}, {}] × cols [{}, \
                 {}] of a [{t_rows}, {}] tensor, but `x` states rows [{}, {}] over a \
                 [{y_rows}, {total}] one — a rope's four regions must share ONE row \
                 window and whole `total` columns over tensors of one height (the \
                 single `Y_TOTAL` names all four views)",
                out.tensor.index(),
                tr.region.rows.start,
                tr.region.rows.len,
                tr.region.cols.start,
                tr.region.cols.len,
                ir.tensors[tr.tensor.index()].cols,
                window.0,
                window.1,
            ));
        }
    }
    let (r_start, mq_rows) = (window.0 * heads, window.1);
    let y_total = y_rows * heads;
    ce("H", Val::Int(i128::from(heads)))?;
    ce("MQ", Val::Int(i128::from(mq_rows)))?;
    ce("HEAD_DIM", Val::Int(i128::from(hd.get())))?;
    ce("HALF", Val::Int(i128::from(hd.get() / 2)))?;
    ce("Y_TOTAL", Val::Int(i128::from(y_total)))?;
    ce("R_START", Val::Int(i128::from(r_start)))?;
    Ok(())
}

/// ⭐ THE fp8-DYNAMIC EPILOGUE'S LIVE `[m, N]` f16 TILES AT ITS PEAK — a fact of
/// the program the `SubOp::MatmulTile { weight: GemmWeight::Fp8Dynamic }` arm of
/// [`kernel_spec`] mints (`matmul_fp8_fwd`): the dot result `p`, the broadcast's
/// shape-defining `tensor.empty`, and the broadcast replica of the per-channel
/// scale row. MEASURED by the isolated single-node probe through the production
/// session entry (K=3840/N=15360): the refusal `1935360 + 952320 > 2097152`
/// decomposes exactly into p + empty + scale-row + the charging broadcast; the
/// zero-init splat is freed by the resident scheduler's dies-at reclaim right
/// after the dot, and the activation/weight loads are uncharged views. The fp8
/// form CANNOT be N-blocked down to fit (`verify_canonical_fp8_matmul_kernel`
/// refuses `BLOCK_N < N` by name), so the only lever that shrinks the epilogue
/// is the baked WIDTH — which is why the emu's prefill-width derivation reads
/// this. Pinned by the live-tile count over the minted program
/// (`the_fp8_epilogue_holds_three_live_mn_tiles_at_its_peak`).
pub const FP8_EPILOGUE_LIVE_TILES: u128 = 3;

/// The fp8-dynamic epilogue's intra-node LX footprint at its peak: the
/// [`FP8_EPILOGUE_LIVE_TILES`] live `[m, N]` f16 tiles plus the collapsed `[N]`
/// scale row. Plain arithmetic on the op's stated extents.
pub const fn fp8_epilogue_peak_bytes(m: u32, n: u64) -> u128 {
    FP8_EPILOGUE_LIVE_TILES * m as u128 * n as u128 * 2 + n as u128 * 2
}

fn kernel_spec<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    file: &str,
    entry: &str,
) -> Result<KernelSpec, String> {
    let out = &node.output;
    let (m, c) = (out.region.rows.len, out.region.cols.len);
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    match (&node.op, entry) {
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
            // The epsilon reaches the program as a splat the lowering stages as a stick
            // constant. The Triton ladder collapses the f32 island around `rsqrt` and
            // rounds the value to f16, so `1e-5f32` becomes `0.00001001358` in the
            // program — the value the descriptor reads.
            ce("EPS", Val::Float(f64::from(*eps)))?;
            // `1.0 / D_MODEL` folded on the host, exactly as the fixture's own
            // `constexprs()` helper does — one multiply instead of a divide.
            ce("INV_D", Val::Float(1.0 / f64::from(c)))?;
        }
        // ⭐ THE (1 + w) FORM'S OWN SPEC — the same signature and constants as the Scale
        // arm, plus the offset itself: the convention's own `offset()` (1.0;
        // `GainConvention::from_offset` refuses every other value), staged by the lowering
        // as a stick constant like the epsilon.
        (
            SubOp::RmsNorm {
                eps,
                gain: GainConvention::OnePlusScale,
            },
            "rmsnorm_oneplus_fwd",
        ) => {
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
            ce("EPS", Val::Float(f64::from(*eps)))?;
            ce("INV_D", Val::Float(1.0 / f64::from(c)))?;
            ce(
                "GAIN_OFFSET",
                Val::Float(f64::from(GainConvention::OnePlusScale.offset())),
            )?;
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
            // ONE tile, the whole region (one linalg.matmul, no K loop). `M_TOTAL` names
            // the OUTPUT TENSOR's row extent (the storage the descriptor addresses) while
            // the store takes the `[M, N]` tile at row 0 — the prefill lm-head fold's m=S
            // tail, whose output is the `[S, vocab]` prefix of the `[mq, vocab]` logits.
            // ⛔ `N` IS THE DEVICE WIDTH — the activation row law (`rung::row_pitch`): the
            // layout reserves rows of whole sticks and every consumer of the buffer reads
            // them at that pitch, so the view and tile name the storage as it is held.
            let (m_total, _a_total) = matmul_window_of(node, ir)?;
            let n_dev = stick_width(*n);
            ce("M", Val::Int(i128::from(m)))?;
            ce("K", Val::Int(i128::from(k)))?;
            ce("N", Val::Int(i128::from(n_dev)))?;
            ce("BLOCK_M", Val::Int(i128::from(m)))?;
            ce("BLOCK_K", Val::Int(i128::from(k)))?;
            ce("BLOCK_N", Val::Int(i128::from(n_dev)))?;
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
            // derivations the dense arm states.
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
            // The region in `[BLOCK_M, N]` row blocks — PLUS the STORAGE the region
            // windows: the descriptor names the TENSOR (`[M, N_TOTAL]`, strides
            // `[N_TOTAL, 1]`) and the load/store names the CORNER (`C_START`). A
            // whole-region node states `N_TOTAL = N`, `C_START = 0`.
            let (n_total, c_start) = window_of(node, ir)?;
            let (block_m, n_blocks, tail_h) = blocks_of(m, c, SILU_MUL_LIVE_TILES);
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
            let (block_m, n_blocks, tail_h) = blocks_of(m, c, EW_SILU_LIVE_TILES);
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
            // C_START — the region in row blocks, PLUS the storage-window facts (see
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
            let (block_m, n_blocks, tail_h) = blocks_of(m, c, EW_BINARY_LIVE_TILES);
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
            // ⛔ THE WINDOW IS THE DEVICE WIDTH, NOT THE LOGICAL ONE: a ScalarMul on
            // the logits runs over the rows its producer matmul wrote, whole sticks wide
            // (`rung::row_pitch`), and the Triton front end refuses a block whose last dim
            // is under 16 bytes (`semantic.py:1863`), so a 3-wide logits tail chunk is not
            // spellable at its logical width. ⛔ AND SO IS `N_TOTAL`: the storage is held
            // at the device width, so a view at the logical width would understate the
            // buffer and a device-width tile at the chunk corner would leave it.
            let c_dev = stick_width(c);
            let (block_m, n_blocks, tail_h) = blocks_of(m, c_dev, 3);
            let n_total_dev = stick_width(n_total);
            ce("M", Val::Int(i128::from(m)))?;
            ce("N", Val::Int(i128::from(c_dev)))?;
            ce("BLOCK_M", Val::Int(i128::from(block_m)))?;
            ce("BLOCK_N", Val::Int(i128::from(c_dev)))?;
            ce("N_TOTAL", Val::Int(i128::from(n_total_dev)))?;
            ce("C_START", Val::Int(i128::from(c_start)))?;
            ce("N_BLOCKS", Val::Int(i128::from(n_blocks)))?;
            ce("TAIL_H", Val::Int(i128::from(tail_h)))?;
            // THE SCALE, from the node's own payload: one splat feeding every
            // `arith.mulf`, staged by the lowering as a stick constant.
            ce("SCALE", Val::Float(f64::from(*scale)))?;
        }
        (SubOp::RopeRotate { head_dim, .. }, "rope_fwd") => {
            rope_fwd(
                node,
                ir,
                &mut signature,
                &mut constexprs,
                HeadWidth::of_whole_sticks(head_dim.get()),
            )?;
        }
        (
            SubOp::RopeAppend {
                attention, rope, ..
            },
            "rope_fwd",
        ) => {
            rope_fwd(
                node,
                ir,
                &mut signature,
                &mut constexprs,
                HeadWidth::of_whole_sticks(rope.head_dim(*attention).get()),
            )?;
        }
        (SubOp::TanhSoftCap { cap }, "tanhsoftcap_fwd") => {
            // The kernel's own parameter spellings: desc_x, desc_o, then the constexprs
            // M / N / BLOCK_M / BLOCK_N / CAP / N_BLOCKS / TAIL_H.
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
            // ⛔ THE CAP DIVIDES THE LOGITS — a non-positive cap makes `x / cap` a
            // division by zero or a sign flip, and no kernel can state it. The
            // producer-side refusal (main's own), re-homed here where the value is
            // read.
            if *cap <= 0.0 {
                return Err(format!(
                    "triton splice: tanhsoftcap t{} states cap {cap} — the cap divides the \
                     logits (`x / cap`) and multiplies the tanh back up, so it must be positive",
                    node.output.tensor.index()
                ));
            }
            // ROW BLOCKING: three live tiles (x, the tanh's
            // operand, the result) over the VOCAB-wide logits, wider than any
            // activation — prefill rungs exceed the LX budget while decode (m=1)
            // never does.
            let (block_m, n_blocks, tail_h) = blocks_of(m, c, 3);
            ce("M", Val::Int(i128::from(m)))?;
            ce("N", Val::Int(i128::from(c)))?;
            ce("BLOCK_M", Val::Int(i128::from(block_m)))?;
            ce("BLOCK_N", Val::Int(i128::from(c)))?;
            ce("CAP", Val::Float(f64::from(*cap)))?;
            ce("N_BLOCKS", Val::Int(i128::from(n_blocks)))?;
            ce("TAIL_H", Val::Int(i128::from(tail_h)))?;
        }
        (SubOp::RmsNormUnit { eps }, "rmsnormunit_fwd") => {
            // The rmsnorm arm's constants minus the weight: desc_x, desc_o, then M /
            // D_MODEL / BLOCK_M / EPS / INV_D — the unit form has ONE tensor parameter
            // where RmsNorm has two, the only structural difference.
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
            ce("D_MODEL", Val::Int(i128::from(c)))?;
            ce("BLOCK_M", Val::Int(i128::from(m)))?;
            ce("EPS", Val::Float(f64::from(*eps)))?;
            ce("INV_D", Val::Float(1.0 / f64::from(c)))?;
        }
        (SubOp::ScalarWeightMul, "scalarwmul_fwd") => {
            // desc_x, `w` (the `[1]` weight as a RUNTIME fp16 SCALAR — rung 3 of the
            // address-provenance ladder, `mul_scale.py`'s own shape), desc_o, then
            // M / N — the logical width (the weight is read in the scalar broadcast mode
            // and pads its own placement). ⛔ THE WEIGHT IS NOT A DESCRIPTOR: a `[1]`-region load
            // would have to broadcast its `[1, 1]`-widened tile over `[M, N]`, an
            // all-unit source `to_ktir` refuses (it would need a rank-0 source), while
            // a rank-0 argument SPLATS — no broadcast anywhere — and the launch binds
            // the same staged buffer through the `[1, 1]` view the rung-3 rewrite
            // mints over the argument.
            for p in ["desc_x", "desc_o"] {
                signature.insert(
                    p.to_string(),
                    ArgSpec::parse("*fp16").map_err(|e| e.to_string())?,
                );
            }
            signature.insert(
                "w".to_string(),
                ArgSpec::parse("fp16").map_err(|e| e.to_string())?,
            );
            let mut ce = |k: &str, v: Val| -> Result<(), String> {
                signature.insert(k.to_string(), ArgSpec::Constexpr);
                constexprs.insert(k.to_string(), v);
                Ok(())
            };
            ce("M", Val::Int(i128::from(m)))?;
            ce("N", Val::Int(i128::from(c)))?;
        }
        (SubOp::Reshape { .. }, "reshape_fwd") => {
            // The stick walk's own facts: both sides at their TRUE region extents
            // (the whole-tensor law is the region guard's), the element-count law,
            // and the stick law the walk's `// 64` arithmetic assumes.
            let src = &node.inputs[0];
            let (r_in, c_in) = (src.region.rows.len, src.region.cols.len);
            let (r_out, c_out) = (out.region.rows.len, out.region.cols.len);
            let elems_in = u64::from(r_in) * u64::from(c_in);
            let elems_out = u64::from(r_out) * u64::from(c_out);
            if elems_in != elems_out {
                return Err(format!(
                    "triton splice: reshape t{}: [{r_in}, {c_in}] → [{r_out}, {c_out}] is \
                     {elems_in} → {elems_out} elements — a re-laying copy preserves the flat \
                     element sequence (the Reshape law), and no kernel can state a count change",
                    node.output.tensor.index()
                ));
            }
            let stk = scratchy_spyre_bundle::rung::FP16_LANES; // the walk copies whole fp16 sticks
            if c_in % stk != 0 || c_out % stk != 0 {
                return Err(format!(
                    "triton splice: reshape t{}: a {c_in}- or {c_out}-wide side is not a whole \
                     {stk}-fp16 stick — a device tensor is stick-blocked, so flat elements \
                     {stk}q..{stk}q+{stk} are one `[1, {stk}]` tile of each side only when both \
                     column extents are whole sticks (the kernel's walk steps `// {stk}`)",
                    node.output.tensor.index()
                ));
            }
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
            ce("R_IN", Val::Int(i128::from(r_in)))?;
            ce("C_IN", Val::Int(i128::from(c_in)))?;
            ce("R_OUT", Val::Int(i128::from(r_out)))?;
            ce("C_OUT", Val::Int(i128::from(c_out)))?;
            ce("N_STICKS", Val::Int(i128::from(elems_in / u64::from(stk))))?;
        }
        (op, entry) => {
            return Err(format!(
                "triton splice: no kernel signature for {op:?} at entry `{entry}` — the row is \
                 incomplete"
            ));
        }
    }
    Ok(KernelSpec {
        kernel: entry.to_string(),
        signature,
        constexprs,
        file: file.to_string(),
    })
}

/// The STORAGE-WINDOW FACTS a pointwise kernel states: the tensor's full column
/// extent (`N_TOTAL` — the descriptor's own shape/strides name the STORAGE) and the
/// region's column corner (`C_START`, `region.cols.start` — the load/store offsets name
/// the WINDOW).
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
    // The operands must window the SAME slice of the SAME storage (one chunk per
    // node).
    for tr in &node.inputs {
        // A SYNTHETIC BEYOND THE GRAPH (the tensor-parallel receive window) has no storage in the
        // graph: its storage IS its region, from its first byte — `matmul_window_of`'s law for the
        // LAST_HIDDEN staging. So it joins the node's one storage window only when that region is
        // exactly the window the output states; anything else is the disagreement refused below.
        let in_total = ir
            .tensors
            .get(tr.tensor.index())
            .map_or(tr.region.cols.len, |s| s.cols);
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
             row {}) — the matmul kernels state the activation's tile at row 0; the prefill \
             lm-head fold is the row-0 window form",
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

/// A row's storage width: whole fp16 sticks, the activation row law.
fn stick_width(n: u32) -> u32 {
    scratchy_spyre_bundle::rung::row_pitch(
        u64::from(n),
        u64::from(scratchy_spyre_bundle::rung::FP16_LANES),
    ) as u32
}

/// The fp16 elements one program's live tiles may hold together: the scratchpad's capacity.
const LIVE_ELEMS: u64 =
    <ktir_core::ddl_binds::Baked as ktir_core::ddl_binds::DeviceGen>::LX_CAPACITY_BYTES / 2;

/// The tiles an elementwise kernel holds live at once: silu's decomposition (`x`, its negation,
/// the exponential, the one, the denominator, the result); a binary op's two operands and result;
/// silu-mul's eight; a block copy's one.
const EW_SILU_LIVE_TILES: u32 = 6;
const EW_BINARY_LIVE_TILES: u32 = 3;
const SILU_MUL_LIVE_TILES: u32 = 8;
const BLOCK_COPY_LIVE_TILES: u32 = 1;

/// How many rows of a `cols`-wide region keep `live` tiles inside [`LIVE_ELEMS`].
fn rows_per_block(cols: u32, live: u32) -> u32 {
    ((LIVE_ELEMS / u64::from(live.max(1)) / u64::from(cols.max(1))) as u32).max(1)
}

/// THE ROW-BLOCK CONSTANTS the pointwise kernels take, stated from the node's own
/// region: `BLOCK_M` is [`rows_per_block`], `N_BLOCKS` full blocks follow, and `TAIL_H`
/// is the shorter last tile when the region does not divide evenly.
///
/// A region that FITS the budget answers `(rows, 1, 0)` — ONE whole-region tile, the
/// constants the one-tile form always implied, so nothing changes for decode (m=1)
/// or any prefill rung inside the LX.
///
/// ⛔ THE `live` COUNT IS THE FAMILY'S OWN — `EW_SILU_LIVE_TILES` /
/// `EW_BINARY_LIVE_TILES` / `SILU_MUL_LIVE_TILES` / 3 for scalarmul — read by the
/// CALLER, because the live set is the kernel's own: a wrong count here would block
/// at a height whose live tiles overflow LX.
fn blocks_of(rows: u32, cols: u32, live: u32) -> (u32, u32, u32) {
    let blk = rows_per_block(cols, live);
    if blk >= rows {
        // The region FITS: one whole-region tile.
        return (rows, 1, 0);
    }
    let n_blocks = rows / blk;
    let tail = rows % blk;
    (blk, n_blocks, tail)
}

/// The launch grid: `[1]` for every kind — the kernel is compiled at the grid the case
/// table states (`vec![1]`).
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
        // spliced program is straight-line — no grid axis at all.
        SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. } => Ok(vec![1]),
        // The same one-work-item law: the row blocks are in-kernel tiles. The MoE
        // sub-ops have no grid row any more — the registry refuses them before the
        // splice (the MoE lowers only from `kernels/moe.py` through KTIR).
        SubOp::TanhSoftCap { .. }
        | SubOp::RmsNormUnit { .. }
        | SubOp::ScalarWeightMul
        | SubOp::Reshape { .. } => Ok(vec![1]),
        // ⛔ NO `_` ARM. A spliced kind is a row above, and a row without a grid here is
        // an unreachable — the same discipline the arity match holds.
        _ => Err("triton splice: no grid for this op kind — the row is incomplete".to_string()),
    }
}

/// THE FULL LADDER for one kernel: parse → TTIR → `make_ttir` → `from_ttir` → `make_ktir`
/// → `to_ktir` — the exact sequence `bake_py.rs` drives, stated once.
///
/// ⭐⭐⭐ MEMOIZED BY SPEC — THE EXPANSION-COST LAW. A bake asks for the SAME (kernel,
/// spec, grid) many times over: the prefill ladder's 21 rungs × 40 layers state one
/// program per (kind, shape) cell, and the shape repeats across every layer of a rung —
/// MEASURED before this cache, the 2b card bake spent ~100 minutes in the models build
/// script, ~17 s per attention program alone (~26 programs in the golden's 442 s run).
/// The compile is a pure function
/// of (source, spec, grid) — the ladder is deterministic — so the second and later
/// callers SHARE the first result (one `Arc`, no per-caller deep copy of the module
/// tree: MEASURED on the tp/8 fp8 bake, 2026-11-11, the clone-on-hit was the memo's own
/// second cost). No consumer mutates the module — `mint` and the attention mints only
/// READ it (`node_for` takes `&Module`), and everything a node re-states (its
/// bindings) is a field of the MINTED NODE, not of the module.
///
/// ⭐ THE KEY RIDES ALONG, INSIDE THE VALUE ([`CompiledKernel`]): the memo's own mint
/// pairs the module with the key that produced it, and the mint hands the pair to
/// [`compile::node_for_compiled`], so the KTIR handoff lowers each kernel
/// ONCE, not once per node that splices it (the same layers-times-rungs law that
/// motivated this memo).
fn compile_kernel(src: &str, spec: &KernelSpec, grid: &[i64]) -> Result<CompiledKernel, String> {
    /// The module memo: one compile per key, and the ONLY mint of the paired
    /// [`CompiledKernel`] the mints take. The key is [`memo_key`]'s canonical `String`,
    /// so it needs no `Hash`/`Ord` on `ArgSpec`/`Val` and is deterministic in
    /// construction.
    static CACHE: crate::compile::ModuleMemo = crate::compile::ModuleMemo::new();
    CACHE.compile_under(memo_key(src, spec, grid), || {
        compile_kernel_uncached(src, spec, grid)
    })
}

/// ⭐ A PROCESS-WIDE MEMO, SHARED BY EVERY THREAD OF THE EMIT — one compile per key, ever.
///
/// It was `thread_local!`, which made a second thread compile what the first already had, so the
/// emit could not lower nodes in parallel without paying each kernel once per thread. Each key owns a
/// `OnceLock`: the first caller computes, a concurrent caller for the SAME key waits on that one
/// computation, and callers for different keys never wait on each other (the map lock is held only
/// to find or insert the cell). A failure is memoized like a success — the compile is a pure function
/// of the key, so a retry could only say the same thing.
struct Memo<T> {
    cells: std::sync::Mutex<std::collections::BTreeMap<String, MemoCell<T>>>,
}

/// One key's result: computed at most once, by whichever caller gets there first. The value is
/// behind an `Arc` — every caller is handed the SAME result, never a deep copy of it (the module
/// tree is megabytes at bake scale; cloning it per caller was the memo's own second cost, measured
/// on the tp/8 fp8 bake, 2026-11-11).
type MemoCell<T> = std::sync::Arc<std::sync::OnceLock<Result<std::sync::Arc<T>, String>>>;

impl<T> Memo<T> {
    const fn new() -> Memo<T> {
        Memo {
            cells: std::sync::Mutex::new(std::collections::BTreeMap::new()),
        }
    }

    /// `f()` once per `key`, the same `Arc` handed to every caller. A `None` key (a spec
    /// [`memo_key`] cannot spell injectively) is never memoized: it computes every time, and the
    /// fresh result is still wrapped so the caller's type is one shape.
    fn get_or_compute(
        &self,
        key: Option<String>,
        f: impl FnOnce() -> Result<T, String>,
    ) -> Result<std::sync::Arc<T>, String> {
        let Some(key) = key else {
            return f().map(std::sync::Arc::new);
        };
        let cell = {
            let mut cells = self.cells.lock().unwrap_or_else(|p| p.into_inner());
            std::sync::Arc::clone(cells.entry(key).or_default())
        };
        match cell.get_or_init(|| f().map(std::sync::Arc::new)) {
            Ok(value) => Ok(std::sync::Arc::clone(value)),
            Err(e) => Err(e.clone()),
        }
    }
}

/// THE CANONICAL KEY of one (source, spec, grid) compile — `None` for a spec this key cannot
/// spell injectively, which sits it out of every memo keyed on it.
fn memo_key(src: &str, spec: &KernelSpec, grid: &[i64]) -> Option<String> {
    use std::collections::BTreeMap;
    // ⛔ THE SOURCE TEXT IS PART OF THE KEY, not the file path: `read_kernel` re-reads
    // the `.py` from disk on every call and the bake runs inside ONE build-script
    // process, so a same-named file edited between two `lower` calls in one expansion
    // (a test that rewrites a fixture) must not be served the first text's module.
    let mut key = String::with_capacity(src.len() + 256);
    key.push_str(spec.file.as_str());
    key.push('\0');
    key.push_str(spec.kernel.as_str());
    key.push('\0');
    let mut params: BTreeMap<&str, String> = BTreeMap::new();
    for (name, arg) in &spec.signature {
        params.insert(
            name.as_str(),
            match arg {
                ArgSpec::Constexpr => "constexpr".to_string(),
                // ⛔ THE FULL TYPE, NOT JUST THE DISCRIMINANT: a `*fp16` and a `*fp8`
                // parameter of one kernel name are different programs, and keying on
                // "ptr" alone would serve one for the other. The debug spelling is
                // injective for `Type` (a plain enum of plain payloads).
                ArgSpec::Ptr(t) => format!("ptr:{t:?}"),
                ArgSpec::Scalar(t) => format!("scalar:{t:?}"),
            },
        );
    }
    for (name, kind) in &params {
        key.push_str(name);
        key.push('\u{1}');
        key.push_str(kind);
        key.push('\u{1}');
    }
    key.push('\0');
    // ⛔ EVERY CONSTEXPR VALUE, CANONICALLY ORDERED — a HashMap iteration order must
    // never leak into the key, or two identical specs could miss. `Val::Float` is keyed
    // by its BITS: -0.0 and 0.0 compile identically but `f64::to_bits` distinguishes
    // them, and distinguishing is the safe direction for a cache key.
    // ⛔ AND A VARIANT WITH NO INJECTIVE SPELLING SITS THE WHOLE SPEC OUT of the memo:
    // `Dtype`/`Seq`/`Ir`/`Desc`/`Slice` carry payloads this key does not spell, so two
    // specs differing only inside one would share a key and the second would be served
    // the first's module. No row binds them today; a row that ever does pays a full
    // compile per call (correct, just unmemoized) rather than risking a wrong hit.
    let mut memo_eligible = true;
    let mut cvals: BTreeMap<&str, String> = BTreeMap::new();
    for (name, v) in &spec.constexprs {
        let s = match v {
            Val::Int(i) => format!("i{i}"),
            Val::Float(f) => format!("f{}", f.to_bits()),
            Val::Bool(b) => format!("b{b}"),
            Val::Str(s) => format!("s{s}"),
            Val::None => "none".to_string(),
            Val::Dtype(_) | Val::Seq(_) | Val::Ir(_) | Val::Desc { .. } | Val::Slice => {
                memo_eligible = false;
                continue;
            }
        };
        cvals.insert(name.as_str(), s);
    }
    for (name, v) in &cvals {
        key.push_str(name);
        key.push('\u{1}');
        key.push_str(v);
        key.push('\u{1}');
    }
    key.push('\0');
    for g in grid {
        key.push_str(&format!("g{g}\u{1}"));
    }
    key.push('\0');
    key.push_str(src);
    memo_eligible.then_some(key)
}

/// The compile itself, unstated above so the memo's key construction cannot interleave
/// with the ladder's phases.
fn compile_kernel_uncached(
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

/// ⛔ THE DEFAULT BINDING LAW — the node's operand prefix, the output LAST. Rope binds
/// three (a `RopeAppend`'s V and KV-cache destinations flow through GRAPH edges, so the
/// kernel consumes x, cos, sin only).
fn default_bindings<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
) -> Vec<BufferId> {
    let t =
        |tr: &scratchy_subtile::subtile_ir::TensorRegion| BufferId::new(tr.tensor.index() as u32);
    let n_bound = match &node.op {
        SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. } => 3,
        _ => node.inputs.len(),
    };
    let mut bindings: Vec<BufferId> = node.inputs.iter().take(n_bound).map(&t).collect();
    bindings.push(t(&node.output));
    bindings
}

/// ⭐ THE PYTHON PARSE, MEMOIZED BY SOURCE TEXT — the parse (a `Lexer`/`Parser` walk over
/// the WHOLE `.py` source) is a pure function of the source alone, and the placement
/// census calls [`operand_dfs`] once per NODE per rung per world
/// (`compute_bundle_layout`'s width walk), so the same kernel source was re-parsed
/// thousands of times in one expansion (MEASURED on the tp/8 fp8 bake, 2026-11-11: the
/// build script spent ~666% CPU / 40+ CPU-min / 17-25 GB RSS, with the parser at the
/// top of the profile). The key is the source text itself — the same law
/// [`memo_key`]'s source clause states — and the memo hands every caller the SAME
/// `Arc<PyModule>`, never a copy. The error is the parse's own display, memoized with
/// the success (the parse is pure, so a retry could only say the same thing); the
/// caller frames it with its own file name.
fn parse_py(src: &str) -> Result<std::sync::Arc<triton_frontend::py::ast::PyModule>, String> {
    static CACHE: Memo<triton_frontend::py::ast::PyModule> = Memo::new();
    CACHE.get_or_compute(Some(src.to_string()), || {
        triton_frontend::py::parser::parse(src).map_err(|e| e.to_string())
    })
}

/// ⭐⭐⭐ THE OPERAND FORMATS A SPLICED KERNEL STATES — every buffer parameter's
/// element format, as `(tensor id, format)` pairs in the node's OPERAND ORDER
/// ([`default_bindings`]' law: the node's inputs, the output LAST). This is the
/// PLACEMENT PASS's width authority: a tensor whose consumer is a spliced
/// kernel is sized at the format that kernel's signature states for it, never
/// at a per-op width census — a `[1, 64]` f32 table or an i32 idx stick states
/// its own element width in the kernel that reads it, and a placement sized at
/// the fp16 default would under-reserve by half and the footprint guard would
/// refuse the CORRECT descriptor.
///
/// The ORDER comes from the kernel's own declared parameter list — the
/// signature map is keyed by NAME and unordered — and each entry is the
/// SIGNATURE's stated pointer dtype, the same statement the compile itself
/// reads. `Ok(None)` for a kind that is not a `lower_one` kernel: a `Refused`
/// row's formats are another producer's statement, and attention — a Spliced
/// row that lowers through `lower_attn` — states its widths in its own operand
/// views, not here.
///
/// A stated dtype with no
/// device format is refused BY NAME, because sizing it at some other format's
/// bytes would turn the footprint guard into a rubber stamp for exactly the
/// access it exists to catch. ⚠ AN `Err` HERE IS THE SEAT'S OWN REFUSAL — the
/// same `kernel_spec` the compile runs — so a caller that does not itself
/// compile the node (the placement pass's width census) treats it as "no
/// formats from this seat" and leaves the refusal to the lowering, which
/// raises it loudly on every node it actually mints; a caller that does
/// compile should propagate it unchanged.
pub fn operand_dfs<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
) -> Result<Option<Vec<(u32, Df)>>, String> {
    // Attention is a Spliced ROW but not a `lower_one` kernel (its bindings are
    // `lower_attn`'s). This names the ROUTING, not a format.
    if matches!(node.op, SubOp::AttnDecode { .. }) {
        return Ok(None);
    }
    let row = row(&node.op);
    let TritonKernelRow::Spliced {
        src, file, entry, ..
    } = &row
    else {
        return Ok(None);
    };
    let spec = kernel_spec(node, ir, file, entry)?;
    let module = parse_py(src).map_err(|e| format!("triton splice: {file}: {e}"))?;
    let f = module
        .function(entry)
        .ok_or_else(|| format!("triton splice: {file} declares no function {entry}"))?;
    let mut dfs: Vec<Df> = Vec::new();
    for p in &f.params {
        if p.is_constexpr() {
            continue;
        }
        let Some(arg) = spec.signature.get(&p.name) else {
            return Err(format!(
                "triton splice: {file}'s {entry} parameter `{}` is missing from the signature",
                p.name
            ));
        };
        match arg {
            ArgSpec::Ptr(t) => dfs.push(df_of_stated_type(&p.name, t)?),
            // A scalar parameter is not an operand buffer; the zip with the
            // binding law below refuses any kernel whose parameters are not
            // exactly its operands.
            ArgSpec::Scalar(_) => {}
            ArgSpec::Constexpr => unreachable!("is_constexpr filtered the parameter above"),
        }
    }
    let bindings = default_bindings(node);
    if bindings.len() != dfs.len() {
        return Err(format!(
            "triton splice: {file}'s {entry} states {} buffer parameter(s) for {} operand(s) — \
             the registry's operand-order contract is violated",
            dfs.len(),
            bindings.len()
        ));
    }
    Ok(Some(
        bindings
            .into_iter()
            .zip(dfs)
            .map(|(b, df)| (b.get(), df))
            .collect(),
    ))
}

/// One stated pointer dtype → its device format
/// (F16→Fp16, F32→Fp32, F8E4M3FN→Fp8, i32→UInt32): a dtype with
/// no device format is refused BY NAME.
fn df_of_stated_type(param: &str, t: &triton_frontend::ttir::Type) -> Result<Df, String> {
    use triton_frontend::ttir::{FloatKind, Type};
    Ok(match t {
        Type::Float(FloatKind::F16) => Df::Fp16,
        Type::Float(FloatKind::F32) => Df::Fp32,
        Type::Float(FloatKind::F8E4M3FN) => Df::Fp8,
        Type::Int(32, _) => Df::UInt32,
        other => {
            return Err(format!(
                "triton splice: parameter `{param}` states element type {other:?}, which has no \
                 device format — the placement would be sized at some other format's bytes and \
                 the footprint guard would then compare an access against a fiction. Give the \
                 buffer a dtype the device has (f16, bf16, f32, f8E4M3FN, i32)."
            ));
        }
    })
}

/// Mint the [`KtirNode`]: the adapter's `node_for` compilation, then RE-STATE the
/// bindings the tape owns (resolved by [`default_bindings`]: the tape's tensor indices,
/// the output LAST).
///
/// ⛔ THE RE-STATE IS THE WHOLE POINT. `node_for` mints positional `BufferId`s
/// (`0, 1, 2`); the tape's rmsnorm node reads `t37`/`t41` and writes `t42`. The
/// bindings are ours to state, because only the splice knows which graph tensor each
/// kernel parameter addresses.
fn mint<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    compiled: &CompiledKernel,
    program: &Program,
    bindings: Vec<BufferId>,
) -> Result<KtirNode, String> {
    // The adapter's mint: `to_ktir_emit::lower` over the module, positional buffer ids —
    // KEYED by the compile's own memo key (riding inside the `CompiledKernel` the memo
    // minted), so a kernel spliced once per layer per rung lowers its handoff ONCE, not
    // once per node.
    let positional = crate::compile::node_for_compiled(compiled, *program)
        .map_err(|e| format!("triton splice: node_for: {e}"))?;
    // ⭐ ONE NAMING LAW, so the op_name and the emulator's function key agree — what the
    // golden pins.
    let name = Arena::global().str(format!(
        "{}_s{}",
        program_stem(node, program),
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
    })
}

/// The program stem of this node's name (`rmsnorm_s{id}`, `add_s{id}`, …). ⛔ READ OFF THE
/// NODE, NOT THE ROW: the elementwise rows carry the CONSUMER's kind (`Elementwise::Add`),
/// and the name comes from the PRODUCER's `EwKind` (`ew_kind_stem` — BiasAdd is named `add`, not `biasadd`). One
/// row may therefore mint several stems; the node states which.
fn program_stem<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    program: &Program,
) -> &'static str {
    if let SubOp::Elementwise(kind) = &node.op {
        return match kind {
            EwKind::Add | EwKind::BiasAdd => "add",
            EwKind::Mul => "mul",
            EwKind::Sub => "sub",
            EwKind::Silu => "silu",
            EwKind::Gelu => "gelu",
            EwKind::QuickGelu => "quickgelu",
            EwKind::GeluErf => "geluerf",
            EwKind::Relu2 => "relu2",
        };
    }
    // ⛔ NO `_` ARM: a new program kind names its stem here or the build fails.
    match program {
        Program::RmsNorm => "rmsnorm",
        Program::SiluMul => "silumul",
        Program::Matmul => "matmul",
        Program::Rope => "rope",
        Program::ScalarMul => "scalarmul",
        // ⭐ THE SANDWICH'S OWN STEM — `rmsnorm_s{split.id}` (the split's id, not the
        // middle norm's), main's naming law for the fused per-head norm.
        Program::WindowedRmsNorm { .. } => "rmsnorm",
        Program::RmsNormUnit => "rmsnormunit",
        Program::TanhSoftCap => "tanhsoftcap",
        Program::ScalarWeightMul => "scalarwmul",
        Program::Reshape => "reshape",
        Program::BlockCopy => "blockcopy",
        Program::Elementwise(_) => "elementwise",
        Program::LmLast => "lmlast",
        Program::PagedAttn => "pagedattn",
        Program::PagedKvWrite => "pagedkvwrite",
        Program::Transpose => "transpose",
    }
}

#[cfg(test)]
mod production_ktir_dump {
    //! ⭐ THE LEGALIZE-TYPES BYTE-IDENTITY PROOF DRIVER (auditor order, 2026-10-26):
    //! `LegalizeTypes` runs on EVERY kernel the ladder compiles, so a change that widens
    //! what stays f32 could move bytes in kernels that work today. This test dumps the
    //! post-`to_ktir` module text — `triton_ktir::text::print::print`, the deterministic
    //! module printer — for EVERY production kernel in `backends/spyre/kernels/`, each at one
    //! representative admitted geometry, into `<workspace>/target/spyre-ktir-dump/`.
    //!
    //! The proof is the EXTERNAL DIFF, not this test: run once with the tree's
    //! `legalize_types.rs`, once with the pre-change file swapped in, and diff the two dump
    //! directories — byte-identical per kernel is the bar, and any nonzero diff names the
    //! kernel it stopped on. That is why the test is `#[ignore]`d: it produces evidence,
    //! asserts nothing about the numbers, and would be dead weight in every ordinary test
    //! run. Run it explicitly with
    //! `cargo test -p scratchy-triton-splice --lib production_ktir_dump -- --ignored`.
    //!
    //! Every spec below mirrors the `kernel_spec` arm (or `lower_attn` / `lower_kv_write` /
    //! `lower_sandwich_node` / `lower_prefill_lm_head_fold` / the MoE numeric tests'
    //! golden geometry) that the production bake states for that kernel — cited per entry
    //! — at a whole-region decode-shaped geometry so every blocking law answers the
    //! one-tile constants. The compile goes through the crate's own [`compile_kernel`]:
    //! the exact memoized ladder (`codegen::compile` → `make_ttir` → `from_ttir` →
    //! `make_ktir` → `to_ktir`) the bake runs, so the dump cannot drift from what
    //! production emits.
    use super::*;

    const PAGED_DECODE: &str = include_str!("../../../../kernels/page_attn_head_major_decode.py");
    const PAGED_PREFILL: &str = include_str!("../../../../kernels/page_attn_head_major_prefill.py");
    const PAGED_KV_WRITE: &str =
        include_str!("../../../../kernels/reshape_and_cache_head_major.py");
    const BATCHED_DECODE: &str = include_str!("../../../../kernels/batched_decode_head_major.py");
    const PAGED_DECODE_FP8: &str =
        include_str!("../../../../kernels/page_attn_head_major_decode_fp8.py");
    const PAGED_PREFILL_FP8: &str =
        include_str!("../../../../kernels/page_attn_head_major_prefill_fp8.py");
    const PAGED_KV_WRITE_FP8: &str =
        include_str!("../../../../kernels/reshape_and_cache_head_major_fp8.py");
    const BATCHED_DECODE_FP8: &str =
        include_str!("../../../../kernels/batched_decode_head_major_fp8.py");
    const ELEMENTWISE: &str = include_str!("../../../../kernels/elementwise.py");
    const LMLAST: &str = include_str!("../../../../kernels/lmlast.py");
    const MATMUL: &str = include_str!("../../../../kernels/matmul.py");
    const MATMUL_FP8: &str = include_str!("../../../../kernels/matmul_fp8.py");
    const MOE: &str = include_str!("../../../../kernels/moe.py");
    const MOE_PREFILL: &str = include_str!("../../../../kernels/moe_prefill.py");
    const RESHAPE: &str = include_str!("../../../../kernels/reshape.py");
    const RMSNORM: &str = include_str!("../../../../kernels/rmsnorm.py");
    const RMSNORMUNIT: &str = include_str!("../../../../kernels/rmsnormunit.py");
    const ROPE: &str = include_str!("../../../../kernels/rope.py");
    const SCALARMUL: &str = include_str!("../../../../kernels/scalarmul.py");
    const SCALARWMUL: &str = include_str!("../../../../kernels/scalarwmul.py");
    const SILUMUL: &str = include_str!("../../../../kernels/silumul.py");
    const TANHSOFTCAP: &str = include_str!("../../../../kernels/tanhsoftcap.py");
    const WINDOWEDRMSNORM: &str = include_str!("../../../../kernels/windowedrmsnorm.py");

    /// The case table's `spec` shape: pointer parameters by stated dtype, then the
    /// constexprs — every constexpr is BOTH a signature entry and a binding.
    fn spec(file: &str, kernel: &str, ptrs: &[(&str, &str)], ces: &[(&str, Val)]) -> KernelSpec {
        let mut signature = HashMap::new();
        for (k, v) in ptrs {
            signature.insert(
                (*k).to_string(),
                ArgSpec::parse(v)
                    .unwrap_or_else(|e| panic!("{file}:{kernel}: bad spec {k}={v}: {e}")),
            );
        }
        let mut constexprs = HashMap::new();
        for (k, v) in ces {
            signature.insert((*k).to_string(), ArgSpec::Constexpr);
            constexprs.insert((*k).to_string(), v.clone());
        }
        KernelSpec {
            kernel: kernel.to_string(),
            signature,
            constexprs,
            file: file.to_string(),
        }
    }

    /// One entry per (kernel file, entry point) — the multi-entry files (`elementwise.py`,
    /// `rmsnorm.py`, `windowedrmsnorm.py`, `matmul.py`, `page_attn_head_major_*.py`) state every entry, so a
    /// guard that only one arm trips is itself covered.
    fn entries() -> Vec<(&'static str, &'static str, KernelSpec, Vec<i64>)> {
        // The whole-region pointwise constants every `blocks_of(1, c, live)` answers.
        let ew = |file: &str, kernel: &str, ptrs: &[(&str, &str)], live: u32| {
            let (block_m, n_blocks, tail_h) = blocks_of(1, 2048, live);
            spec(
                file,
                kernel,
                ptrs,
                &[
                    ("M", Val::Int(1)),
                    ("N", Val::Int(2048)),
                    ("BLOCK_M", Val::Int(i128::from(block_m))),
                    ("BLOCK_N", Val::Int(2048)),
                    ("N_TOTAL", Val::Int(2048)),
                    ("C_START", Val::Int(0)),
                    ("N_BLOCKS", Val::Int(i128::from(n_blocks))),
                    ("TAIL_H", Val::Int(i128::from(tail_h))),
                ],
            )
        };
        let mut v: Vec<(&'static str, &'static str, KernelSpec, Vec<i64>)> = Vec::new();
        // ── rmsnorm.py: the Scale arm and the (1 + w) form's own spec. ──
        for entry in ["rmsnorm_fwd", "rmsnorm_oneplus_fwd"] {
            let mut ces = vec![
                ("M", Val::Int(1)),
                ("D_MODEL", Val::Int(2048)),
                ("BLOCK_M", Val::Int(1)),
                ("EPS", Val::Float(1e-5)),
                ("INV_D", Val::Float(1.0 / 2048.0)),
            ];
            if entry == "rmsnorm_oneplus_fwd" {
                ces.push((
                    "GAIN_OFFSET",
                    Val::Float(f64::from(GainConvention::OnePlusScale.offset())),
                ));
            }
            v.push((
                entry,
                RMSNORM,
                spec(
                    "rmsnorm.py",
                    entry,
                    &[
                        ("desc_x", "*fp16"),
                        ("desc_w", "*fp16"),
                        ("desc_o", "*fp16"),
                    ],
                    &ces,
                ),
                vec![1],
            ));
        }
        // ── matmul.py (Dense): N is the DEVICE width, the arm's own padding law. ──
        let n_dev = stick_width(2048);
        v.push((
            "matmul_fwd",
            MATMUL,
            spec(
                "matmul.py",
                "matmul_fwd",
                &[
                    ("desc_a", "*fp16"),
                    ("desc_w", "*fp16"),
                    ("desc_o", "*fp16"),
                ],
                &[
                    ("M", Val::Int(1)),
                    ("K", Val::Int(2048)),
                    ("N", Val::Int(i128::from(n_dev))),
                    ("BLOCK_M", Val::Int(1)),
                    ("BLOCK_K", Val::Int(2048)),
                    ("BLOCK_N", Val::Int(i128::from(n_dev))),
                    ("M_TOTAL", Val::Int(1)),
                ],
            ),
            vec![1],
        ));
        // ── matmul_fp8.py: the arity-3 twin, LOGICAL widths, one whole-region tile. ──
        v.push((
            "matmul_fp8_fwd",
            MATMUL_FP8,
            spec(
                "matmul_fp8.py",
                "matmul_fp8_fwd",
                &[
                    ("desc_x", "*fp16"),
                    ("desc_w", "*fp8e4nv"),
                    ("desc_ws", "*fp16"),
                    ("desc_o", "*fp16"),
                ],
                &[
                    ("M", Val::Int(1)),
                    ("K", Val::Int(2048)),
                    ("N", Val::Int(2048)),
                    ("BLOCK_M", Val::Int(1)),
                    ("BLOCK_K", Val::Int(2048)),
                    ("BLOCK_N", Val::Int(2048)),
                    ("M_TOTAL", Val::Int(1)),
                ],
            ),
            vec![1],
        ));
        // ── elementwise.py: every binary entry plus both unary acts. ──
        for (entry, ptrs, live) in [
            (
                "add_fwd",
                &[
                    ("desc_a", "*fp16"),
                    ("desc_b", "*fp16"),
                    ("desc_o", "*fp16"),
                ][..],
                EW_BINARY_LIVE_TILES,
            ),
            (
                "mul_fwd",
                &[
                    ("desc_a", "*fp16"),
                    ("desc_b", "*fp16"),
                    ("desc_o", "*fp16"),
                ][..],
                EW_BINARY_LIVE_TILES,
            ),
            (
                "sub_fwd",
                &[
                    ("desc_a", "*fp16"),
                    ("desc_b", "*fp16"),
                    ("desc_o", "*fp16"),
                ][..],
                EW_BINARY_LIVE_TILES,
            ),
            (
                "silu_fwd",
                &[("desc_x", "*fp16"), ("desc_o", "*fp16")][..],
                EW_SILU_LIVE_TILES,
            ),
            (
                "gelu_fwd",
                &[("desc_x", "*fp16"), ("desc_o", "*fp16")][..],
                EW_SILU_LIVE_TILES,
            ),
        ] {
            v.push((
                entry,
                ELEMENTWISE,
                ew("elementwise.py", entry, ptrs, live),
                vec![1],
            ));
        }
        // ── silumul.py: the [M, N] gate·up product, whole-region single tile. ──
        v.push((
            "silumul_fwd",
            SILUMUL,
            ew(
                "silumul.py",
                "silumul_fwd",
                &[
                    ("desc_g", "*fp16"),
                    ("desc_u", "*fp16"),
                    ("desc_o", "*fp16"),
                ],
                SILU_MUL_LIVE_TILES,
            ),
            vec![1],
        ));
        // ── scalarmul.py: the WINDOW is the device width — whole sticks on both N and
        // N_TOTAL, the arm's own law. ──
        {
            let c_dev = stick_width(2048);
            let (_, n_blocks, tail_h) = blocks_of(1, c_dev, 3);
            v.push((
                "scalarmul_fwd",
                SCALARMUL,
                spec(
                    "scalarmul.py",
                    "scalarmul_fwd",
                    &[("desc_x", "*fp16"), ("desc_o", "*fp16")],
                    &[
                        ("M", Val::Int(1)),
                        ("N", Val::Int(i128::from(c_dev))),
                        ("BLOCK_M", Val::Int(1)),
                        ("BLOCK_N", Val::Int(i128::from(c_dev))),
                        ("N_TOTAL", Val::Int(i128::from(c_dev))),
                        ("C_START", Val::Int(0)),
                        ("N_BLOCKS", Val::Int(i128::from(n_blocks))),
                        ("TAIL_H", Val::Int(i128::from(tail_h))),
                        ("SCALE", Val::Float(8.0)),
                    ],
                ),
                vec![1],
            ));
        }
        // ── rope.py: a whole-region node — R_START = 0, Y_TOTAL = MQ·H. ──
        v.push((
            "rope_fwd",
            ROPE,
            spec(
                "rope.py",
                "rope_fwd",
                &[
                    ("desc_x", "*fp16"),
                    ("desc_cos", "*fp16"),
                    ("desc_sin", "*fp16"),
                    ("desc_o", "*fp16"),
                ],
                &[
                    ("H", Val::Int(8)),
                    ("MQ", Val::Int(1)),
                    ("HEAD_DIM", Val::Int(128)),
                    ("HALF", Val::Int(64)),
                    ("Y_TOTAL", Val::Int(8)),
                    ("R_START", Val::Int(0)),
                ],
            ),
            vec![1],
        ));
        // ── tanhsoftcap.py: the vocab-wide cap, three live tiles. ──
        {
            let (_, n_blocks, tail_h) = blocks_of(1, 8192, 3);
            v.push((
                "tanhsoftcap_fwd",
                TANHSOFTCAP,
                spec(
                    "tanhsoftcap.py",
                    "tanhsoftcap_fwd",
                    &[("desc_x", "*fp16"), ("desc_o", "*fp16")],
                    &[
                        ("M", Val::Int(1)),
                        ("N", Val::Int(8192)),
                        ("BLOCK_M", Val::Int(1)),
                        ("BLOCK_N", Val::Int(8192)),
                        ("CAP", Val::Float(30.0)),
                        ("N_BLOCKS", Val::Int(i128::from(n_blocks))),
                        ("TAIL_H", Val::Int(i128::from(tail_h))),
                    ],
                ),
                vec![1],
            ));
        }
        // ── rmsnormunit.py: the rmsnorm arm's constants minus the weight. ──
        v.push((
            "rmsnormunit_fwd",
            RMSNORMUNIT,
            spec(
                "rmsnormunit.py",
                "rmsnormunit_fwd",
                &[("desc_x", "*fp16"), ("desc_o", "*fp16")],
                &[
                    ("M", Val::Int(1)),
                    ("D_MODEL", Val::Int(128)),
                    ("BLOCK_M", Val::Int(1)),
                    ("EPS", Val::Float(1e-5)),
                    ("INV_D", Val::Float(1.0 / 128.0)),
                ],
            ),
            vec![1],
        ));
        // ── scalarwmul.py: `w` is a RUNTIME fp16 SCALAR, not a descriptor. ──
        v.push((
            "scalarwmul_fwd",
            SCALARWMUL,
            spec(
                "scalarwmul.py",
                "scalarwmul_fwd",
                &[("desc_x", "*fp16"), ("w", "fp16"), ("desc_o", "*fp16")],
                &[("M", Val::Int(1)), ("N", Val::Int(2048))],
            ),
            vec![1],
        ));
        // ── reshape.py: both sides whole sticks, element count preserved. ──
        v.push((
            "reshape_fwd",
            RESHAPE,
            spec(
                "reshape.py",
                "reshape_fwd",
                &[("desc_src", "*fp16"), ("desc_dst", "*fp16")],
                &[
                    ("R_IN", Val::Int(1)),
                    ("C_IN", Val::Int(1024)),
                    ("R_OUT", Val::Int(2)),
                    ("C_OUT", Val::Int(512)),
                    ("N_STICKS", Val::Int(16)),
                ],
            ),
            vec![1],
        ));
        // ── lmlast.py: the extraction, one entry per request count — k = 1, 2 and 3 (3 is
        // not a power of two), each at w = 5 so MQ is the pair's own product. ──
        for (name, s) in [
            ("lmlast_fwd_s1", 1i128),
            ("lmlast_fwd_s2", 2),
            ("lmlast_fwd_s3", 3),
        ] {
            v.push((
                name,
                LMLAST,
                spec(
                    "lmlast.py",
                    "lmlast_fwd",
                    &[("desc_src", "*fp16"), ("desc_dst", "*fp16")],
                    &[
                        ("MQ", Val::Int(5 * s)),
                        ("HIDDEN", Val::Int(2048)),
                        ("W", Val::Int(5)),
                        ("S", Val::Int(s)),
                        ("N_STICKS", Val::Int(32)),
                    ],
                ),
                vec![1],
            ));
        }
        // ── windowedrmsnorm.py: all three entries of the one body. ──
        for (entry, gained, oneplus) in [
            ("windowedrmsnorm_gained_fwd", true, false),
            ("windowedrmsnorm_oneplus_fwd", true, true),
            ("windowedrmsnorm_unit_fwd", false, false),
        ] {
            let ptrs: &[(&str, &str)] = if gained {
                &[
                    ("desc_x", "*fp16"),
                    ("desc_gamma", "*fp16"),
                    ("desc_o", "*fp16"),
                ]
            } else {
                &[("desc_x", "*fp16"), ("desc_o", "*fp16")]
            };
            let mut ces = vec![
                ("M", Val::Int(1)),
                ("D", Val::Int(128)),
                ("HEADS", Val::Int(8)),
                ("EPS", Val::Float(1e-6)),
                ("INV_D", Val::Float(1.0 / 128.0)),
            ];
            if oneplus {
                ces.push((
                    "GAIN_OFFSET",
                    Val::Float(f64::from(GainConvention::OnePlusScale.offset())),
                ));
            }
            v.push((
                entry,
                WINDOWEDRMSNORM,
                spec("windowedrmsnorm.py", entry, ptrs, &ces),
                vec![1],
            ));
        }
        // ── spyre-inference's paged step: `page_attn_head_major_decode` (one query row) and
        // `page_attn_head_major_prefill` (a prompt chunk) at granite's head geometry over a four-page
        // pool (`lower_attn`'s spec), and `reshape_and_cache_head_major`: one decode token's write
        // (`lower_kv_write`'s spec) at granite's 8 kv heads and at 2. Grid [1]. ──
        // ...plus the walk at granite-3.1-2b's own pool — `max_position_embeddings` 131072 in 2048
        // pages — decoding over the whole of it and a prompt chunk walking four pages, so the card
        // compile is checked at the extents a bake states, not only at a toy pool.
        for (name, mq, blocks, pages, hd, pool, (heads, kvh)) in [
            (
                "paged_attn_decode",
                1i128,
                4i128,
                4i128,
                64i128,
                KvPool::Fp16,
                (32, 8),
            ),
            ("paged_attn_prefill", 64, 1, 4, 64, KvPool::Fp16, (32, 8)),
            (
                "paged_attn_decode_pool",
                1,
                2048,
                2048,
                64,
                KvPool::Fp16,
                (32, 8),
            ),
            (
                "paged_attn_prefill_walk",
                64,
                4,
                2048,
                64,
                KvPool::Fp16,
                (32, 8),
            ),
            // granite-3.1-8b's head dim: a kv-head slab is two sticks wide.
            (
                "paged_attn_decode_hd128",
                1,
                4,
                4,
                128,
                KvPool::Fp16,
                (32, 8),
            ),
            (
                "paged_attn_prefill_hd128",
                64,
                4,
                4,
                128,
                KvPool::Fp16,
                (32, 8),
            ),
            // under Triton's 8-row gather floor: the rows read and written at the first row
            // `query_row_index` names.
            ("paged_attn_prefill_p7", 7, 1, 4, 64, KvPool::Fp16, (32, 8)),
            // a prompt chunk over granite-3.1-2b's whole pool, sixteen pages a trip
            (
                "paged_attn_prefill_trips",
                64,
                1024,
                1024,
                64,
                KvPool::Fp16,
                (32, 8),
            ),
            // the fp8 pool's twins, at the head dim whose fp8 row is whole sticks
            (
                "paged_attn_decode_hd128_fp8",
                1,
                4,
                4,
                128,
                KvPool::Fp8,
                (32, 8),
            ),
            (
                "paged_attn_prefill_hd128_fp8",
                64,
                4,
                4,
                128,
                KvPool::Fp8,
                (32, 8),
            ),
            ("paged_attn_decode_fp8", 1, 4, 4, 64, KvPool::Fp8, (32, 8)),
            ("paged_attn_prefill_fp8", 64, 1, 4, 64, KvPool::Fp8, (32, 8)),
            (
                "paged_attn_prefill_trips_fp8",
                64,
                1024,
                1024,
                64,
                KvPool::Fp8,
                (32, 8),
            ),
            // gemma-4's two classes: the sliding layers' 8 kv heads of 256, the global layers' one
            // kv head of 512, both under 16 query heads
            (
                "paged_attn_decode_g256",
                1,
                4,
                4,
                256,
                KvPool::Fp16,
                (16, 8),
            ),
            (
                "paged_attn_decode_g512",
                1,
                4,
                4,
                512,
                KvPool::Fp16,
                (16, 1),
            ),
            (
                "paged_attn_prefill_g512",
                64,
                4,
                4,
                512,
                KvPool::Fp16,
                (16, 1),
            ),
            (
                "paged_attn_decode_g512_fp8",
                1,
                4,
                4,
                512,
                KvPool::Fp8,
                (16, 1),
            ),
            (
                "paged_attn_prefill_g256_fp8",
                64,
                4,
                4,
                256,
                KvPool::Fp8,
                (16, 8),
            ),
        ] {
            let decode = mq == 1;
            let ptrs: Vec<(&str, &str)> = vec![
                ("query", "*fp16"),
                ("query_row_index", "*i32"),
                ("k_pages", pool.pointer()),
                ("v_pages", pool.pointer()),
                ("page_index_table", "*i32"),
                ("mask_stack", "*fp16"),
                ("out", "*fp16"),
            ];
            let mut ces = vec![
                ("NUM_TOKENS", Val::Int(mq)),
                ("NUM_HEADS", Val::Int(heads)),
                ("NUM_KV_HEADS", Val::Int(kvh)),
                ("HEAD_SIZE", Val::Int(hd)),
                (
                    "BLOCK_SIZE",
                    Val::Int(i128::from(SchedulerBlockTokens::REFERENCE.get())),
                ),
                ("NUM_PAGES", Val::Int(pages)),
                ("NUM_BLOCKS", Val::Int(blocks)),
                ("SCALE", Val::Float(0.125)),
            ];
            let (src, file, entry) = if decode {
                (
                    pool.pick(PAGED_DECODE, PAGED_DECODE_FP8),
                    pool.pick(
                        "page_attn_head_major_decode.py",
                        "page_attn_head_major_decode_fp8.py",
                    ),
                    pool.pick(
                        "page_attn_head_major_decode_kernel",
                        "page_attn_head_major_decode_fp8_kernel",
                    ),
                )
            } else {
                ces.push(("PADDED_QUERY_LEN", Val::Int(mq)));
                ces.push((
                    "BLOCKS_PER_TRIP",
                    Val::Int(i128::from(trip_pages(blocks as u32))),
                ));
                (
                    pool.pick(PAGED_PREFILL, PAGED_PREFILL_FP8),
                    pool.pick(
                        "page_attn_head_major_prefill.py",
                        "page_attn_head_major_prefill_fp8.py",
                    ),
                    pool.pick(
                        "page_attn_head_major_prefill_kernel",
                        "page_attn_head_major_prefill_fp8_kernel",
                    ),
                )
            };
            v.push((name, src, spec(file, entry, &ptrs, &ces), vec![1]));
        }
        // ── and the PREFILL kernel at a PACKED rung's split extents (M2's S=2 shape, stated at the
        // splice level): `NUM_TOKENS` is the graph's rows (two requests of 64) while
        // `PADDED_QUERY_LEN` and the walk are ONE request's 64 rows — the two extents
        // [`QueryExtent`] separates, compiled at the pair a (64, 2) rung mints. The walk is the
        // rung's own law at `ActiveCap::NONE` (a start==0 chunk: the pages its own per-request
        // rows fill), stated by the same [`walk_pages`] the splice bakes it with.
        {
            let ptrs: Vec<(&str, &str)> = vec![
                ("query", "*fp16"),
                ("query_row_index", "*i32"),
                ("k_pages", "*fp16"),
                ("v_pages", "*fp16"),
                ("page_index_table", "*i32"),
                ("mask_stack", "*fp16"),
                ("out", "*fp16"),
            ];
            let blocks = walk_pages(4, 64, scratchy_spyre_bundle::program::ActiveCap::NONE);
            let ces = vec![
                ("NUM_TOKENS", Val::Int(128)),
                ("NUM_HEADS", Val::Int(32)),
                ("NUM_KV_HEADS", Val::Int(8)),
                ("HEAD_SIZE", Val::Int(64)),
                (
                    "BLOCK_SIZE",
                    Val::Int(i128::from(SchedulerBlockTokens::REFERENCE.get())),
                ),
                ("NUM_PAGES", Val::Int(4)),
                ("NUM_BLOCKS", Val::Int(i128::from(blocks))),
                ("PADDED_QUERY_LEN", Val::Int(64)),
                ("BLOCKS_PER_TRIP", Val::Int(i128::from(trip_pages(blocks)))),
                ("SCALE", Val::Float(0.125)),
            ];
            v.push((
                "paged_attn_prefill_packed",
                PAGED_PREFILL,
                spec(
                    "page_attn_head_major_prefill.py",
                    "page_attn_head_major_prefill_kernel",
                    &ptrs,
                    &ces,
                ),
                vec![1],
            ));
        }
        // ...and `batched_decode_head_major` over that pool, the chunkings a decode batch rung
        // states (`BatchChunking::of`): 4 sequences on the 4-page rung (one chunk), on the 64-page
        // rung (8 chunks, each block slot its own softmax), 2 sequences on the 2-page rung (four
        // pages, 32 kv rows), and 32 on the 4-page rung (4 chunks of ONE block slot —
        // the widest batch rung, where the slots' merge folds one). At head dim 128 (granite-3.1-8b)
        // the one-chunk and many-chunk batches, and the widest rung on its one-page walk.
        for (name, seqs, walk, hd, pool) in [
            ("batched_decode", 4u32, 4u32, 64i128, KvPool::Fp16),
            ("batched_decode_chunks", 4, 64, 64, KvPool::Fp16),
            ("batched_decode_pair", 2, 2, 64, KvPool::Fp16),
            ("batched_decode_one_slot", 32, 4, 64, KvPool::Fp16),
            ("batched_decode_hd128", 4, 4, 128, KvPool::Fp16),
            ("batched_decode_chunks_hd128", 4, 64, 128, KvPool::Fp16),
            ("batched_decode_one_slot_hd128", 32, 1, 128, KvPool::Fp16),
            ("batched_decode_hd128_fp8", 4, 4, 128, KvPool::Fp8),
            ("batched_decode_one_slot_hd128_fp8", 32, 1, 128, KvPool::Fp8),
            ("batched_decode_fp8", 4, 4, 64, KvPool::Fp8),
            ("batched_decode_chunks_fp8", 4, 64, 64, KvPool::Fp8),
        ] {
            let c = BatchChunking::of(
                DecodeWalk {
                    seqs,
                    blocks: walk,
                    pages: 2048,
                },
                DecodeHeads {
                    kv_heads: 8,
                    q_width: 32 * hd as u32,
                    head_size: hd as u32,
                },
            );
            v.push((
                name,
                pool.pick(BATCHED_DECODE, BATCHED_DECODE_FP8),
                spec(
                    pool.pick(
                        "batched_decode_head_major.py",
                        "batched_decode_head_major_fp8.py",
                    ),
                    pool.pick(
                        "batched_decode_head_major_kernel",
                        "batched_decode_head_major_fp8_kernel",
                    ),
                    &[
                        ("query", "*fp16"),
                        ("rep_row_ids", "*i32"),
                        ("k_pages", pool.pointer()),
                        ("v_pages", pool.pointer()),
                        ("chunk_page_ids", "*i32"),
                        ("mask_by_chunk", "*fp16"),
                        ("out", "*fp16"),
                    ],
                    &[
                        ("NUM_TOKENS", Val::Int(i128::from(seqs))),
                        ("NUM_SEQS", Val::Int(i128::from(c.seqs))),
                        ("BLOCKS_PER_CHUNK", Val::Int(i128::from(c.blocks_per_chunk))),
                        ("NUM_CHUNKS", Val::Int(i128::from(c.num_chunks))),
                        ("NUM_KV_HEADS", Val::Int(8)),
                        ("NUM_QUERIES_PER_KV", Val::Int(4)),
                        (
                            "BLOCK_SIZE",
                            Val::Int(i128::from(SchedulerBlockTokens::REFERENCE.get())),
                        ),
                        ("HEAD_SIZE", Val::Int(hd)),
                        ("NUM_PAGES", Val::Int(2048)),
                        ("ROWS_PER_PAGE", Val::Int(i128::from(c.rows_per_page))),
                        ("SCALE", Val::Float(0.125)),
                    ],
                ),
                vec![1],
            ));
        }
        // A decode step's write (8 kv heads, one store per head), one below the scatter floor (2
        // heads, one store per row), a 20-token prompt chunk's (160 index entries, five scatter ops
        // of 32), and at head dim 128 the first two and a prompt chunk's.
        for (name, t, nkvh, hd, pool) in [
            ("paged_kv_write", 1i128, 8i128, 64i128, KvPool::Fp16),
            ("paged_kv_write_rows", 1, 2, 64, KvPool::Fp16),
            ("paged_kv_write_prompt", 20, 8, 64, KvPool::Fp16),
            ("paged_kv_write_hd128", 1, 8, 128, KvPool::Fp16),
            ("paged_kv_write_rows_hd128", 1, 2, 128, KvPool::Fp16),
            ("paged_kv_write_prompt_hd128", 96, 8, 128, KvPool::Fp16),
            ("paged_kv_write_hd128_fp8", 1, 8, 128, KvPool::Fp8),
            ("paged_kv_write_prompt_hd128_fp8", 96, 8, 128, KvPool::Fp8),
            ("paged_kv_write_fp8", 1, 8, 64, KvPool::Fp8),
            ("paged_kv_write_prompt_fp8", 20, 8, 64, KvPool::Fp8),
        ] {
            let mut ces = vec![
                ("T", Val::Int(t)),
                ("NUM_KV_HEADS", Val::Int(nkvh)),
                ("HEAD_SIZE", Val::Int(hd)),
                (
                    "NUM_ROWS",
                    Val::Int(4 * nkvh * i128::from(SchedulerBlockTokens::REFERENCE.get())),
                ),
            ];
            if let PageOrder::TokenMajor = pool.page_order() {
                ces.push(("STICK", Val::Int(i128::from(pool.stick_lanes()))));
            }
            v.push((
                name,
                pool.pick(PAGED_KV_WRITE, PAGED_KV_WRITE_FP8),
                spec(
                    pool.pick(
                        "reshape_and_cache_head_major.py",
                        "reshape_and_cache_head_major_fp8.py",
                    ),
                    pool.pick(
                        "reshape_and_cache_head_major_kernel",
                        "reshape_and_cache_head_major_fp8_kernel",
                    ),
                    &[
                        ("key", "*fp16"),
                        ("value", "*fp16"),
                        ("k_rows", pool.pointer()),
                        ("v_rows", pool.pointer()),
                        ("row_index", "*i32"),
                    ],
                    &ces,
                ),
                vec![1],
            ));
        }
        // ── moe.py / moe_prefill.py: the MoE numeric tests' golden geometry — the
        // smallest geometry the real-Triton compile admits — at grid [1]. ──
        let h = 128.0f64;
        let moe_ces = |rows: &'static str, m: i128| -> Vec<(&'static str, Val)> {
            vec![
                (rows, Val::Int(m)),
                ("H", Val::Int(128)),
                ("I", Val::Int(64)),
                ("E", Val::Int(64)),
                ("K", Val::Int(8)),
                ("EPS", Val::Float(1e-6)),
                ("INV_H", Val::Float(1.0 / h)),
                ("TEMP", Val::Float(h.powf(-0.5))),
                ("INV_QMAX", Val::Float(1.0 / 448.0)),
                ("SCALE_FLOOR", Val::Float(1.0 / 448.0 / 448.0)),
            ]
        };
        let moe_ptrs: &[(&str, &str)] = &[
            ("desc_xr", "*fp16"),
            ("desc_rg", "*fp16"),
            ("desc_rw", "*fp16"),
            ("desc_pes", "*fp16"),
            ("desc_xe", "*fp16"),
            ("desc_wg", "*fp8e4nv"),
            ("desc_sg", "*fp16"),
            ("desc_wu", "*fp8e4nv"),
            ("desc_su", "*fp16"),
            ("desc_wd", "*fp8e4nv"),
            ("desc_sd", "*fp16"),
            ("desc_o", "*fp16"),
            ("desc_ids", "*i32"),
            ("desc_rs", "*fp16"),
            ("desc_act", "*fp16"),
        ];
        v.push((
            "moe_fwd",
            MOE,
            spec("moe.py", "moe_fwd", moe_ptrs, &moe_ces("M", 1)),
            vec![1],
        ));
        // `moe_prefill.py`'s signature: `moe.py`'s first twelve, then the dense route
        // (`desc_dense`) — no id/scratch sticks, the route is dense in the prompt form.
        v.push((
            "moe_prefill_fwd",
            MOE_PREFILL,
            spec(
                "moe_prefill.py",
                "moe_prefill_fwd",
                &[
                    ("desc_xr", "*fp16"),
                    ("desc_rg", "*fp16"),
                    ("desc_rw", "*fp16"),
                    ("desc_pes", "*fp16"),
                    ("desc_xe", "*fp16"),
                    ("desc_wg", "*fp8e4nv"),
                    ("desc_sg", "*fp16"),
                    ("desc_wu", "*fp8e4nv"),
                    ("desc_su", "*fp16"),
                    ("desc_wd", "*fp8e4nv"),
                    ("desc_sd", "*fp16"),
                    ("desc_o", "*fp16"),
                    ("desc_dense", "*fp16"),
                ],
                &moe_ces("T", 16),
            ),
            vec![1],
        ));
        v
    }

    /// The `Program` each dumped kernel's registry row states — the same fact
    /// `TritonKernelRow::Spliced { program }` (or the attention/MoE producers) carries,
    /// restated here because the entries table is keyed by kernel name.
    /// ⛔ NO `_` ARM: a new entry without a program here is a panic naming it, never a
    /// silent skip — the no-silent-skip law, applied to the dump driver itself.
    fn program_of(name: &str) -> Program {
        match name {
            "rmsnorm_fwd" | "rmsnorm_oneplus_fwd" => Program::RmsNorm,
            "matmul_fwd" | "matmul_fp8_fwd" => Program::Matmul,
            "add_fwd" => Program::Elementwise(Elementwise::Add),
            "mul_fwd" => Program::Elementwise(Elementwise::Mul),
            "sub_fwd" => Program::Elementwise(Elementwise::Sub),
            "silu_fwd" => Program::Elementwise(Elementwise::Silu),
            "gelu_fwd" => Program::Elementwise(Elementwise::Gelu),
            "silumul_fwd" => Program::SiluMul,
            "scalarmul_fwd" => Program::ScalarMul,
            "tanhsoftcap_fwd" => Program::TanhSoftCap,
            "rmsnormunit_fwd" => Program::RmsNormUnit,
            "scalarwmul_fwd" => Program::ScalarWeightMul,
            "reshape_fwd" => Program::Reshape,
            "lmlast_fwd_s1" | "lmlast_fwd_s2" | "lmlast_fwd_s3" => Program::LmLast,
            "windowedrmsnorm_gained_fwd" | "windowedrmsnorm_oneplus_fwd" => {
                Program::WindowedRmsNorm { unit: false }
            }
            "windowedrmsnorm_unit_fwd" => Program::WindowedRmsNorm { unit: true },
            "rope_fwd" => Program::Rope,
            "paged_attn_decode"
            | "paged_attn_prefill"
            | "paged_attn_prefill_packed"
            | "paged_attn_decode_pool"
            | "paged_attn_prefill_walk"
            | "paged_attn_decode_hd128"
            | "paged_attn_prefill_hd128"
            | "paged_attn_prefill_p7"
            | "paged_attn_prefill_trips"
            | "batched_decode"
            | "batched_decode_chunks"
            | "batched_decode_pair"
            | "batched_decode_one_slot"
            | "batched_decode_hd128"
            | "batched_decode_chunks_hd128"
            | "batched_decode_one_slot_hd128"
            | "paged_attn_decode_hd128_fp8"
            | "paged_attn_prefill_hd128_fp8"
            | "batched_decode_hd128_fp8"
            | "batched_decode_one_slot_hd128_fp8"
            | "paged_attn_decode_fp8"
            | "paged_attn_prefill_fp8"
            | "paged_attn_prefill_trips_fp8"
            | "batched_decode_fp8"
            | "batched_decode_chunks_fp8"
            | "paged_attn_decode_g256"
            | "paged_attn_decode_g512"
            | "paged_attn_prefill_g512"
            | "paged_attn_decode_g512_fp8"
            | "paged_attn_prefill_g256_fp8" => Program::PagedAttn,
            "paged_kv_write"
            | "paged_kv_write_rows"
            | "paged_kv_write_prompt"
            | "paged_kv_write_hd128"
            | "paged_kv_write_rows_hd128"
            | "paged_kv_write_prompt_hd128"
            | "paged_kv_write_hd128_fp8"
            | "paged_kv_write_prompt_hd128_fp8"
            | "paged_kv_write_fp8"
            | "paged_kv_write_prompt_fp8" => Program::PagedKvWrite,
            other => panic!(
                "`{other}` is in the dump table but `program_of` states no Program for it — \
                 a dumped kernel with no program is a silent skip of the SDSC leg, which the \
                 no-silent-skip law refuses. Add the row."
            ),
        }
    }

    /// The kernels with no `Program` to mint them under: the MoE lowers only from `moe.py`
    /// through KTIR, and no per-op row names a program for it.
    fn sdsc_skip_of(name: &str) -> Option<&'static str> {
        match name {
            "moe_fwd" | "moe_prefill_fwd" => Some("no Program variant — the MoE has no row"),
            _ => None,
        }
    }

    /// One emitted descriptor as stable text for the byte-identity diff.
    fn sdsc_line(op: &(spyre_superdsc::opfunc::OpFunc, spyre_superdsc::wire::Sdsc)) -> String {
        serde_json::json!({ "op": op.0.wire(), "sdsc": op.1 }).to_string()
    }

    /// Lower one dumped kernel the way the bake does: every parameter device-resident, each at a
    /// distinct address, every constant on a stick of its own.
    fn emit_sdsc(name: &str, node: &KtirNode) -> Result<Vec<String>, String> {
        type G = ktir_core::ddl_binds::Baked;
        let params = node.func.arguments.len();
        let planned = spyre_superdsc::plan::<G>(&node.func, &vec![None; params])
            .map_err(|e| format!("{name}: {e:?}"))?;
        let addresses: Vec<spyre_superdsc::DevAddr> = (0..params as u64)
            .map(|p| spyre_superdsc::DevAddr::new((p + 1) << 28))
            .collect();
        let consts: Vec<(spyre_superdsc::graph::ConstStick, spyre_superdsc::DevAddr)> = planned
            .graph
            .consts
            .iter()
            .enumerate()
            .map(|(i, c)| {
                (
                    *c,
                    spyre_superdsc::DevAddr::new((params as u64 + 1) << 28 | (i as u64 * 128)),
                )
            })
            .collect();
        let bindings = spyre_superdsc::Bindings {
            params: &addresses,
            consts: &consts,
            scratch: Some(spyre_superdsc::DevAddr::new((params as u64 + 2) << 28)),
        };
        let ops = planned
            .emit::<G>(&bindings)
            .map_err(|e| format!("{name}: {e:?}"))?;
        if ops.is_empty() {
            return Err(format!("`{name}` emitted zero ops"));
        }
        Ok(ops.iter().map(sdsc_line).collect())
    }

    /// ⭐ THE STRUCTURAL TWIN, RUN IN EVERY TEST PASS — the same driver over the same `entries()`,
    /// writing nothing and asserting the SHAPE of the result rather than its bytes. Every production
    /// kernel compiles to a non-empty KTIR module; every one the lowering takes emits at least one
    /// op; and the kernels it does NOT lower are exactly the two named tables — the skips
    /// (`sdsc_skip_of`) and the refusals below (kernels the lowering refuses by name, each with
    /// the substring its refusal must carry). A kernel that starts
    /// refusing, stops refusing, or refuses for another reason fails here, by name, with no golden
    /// to regenerate: the byte-level proof stays the ignored driver's diff.
    #[test]
    fn every_production_kernel_lowers_or_is_refused_by_name() {
        // (kernel, a substring its refusal carries).
        const EXPECTED_REFUSALS: &[(&str, &str)] = &[];
        let mut refused: Vec<(&str, String)> = Vec::new();
        let mut skipped: Vec<&str> = Vec::new();
        for (name, src, spec, grid) in entries() {
            let module = compile_kernel(src, &spec, &grid)
                .unwrap_or_else(|e| panic!("`{name}` did not compile: {e}"));
            assert!(
                !triton_ktir::text::print::print(&module).trim().is_empty(),
                "`{name}` printed an empty module"
            );
            if sdsc_skip_of(name).is_some() {
                skipped.push(name);
                continue;
            }
            let node = crate::compile::node_for(&module, program_of(name))
                .unwrap_or_else(|e| panic!("`{name}` did not mint a node: {e}"));
            match emit_sdsc(name, &node) {
                Ok(lines) => assert!(!lines.is_empty(), "`{name}` emitted no ops"),
                Err(reason) => refused.push((name, reason)),
            }
        }
        let names: Vec<&str> = refused.iter().map(|(n, _)| *n).collect();
        let expected: Vec<&str> = EXPECTED_REFUSALS.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names, expected,
            "the kernels the lowering refuses moved — refusals: {refused:#?}"
        );
        for ((name, reason), (_, why)) in refused.iter().zip(EXPECTED_REFUSALS) {
            assert!(
                reason.contains(why),
                "`{name}` is refused, but not for the named reason ({why:?}): {reason}"
            );
        }
        skipped.sort_unstable();
        assert_eq!(
            skipped,
            ["moe_fwd", "moe_prefill_fwd"],
            "the named skips moved"
        );
    }

    #[test]
    #[ignore = "the LegalizeTypes byte-identity proof driver: dumps every production \
                kernel's post-to_ktir module text and asserts nothing about the numbers — \
                the proof is the diff of two runs (see the module docs)"]
    fn dump_production_ktir() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../target/spyre-ktir-dump");
        std::fs::create_dir_all(&dir).expect("create the dump dir");
        for (name, src, spec, grid) in entries() {
            let module = compile_kernel(src, &spec, &grid)
                .unwrap_or_else(|e| panic!("`{name}` did not compile: {e}"));
            let text = triton_ktir::text::print::print(&module);
            assert!(
                !text.trim().is_empty(),
                "`{name}` printed an empty module — a dump of nothing proves nothing"
            );
            let path = dir.join(format!("{name}.ktir"));
            std::fs::write(&path, &text)
                .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
            eprintln!(
                "dumped {name} -> {} ({} lines)",
                path.display(),
                text.lines().count()
            );

            // THE SDSC LEG — the emitted-descriptor half of the same proof: the byte-identity
            // bar applies to what the lowering EMITS, not only to the module text. Same law:
            // the diff of two runs is the proof; a named skip (`sdsc_skip_of`) is the
            // only kernel without an `.sdsc.jsonl`, and every other kernel must lower.
            match sdsc_skip_of(name) {
                Some(reason) => {
                    eprintln!("SDSC-SKIP {name}: {reason}");
                }
                None => {
                    let node = crate::compile::node_for(&module, program_of(name))
                        .unwrap_or_else(|e| panic!("`{name}` did not mint a node: {e}"));
                    let sdsc_path = dir.join(format!("{name}.sdsc.jsonl"));
                    match emit_sdsc(name, &node) {
                        Ok(lines) => {
                            std::fs::write(&sdsc_path, format!("{}\n", lines.join("\n")))
                                .unwrap_or_else(|e| panic!("write {}: {e}", sdsc_path.display()));
                            eprintln!(
                                "dumped {name} -> {} ({} ops)",
                                sdsc_path.display(),
                                lines.len()
                            );
                        }
                        // ⭐ A KERNEL THE LOWERING REFUSES IS A NAMED SKIP, CARRIED AS DATA — the
                        // driver mirrors what production lowers and NOTHING wider. The refusal
                        // is the lowering's own message, written as the kernel's one
                        // `.sdsc.jsonl` line, so the before/after diff still sees it: a kernel
                        // that flips refused↔emitted, or whose refusal reason moves, is a diff.
                        Err(reason) => {
                            std::fs::write(
                                &sdsc_path,
                                format!("{{\"refused\": {}}}\n", serde_json::json!(reason)),
                            )
                            .unwrap_or_else(|e| panic!("write {}: {e}", sdsc_path.display()));
                            eprintln!("SDSC-REFUSED {name}: {reason}");
                        }
                    }
                }
            }
        }
    }
}
