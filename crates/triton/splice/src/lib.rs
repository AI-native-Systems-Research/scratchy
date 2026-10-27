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
//!   INFER this from the op soup; scratchy has the tape, so it STATES it, and the golden
//!   pins the `op_name` this row's stem produces rather than trusting a classifier that
//!   could mis-recognize.
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
//!
//! `GainConvention::OnePlusScale` WAS this list's third member and is a ROW now: the
//! kernel states the `+1` as an addf over the loaded gain and the door states the same
//! add against the offset's `[1,1]` registry const (the epsilon's own mechanism) —
//! see [`rmsnorm_row`].
//!
//! # THE GATE, AND WHY IT IS AT THE DESCRIPTOR LEVEL
//!
//! The golden (`tests/triton_splice_golden.rs`) checks the splice's `EmittedOp`s: the
//! `op_name` (the builder's naming law), the attention arm the shape took, and an
//! emulator `max_abs` numeric comparison for the families the emulator runs (matmul,
//! elementwise, gelu, rope, fp8 matmul). The builder that preceded the splice is
//! DELETED — there is no byte-identity control anymore, and nothing in this crate
//! should claim one. The strongest surviving gates are the numeric ones and the
//! canonical card A/B per model/quant pair at the PR level.
//!
//! # ⛔ THE SHAPE WORK LIST, AS NAMED ERRORS
//!
//! The kernel families here state ONE whole-tensor tile at corner 0. A node whose shape
//! needs more than that is refused LOUDLY, naming the kernel capability that must land.
//! The historical items (windowed regions, LX row-blocking, the odd-N vocab matmul, the
//! prefill lm-head fold) have all LANDED — their kernels state the corner (`C_START`),
//! the row blocks, the device width, and the fold (`lmlast.py` + the m=1 tail); what
//! remains is whatever the current guards name, which is the honest work list.
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

use ktir_core::arena::Arena;
use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::ktir_node::{BufferId, Elementwise, KtirNode, Program};
use ktir_superdsc::superdsc_opspec::Df;
use scratchy_subtile::lower::GemmWeight;
use scratchy_subtile::subtile_ir::{EwKind, GainConvention, SubOp, SubtileIR, SubtileNode};

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
/// offset over the load first, a constant the door resolves through the SAME
/// `[1,1]` registry slot mechanism the epsilon uses (`program_gain_offset` reads
/// it off the program; `compute_bundle_layout` collects `gain.offset()`).
pub fn rmsnorm_row(gain: GainConvention) -> TritonKernelRow {
    match gain {
        GainConvention::Scale => TritonKernelRow::Spliced {
            src: include_str!("../../../targets/spyre/kernels/rmsnorm.py"),
            file: "rmsnorm.py",
            entry: "rmsnorm_fwd",
            program: Program::RmsNorm,
        },
        // ⭐ THE (1 + w) ROW — the offset is CARRIED, never folded into the weights
        // (folding would make the loaded gain disagree with the checkpoint, so every
        // consumer that re-reads it — dumps, the numeric reference, a second target —
        // would see a different tensor). The kernel states the addf over the load;
        // the door states the SAME add against the offset's registry const, so the
        // descriptor computes the formula the program runs — metal's own
        // `ScalarOffsetRmsNorm` (whose kernel applies `weight + offset`) is the
        // precedent shape.
        GainConvention::OnePlusScale => TritonKernelRow::Spliced {
            src: include_str!("../../../targets/spyre/kernels/rmsnorm.py"),
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
            src: include_str!("../../../targets/spyre/kernels/matmul.py"),
            file: "matmul.py",
            entry: "matmul_fwd",
            program: Program::Matmul,
        },
        // THE fp8 W8A8 ROW — the delivery target (granite 8b fp8). Same
        // `Program::Matmul` classification as dense: the DOOR discriminates fp8 from
        // the weight view's `is_fp8` + arity-3 bindings, never from the program kind,
        // so both rows reach the same `matmul` door arm. The kernel's spelled
        // `* w_scale` epilogue is what the ladder's
        // `verify_canonical_fp8_matmul_kernel` requires.
        GemmWeight::Fp8Dynamic => TritonKernelRow::Spliced {
            src: include_str!("../../../targets/spyre/kernels/matmul_fp8.py"),
            file: "matmul_fp8.py",
            entry: "matmul_fp8_fwd",
            program: Program::Matmul,
        },
        // ⛔ NO KERNEL: the wavefront lowering skips affine presets before a tape is
        // lowered (`codegen`'s `no superdsc bundle` gate) and no consumer realizes an
        // affine contraction on this path, so nothing constructs this node on the spyre
        // path. The arm is the compile-time enumeration — a realization lands here
        // WITH its kernel (metal's qmv family is the precedent shape).
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
    let ew_src = || include_str!("../../../targets/spyre/kernels/elementwise.py");
    match kind {
        // Add AND BiasAdd share `add_fwd` — both map to `Elementwise::Add` and the same
        // `add_s{id}` name (the builder's own law).
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
        // emitting "gelu" for it would run the wrong model and report success (the
        // consumer's own `elementwise_op_func` refusal names exactly this). A
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
        // No elementwise kernel for squared ReLU in `elementwise.py`; the arch using it
        // (Nemotron-H) is not spyre-capable.
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
        src: include_str!("../../../targets/spyre/kernels/lmlast.py"),
        file: "lmlast.py",
        entry: "lmlast_fwd",
        program: Program::LmLast,
    }
}

/// THE ATTENTION FAMILY — one row, one kernel, every shape: `attn.py`'s own three
/// constexpr-selected arms (the causal one-pass, decode, prefill continuation) are the
/// builder's own three arms restated, so the row does not branch on shape at all.
pub fn attn_row() -> TritonKernelRow {
    TritonKernelRow::Spliced {
        src: include_str!("../../../targets/spyre/kernels/attn.py"),
        file: "attn.py",
        entry: "attn_fwd",
        program: Program::Attn,
    }
}

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
            src: include_str!("../../../targets/spyre/kernels/silumul.py"),
            file: "silumul.py",
            entry: "silumul_fwd",
            program: Program::SiluMul,
        },
        SubOp::ScalarMul { .. } => TritonKernelRow::Spliced {
            src: include_str!("../../../targets/spyre/kernels/scalarmul.py"),
            file: "scalarmul.py",
            entry: "scalarmul_fwd",
            program: Program::ScalarMul,
        },
        // `RopeAppend` carries 6 inputs but only the first three are the rotation (V
        // and the KV-cache destinations flow through GRAPH edges — the builder's own
        // `lower_rope_node` reads `inputs[0..3]`), so the kernel consumes x/cos/sin/out
        // and the row covers BOTH `RopeAppend` and the standalone `RopeRotate`.
        SubOp::RopeRotate { .. } | SubOp::RopeAppend { .. } => TritonKernelRow::Spliced {
            src: include_str!("../../../targets/spyre/kernels/rope.py"),
            file: "rope.py",
            entry: "rope_fwd",
            program: Program::Rope,
        },
        // ATTENTION SPLICES — `lower_attn` is the entry: the kernel states the same
        // three arms as constexpr-selected cases, and the door (`attn_operands`) reads
        // every fact — the q/out views, the swept rung, the scale — off the program
        // itself. This row exists so `row()` stays exhaustive; the walk never routes
        // attention through `lower_one` (attention needs door-order bindings, the mask
        // binding, and the dead-prefix view injection that no pure-Triton program can
        // state).
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
        SubOp::TanhSoftCap { .. } => TritonKernelRow::Spliced {
            src: include_str!("../../../targets/spyre/kernels/tanhsoftcap.py"),
            file: "tanhsoftcap.py",
            entry: "tanhsoftcap_fwd",
            program: Program::TanhSoftCap,
        },
        SubOp::RmsNormUnit { .. } => TritonKernelRow::Spliced {
            src: include_str!("../../../targets/spyre/kernels/rmsnormunit.py"),
            file: "rmsnormunit.py",
            entry: "rmsnormunit_fwd",
            program: Program::RmsNormUnit,
        },
        SubOp::ScalarWeightMul => TritonKernelRow::Spliced {
            src: include_str!("../../../targets/spyre/kernels/scalarwmul.py"),
            file: "scalarwmul.py",
            entry: "scalarwmul_fwd",
            program: Program::ScalarWeightMul,
        },
        SubOp::Reshape { .. } => TritonKernelRow::Spliced {
            src: include_str!("../../../targets/spyre/kernels/reshape.py"),
            file: "reshape.py",
            entry: "reshape_fwd",
            program: Program::Reshape,
        },
        // ⭐ THE TOKEN DRAW — the whole vocab scan as ONE kernel per request
        // row: the row, its per-row noise block and temperature reciprocals,
        // the nine staged tables (functions of the vocab alone, host-staged at
        // load — the seventh is the pad law's offset row), and the `[rows,
        // 64]` i32 idx stick out. The vocab and the row count ride the node
        // (never a model name); every constexpr below derives from them.
        SubOp::Sample { .. } => TritonKernelRow::Spliced {
            src: include_str!("../../../targets/spyre/kernels/sample_fwd.py"),
            file: "sample_fwd.py",
            entry: "sample_fwd",
            program: Program::Sample,
        },
        // ⛔⭐ THE OPS `lower_one_node` NEVER ROUTES HERE — enumerated by NAME, never
        // `_`, so adding a SubOp is an E0004 in this crate too. `lower_one_node`'s own
        // match is the gate: these kinds have their own arms there (attention's
        // const-generic geometry door, the host-routed rmsnorm pair, the by-name
        // refusals), and a kind reaching THIS match means the routing changed — the
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
/// `op_name` and the emulator's function key follow the same naming law the builder
/// path established. That is what the golden's `op_name` pins.
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

/// The prefill lm-head tail predicate, stated ONCE — every consumer of the fold
/// (the `lower_all` rewrite, the one-op `lower`'s routing echo below, and the
/// bake-time numeric gate's tail compare, which states the fold's last-row law
/// against the golden) reads
/// the SAME four conjuncts, because the width test alone was falsified by the
/// gemma-4 parity fixture (vocab 512), whose GLOBAL-layer q projection is ALSO
/// 512 cols (4 heads × hd 128) — the fold fired on every global layer's Q, and
/// the model's first sampled token was garbage while every width-128 tensor
/// stayed fp16-clean. A width COLLISION is not an identity: only the tail's own
/// rows are dead, and only the result's ancestors are the tail. The walk is
/// [`tensor_feeds_result_through_row_local_ops`] (main's own law).
pub fn is_prefill_lm_head_tail<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    rows_are_requests: bool,
) -> bool {
    // ⭐ THE MINTED DRAW STATES THE TAIL'S PRODUCT. A graph the sampler minted
    // re-points its result at the idx stick and pads the tail's storage width,
    // so the original width conjunct (the node's output region width equals the
    // result tensor's cols) can no longer see the tail. The draw restates it:
    // the lm-head's OWN full output width N equals the vocab the draw serves,
    // and an elementwise tail node's region width equals that vocab — the same
    // "this node is vocab-wide like the result" statement, read off the ops'
    // own fields, immune to the re-point and the padding. A graph without a
    // draw keeps the original conjunct verbatim. One predicate, read by every
    // caller — the `lower_all` rewrite, the one-op routing echo, and the
    // bake-time numeric gate — so all three state the SAME fold. (The width
    // alone was never sufficient — the gemma-4 q-projection collision is what
    // the feeds-walk conjunct below kills — and that protection is unchanged.)
    let draw = ir
        .nodes
        .iter()
        .find(|n| matches!(n.op, SubOp::Sample { .. }));
    let feeds_the_tail_product = match (&draw, &node.op) {
        (Some(sample), SubOp::MatmulTile { n, .. }) => match &sample.op {
            SubOp::Sample { vocab, .. } => *n == *vocab,
            _ => false,
        },
        (Some(sample), _) => match &sample.op {
            SubOp::Sample { vocab, .. } => node.output.region.cols.len == *vocab,
            _ => false,
        },
        (None, _) => {
            let result_cols = ir.tensors[ir.result.index()].cols;
            node.output.region.cols.len == result_cols
        }
    };
    feeds_the_tail_product
        && node.output.region.rows.len > 1
        && !rows_are_requests
        && tensor_feeds_result_through_row_local_ops(ir, node.output.tensor)
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
    // ── THE HEAD-NORM SANDWICH'S THREE SEATS — the split (head) is where the fused
    // program is emitted; the flatten-back (tail) and the middle (norm) self-identify
    // as fused-away and emit NOTHING. The match is dataflow-based
    // ([`sandwich_at_split`]), so it is a pure function of the graph — every walk
    // reaches the same answer at the same nodes, and any failed check falls back to
    // the byte-identical un-fused lowering (the ordinary reshape/norm rows below).
    // ⛔ THE MATCHER ACCEPTS GAINED (`Scale`/`OnePlusScale`) AND `RmsNormUnit`
    // MIDDLES. The two gained conventions are two ENTRIES of the same windowed
    // body (`windowedrmsnorm_gained_fwd` / `windowedrmsnorm_oneplus_fwd`), and the
    // door reads the oneplus form's offset off the program's own addf — the same
    // law the plain rmsnorm row carries — so neither convention needs to stay
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
        return lower_prefill_lm_head_fold(node, ir);
    }
    // ⭐ THE LOGITS SCALARMUL IS THE TAIL'S SECOND HALF — main's own law
    // (`lower_scalarmul_node(&node_at_one_row(node), …)` under the same
    // `is_prefill_lm_head_tail`): the m>1 matmul above it is folded to the m=1
    // tail that writes ONE logits row, so the scale must run over that one row
    // too. Left un-folded it scales mq × vocab logits of which mq−1 rows were
    // never written — row 0 is still right (which is why text A/Bs pass), but
    // the device pays mq−1 rows of reads of unwritten logits for nothing.
    if let SubOp::ScalarMul { .. } = &node.op
        && is_prefill_lm_head_tail
    {
        let at_m1 = node_at_one_row(node);
        return lower_one(&at_m1, ir, rows_are_requests).map(|e| vec![e]);
    }
    // ⭐ AND THE LOGITS TANH-SOFTCAP IS THE TAIL'S THIRD KIND — the same
    // `node_at_one_row` law (main gives TanhSoftCap the fold ScalarMul has): the
    // softcap sits ON the logits, so the folded tail caps ONE row of them instead
    // of mq rows of which mq−1 were never written.
    if let SubOp::TanhSoftCap { .. } = &node.op
        && is_prefill_lm_head_tail
    {
        let at_m1 = node_at_one_row(node);
        return lower_one(&at_m1, ir, rows_are_requests).map(|e| vec![e]);
    }
    // ⭐ AND THE TOKEN DRAW'S ROW INPUT IS THE FOLDED TAIL'S ONE ROW — the same
    // `node_at_one_row` law the tail's ScalarMul and TanhSoftCap get: the mint
    // states the prefill draw at rows = 1 reading the result's row 0 (the row
    // the fold writes), so the kernel is handed the whole one-row region at
    // corner 0. The gate is the mint's own typed fact — a one-row region of a
    // taller tensor — never a model name or a rung label.
    if let SubOp::Sample { .. } = &node.op
        && node.inputs[0].region.rows.len == 1
        && ir.tensors[node.inputs[0].tensor.index()].rows > 1
    {
        let at_m1 = node_at_one_row(node);
        return lower_one(&at_m1, ir, rows_are_requests).map(|e| vec![e]);
    }
    lower_one(node, ir, rows_are_requests).map(|e| vec![e])
}

/// Main's own `node_at_one_row`, verbatim law: slice the FIRST row (the region's
/// own `rows.start`, length 1) of the input and the output — the m=1 tail's
/// addressing, which the folded matmul's output row 0 matches.
fn node_at_one_row<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
) -> SubtileNode<F> {
    let one_row = |tr: &scratchy_subtile::subtile_ir::TensorRegion| {
        scratchy_subtile::subtile_ir::TensorRegion {
            tensor: tr.tensor,
            region: scratchy_subtile::subtile_ir::Region {
                rows: scratchy_subtile::subtile_ir::Range::new(tr.region.rows.start, 1),
                cols: tr.region.cols,
            },
        }
    };
    let mut inputs = node.inputs.clone();
    if let Some(a) = inputs.first_mut() {
        *a = one_row(a);
    }
    SubtileNode {
        id: node.id,
        op: node.op,
        inputs,
        output: one_row(&node.output),
    }
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
//  splice's own addition: the kernel addresses the TENSOR BASES where the builder's
//  `load_region` honored a region's corner) — the matcher returns `None` and all three
//  nodes lower exactly as they did before this existed, through their ordinary rows.
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
    /// carried: it is `full / D` of the split input's own extents, re-derived by the
    /// spec and the door alike, so no second spelling can disagree.
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
/// entries of the same windowed body, and the door resolves the OnePlusScale form's
/// offset off the program's own addf (the plain rmsnorm row's law), so the matcher
/// stays a pure function of the graph for every convention.
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
    // ⭐ THE SPLICE'S OWN LAW, the one the builder did not need: the kernel addresses
    // the TENSOR BASES (its descriptors bind the split's source and the flatten's
    // output directly), where the builder's `load_region` honored a region's corner.
    // A windowed sandwich stays un-fused — the ordinary reshape + norm + reshape rows
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
fn lower_sandwich_node<F: scratchy_subtile::subtile_ir::RopeForm>(
    sandwich: &HeadNormSandwich<'_, F>,
) -> Result<Vec<EmittedOp>, String> {
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
    let src = include_str!("../../../targets/spyre/kernels/windowedrmsnorm.py");
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
        // The convention's own `offset()` (1.0), the same value the census
        // collects and the door resolves — one constant, three readers, one
        // value, the epsilon's law.
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
    let k = mint(&synth, module, &program, default_bindings(&synth))?;
    // `mint` already named the node `rmsnorm_s{split.id}` (the program stem + the
    // split's id, main's own naming law for the fused form).
    let name = k.func.name.to_string();
    let mut e = EmittedOp::bare(name);
    e.ktir = Some(k);
    Ok(vec![e])
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE PREFILL TAIL'S DATAFLOW WALK — main's own law, ported verbatim.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The ROW-LOCAL op set: ops whose output row `r` depends only on input row `r` (plus
/// weights and per-row tables). The prefill lm-head fold is sound only inside this set —
/// see [`tensor_feeds_result_through_row_local_ops`].
fn row_local_tail_op<F: scratchy_subtile::subtile_ir::RopeForm>(op: &SubOp<F>) -> bool {
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
            // ⭐ THE DRAW IS ROW-LOCAL IN THE TAIL'S OWN SENSE: it reads one
            // row region of its input (the fold's product) and writes the idx
            // stick — no row mixing — so a minted graph's feeds-walk traverses
            // it from the tail to the re-pointed result.
            | SubOp::Sample { .. }
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

fn lower_one<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    rows_are_requests: bool,
) -> Result<EmittedOp, String> {
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
    // rewritten THERE (last-row extraction + the re-lowered m=1 matmul), so a
    // MatmulTile reaching THIS one-op body as the tail means the one-op `lower`
    // was called where the walk's `lower_all` belongs. The test is the FULL
    // [`is_prefill_lm_head_tail`] — width, m>1, rows-not-requests, AND the
    // row-local dataflow walk — because the echo must not fire on a width
    // COLLISION the walk already refused (the q-projection law: a collision
    // matmul that feeds the result through rope lowers through here normally,
    // un-folded, every row kept).
    //
    // ⭐ THE ODD VOCAB SPLICES. A decode lm_head at vocab 49155 (granite) has N odd,
    //    and the ladder's `PlanCorelets` now re-patterns an odd N to `single_corelet`
    //    (the same re-patterning the C++ itself made for `split` at one stick) — the
    //    builder path has emitted exactly this matmul for this repo's whole life, so
    //    the device runs it, and the two-corelet plan was a LADDER artifact, not a
    //    device fact.
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
        // ⭐ THE TANH-SOFTCAP AND THE TOKEN DRAW STATE NEITHER CORNER — the softcap
        // kernel's view IS the region (`[M, N]`, tiles from row 0 / column 0), so BOTH
        // corners must be 0 while the extents are free: the prefill lm-head fold
        // re-lowers the vocab-wide softcap at m=1 (row 0 of the `[mq, vocab]` logits
        // storage, a legal prefix view), and the front end's chunker never tiles this
        // op (its chunk set is the elementwise/scalarmul/silumul family), so a column
        // window never arises. The draw's kernel is the same law: its view is the
        // region — the minted prefill draw reads row 0 of the `[mq, pad_vocab]`
        // storage at rows = 1, a legal prefix view — and every operand the draw
        // binds (the per-row pair, the nine tables, the idx stick) is whole at
        // corner 0.
        let states_neither_corner = matches!(
            &node.op,
            SubOp::TanhSoftCap { .. } | SubOp::Sample { .. }
        );
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
        // ⭐ SCALAR-WEIGHT MUL IS ARITY-2 (`[act, weight]`, the builder's own
        // binding). The MoE sub-ops have NO arity here any more: the registry
        // refuses them before the splice — the MoE lowers only from
        // `kernels/moe.py` through KTIR.
        SubOp::ScalarWeightMul => 2,
        // ⭐ THE SAMPLER'S ARITY IS 12: the logits row, its per-row noise
        // block and its per-row temperature reciprocals, then the nine
        // staged tables (two decreasing ramps, two increasing index tables,
        // two widen tables, the pad law's offset row, the two-level ramp's
        // group pair) — every kernel parameter but the output is a tape
        // operand, per the splice's own positional law.
        SubOp::Sample { .. } => 12,
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
    let k = mint(node, m, program, default_bindings(node))?;
    let name = format!("{}_s{}", program_stem(node, program), node.id.index());
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
    // The bake-log proof the fold fired: every prefill rung's bake names its
    // own LAST_HIDDEN extraction — the op, the row it takes, and the staging
    // it lands in. A rung whose bake lacks this line folded nothing.
    eprintln!(
        "[spyre-superdsc] prefill lm-head tail folded — lmlast_s{} extracts row {} of {} \
         (hidden {}) into LAST_HIDDEN t{}",
        node.id.index(),
        row,
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
    ce("ROW", Val::Int(i128::from(row)))?;
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
                rows: scratchy_subtile::subtile_ir::Range::new(0, 1),
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
        module,
        &lmlast_program,
        default_bindings(&extraction_node),
    )?;
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

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  ATTENTION — the one spliced kind whose program is NOT a straight parameter-for-tensor
//  substitution, which is why it gets its own entry rather than riding `lower_one`:
//
//  * the DOOR's parameter order is q, out, then kc, new_k, vc, new_v, then the mask — the
//    output is SECOND, not last, because `attn_operands` reads `r[0]`/`r[1]` as q/out and
//    the next four as the segment pairs (`KtirFunc::attn`'s own `arg_for` order);
//  * the MASK is a bound synthetic tensor beyond the graph (`k.mask`, not a binding slot
//    the positional mint could name);
//  * the DEAD PREFIX is a SPLICE-INJECTED VIEW: at `SWEPT == 0` the kernel's `if SWEPT > 0`
//    guards leave the resident cache parameters addressed nowhere, the ladder DCEs an
//    unconsumed descriptor, and no Triton program can state "a parameter with a view and no
//    access tile" — so this entry injects the two `ktdp.construct_memory_view` ops into the
//    compiled module post-hoc, the same law as `lmlast`'s `node_out_tid` (a stated tape
//    fact, not an inferred one).
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE ATTENTION SPLICE — compile `attn.py` for this node's geometry and swept rung, and
/// mint the `Program::Attn` `KtirNode` with the door's own parameter order.
///
/// `cap` is the resident cache tensor's row extent (`lower_attn_node`'s own read), and
/// `active_cap` is the swept rung the bundle was baked for — THE one bundle fact that must
/// meet the program here, because `attn_at` reads the swept extent back off the program
/// (`param_read_rows`) and the two must agree by the door's own hard error.
pub fn lower_attn<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    cap: u32,
    active_cap: ktir_superdsc::ktir_node::ActiveCap,
) -> Result<EmittedOp, String> {
    let row = attn_row();
    let (src, attn_program) = match &row {
        TritonKernelRow::Spliced { src, program, .. } => (*src, program),
        TritonKernelRow::Refused(why) => return Err(why.to_string()),
    };
    let SubOp::AttnDecode { geom, scale, .. } = &node.op else {
        return Err(
            "triton splice: lower_attn called on a node that is not an AttnDecode — the \
             entry is AttnDecode's"
                .to_string(),
        );
    };
    if node.inputs.len() < 5 {
        return Err(format!(
            "triton splice: AttnDecode t{} expects 5 inputs [q, prefix_k, prefix_v, new_k, \
             new_v], found {}",
            node.output.tensor.index(),
            node.inputs.len()
        ));
    }
    let (nqh, nkvh, hd) = (geom.nqh().get(), geom.nkvh().get(), geom.hd().get());
    let stick = ktir_superdsc::sdsc_abstract::POOL_STICK; // 64 — the rung alignment law
    let swept = active_cap.resolve(cap, stick);
    let mq = node.output.region.rows.len;
    let gqa = geom.gqa().get();
    // ⛔ THE PREFIX SEGMENT'S OWN LAW, `KtirSeg`'s and not the rung's: a MASKED prefix
    // (this forward's own rope-append) is read from row 0 to the SWEPT extent — the
    // rung, bounded by the cache TENSOR's own rows (`graph.shape(kr.tensor).rows`,
    // NOT the region's: `lower_region` slices a rope-append prefix's region to
    // `valid_len - 1` rows while the builder's tile sweeps the rung and lets the
    // runtime length mask bound it); a PRE-POPULATED one (the emulator's host-threaded
    // cache) is read at its own region rows and corner. The door reads this extent
    // back off the program (`param_read_rows`), so the spliced tile must state exactly
    // the builder's.
    let mask_prefix = matches!(
        &node.op,
        SubOp::AttnDecode {
            producer: scratchy_subtile::subtile_ir::KvCacheProducer::SameForwardRopeAppend { .. },
            ..
        }
    );
    // The cache tensor's own row count — the builder's bound (`swept.min` of exactly
    // this), and the same reading `lower_one_node`'s walk used for `cap`.
    let cache_rows = ir.tensors[node.inputs[1].tensor.index()].rows;
    let (prefix_len, kc_row) = if mask_prefix {
        (swept.min(cache_rows), 0)
    } else {
        (
            node.inputs[1].region.rows.len,
            node.inputs[1].region.rows.start,
        )
    };
    // `KtirFunc::attn`'s zero-length segment law: a dead prefix (prefix_len == 0) keeps
    // its VIEW (injected below) and drops its compute, AND CLEARS `mask_prefix` — with no
    // prefix segment there is nothing the runtime length mask could bound. That clearing
    // is what makes a no-prefix prompt chunk take the causal ONE-PASS arm.
    let mask_eff = mask_prefix && prefix_len > 0;
    let new_len = node.inputs[3].region.rows.len;
    // The builder's own one-pass arm condition, verbatim: ONE LIVE SEGMENT (the dead
    // prefix leaves exactly the new block), mq > 1, no runtime length mask, and the new
    // block spanning the whole chunk (`seq_len == mq`) — the additive `[mq, mq]` causal
    // triangle can then replace the per-row slice. (A temporary `mq >= 8` floor sat
    // here to satisfy the front end's since-removed TMA 16-byte check — a GPU law the
    // Spyre target does not have; with the check gone the builder's own condition
    // stands again, and the ladder's bottom rung takes the one-pass arm like main.)
    let one_pass = !mask_eff && prefix_len == 0 && new_len == mq && mq > 1;
    // Every other multi-row shape is the whole-chunk continuation (`attn.py`'s
    // `_attn_cont_head`), which states its causality in the mask rather than in per-row
    // tile extents — so it always binds one.
    let continuation = mq > 1 && !one_pass;
    if continuation && (prefix_len == 0 || new_len < mq) {
        return Err(format!(
            "triton splice: AttnDecode t{}: a {mq}-row attention with a {prefix_len}-row prefix and \
             a {new_len}-row new block — the whole-chunk continuation needs a live prefix and a new \
             block covering every row, and a dead prefix with a full new block is the one-pass arm",
            node.output.tensor.index()
        ));
    }
    // THE MASK IS BOUND WHEN A SEGMENT CONSUMES IT — the runtime length mask (decode,
    // `[1, prefix_len]`), the causal triangle (the one-pass, `[mq, mq]`), or the
    // continuation's `[mq, prefix_len + mq]` prefix-plus-new-block mask.
    let has_mask = mask_eff || one_pass || continuation;
    // The scale, stated as the kernel's own constexpr so the program's `qk * SCALE` mulf
    // carries exactly the value `program_score_scale` reads back — the door resolves the
    // registry slot FROM that value, so the program and `scalarmul_scales` cannot disagree.
    let spec = attn_kernel_spec(
        node,
        &AttnFacts {
            nqh,
            nkvh,
            hd,
            gqa,
            mq,
            cap,
            prefix_len,
            kc_row,
            new_len,
            scale: *scale,
            has_mask,
            one_pass,
        },
    )?;
    // ⭐ THE GRID IS THE HEAD COUNT AT mq > 1 — the builder's own structure. `KtirFunc::attn`
    // runs one head per program instance when `mq > 1` (`KtdpGetComputeTileId`, grid
    // `(nq, 1)`), and the kernel states the same shape: `tl.program_id(0)` IS the head in
    // its mq > 1 arms, so the grid folds the head axis and the program carries ONE head's
    // body — 32x fewer ops per program than a static_range head loop unrolled into one
    // grid-[1] program (granite: 32 query heads), which is the compile-time law the
    // builder's form sets. Decode keeps grid [1] and the kernel's own static head loop,
    // exactly the builder's `mq == 1` arm.
    let grid: Vec<i64> = if mq > 1 {
        vec![i64::from(nqh)]
    } else {
        vec![1]
    };
    let positional = attn_positional(
        src,
        &spec,
        &grid,
        *attn_program,
        AttnPostCompile {
            cap,
            kv_width: nkvh * hd,
            dead_prefix: prefix_len == 0,
            has_mask,
        },
    )?;
    let k = mint_attn(node, ir, positional, has_mask)?;
    let name = format!("attn_s{}", node.id.index());
    let mut e = EmittedOp::bare(name);
    e.ktir = Some(k);
    Ok(e)
}

/// What [`attn_positional`] does to the compiled module before `node_for` reads it — every input
/// of those edits, so the memo keys on all of them by name rather than on the claim that the
/// spec's constexprs already spell them.
struct AttnPostCompile {
    cap: u32,
    kv_width: u32,
    dead_prefix: bool,
    has_mask: bool,
}

/// ⭐ THE ATTENTION PROGRAM, MEMOIZED PAST `node_for` — the per-LAYER cost law.
///
/// Every layer of a rung states the SAME spec (the geometry, the widths and the corners are the
/// rung's, not the layer's), and the wiring walk lowers every layer of the unrolled graph. With
/// only [`compile_kernel`] memoized, each of the 40 layers still cloned the cached module, edited
/// it and re-ran `node_for` over it — MEASURED, after the compile memo, `node_for` alone was a
/// third of the granite-2b models build script on the 256-row prompt chunk. What differs per
/// layer is exactly what [`mint_attn`] re-states (name, bindings, mask slot), so the positional
/// node is computed once per (compile key, post-compile edits, program) and cloned.
fn attn_positional(
    src: &str,
    spec: &KernelSpec,
    grid: &[i64],
    program: Program,
    post: AttnPostCompile,
) -> Result<KtirNode, String> {
    static NODES: Memo<KtirNode> = Memo::new();
    let AttnPostCompile {
        cap,
        kv_width,
        dead_prefix,
        has_mask,
    } = post;
    let key = memo_key(src, spec, grid).map(|k| {
        format!("{k}\0{program:?}\u{1}{cap}\u{1}{kv_width}\u{1}{dead_prefix}\u{1}{has_mask}")
    });
    NODES.get_or_compute(key, || {
        let mut module = compile_kernel(src, spec, grid)?;
        // ⛔ THE DEAD PREFIX'S VIEWS, INJECTED. At `prefix_len == 0` the kernel guards its
        // kc/vc descriptors away, the ladder DCEs them, and the resident cache parameters would
        // be "addressed NOWHERE" — but the door requires them: `attn_operands` reads the cache's
        // identity and capacity off those views, and the card keeps the cache RESIDENT through
        // them. The builder keeps the view by construction (`KtirFunc::attn`'s zero-length
        // segment); the splice states the same fact post-compile, as one
        // `ktdp.construct_memory_view` per cache parameter over the SAME shape the live
        // segment's view states (the full `[cap, nkvh·hd]` capacity).
        if dead_prefix {
            inject_dead_cache_views(&mut module, cap, kv_width)?;
        }
        // ⛔ THE MASKLESS SHAPE TRUNCATES THE MASK PARAMETER. The kernel's parameter list is
        // static Python, so `desc_mask` is declared in every arm; a maskless shape (a
        // pre-populated cache at decode, a continuation chunk) traces no view over it, and the
        // handoff refuses an unaddressed parameter rather than inventing a width. The builder's
        // own maskless arm mints NO mask parameter (`KtirFunc::attn` registers the mask tile
        // only under `mask_prefix` / the one-pass arm), so the same law holds here: the
        // parameter must not survive. `desc_mask` is the LAST parameter — the consumer numbers
        // its buffers by parameter position, so removing the tail renumbers nothing (the
        // handoff's own law).
        if !has_mask {
            truncate_unused_mask_param(&mut module)?;
        }
        triton_ktir_superdsc::node_for(&module, program)
            .map_err(|e| format!("triton splice: node_for: {e}"))
    })
}

/// THE ATTENTION KERNEL'S LAUNCH CONTRACT — the door's own parameter order (q, out, the
/// four segment buffers in `attn_operands`'s positional order — resident K, new K,
/// resident V, new V — then the mask), plus every extent as a constexpr: the geometry
/// (`NQH/NKVH/HD/GQA`), the widths (`MQ/CAP/SWEPT/NEW_LEN`), the segment corners
/// (`KC_*/KD_*`, the region row/col starts), the scale, and the two arm selectors
/// (`HAS_MASK`, `ONE_PASS`).
///
/// ⭐ THE MASK PARAMETER IS STATED ONLY WHEN A SEGMENT CONSUMES IT — the kernel builds its
/// mask descriptor under the same `HAS_MASK` guard, so an always-declared parameter would
/// be a binding slot nothing addresses (the ladder DCEs the unconsumed descriptor and the
/// handoff refuses "addressed NOWHERE"). `HAS_MASK` is the builder's own `mask_prefix`,
/// and `ONE_PASS` the builder's own one-pass arm condition (`live.len() == 1 && mq > 1 &&
/// !mask_prefix && seq_len == mq`).
///
/// The facts arrive as ONE struct — the group is the shape's own frame (geometry, widths,
/// segment extents, arm selectors), minted by `lower_attn` beside the laws that derive
/// each member, so this mint holds no derivation of its own.
struct AttnFacts {
    nqh: u32,
    nkvh: u32,
    hd: u32,
    gqa: u32,
    mq: u32,
    cap: u32,
    prefix_len: u32,
    kc_row: u32,
    new_len: u32,
    scale: f32,
    has_mask: bool,
    one_pass: bool,
}

fn attn_kernel_spec<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    f: &AttnFacts,
) -> Result<KernelSpec, String> {
    let AttnFacts {
        nqh,
        nkvh,
        hd,
        gqa,
        mq,
        cap,
        prefix_len,
        kc_row,
        new_len,
        scale,
        has_mask,
        one_pass,
    } = *f;
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    // The door's positional order: q, out, kc, kd, vc, vd — the cache's K before the new
    // block's K, then the same for V — then the mask when a segment reads it.
    let mut params: Vec<&str> = vec![
        "desc_q", "desc_o", "desc_kc", "desc_kd", "desc_vc", "desc_vd",
    ];
    // ⛔ THE MASK PARAMETER IS ALWAYS IN THE SIGNATURE: the kernel's parameter list is static
    // Python, so a shape without a mask still declares `desc_mask` — and the trace (every
    // descriptor and load under `if HAS_MASK:` / `if SWEPT > 0`) never references it, so the
    // pipeline DCEs the unaddressed parameter and the compiled program carries exactly the
    // arguments the builder's own arm states (six, for a maskless shape). Same law as the dead
    // prefix's cache views: the program's parameter set is decided by what the trace addresses.
    params.push("desc_mask");
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
    // The segment corners, from the node's own regions — the row/col starts the builder's
    // `KtirSeg` states (`kr.region.rows.start`, `kr.region.cols.start`). The MASKED prefix's
    // row corner is 0 by the builder's own law (`row_start = 0` at `mask_prefix && i == 0`),
    // which is why `kc_row` arrives as its own parameter and is not read off the region.
    let (kc_c, kd_r, kd_c) = (
        node.inputs[1].region.cols.start,
        node.inputs[3].region.rows.start,
        node.inputs[3].region.cols.start,
    );
    ce("NQH", Val::Int(i128::from(nqh)))?;
    ce("NKVH", Val::Int(i128::from(nkvh)))?;
    ce("HD", Val::Int(i128::from(hd)))?;
    ce("GQA", Val::Int(i128::from(gqa)))?;
    ce("MQ", Val::Int(i128::from(mq)))?;
    ce("CAP", Val::Int(i128::from(cap)))?;
    ce("SWEPT", Val::Int(i128::from(prefix_len)))?;
    ce("NEW_LEN", Val::Int(i128::from(new_len)))?;
    ce("KC_ROW", Val::Int(i128::from(kc_row)))?;
    ce("KC_COL", Val::Int(i128::from(kc_c)))?;
    ce("KD_ROW", Val::Int(i128::from(kd_r)))?;
    ce("KD_COL", Val::Int(i128::from(kd_c)))?;
    ce("SCALE", Val::Float(f64::from(scale)))?;
    ce("HAS_MASK", Val::Bool(has_mask))?;
    ce("ONE_PASS", Val::Bool(one_pass))?;
    Ok(KernelSpec {
        kernel: "attn_fwd".to_string(),
        signature,
        constexprs,
        file: "attn.py".to_string(),
    })
}

/// THE DEAD PREFIX'S VIEWS — one `ktdp.construct_memory_view` per resident-cache
/// parameter, spliced at the top of the compiled kernel's body.
///
/// ⛔ THE PARAMETER POSITIONS ARE THE DOOR'S, and the injection must find its own
/// parameter: the kernel's pointer parameters are `desc_q` (0), `desc_o` (1), `desc_kc`
/// (2), `desc_kd` (3), `desc_vc` (4), `desc_vd` (5), so the cache parameters are 2 and 4.
/// The view shape is the segment's own `[cap, kv_width]` — the capacity
/// `attn_operands` reads as `cap = kc.v_rows`, exactly what a live segment's view states
/// and what the builder's dead segment keeps.
fn inject_dead_cache_views(
    module: &mut triton_ktir::ir::Module,
    cap: u32,
    kv_width: u32,
) -> Result<(), String> {
    use triton_ktir::ir::{Attr, AttrKey, IrType, Op, OpKind};
    // Mint the view names BEFORE the body borrow: `fresh_named` needs `&mut module` and
    // the splice needs `&mut` the body, so the two phases cannot interleave (the rung-3
    // rewrite's own law).
    let view_names = [module.fresh_named("kc_dead"), module.fresh_named("vc_dead")];
    let kernel = module
        .kernel_mut()
        .map_err(|e| format!("triton splice: attn dead-view injection: {e}"))?;
    let region = kernel.regions.first_mut().ok_or_else(|| {
        "triton splice: attn dead-view injection: the kernel has no body".to_string()
    })?;
    // The dead segment's view shape, as the ladder's own `build_base_memory_view` spells
    // it: `Shape`/`Strides` as element counts, `CoordinateSet` from the range-set law, HBM.
    let dims = vec![i64::from(cap), i64::from(kv_width)];
    let range_set = format!(
        "(d0, d1) : (d0 >= 0, -d0 + {r} >= 0, d1 >= 0, -d1 + {c} >= 0)",
        r = cap - 1,
        c = kv_width - 1,
    );
    // Positions 2 and 4: the resident K and V cache parameters (`desc_kc`, `desc_vc`).
    for (n, (view, param_idx)) in view_names.iter().zip([2usize, 4usize]).enumerate() {
        let Some(&(ptr, _)) = region.args.get(param_idx) else {
            return Err(format!(
                "triton splice: attn dead-view injection: the kernel states fewer than {} \
                 parameters — the door's order is q, out, kc, kd, vc, vd",
                param_idx + 1
            ));
        };
        // The parameter must be addressed nowhere — that is the dead prefix's own shape,
        // and a view that already exists means the kernel's guards diverged from SWEPT==0.
        if region
            .ops
            .iter()
            .any(|o| o.kind == OpKind::KtdpConstructMemoryView && o.operands.first() == Some(&ptr))
        {
            return Err(format!(
                "triton splice: attn dead-view injection: parameter {param_idx} already states \
                 a view — the kernel's SWEPT==0 guards left the cache descriptors live, and \
                 injecting a second view over one pointer would give the door two extents to \
                 read where it takes the first"
            ));
        }
        let view_op = Op::new(OpKind::KtdpConstructMemoryView)
            .with_result(
                *view,
                IrType::MemRef {
                    dims: dims.clone(),
                    elem: triton_ktir::ir::DType::F16,
                },
            )
            .with_operands([ptr])
            .with_attr(AttrKey::Shape, Attr::IntList(dims.clone()))
            .with_attr(
                AttrKey::Strides,
                Attr::IntList(vec![i64::from(kv_width), 1]),
            )
            .with_attr(AttrKey::CoordinateSet, Attr::AffineSet(range_set.clone()))
            .with_attr(AttrKey::MemorySpace, Attr::Str("HBM".into()));
        // Splice at the TOP of the body, in parameter order: a view is loop-invariant by
        // construction (its one operand is the base address), and `regions()` finds it by
        // a non-recursive search of the function's top-level ops.
        region.ops.insert(n, view_op);
    }
    Ok(())
}

/// THE MASKLESS SHAPE'S PARAMETER TRUNCATION — drop the tail `desc_mask` parameter the
/// kernel declares but no arm traced a view over.
///
/// ⛔ THE PARAMETER MUST BE UNUSED **AND LAST**. The consumer numbers its buffers by
/// parameter position, so only removing the tail renumbers nothing — anything but the last
/// slot would silently re-index every later buffer. And the parameter must be addressed
/// NOWHERE (the same `every_parameter_states_its_width` precondition the handoff enforces):
/// a body that still holds a view over it means `HAS_MASK` and the trace's own guards
/// disagreed, which is a kernel bug this refuses rather than papers over.
fn truncate_unused_mask_param(module: &mut triton_ktir::ir::Module) -> Result<(), String> {
    use triton_ktir::ir::OpKind;
    let kernel = module
        .kernel_mut()
        .map_err(|e| format!("triton splice: attn mask truncation: {e}"))?;
    let region = kernel
        .regions
        .first_mut()
        .ok_or_else(|| "triton splice: attn mask truncation: the kernel has no body".to_string())?;
    let Some(&(ptr, _)) = region.args.last() else {
        return Err(
            "triton splice: attn mask truncation: the kernel states no parameters".to_string(),
        );
    };
    if region
        .ops
        .iter()
        .any(|o| o.kind == OpKind::KtdpConstructMemoryView && o.operands.first() == Some(&ptr))
    {
        return Err(
            "triton splice: attn mask truncation: the last parameter is addressed by a view — \
             HAS_MASK is false for this shape, so the kernel's mask guards left the mask \
             descriptor live; the maskless arm must not reference `desc_mask` at all"
                .to_string(),
        );
    }
    region.args.pop();
    Ok(())
}

/// THE ATTENTION MINT — the adapter's `node_for` compilation ([`attn_positional`]), then the
/// attention's own re-state: the bindings in the DOOR's parameter order (q, out, then the
/// segment pairs, with the output SECOND, not last), the mask as a bound synthetic tensor id
/// (the graph's next free one — the builder's own `graph.tensors.len()` law for the mask id),
/// and the builder's `attn_s{id}` name.
fn mint_attn<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    positional: KtirNode,
    mask_bound: bool,
) -> Result<KtirNode, String> {
    // ⛔ THE DOOR'S ORDER, NOT `lower_one`'s: q (inputs[0]), OUT (the node's output), then
    // kc (inputs[1]), kd (inputs[3]), vc (inputs[2]), vd (inputs[4]) — the order
    // `KtirFunc::attn`'s `arg_for` mints and `attn_operands` reads back. `regions()` zips
    // `bindings[i]` against `arguments[i]`, so a swapped binding is a descriptor over the
    // wrong tensor, and the door's `is_out` check is what catches a transposed q/out.
    let t =
        |tr: &scratchy_subtile::subtile_ir::TensorRegion| BufferId::new(tr.tensor.index() as u32);
    let mut bindings = vec![
        t(&node.inputs[0]),
        t(&node.output),
        t(&node.inputs[1]),
        t(&node.inputs[3]),
        t(&node.inputs[2]),
        t(&node.inputs[4]),
    ];
    // ⭐ THE MASK'S BINDING SLOT — the synthetic tensor BEYOND the graph, numbered the way
    // the builder numbers it (`self.graph.tensors.len()`): the door skips the parameter
    // by this very tid (`k.mask`), so beyond-the-graph is the only requirement, and the
    // builder's own law is the graph's extent.
    let mask_tid = mask_bound.then(|| BufferId::new(ir.tensors.len() as u32));
    if let Some(m) = mask_tid {
        bindings.push(m);
    }
    let name = Arena::global().str(format!("attn_s{}", node.id.index()));
    let KtirNode { func, program, .. } = positional;
    if func.arguments.len() != bindings.len() {
        return Err(format!(
            "triton splice: {} compiled to {} parameters but the door's order states {} \
             bindings — the kernel's parameter list diverged from the row's contract",
            func.name,
            func.arguments.len(),
            bindings.len()
        ));
    }
    Ok(KtirNode {
        func: ktir_core::ir::IRFunction { name, ..func },
        program,
        bindings,
        mask: mask_tid,
        node_out_tid: None,
        minted: Default::default(),
    })
}

/// THE KERNEL'S LAUNCH CONTRACT, stated from the node's own shapes — the signature (one
/// descriptor per operand plus the output), the constexprs (the node's facts as
/// `tl.constexpr`s, the monomorphisation key), all in the case-table's own shape
/// (`cases::spec`).
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
        // ⭐ THE (1 + w) FORM'S OWN SPEC — the same signature and constants as the Scale
        // arm, plus the offset itself. The value is the convention's own `offset()`
        // (1.0; `GainConvention::from_offset` refuses every other value), and it is the
        // SAME value `compute_bundle_layout` collects into `scalarmul_scales` and the
        // door reads back off the program's splat — one constant, three readers, one
        // value, the epsilon's law.
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
            // ONE tile, the whole region — `KtirFunc::matmul`'s own whole-region law
            // (one linalg.matmul, no K loop). `M_TOTAL` names the OUTPUT TENSOR's row
            // extent (the storage the descriptor addresses) while the store takes the
            // `[M, N]` tile at row 0 — the prefill lm-head fold's m=1 tail, whose
            // output is row 0 of the `[mq, vocab]` logits storage.
            // ⛔ `N` IS THE DEVICE WIDTH — the door's own padding law, stated here so
            // the view/tile name the storage every consumer addresses: the card writes
            // `n_dev` columns (`out_width_the_weight_holds`'s staged-weight contract),
            // the layout reserves `n_dev` (`pointwise_width_the_output_holds`'s
            // case-(b) proof), and a pointwise consumer on the same buffer (granite's
            // logits scalarmul, whose chunk-6 window is `for_pointwise`-padded) needs
            // the VIEW to hold the tile. `for_matmul` is IDEMPOTENT at a padded width
            // and the weight-holds drop is the door's, applied to both paths — so this
            // spelling and the builder's logical one reach the same emit. The fp8 arm
            // stays LOGICAL: its door arm returns early through
            // `matmul_fp8_descriptors(m, k, n)` and is not pad-idempotent.
            let (m_total, _a_total) = matmul_window_of(node, ir)?;
            let n_dev = ktir_superdsc::work::DeviceWidth::for_matmul(m, *n, k, false).get();
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
            // ⛔ AND SO IS `N_TOTAL` — the STORAGE is the device width too: the
            // producer matmul wrote `for_matmul`-padded columns and the layout
            // reserved exactly that (the weight-holds / output-holds proofs both rely
            // on the pad being real), so a view at the logical width understates the
            // buffer and a device-width TILE at the chunk corner leaves it — granite's
            // chunk 6 (`[31, 512]` at 49152 of a 49159-wide view) is exactly that
            // refusal. `for_pointwise` of the logical storage IS the producer's
            // `for_output` width (the documented equality, for any matmul above the
            // util floor), so the widened view names the storage the placement
            // actually holds.
            let c_dev = ktir_superdsc::work::DeviceWidth::for_pointwise(c).get();
            let (block_m, n_blocks, tail_h) = blocks_of(m, c_dev, 3);
            let n_total_dev = ktir_superdsc::work::DeviceWidth::for_pointwise(n_total).get();
            ce("M", Val::Int(i128::from(m)))?;
            ce("N", Val::Int(i128::from(c_dev)))?;
            ce("BLOCK_M", Val::Int(i128::from(block_m)))?;
            ce("BLOCK_N", Val::Int(i128::from(c_dev)))?;
            ce("N_TOTAL", Val::Int(i128::from(n_total_dev)))?;
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
            // only this arm reads the facts: all four regions share ONE row
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
                        node.output.tensor.index(),
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
            ce("HEAD_DIM", Val::Int(i128::from(hd)))?;
            ce("HALF", Val::Int(i128::from(hd / 2)))?;
            ce("Y_TOTAL", Val::Int(i128::from(y_total)))?;
            ce("R_START", Val::Int(i128::from(r_start)))?;
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
            // ROW BLOCKING, the builder's own law: three live tiles (x, the tanh's
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
            // M / N — the logical width (the door's own `check_pointwise_cols` law for
            // this family; the weight is read in the scalar broadcast mode and pads its
            // own placement). ⛔ THE WEIGHT IS NOT A DESCRIPTOR: a `[1]`-region load
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
            let stk: u32 = 64; // Fp16::ELEMS_PER_STICK — the door's own stick law
            if c_in % stk != 0 || c_out % stk != 0 {
                return Err(format!(
                    "triton splice: reshape t{}: a {c_in}- or {c_out}-wide side is not a whole \
                     {stk}-fp16 stick — a device tensor is stick-blocked, so flat elements \
                     {stk}q..{stk}q+{stk} are one `[1, {stk}]` tile of each side only when both \
                     column extents are whole sticks (the door's own law, stated here so the \
                     walk's `// {stk}` arithmetic is well-formed)",
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
        (SubOp::Sample { vocab, rows }, "sample_fwd") => {
            // The token draw's own facts, all derived from the node's vocab and
            // row count — never a model name: PAD_V is the power-of-two padded
            // row, STRICTLY above the vocab (the descriptor block's law and the
            // sentinel's — a power-of-two vocab still needs pad lanes), ONE
            // derivation shared with the host reference and the load-time table
            // staging), NB its block count, BLOCK the stick (the two-level
            // ramp's lane width and the output stick), B the row count this
            // rung serves (the grid's own width). The signature states every
            // kernel parameter's dtype: the fp16 row, the two PER-ROW staged
            // operands in f32 (the noise block, the temperature reciprocals),
            // the nine tables in f32 (the selection is f32 — a sum the f16
            // grid would round), and the i32 output stick (the
            // index-ingestion law).
            let pad_v = scratchy_subtile::subtile_ir::sample_pad_vocab(*vocab);
            let nb = pad_v / scratchy_subtile::subtile_ir::SAMPLE_BLOCK;
            signature.insert(
                "logits_ptr".to_string(),
                ArgSpec::parse("*fp16").map_err(|e| e.to_string())?,
            );
            for p in [
                "noise_ptr",
                "invt_ptr",
                "bramp_ptr",
                "lramp_ptr",
                "binc_ptr",
                "linc_ptr",
                "ones_ptr",
                "sixtyfour_ptr",
                "padoff_ptr",
                "gramp_ptr",
                "ginc_ptr",
            ] {
                signature.insert(
                    p.to_string(),
                    ArgSpec::parse("*fp32").map_err(|e| e.to_string())?,
                );
            }
            signature.insert(
                "idx_ptr".to_string(),
                ArgSpec::parse("*i32").map_err(|e| e.to_string())?,
            );
            let mut ce = |k: &str, v: Val| -> Result<(), String> {
                signature.insert(k.to_string(), ArgSpec::Constexpr);
                constexprs.insert(k.to_string(), v);
                Ok(())
            };
            ce("V", Val::Int(i128::from(*vocab)))?;
            ce("PAD_V", Val::Int(i128::from(pad_v)))?;
            ce("NB", Val::Int(i128::from(nb)))?;
            ce(
                "BLOCK",
                Val::Int(i128::from(scratchy_subtile::subtile_ir::SAMPLE_BLOCK)),
            )?;
            ce("B", Val::Int(i128::from(*rows)))?;
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
        // The same one-work-item law: the row blocks are in-kernel tiles. The MoE
        // sub-ops have no grid row any more — the registry refuses them before the
        // splice (the MoE lowers only from `kernels/moe.py` through KTIR).
        SubOp::TanhSoftCap { .. }
        | SubOp::RmsNormUnit { .. }
        | SubOp::ScalarWeightMul
        | SubOp::Reshape { .. } => Ok(vec![1]),
        // ⭐ ONE PROGRAM PER REQUEST ROW: the sampler's grid is the rung's
        // own row count — 1 on the solo rung, the batched rung's baked width
        // on every batch rung. Every per-row operand is a `[B, ·]` descriptor
        // this program's own row indexes (`[pid, 0]`).
        SubOp::Sample { rows, .. } => Ok(vec![i64::from(*rows)]),
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
/// script, ~17 s per attention program alone (~26 programs in the golden's 442 s run),
/// where main's direct-emission builder paid none of it. The compile is a pure function
/// of (source, spec, grid) — the ladder is deterministic — so the second and later
/// callers get a CLONE of the first result. Every consumer mutates only its own copy
/// (`mint`/`mint_attn` re-state names and bindings; `inject_dead_cache_views` and
/// `truncate_unused_mask_param` edit their own copy), so a cached pristine `Module` is
/// safe to hand out again.
fn compile_kernel(
    src: &str,
    spec: &KernelSpec,
    grid: &[i64],
) -> Result<triton_ktir::ir::Module, String> {
    /// The memo. The key is [`memo_key`]'s canonical `String`, so it needs no `Hash`/`Ord` on
    /// `ArgSpec`/`Val` and is deterministic in construction.
    static CACHE: Memo<triton_ktir::ir::Module> = Memo::new();
    CACHE.get_or_compute(memo_key(src, spec, grid), || {
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

/// One key's result: computed at most once, by whichever caller gets there first.
type MemoCell<T> = std::sync::Arc<std::sync::OnceLock<Result<T, String>>>;

impl<T: Clone> Memo<T> {
    const fn new() -> Memo<T> {
        Memo {
            cells: std::sync::Mutex::new(std::collections::BTreeMap::new()),
        }
    }

    /// `f()` once per `key`, cloned out to every caller. A `None` key (a spec [`memo_key`] cannot
    /// spell injectively) is never memoized: it computes every time.
    fn get_or_compute(
        &self,
        key: Option<String>,
        f: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let Some(key) = key else {
            return f();
        };
        let cell = {
            let mut cells = self.cells.lock().unwrap_or_else(|p| p.into_inner());
            std::sync::Arc::clone(cells.entry(key).or_default())
        };
        cell.get_or_init(f).clone()
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
/// row that lowers through `lower_attn`'s door-order bindings — states its
/// widths at the door (the per-parameter view dtype), not here.
///
/// The device-format mapping is the door's own law: a stated dtype with no
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
    // Attention is a Spliced ROW but not a `lower_one` kernel (its door-order
    // bindings and mask slot are `lower_attn`'s); its widths are the door's
    // statement. This names the ROUTING, not a format.
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
    let module = triton_frontend::py::parser::parse(src)
        .map_err(|e| format!("triton splice: {file}: {e}"))?;
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

/// One stated pointer dtype → its device format — the door's own mapping law
/// (F16→Fp16, BF16→Bf16, F32→Fp32, F8E4M3FN→Fp8, i32→SenUint32): a dtype with
/// no device format is refused BY NAME.
fn df_of_stated_type(param: &str, t: &triton_frontend::ttir::Type) -> Result<Df, String> {
    use triton_frontend::ttir::{FloatKind, Type};
    Ok(match t {
        Type::Float(FloatKind::F16) => Df::Fp16,
        Type::Float(FloatKind::BF16) => Df::Bf16,
        Type::Float(FloatKind::F32) => Df::Fp32,
        Type::Float(FloatKind::F8E4M3FN) => Df::Fp8,
        Type::Int(32, _) => Df::SenUint32,
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

/// Mint the [`KtirNode`]: the adapter's `node_for` compilation, then RE-STATE the fields
/// the tape owns — the name (the builder's naming law), the bindings (resolved by
/// [`default_bindings`]: the tape's tensor indices, the output LAST), `mask` (none for
/// rmsnorm), and `node_out_tid` (None — the program writes its own node's output).
///
/// ⛔ THE RE-STATE IS THE WHOLE POINT. `node_for` mints positional `BufferId`s
/// (`0, 1, 2`); the tape's rmsnorm node reads `t37`/`t41` and writes `t42`. The
/// bindings are ours to state, exactly as `KtirFunc::finish_shaped` states them, because
/// only the splice knows which graph tensor each kernel parameter addresses.
fn mint<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    module: triton_ktir::ir::Module,
    program: &Program,
    bindings: Vec<BufferId>,
) -> Result<KtirNode, String> {
    // The adapter's mint: `to_ktir_emit::lower` over the module, positional buffer ids.
    let positional = triton_ktir_superdsc::node_for(&module, *program)
        .map_err(|e| format!("triton splice: node_for: {e}"))?;
    // ⭐ THE NAME IS THE BUILDER'S LAW, so the op_name and the emulator's function key
    // follow the same law the builder path established — what the golden pins.
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
        mask: None,
        node_out_tid: None,
        minted: Default::default(),
    })
}

/// The program stem the builder's naming law uses for this node (`rmsnorm_s{id}`,
/// `add_s{id}`, …). ⛔ READ OFF THE NODE, NOT THE ROW: the elementwise rows carry the
/// CONSUMER's kind (`Elementwise::Add`), and the builder's name comes from the
/// PRODUCER's `EwKind` (`ew_kind_stem` — BiasAdd is named `add`, not `biasadd`). One
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
            // ⛔ NO `_` ARM. A spliced elementwise kind has a row above; reaching here
            // with a kind that has none is an unreachable — the same discipline the
            // arity match holds.
            other => unreachable!("elementwise kind {other:?} has no program stem"),
        };
    }
    // ⭐ THE 2a KINDS THAT SHARE ANOTHER FAMILY'S PROGRAM but keep their OWN builder
    // names — read off the NODE, like the elementwise stems above, because the
    // program match below would answer the shared family's name.
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
        // ⭐ THE SAMPLER'S OWN STEM — #813's argmax law: a spliced whole-function kernel
        // keeps its own name (`argmax_s{id}`), never the generic `triton`, so a bake log
        // names the draw's descriptors for what they are.
        Program::Sample => "sample",
        _ => "triton",
    }
}

#[cfg(test)]
mod production_ktir_dump {
    //! ⭐ THE LEGALIZE-TYPES BYTE-IDENTITY PROOF DRIVER (auditor order, 2026-10-26):
    //! `LegalizeTypes` runs on EVERY kernel the ladder compiles, so a change that widens
    //! what stays f32 could move bytes in kernels that work today. This test dumps the
    //! post-`to_ktir` module text — `triton_ktir::text::print::print`, the deterministic
    //! module printer — for EVERY production kernel in `crates/targets/spyre/kernels/`
    //! except `sample_fwd.py` (the sampler this proof guards), each at one representative
    //! admitted geometry, into `<workspace>/target/spyre-ktir-dump/`.
    //!
    //! The proof is the EXTERNAL DIFF, not this test: run once with the tree's
    //! `legalize_types.rs`, once with the pre-change file swapped in, and diff the two dump
    //! directories — byte-identical per kernel is the bar, and any nonzero diff names the
    //! kernel it stopped on. That is why the test is `#[ignore]`d: it produces evidence,
    //! asserts nothing about the numbers, and would be dead weight in every ordinary test
    //! run. Run it explicitly with
    //! `cargo test -p scratchy-triton-splice --lib production_ktir_dump -- --ignored`.
    //!
    //! Every spec below mirrors the `kernel_spec` arm (or `attn_kernel_spec` /
    //! `lower_sandwich_node` / `lower_prefill_lm_head_fold` / the MoE numeric tests'
    //! golden geometry) that the production bake states for that kernel — cited per entry
    //! — at a whole-region decode-shaped geometry so every blocking law answers the
    //! one-tile constants. The compile goes through the crate's own [`compile_kernel`]:
    //! the exact memoized ladder (`codegen::compile` → `make_ttir` → `from_ttir` →
    //! `make_ktir` → `to_ktir`) the bake runs, so the dump cannot drift from what
    //! production emits.
    use super::*;

    const ATTN: &str = include_str!("../../../targets/spyre/kernels/attn.py");
    const ELEMENTWISE: &str = include_str!("../../../targets/spyre/kernels/elementwise.py");
    const LMLAST: &str = include_str!("../../../targets/spyre/kernels/lmlast.py");
    const MATMUL: &str = include_str!("../../../targets/spyre/kernels/matmul.py");
    const MATMUL_FP8: &str = include_str!("../../../targets/spyre/kernels/matmul_fp8.py");
    const MOE: &str = include_str!("../../../targets/spyre/kernels/moe.py");
    const MOE_PREFILL: &str = include_str!("../../../targets/spyre/kernels/moe_prefill.py");
    const RESHAPE: &str = include_str!("../../../targets/spyre/kernels/reshape.py");
    const RMSNORM: &str = include_str!("../../../targets/spyre/kernels/rmsnorm.py");
    const RMSNORMUNIT: &str = include_str!("../../../targets/spyre/kernels/rmsnormunit.py");
    const ROPE: &str = include_str!("../../../targets/spyre/kernels/rope.py");
    const SAMPLE: &str = include_str!("../../../targets/spyre/kernels/sample_fwd.py");
    const SCALARMUL: &str = include_str!("../../../targets/spyre/kernels/scalarmul.py");
    const SCALARWMUL: &str = include_str!("../../../targets/spyre/kernels/scalarwmul.py");
    const SILUMUL: &str = include_str!("../../../targets/spyre/kernels/silumul.py");
    const TANHSOFTCAP: &str = include_str!("../../../targets/spyre/kernels/tanhsoftcap.py");
    const WINDOWEDRMSNORM: &str = include_str!("../../../targets/spyre/kernels/windowedrmsnorm.py");

    /// The case table's `spec` shape: pointer parameters by stated dtype, then the
    /// constexprs — every constexpr is BOTH a signature entry and a binding.
    fn spec(
        file: &str,
        kernel: &str,
        ptrs: &[(&str, &str)],
        ces: &[(&str, Val)],
    ) -> KernelSpec {
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

    use ktir_superdsc::superdsc_opspec::{
        EW_BINARY_LIVE_TILES, EW_SILU_LIVE_TILES, SILU_MUL_LIVE_TILES,
    };

    /// One entry per (kernel file, entry point) — the multi-entry files (`elementwise.py`,
    /// `rmsnorm.py`, `windowedrmsnorm.py`, `matmul.py`, `attn.py`) state every entry, so a
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
                    &[("desc_x", "*fp16"), ("desc_w", "*fp16"), ("desc_o", "*fp16")],
                    &ces,
                ),
                vec![1],
            ));
        }
        // ── matmul.py (Dense): N is the DEVICE width, the arm's own padding law. ──
        let n_dev = ktir_superdsc::work::DeviceWidth::for_matmul(1, 2048, 2048, false).get();
        v.push((
            "matmul_fwd",
            MATMUL,
            spec(
                "matmul.py",
                "matmul_fwd",
                &[("desc_a", "*fp16"), ("desc_w", "*fp16"), ("desc_o", "*fp16")],
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
        // ── matmul_fp8.py: the arity-3 twin, LOGICAL widths (the arm's own early-return
        // door is not pad-idempotent), one whole-region tile. ──
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
                &[("desc_a", "*fp16"), ("desc_b", "*fp16"), ("desc_o", "*fp16")][..],
                EW_BINARY_LIVE_TILES,
            ),
            (
                "mul_fwd",
                &[("desc_a", "*fp16"), ("desc_b", "*fp16"), ("desc_o", "*fp16")][..],
                EW_BINARY_LIVE_TILES,
            ),
            (
                "sub_fwd",
                &[("desc_a", "*fp16"), ("desc_b", "*fp16"), ("desc_o", "*fp16")][..],
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
                ew("elementwise.py", entry, &ptrs, live),
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
                &[("desc_g", "*fp16"), ("desc_u", "*fp16"), ("desc_o", "*fp16")],
                SILU_MUL_LIVE_TILES,
            ),
            vec![1],
        ));
        // ── scalarmul.py: the WINDOW is the device width — `for_pointwise` on both N and
        // N_TOTAL, the arm's own law. ──
        {
            let c_dev = ktir_superdsc::work::DeviceWidth::for_pointwise(2048).get();
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
        // ── lmlast.py: the extraction, at the row the SSOT names for MQ=5. ──
        v.push((
            "lmlast_fwd",
            LMLAST,
            spec(
                "lmlast.py",
                "lmlast_fwd",
                &[("desc_src", "*fp16"), ("desc_dst", "*fp16")],
                &[
                    ("MQ", Val::Int(5)),
                    ("HIDDEN", Val::Int(2048)),
                    (
                        "ROW",
                        Val::Int(
                            scratchy_subtile::sdsc_abstract::selector_lastrow_col(5) as i128,
                        ),
                    ),
                    ("N_STICKS", Val::Int(32)),
                ],
            ),
            vec![1],
        ));
        // ── windowedrmsnorm.py: all three entries of the one body. ──
        for (entry, gained, oneplus) in [
            ("windowedrmsnorm_gained_fwd", true, false),
            ("windowedrmsnorm_oneplus_fwd", true, true),
            ("windowedrmsnorm_unit_fwd", false, false),
        ] {
            let ptrs: &[(&str, &str)] = if gained {
                &[("desc_x", "*fp16"), ("desc_gamma", "*fp16"), ("desc_o", "*fp16")]
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
        // ── attn.py: the three constexpr-selected arms — decode with a live masked
        // prefix, the causal one-pass, and the whole-chunk continuation. The grid is the
        // head count at MQ > 1 (`attn_positional`'s own law), [1] at decode. ──
        let attn_ptrs: &[(&str, &str)] = &[
            ("desc_q", "*fp16"),
            ("desc_o", "*fp16"),
            ("desc_kc", "*fp16"),
            ("desc_kd", "*fp16"),
            ("desc_vc", "*fp16"),
            ("desc_vd", "*fp16"),
            ("desc_mask", "*fp16"),
        ];
        for (name, mq, swept, new_len, one_pass, grid) in [
            ("attn_decode", 1i128, 127i128, 1i128, false, vec![1i64]),
            ("attn_one_pass", 64, 0, 64, true, vec![8]),
            ("attn_continuation", 64, 64, 64, false, vec![8]),
        ] {
            v.push((
                name,
                ATTN,
                spec(
                    "attn.py",
                    "attn_fwd",
                    attn_ptrs,
                    &[
                        ("NQH", Val::Int(8)),
                        ("NKVH", Val::Int(2)),
                        ("HD", Val::Int(128)),
                        ("GQA", Val::Int(4)),
                        ("MQ", Val::Int(mq)),
                        ("CAP", Val::Int(2048)),
                        ("SWEPT", Val::Int(swept)),
                        ("NEW_LEN", Val::Int(new_len)),
                        ("KC_ROW", Val::Int(0)),
                        ("KC_COL", Val::Int(0)),
                        ("KD_ROW", Val::Int(0)),
                        ("KD_COL", Val::Int(0)),
                        ("SCALE", Val::Float(0.125)),
                        ("HAS_MASK", Val::Bool(true)),
                        ("ONE_PASS", Val::Bool(one_pass)),
                    ],
                ),
                grid,
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
        // ── sample_fwd.py: the token draw at solo and batched grids — the sampler this
        // proof guards, and the ONE kernel the grid-[B] fix is allowed to move. The
        // spec mirrors the `SubOp::Sample` arm of `kernel_spec`'s registry row: the
        // logits row, its per-row noise and temperature reciprocal, the nine staged
        // tables, and the `[B, 64]` idx stick out. ──
        {
            let sample_ptrs: &[(&str, &str)] = &[
                ("logits_ptr", "*fp16"),
                ("noise_ptr", "*fp32"),
                ("invt_ptr", "*fp32"),
                ("bramp_ptr", "*fp32"),
                ("lramp_ptr", "*fp32"),
                ("binc_ptr", "*fp32"),
                ("linc_ptr", "*fp32"),
                ("ones_ptr", "*fp32"),
                ("sixtyfour_ptr", "*fp32"),
                ("padoff_ptr", "*fp32"),
                ("gramp_ptr", "*fp32"),
                ("ginc_ptr", "*fp32"),
                ("idx_ptr", "*i32"),
            ];
            for (name, b, grid) in [
                ("sample_fwd_b1", 1i128, vec![1i64]),
                ("sample_fwd_b2", 2, vec![2]),
                ("sample_fwd_b32", 32, vec![32]),
            ] {
                v.push((
                    name,
                    SAMPLE,
                    spec(
                        "sample_fwd.py",
                        "sample_fwd",
                        sample_ptrs,
                        &[
                            ("V", Val::Int(49155)),
                            ("PAD_V", Val::Int(65536)),
                            ("NB", Val::Int(1024)),
                            ("BLOCK", Val::Int(64)),
                            ("B", Val::Int(b)),
                        ],
                    ),
                    grid,
                ));
            }
        }
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
            "lmlast_fwd" => Program::LmLast,
            "windowedrmsnorm_gained_fwd" | "windowedrmsnorm_oneplus_fwd" => {
                Program::WindowedRmsNorm { unit: false }
            }
            "windowedrmsnorm_unit_fwd" => Program::WindowedRmsNorm { unit: true },
            "sample_fwd_b1" | "sample_fwd_b2" | "sample_fwd_b32" => Program::Sample,
            other => panic!(
                "`{other}` is in the dump table but `program_of` states no Program for it — \
                 a dumped kernel with no program is a silent skip of the SDSC leg, which the \
                 no-silent-skip law refuses. Add the row."
            ),
        }
    }

    /// The kernels whose SDSC this driver CANNOT emit, each for a stated reason — a
    /// NAMED skip, never a silent one. Everything else must lower or the dump fails.
    fn sdsc_skip_of(name: &str) -> Option<&'static str> {
        match name {
            // The geometry const door: `rope_at::<HD>` / `attn_at::<NQH, NKVH, HD>`
            // monomorphise on head geometry the driver would have to state from the
            // fixture's constexprs (`bake_py`'s rope arm) plus the bundle attention
            // facts no single-kernel driver holds. The KTIR-text leg still covers them.
            "rope_fwd" => Some("the model-geometry const door (rope_at::<HD>)"),
            "attn_decode" | "attn_one_pass" | "attn_continuation" => {
                Some("the model-geometry const door (attn_at::<NQH, NKVH, HD>)")
            }
            // No `Program` variant exists and none is inventable here: the MoE lowers
            // only from `kernels/moe.py` through KTIR, and no per-op row names a
            // program for it. Stating one in a dump driver would be routing invented
            // outside the registry.
            "moe_fwd" | "moe_prefill_fwd" => Some("no Program variant — the MoE has no door row"),
            _ => None,
        }
    }

    /// One emitted op as STABLE TEXT for the byte-identity diff: `wire::SdscOp` derives
    /// `Serialize` (the wire format's own serializer — deterministic maps, no Debug
    /// float spelling), and every non-serializable field rides as its `Debug` string.
    /// `ktir` is dropped on purpose: it is the program the op CARRIES, not bytes the
    /// device consumes, and the `.ktir` file beside this one already pins it.
    fn sdsc_line(op: &ktir_superdsc::emit::EmittedOp) -> String {
        let mut obj = serde_json::Map::new();
        obj.insert("op_name".into(), serde_json::json!(op.op_name));
        obj.insert(
            "op".into(),
            match &op.op {
                Some(o) => serde_json::to_value(o)
                    .unwrap_or_else(|e| panic!("serialize {}: {e}", op.op_name)),
                // `None` is the answer for a KTIR op, not an empty descriptor.
                None => serde_json::Value::Null,
            },
        );
        obj.insert("arg_bindings".into(), serde_json::json!(format!("{:?}", op.arg_bindings)));
        obj.insert("time".into(), serde_json::json!(op.time));
        obj.insert(
            "affine_strides".into(),
            serde_json::json!(format!("{:?}", op.affine_strides)),
        );
        obj.insert("slot_stride_bytes".into(), serde_json::json!(op.slot_stride_bytes));
        obj.insert("slab_write".into(), serde_json::json!(op.slab_write));
        obj.insert(
            "slab_stride_bytes".into(),
            serde_json::json!(op.slab_stride_bytes),
        );
        serde_json::Value::Object(obj).to_string()
    }

    /// Emit one dumped kernel's SDSC through the SAME dispatch the production door
    /// (`scratchy-target-spyre`'s `ktir_superdsc_door::lower`) states, arm for arm:
    /// the pointwise kinds, the matmuls, the sampler and LmLast through
    /// `whole_function::lower_function`; the leaf kinds through their own
    /// `lower_ktir_to_superdsc` bodies; the column-window guard on exactly the kinds
    /// the door wraps it around. This crate cannot call that door (the dependency runs
    /// the other way — the target crate depends on the splice), so the arms are
    /// mirrored here and the mirror is the dump's own cited limitation.
    fn emit_sdsc(name: &str, node: &KtirNode) -> Result<Vec<String>, String> {
        use ktir_superdsc::emit::lower_ktir_to_superdsc as lk;
        use ktir_superdsc::emit::whole_function::lower_function;

        let lk_name = node.func.name;
        let regions =
            lk::regions(node).map_err(|e| format!("{name}: regions: {e:?}"))?;
        let layout = triton_ktir_superdsc::layout::for_regions(node, &regions)
            .map_err(|e| format!("{name}: layout: {e}"))?;
        let windowed = |ops: Result<Vec<ktir_superdsc::emit::EmittedOp>, lk::Error>| {
            let ops = ops.map_err(|e| format!("{name}: {e:?}"))?;
            lk::column_windows_are_honoured(lk_name, &regions, &ops)
                .map_err(|e| format!("{name}: {e:?}"))?;
            Ok::<_, String>(ops)
        };
        let mut sym = 0i64;
        let ops = match program_of(name) {
            // The door's `windowed(...)`-wrapped whole-function kinds.
            Program::Elementwise(_)
            | Program::SiluMul
            | Program::RmsNorm
            | Program::ScalarMul
            | Program::Sample => windowed(lower_function(node, Some(&layout), &mut sym, None)),
            // The door's unwrapped whole-function kinds.
            Program::Matmul | Program::LmLast => Ok(lower_function(
                node,
                Some(&layout),
                &mut sym,
                None,
            )
            .map_err(|e| format!("{name}: {e:?}"))?),
            Program::RmsNormUnit => {
                windowed(lk::rmsnorm_unit(lk_name, node, &regions, &mut sym, Some(&layout)))
            }
            Program::TanhSoftCap => {
                windowed(lk::tanhsoftcap(lk_name, node, &regions, &mut sym, Some(&layout)))
            }
            Program::ScalarWeightMul => {
                windowed(lk::scalarweightmul(lk_name, &regions, &mut sym, Some(&layout)))
            }
            Program::Reshape => Ok(lk::reshape(lk_name, &regions, &mut sym, Some(&layout))
                .map_err(|e| format!("{name}: {e:?}"))?),
            Program::WindowedRmsNorm { .. } => Ok(lk::windowed_rmsnorm(
                lk_name,
                node,
                &regions,
                &mut sym,
                Some(&layout),
            )
            .map_err(|e| format!("{name}: {e:?}"))?),
            // `program_of` states none of these and `sdsc_skip_of` names the ones that
            // reach this far — a geometry-door or MoE name here means the two tables
            // disagree, which is a defect in the driver, not a skip.
            other => {
                return Err(format!(
                    "`{name}` states {other:?}, which the dump's door mirror has no arm for — \
                     `sdsc_skip_of` must name it or `program_of` must not state it"
                ));
            }
        }?;
        if ops.is_empty() {
            return Err(format!(
                "`{name}` emitted ZERO ops — a dump of nothing proves nothing (the \
                 no-silent-skip law, applied to the SDSC leg)"
            ));
        }
        Ok(ops.iter().map(sdsc_line).collect())
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
            eprintln!("dumped {name} -> {} ({} lines)", path.display(), text.lines().count());

            // THE SDSC LEG — the emitted-descriptor half of the same proof: the grid-[B]
            // fix changes `Region`/`windowed_region`/the mint, so the byte-identity bar
            // applies to what the doors EMIT, not only to the module text. Same law:
            // the diff of two runs is the proof; a named skip (`sdsc_skip_of`) is the
            // only kernel without an `.sdsc.jsonl`, and every other kernel must lower.
            match sdsc_skip_of(name) {
                Some(reason) => {
                    eprintln!("SDSC-SKIP {name}: {reason}");
                }
                None => {
                    let node = triton_ktir_superdsc::node_for(&module, program_of(name))
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
                        // ⭐ A KERNEL THE DOOR REFUSES IS A NAMED SKIP, CARRIED AS DATA — the
                        // driver mirrors what the production door lowers and NOTHING wider
                        // (auditor ruling 2026-10-27: never fix the door to suit the driver).
                        // The refusal is the door's own message, written as the kernel's one
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
