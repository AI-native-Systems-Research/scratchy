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
        | SubOp::AllRowsMatmul
        // ⛔ THE CHAINED HEAD'S PAIR — its by-name refusals DIED with the
        // chained-head emission rung: both ops are HAND-MINTED KTIR programs
        // (`lower_one_node`'s chain arms, the spyre emitter — the door bodies
        // `lk::embed_row_gather`/`lk::chain_stage`, never Triton kernels), so
        // they are ordinary members of the "not a spliced kind" class: reaching
        // the splice means the routing changed, not that a kernel is missing.
        | SubOp::EmbedRowGather { .. }
        | SubOp::ChainStage { .. } => TritonKernelRow::Refused(
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
        let states_neither_corner =
            matches!(&node.op, SubOp::TanhSoftCap { .. } | SubOp::Sample { .. });
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
        // ⭐ THE SAMPLER'S ARITY IS 26: the logits row, its per-row seed
        // quadruple's FOUR `[rows, 1]` word tables and its per-row temperature
        // reciprocals, then the twenty staged tables (two decreasing ramps,
        // two increasing index tables, two widen tables, the pad law's offset
        // row, the two-level ramp's group pair, then the noise section's own
        // eleven — the four i32 hash/fold constants, the lane table, the
        // fold's five f32 constants, the fp16 clamp) — every kernel parameter
        // but the output is a tape operand, per the splice's own positional
        // law. (Four word tables, not one `[rows, 4]` table: the Col operand
        // mode addresses the stick head alone — the int-bcast leg's measured
        // law — so each word rides its own corner-free table, the auditor's
        // wall-4 ruling (a); the row word is GONE from the key — the P3
        // localization ruling, 2026-10-27, request-owned noise alone.)
        SubOp::Sample { .. } => 26,
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
//  PAGED ATTENTION — spyre-inference's serving step (`SpyreHeadMajorAttentionImpl`, the decoder
//  backend its default `SPYRE_ATTN_KV_LAYOUT=head_major` selects): the KV cache write
//  (`reshape_and_cache_head_major`) and the online-softmax page walk
//  (`page_attn_head_major_decode` / `_prefill`), as `kernels/reshape_and_cache_head_major.py`,
//  `kernels/page_attn_head_major_decode.py` and `kernels/page_attn_head_major_prefill.py` —
//  their kernels, operand for operand.
//
//  * the K and V CACHE TENSORS the layout names ARE THE POOLS, read head-major as
//    `[pages, nkvh, PAGE_SLOTS, hd]` — the same bytes as the cache's `[cap, nkvh·hd]`, so the
//    graph's own tensors carry the pages and nothing is re-allocated;
//  * a PAGE is the scheduler's KV block ([`PAGE_SLOTS`] slots, the block size the worker hands
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
/// page_idx)`, every kv head) is 128 KB at 8 kv heads × 64 head dims, past dxp's bound on one
/// gathered entry ([`ktir_superdsc::emit::lower_ktir_to_superdsc::GATHER_ENTRY_BYTES_MAX`]); the
/// gather copies such an entry in pieces ([`ktir_superdsc::emit::lower_ktir_to_superdsc::gather_pieces`]).
pub const PAGE_SLOTS: u32 = 128;

/// An int32 stick (spyre-inference's `INT32_ELEMS_PER_STICK`).
pub const INT32_STICK: u32 = 32;

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

/// ⭐ A DECODE BATCH'S CHUNKING — spyre-inference's `batched_decode_chunking`
/// (`spyre_attn_bucketer.py`, 297fc48) over a batch of `seqs` sequences walking `blocks` blocks:
/// `blocks_per_chunk` block slots per chunk, so a chunk gathers `entries = blocks_per_chunk · seqs`
/// pages (their target: one per core), and `num_chunks` chunks cover the walk.
///
/// One difference, Triton's: its gather takes at least 8 indices (`semantic.py`'s
/// `descriptor_gather`), so a chunk is at least `ceil(8 / seqs)` block slots; the slots past a
/// sequence's blocks gather page 0 under a fully masked tile, their own padding law ("padding
/// columns gather page 0 under an all -inf mask and contribute zero").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchChunking {
    pub seqs: u32,
    pub blocks_per_chunk: u32,
    pub num_chunks: u32,
}

impl BatchChunking {
    /// `spyre_attn_bucketer._SPYRE_CORE_COUNT`: the cores a chunk's entries are sized to.
    pub const CORES: u32 = 32;
    /// Triton's gather floor.
    pub const GATHER_FLOOR: u32 = 8;

    pub const fn of(seqs: u32, blocks: u32) -> BatchChunking {
        let theirs = {
            let j = Self::CORES / seqs;
            let j = if j < blocks { j } else { blocks };
            if j < 1 { 1 } else { j }
        };
        let floor = Self::GATHER_FLOOR.div_ceil(seqs);
        let blocks_per_chunk = if theirs < floor { floor } else { theirs };
        BatchChunking {
            seqs,
            blocks_per_chunk,
            num_chunks: blocks.div_ceil(blocks_per_chunk),
        }
    }

    /// Pages one chunk gathers.
    pub const fn entries(self) -> u32 {
        self.blocks_per_chunk * self.seqs
    }
}

/// ⭐⭐⭐ THE PAGED STEP'S HOST SOURCES — the buffers a paged forward's attention and KV write read
/// besides the pools, at their RESERVED tids (`ktir_superdsc::reserved_tids::PAGED_STEP_BASE`): one
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
    /// `kv_row_pool` `[num_pages, num_kv_heads, 1]` int32: every page's folded-cache rows
    /// `page * num_kv_heads + kv` (the decode kernel's).
    pub kv_row_pool: u32,
    /// A decode batch's `rep_row_ids` `[entries]` int32: entry `j · seqs + s` holds sequence `s`'s
    /// query row.
    pub rep_rows: u32,
    /// A decode batch's `chunk_page_ids` `[num_chunks, entries, 1]` int32: chunk `c`'s entry
    /// `j · seqs + s` holds the page of block slot `j` of sequence `s`.
    pub chunk_pages: u32,
    /// A decode batch's `mask_by_chunk` `[num_chunks, blocks_per_chunk, seqs, num_kv_heads, 1,
    /// block_size]`.
    pub chunk_masks: u32,
    /// The decode step's int32 zero stick, staged beside `kv_row_pool`: the add identity the chained
    /// index's run copy reads on the card.
    pub int32_zero: u32,
}

impl PagedStepTids {
    pub const RESERVED: PagedStepTids = PagedStepTids {
        mask: ktir_superdsc::reserved_tids::PAGED_MASK_TID,
        page_table: ktir_superdsc::reserved_tids::PAGED_PAGE_TABLE_TID,
        write_rows: ktir_superdsc::reserved_tids::PAGED_WRITE_ROWS_TID,
        query_rows: ktir_superdsc::reserved_tids::PAGED_QUERY_ROWS_TID,
        kv_row_pool: ktir_superdsc::reserved_tids::PAGED_KV_ROW_POOL_TID,
        rep_rows: ktir_superdsc::reserved_tids::PAGED_REP_ROWS_TID,
        chunk_pages: ktir_superdsc::reserved_tids::PAGED_CHUNK_PAGES_TID,
        chunk_masks: ktir_superdsc::reserved_tids::PAGED_CHUNK_MASKS_TID,
        int32_zero: ktir_superdsc::reserved_tids::PAGED_INT32_ZERO_TID,
    };
}

/// THE ATTENTION FAMILY's prefill kernel — `page_attn_head_major_prefill.py`, spyre-inference's
/// `page_attn_head_major_prefill_kernel`. [`lower_attn`] dispatches as spyre-inference does: this
/// kernel past one query row, [`attn_decode_row`]'s at one.
pub fn attn_row() -> TritonKernelRow {
    TritonKernelRow::Spliced {
        src: include_str!("../../../targets/spyre/kernels/page_attn_head_major_prefill.py"),
        file: "page_attn_head_major_prefill.py",
        entry: "page_attn_head_major_prefill_kernel",
        program: Program::PagedAttn,
    }
}

/// THE ATTENTION FAMILY's decode kernel — `page_attn_head_major_decode.py`, spyre-inference's
/// `page_attn_head_major_decode_kernel`, at one query row.
pub fn attn_decode_row() -> TritonKernelRow {
    TritonKernelRow::Spliced {
        src: include_str!("../../../targets/spyre/kernels/page_attn_head_major_decode.py"),
        file: "page_attn_head_major_decode.py",
        entry: "page_attn_head_major_decode_kernel",
        program: Program::PagedAttn,
    }
}

/// THE ATTENTION FAMILY's batched decode kernel — `batched_decode_head_major.py`, spyre-inference's
/// `batched_decode_head_major_kernel`, for a decode rung whose rows are separate sequences.
pub fn batched_decode_row() -> TritonKernelRow {
    TritonKernelRow::Spliced {
        src: include_str!("../../../targets/spyre/kernels/batched_decode_head_major.py"),
        file: "batched_decode_head_major.py",
        entry: "batched_decode_head_major_kernel",
        program: Program::PagedAttn,
    }
}

/// The paged KV write's row — `reshape_and_cache_head_major.py`, spyre-inference's
/// `reshape_and_cache_head_major_kernel`: the scatter half of a KV writer (`RopeAppend`), its own
/// program.
pub fn kv_write_row() -> TritonKernelRow {
    TritonKernelRow::Spliced {
        src: include_str!("../../../targets/spyre/kernels/reshape_and_cache_head_major.py"),
        file: "reshape_and_cache_head_major.py",
        entry: "reshape_and_cache_head_major_kernel",
        program: Program::PagedKvWrite,
    }
}

/// The pool a cache tensor is: `pages` of `PAGE_SLOTS` slots at `kv_width = nkvh · hd` per slot.
/// Refused unless the cache is exactly that — a whole number of pages of the stated width.
fn pool_pages<F: scratchy_subtile::subtile_ir::RopeForm>(
    ir: &SubtileIR<F>,
    cache: &scratchy_subtile::subtile_ir::TensorRegion,
    kv_width: u32,
    what: &str,
) -> Result<u32, String> {
    let t = &ir.tensors[cache.tensor.index()];
    if t.cols != kv_width || t.rows == 0 || !t.rows.is_multiple_of(PAGE_SLOTS) {
        return Err(format!(
            "triton splice: the {what} cache t{} is [{}, {}], which is not a whole number of \
             {PAGE_SLOTS}-slot pages at {kv_width} columns — the pool it is read as is \
             [pages, nkvh, {PAGE_SLOTS}, hd] over exactly those bytes",
            cache.tensor.index(),
            t.rows,
            t.cols
        ));
    }
    Ok(t.rows / PAGE_SLOTS)
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

/// ⭐ THE WALK — how many blocks of the page table a rung's attention reads (`NUM_BLOCKS`): the rung's
/// swept extent in pages. Every page of the pool at `ActiveCap::FULL` (a decode step's keys can sit
/// anywhere below the capacity), and at `ActiveCap::NONE` (a prompt chunk baked with no resident
/// prefix) the pages its own `mq` rows fill.
///
/// Blocks past a sequence's length carry a fully-masked tile, so a walk wider than the context costs
/// time, never correctness; a walk NARROWER than the context drops keys. So the walk is also the reach
/// of the rung: a step whose positions end at or below `walk · PAGE_SLOTS` can run on it, whatever its
/// start — the mask stack, not the rung, says which of the walked keys each row attends. The splice
/// bakes it and the wiring states it (`PagedStep::walk`) from this one function.
pub fn walk_pages(pages: u32, mq: u32, active_cap: ktir_superdsc::ktir_node::ActiveCap) -> u32 {
    let swept = active_cap.resolve(pages * PAGE_SLOTS, ktir_superdsc::sdsc_abstract::POOL_STICK);
    let walked = if swept == 0 { mq } else { swept };
    walked.div_ceil(PAGE_SLOTS).min(pages)
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

/// The KV cache's allocation in a one-kernel bundle — see
/// [`triton_ktir_superdsc::layout::allocate_paged_cache`], where the one-kernel layout lives.
pub use triton_ktir_superdsc::layout::allocate_paged_cache;

/// THE ATTENTION SPLICE — compile spyre-inference's kernel for this node's rows (one sequence's:
/// the decode kernel at one row, the prefill kernel past it; a decode batch's: the batched decode
/// kernel) at its geometry and the rung's walk ([`walk_pages`]), and mint the `Program::PagedAttn`
/// node over their operands, in their order: `[query, query_row_index, k_pages, v_pages,
/// page_index_table, (kv_row_pool,) mask_stack, out]`, or the batch's `[query, rep_row_ids, k_pages,
/// v_pages, chunk_page_ids, mask_by_chunk, out]`.
pub fn lower_attn<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    active_cap: ktir_superdsc::ktir_node::ActiveCap,
    paged_rows: PagedRows,
) -> Result<EmittedOp, String> {
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
    let SubOp::AttnDecode {
        geom,
        scale,
        producer,
        ..
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
    let (nqh, nkvh, hd) = (geom.nqh().get(), geom.nkvh().get(), geom.hd().get());
    let mq = node.output.region.rows.len;
    let kv_width = nkvh * hd;
    let pages = pool_pages(ir, &node.inputs[1], kv_width, "K")?;
    if pool_pages(ir, &node.inputs[2], kv_width, "V")? != pages {
        return Err(format!(
            "triton splice: AttnDecode t{}: the K and V pools differ in pages",
            node.output.tensor.index()
        ));
    }
    whole(ir, &node.inputs[0], mq, nqh * hd, "query")?;
    whole(ir, &node.output, mq, nqh * hd, "attention output")?;
    let blocks = walk_pages(pages, mq, active_cap);
    let tids = PagedStepTids::RESERVED;
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    let t =
        |tr: &scratchy_subtile::subtile_ir::TensorRegion| BufferId::new(tr.tensor.index() as u32);
    if batch {
        // spyre-inference's eligibility (`_batched_decode_preconditions_met`): a chunk gathering as
        // many pages as the pool holds faults their device, and their per-sequence loop serves it.
        let chunking = BatchChunking::of(mq, blocks);
        if chunking.entries() >= pages {
            return Err(format!(
                "triton splice: AttnDecode t{}: a {mq}-sequence batch walking {blocks} block(s) \
                 gathers {} pages a chunk, and the pool holds {pages} — spyre-inference serves \
                 that batch one sequence at a time",
                node.output.tensor.index(),
                chunking.entries()
            ));
        }
        for (p, ty) in [
            ("query", "*fp16"),
            ("rep_row_ids", "*i32"),
            ("k_pages", "*fp16"),
            ("v_pages", "*fp16"),
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
            ("BLOCK_SIZE", PAGE_SLOTS),
            ("HEAD_SIZE", hd),
            ("NUM_PAGES", pages),
        ] {
            signature.insert(k.to_string(), ArgSpec::Constexpr);
            constexprs.insert(k.to_string(), Val::Int(i128::from(v)));
        }
        signature.insert("SCALE".to_string(), ArgSpec::Constexpr);
        constexprs.insert("SCALE".to_string(), Val::Float(f64::from(*scale)));
        let spec = KernelSpec {
            kernel: entry.to_string(),
            signature,
            constexprs,
            file: file.to_string(),
        };
        let module = compile_kernel(src, &spec, &[1])?;
        let positional = triton_ktir_superdsc::node_for(&module, program)
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
            Some(BufferId::new(tids.chunk_masks)),
        )?;
        let mut e = EmittedOp::bare(format!("attn_s{}", node.id.index()));
        e.ktir = Some(k);
        return Ok(e);
    }
    let mut pointers = vec![
        ("query", "*fp16"),
        ("query_row_index", "*i32"),
        ("k_pages", "*fp16"),
        ("v_pages", "*fp16"),
        ("page_index_table", "*i32"),
    ];
    if decode {
        pointers.push(("kv_row_pool", "*i32"));
    }
    pointers.extend([("mask_stack", "*fp16"), ("out", "*fp16")]);
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
        ("HEAD_SIZE", hd),
        ("BLOCK_SIZE", PAGE_SLOTS),
        ("NUM_PAGES", pages),
        ("NUM_BLOCKS", blocks),
    ];
    if !decode {
        sizes.push(("PADDED_QUERY_LEN", mq));
    }
    for (k, v) in sizes {
        signature.insert(k.to_string(), ArgSpec::Constexpr);
        constexprs.insert(k.to_string(), Val::Int(i128::from(v)));
    }
    signature.insert("SCALE".to_string(), ArgSpec::Constexpr);
    constexprs.insert("SCALE".to_string(), Val::Float(f64::from(*scale)));
    let spec = KernelSpec {
        kernel: entry.to_string(),
        signature,
        constexprs,
        file: file.to_string(),
    };
    let module = compile_kernel(src, &spec, &[1])?;
    let positional = triton_ktir_superdsc::node_for(&module, program)
        .map_err(|e| format!("triton splice: node_for: {e}"))?;
    let mut bindings = vec![
        t(&node.inputs[0]),
        BufferId::new(tids.query_rows),
        t(&node.inputs[1]),
        t(&node.inputs[2]),
        BufferId::new(tids.page_table),
    ];
    if decode {
        bindings.push(BufferId::new(tids.kv_row_pool));
    }
    bindings.extend([BufferId::new(tids.mask), t(&node.output)]);
    let k = rebind(
        positional,
        format!("attn_s{}", node.id.index()),
        bindings,
        Some(BufferId::new(tids.mask)),
    )?;
    let mut e = EmittedOp::bare(format!("attn_s{}", node.id.index()));
    e.ktir = Some(k);
    Ok(e)
}

/// THE KV WRITE'S SPLICE — the scatter half of a `RopeAppend`: compile spyre-inference's
/// `reshape_and_cache_head_major_kernel` for the node's rows and pools, and mint the
/// `Program::PagedKvWrite` node over their operands: `[key, value, k_rows, v_rows, row_index]`, where
/// `key` is the RopeAppend's own output (the rotated K) and `value` its V input.
pub fn lower_kv_write<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
) -> Result<EmittedOp, String> {
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
    let SubOp::RopeAppend { head_dim, .. } = &node.op else {
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
    let hd = head_dim.get();
    let (rows, kv_width) = (node.output.region.rows.len, node.output.region.cols.len);
    if hd == 0 || !kv_width.is_multiple_of(hd) {
        return Err(format!(
            "triton splice: RopeAppend t{}: its {kv_width}-column K is not whole heads of {hd}",
            node.output.tensor.index()
        ));
    }
    let nkvh = kv_width / hd;
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
        ("k_rows", "*fp16"),
        ("v_rows", "*fp16"),
        ("row_index", "*i32"),
    ] {
        signature.insert(
            p.to_string(),
            ArgSpec::parse(ty).map_err(|e| e.to_string())?,
        );
    }
    for (k, v) in [
        ("T", rows),
        ("NUM_KV_HEADS", nkvh),
        ("HEAD_SIZE", hd),
        ("NUM_ROWS", pages * nkvh * PAGE_SLOTS),
    ] {
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
    let positional = triton_ktir_superdsc::node_for(&module, program)
        .map_err(|e| format!("triton splice: node_for: {e}"))?;
    let t =
        |tr: &scratchy_subtile::subtile_ir::TensorRegion| BufferId::new(tr.tensor.index() as u32);
    let bindings = vec![
        t(&node.output),
        t(&node.inputs[3]),
        t(&node.inputs[4]),
        t(&node.inputs[5]),
        BufferId::new(PagedStepTids::RESERVED.write_rows),
    ];
    let name = format!("kvwrite_s{}", node.id.index());
    let k = rebind(positional, name.clone(), bindings, None)?;
    let mut e = EmittedOp::bare(name);
    e.ktir = Some(k);
    Ok(e)
}

/// A compiled program re-stated with this node's own name and bindings (`node_for` mints positional
/// buffer ids), refused if the kernel's parameter list and the bindings disagree in length.
fn rebind(
    positional: KtirNode,
    name: String,
    bindings: Vec<BufferId>,
    mask: Option<BufferId>,
) -> Result<KtirNode, String> {
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
        mask,
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
            // kernel parameter's dtype: the fp16 row, the PER-ROW SEED WORD
            // TABLES in i32 — four `[rows, 1]` tables, one per word of the
            // quadruple (seed_lo, seed_hi, step, greedy_gate), each
            // loaded corner-free at `[pid, 0]` (the Col operand mode reads
            // the stick head alone — the int-bcast leg's measured law — so
            // no word rides a windowed column; the auditor's wall-4 ruling
            // (a); the row word is gone from the key — the P3 localization
            // ruling, request-owned noise alone); the noise is computed on
            // the card, no f32 noise staging
            // rides — then the per-row temperature reciprocals in f32,
            // the nine selection tables in f32 (the selection is f32 — a sum
            // the f16 grid would round), the noise section's own tables — the
            // hash multipliers and the wrap-fold's 2^8 in i32, the lane and the
            // fold's five f32 constants, the FEST-domain clamp in fp16 — and
            // the i32 output stick (the index-ingestion law).
            let pad_v = scratchy_subtile::subtile_ir::sample_pad_vocab(*vocab);
            let nb = pad_v / scratchy_subtile::subtile_ir::SAMPLE_BLOCK;
            signature.insert(
                "logits_ptr".to_string(),
                ArgSpec::parse("*fp16").map_err(|e| e.to_string())?,
            );
            for p in ["slo_ptr", "shi_ptr", "stp_ptr", "sgate_ptr"] {
                signature.insert(
                    p.to_string(),
                    ArgSpec::parse("*i32").map_err(|e| e.to_string())?,
                );
            }
            for p in [
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
            for p in ["hc1_ptr", "hc2_ptr", "hc3_ptr", "c256_ptr"] {
                signature.insert(
                    p.to_string(),
                    ArgSpec::parse("*i32").map_err(|e| e.to_string())?,
                );
            }
            for p in [
                "lane_ptr",
                "fm8_ptr",
                "ftop_ptr",
                "fzero_ptr",
                "fhalf_ptr",
                "fm22_ptr",
            ] {
                signature.insert(
                    p.to_string(),
                    ArgSpec::parse("*fp32").map_err(|e| e.to_string())?,
                );
            }
            signature.insert(
                "cmin_ptr".to_string(),
                ArgSpec::parse("*fp16").map_err(|e| e.to_string())?,
            );
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
    // ⭐ THE CHAINED HEAD'S PAIR — HAND-MINTED KTIR, NOT SPLICED, but its
    // operand formats are still THIS census's to state, because the placement
    // widths read no other seat: the door's own u32-stick law (a staged
    // table's gathered destination is exactly one `[1, 32]` SENUINT32 stick,
    // and the pick's `[1, 64]` i32 lanes are the same bytes at the argmax's
    // own SenUint32 statement) sizes those tensors at FOUR bytes an element.
    // Left to the fp16 default the placement under-reserves by half and the
    // footprint guard refuses the CORRECT descriptor — the exact defect this
    // census exists to prevent. The formats are the DOOR's law
    // (`lk::embed_row_gather`'s idx-stick class check, `lk::chain_stage`'s
    // destination class check), stated here because the emitter that mints
    // the programs has no kernel signature to read them from.
    if matches!(
        node.op,
        SubOp::EmbedRowGather { .. } | SubOp::ChainStage { .. }
    ) {
        let bindings = default_bindings(node);
        let dfs = match &node.op {
            // `[s_stick, l_stick, weight, out]`: the two index sticks are
            // SENUINT32; the weight and the `[1, hidden]` row are fp16. ⭐ The
            // list is the COMPLETE operand census in `default_bindings` order,
            // the output LAST — the ktir consumers (`node_output_on_f16_stick`,
            // `result_dtype_of_ir`) read it as the node's full parameter list
            // and refuse a node whose own output is absent (the check-8
            // pre-scan catch: the u32-only partial list panicked the reference
            // legs' stick walk at the gather). The Fp16 entries are inert for
            // the placement census by its own skip law (`df != Df::Fp16`), so
            // completion changes no sizing.
            SubOp::EmbedRowGather { .. } if bindings.len() == 4 => vec![
                (bindings[0].get(), Df::SenUint32),
                (bindings[1].get(), Df::SenUint32),
                (bindings[2].get(), Df::Fp16),
                (bindings[3].get(), Df::Fp16),
            ],
            // `[pick, s_stick, l_stick, s_tab, l_tab, out]`: every operand is
            // a u32 stick or a u32 table — all SENUINT32.
            SubOp::ChainStage { .. } if bindings.len() == 6 => {
                bindings.iter().map(|b| (b.get(), Df::SenUint32)).collect()
            }
            // A chain node off the mint's arity — the operand-order contract
            // this function's own zip law states below.
            _ => {
                return Err(format!(
                    "triton splice: a chained-head node states {} operand(s) — the mint's \
                     shape is exactly four for the gather and six for the stage (see \
                     `mint_embed_chain`)",
                    bindings.len()
                ));
            }
        };
        return Ok(Some(dfs));
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
    //! Every spec below mirrors the `kernel_spec` arm (or `lower_attn` / `lower_kv_write` /
    //! `lower_sandwich_node` / `lower_prefill_lm_head_fold` / the MoE numeric tests'
    //! golden geometry) that the production bake states for that kernel — cited per entry
    //! — at a whole-region decode-shaped geometry so every blocking law answers the
    //! one-tile constants. The compile goes through the crate's own [`compile_kernel`]:
    //! the exact memoized ladder (`codegen::compile` → `make_ttir` → `from_ttir` →
    //! `make_ktir` → `to_ktir`) the bake runs, so the dump cannot drift from what
    //! production emits.
    use super::*;

    const PAGED_DECODE: &str =
        include_str!("../../../targets/spyre/kernels/page_attn_head_major_decode.py");
    const PAGED_PREFILL: &str =
        include_str!("../../../targets/spyre/kernels/page_attn_head_major_prefill.py");
    const PAGED_KV_WRITE: &str =
        include_str!("../../../targets/spyre/kernels/reshape_and_cache_head_major.py");
    const BATCHED_DECODE: &str =
        include_str!("../../../targets/spyre/kernels/batched_decode_head_major.py");
    const ELEMENTWISE: &str = include_str!("../../../targets/spyre/kernels/elementwise.py");
    const LMLAST: &str = include_str!("../../../targets/spyre/kernels/lmlast.py");
    const MATMUL: &str = include_str!("../../../targets/spyre/kernels/matmul.py");
    const MATMUL_FP8: &str = include_str!("../../../targets/spyre/kernels/matmul_fp8.py");
    const MOE: &str = include_str!("../../../targets/spyre/kernels/moe.py");
    const MOE_PREFILL: &str = include_str!("../../../targets/spyre/kernels/moe_prefill.py");
    const RESHAPE: &str = include_str!("../../../targets/spyre/kernels/reshape.py");
    const SAMPLE: &str = include_str!("../../../targets/spyre/kernels/sample_fwd.py");
    const RMSNORM: &str = include_str!("../../../targets/spyre/kernels/rmsnorm.py");
    const RMSNORMUNIT: &str = include_str!("../../../targets/spyre/kernels/rmsnormunit.py");
    const ROPE: &str = include_str!("../../../targets/spyre/kernels/rope.py");
    const SCALARMUL: &str = include_str!("../../../targets/spyre/kernels/scalarmul.py");
    const SCALARWMUL: &str = include_str!("../../../targets/spyre/kernels/scalarwmul.py");
    const SILUMUL: &str = include_str!("../../../targets/spyre/kernels/silumul.py");
    const TANHSOFTCAP: &str = include_str!("../../../targets/spyre/kernels/tanhsoftcap.py");
    const WINDOWEDRMSNORM: &str = include_str!("../../../targets/spyre/kernels/windowedrmsnorm.py");

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

    use ktir_superdsc::superdsc_opspec::{
        EW_BINARY_LIVE_TILES, EW_SILU_LIVE_TILES, SILU_MUL_LIVE_TILES,
    };

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
        let n_dev = ktir_superdsc::work::DeviceWidth::for_matmul(1, 2048, 2048, false).get();
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
                        Val::Int(scratchy_subtile::sdsc_abstract::selector_lastrow_col(5) as i128),
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
        for (name, mq, blocks, pages, hd) in [
            ("paged_attn_decode", 1i128, 4i128, 4i128, 64i128),
            ("paged_attn_prefill", 64, 1, 4, 64),
            ("paged_attn_decode_pool", 1, 2048, 2048, 64),
            ("paged_attn_prefill_walk", 64, 4, 2048, 64),
            // granite-3.1-8b's head dim: a kv-head slab is two sticks wide.
            ("paged_attn_decode_hd128", 1, 4, 4, 128),
            ("paged_attn_prefill_hd128", 64, 4, 4, 128),
        ] {
            let decode = mq == 1;
            let mut ptrs: Vec<(&str, &str)> = vec![
                ("query", "*fp16"),
                ("query_row_index", "*i32"),
                ("k_pages", "*fp16"),
                ("v_pages", "*fp16"),
                ("page_index_table", "*i32"),
            ];
            if decode {
                ptrs.push(("kv_row_pool", "*i32"));
            }
            ptrs.extend([("mask_stack", "*fp16"), ("out", "*fp16")]);
            let mut ces = vec![
                ("NUM_TOKENS", Val::Int(mq)),
                ("NUM_HEADS", Val::Int(32)),
                ("NUM_KV_HEADS", Val::Int(8)),
                ("HEAD_SIZE", Val::Int(hd)),
                ("BLOCK_SIZE", Val::Int(i128::from(PAGE_SLOTS))),
                ("NUM_PAGES", Val::Int(pages)),
                ("NUM_BLOCKS", Val::Int(blocks)),
                ("SCALE", Val::Float(0.125)),
            ];
            let (src, file, entry) = if decode {
                (
                    PAGED_DECODE,
                    "page_attn_head_major_decode.py",
                    "page_attn_head_major_decode_kernel",
                )
            } else {
                ces.push(("PADDED_QUERY_LEN", Val::Int(mq)));
                (
                    PAGED_PREFILL,
                    "page_attn_head_major_prefill.py",
                    "page_attn_head_major_prefill_kernel",
                )
            };
            v.push((name, src, spec(file, entry, &ptrs, &ces), vec![1]));
        }
        // ...and `batched_decode_head_major` over that pool, the chunkings a decode batch rung
        // states (`BatchChunking::of`): 4 sequences on the 4-page rung (one chunk), on the 64-page
        // rung (8 chunks, each block slot its own softmax), 2 sequences on the 2-page rung (a
        // chunk under the gather floor), and 32 on the 4-page rung (4 chunks of ONE block slot —
        // the widest batch rung, where the slots' merge folds one). At head dim 128 (granite-3.1-8b)
        // the one-chunk and many-chunk batches, and the widest rung on its one-page walk.
        for (name, seqs, walk, hd) in [
            ("batched_decode", 4u32, 4u32, 64i128),
            ("batched_decode_chunks", 4, 64, 64),
            ("batched_decode_floor", 2, 2, 64),
            ("batched_decode_one_slot", 32, 4, 64),
            ("batched_decode_hd128", 4, 4, 128),
            ("batched_decode_chunks_hd128", 4, 64, 128),
            ("batched_decode_one_slot_hd128", 32, 1, 128),
        ] {
            let c = BatchChunking::of(seqs, walk);
            v.push((
                name,
                BATCHED_DECODE,
                spec(
                    "batched_decode_head_major.py",
                    "batched_decode_head_major_kernel",
                    &[
                        ("query", "*fp16"),
                        ("rep_row_ids", "*i32"),
                        ("k_pages", "*fp16"),
                        ("v_pages", "*fp16"),
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
                        ("BLOCK_SIZE", Val::Int(i128::from(PAGE_SLOTS))),
                        ("HEAD_SIZE", Val::Int(hd)),
                        ("NUM_PAGES", Val::Int(2048)),
                        ("SCALE", Val::Float(0.125)),
                    ],
                ),
                vec![1],
            ));
        }
        // A decode step's write (8 kv heads, one store per head), one below the scatter floor (2
        // heads, one store per row), a 20-token prompt chunk's (160 index entries, five scatter ops
        // of 32), and at head dim 128 the first two and a prompt chunk's.
        for (name, t, nkvh, hd) in [
            ("paged_kv_write", 1i128, 8i128, 64i128),
            ("paged_kv_write_rows", 1, 2, 64),
            ("paged_kv_write_prompt", 20, 8, 64),
            ("paged_kv_write_hd128", 1, 8, 128),
            ("paged_kv_write_rows_hd128", 1, 2, 128),
            ("paged_kv_write_prompt_hd128", 96, 8, 128),
        ] {
            v.push((
                name,
                PAGED_KV_WRITE,
                spec(
                    "reshape_and_cache_head_major.py",
                    "reshape_and_cache_head_major_kernel",
                    &[
                        ("key", "*fp16"),
                        ("value", "*fp16"),
                        ("k_rows", "*fp16"),
                        ("v_rows", "*fp16"),
                        ("row_index", "*i32"),
                    ],
                    &[
                        ("T", Val::Int(t)),
                        ("NUM_KV_HEADS", Val::Int(nkvh)),
                        ("HEAD_SIZE", Val::Int(hd)),
                        ("NUM_ROWS", Val::Int(4 * nkvh * i128::from(PAGE_SLOTS))),
                    ],
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
        // ── sample_fwd.py: the token draw at solo and batched grids — the sampler this
        // proof guards, and the ONE kernel the grid-[B] fix is allowed to move. The
        // spec mirrors the `SubOp::Sample` arm of `kernel_spec`'s registry row: the
        // logits row, its per-row seed quadruple's four word tables and temperature
        // reciprocal, the twenty staged tables (the nine of the selection cascade and
        // the noise section's own eleven), and the `[B, 64]` idx stick out. ──
        {
            let sample_ptrs: &[(&str, &str)] = &[
                ("logits_ptr", "*fp16"),
                ("slo_ptr", "*i32"),
                ("shi_ptr", "*i32"),
                ("stp_ptr", "*i32"),
                ("sgate_ptr", "*i32"),
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
                ("hc1_ptr", "*i32"),
                ("hc2_ptr", "*i32"),
                ("hc3_ptr", "*i32"),
                ("c256_ptr", "*i32"),
                ("lane_ptr", "*fp32"),
                ("fm8_ptr", "*fp32"),
                ("ftop_ptr", "*fp32"),
                ("fzero_ptr", "*fp32"),
                ("fhalf_ptr", "*fp32"),
                ("fm22_ptr", "*fp32"),
                ("cmin_ptr", "*fp16"),
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
            "paged_attn_decode"
            | "paged_attn_prefill"
            | "paged_attn_decode_pool"
            | "paged_attn_prefill_walk"
            | "paged_attn_decode_hd128"
            | "paged_attn_prefill_hd128"
            | "batched_decode"
            | "batched_decode_chunks"
            | "batched_decode_floor"
            | "batched_decode_one_slot"
            | "batched_decode_hd128"
            | "batched_decode_chunks_hd128"
            | "batched_decode_one_slot_hd128" => Program::PagedAttn,
            "paged_kv_write"
            | "paged_kv_write_rows"
            | "paged_kv_write_prompt"
            | "paged_kv_write_hd128"
            | "paged_kv_write_rows_hd128"
            | "paged_kv_write_prompt_hd128" => Program::PagedKvWrite,
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
            // The geometry const door: `rope_at::<HD>` monomorphises on head geometry the
            // driver would have to state from the fixture's constexprs (`bake_py`'s rope
            // arm). The KTIR-text leg still covers it.
            "rope_fwd" => Some("the model-geometry const door (rope_at::<HD>)"),
            // The paged step's card lowering is not ported: the door refuses both programs by
            // name. The KTIR-text leg covers them.
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
        obj.insert(
            "arg_bindings".into(),
            serde_json::json!(format!("{:?}", op.arg_bindings)),
        );
        obj.insert("time".into(), serde_json::json!(op.time));
        obj.insert(
            "affine_strides".into(),
            serde_json::json!(format!("{:?}", op.affine_strides)),
        );
        obj.insert(
            "slot_stride_bytes".into(),
            serde_json::json!(op.slot_stride_bytes),
        );
        obj.insert("slab_write".into(), serde_json::json!(op.slab_write));
        obj.insert(
            "slab_stride_bytes".into(),
            serde_json::json!(op.slab_stride_bytes),
        );
        obj.insert("walk".into(), serde_json::json!(format!("{:?}", op.walk)));
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
        // A WALKING program's windows are its trip-0 windows (`loop_fold::flatten`), the reading
        // the walk's own lowering takes.
        let flat = ktir_superdsc::emit::loop_fold::flatten(node)
            .map_err(|e| format!("{name}: flatten: {e:?}"))?;
        let at_trip0 = flat.as_ref().map_or(node, |f| &f.node);
        let regions = lk::regions(at_trip0).map_err(|e| format!("{name}: regions: {e:?}"))?;
        let layout = triton_ktir_superdsc::layout::for_regions(node, &regions)
            .map_err(|e| format!("{name}: layout: {e}"))?;
        if matches!(program_of(name), Program::PagedAttn | Program::PagedKvWrite) {
            allocate_paged_cache(node, &layout)?;
        }
        let windowed = |ops: Result<Vec<ktir_superdsc::emit::EmittedOp>, lk::Error>| {
            let ops = ops.map_err(|e| format!("{name}: {e:?}"))?;
            lk::column_windows_are_honoured(lk_name, &regions, &ops)
                .map_err(|e| format!("{name}: {e:?}"))?;
            Ok::<_, String>(ops)
        };
        let mut sym = 0i64;
        let ops = match program_of(name) {
            // The door's walk, whatever the program computes: the body at trip 0, marked for the
            // card's relaunch.
            _ if flat.is_some() => {
                Ok(
                    ktir_superdsc::emit::whole_function::lower_walk(node, Some(&layout), &mut sym)
                        .map_err(|e| format!("{name}: {e:?}"))?,
                )
            }
            // The door's `windowed(...)`-wrapped whole-function kinds.
            Program::Elementwise(_)
            | Program::SiluMul
            | Program::RmsNorm
            | Program::ScalarMul
            | Program::Sample
            | Program::PagedAttn
            | Program::PagedKvWrite => {
                windowed(lower_function(node, Some(&layout), &mut sym, None))
            }
            // The door's unwrapped whole-function kinds.
            Program::Matmul | Program::LmLast => {
                Ok(lower_function(node, Some(&layout), &mut sym, None)
                    .map_err(|e| format!("{name}: {e:?}"))?)
            }
            Program::RmsNormUnit => windowed(lk::rmsnorm_unit(
                lk_name,
                node,
                &regions,
                &mut sym,
                Some(&layout),
            )),
            Program::TanhSoftCap => windowed(lk::tanhsoftcap(
                lk_name,
                node,
                &regions,
                &mut sym,
                Some(&layout),
            )),
            Program::ScalarWeightMul => windowed(lk::scalarweightmul(
                lk_name,
                &regions,
                &mut sym,
                Some(&layout),
            )),
            Program::Reshape => Ok(lk::reshape(lk_name, &regions, &mut sym, Some(&layout))
                .map_err(|e| format!("{name}: {e:?}"))?),
            Program::WindowedRmsNorm { .. } => {
                Ok(
                    lk::windowed_rmsnorm(lk_name, node, &regions, &mut sym, Some(&layout))
                        .map_err(|e| format!("{name}: {e:?}"))?,
                )
            }
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
            eprintln!(
                "dumped {name} -> {} ({} lines)",
                path.display(),
                text.lines().count()
            );

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

/// ⭐ THE PAGE WALK ON THE CARD — spyre-inference's page walk at granite-3.1-2b's own pool (2048 pages of
/// [`PAGE_SLOTS`] slots, 32 query heads over 8 kv heads at head dim 64), through the card's walk lowering
/// (`whole_function::lower_walk`): the loop becomes ONE body built at trip 0 that the card relaunches
/// once per trip, and every fact of that relaunch is read off the program.
#[cfg(test)]
mod walk_on_the_card {
    use super::*;
    use ktir_superdsc::emit::loop_fold::{WalkTrips, flatten};

    fn walk(mq: i128, blocks: i128) -> KtirNode {
        walk_at(mq, blocks, 64)
    }

    /// [`walk`] at head dim `hd`.
    fn walk_at(mq: i128, blocks: i128, hd: i128) -> KtirNode {
        let decode = mq == 1;
        let row = if decode {
            attn_decode_row()
        } else {
            attn_row()
        };
        let TritonKernelRow::Spliced {
            src, file, entry, ..
        } = row
        else {
            panic!("the attention rows are spliced");
        };
        let mut signature = HashMap::new();
        let mut constexprs = HashMap::new();
        for p in ["query", "k_pages", "v_pages", "mask_stack", "out"] {
            signature.insert(p.to_string(), ArgSpec::parse("*fp16").expect("spec"));
        }
        let mut ints = vec!["query_row_index", "page_index_table"];
        if decode {
            ints.push("kv_row_pool");
        }
        for p in ints {
            signature.insert(p.to_string(), ArgSpec::parse("*i32").expect("spec"));
        }
        let mut sizes = vec![
            ("NUM_TOKENS", mq),
            ("NUM_HEADS", 32),
            ("NUM_KV_HEADS", 8),
            ("HEAD_SIZE", hd),
            ("BLOCK_SIZE", i128::from(PAGE_SLOTS)),
            ("NUM_PAGES", 2048),
            ("NUM_BLOCKS", blocks),
        ];
        if !decode {
            sizes.push(("PADDED_QUERY_LEN", mq));
        }
        for (k, v) in sizes {
            signature.insert(k.to_string(), ArgSpec::Constexpr);
            constexprs.insert(k.to_string(), Val::Int(v));
        }
        signature.insert("SCALE".to_string(), ArgSpec::Constexpr);
        constexprs.insert("SCALE".to_string(), Val::Float(0.125));
        let spec = KernelSpec {
            kernel: entry.to_string(),
            signature,
            constexprs,
            file: file.to_string(),
        };
        let module = compile_kernel(src, &spec, &[1]).expect("the walk compiles");
        triton_ktir_superdsc::node_for(&module, Program::PagedAttn).expect("the walk mints")
    }

    /// `batched_decode_head_major` at granite's geometry over its 2048-page pool: `seqs` sequences
    /// walking `walk` blocks, at the chunking a decode batch rung states.
    fn batched_walk(seqs: u32, walk: u32) -> (KtirNode, BatchChunking) {
        batched_walk_at(seqs, walk, 64)
    }

    /// [`batched_walk`] at head dim `hd`.
    fn batched_walk_at(seqs: u32, walk: u32, hd: u32) -> (KtirNode, BatchChunking) {
        let TritonKernelRow::Spliced {
            src, file, entry, ..
        } = batched_decode_row()
        else {
            panic!("the batched decode row is spliced");
        };
        let c = BatchChunking::of(seqs, walk);
        let mut signature = HashMap::new();
        let mut constexprs = HashMap::new();
        for p in ["query", "k_pages", "v_pages", "mask_by_chunk", "out"] {
            signature.insert(p.to_string(), ArgSpec::parse("*fp16").expect("spec"));
        }
        for p in ["rep_row_ids", "chunk_page_ids"] {
            signature.insert(p.to_string(), ArgSpec::parse("*i32").expect("spec"));
        }
        for (k, v) in [
            ("NUM_TOKENS", seqs),
            ("NUM_SEQS", c.seqs),
            ("BLOCKS_PER_CHUNK", c.blocks_per_chunk),
            ("NUM_CHUNKS", c.num_chunks),
            ("NUM_KV_HEADS", 8),
            ("NUM_QUERIES_PER_KV", 4),
            ("BLOCK_SIZE", PAGE_SLOTS),
            ("HEAD_SIZE", hd),
            ("NUM_PAGES", 2048),
        ] {
            signature.insert(k.to_string(), ArgSpec::Constexpr);
            constexprs.insert(k.to_string(), Val::Int(i128::from(v)));
        }
        signature.insert("SCALE".to_string(), ArgSpec::Constexpr);
        constexprs.insert("SCALE".to_string(), Val::Float(0.125));
        let spec = KernelSpec {
            kernel: entry.to_string(),
            signature,
            constexprs,
            file: file.to_string(),
        };
        let module = compile_kernel(src, &spec, &[1]).expect("the batched walk compiles");
        let node =
            triton_ktir_superdsc::node_for(&module, Program::PagedAttn).expect("the walk mints");
        (node, c)
    }

    /// A decode batch's walk relaunches its chunk body once per chunk: its trips are the
    /// chunks, and ONE trip moves the chunk's mask tile — `entries · kv heads` one-stick rows of the
    /// stick-major stack — which is the one shift a relaunch applies (the chunk's page ids pitched
    /// to it). The body is one contiguous run of descriptors, one launch group.
    #[test]
    fn a_decode_batch_relaunches_its_chunk_body_once_per_chunk() {
        let (node, c) = batched_walk(4, 64);
        assert_eq!(
            (c.blocks_per_chunk, c.num_chunks),
            (8, 8),
            "4 sequences over 64 blocks"
        );
        let flat = flatten(&node)
            .expect("the walk flattens")
            .expect("a multi-chunk walk states its loop");
        assert_eq!(flat.fold.trips, c.num_chunks);
        assert_eq!(flat.fold.step_bytes, c.entries() * 8 * 128);
        assert_eq!(
            flat.fold.carried.len(),
            3,
            "each block slot's running max, sum and output"
        );
        let layout = triton_ktir_superdsc::layout::for_node(&node).expect("the walk places");
        allocate_paged_cache(&node, &layout).expect("the cache allocates");
        let mut sym = 0i64;
        let ops = ktir_superdsc::emit::whole_function::lower_walk(&node, Some(&layout), &mut sym)
            .unwrap_or_else(|e| panic!("the batched walk lowers: {}", e.message));
        let marked: Vec<usize> = (0..ops.len()).filter(|&i| ops[i].walk.is_some()).collect();
        let (first, last) = (marked[0], *marked.last().expect("a body"));
        assert_eq!(
            marked.len(),
            last - first + 1,
            "the body's descriptors are one contiguous run — one launch group"
        );
        for e in &ops[first..=last] {
            assert_eq!(
                e.walk,
                Some(WalkTrips {
                    trips: c.num_chunks,
                    step_bytes: c.entries() * 8 * 128
                }),
                "{}",
                e.op_name
            );
            assert!(e.kv_page_fold && e.kv_gathered, "{}", e.op_name);
        }
    }

    /// ⭐ A DECODE BATCH LOWERS AT EVERY RUNG SHAPE THE BAKE ASKS FOR — two sequences to thirty-two,
    /// one chunk to many, at head dim 64 and at 128 (granite-3.1-8b, its query rows split into head
    /// rows of two sticks) — its body marked for the relaunch exactly when the walk loops. The widest
    /// rung's one-page walk (32 sequences, one chunk of one block slot) is the one a bake used to skip.
    #[test]
    fn a_decode_batch_lowers_at_every_rung_shape() {
        for hd in [64, 128] {
            for (seqs, walk) in [(2, 1), (2, 64), (32, 1), (32, 64), (4, 2048)] {
                let at = format!("{seqs} sequences over {walk} blocks at head dim {hd}");
                let (node, c) = batched_walk_at(seqs, walk, hd);
                let layout = triton_ktir_superdsc::layout::for_node(&node)
                    .unwrap_or_else(|e| panic!("{at}: {e:?}"));
                allocate_paged_cache(&node, &layout).expect("the cache allocates");
                let mut sym = 0i64;
                let ops =
                    ktir_superdsc::emit::whole_function::lower_walk(&node, Some(&layout), &mut sym)
                        .unwrap_or_else(|e| panic!("{at}: {}", e.message));
                assert_eq!(
                    ops.iter().any(|e| e.walk.is_some()),
                    c.num_chunks > 1,
                    "{at}"
                );
            }
        }
    }

    /// One chunk is straight-line: no loop to relaunch, every descriptor once.
    #[test]
    fn a_one_chunk_decode_batch_is_straight_line() {
        let (node, c) = batched_walk(4, 4);
        assert_eq!(c.num_chunks, 1);
        assert!(
            flatten(&node).expect("the walk flattens").is_none(),
            "a single chunk states no loop"
        );
        let layout = triton_ktir_superdsc::layout::for_node(&node).expect("the walk places");
        allocate_paged_cache(&node, &layout).expect("the cache allocates");
        let mut sym = 0i64;
        let ops = ktir_superdsc::emit::whole_function::lower_walk(&node, Some(&layout), &mut sym)
            .unwrap_or_else(|e| panic!("the one-chunk batch lowers: {}", e.message));
        assert!(ops.iter().all(|e| e.walk.is_none()));
    }

    /// The walk's trips are its blocks, and ONE trip moves the mask tile `[P, block]` — `P` stick rows
    /// of the stick-major stack — and the page index table's row by the same card bytes, which is
    /// the one shift a relaunch applies.
    #[test]
    fn the_walk_relaunches_its_body_once_per_block() {
        for (mq, blocks) in [(1i128, 2048i128), (64, 4)] {
            let node = walk(mq, blocks);
            let flat = flatten(&node)
                .expect("the walk flattens")
                .expect("a multi-block walk states its loop");
            assert_eq!(flat.fold.trips, blocks as u32, "mq={mq}");
            assert_eq!(flat.fold.step_bytes, mq as u32 * 128, "mq={mq}");
            assert_eq!(
                flat.fold.carried.len(),
                3,
                "the running max, sum and output (mq={mq})"
            );
            let layout = triton_ktir_superdsc::layout::for_node(&node).expect("the walk places");
            allocate_paged_cache(&node, &layout).expect("the cache allocates");
            let mut sym = 0i64;
            let ops =
                ktir_superdsc::emit::whole_function::lower_walk(&node, Some(&layout), &mut sym)
                    .unwrap_or_else(|e| panic!("mq={mq}: the walk lowers: {}", e.message));
            let marked: Vec<usize> = (0..ops.len()).filter(|&i| ops[i].walk.is_some()).collect();
            let (first, last) = (marked[0], *marked.last().expect("a body"));
            assert_eq!(
                marked.len(),
                last - first + 1,
                "mq={mq}: the body's descriptors are one contiguous run — one launch group"
            );
            for e in &ops[first..=last] {
                assert_eq!(
                    e.walk,
                    Some(WalkTrips {
                        trips: blocks as u32,
                        step_bytes: mq as u32 * 128
                    }),
                    "mq={mq}: {}",
                    e.op_name
                );
                assert!(e.kv_page_fold && e.kv_gathered, "mq={mq}: {}", e.op_name);
            }
            // The carried buffers are seeded, and the body's constants filled, BEFORE the body (a
            // relaunch shifts the segment the registry constants live in) — beside the query's own
            // `index_select`, which spyre-inference runs before its walk.
            assert!(
                ops[..first]
                    .iter()
                    .all(|e| e.op_name.starts_with("fill") || e.op_name.starts_with("gather")),
                "mq={mq}: only fills and the query's gather precede the body"
            );
            // The decode walk copies its `kv_row_pool` run against the int32 zero the host stages
            // beside the pool, as it stages every int32 the attention reads — nothing makes it on
            // the card.
            let zero = ktir_superdsc::place::act_name(PagedStepTids::RESERVED.int32_zero);
            assert_eq!(
                ops[first..=last]
                    .iter()
                    .any(|e| e.arg_bindings.iter().any(|a| a.buffer == zero)),
                mq == 1,
                "mq={mq}: the body reads the host-staged int32 zero exactly when the walk chains \
                 `kv_row_pool`"
            );
            assert!(
                ops[..first].iter().any(|e| e.op_name.starts_with("gather")),
                "mq={mq}: the query's rows are gathered through `query_row_index`"
            );
            assert!(
                ops[last + 1..]
                    .iter()
                    .any(|e| e.op_name.starts_with("scatter")),
                "mq={mq}: the output rows are scattered through `query_row_index`"
            );
            assert!(
                ops[first..=last]
                    .iter()
                    .all(|e| !e.op_name.starts_with("fill") && !e.op_name.starts_with("scalarmul")),
                "mq={mq}: the body reads no registry constant"
            );
            assert!(
                ops[first..=last]
                    .iter()
                    .any(|e| e.op_name.starts_with("gather")),
                "mq={mq}: the body gathers its pages"
            );
            assert!(
                ops[last + 1..].iter().all(|e| e.walk.is_none()),
                "mq={mq}: the epilogue runs once"
            );
        }
    }

    /// ⭐ A HEAD TWO STICKS WIDE (granite-3.1-8b's 128): a gathered page read `[KV, BLOCK, 128]` is
    /// permuted per kv head, inside the relaunched body — block `b` read and written at its own base
    /// `b · BLOCK · 128` — into the blocks the batched transpose and the PV contraction read one at a
    /// time. The query row is permuted once into the stick-major `[KV·G, 128]` before the walk, and
    /// the output back to its `[1, H·128]` row after it, which the scatter then writes.
    #[test]
    fn a_two_stick_head_permutes_each_kv_heads_page() {
        let node = walk_at(1, 4, 128);
        let layout = triton_ktir_superdsc::layout::for_node(&node).expect("the walk places");
        allocate_paged_cache(&node, &layout).expect("the cache allocates");
        let mut sym = 0i64;
        let ops = ktir_superdsc::emit::whole_function::lower_walk(&node, Some(&layout), &mut sym)
            .unwrap_or_else(|e| panic!("the hd128 walk lowers: {}", e.message));
        let block = PAGE_SLOTS * 128;
        let mut paged: Vec<(String, u32)> = Vec::new();
        for e in &ops {
            let Some(rest) = e.op_name.strip_prefix("permute") else {
                continue;
            };
            match rest.split_once("_o") {
                Some(("", _)) => assert!(
                    e.walk.is_none(),
                    "{}: a whole-buffer permute is the query's or the output's, outside the body",
                    e.op_name
                ),
                Some((b, dst)) => {
                    let b: u32 = b.parse().expect("a block index");
                    assert!(
                        e.walk.is_some(),
                        "{}: a page permute is in the body",
                        e.op_name
                    );
                    for a in &e.arg_bindings {
                        assert_eq!(a.offset_elems, b * block, "{}: {}", e.op_name, a.buffer);
                    }
                    paged.push((dst.to_string(), b));
                }
                None => panic!("{}: not a permute's name", e.op_name),
            }
        }
        let mut dsts: Vec<&str> = paged.iter().map(|(d, _)| d.as_str()).collect();
        dsts.sort_unstable();
        dsts.dedup();
        assert_eq!(dsts.len(), 2, "the K page and the V page: {paged:?}");
        for d in dsts {
            let mut blocks: Vec<u32> = paged
                .iter()
                .filter(|(x, _)| x == d)
                .map(|&(_, b)| b)
                .collect();
            blocks.sort_unstable();
            assert_eq!(blocks, (0..8).collect::<Vec<_>>(), "one per kv head: t{d}");
        }
        assert_eq!(
            ops.iter()
                .filter(|e| e.op_name.starts_with("permute_o"))
                .count(),
            2,
            "the query row in, the output row out"
        );
        assert!(
            ops.iter().any(|e| e.op_name.starts_with("scatter")),
            "the output row is scattered to its query row"
        );
        // The score contraction reads the query as one `[4, 128]` block per kv head: two stick-run
        // copies, laid once before the walk — the same bytes on every trip. The value contraction
        // reads each trip's probabilities `[4, PAGE_SLOTS]` the same way once a page is wider than
        // a stick: one copy per stick of the page, in the body.
        let first_body = ops.iter().position(|e| e.walk.is_some()).expect("a body");
        let blocks: Vec<usize> = (0..ops.len())
            .filter(|&i| ops[i].op_name.starts_with("heads"))
            .collect();
        let page_sticks = PAGE_SLOTS / 64;
        let probs_copies = if page_sticks > 1 { page_sticks } else { 0 };
        assert_eq!(
            blocks.len() as u32,
            2 + probs_copies,
            "one copy per stick of the head, and one per stick of a page"
        );
        let (before, inside): (Vec<usize>, Vec<usize>) = blocks
            .into_iter()
            .partition(|&i| ops[i].walk.is_none() && i < first_body);
        assert_eq!(before.len(), 2, "the query's blocks precede the walk");
        assert_eq!(
            inside.len() as u32,
            probs_copies,
            "the probabilities' blocks"
        );
        for i in inside {
            assert!(
                ops[i].walk.is_some(),
                "{}: a trip's probabilities are laid out on the trip",
                ops[i].op_name
            );
        }
    }

    /// ⭐ THE PROMPT WALK AT A HEAD TWO STICKS WIDE. The query's head-major relayout is two
    /// stick-run copies before the walk — head `a`'s stick `s`, the 64 rows at stick column `2a + s`
    /// of the gathered `[64, 32·128]` block, to `s·32·64 + a·64` of the head-major `[32·64, 128]` —
    /// and the output's is their inverse after it. A page (8 kv heads × [`PAGE_SLOTS`] slots × 128)
    /// is past dxp's 64 KB entry bound, so the body gathers it in 64 KB pieces through its page id
    /// doubled once per halving, piece `p` read and landed `p · 64 KB` in.
    #[test]
    fn a_two_stick_head_prompt_walk_relays_its_heads_and_gathers_pages_in_pieces() {
        let node = walk_at(64, 4, 128);
        let layout = triton_ktir_superdsc::layout::for_node(&node).expect("the walk places");
        allocate_paged_cache(&node, &layout).expect("the cache allocates");
        let mut sym = 0i64;
        let ops = ktir_superdsc::emit::whole_function::lower_walk(&node, Some(&layout), &mut sym)
            .unwrap_or_else(|e| panic!("the hd128 prompt walk lowers: {}", e.message));
        let first_body = ops.iter().position(|e| e.walk.is_some()).expect("a body");
        let named = |prefix: &str| -> Vec<usize> {
            (0..ops.len())
                .filter(|&i| ops[i].op_name.starts_with(prefix))
                .collect()
        };
        // (heads, rows, sticks a head)
        let (a, b, n) = (32u32, 64u32, 2u32);
        let stick = 64u32;
        // The score contraction's per-kv-head query blocks: 8 kv heads of 4·64 rows each. The
        // value contraction reads the probabilities `[8·4·64, PAGE_SLOTS]` the same way, a trip at
        // a time, once a page is wider than one stick — the same rows, so the same offsets.
        let (kv, m) = (8u32, a * b / 8);
        assert_eq!(
            PAGE_SLOTS / stick,
            n,
            "a page two sticks wide, as the head is"
        );
        let heads = named("heads");
        assert_eq!(
            heads.len(),
            8,
            "two copies in, two into the score contraction's kv-head blocks, two into the value \
             contraction's per trip, two out"
        );
        for (j, &i) in heads.iter().enumerate() {
            let e = &ops[i];
            let s = (j as u32) % n;
            let offs: Vec<u32> = e.arg_bindings.iter().map(|x| x.offset_elems).collect();
            let (block, head_major) = (s * b * stick, s * a * b * stick);
            if j < 2 {
                assert!(
                    i < first_body,
                    "{}: the query's copy precedes the walk",
                    e.op_name
                );
                assert_eq!(offs, [block, head_major], "{}", e.op_name);
            } else if j < 4 {
                assert!(
                    e.walk.is_none() && i < first_body,
                    "{}: the query's kv-head blocks are laid once, before the walk",
                    e.op_name
                );
                assert_eq!(
                    offs,
                    [s * kv * m * stick, s * m * stick],
                    "{}: stick {s} of every kv head's rows, to its block",
                    e.op_name
                );
            } else if j < 6 {
                assert!(
                    e.walk.is_some(),
                    "{}: a trip's probabilities are laid out on the trip",
                    e.op_name
                );
                assert_eq!(
                    offs,
                    [s * kv * m * stick, s * m * stick],
                    "{}: stick {s} of every kv head's probabilities, to its block",
                    e.op_name
                );
            } else {
                assert!(
                    e.walk.is_none() && i > first_body,
                    "{}: after the walk",
                    e.op_name
                );
                assert_eq!(offs, [head_major, block], "{}", e.op_name);
            }
        }
        let piece = 64 * 1024 / 2;
        let pieces = 8 * PAGE_SLOTS * 128 / piece;
        assert!(
            named(&format!("gather{pieces}_o")).is_empty(),
            "{pieces} pieces"
        );
        for d in (0..pieces).map(|p| named(&format!("gather{p}_o"))) {
            assert_eq!(d.len(), 2, "the K page and the V page");
            for i in d {
                let e = &ops[i];
                let p: u32 = e.op_name["gather".len()..e.op_name.len()]
                    .split('_')
                    .next()
                    .and_then(|x| x.parse().ok())
                    .expect("a piece index");
                assert!(
                    e.walk.is_some(),
                    "{}: a page piece is in the body",
                    e.op_name
                );
                let landed = e
                    .arg_bindings
                    .iter()
                    .find(|x| !x.is_input)
                    .expect("the landing")
                    .offset_elems;
                assert_eq!(landed, p * piece, "{}", e.op_name);
            }
        }
        let doubled = named("addi32toi32");
        assert_eq!(
            doubled.len() as u32,
            2 * pieces.trailing_zeros(),
            "each page's id doubled once per halving"
        );
        assert!(
            doubled.iter().all(|&i| ops[i].walk.is_some()),
            "per trip, in the body"
        );
    }

    /// ⛔ A PAGE IN BLOCKS IS READ ONLY AS BLOCKS. Read `[KV, BLOCK, 128]`, a page lands as its kv
    /// heads' stick-major blocks one after another — the frame a batched transpose and a batched
    /// contraction's kernel take a block of at its base, and not the flat frame any other op reads.
    /// An elementwise op over two such pages is refused by name, not lowered at the flat frame's
    /// addresses — and so is a shape op naming one in any frame.
    #[test]
    fn an_elementwise_read_of_a_page_in_blocks_is_refused() {
        // (what the program stores, the refusal's own words)
        for (stored, says) in [
            (
                "tl.reshape(k + v, [4 * 64, 128])",
                "only a batched transpose's input or a batched contraction's kernel reads",
            ),
            (
                "tl.reshape(k, [4 * 64, 128]) + tl.reshape(v, [4 * 64, 128])",
                "every frame a shape op names is the flat frame's",
            ),
        ] {
            refused_over_pages_in_blocks(stored, says);
        }
    }

    fn refused_over_pages_in_blocks(stored: &str, says: &str) {
        const SRC: &str = r#"
import triton
import triton.language as tl


@triton.jit
def two_pages(k_index, v_index, k_pages, v_pages, out, NUM_PAGES: tl.constexpr):
    k_desc = tl.make_tensor_descriptor(k_pages, shape=[NUM_PAGES, 4 * 64 * 128],
                                       strides=[4 * 64 * 128, 1], block_shape=[1, 4 * 64 * 128])
    v_desc = tl.make_tensor_descriptor(v_pages, shape=[NUM_PAGES, 4 * 64 * 128],
                                       strides=[4 * 64 * 128, 1], block_shape=[1, 4 * 64 * 128])
    rows = tl.make_tensor_descriptor(out, shape=[4 * 64, 128], strides=[128, 1],
                                     block_shape=[4 * 64, 128])
    k = tl.reshape(k_desc.load([tl.load(k_index), 0]), [4, 64, 128])
    v = tl.reshape(v_desc.load([tl.load(v_index), 0]), [4, 64, 128])
    rows.store([0, 0], STORED)
"#;
        let src = SRC.replace("STORED", stored);
        let mut signature = HashMap::new();
        let mut constexprs = HashMap::new();
        for (p, ty) in [
            ("k_index", "*i32"),
            ("v_index", "*i32"),
            ("k_pages", "*fp16"),
            ("v_pages", "*fp16"),
            ("out", "*fp16"),
        ] {
            signature.insert(p.to_string(), ArgSpec::parse(ty).expect("spec"));
        }
        signature.insert("NUM_PAGES".to_string(), ArgSpec::Constexpr);
        constexprs.insert("NUM_PAGES".to_string(), Val::Int(4));
        let spec = KernelSpec {
            kernel: "two_pages".to_string(),
            signature,
            constexprs,
            file: "two_pages.py".to_string(),
        };
        let module = compile_kernel(&src, &spec, &[1]).expect("the kernel compiles");
        let node =
            triton_ktir_superdsc::node_for(&module, Program::PagedAttn).expect("the program mints");
        let layout = triton_ktir_superdsc::layout::for_node(&node).expect("the program places");
        allocate_paged_cache(&node, &layout).expect("the cache allocates");
        let mut sym = 0i64;
        let Err(e) = ktir_superdsc::emit::whole_function::lower_function(
            &node,
            Some(&layout),
            &mut sym,
            None,
        ) else {
            panic!("{stored}: an add over two pages in blocks lowered");
        };
        assert!(e.message.contains(says), "{stored}: {}", e.message);
    }

    /// A one-block walk states no loop — the front end promotes a one-trip `scf.for` to its body —
    /// and lowers as the straight-line program it is, nothing marked for a relaunch.
    #[test]
    fn a_one_block_walk_is_straight_line() {
        let node = walk(1, 1);
        assert!(flatten(&node).expect("flattens").is_none());
        let layout = triton_ktir_superdsc::layout::for_node(&node).expect("places");
        allocate_paged_cache(&node, &layout).expect("the cache allocates");
        let mut sym = 0i64;
        let ops = ktir_superdsc::emit::whole_function::lower_walk(&node, Some(&layout), &mut sym)
            .unwrap_or_else(|e| panic!("the one-block walk lowers: {}", e.message));
        assert!(!ops.is_empty() && ops.iter().all(|e| e.walk.is_none()));
    }

    /// `reshape_and_cache_head_major` writing `t` tokens of `nkvh` kv heads at head dim `hd` into a
    /// four-page-per-head pool, through the card's lowering against its one-kernel layout.
    fn kv_write_ops(t: i128, nkvh: i128, hd: i128) -> Vec<ktir_superdsc::emit::EmittedOp> {
        let TritonKernelRow::Spliced {
            src, file, entry, ..
        } = kv_write_row()
        else {
            panic!("the KV write row is spliced");
        };
        let mut signature = HashMap::new();
        let mut constexprs = HashMap::new();
        for p in ["key", "value", "k_rows", "v_rows"] {
            signature.insert(p.to_string(), ArgSpec::parse("*fp16").expect("spec"));
        }
        signature.insert(
            "row_index".to_string(),
            ArgSpec::parse("*i32").expect("spec"),
        );
        for (k, v) in [
            ("T", t),
            ("NUM_KV_HEADS", nkvh),
            ("HEAD_SIZE", hd),
            ("NUM_ROWS", 4 * nkvh * i128::from(PAGE_SLOTS)),
        ] {
            signature.insert(k.to_string(), ArgSpec::Constexpr);
            constexprs.insert(k.to_string(), Val::Int(v));
        }
        let spec = KernelSpec {
            kernel: entry.to_string(),
            signature,
            constexprs,
            file: file.to_string(),
        };
        let module = compile_kernel(src, &spec, &[1]).expect("the write compiles");
        let node = triton_ktir_superdsc::node_for(&module, Program::PagedKvWrite)
            .expect("the write mints");
        let layout = triton_ktir_superdsc::layout::for_node(&node).expect("the write places");
        allocate_paged_cache(&node, &layout).expect("the cache allocates");
        let mut sym = 0i64;
        ktir_superdsc::emit::whole_function::lower_walk(&node, Some(&layout), &mut sym)
            .unwrap_or_else(|e| {
                panic!("t={t} nkvh={nkvh} hd={hd}: the write lowers: {}", e.message)
            })
    }

    /// ⭐ A HEAD TWO STICKS WIDE IS WRITTEN WHOLE, one index entry per row. The pool is row-outermost,
    /// so a row's two sticks are one entry there; a prompt chunk's rows are first relaid row-major,
    /// one stick permute per kv head of K and of V, and a decode step's one row per head already is.
    /// At head dim 64 nothing is relaid.
    #[test]
    fn a_two_stick_head_is_scattered_one_row_per_entry() {
        let count = |ops: &[ktir_superdsc::emit::EmittedOp], prefix: &str| {
            ops.iter().filter(|e| e.op_name.starts_with(prefix)).count()
        };
        for (t, hd, permutes) in [(1, 64, 0), (96, 64, 0), (1, 128, 0), (96, 128, 2 * 8)] {
            let ops = kv_write_ops(t, 8, hd);
            assert_eq!(
                count(&ops, "scatter_rows"),
                permutes,
                "t={t} hd={hd}: one relayout per kv head of K and of V, and only at two sticks"
            );
            // 32 index entries an op, `t · 8` rows each for K and for V.
            let per_op = 32;
            assert_eq!(
                count(&ops, "scatter_run"),
                2 * (t as usize * 8).div_ceil(per_op),
                "t={t} hd={hd}"
            );
            assert_eq!(
                ops.len(),
                count(&ops, "scatter_"),
                "t={t} hd={hd}: nothing else"
            );
        }
        // Below the scatter floor (two kv heads, one store per row) the rows coalesce the same way.
        let rows = kv_write_ops(1, 2, 128);
        assert_eq!(rows.len(), 2, "one scatter each for K and V: {:?}", {
            rows.iter().map(|e| e.op_name.clone()).collect::<Vec<_>>()
        });
    }
}
