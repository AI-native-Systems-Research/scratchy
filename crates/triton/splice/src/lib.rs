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

/// THE ATTENTION FAMILY — one row, one kernel: `batched_decode.py`'s own batched paged-decode
/// kernel, the Spyre-native `batched_decode_hm`. The old `attn.py` row (the three
/// constexpr-selected arms over the flat cache) is deleted, not left beside it; the prefill
/// shapes it served are the prefill kernel's to serve.
pub fn attn_decode_row() -> TritonKernelRow {
    TritonKernelRow::Spliced {
        src: include_str!("../../../targets/spyre/kernels/batched_decode.py"),
        file: "batched_decode.py",
        entry: "batched_decode_hm",
        program: Program::Attn,
    }
}

/// THE ATTENTION FAMILY'S PREFILL ROW — `batched_prefill.py`'s own chunked-prefill kernel,
/// the Spyre-native `batched_prefill_hm`: the token-row shapes (the prompt chunk and its
/// continuations), torch's `page_attn_head_major_prefill` algorithm over the same pool and
/// id-stick laws as the decode row.
pub fn attn_prefill_row() -> TritonKernelRow {
    TritonKernelRow::Spliced {
        src: include_str!("../../../targets/spyre/kernels/batched_prefill.py"),
        file: "batched_prefill.py",
        entry: "batched_prefill_hm",
        program: Program::Attn,
    }
}

/// THE PAGED-FOLD FAMILY'S SEED ROW — `paged_fold.py`'s `paged_fold_seed_hm`: the once-per-step
/// program that writes the fold carry's seed values (max at the finite −65504 sentinel, sum and
/// accumulator at zero). The per-page body of a chunked ladder's `(FULL, no-chunk)` rung lowers
/// as this THREE-PROGRAM family; see [`lower_attn`] for the family's contract.
pub fn paged_fold_seed_row() -> TritonKernelRow {
    TritonKernelRow::Spliced {
        src: include_str!("../../../targets/spyre/kernels/paged_fold.py"),
        file: "paged_fold.py",
        entry: "paged_fold_seed_hm",
        program: Program::Attn,
    }
}

/// THE PAGED-FOLD FAMILY'S BODY ROW — `paged_fold.py`'s `paged_fold_hm`: ONE page of the walk,
/// folded into the carried online-softmax state. Repeating this launch IS the loop; the runtime
/// re-launches it once per resident page, shifting the activation segment's base one page's
/// worth of bytes per pass (the tables and mask are page-major over the region).
pub fn paged_fold_row() -> TritonKernelRow {
    TritonKernelRow::Spliced {
        src: include_str!("../../../targets/spyre/kernels/paged_fold.py"),
        file: "paged_fold.py",
        entry: "paged_fold_hm",
        program: Program::Attn,
    }
}

/// THE PAGED-FOLD FAMILY'S FINALIZE ROW — `paged_fold.py`'s `paged_fold_finalize_hm`: the one
/// normalize (`out = acc / l`), once, after the last pass.
pub fn paged_fold_finalize_row() -> TritonKernelRow {
    TritonKernelRow::Spliced {
        src: include_str!("../../../targets/spyre/kernels/paged_fold.py"),
        file: "paged_fold.py",
        entry: "paged_fold_finalize_hm",
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
        // ⭐ THE TWO ROPE FORMS ARE TWO CONTRACTS, told apart by the SubOp — never by sniffing a
        // chain. `RopeRotate` is the q stream's rotation (arity 3, no cache); `RopeAppend` is the
        // front end's arity-6 `[K, cos, sin, V, K_cache, V_cache]` whose WRITE HALF rides the
        // door's recipe — its own kernel (the never-walked contract statement), its own
        // `Program`, and its own lowering entry ([`lower_rope_append`]), because one node
        // lowers to TWO programs (the rotation plus the carrier).
        SubOp::RopeRotate { .. } => TritonKernelRow::Spliced {
            src: include_str!("../../../targets/spyre/kernels/rope.py"),
            file: "rope.py",
            entry: "rope_fwd",
            program: Program::Rope,
        },
        SubOp::RopeAppend { .. } => TritonKernelRow::Spliced {
            src: include_str!("../../../targets/spyre/kernels/rope.py"),
            file: "rope.py",
            entry: "rope_append_fwd",
            program: Program::RopeAppend,
        },
        // ATTENTION SPLICES — `lower_attn` is the entry: the batched paged-decode kernel
        // states its whole contract as constexprs off the SubOp's own fields, and the door
        // reads every fact — the q/out views, the pool planes, the scale — off the program
        // itself. This row exists so `row()` stays exhaustive; the walk never routes
        // attention through `lower_one` (attention needs door-order bindings and the
        // synthetic id-table and mask bindings that no pure-Triton program can state).
        SubOp::AttnDecode { .. } => attn_decode_row(),
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
///
/// ⭐ THE WIDTH CONJUNCT READS THE OUTPUT **TENSOR'S** COLS, NOT THE REGION'S.
/// The front end column-chunks every elementwise op (`n_blocks(out_cols, nb)`,
/// subtile_ir.rs) — granite's 49155-wide logits ScalarMul arrives here as SEVEN
/// chunk nodes (six of 8192 + one of 3) — so a region-keyed test sees each
/// chunk's 8192 against the result's 49155 and the fold's elementwise arms
/// (ScalarMul, TanhSoftCap) never fire, while the MatmulTile above them IS
/// whole-width and folds: the tail's own scale then reads mq rows of which
/// mq−1 were never written (the read-before-write walk's first true positive,
/// prefill leg 55). Keying on the TENSOR's width makes a chunk fold iff its
/// tensor would fold as a whole node — the chunking stops changing the fold's
/// decision. For a whole-region node the two readings agree (the region IS the
/// tensor), so the MatmulTile arm and the gemma-4 collision refusal (still
/// rejected by the dataflow conjunct — a width collision is not an identity)
/// are unchanged.
pub fn is_prefill_lm_head_tail<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    rows_are_requests: bool,
) -> bool {
    let result_cols = ir.tensors[ir.result.index()].cols;
    ir.tensors[node.output.tensor.index()].cols == result_cols
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

/// ⭐⭐⭐ THE APPEND LOWERING — `RopeAppend`'s own entry, because one node lowers to TWO
/// PROGRAMS (the wall-22 ruling: two card-proven mechanisms, both main's, wired through
/// the door's own arms):
///
/// 1. THE ROTATION — `rope.py`'s `rope_fwd` minted as [`Program::Rope`], exactly the
///    `RopeRotate` row's own op: the door's `fn rope` lowers it through `rope_at` (the
///    P-table matmul), so no kernel body ever runs and the sub-stick rotation tiles that
///    wall 22 refused are never walked. The rotation writes the node's output — the
///    rotated K, PLAIN (the fp8 k_proj's scale fold is refused, so the pool readers own
///    the scale).
/// 2. THE CARRIER — `rope.py`'s `rope_append_fwd` minted as [`Program::RopeAppend`], the
///    arity-6 contract statement `[K, cos, sin, V, K_cache, V_cache]`. Its body is NEVER
///    WALKED (rope_fwd's own precedent): the door's `Program::RopeAppend` arm lowers the
///    node's write contract through `kv_cache_write` — main's recipe, re-homed from
///    `attn_at` to the rope arm — which self-stamps every op it mints (the paged latch's
///    own source). The carrier carries NO stamps and states its node's output tensor on
///    [`KtirNode::node_out_tid`] (the two-halves law: the rotation writes it, the carrier
///    names it).
///
/// Program order is the ruling's own: rotate (the P-matmul) → write (the copies + stamps
/// + the Kᵀ tail) → the attention gather that reads the pool.
///
/// ⛔ THE RECIPE CALL LIVES IN THE WALK, NEVER HERE — the splice mints TYPED programs and
/// states their bindings; lowering a node's contract through a recipe is the door's own
/// class of work (the same class as `fn rope` itself), and a splice that minted door ops
/// from a registry row would be the chain-recognizer shape the wall-21 ruling rejected.
pub fn lower_rope_append<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
) -> Result<Vec<EmittedOp>, String> {
    let SubOp::RopeAppend {
        head_dim, layout, ..
    } = &node.op
    else {
        return Err(
            "triton splice: lower_rope_append called on a node that is not a RopeAppend"
                .to_string(),
        );
    };
    // ⛔ THE CONTRACT IS THE ARITY AND THE LAYOUT WITNESS, stated once here: six inputs,
    // and the cache pair the sealed `KvCacheLayout` names IS the pair the node carries as
    // inputs[4]/[5] — the same tensors `AttnDecode`'s prefix reads, so a divergent pair is
    // a malformed graph, not a lowering decision.
    if node.inputs.len() != 6 {
        return Err(format!(
            "triton splice: rope_append t{} states {} operand(s) — the append contract is \
             exactly [K, cos, sin, V, K_cache, V_cache]",
            node.output.tensor.index(),
            node.inputs.len()
        ));
    }
    let t =
        |tr: &scratchy_subtile::subtile_ir::TensorRegion| BufferId::new(tr.tensor.index() as u32);
    let (kc_tid, vc_tid) = (layout.cache_tensor(), layout.v_cache_tensor());
    for (bound, stated, role) in [
        (kc_tid, node.inputs[4].tensor, "the K cache"),
        (vc_tid, node.inputs[5].tensor, "the V cache"),
    ] {
        if bound != stated {
            return Err(format!(
                "triton splice: rope_append t{} binds {role} as t{} but the node's operand \
                 is t{} — the KvCacheLayout's sealed pair and the graph edges diverged, which \
                 is a malformed graph (the layout is the same witness AttnDecode reads)",
                node.output.tensor.index(),
                bound.index(),
                stated.index()
            ));
        }
    }
    // The door's contract, stated from the node's own facts — the same derivation the
    // `rope_fwd` arm makes: total = the output's declared width (`heads * hd`), and the
    // builder's own refusal (`total` not a whole number of `hd`-wide heads) is mirrored.
    let hd = head_dim.get();
    let total = node.output.region.cols.len;
    if hd == 0 || !total.is_multiple_of(hd) {
        return Err(format!(
            "triton splice: rope_append t{}: {total} cols is not a whole number of \
             {hd}-wide heads",
            node.output.tensor.index()
        ));
    }
    let heads = total / hd;
    // ⛔ THE WINDOWING LAW, the `rope_fwd` arm's own, minus the dead `out`: x, cos, sin
    // and V share ONE row window, take WHOLE columns of `total`, and sit over tensors of
    // ONE height (the head-replicated cos/sin reinterpretation is same-bytes only at that
    // width). The rotation's own mint re-states the law OVER THE OUTPUT (`kernel_spec`'s
    // `rope_fwd` arm checks all four regions), so a divergent out refuses there. The
    // caches are bound by identity, not by window — the pool planes are the layout's own
    // tensors.
    let x = &node.inputs[0];
    let window = (x.region.rows.start, x.region.rows.len);
    let y_rows = ir.tensors[x.tensor.index()].rows;
    for (tr, role) in [
        (x, "the `x` view"),
        (&node.inputs[1], "the `cos` view"),
        (&node.inputs[2], "the `sin` view"),
        (&node.inputs[3], "the `v` view"),
    ] {
        let t_rows = ir.tensors[tr.tensor.index()].rows;
        let shared = (tr.region.rows.start, tr.region.rows.len) == window;
        let cols_whole = tr.region.cols.start == 0 && tr.region.cols.len == total;
        if !shared || !cols_whole || t_rows != y_rows {
            return Err(format!(
                "triton splice: rope_append t{}: {role} states rows [{}, {}] × cols [{}, \
                 {}] of a [{t_rows}, {}] tensor, but `x` states rows [{}, {}] over a \
                 [{y_rows}, {total}] one — an append's four stream regions must share ONE \
                 row window and whole `total` columns over tensors of one height",
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
    let src = match row(&node.op) {
        TritonKernelRow::Spliced { src, .. } => src,
        TritonKernelRow::Refused(why) => return Err(why.to_string()),
    };

    // ── HALF 1: THE ROTATION — `rope_fwd` minted as `Program::Rope`, the `RopeRotate`
    // row's own op. `kernel_spec`'s rope arm derives the same constexprs off THIS node
    // (it matches `RopeAppend` too), and the door's `fn rope` lowers the program through
    // `rope_at` — the card-proven P-table matmul, no kernel body, no halves.
    let spec = kernel_spec(node, ir, "rope.py", "rope_fwd")?;
    let module = compile_kernel(src, &spec, &[1])?;
    let rotation = mint(
        node,
        module,
        &Program::Rope,
        vec![
            t(&node.inputs[0]),
            t(&node.inputs[1]),
            t(&node.inputs[2]),
            t(&node.output),
        ],
    )?;
    let mut e = EmittedOp::bare(format!("rope_s{}", node.id.index()));
    e.ktir = Some(rotation);

    // ── HALF 2: THE CARRIER — `rope_append_fwd`, the arity-6 contract statement. Its
    // body is never walked (the door's `Program::RopeAppend` arm lowers the write through
    // `kv_cache_write`), so its constexprs state the CONTRACT: the node's window and the
    // pool's plane law, `SLOT_START = 0` because the runtime aims every slot (the
    // recipe's `kv_part` / `slot_stride_bytes` stamps, never a baked corner).
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    for param in [
        "desc_x", "desc_cos", "desc_sin", "desc_v", "desc_kc", "desc_vc",
    ] {
        signature.insert(
            param.to_string(),
            ArgSpec::parse("*fp16").map_err(|e| e.to_string())?,
        );
    }
    let mut ce = |k: &str, v: Val| -> Result<(), String> {
        signature.insert(k.to_string(), ArgSpec::Constexpr);
        constexprs.insert(k.to_string(), v);
        Ok(())
    };
    ce("H", Val::Int(i128::from(heads)))?;
    ce("HEAD_DIM", Val::Int(i128::from(hd)))?;
    ce("HALF", Val::Int(i128::from(hd / 2)))?;
    ce("X_ROWS", Val::Int(i128::from(y_rows)))?;
    ce("ROW_START", Val::Int(i128::from(window.0)))?;
    ce("PART_ROWS", Val::Int(i128::from(window.1)))?;
    ce("SLOT_START", Val::Int(0))?;
    ce(
        "PLANE_SLOTS",
        Val::Int(ktir_superdsc::sdsc_abstract::PagedKvPool::PLANE_SLOTS as i128),
    )?;
    let spec = KernelSpec {
        file: "rope.py".to_string(),
        kernel: "rope_append_fwd".to_string(),
        signature,
        constexprs,
    };
    let module = compile_kernel(src, &spec, &[1])?;
    // The bindings in the KERNEL's parameter order: the four stream operands, then the
    // two pool planes — the pair's OWN tensors, the WRITE's destinations (natural K and V;
    // the Kᵀ tail re-transposes from the K one into the reserved Kᵀ plane
    // (`KctResident::of(k).wire_tid()`),
    // which is what `mint_attn`'s gather reads — the pair's K tid KEYS that plane, and the
    // carrier never reads it, only fills its source). The pair is the one the contract
    // check above pinned to the graph edges.
    let bindings = vec![
        t(&node.inputs[0]),
        t(&node.inputs[1]),
        t(&node.inputs[2]),
        t(&node.inputs[3]),
        BufferId::new(kc_tid.index() as u32),
        BufferId::new(vc_tid.index() as u32),
    ];
    let mut carrier = mint(node, module, &Program::RopeAppend, bindings)?;
    // ⭐ THE TWO-HALVES LAW, STATED ON THE CARRIER: this program is HALF of the node —
    // the rotation writes the node's output, the carrier only names it (the recipe
    // suffixes its ops `_o{t}` after it) — so `node_out_tid` carries the node's output
    // tensor, exactly `LmLast`'s own precedent for a program that does not write its
    // node's output itself.
    carrier.node_out_tid = Some(t(&node.output));
    // ⭐ AND ITS OWN NAME — `ropewr_s{id}`, never the rotation's `rope_s{id}`: both
    // halves mint through `program_stem`'s `rope` stem, and two programs of one node
    // with one name is a collision the emulator's function key would take for one
    // function.
    let carrier_name = format!("ropewr_s{}", node.id.index());
    carrier.func.name = Arena::global().str(carrier_name.clone());
    let mut w = EmittedOp::bare(carrier_name);
    w.ktir = Some(carrier);
    // NO STAMPS ON THE CARRIER: the recipe self-stamps every op it mints (the paged
    // latch's own source — `latch_paged_geometry` reads the FINAL ops), so a carrier
    // carrying write facts would be a second stamping site for one fact.
    Ok(vec![e, w])
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
        // ⭐ THE TANH-SOFTCAP STATES NEITHER CORNER — its kernel's view IS the region
        // (`[M, N]`, tiles from row 0 / column 0), so BOTH corners must be 0 while the
        // extents are free: the prefill lm-head fold re-lowers the vocab-wide softcap
        // at m=1 (row 0 of the `[mq, vocab]` logits storage, a legal prefix view), and
        // the front end's chunker never tiles this op (its chunk set is the
        // elementwise/scalarmul/silumul family), so a column window never arises.
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

    // ⛔ THE ARITY IS THE NODE'S OWN CONTRACT, stated once per op kind so the splice and
    // the builder cannot disagree about it. The builder arm's own check is identical.
    // ⭐ ROPE-ROTATE'S ARITY IS 3, NOT THE NODE'S INPUT COUNT: `lower_rope_node` reads only
    // `inputs[0..3]` (x, cos, sin) — the rotation's own operands, the output binding
    // appended by `default_bindings`. `RopeAppend` has NO arity here on purpose: it routes
    // through [`lower_rope_append`] before `lower_one` is reachable, so one reaching THIS
    // match is a routing change and the `_` arm's refusal names it.
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
        SubOp::RopeRotate { .. } => 3,
        // ⭐ SCALAR-WEIGHT MUL IS ARITY-2 (`[act, weight]`, the builder's own
        // binding). The MoE sub-ops have NO arity here any more: the registry
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
//  * the KERNEL is `batched_decode.py`'s `batched_decode_hm` — the Spyre-native batched
//    paged-decode kernel, ONE program for the whole decode batch, the pool swept through
//    the device-resident page-id table (the old `attn.py` row is deleted; the prefill
//    shapes it served are the prefill kernel's to serve, refused by name until that row
//    lands);
//  * the DOOR's parameter order is q, out, kt, v, ids, mask — the output is SECOND, the
//    Kᵀ plane is the RESERVED name keyed on the sealed pair's K tid (the door's own naming
//    law, `KctResident::of(k).wire_tid()` — the same derivation the write's tail fills and main's
//    card-proven reader reads; the pair's OWN K tensor is the NATURAL plane the write's
//    cachewr copies fill, and reading it here would misread real K as transposed — same
//    size, wrong plane, invisible to every extent law), the V plane is the pair's own V
//    tensor (read natural), and the id table and the mask are bound
//    synthetic tensors beyond the graph (`k.mask`'s own law), because no graph tensor
//    names them;
//  * EVERY CONSTEXPR IS THE NODE'S OWN FACT OR A SHARED LAW — the geometry and scale off
//    `SubOp::AttnDecode`, the page geometry off `PagedKvPool`/`CopyDims` (the pool's page
//    width and the id stick's entry count are device laws, not per-model numbers), and
//    the batching off the request count the walk itself carries. Nothing is hand-typed
//    per model.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE ATTENTION SPLICE — compile `batched_decode.py` for this node's geometry, and mint the
/// `Program::Attn` `KtirNode` with the kernel's own parameter order.
///
/// ⭐ THE SYNTHETIC BINDING SLOTS — the host-staged inputs past the graph's own tensors,
/// numbered through the ONE shared law (`step_input_tid`) so the producer's placement pass
/// places exactly these tids and refuses any disagreement. The slot table is the CONTRACT's
/// own (`ktir_superdsc::sdsc_abstract::attn_step_slots` — the splice is the STATER, the worker
/// the FILLER, and the table lives in the one vocabulary crate both link; see its doc for why
/// it is a fixed ABI and not wiring data). Slot 0 keeps the mask on the tid the wiring's own
/// one-past-the-graph law already assigns it, so nothing pre-existing moves. EVERY attention
/// program of a bundle — the one-shot kernels and the paged-fold family's body — states the
/// SAME slots, so the placement pass and the runtime's staging treat them alike.
use ktir_superdsc::sdsc_abstract::attn_step_slots;

/// `cap` is the resident pool tensor's row extent (`lower_attn_node`'s own read) — for the
/// paged pool a row IS a page, so it is the decode kernel's `PAGES` constexpr. `_active_cap`
/// is the walk's swept rung, carried for the caller's contract and unread here: both batched
/// kernels sweep the pool through the id table, not a baked rung. `rows_are_requests` is the
/// ONE routing bit — the decode rung's rows are requests and `batched_decode_hm` serves
/// them; a token-row shape is a chunk and `batched_prefill_hm` serves it.
pub fn lower_attn<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    cap: u32,
    _active_cap: ktir_superdsc::ktir_node::ActiveCap,
    rows_are_requests: bool,
    paged_fold: Option<&PagedFoldRung>,
    // ⭐⭐⭐ THE BUNDLE'S SCORE-SCALE LAW — threaded from the walk that minted it
    // (`score_scale_law`'s ONE derivation, the same one the door's `BundleAttnParams` and the
    // pool write read). `None` only over a graph with no attention node; an AttnDecode under a
    // `None` is a walk/splice desync and refuses below.
    scale_law: Option<ktir_superdsc::score_scale::ScoreScaleLaw>,
) -> Result<Vec<EmittedOp>, String> {
    let row = if rows_are_requests {
        attn_decode_row()
    } else {
        attn_prefill_row()
    };
    let (src, attn_program) = match &row {
        TritonKernelRow::Spliced { src, program, .. } => (*src, program),
        TritonKernelRow::Refused(why) => return Err(why.to_string()),
    };
    let SubOp::AttnDecode {
        geom,
        scale,
        valid_len,
        layout: kv,
        ..
    } = &node.op
    else {
        return Err(
            "triton splice: lower_attn called on a node that is not an AttnDecode — the \
             entry is AttnDecode's"
                .to_string(),
        );
    };
    let (nkvh, hd) = (geom.nkvh().get(), geom.hd().get());
    let gqa = geom.gqa().get();
    // ⭐⭐⭐⭐⭐ THE SCALE, DERIVED BY EXHAUSTIVE MATCH OVER THE LAW — never a naked read off
    // the node. The node handed in may be a REWRITE (the walk rewrites `scale = 1.0` under the
    // fold's receipt, and only then); the ORIGINAL IR payload is the arch's own statement (the
    // IR is `&`-shared and every rewrite happens on a clone, so `ir.nodes[node.id]` is the
    // model's number and never a rewrite's residue); and the LAW is the bundle's proof of
    // WHERE the scale lives. The three arms:
    //   * `StatedByProgram` — the fold was REFUSED (fp8 payload, shared readers, the f16
    //     floor), the pool is RAW ([`ScoreScaleLaw::pool_k`]), and the program MUST state the
    //     arch's whole multiplier;
    //   * `CommutedIntoWeights` — the receipt: √s lives in BOTH projection weights, the pool
    //     is born scaled, and the program states the identity;
    //   * `UnitByModel` — the arch's own multiplier IS 1.0; nothing for anyone to carry.
    // The derived value is then CROSS-CHECKED against the node's carried scale: a rewritten
    // 1.0 arriving under `StatedByProgram` is the HALF-APPLIED FOLD (the recognition rewrote
    // the node while the gates refused the weight side — scores missing s, a flat softmax,
    // attention ≈ mean-V, invisible to every extent law), and a real multiplier arriving
    // under a carried-elsewhere law is the DOUBLE-APPLY. Both are build errors naming the
    // layer and both values, not emissions.
    let original = match &ir.nodes[node.id.index()].op {
        SubOp::AttnDecode { scale, .. } => *scale,
        _ => {
            return Err(format!(
                "triton splice: AttnDecode t{} is not in the graph the walk handed over — the \
                 ORIGINAL scale payload is read off `ir` (the rewrite happens on a clone), so a \
                 node the IR does not carry has no arch statement to derive the scale from",
                node.output.tensor.index()
            ));
        }
    };
    let Some(law) = scale_law else {
        return Err(format!(
            "triton splice: AttnDecode t{} reached the splice with no score-scale law — the \
             walk mints one over every graph with an attention node, so a None here is a \
             walk/splice desync, not a lawless graph",
            node.output.tensor.index()
        ));
    };
    let lawful = match law {
        ktir_superdsc::score_scale::ScoreScaleLaw::StatedByProgram => original,
        ktir_superdsc::score_scale::ScoreScaleLaw::CommutedIntoWeights => 1.0,
        ktir_superdsc::score_scale::ScoreScaleLaw::UnitByModel => original,
    };
    if *scale != lawful {
        return Err(format!(
            "triton splice: AttnDecode t{} carries scale {} but the bundle's law is {law:?} over \
             an IR payload of {original}, which mints {}. A rewritten 1.0 under StatedByProgram is \
             the HALF-APPLIED FOLD — the recognition rewrote the node while its gates refused the \
             weight side, so the pool is raw K and this program would compute scores at t = 1, a \
             flat softmax no extent law can see; a real multiplier under a carried-elsewhere law \
             is the DOUBLE-APPLY. Either the recognition must not rewrite when its gates refuse, \
             or the gates must pass",
            node.output.tensor.index(),
            scale,
            lawful,
        ));
    }
    let scale = lawful;
    // THE PAGE GEOMETRY IS A SHARED DEVICE LAW, NOT A MODEL NUMBER: the pool's page width
    // and the id stick's entry count live beside the pool types every other consumer reads.
    let page_slots = ktir_superdsc::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32;
    // ⛔ THE POOL'S ROWS ARE ITS PAGES — `cap` is the cache tensor's own row extent, and a
    // paged pool row is one whole page for every KV head. The decode kernel states it as
    // its `PAGES` constexpr.
    if cap == 0 {
        return Err(format!(
            "triton splice: AttnDecode t{} reads a zero-page pool — PAGES is the cache \
             tensor's row extent and cannot be 0",
            node.output.tensor.index()
        ));
    }
    // ── THE PAGED-FOLD FAMILY — the `(FULL, no-chunk)` rung of a chunked ladder. ──
    //
    // That rung's attention is the ONE body every context past the widest chunk falls back to,
    // so it must serve any length the pool holds — which a baked walk extent cannot. The
    // family below is the other shape: the online-softmax carry in DEVICE MEMORY, ONE page per
    // launch, the runtime re-launching the body once per resident page. Three programs, in
    // program order (seed, body, finalize): the seed is separate because a seeding PASS would
    // reset the running state mid-accumulation, and the finalize is separate because the
    // normalize runs ONCE, after the last pass.
    //
    // ⛔ THE FAMILY IS THE TOKEN-ROW (MQ==1) FORM — the kernel's own law: the carry's rows and
    // the tables' per-page blocks agree only at one query row. A batched decode rung is the
    // m>1 follow-up; stating the fold body on one is refused here rather than baked wrong.
    if let Some(pf) = paged_fold.filter(|pf| pf.fold_body) {
        if rows_are_requests {
            return Err(format!(
                "triton splice: AttnDecode t{} states the paged-fold body on a REQUEST-ROW \
                 rung — the fold family is the MQ==1 token-row form; the batched decode fold \
                 is the m>1 follow-up",
                node.output.tensor.index()
            ));
        }
        // ⛔ THE REGION CANNOT OUT-REACH THE POOL IT GATHERS: the fold's plane descriptors
        // state the region's page count, and a region past the pool's own rows sweeps past
        // the cache tensor's placement.
        if pf.region_pages > cap {
            return Err(format!(
                "triton splice: AttnDecode t{} states a {}-page table region over a {}-page \
                 pool — the region the ladder stages cannot out-reach the pool its entries \
                 gather",
                node.output.tensor.index(),
                pf.region_pages,
                cap
            ));
        }
        return mint_paged_fold(node, ir, nkvh, gqa, hd, scale, pf.region_pages);
    }
    // ⛔ A RUNG ON A CHUNKED LADDER THAT IS NOT THE FOLD BODY still states the region: every
    // rung of the ladder places its step inputs identically (`ladder_stages_one_layout`), so
    // the sweep and chunk rungs' tables and mask state the SAME region the fold reads.
    let region_pages = paged_fold.map(|pf| pf.region_pages);
    // ⭐ THE REGION VALUE IS BOUND PER ARM, from the same pages fact that arm's spec got — the
    // spec's own law, not a second derivation: the prefill arm's spec defaults a `None` region
    // to the walk's own extent, and the decode arm's spec receives the pool capacity as its
    // `pages`. The mint below states it on the carrier unconditionally.
    let (spec, region) = if rows_are_requests {
        // ── THE DECODE ROW: the request count is the batch. ──
        let num_seqs = node.output.region.rows.len;
        if num_seqs == 0 {
            return Err(format!(
                "triton splice: AttnDecode t{} states a zero-row decode output — the batched \
                 kernel's NUM_SEQS is the request count and cannot be 0",
                node.output.tensor.index()
            ));
        }
        // THE CHUNK'S ENTRY SLOTS FILL ONE WHOLE STICK: `ENTRIES = NUM_SEQS * BLOCKS_PER_CHUNK`
        // must fill the SenUint32 stick exactly (torch's own chunking law — one entry slot per
        // core), so the chunk depth is the stick divided by the batch, and a batch that does
        // not divide it has no form under this kernel. The BLOCK-ENTRY tables the gathers
        // read are wider (`ENTRIES * NKVH * windows` a chunk); the stick law governs the
        // CHUNKING, which is what feeds the softmax merge shape.
        let entries = ktir_superdsc::sdsc_abstract::CopyDims::ENTRIES_PER_OP;
        if !entries.is_multiple_of(num_seqs) {
            return Err(format!(
                "triton splice: AttnDecode t{}: a {num_seqs}-request decode batch does not \
                 divide the {entries}-entry id stick, so no BLOCKS_PER_CHUNK fills it exactly — \
                 the kernel's one-stick contract has no form at this batch width",
                node.output.tensor.index()
            ));
        }
        let blocks_per_chunk = entries / num_seqs;
        // THE CHUNK COUNT COVERS THE NODE'S OWN VALID LENGTH: `valid_len` is the SubOp's
        // stated logical length (prefix ++ new), the one sequence-extent fact the node
        // carries. Pages per sequence, then chunks at this batch's own chunk depth.
        let pages_per_seq = valid_len.div_ceil(page_slots);
        let num_chunks = pages_per_seq.div_ceil(blocks_per_chunk);
        (
            batched_decode_spec(&BatchedDecodeFacts {
                num_seqs,
                nkvh,
                gqa,
                hd,
                pages: cap,
                page_slots,
                blocks_per_chunk,
                num_chunks,
                scale,
            })?,
            // The value that arm's spec received as `pages`: the pool capacity.
            cap,
        )
    } else {
        // ── THE PREFILL ROW: the chunk's query rows are M. ──
        if node.inputs.len() < 5 {
            return Err(format!(
                "triton splice: AttnDecode t{} expects 5 inputs [q, prefix_k, prefix_v, \
                 new_k, new_v], found {}",
                node.output.tensor.index(),
                node.inputs.len()
            ));
        }
        let mq = node.output.region.rows.len;
        if mq == 0 {
            return Err(format!(
                "triton splice: AttnDecode t{} states a zero-row chunk — the prefill \
                 kernel's MQ is the chunk's query rows and cannot be 0",
                node.output.tensor.index()
            ));
        }
        // THE PAGES THE CHUNK ATTENDS: the prefix segment's rows plus the new block's —
        // the graph's own regions, the same reading the old attn.py splice took — padded
        // to whole pages. The kernel's own static_assert carries the rest of the contract
        // (one full id stick per page-window trip: `NKVH*GQA == 32`).
        let prefix_len = node.inputs[1].region.rows.len;
        let new_len = node.inputs[3].region.rows.len;
        let pages = (prefix_len + new_len).div_ceil(page_slots);
        if pages == 0 {
            return Err(format!(
                "triton splice: AttnDecode t{} attends zero pages — a {prefix_len}-row \
                 prefix plus a {new_len}-row new block has no page under PAGE_SLOTS = \
                 {page_slots}",
                node.output.tensor.index()
            ));
        }
        (
            batched_prefill_spec(&BatchedPrefillFacts {
                mq,
                nkvh,
                gqa,
                hd,
                pages,
                page_slots,
                region_pages,
                scale,
            })?,
            // The spec's own default law (`batched_prefill_spec`'s `region_pages.unwrap_or(pages)`):
            // a rung whose ladder stated no fold region keeps the region at the walk's own extent.
            region_pages.unwrap_or(pages),
        )
    };
    // ONE PROGRAM FOR THE WHOLE BATCH/CHUNK — grid [1, 1] is both kernels' own contract:
    // every descriptor corner is a compile-time constant (the page/chunk loop is
    // `tl.static_range`).
    let grid: Vec<i64> = vec![1, 1];
    let positional = attn_positional(src, &spec, &grid, *attn_program)?;
    let k = mint_attn(node, ir, positional, rows_are_requests)?;
    let name = format!("attn_s{}", node.id.index());
    // ⭐⭐⭐⭐⭐ THE REGION FACT AT THE MINT — UNCONDITIONAL, the SHARED TAIL over both row
    // arms. Every op this arm mints is a one-shot attention carrier whose program gathers the
    // paged pool through the absolute-entry tables — `mint_attn` binds the id tables in BOTH
    // row arms — whether or not its ladder stated a fold region: a PREFIX-FREE rung states
    // none (its bake is `ActiveCap::NONE`, because a start==0 chunk has nothing resident to
    // attend) yet its kernel gathers the pool the same way, so keying the mint on
    // `region_pages` left those carriers BARE and their reads double-based (the page base
    // composed on top of the absolute entries — token 0's defect). The region VALUE is each
    // arm's own contract, bound above. The pool's tids ride with it: the walk's stamp site
    // checks the layout's gather reach for exactly those tids before stamping, so a graph
    // that placed no pool REFUSES the build instead of taking the zero shift over an
    // unpaged read.
    let mut e = EmittedOp::paged_region(
        name,
        ktir_superdsc::sdsc_abstract::PagedRegion {
            region_pages: region,
            k_tid: kv.cache_tensor().index() as u32,
            v_tid: kv.v_cache_tensor().index() as u32,
        },
    );
    e.ktir = Some(k);
    Ok(vec![e])
}

/// ⭐ THE ATTENTION PROGRAM, MEMOIZED PAST `node_for` — the per-LAYER cost law.
///
/// Every layer of a rung states the SAME spec (the geometry, the widths and the corners are the
/// rung's, not the layer's), and the wiring walk lowers every layer of the unrolled graph. With
/// only [`compile_kernel`] memoized, each of the 40 layers still cloned the cached module and
/// re-ran `node_for` over it — MEASURED, after the compile memo, `node_for` alone was a
/// third of the granite-2b models build script on the 256-row prompt chunk. What differs per
/// layer is exactly what [`mint_attn`] re-states (name, bindings, mask slot), so the positional
/// node is computed once per (compile key, program) and cloned.
///
/// ⛔ THE MODULE IS NOT EDITED PAST COMPILE. The old row carried two post-compile surgeries —
/// injecting dead cache views at `prefix_len == 0`, truncating an unconsumed tail mask
/// parameter — both attention-named edits of the compiled TTIR, both unreachable on the
/// batched rows (the kernels always trace their pool views and always consume the mask), and
/// both DELETED rather than generalized: a fact the kernel does not state belongs in the
/// kernel (or in the descriptor's spec), never in a splice-side rewrite of the module the
/// walk then misreads as the kernel's own.
fn attn_positional(
    src: &str,
    spec: &KernelSpec,
    grid: &[i64],
    program: Program,
) -> Result<KtirNode, String> {
    static NODES: Memo<KtirNode> = Memo::new();
    let key = memo_key(src, spec, grid).map(|k| format!("{k}\0{program:?}"));
    NODES.get_or_compute(key, || {
        let module = compile_kernel(src, spec, grid)?;
        triton_ktir_superdsc::node_for(&module, program)
            .map_err(|e| format!("triton splice: node_for: {e}"))
    })
}

/// The batched paged-decode kernel's facts — every one the node's own or a shared law,
/// minted by [`lower_attn`] beside the law that derives each member, so this struct holds
/// no derivation of its own.
struct BatchedDecodeFacts {
    num_seqs: u32,
    nkvh: u32,
    gqa: u32,
    hd: u32,
    pages: u32,
    page_slots: u32,
    blocks_per_chunk: u32,
    num_chunks: u32,
    scale: f32,
}

/// THE BATCHED DECODE KERNEL'S LAUNCH CONTRACT — the kernel's own parameter order
/// (`batched_decode_hm`: q, out, the two pool planes, Q's row-repetition table, the two
/// block-entry tables, the mask), plus every extent as a constexpr: the geometry
/// (`NUM_SEQS/NKVH/GQA/HD`), the pool (`PAGES`, `PAGE_SLOTS`), the batching
/// (`BLOCKS_PER_CHUNK`, `NUM_CHUNKS`), and the scale.
///
/// The ids tables are `*i32` — the kernel gathers pages through them and the door sizes an
/// index buffer `SenUint32` by its own gather law; `desc_rep_ids` is the same `*i32` form
/// stating Q's entry-order row repetition. The mask is ALWAYS consumed (the kernel
/// loads it unconditionally in the chunk loop — the additive -65504 sentinel is what makes
/// a padded page merge with weight zero), so the parameter is always in the signature.
fn batched_decode_spec(f: &BatchedDecodeFacts) -> Result<KernelSpec, String> {
    let BatchedDecodeFacts {
        num_seqs,
        nkvh,
        gqa,
        hd,
        pages,
        page_slots,
        blocks_per_chunk,
        num_chunks,
        scale,
    } = *f;
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    // THE BLOCK-ENTRY CONTRACT: the decode kernel gathers the pool ONE STICK BLOCK per
    // index entry, through TWO tables — each plane's own block numbering — whose values
    // are the host's `GatherEntry` bytes. No page stride or block offset is a constexpr
    // here: those live in the values, composed at the pool's one door on the host.
    // `desc_rep_ids` is Q's own row-repetition table (torch's `rep_row_ids` values) —
    // the same host-staged index-data family, staging the entry-order repetition the
    // kernel states as its Q gather.
    for (p, ty) in [
        ("desc_q", "*fp16"),
        ("desc_o", "*fp16"),
        ("desc_kt", "*fp16"),
        ("desc_v", "*fp16"),
        ("desc_rep_ids", "*i32"),
        ("desc_ids_kt", "*i32"),
        ("desc_ids_v", "*i32"),
        ("desc_mask", "*fp16"),
    ] {
        signature.insert(
            p.to_string(),
            ArgSpec::parse(ty).map_err(|e| e.to_string())?,
        );
    }
    let mut ce = |k: &str, v: Val| -> Result<(), String> {
        signature.insert(k.to_string(), ArgSpec::Constexpr);
        constexprs.insert(k.to_string(), v);
        Ok(())
    };
    ce("NUM_SEQS", Val::Int(i128::from(num_seqs)))?;
    ce("NKVH", Val::Int(i128::from(nkvh)))?;
    ce("GQA", Val::Int(i128::from(gqa)))?;
    ce("HD", Val::Int(i128::from(hd)))?;
    ce("PAGE_SLOTS", Val::Int(i128::from(page_slots)))?;
    ce("PAGES", Val::Int(i128::from(pages)))?;
    ce("BLOCKS_PER_CHUNK", Val::Int(i128::from(blocks_per_chunk)))?;
    ce("NUM_CHUNKS", Val::Int(i128::from(num_chunks)))?;
    ce("SCALE", Val::Float(f64::from(scale)))?;
    Ok(KernelSpec {
        kernel: "batched_decode_hm".to_string(),
        signature,
        constexprs,
        file: "batched_decode.py".to_string(),
    })
}

/// The batched prefill kernel's facts — every one the node's own or a shared law, minted by
/// [`lower_attn`] beside the law that derives each member, so this struct holds no
/// derivation of its own.
struct BatchedPrefillFacts {
    mq: u32,
    nkvh: u32,
    gqa: u32,
    hd: u32,
    pages: u32,
    page_slots: u32,
    /// The page ceiling of the region the tables and mask STATE, when this rung sits on a
    /// chunked ladder (the census's ONE `max_pages` derivation, threaded by the walk). `None`
    /// states the region at the walk's own extent — byte for byte the region-less statement.
    region_pages: Option<u32>,
    scale: f32,
}

/// THE BATCHED PREFILL KERNEL'S LAUNCH CONTRACT — the same parameter order as the decode
/// kernel's (`batched_prefill_hm`: q, out, the two pool planes, the TWO block-entry tables,
/// the mask), plus every extent as a constexpr: the geometry (`MQ/NKVH/GQA/HD`), the pool
/// walk (`PAGES`, `PAGE_SLOTS`), and the scale.
///
/// THE BLOCK-ENTRY CONTRACT, under the same laws as the decode spec: the kernel gathers the
/// pool ONE STICK BLOCK per index entry, through TWO tables — each plane's own block
/// numbering — whose values are the host's `GatherEntry` bytes. No page stride or block
/// offset is a constexpr here: those live in the values, composed at the pool's one door on
/// the host. The ids table is `*i32` and the mask always consumed. The kernel's own
/// `static_assert` carries the one-stick-per-trip law (`NKVH*GQA == 32`), so
/// a model class the form does not cover refuses at compile, in the kernel's own terms.
fn batched_prefill_spec(f: &BatchedPrefillFacts) -> Result<KernelSpec, String> {
    let BatchedPrefillFacts {
        mq,
        nkvh,
        gqa,
        hd,
        pages,
        page_slots,
        region_pages,
        scale,
    } = *f;
    // ⛔ THE REGION IS THE LADDER'S, NOT THE WALK'S: on a chunked ladder every rung states the
    // SAME region so the step-input placements the placement pass mints agree across rungs
    // (`ladder_stages_one_layout`), and the per-page fold body the ladder also bakes reads the
    // same region one page per launch. `None` (an unchunked bundle) keeps the region at the
    // walk's own extent, which is the byte-for-byte region-less statement.
    let region_pages = region_pages.unwrap_or(pages);
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    for (p, ty) in [
        ("desc_q", "*fp16"),
        ("desc_o", "*fp16"),
        ("desc_kt", "*fp16"),
        ("desc_v", "*fp16"),
        ("desc_ids_kt", "*i32"),
        ("desc_ids_v", "*i32"),
        ("desc_mask", "*fp16"),
    ] {
        signature.insert(
            p.to_string(),
            ArgSpec::parse(ty).map_err(|e| e.to_string())?,
        );
    }
    let mut ce = |k: &str, v: Val| -> Result<(), String> {
        signature.insert(k.to_string(), ArgSpec::Constexpr);
        constexprs.insert(k.to_string(), v);
        Ok(())
    };
    ce("MQ", Val::Int(i128::from(mq)))?;
    ce("NKVH", Val::Int(i128::from(nkvh)))?;
    ce("GQA", Val::Int(i128::from(gqa)))?;
    ce("HD", Val::Int(i128::from(hd)))?;
    ce("PAGE_SLOTS", Val::Int(i128::from(page_slots)))?;
    ce("PAGES", Val::Int(i128::from(pages)))?;
    ce("REGION_PAGES", Val::Int(i128::from(region_pages)))?;
    ce("SCALE", Val::Float(f64::from(scale)))?;
    Ok(KernelSpec {
        kernel: "batched_prefill_hm".to_string(),
        signature,
        constexprs,
        file: "batched_prefill.py".to_string(),
    })
}

/// ⭐ WHAT THE LADDER STATES TO THE SPLICE ABOUT THE PAGED FOLD, at the rung being lowered.
///
/// A chunked ladder states this for EVERY rung (the ceiling included): the sweep and chunk
/// rungs need the REGION so their step-input placements agree with the fold body's
/// (`ladder_stages_one_layout` — one session stages one layout), and the fold body is the
/// `(FULL, no-chunk)` rung the ladder's every-past-the-widest-chunk context falls back to.
/// `region_pages` is the census's ONE `max_pages` derivation, threaded by the walk — the
/// splice never re-derives it.
pub struct PagedFoldRung {
    /// The page ceiling of the region the block-entry tables and the mask state.
    pub region_pages: u32,
    /// THIS rung is the per-page fold body — the `(FULL, no-chunk)` rung.
    pub fold_body: bool,
}

/// The paged-fold family's facts — the fold body's own geometry, minted by
/// [`mint_paged_fold`] beside the law that derives each member. The seed and finalize
/// programs read the SUBSET of these their signatures state.
struct PagedFoldFamilyFacts {
    mq: u32,
    nkvh: u32,
    gqa: u32,
    hd: u32,
    page_slots: u32,
    region_pages: u32,
    scale: f32,
}

/// THE SEED PROGRAM'S LAUNCH CONTRACT — `paged_fold_seed_hm`'s own parameter order (the
/// carry's max, sum, accumulator), plus the geometry as constexprs. The carry's memory image
/// is f16 throughout — the one-shot kernel's own legalized grading (the kernel's module doc
/// states the law), so the fold is numerically identical to the body it replaces.
fn paged_fold_seed_spec(f: &PagedFoldFamilyFacts) -> Result<KernelSpec, String> {
    let PagedFoldFamilyFacts {
        mq, nkvh, gqa, hd, ..
    } = *f;
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    for (p, ty) in [
        ("desc_m", "*fp16"),
        ("desc_l", "*fp16"),
        ("desc_acc", "*fp16"),
    ] {
        signature.insert(
            p.to_string(),
            ArgSpec::parse(ty).map_err(|e| e.to_string())?,
        );
    }
    let mut ce = |k: &str, v: Val| -> Result<(), String> {
        signature.insert(k.to_string(), ArgSpec::Constexpr);
        constexprs.insert(k.to_string(), v);
        Ok(())
    };
    ce("MQ", Val::Int(i128::from(mq)))?;
    ce("NKVH", Val::Int(i128::from(nkvh)))?;
    ce("GQA", Val::Int(i128::from(gqa)))?;
    ce("HD", Val::Int(i128::from(hd)))?;
    Ok(KernelSpec {
        kernel: "paged_fold_seed_hm".to_string(),
        signature,
        constexprs,
        file: "paged_fold.py".to_string(),
    })
}

/// THE FOLD BODY'S LAUNCH CONTRACT — `paged_fold_hm`'s own parameter order (q, the carried
/// max/sum/accumulator, the two pool planes, the two block-entry tables, the mask), plus every
/// extent as a constexpr: the geometry (`MQ/NKVH/GQA/HD`), the pool (`PAGE_SLOTS`), the region
/// the tables and mask state (`REGION_PAGES`), and the scale. The block-entry contract is the
/// prefill kernel's own — global stick-block entries, one per table row — and the per-pass
/// shift the runtime applies rides the ACTIVATION segment, page-major, which is why the tables
/// and the mask state the REGION and not the page.
fn paged_fold_spec(f: &PagedFoldFamilyFacts) -> Result<KernelSpec, String> {
    let PagedFoldFamilyFacts {
        mq,
        nkvh,
        gqa,
        hd,
        page_slots,
        region_pages,
        scale,
    } = *f;
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    for (p, ty) in [
        ("desc_q", "*fp16"),
        ("desc_m", "*fp16"),
        ("desc_l", "*fp16"),
        ("desc_acc", "*fp16"),
        ("desc_kt", "*fp16"),
        ("desc_v", "*fp16"),
        ("desc_ids_kt", "*i32"),
        ("desc_ids_v", "*i32"),
        ("desc_mask", "*fp16"),
    ] {
        signature.insert(
            p.to_string(),
            ArgSpec::parse(ty).map_err(|e| e.to_string())?,
        );
    }
    let mut ce = |k: &str, v: Val| -> Result<(), String> {
        signature.insert(k.to_string(), ArgSpec::Constexpr);
        constexprs.insert(k.to_string(), v);
        Ok(())
    };
    ce("MQ", Val::Int(i128::from(mq)))?;
    ce("NKVH", Val::Int(i128::from(nkvh)))?;
    ce("GQA", Val::Int(i128::from(gqa)))?;
    ce("HD", Val::Int(i128::from(hd)))?;
    ce("PAGE_SLOTS", Val::Int(i128::from(page_slots)))?;
    ce("REGION_PAGES", Val::Int(i128::from(region_pages)))?;
    ce("SCALE", Val::Float(f64::from(scale)))?;
    Ok(KernelSpec {
        kernel: "paged_fold_hm".to_string(),
        signature,
        constexprs,
        file: "paged_fold.py".to_string(),
    })
}

/// THE FINALIZE PROGRAM'S LAUNCH CONTRACT — `paged_fold_finalize_hm`'s own parameter order
/// (the output, the carried sum, the carried accumulator). The max is spent by the last pass,
/// so nothing reads it here — the signature states no `desc_m`, and an unused parameter is
/// not merely dead weight: Triton may drop it from the compiled signature and break the
/// parameter-count guard below.
fn paged_fold_finalize_spec(f: &PagedFoldFamilyFacts) -> Result<KernelSpec, String> {
    let PagedFoldFamilyFacts {
        mq, nkvh, gqa, hd, ..
    } = *f;
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    for (p, ty) in [
        ("desc_o", "*fp16"),
        ("desc_l", "*fp16"),
        ("desc_acc", "*fp16"),
    ] {
        signature.insert(
            p.to_string(),
            ArgSpec::parse(ty).map_err(|e| e.to_string())?,
        );
    }
    let mut ce = |k: &str, v: Val| -> Result<(), String> {
        signature.insert(k.to_string(), ArgSpec::Constexpr);
        constexprs.insert(k.to_string(), v);
        Ok(())
    };
    ce("MQ", Val::Int(i128::from(mq)))?;
    ce("NKVH", Val::Int(i128::from(nkvh)))?;
    ce("GQA", Val::Int(i128::from(gqa)))?;
    ce("HD", Val::Int(i128::from(hd)))?;
    Ok(KernelSpec {
        kernel: "paged_fold_finalize_hm".to_string(),
        signature,
        constexprs,
        file: "paged_fold.py".to_string(),
    })
}

/// THE PAGED-FOLD FAMILY'S MINT — three programs, in PROGRAM ORDER (seed, fold body,
/// finalize), each re-stated like [`mint_attn`] re-states the one-shot kernels: the bindings
/// in the KERNEL's parameter order, the Kᵀ plane bound to the RESERVED tid keyed on the
/// sealed pair's K tid and the V plane to the pair's own V tensor (the same naming law
/// [`mint_attn`] states), the tables and mask as host-staged step inputs past the graph (the
/// SAME slot law [`mint_attn`] numbers), and the carry bound to its three reserved tids — the
/// placements the census mints, gated on the ladder's fold statement.
///
/// ⭐ ONLY THE BODY CARRIES THE FOLD ATTRIBUTE ([`EmittedOp::paged_fold`], the ONE door): the
/// seed and the finalize launch once per step, not once per page, so stamping a fold onto
/// them would re-launch a seeding pass mid-accumulation and a normalize per page — the two
/// defects the three-program split exists to prevent.
fn mint_paged_fold<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    nkvh: u32,
    gqa: u32,
    hd: u32,
    scale: f32,
    region_pages: u32,
) -> Result<Vec<EmittedOp>, String> {
    let SubOp::AttnDecode { layout, .. } = &node.op else {
        return Err(
            "triton splice: mint_paged_fold called on a node that is not an AttnDecode".to_string(),
        );
    };
    let mq = node.output.region.rows.len;
    let page_slots = ktir_superdsc::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32;
    let facts = PagedFoldFamilyFacts {
        mq,
        nkvh,
        gqa,
        hd,
        page_slots,
        region_pages,
        scale,
    };
    // ONE PROGRAM PER FAMILY MEMBER — grid [1, 1] is the family's own contract: every
    // descriptor corner is a compile-time constant, the body's page loop is the RUNTIME's
    // re-launch, not a `tl.static_range` walk.
    let grid: Vec<i64> = vec![1, 1];
    let row_src = |row: &TritonKernelRow| -> Result<(&str, Program), String> {
        match row {
            TritonKernelRow::Spliced { src, program, .. } => Ok((src, *program)),
            TritonKernelRow::Refused(why) => Err(why.to_string()),
        }
    };
    let (seed_src, seed_prog) = row_src(&paged_fold_seed_row())?;
    let (fold_src, fold_prog) = row_src(&paged_fold_row())?;
    let (fin_src, fin_prog) = row_src(&paged_fold_finalize_row())?;
    let seed_pos = attn_positional(seed_src, &paged_fold_seed_spec(&facts)?, &grid, seed_prog)?;
    let fold_pos = attn_positional(fold_src, &paged_fold_spec(&facts)?, &grid, fold_prog)?;
    let fin_pos = attn_positional(fin_src, &paged_fold_finalize_spec(&facts)?, &grid, fin_prog)?;
    // ⭐ THE TAPE'S REAL TENSOR INDICES over the adapter's positional ones — the same law
    // `mint_attn` states: a splice that kept the positional ids would emit descriptors over
    // `t0/t1/t2` while the tape's buffers are elsewhere.
    let t =
        |tr: &scratchy_subtile::subtile_ir::TensorRegion| BufferId::new(tr.tensor.index() as u32);
    // The carry's three reserved tids — the census places exactly these, gated on the fold
    // statement, so a binding and a placement can never disagree about which tensor is which.
    let carry = |which: ktir_superdsc::reserved_tids::PagedFoldCarry| BufferId::new(which.tid());
    let m_tid = carry(ktir_superdsc::reserved_tids::PagedFoldCarry::Max);
    let l_tid = carry(ktir_superdsc::reserved_tids::PagedFoldCarry::Sum);
    let acc_tid = carry(ktir_superdsc::reserved_tids::PagedFoldCarry::Acc);
    // The step-input slot law is `mint_attn`'s own (the mask at 0, the two tables at 1 and 2)
    // — the fold body states the SAME host-staged inputs the one-shot kernels do, at the
    // SAME tids, so the placement pass and the runtime's staging treat every attention
    // program of the bundle alike.
    let n_tensors = ir.tensors.len() as u32;
    let law = |slot: u32| BufferId::new(ktir_superdsc::ktir_node::step_input_tid(n_tensors, slot));
    let mask_tid = law(attn_step_slots::MASK);
    let kt_ids_tid = law(attn_step_slots::KT_BLOCK_IDS);
    let v_ids_tid = law(attn_step_slots::V_BLOCK_IDS);
    // The re-state: the positional node's func with THIS program's own name, bindings, mask
    // and step inputs. The parameter-count guard is `mint_attn`'s own — the kernel's
    // parameter list and the row's contract cannot diverge silently.
    let restate = |positional: KtirNode,
                   name: &'static str,
                   bindings: Vec<BufferId>,
                   mask: Option<BufferId>,
                   step_inputs: Vec<ktir_superdsc::ktir_node::StepInput>|
     -> Result<KtirNode, String> {
        let KtirNode { func, program, .. } = positional;
        if func.arguments.len() != bindings.len() {
            return Err(format!(
                "triton splice: {} compiled to {} parameters but the kernel's order states {} \
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
            mask,
            step_inputs,
            node_out_tid: None,
            minted: Default::default(),
        })
    };
    let id = node.id.index();
    // ── THE SEED: the carry's three tensors, in the kernel's order (max, sum, accumulator).
    // No step inputs, no mask — the program touches nothing the host stages. ──
    let seed = restate(
        seed_pos,
        Arena::global().str(format!("attn_s{id}_fseed")),
        vec![m_tid, l_tid, acc_tid],
        None,
        Vec::new(),
    )?;
    // ── THE BODY: q, the carry, the pool planes, the tables, the mask — the kernel's own
    // parameter order. ──
    let fold = restate(
        fold_pos,
        Arena::global().str(format!("attn_s{id}_fold")),
        vec![
            t(&node.inputs[0]),
            m_tid,
            l_tid,
            acc_tid,
            // ⭐ THE SCORE GATHER READS THE RESERVED Kᵀ PLANE — `KctResident` keyed on the
            // sealed pair's K tid, the SAME expression the door's write tail
            // (ktir_superdsc_door.rs) and main's card-proven reader spell, and the tape layer's
            // body-representative table before them: the THIRD sayer of one derivation. The
            // pair's own K tensor is the NATURAL plane the write's cachewr copies fill; binding
            // it here misreads real K as transposed — same size, wrong plane, invisible to every
            // extent law (both planes are `plane_stride` per kv head). Minted through the TYPE
            // so the binding cannot be spelled as the source tid: the gather door classifies it
            // by kind, and a `KctResident` resolves to the reserved plane placement and nothing
            // else.
            BufferId::new(
                ktir_superdsc::reserved_tids::KctResident::of(layout.cache_tensor().index() as u32)
                    .wire_tid(),
            ),
            BufferId::new(layout.v_cache_tensor().index() as u32),
            kt_ids_tid,
            v_ids_tid,
            mask_tid,
        ],
        Some(mask_tid),
        vec![
            ktir_superdsc::ktir_node::StepInput {
                buffer: kt_ids_tid,
                slot: attn_step_slots::KT_BLOCK_IDS,
            },
            ktir_superdsc::ktir_node::StepInput {
                buffer: v_ids_tid,
                slot: attn_step_slots::V_BLOCK_IDS,
            },
            ktir_superdsc::ktir_node::StepInput {
                buffer: mask_tid,
                slot: attn_step_slots::MASK,
            },
        ],
    )?;
    // ── THE FINALIZE: the output, the sum, the accumulator — in the kernel's order. ──
    let fin = restate(
        fin_pos,
        Arena::global().str(format!("attn_s{id}_ffin")),
        vec![t(&node.output), l_tid, acc_tid],
        None,
        Vec::new(),
    )?;
    // The BODY carries the fold attribute — the family's launch-ABI laws, stated at the one
    // door the walk's stamp and the bake's proof read. `page_slots` is restated from the
    // shared device law so the stamp never re-derives it.
    let mut body = EmittedOp::paged_fold(
        format!("attn_s{id}_fold"),
        ktir_superdsc::sdsc_abstract::PagedFold {
            gathered: true,
            batched_requests: false,
            fold_rows: ktir_superdsc::sdsc_abstract::FoldRowRegime::WholeBatch,
            page_slots,
        },
    );
    body.ktir = Some(fold);
    let mut seed_op = EmittedOp::bare(format!("attn_s{id}_fseed"));
    seed_op.ktir = Some(seed);
    let mut fin_op = EmittedOp::bare(format!("attn_s{id}_ffin"));
    fin_op.ktir = Some(fin);
    Ok(vec![seed_op, body, fin_op])
}

/// THE ATTENTION MINT — the adapter's `node_for` compilation ([`attn_positional`]), then the
/// attention's own re-state: the bindings in the KERNEL's parameter order (q, out, the two
/// pool planes, the two block-entry tables, the mask), the Kᵀ plane bound to the RESERVED name
/// keyed on the sealed pair's K tid (the door's own naming law, `KctResident::of(k).wire_tid()` — the same
/// derivation the write's tail fills and main's card-proven reader reads; the pair's OWN K
/// tensor is the NATURAL plane the write's cachewr copies fill, and the score gather must not
/// read it — same size, wrong plane, invisible to every extent law) and the V plane to the
/// pair's own V tensor (read natural), the two block-entry tables and the mask as host-staged
/// step inputs past the graph (numbered through the one shared `step_input_tid` law and STATED
/// on the node, so the producer's placement pass places exactly those tids and the runtime
/// discovers them by placement), and the builder's `attn_s{id}` name.
fn mint_attn<F: scratchy_subtile::subtile_ir::RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    positional: KtirNode,
    rows_are_requests: bool,
) -> Result<KtirNode, String> {
    let SubOp::AttnDecode { layout, .. } = &node.op else {
        return Err(
            "triton splice: mint_attn called on a node that is not an AttnDecode".to_string(),
        );
    };
    // ⛔ THE KERNEL'S ORDER, NOT `lower_one`'s — and the two rows state DIFFERENT contracts:
    // the decode kernel (`batched_decode_hm`) takes q, OUT, the two pool planes, Q's
    // row-repetition table, the two block-entry tables, the mask; the prefill kernel
    // (`batched_prefill_hm`) states no repetition (its BATCH is Q's own rows, each used
    // once), so its list stops at the mask. `rows_are_requests` is the same routing bit
    // that chose the kernel, so the binding list states the contract of the kernel the
    // router compiled. `regions()` zips `bindings[i]` against `arguments[i]`, so a swapped
    // or extra binding is a descriptor over the wrong tensor — the parameter-count guard
    // below refuses the divergence.
    let t =
        |tr: &scratchy_subtile::subtile_ir::TensorRegion| BufferId::new(tr.tensor.index() as u32);
    // ⭐ THE SYNTHETIC BINDING SLOTS — the contract's own `attn_step_slots` table (the
    // paged-fold family's body states the same slots), numbered through the ONE shared law
    // (`step_input_tid`).
    let n_tensors = ir.tensors.len() as u32;
    let law = |slot: u32| BufferId::new(ktir_superdsc::ktir_node::step_input_tid(n_tensors, slot));
    let mask_tid = law(attn_step_slots::MASK);
    let kt_ids_tid = law(attn_step_slots::KT_BLOCK_IDS);
    let v_ids_tid = law(attn_step_slots::V_BLOCK_IDS);
    let rep_ids_tid = law(attn_step_slots::REP_ROW_IDS);
    // The decode row's repetition table (slot 3); the prefill row states none.
    let mut bindings = vec![
        t(&node.inputs[0]),
        t(&node.output),
        // ⭐ THE SCORE GATHER READS THE RESERVED Kᵀ PLANE — `KctResident` keyed on the
        // sealed pair's K tid, the SAME expression the door's write tail
        // (ktir_superdsc_door.rs) and main's card-proven reader spell, and the tape layer's
        // body-representative table before them: the THIRD sayer of one derivation. The pair's
        // own K tensor is the NATURAL plane the write's cachewr copies fill; binding it here
        // misreads real K as transposed — same size, wrong plane, invisible to every extent law
        // (both planes are `plane_stride` per kv head). Minted through the TYPE so the binding
        // cannot be spelled as the source tid: the gather door classifies it by kind, and a
        // `KctResident` resolves to the reserved plane placement and nothing else.
        BufferId::new(
            ktir_superdsc::reserved_tids::KctResident::of(layout.cache_tensor().index() as u32)
                .wire_tid(),
        ),
        BufferId::new(layout.v_cache_tensor().index() as u32),
        kt_ids_tid,
        v_ids_tid,
        mask_tid,
    ];
    if rows_are_requests {
        // Position 4: after the two pool planes, before the block-entry tables —
        // the decode kernel's own parameter order (`desc_rep_ids` past `desc_v`).
        bindings.insert(4, rep_ids_tid);
    }
    let name = Arena::global().str(format!("attn_s{}", node.id.index()));
    let KtirNode { func, program, .. } = positional;
    if func.arguments.len() != bindings.len() {
        return Err(format!(
            "triton splice: {} compiled to {} parameters but the kernel's order states {} \
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
        mask: Some(mask_tid),
        step_inputs: {
            let mut staged = vec![
                ktir_superdsc::ktir_node::StepInput {
                    buffer: kt_ids_tid,
                    slot: attn_step_slots::KT_BLOCK_IDS,
                },
                ktir_superdsc::ktir_node::StepInput {
                    buffer: v_ids_tid,
                    slot: attn_step_slots::V_BLOCK_IDS,
                },
                ktir_superdsc::ktir_node::StepInput {
                    buffer: mask_tid,
                    slot: attn_step_slots::MASK,
                },
            ];
            if rows_are_requests {
                staged.push(ktir_superdsc::ktir_node::StepInput {
                    buffer: rep_ids_tid,
                    slot: attn_step_slots::REP_ROW_IDS,
                });
            }
            staged
        },
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
/// (`mint`/`mint_attn` re-state names and bindings), so a cached pristine `Module` is
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
    // ⭐ `RopeAppend` never routes through here — [`lower_rope_append`] states its bindings
    // itself (the four stream operands plus the two pool planes off the sealed
    // `KvCacheLayout`, no output binding: the write rides the door's recipe, so the
    // carrier names its node's output on `node_out_tid` instead). Keeping it out of
    // this match makes a misroute loud at the arity check instead of binding six
    // as if three were the rotation.
    let n_bound = match &node.op {
        SubOp::RopeRotate { .. } => 3,
        _ => node.inputs.len(),
    };
    let mut bindings: Vec<BufferId> = node.inputs.iter().take(n_bound).map(&t).collect();
    bindings.push(t(&node.output));
    bindings
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
        step_inputs: Vec::new(),
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
        // The append keeps the builder's `rope` stem at the mint — it IS the rope node —
        // and `lower_rope_append` then renames the carrier `ropewr_s{id}` so the node's
        // two programs (rotation + carrier) never share one name.
        Program::RopeAppend => "rope",
        Program::ScalarMul => "scalarmul",
        // ⭐ THE SANDWICH'S OWN STEM — `rmsnorm_s{split.id}` (the split's id, not the
        // middle norm's), main's naming law for the fused per-head norm.
        Program::WindowedRmsNorm { .. } => "rmsnorm",
        Program::RmsNormUnit => "rmsnormunit",
        Program::TanhSoftCap => "tanhsoftcap",
        Program::ScalarWeightMul => "scalarwmul",
        Program::Reshape => "reshape",
        _ => "triton",
    }
}
