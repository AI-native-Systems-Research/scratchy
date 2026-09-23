// SPDX-License-Identifier: Apache-2.0
//! THE REQUEST TYPE — one node's KTIR, plus the facts its program does not carry.
//!
//! ⭐ THIS IS THE LOWERING'S INPUT, AND IT IS WHY THE CRATE EXISTS. A KTIR producer — whatever its
//! front end — hands one [`KtirNode`] per node and the lowering answers with SuperDSC descriptors.
//! Everything here is the REQUEST: not the device's laws, and not the caller's bake plan.
//!
//! ⛔ WHAT IS LEFT HERE IS A BINDING, NOT A FACT ABOUT THE PROGRAM. `out_shape`,
//! `scalarmul_scale_idx` and `rmsnorm_eps_idx` are GONE — each was a value the producer computed and
//! stapled to the program while the program already stated it, so the two could disagree and a
//! third-party producer had to supply a number its own KTIR carries. They are read from the IR now:
//! the row count from the output's own store windows, the query rows from `q`'s view, rope's rows from
//! its access tile, and both constants from the splats they feed.
//!
//! ⛔ THE ONE THING NO PROGRAM CAN STATE IS WHICH BUFFER A PARAMETER ADDRESSES. A KTIR function's
//! parameters are `index` start addresses; nothing inside the function says which allocation each one
//! points at, and a launch has to bind an address per parameter. So [`KtirNode::bindings`] carries
//! that, as [`BufferId`]s the CALLER numbers — and the crate never asks what a buffer is, only names
//! it ([`act_name`](crate::place::act_name)) and looks it up in the caller's own
//! [`BundleLayout`](crate::placement::BundleLayout). Every other property comes from the program: the
//! extents and strides from `ktdp.construct_memory_view`, the element format from that view's own
//! attributes. That is why this needs no trait over the producer's tensor type — no question ever
//! travels back.
//!
//! ⛔⛔⛔ AND ATTENTION HAS NO ENTRY HERE, WHICH IS THE POINT OF THE TYPE. A sidecar named
//! `AttnFacts` used to ride beside the bindings: eleven fields the PRODUCER computed off its own
//! graph node and stapled to the program, which the lowering then read INSTEAD OF THE PROGRAM.
//! Information flowed producer → sidecar → SuperDSC, straight past the IR — so a producer that did
//! not have that graph could not fill it (making this crate's attention path unusable to it), and the
//! interpreter (which executes the ops) and the CARD (which read the sidecar) were not obliged to
//! compute the same thing. Seven of the eleven are read out of the program by
//! [`attn_operands`](crate::emit::lower_ktir_to_superdsc::attn_operands); of the four that remained,
//! the multiplier and the swept KV extent have since been read off the program too, leaving the ones
//! that are facts of the MODEL and the BUNDLE as ordinary arguments of
//! [`AttnAt`](crate::emit::lower_ktir_to_superdsc::AttnAt) —
//! `ibm/main`'s own `LowerAttn` fields — from the caller that crosses the geometry door.
//!
//! [`ActiveCap`] stays here because it is that door's own vocabulary: the swept KV extent is a fact
//! of the BUNDLE the caller is told and the bake never sees, and the lowering's argument pack names
//! its type.

// `DataFormat` is in scope for its `ELEMS_PER_STICK` associated const, which
// `ActiveCap::decode_ladder` aligns each rung to; `Fp16` is the format that answers it.
use crate::superdsc_opspec::{DataFormat, Fp16};

/// Which elementwise function a program computes — see [`Program::Elementwise`].
///
/// ⛔ THE KINDS THE DEVICE CANNOT DO ARE VARIANTS TOO. `QuickGelu`/`GeluErf` have no DDL primitive,
/// and they are named here rather than absent so a producer emitting one gets a refusal that NAMES
/// it — substituting the nearest primitive would run a different function and report success.
/// `QuickGelu` is x·σ(1.702x) and `GeluErf` the exact-erf gelu; neither is `OpFunc::Gelu`'s tanh
/// polynomial, and there is no `OpFunc` for either.
///
/// ⛔⛔ `Sub` WAS IN THAT LIST AND IT DID NOT BELONG THERE. The refusal's stated reason was "Sub
/// takes a `[m, 1]` broadcast operand that `pw2` cannot express", and every clause of that was
/// false: `OpFunc::Subtract` exists, spells `"sub"` — which is in the on-card-verified recognized
/// set (`crates/targets/spyre/tests/superdsc_time_tile.rs`, asserted against dxp's own
/// `dscdefn.cpp opFuncsToString`) — `pw2` has no arity law and `In::col` expresses exactly the
/// `[m, 1]` operand it claimed was inexpressible, and
/// [`elementwise`](crate::emit::lower_ktir_to_superdsc::elementwise) never calls `pw2` at all. A
/// NON-broadcasting subtract is structurally identical to `Add`/`Mul`; the reference evaluator
/// permits `bc == ac || bc == 1` for all three ALIKE, so broadcasting was never `Sub`-specific. The
/// real hazard — an operand whose extent differs from the output's — is now guarded for EVERY
/// binary kind by the extent check in `elementwise`, which is where it always belonged.
///
/// ⭐ THE POINTWISE SET IS THE `OpFunc` SET, NOT A HAND-PICKED FOUR. Every variant below resolves
/// through `op_func_from_str` to an `OpFunc` that already existed; a target that has the primitive
/// should not refuse the op merely because this enum never named it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Elementwise {
    Silu,
    Gelu,
    Mul,
    Add,
    QuickGelu,
    GeluErf,
    /// `a - b`. `OpFunc::Subtract`, dxp `"sub"`.
    Sub,
    // ── UNARY (`OpFunc` + `op_func_from_str` arm already existed for every one) ──
    /// `e^x`. Live on card already via the attention softmax (`attn.rs`).
    Exp,
    /// `1/√x`. Live on card already via the rms normalisation (`rmsnorm.rs`).
    Rsqrt,
    /// `√x`.
    Sqrt,
    /// `|x|`. Live on card already via the fp8 quantiser's absmax (`ktir_matmul_fp8.rs`).
    Abs,
    /// `1/x`. Live on card already via the fp8 quantiser's scale (`ktir_matmul_fp8.rs`).
    Reciprocal,
    /// `σ(x)`.
    Sigmoid,
    /// `tanh(x)`.
    Tanh,
    /// `x·tanh(ln(1 + e^x))`.
    Mish,
    // ── BINARY ──
    /// `a / b`. Live on card already via the attention rescale (`attn.rs`).
    RealDiv,
    /// `max(a, b)`. Live on card already via the attention running max (`attn.rs`) and the fp8
    /// quantiser's clamp (`ktir_matmul_fp8.rs`).
    Maximum,
    /// `min(a, b)`. Live on card already via the fp8 quantiser's clamp (`ktir_matmul_fp8.rs`).
    Minimum,
}

impl Elementwise {
    /// EVERY variant, declared once — the set the consumer-door tests sweep.
    ///
    /// ⛔ COMPLETENESS IS PROVEN, NOT PROMISED. A list like this normally rots the first time a
    /// variant is added, so the test module's `declared_index` is an exhaustive `match` with NO `_`
    /// arm: a new variant is an E0004 there, and the sweep then fails unless it is also added here.
    pub const ALL: &'static [Elementwise] = &[
        Elementwise::Silu,
        Elementwise::Gelu,
        Elementwise::Mul,
        Elementwise::Add,
        Elementwise::QuickGelu,
        Elementwise::GeluErf,
        Elementwise::Sub,
        Elementwise::Exp,
        Elementwise::Rsqrt,
        Elementwise::Sqrt,
        Elementwise::Abs,
        Elementwise::Reciprocal,
        Elementwise::Sigmoid,
        Elementwise::Tanh,
        Elementwise::Mish,
        Elementwise::RealDiv,
        Elementwise::Maximum,
        Elementwise::Minimum,
    ];
}

/// WHAT THIS PROGRAM COMPUTES — the node kind, as a type.
///
/// ⛔⛔⛔ IT WAS A SUBSTRING OF `func.name`, AND THAT IS A STRINGY CONTRACT THIS CRATE OWNS. The
/// producer named every program `<kind>_s<node index>` and the consumer recovered the kind with
/// `name.split('_').next()`. `IRFunction::name` being `ktir_core`'s own `&str` field is no excuse:
/// [`KtirNode`] is OUR struct, so encoding a fact in a name and parsing it back out was our choice.
/// A typo'd stem was a runtime refusal, a new kind was a silently unmatched arm, and the dispatch had
/// to split a string to know which body to run.
///
/// ⛔ THIS FIELD IS NOT HERE BECAUSE THE FACT IS UNDERIVABLE. IT IS DERIVABLE, AND THE CLAIM THAT IT WAS
/// NOT — "a pointwise KTIR body is the same op-DAG shape whichever function it applies" — WAS FALSE. This
/// tree's own producer emits `arith.addf` for an add, `arith.mulf` for a mul and a SIX-op longhand for a
/// silu; a Triton producer's ops are 1:1 likewise. Every variant has a discriminator in the op stream:
/// `math.sqrt` appears only in rmsnorm, `arith.maximumf` only in attention, `linalg.matmul` WITH
/// `indexing_maps` only in a matmul (attention's `matmul_plain` omits it), and the lm-head extraction is
/// the only program with no compute op at all. Elementwise `Mul` and `ScalarMul` are the one pair that
/// needs a tiebreak, and it already exists: `program_scalarmul_scale` tells them apart by whether an
/// operand is a `tensor.splat`.
///
/// ⭐ WHAT IT IS ACTUALLY FOR, and it is narrower: a REFUSAL BY NAME cannot be derived from ops a producer
/// never emits. KTIR has no gelu op, so `QuickGelu`/`GeluErf` — which have no DDL primitive and must be
/// refused by name rather than silently lowered as the nearest one — arrive only as a stated kind. And the
/// label is currently UNCHECKED: `elementwise` emits its op-func string from the kind alone and never
/// inspects the ops, so a mislabelled program lowers to the wrong descriptor silently. The crate already
/// has the antidote pattern three times over (`program_rmsnorm_eps`, `program_scalarmul_scale`,
/// `program_score_scale`, the last of which `attn_at` uses to PROVE a caller's value matches the
/// program's). Applying it here — deriving the kind and refusing a disagreement — is the outstanding work,
/// and until then a producer that could state the fact from its own ops is being asked to classify by hand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Program {
    /// A pointwise function over whole rows.
    Elementwise(Elementwise),
    /// `silu(gate) · up` as ONE device primitive — the AIU has `OpFunc::Silu` and KTIR's `math.*` set
    /// does not, so the producer writes silu longhand and this recognises it.
    SiluMul,
    RmsNorm,
    ScalarMul,
    Matmul,
    /// The last-row extraction half of an lm-head tail — see [`KtirNode::node_out_tid`].
    LmLast,
    Rope,
    Attn,
    /// A 2-D BLOCK TRANSPOSE of a whole tensor — `[mb, out]` → `[out, mb]`, one
    /// `interslicetranspose_fp16` on the PT unit. See
    /// [`transpose`](crate::emit::lower_ktir_to_superdsc::transpose).
    ///
    /// ⭐ IT IS HERE BECAUSE A TRANSPOSE OF A COMPUTED VALUE HAS NOWHERE ELSE TO GO. A `tt.trans` on a
    /// value the producer LOADED can be folded into the load's access-tile order and never becomes an
    /// op; a `tt.trans` on a value the program COMPUTED (a RoPE'd K) has no access tile to fold into,
    /// so the reorder has to be a descriptor. `OpFunc::Transpose` is the device primitive for it, and
    /// [`assemble_transpose`](crate::emit::assemble_transpose) has been written for it since the
    /// SubtileIR path — with no entry point, so no producer could ask.
    ///
    /// ⛔⛔⛔ AND IT IS THE STICK-ALIGNED CASE ONLY. Both extents must be whole 64-element fp16 sticks.
    /// The transpose of a per-step `[1, 64]` key is exactly the shape that deleted this file's former
    /// generic op walk (see the module header of
    /// [`lower_ktir_to_superdsc`](crate::emit::lower_ktir_to_superdsc)): a 1-element stick extent is
    /// not a whole stick, the dxp scheduler refuses it, and the on-card ReStickify that was asked for
    /// it anyway wrote the SECOND Kᵀ stick wrong and garbled decode past 64 tokens. The lowering
    /// REFUSES that shape by name rather than emitting it; the shipped attention's answer — pad the
    /// row count to a stick multiple and loop whole 64×64 tiles — is named in the refusal.
    Transpose,
}

/// One buffer of a bundle, as the CALLER numbers it.
///
/// ⭐ OPAQUE TO THIS CRATE, AND DELIBERATELY SO. The lowering does exactly two things with a
/// `BufferId`: renders the descriptor operand name for it ([`act_name`](crate::place::act_name)) and
/// looks it up in the [`BundleLayout`](crate::placement::BundleLayout) the caller supplies. It never
/// asks what the buffer holds — extents, strides and element format all come from the program's own
/// `ktdp.construct_memory_view`. So the producer's tensor type needs no trait here: it numbers its
/// buffers, and the numbers are keys.
///
/// ⛔ A NEWTYPE BECAUSE THE ALTERNATIVE HAS BITTEN THIS TREE. It was a bare `Vec<usize>` documented
/// as "the tensor the i-th parameter points at" IN A PRODUCER'S OWN VOCABULARY — a word this crate has
/// no business knowing — and an integer indistinguishable from the parameter indices, row counts and
/// registry slots beside it. Conflating two `u32` meanings is how `kv-launch-slot`-vs-`kv-row` and
/// `mq`-as-two-widths both happened; the type makes the wrong one not compile.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BufferId(u32);

impl BufferId {
    /// Number a buffer. The value means nothing to this crate beyond being distinct per buffer within
    /// one bundle, and matching the key the caller's `BundleLayout` places it under.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The caller's number, for the two crate-side uses: the operand name and the layout lookup.
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for BufferId {
    /// Bare, so a message reads `t{id}` the way every descriptor-facing diagnostic here already does.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The ACTIVE KV sweep extent baked into a decode-attention bundle — one `sk_bucket` ladder rung.
///
/// The KV STORAGE is always the full `cap`; a bundle baked at a smaller `ActiveCap` sweeps only its
/// first [`resolve`](ActiveCap::resolve) slots (O(active) attention over the SAME resident cache). A
/// LADDER of decode bundles — one per rung — lets the runtime pick the smallest sweep ≥ the live KV
/// length while sharing one resident weights+KV. A newtype so the swept extent is never confused with
/// the storage `cap`, the query count `mq`, or a bare token count floating around as `u32`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ActiveCap(u32);

impl ActiveCap {
    /// Sweep the full storage cap — the ceiling rung (ladder disabled). Emits byte-identically to the
    /// pre-ladder path.
    pub const FULL: ActiveCap = ActiveCap(0);

    /// Sweep NO resident prefix at all — `nb == 0`, so the attention emits ZERO prefix blocks and the
    /// new-token block alone seeds and finalizes the online softmax.
    ///
    /// Valid ONLY for a chunk whose `start == 0`, i.e. one with no resident prefix to attend. That is
    /// every single-chunk prompt, which is the common TTFT case — and there the 4 prefix blocks are
    /// masked out in full by pmask, so they launch 308 of the 710 ops in a prefill layer (43%) to
    /// compute nothing: 12,320 launches per forward, ~4-6 ms at the measured 0.31-0.47 us per launch,
    /// plus ~1.2 ms of masked MACs.
    ///
    /// Correct because the new-token block is already the `first` block: with `first` it writes
    /// `run_m`/`run_l`/`run_o` directly rather than folding, and the finalize is `out = run_o / run_l`
    /// — which for a single block is exactly `exp(s - max) / sum`. (`assemble_attn`'s note that the
    /// seed "is still followed by at least one prefix-block combine" describes the numerical ORDERING
    /// argument for seeding from the new block, not a correctness requirement for a combine to exist.)
    ///
    /// A DISTINCT sentinel is required because `FULL` is `0`: "sweep everything" and "sweep nothing"
    /// would otherwise be the same value. `u32::MAX` can never be a legitimate slot request (it is
    /// neither `<= cap` nor stick-aligned), so it cannot collide with a real rung.
    pub const NONE: ActiveCap = ActiveCap(u32::MAX);

    /// A rung requesting a swept extent of `slots` KV positions. Stick-alignment + `≤ cap` validation
    /// is applied by [`resolve`](ActiveCap::resolve) against the bundle's real storage cap.
    pub const fn new(slots: u32) -> ActiveCap {
        ActiveCap(slots)
    }

    /// The raw requested slot count (0 for [`FULL`](ActiveCap::FULL)). For serialization/display only;
    /// use [`resolve`](ActiveCap::resolve) for the concrete tile extent.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The concrete swept extent for a bundle whose storage cap is `cap` (`stick` = 64-elem tile
    /// alignment): [`FULL`](ActiveCap::FULL), or any out-of-range / mis-aligned request, resolves to
    /// the full `cap` (sweep everything); otherwise the requested extent. The ONLY place a rung becomes
    /// a tile extent.
    pub fn resolve(self, cap: u32, stick: u32) -> u32 {
        // NONE resolves to 0 ⇒ `nb = 0` ⇒ no prefix blocks at all. Checked FIRST so it can never be
        // mistaken for an out-of-range request and silently widened to the full cap.
        if self == ActiveCap::NONE {
            return 0;
        }
        if self.0 > 0 && self.0 <= cap && self.0.is_multiple_of(stick) {
            self.0
        } else {
            cap
        }
    }

    /// The INTERIOR decode ladder rungs strictly below `cap` — the KV spans where bounding the sweep is
    /// worth a separate bundle. The caller adds `cap` itself as the ceiling rung (the full-cap fallback
    /// that MUST exist so any context ≤ cap is correct). Env `SCRATCHY_SUPERDSC_DECODE_RUNGS="512,2048"`
    /// overrides the default set. Each rung is stick-aligned (up) and `< cap`; sorted ascending, deduped.
    /// Empty when `cap` is small enough that no interior rung fits (⇒ single full-cap bundle = disabled).
    pub fn decode_ladder(cap: u32) -> Vec<ActiveCap> {
        let stick = Fp16::ELEMS_PER_STICK;
        // THE LADDER, and there is no other. It used to sit behind
        // `SCRATCHY_SUPERDSC_DECODE_RUNGS`, so which rungs a bundle baked depended on the shell
        // that ran the build.
        //
        // Geometric: each rung bounds the decode sweep to ~2x the true length. The low rungs
        // (64/128/256) recover short-context tok/s; the high rungs (1024/2048) keep long context
        // bounded below the full cap. Only rungs < cap survive; `cap` is always the ceiling.
        // Cheap: N x KB body programs, one shared resident KV.
        //
        // 64 added (2026-07-28): the attention emitter blocks the resident cache in single-stick
        // (64-element) chunks (a real hardware reduce-MAX limit — see
        // reference_spyre_multirow_reduce_not_broken.md). At rung 128 (the previous floor) that is
        // ALWAYS 2 resident blocks (128/64) even at the very start of generation, when the true
        // context is far shorter — one whole block-fold's worth of matmuls + reduce/pointwise ops
        // wasted on masked-out slots every single step. A 64 rung makes nb=1 for short contexts,
        // recovering that op count through a proven-safe mechanism (a smaller sweep, not a wider
        // reduce) — unlike widening the block itself past one stick, which a real on-card test
        // showed breaks coherence (see the reverted "LX-tile block" commit).
        let base: Vec<u32> = vec![64, 128, 256, 512, 1024, 2048];
        let mut rungs: Vec<u32> = base
            .into_iter()
            .map(|r| r.next_multiple_of(stick))
            .filter(|&r| r > 0 && r < cap)
            .collect();
        rungs.sort_unstable();
        rungs.dedup();
        rungs.into_iter().map(ActiveCap::new).collect()
    }
}

/// One node's KTIR: the function, and which tensor each of its parameters carries.
#[derive(Clone)]
pub struct KtirNode {
    pub func: ktir_core::ir::IRFunction<'static>,
    /// What this program computes. See [`Program`] — carried because no reading of the IR recovers it.
    pub program: Program,
    /// Parameter order — `bindings[i]` is the buffer the i-th parameter addresses.
    ///
    /// ⛔ THE PAIRING IS CARRIED BECAUSE NO PROGRAM STATES IT. A KTIR function's parameters are
    /// `index` start addresses; nothing inside the function says which allocation each points at, and
    /// a launch binds an address per parameter. This is the one input that is a binding rather than a
    /// restatement of the IR — and it is the caller's own numbering, not this crate's ([`BufferId`]).
    pub bindings: Vec<BufferId>,
    /// The buffer holding the runtime length mask, when this node reads one.
    ///
    /// ⛔ A BINDING, LIKE [`Self::bindings`], AND FOR THE SAME REASON. The mask is SYNTHETIC — it is
    /// not a tensor of the producer's graph, so the producer numbers it itself and no program can say
    /// which of its parameters that number is. The lowering needs only that: which parameter to leave
    /// out of the K/V segment list.
    ///
    /// ⛔ IT WAS `(u32, u32)` AND THE SECOND HALF WAS THE PRODUCER'S. The pair carried the prefix
    /// CAPACITY, which this crate never read — `k.mask.map(|(t, _)| t)` threw it away at the one use
    /// site — while the CALLER read it back off this type to size a host buffer. A public input type
    /// carrying a field only its caller reads is the same leak in the other direction, and the capacity
    /// was never absent from the IR anyway: the mask parameter's own view states `[1, capacity]`, so a
    /// caller that wants it reads it from [`regions`](crate::emit::lower_ktir_to_superdsc::regions).
    pub mask: Option<BufferId>,
    /// The output tensor of the NODE this program is HALF OF, when that is not the program's own
    /// output. `None` for every program that writes its node's output itself, which is all but one.
    ///
    /// ⛔⛔⛔ ONE PROGRAM IS NOT ALWAYS ONE NODE, AND THE DESCRIPTOR NAMES ARE NAMED BY THE NODE.
    /// `ibm/main` lowered a node's whole tail in one body, so it named every descriptor it emitted
    /// `<kind>{j}_o{node.output.tensor}` — the lm-head tail's `hidden/64` extraction copies included,
    /// even though they write the reserved `LAST_HIDDEN_TID` staging rather than the logits. Split in
    /// two, the extraction became its own program whose only output IS that staging, so
    /// [`lmlast`](crate::emit::lower_ktir_to_superdsc::lmlast) named its copies
    /// `lmlast{j}_o4294967295` (`u32::MAX`) for every model — a gratuitous divergence from main's
    /// `lmlast{j}_o1128`, on the exact artifact this port is validated by byte-comparing
    /// (`dip_standalone -s`), and one that drops the node identity main's names carry.
    ///
    /// ⛔ AND IT CANNOT BE THE PROGRAM'S NAME. A program is named `<kind>_s<node index>` and MUST be
    /// (the emulator keys functions by name and the front end column-chunks one wide node into
    /// several nodes sharing ONE output tensor, so `_o{tid}` collides — MEASURED, see `KtirFunc`'s
    /// note). So the node's output tensor is stated here, by the producer that built both halves.
    ///
    /// ⭐ THIS ONE STAYS, AND IT IS A BINDING, NOT A SIDECAR. Unlike the shapes and the constants that
    /// used to sit beside it, no reading can recover this: the extraction's parameters are the staging
    /// it writes and the activation it reads, and the node's real output is neither. Every other home
    /// is worse — on the emitted op it joins the bake-plan fields that should not be in this crate's
    /// output at all, and as a parameter of the lowering it is the same `Option` one level out, since
    /// dispatch happens per BUNDLE and later than the producer that knows. So it is stated here, as a
    /// [`BufferId`] like every other buffer this type names.
    pub node_out_tid: Option<BufferId>,
}
///
/// ⭐ DELIBERATELY *NOT* A [`Program`] VARIANT, and the reason is a real cost rather than taste.
/// [`Program`] is matched EXHAUSTIVELY by consumers outside this crate (a caller's layout planner has
/// one arm per kind), so a new variant is a breaking change to every one of them for a kind that only
/// the whole-function door can reach — `linalg.reduce` never arrives as a per-`Program` node, because
/// a producer stating one node per model-graph op states the FUSED kind that contains it
/// (`Program::RmsNorm`), not the bare reduce. So the kind travels as an argument to
/// [`crate::emit::lower_ktir_to_superdsc::reduce`] and the door that reads it dispatches locally.
///
/// ⛔⛔⛔ AND `Max` IS NOT SYMMETRIC WITH `Sum`, WHICH IS WHY THIS IS A TYPE AND NOT A STRING.
/// `ir/bridge/tiled_op_sdsc_op/reduce.rs` records a MEASURED device defect at its own code: the
/// on-card reduce-MAX returns 0 — the SEED — whenever `rows > 1`, proven by an attention diagnostic,
/// and is correct only at `rows == 1`; reduce-SUM is fine multi-row. A multi-row max therefore
/// lowers, bakes, exits 0, and returns zero for every row. The entry point refuses it by name.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReduceKind {
    /// `arith.addf` as the combiner. Correct multi-row on card.
    Sum,
    /// `arith.maxnumf` as the combiner. ⛔ CORRECT ONLY AT `rows == 1` — see [`Program::Reduce`].
    Max,
}

impl ReduceKind {
    /// The `opFuncName` this kind emits. `sfp` for both.
    pub const fn op_func(self) -> &'static str {
        match self {
            ReduceKind::Sum => "sum",
            ReduceKind::Max => "max",
        }
    }
}
