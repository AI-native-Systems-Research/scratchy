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
    /// `ln(x)`. The card's own `log` opFunc (`OpFuncs::LOG_FWD`, `arch_enums.cpp`) —
    /// the on-card Gumbel noise's `g = -log(-log(u))` (front-b-noise, auditor order
    /// 2026-10-26). Reached only from the whole-function walk over a Triton
    /// program's `math.log`; a `Program::Elementwise(Log)` producer would state it
    /// directly the way an `Exp` one does.
    Log,
    /// `-x`. The card's own `neg` opFunc (`OpFuncs::NEG`, `arch_enums.cpp`) — the
    /// on-card Gumbel noise's `g = -log(-log(u))` negates twice around the inner
    /// log (front-b-noise, auditor order 2026-10-26). Reached from the
    /// whole-function walk over a Triton program's `arith.negf` once the silu
    /// longhand recogniser stops claiming every negf as its own.
    Neg,
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
    /// `a == b` as 0/1 fp16. `OpFunc::Equal`, dxp `"equal"` — the vendor realizes it as `EQUAL`
    /// into a BOOL state reg then `SELECT(plus1, zero)` (`broadcast_ops.ddl:41`), so the
    /// wire-level result IS fp16 0.0/1.0, the same arity-2 broadcast-op shape as
    /// `maximum`/`minimum`. NaN compares FALSE, matching the emulator's ordered `cmpf oeq`.
    ///
    /// ⭐ IT IS BACK FOR THE SAMPLER (#813 carried it for the argmax; #926's revert took it
    /// with the rest). The Gumbel-max body is an EQUALITY SELECTION — `score == max` is the
    /// one-hot the staged ramps and index tables read the draw back out through — and no
    /// other elementwise op expresses it: `maximum` answers the larger operand, not the
    /// compare, and a subtract-then-sign trick rounds near-ties into false zeros in f16.
    Equal,
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
        Elementwise::Log,
        Elementwise::Neg,
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
        Elementwise::Equal,
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
    /// `x · rsqrt(mean(x²) + eps)` with NO gain — gemma's per-head V/K norms
    /// (`rmsnorm_unit` in the DSL). Same chain as [`Program::RmsNorm`] minus the
    /// final gamma multiply: the normalising multiply IS the output, so the
    /// program has ONE tensor parameter where RmsNorm has two.
    RmsNormUnit,
    /// `cap · tanh(x / cap)` — Gemma's final-logit soft cap, written longhand as
    /// `realdiv` by the cap, `tanh`, then `multiply` by the cap. The cap is a
    /// model constant (`final_logit_softcapping`) carried on the SubOp; the
    /// program splats it as an immediate, and [`program_tanhsoftcap_cap`] reads
    /// that splat back so the emit side resolves the `[1,1]` registry slot by
    /// value — the same contract `program_rmsnorm_eps` holds for the epsilon.
    TanhSoftCap,
    /// `out = x · w` where `w` is a LOADED `[1]`-shaped weight (gemma4
    /// `layer_scalar[layer]`) — a bound weight-source operand broadcast on
    /// both dims, the same operand mode ScalarMul gives its registry const
    /// but with the value staged by the host rather than baked. TWO tensor
    /// parameters (x and the weight) where ScalarMul has one.
    ScalarWeightMul,
    /// A RE-LAYING COPY that re-sticks a whole buffer `[r_in, c_in]` →
    /// `[r_out, c_out]`, preserving the flat element sequence (gemma4's
    /// per-head q/k/v-norm views `[1, H·D] → [H, D]` and their
    /// flatten-backs). ONE tensor parameter. It cannot be a view alias —
    /// the device layout is a function of the ROW COUNT, so two views over
    /// one buffer with different extents disagree about every element; the
    /// door lowers it to single-stick `identity` copies (the `lmlast`
    /// mechanism). See
    /// [`reshape`](crate::emit::lower_ktir_to_superdsc::reshape).
    Reshape,
    ScalarMul,
    Matmul,
    /// The token draw — `kernels/sample_fwd.py`: the FIRST-maximum index of
    /// `xm·inv_T + g·gate` over each row's `[1, PAD_V]` padded span, as that
    /// row's `[1, 64]` i32 idx stick. TWENTY-SEVEN tensor parameters — the row,
    /// its SEED quadruple's four word tables and its temperature reciprocal
    /// (the per-row operand classes: `[B, PAD_V]` f32, four `[B, 1]` si32
    /// words — request-owned alone, no row word, the P3 localization ruling —
    /// and `[B, 1]` f32), the twenty
    /// staged tables that are functions of the vocab alone (the pad law's
    /// offset row, the two-level ramp's three decreasing tables, the three
    /// increasing index tables, the ones and sixtyfour widen tables, then the
    /// noise section's own eleven — the four i32 hash/fold constants, the
    /// lane table, the fold's five f32 constants, the fp16 island's clamp),
    /// and the output stick. The body is whole-row `tt.reduce` maxima,
    /// equality compares and staged-table multiplies over generic ops, with
    /// the whole selection IN F32 (the score is a sum; f16's 10-bit mantissa
    /// would round near-ties into ties the reference never sees), the noise
    /// a wrapping-i32 counter hash through the wrap-fold and the fp16-island
    /// Gumbel (the card's only log is fp16). The door reads NO fact off this
    /// program — no epsilon, no scale, no geometry: the tables and the
    /// per-row operands ride as ordinary tensor operands — so its arm is the
    /// plain `whole_function::lower_function` route, exactly ArgMax's (#813).
    Sample,
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
    /// The chained head's embedding-row production (#603) — `table[pick]` off the lm-head's own
    /// TIED weight, read IN PLACE in its kernel slot, as three door-assembled legs: a slab
    /// gather (`gather_copy_opspec`, the MoE expert-bank precedent — one entry is one whole
    /// contiguous `[page, 64]` chunk of the slot's slab-major device layout, `page = hidden/k`;
    /// the `k` entries tile the slab), the
    /// restickify-transpose of that slab `[hidden, 64] → [64, hidden]`
    /// ([`transpose`](crate::emit::lower_ktir_to_superdsc::transpose)'s own door), and a row
    /// gather off the transposed slab (`hidden/64` one-stick entries, the ramp lane `i`
    /// naming source stick `64·i + l` — the same copy door). THREE tensor inputs plus the
    /// out, in mint order: the slab index stick `[width, 32]` (row r's lanes the LANE
    /// PATTERN `k·s_r + (i mod k)`, `s_r = pick_r >> 6`), the lane index stick
    /// `[width, 32·l_sticks]` (row r's stick b at flat index stick `r·l_sticks + b`, its
    /// lanes the LANE RAMP `64·i + (pick_r & 63)` — the transposed row's own sticks), the
    /// weight `[hidden, pad_vocab]`, and the `[width, hidden]` row out.
    ///
    /// ⭐ IT IS A PROGRAM KIND AND NOT A TRITON KERNEL BECAUSE THE READ IS NOT
    /// KERNEL-EXPRESSIBLE, AND THAT IS A CITED FACT, NOT A PREFERENCE. A descriptor gather
    /// requires contiguous rows (`strides=[D, 1]`, `test-fixtures/embedding.py`'s own contract
    /// and Triton `semantic.py`'s asserts at the definition), while the token's row in the
    /// kernel slot is lane `pick % 64` across `hidden` sticks — stride 64, never contiguous
    /// (the slab-major law, [`crate::sdsc_abstract`]'s `dev_off_stk`). And the whole-slab row
    /// a gather would need instead is `hidden · 64` elements (512 KB at hidden 4096), past
    /// every tile budget. The `Transpose` kind is the precedent class: a computed-value op
    /// with no Triton home, door-assembled, its stick law stated in its doc.
    ///
    /// ⛔ THE DECOMPOSE (`s`, `l`) IS NOT THIS OP'S. The indices arrive as STICKS this op
    /// reads — SENUINT32, one row of the `[width, ·]` sticks per request, whatever produced
    /// them (the `ChainStage` node, or the host's one-time seed). Producing them is the
    /// chain's consumption fact and lives with its producer; if the chain's realization ever
    /// changes geometry, this op must not change with it.
    ///
    /// ⛔ THE WEIGHT IS READ IN PLACE — ONE source, the lm-head's own operand. Any second
    /// staging of the embedding matrix anywhere in this op's realization is the dead hack,
    /// refused on sight. The `[width·hidden, 64]` slab scratch and the
    /// `[width·(64+1), hidden]` transposed PLACEMENT (each request's `[64, hidden]`
    /// transpose plus the sentinel's reserved zero tail — the unwritten sticks
    /// `[hidden, hidden + hidden/64)` per request, never written by any op, staged zero
    /// once at prepare) are per-step activations of the step's arena, sized at expansion —
    /// never runtime allocations, never weight-sized.
    ///
    /// `k` — the SLAB-GATHER ENTRY COUNT: the s stick's LANE PATTERN modulus
    /// (`k·s + (i mod k)`) and the slab gather's page split (`page = hidden/k`), the
    /// number that keeps the gather's per-core LX charge under the vendor budget. Stated
    /// by the lowering that builds this payload — the one derivation's spelling, called
    /// once off this op's own `hidden` — and read here by the door, never re-derived.
    EmbedRowGather { k: u32 },
    /// The chained head's pick decompose (#603) — the pick, staged into the persistent s/l
    /// placements through TWO STAGED TABLES, as THREE one-entry u32 table gathers the door
    /// assembles (`gather_copy_opspec`, the same copy door): `s_tab[pick] → s_stick`,
    /// `s_tab[pick] → s_out` (the node's fresh output — the s-value stick the host
    /// reference states, so the two ends agree on what the node produced), and
    /// `l_tab[pick] → l_stick`. FIVE tensor inputs plus the out, in mint order: the pick
    /// stick `[width, 64]` (the argmax idx stick's own class, i32 lanes, every lane of row
    /// r equal to `pick_r` — read as ONE SENUINT32 entry per request, the value is below
    /// 2^24 so the bytes are the same number), the persistent s-stick `[width, 32]`, the
    /// persistent l-stick `[width, 32·l_sticks]`, the two decompose tables, and the fresh
    /// `[width, 32]` s-value stick out.
    ///
    /// ⭐ THE TABLES ARE THE ARGMAX NINE'S OWN IDIOM, RULED — staged tables, exact by
    /// construction where fp16 arithmetic is not: the pick reaches `pad_vocab + 4160`,
    /// past fp16's RANGE entirely, and integers past 2048 are not fp16-exact anyway.
    /// `s_tab`'s row `t` is the LANE PATTERN — lane `i` stages `k·(t>>6) + (i mod k)`,
    /// clamped to the s=0 pattern `i mod k` past `pad_vocab` (the s-clamp is BUS
    /// SAFETY — the only pick past `pad_vocab` the argmax can answer is the all-NaN
    /// sentinel, and it reads slab 0's pattern harmlessly); `l_tab[t]` carries the LANE
    /// RAMP at `t & 63` (lane `i` = `64·(i mod (hidden/64)) + (t & 63)`), mapping
    /// past-`pad_vocab` picks to the SENTINEL RAMP (lanes naming the transposed placement's
    /// unwritten zero tail, word 0 = `hidden` — a real pick's word 0 is a lane `≤ 63`, so
    /// the mark never collides). `s_tab` is `[table_rows, 32]` SENUINT32 and `l_tab` is
    /// `[table_rows, 32·l_sticks]` — the payload's own stated `table_rows`, one patterned
    /// row per answerable pick, a function of `pad_vocab`, `k`, and `hidden`, staged once
    /// at prepare.
    /// ⛔ A TWO-LEVEL SPLIT IS DEAD BY ARITHMETIC, NOT PREFERENCE: `t = a·2¹² + b` ⇒
    /// `s = a·64 + (b >> 6)` needs a u32 add no door emits, and `l = t & 63` has no
    /// small-table form (any table indexed by `t` is range-sized).
    ///
    /// ⛔ THE PERSISTENT WRITES ARE GATHER DESTINATIONS, the one carrier the doors provide
    /// for u32 data (a plain u32 copy door does not exist — `reshape`'s identity copies
    /// are fp16-only), and the placement's own name is the destination's name: the KV
    /// write's identity-matmul form was for fp16 GEOMETRY, not a gather prohibition. The
    /// sticks are PLACEMENTS the device owns across steps (the KV pool's law), never node
    /// outputs — a node output with no in-graph consumer is a free slot the coloring
    /// reuses, and the body would clobber the chain's state between steps.
    ///
    /// ⛔ ONE OP PER ROW IS FORCED BY THE PICK'S OWN SHAPE, not chosen: the pick is splat
    /// per row, so a single contiguous run of `width` entries from word 0 would read row
    /// 0's lanes `width` times (every one of them `pick_0`). The door cuts one op per
    /// request, each reading its row's own index stick — the KV fold's per-stick cut
    /// (`GatherScratch::copies`), which at width 1 is the single whole-buffer op
    /// byte-for-byte.
    ///
    /// `pad_vocab` — the table range's derivation, stated once at the mint from the pick's
    /// own argmax (`sample_pad_vocab`) so producer and consumer can never disagree on the
    /// range the decompose is exact over. The mint asserts the group law on it (a whole
    /// multiple of the sampler group) beside the extent; nothing derives it twice.
    ///
    /// `table_rows` — the tables' OWN row count, stated ONCE at the mint as
    /// `pad_vocab + SAMPLE_SENTINEL_OFFSET + 1` (the sentinel row included). The door's
    /// extent check reads THIS payload, never a re-derivation: this leaf crate carries no
    /// sentinel arithmetic, and the one mint that computed the extent is the one statement
    /// of it.
    ///
    /// `k` — the SLAB-GATHER ENTRY COUNT, the s table's LANE PATTERN modulus
    /// (`k·(t>>6) + (i mod k)` per row `t`, the s=0 pattern `i mod k` past `pad_vocab`)
    /// and the consumer gather's page split (`page = hidden/k`). The fill stages it and
    /// the door reads it, both off THIS payload — the mint's own stated entry count,
    /// carried here so no consumer re-derives it.
    ///
    /// `hidden` — the MODEL'S OWN hidden width, the l table's LANE RAMP modulus
    /// (`64·(i mod (hidden/64)) + (t & 63)` per row `t`) and the sentinel ramp's mark
    /// (its word 0 is `hidden` itself). Stated ONCE at the mint off the paired gather's
    /// own `hidden` payload — the same single statement `load_consts_of_ir` stages the
    /// `ChainStageTable`s from — and carried here so no consumer re-derives it: the
    /// index stick's width (`CHAIN_IDX_BLOCK`, 32 for EVERY model) is not a statement
    /// of hidden, and `lanes·64` is one model's hidden by coincidence, never a law.
    ChainStage {
        pad_vocab: u32,
        table_rows: u32,
        k: u32,
        hidden: u32,
    },
    /// `Reshape{Times(H>1)} → RmsNorm{eps, gain}/RmsNormUnit{eps} →
    /// Reshape{Times(1)}` — the per-head q/k norm SANDWICH, fused to per-head
    /// windows over the ORIGINAL tensor. The flatten-back restores exactly the
    /// split's source arrangement (same element count, row-major order, the
    /// split is `Times(H)` and the flatten `Times(1)`), so the sandwich is
    /// `rmsnorm` applied to head windows: read `t_in[:, h·D..(h+1)·D]`, norm
    /// over `[m, D]`, write `t_out`'s head window — and BOTH reshapes (true
    /// re-laying copies, 87% of gemma-4's descriptors) vanish.
    ///
    /// THREE tensor parameters in the gained form (the split's source `x`, the
    /// `gamma` row `[1, D]` shared across heads, the flatten's output `t_out`);
    /// TWO in the unit form (no gamma). Which form a program is — and the head
    /// count, width and gain convention — is derived at the door from the
    /// program's own views, the same way [`Program::Attn`]'s geometry is: `x`
    /// states `[m, H·D]`, `gamma` states `[1, D]` (or is absent), the output
    /// states `[m, H·D]` and its first access tile is one head's window
    /// `[m, D]`.
    ///
    /// ⭐ THE HEAD-WINDOW / BLOCK-ALIGNMENT LAW (why the offsets are legal):
    /// a stick-blocked `[m, H·D]` tensor places element `(r, h·D + c)` at
    /// `h·D·m + (c/64)·(m·64) + r·64 + c%64` (D is a 64-multiple — every head
    /// dim this crate serves is), and a `[m, D]` op at base offset `h·D·m`
    /// places `(r, c)` at exactly the same address — the head window is a
    /// WHOLE-BLOCK SHIFT of the op's own block, which is the one offset class
    /// `pointwise.rs`/`reduce.rs` accept for their stick-major rank-2 form.
    /// The synthetics ([`crate::place::SynthRole`] `Sq16`/`Mean`/`Meps`/`Rinv`,
    /// plus `Xn` in the gained form) are declared ONCE at the full re-laid
    /// footprints — `[H·m, D]` for the pointwise ones, `[H·m, stick]` for the
    /// reduced ones — and addressed per head through the same law, so a
    /// program's heads share one set of buffers the way the un-fused form's
    /// one norm did.
    WindowedRmsNorm {
        unit: bool,
    },
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
        // (64-element) chunks — the conservative, card-proven form of the online-softmax
        // recurrence's kv-block tiling (torch-spyre tiles over kv-blocks the same way). At rung
        // 128 (the previous floor) that is ALWAYS 2 resident blocks (128/64) even at the very
        // start of generation, when the true context is far shorter — one whole block-fold's
        // worth of matmuls + reduce/pointwise ops wasted on masked-out slots every single step.
        // A 64 rung makes nb=1 for short contexts, recovering that op count through a
        // proven-safe mechanism: a smaller sweep, not a wider reduce. (The old justification
        // called the single-stick blocking "a real hardware reduce-MAX limit" — an attribution
        // the template's own DDL refutes; see
        // [`crate::emit::lower_ktir_to_superdsc::reduce`]'s law doc. The blocking itself stays:
        // it is this emitter's proven form. Widening the block is a separate experiment, gated
        // on the same card law the wide reduce now rides.)
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
    /// several nodes sharing ONE output tensor, so `_o{tid}` collides — MEASURED, see the deleted
    /// builder's note). So the node's output tensor is stated here, by the producer that built both halves.
    ///
    /// ⭐ THIS ONE STAYS, AND IT IS A BINDING, NOT A SIDECAR. Unlike the shapes and the constants that
    /// used to sit beside it, no reading can recover this: the extraction's parameters are the staging
    /// it writes and the activation it reads, and the node's real output is neither. Every other home
    /// is worse — on the emitted op it joins the bake-plan fields that should not be in this crate's
    /// output at all, and as a parameter of the lowering it is the same `Option` one level out, since
    /// dispatch happens per BUNDLE and later than the producer that knows. So it is stated here, as a
    /// [`BufferId`] like every other buffer this type names.
    pub node_out_tid: Option<BufferId>,
    /// ⭐ THE MINT RECORD — every intermediate tid this node's lowering minted, by the SSA
    /// value that needed it. `whole_function`'s `mint_inter` consults it FIRST and replays
    /// the recorded tid when the SAME layout already declared that tid's synthetic — which is
    /// how the declare pass and the real pass (two lowerings of the same node into the same
    /// layout, [the splice memo's](crate::emit::whole_function) shared instance) land on the
    /// SAME names instead of minting fresh buffers per pass.
    ///
    /// A `Mutex` and not a `RefCell` because the splice's whole-function memo holds
    /// `KtirNode` in a process-wide `static` and requires `Sync`. The clone the memo hands
    /// each rung carries the record BY CONTENTS — see the manual `Clone` below.
    pub minted: std::sync::Mutex<std::collections::HashMap<ktir_core::ir::Ssa, u32>>,
}

impl Clone for KtirNode {
    /// Manual for ONE field: the mint record clones by CONTENTS — a fresh `Mutex` over the
    /// map's clone, NOT a shared `Arc<Mutex<…>>`. The splice's memo hands each rung its own
    /// `KtirNode`, and a shared record would let one rung's declare pass overwrite another
    /// rung's minted tids even though the two rungs lower into DIFFERENT layouts; the
    /// contents-clone at `Memo` time freezes each rung's starting record at what the shared
    /// instance had. The one safety valve is that replay only fires when the layout itself
    /// confirms the tid — [`mint_inter`](crate::emit::whole_function) checks the layout's
    /// synth map before trusting the record — so a record carried into a layout that never
    /// declared that tid mints a fresh one instead of colliding.
    fn clone(&self) -> Self {
        KtirNode {
            func: self.func,
            program: self.program,
            bindings: self.bindings.clone(),
            mask: self.mask,
            node_out_tid: self.node_out_tid,
            minted: std::sync::Mutex::new(
                self.minted
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .clone(),
            ),
        }
    }
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
/// ⭐ `Sum` AND `Max` SHARE ONE TEMPLATE — `summeanmaxexx2.ddl` — differing only in compute type
/// and the identity the template seeds internally (zero vs negInf, its own `ddl.define_constant`s
/// at `:69-72`). The old doc here claimed reduce-MAX was "correct only at `rows == 1`"; that
/// attribution (and its successor, "the conjunction `rows>1` AND width past one stick") was
/// refuted by the DDL — the one measured `mxp = 0` cell carried this crate's own row-mixing and
/// pad-lane defects, both since fixed. See [`crate::emit::lower_ktir_to_superdsc::reduce`]'s law
/// doc: the wide multi-row max lowers, and the card gate arbitrates it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReduceKind {
    /// `arith.addf` as the combiner. Card-proven multi-row at any width.
    Sum,
    /// `arith.maxnumf` as the combiner.
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
