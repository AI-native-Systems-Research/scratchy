//! Bridge 3 (`TiledOp -> SdscOp`), AttnDecode composite decomposition — ONE algorithm for any
//! query-row count `mq` (mq=1 decode, mq>1 prefill/chunked-prefill), no separate decode/prefill
//! implementation. Ported from torch-spyre's `spyre__sdpa_overrideable` (`decompositions.py:527`):
//!
//! ⛔⛔⛔⭐⭐⭐⭐⭐ THE SCORE LEG CANNOT READ NATURAL K, AND THE REASON IS ONE LINE OF THE OP LIBRARY.
//!
//! The resident Kᵀ plane looks like pure waste: natural K is what the cache write produces, a third of
//! every KV page is spent re-transposing it, and `dev_off` addresses BOTH orientations out of one
//! function. The vocabulary to ask for the other one now exists and is proven element-for-element against
//! `PagedKvPool::addr` (`StickKind::KernelNt` … `MatK::of_head_dim_nt`, and
//! `subtile/tests/kernel_nt_is_natural_k.rs` at hd 64/128/256).
//!
//! ⛔ A FIRST ATTEMPT WAS REFUSED ON CARD, AND THE REFUSAL WAS MISREAD.
//! `KernelOrient::ContractionOnStick` here (dims `[out,in]`, stick `["in"]`) with its corner at
//! `KvPlane::Knat` gives `sbf-ddc: DtException: Could not find any suitable dimension mapping`
//! (`ddl_conversion.cpp:2521`, granite-3.1-2b fp8, decode rungs 64 and 128). `bmm.ddl:22` says
//! `%slice_layout_kernel = ddl.layout(%in,%out) {is_order_fixed=true}`, which LOOKS like "a bmm kernel's
//! dim order is non-permutable". It is not that.
//!
//! ⭐⭐⭐ `ddl_conversion.cpp:2339-2356` builds THREE dim lists per dataspace and matches a different ddl
//! layout against each: `globalDims` IS `layoutDimOrder_`, while `sliceDims`/`stickDims` are derived from
//! `stickDimOrder_` against `elemInSlice = 128/wordLength/8`. The `is_order_fixed=true` one is the SLICE
//! layout — so it constrains the STICK decomposition, not the declared order. `bmm.ddl:25`'s global layout
//! is `{}`, i.e. order NOT fixed, so `layoutDimOrder_` is permutable.
//!
//! ⇒ The `[out,in]` order was never the problem. **THE SLICE DIM WAS**, and `bmm.ddl` has TWO kernel slice
//! layouts chosen by ELEMENT TYPE:
//!
//! ```text
//! 22: %slice_layout_kernel       = ddl.layout(%in,%out) {is_order_fixed=true}   // int8 / fp8 / int4
//! 23: %slice_layout_kernel_16bit = ddl.layout(%out)     {is_order_fixed=true}   // fp16
//! ```
//!
//! `%kertensor_fp16` (line 41) takes the **one-dim** `(%out)`. So an **fp16** bmm kernel's slice
//! decomposition must be exactly `["out"]` — the SCORE axis, i.e. slots. Kᵀ sticks on slots; natural K
//! sticks on `hd` = `in`, so `{out} ∩ {in} = {}` and the mapping search fails. **At fp16 the resident Kᵀ
//! plane is required**, and that is the real reason.
//!
//! ⭐⭐ THE LEVER, UNTRIED: line 22's two-dim `(%in,%out)` is what the NON-16-bit kernels take, which is why
//! `emit/mod.rs`'s `is_fp8_kernel` branch (`["in","out"]` / `stickSize_ [2,64]`) bakes. An **fp8 KV cache**
//! could therefore declare the natural-K orientation — killing this plane and halving KV bytes at once.
//!
//! ⭐⭐⭐⭐⭐ AND WHAT IBM DOES, WHICH SETTLES WHY THIS PLANE EXISTS AT ALL. Their paged K cache is ONE
//! tensor in NATURAL layout — `spyre-inference/experimental/attention_backend/paged_vector_add_target.py:64`
//! is `[pages, slots, heads, hd]` sticked on `hd`, the same layout as its `value_cache`, with no transposed
//! twin. They get away with it because their attention TILES the KV axis
//! (`spyre_hint(tiles={"max_seqlen_kv": kv_block_size})`) and transposes EACH TILE as it streams, so their
//! Kᵀ is transient per-tile scratch. They still pay a restickify for it (`dump_cost_model.py:454`).
//!
//! ⇒ **THIS PLANE IS A CACHE OF THE TRANSPOSE, not a transpose IBM avoids.** We transpose a stick-block once
//! when it is written and read it for the rest of the request; they re-transpose every tile every step. With
//! this fold sweeping the WHOLE prefix per step, their way costs ~118M elements/step at 576 tokens
//! (9 blocks x 8 heads x 4096 x 40 layers) ≈ 236 MB against a 24 ms budget. The plane buys that back.
//!
//! ⛔ SO THE DEPENDENCY IS THE SWEEP, NOT THE LAYOUT: the third plane goes away when attention becomes
//! tile-streaming (flash-style), and not before. Do not try to delete it while the fold reads the whole
//! prefix per step.
//!
//! ```python
//! expansion = num_heads // num_kvheads
//! if expansion != 1:
//!     key = key.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
//!     value = value.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
//! scores = torch.matmul(query * scale, (key * scale).transpose(-1, -2))
//! scores = scores + causal_mask
//! block_max = torch.amax(scores, dim=-1); max_running = torch.maximum(M, block_max)
//! exp_scores = torch.exp(scores - max_running.unsqueeze(-1))
//! denominator = denominator * correction + exp_scores.sum(dim=-1)
//! output = output * correction.unsqueeze(-1) + torch.matmul(exp_scores, value)
//! ```
//!
//! GQA is handled EXACTLY as torch-spyre does it: `key`/`value` are expanded to `nqh` copies
//! (one per query head) BEFORE attention, not deduped to `nkvh` — matching the reference literally,
//! and matching the pre-existing, on-hardware-proven cache-write addressing (`qh*cap*hd`, looping
//! all `nqh` heads), so there is exactly ONE cache layout convention, not two.
//!
//! torch-spyre's version tiles this over kv-blocks with an online-softmax running-max recurrence
//! (for arbitrarily long `max_seqlen_kv`). This port does the SAME — blocked, ≤stick(64)-wide per
//! reduce — not the untiled "whole active_cap in one reduce" joint form: that form is mathematically
//! equivalent on paper but re-triggers a documented, real hardware defect where reduce-MAX over more
//! than one stick of columns mis-combines partial per-tile maxima (see
//! `reference_spyre_multirow_reduce_not_broken.md`).
//!
//! BATCHED OVER HEADS (2026-07-27, matching `superdsc-batch-perf`'s hardware-proven flash decode
//! exactly): the per-head score/value MATMULS stay separate (each head needs its own K/V slice —
//! unavoidable), but every reduce/pointwise op in the online-softmax recurrence runs ONCE per block
//! across ALL `nqh` (and `mq`) rows at once (`rows=nqh·mq`, one shared buffer), not once per
//! (head,block) pair. A prior version of this file looped a per-head function once per head, so
//! EVERY reduce/pointwise op — not just the matmuls — was reissued `nqh`× per block: `nqh`× the op
//! count of the proven old code for the identical computation (the real cause of a 31→13.7 tok/s
//! regression), and `rows=mq` (mq=1 for decode) per op instead of the proven-safe `rows=nqh` shape.
//! Each head's per-head matmul writes its `[mq,width]` score/value block into ROWS `h*mq..h*mq+mq`
//! of a shared `[nqh·mq,*]` buffer (row-major); every subsequent op processes the WHOLE buffer in one
//! call — matching the original design note this file started from: "the bmm OUTPUT is nqh-major
//! [nqh,mq,*], so the ew/reduce ops run over rows=nqh·mq".
//!
//! STAGE 1 / STAGE 2 (2026-07-27/28, user-directed): STAGE 1 is decode (mq=1) — sequential-prefill-
//! only, running through this SAME tiled/batched-over-heads path — correct and fast. `pmask` (prefix
//! validity) is head-and-query-row independent, so a single-row mb-broadcast is exact for any `mq`.
//! `cmask` (new-block causal) DOES vary per query row (row r attends new-block col c iff c<=r) but
//! not per head; at mq=1 this degenerates to "repeat one row `nqh` times", identical to the SAME
//! mb-broadcast pmask uses, so decode is exact. At mq>1 (STAGE 2, batched prefill) a plain
//! mb-broadcast would read only query-row 0's causal pattern for every row (structurally runs, no
//! crash, but numerically wrong for rows other than the first). The worker (`run_prefill_batch`,
//! 2026-07-28) tiles `cmask` `[nqh·mq,mq_pad]`, one real per-head repeat, matching this file's own
//! `h*mq+qrow` row order — but staging alone did NOT close the gap: `assemble_attn_block`'s mask-add
//! read cmask via the SAME `mb_at` broadcast as pmask (one row replicated to all `rows`), silently
//! discarding every tiled row but row 0. Fixed (2026-07-28) via a `mask_bcast` param: pmask keeps the
//! broadcast read (correct, head/query-row independent); cmask now reads per-row (`In::sliced`, no
//! broadcast), lining up 1:1 with the worker's pre-tiled rows. Not yet verified on real hardware.

use super::matmul::{
    SharedKernelBmmForm, assemble_matmul_off_maybe_epilogue,
    assemble_matmul_off_phys_m_maybe_epilogue, assemble_matmul_off_phys_m_with_epilogue,
    assemble_matmul_off_with_epilogue,
};
use super::reduce::assemble_reduce_off;
use crate::addr::{Head, Idx, Nest};
use crate::emit::{
    EmittedOp, In, assemble_pointwise_broadcast_off, assemble_restickify_kt_2d, fl, rb,
};
use crate::placement::BundleLayout;
use crate::sdsc_abstract::{
    BlockCols, FeatIdx, FlatTag, KernelTag, KtTileFeats, KtTileSlots, Lanes, MaskRows, MatK, MatM,
    MatN, MatY, PerRequestRows, RowCount, RowWindow, SlotWindow, Stk,
};
use crate::superdsc_error::SuperDscError;
use crate::superdsc_opspec::{DataFormat, Df, Fp16};

/// HEAD-MAJOR (`Stk<FlatTag>`) handle for the shared, batched-over-heads online-softmax buffers —
/// the attention-correct kind (matches the old proven flash decode's `assemble_pointwise_broadcast_hm`
/// usage throughout). Matmul activation reads stay `rb()` (`Stk<RowBlockedTag>` — `assemble_matmul_off`
/// requires it by type); only the reduce/pointwise path uses this. `Stk`'s layout is annotation-only
/// (addressing is driven by NAME), so the same buffer name can be `rb()`-wrapped at a matmul call site
/// and `hm()`-wrapped at a pointwise call site with no conflict.
fn hm(name: &str) -> Stk<FlatTag> {
    fl(name, 1, BlockCols::of_one_stick(Lanes::FP16).get())
}

/// The ONE-STICK-WIDE online-softmax buffers, IN DECLARATION ORDER.
///
/// ⛔ THE ORDER IS THE ADDRESS ASSIGNMENT. `BundleLayout::synth` is a bump allocator, so moving an
/// entry moves every tensor after it and changes the emitted bundle.
pub(crate) const STICK_WIDE_BUFS: [crate::place::SynthRole; 11] = {
    use crate::place::SynthRole as R;
    [
        R::RunM,
        R::RunL,
        R::BMax,
        R::NewM,
        R::Corr,
        R::CorrSubT,
        R::BSum,
        R::LTmp,
        R::Sc,
        R::ExpB,
        R::ESubT,
    ]
};

/// The HEAD-DIM-wide output accumulators, in declaration order.
pub(crate) const HEAD_WIDE_BUFS: [crate::place::SynthRole; 2] = {
    use crate::place::SynthRole as R;
    [R::RunO, R::OTmp]
};

/// The shared, batched-over-heads online-softmax scratch/state buffer NAMES for one AttnDecode node.
struct BlockBufs {
    sc: String,
    bmax: String,
    newm: String,
    corr: String,
    corrsubt: String,
    expb: String,
    esubt: String,
    bsum: String,
    otmp: String,
    ltmp: String,
    run_m: String,
    run_l: String,
    run_o: String,
}

/// One block's contribution, folded into the shared online-softmax state via the correction
/// recurrence. `width` is this block's real column count (`stick` for a resident-cache block,
/// `mq_pad` for the new-token block). `kt_kernel`/`kt_stride` describe the K-side operand's PHYSICAL
/// tensor (`kct`, stride=`cap`, for a resident block; `new_kt`, stride=`mq_pad`, for the new block) —
/// `kt_stride` is the tensor's TRUE stride, never `width` (a per-block column slice must not lie about
/// the tensor's real row stride, or every row past the first addresses the wrong location once
/// `cap>width`). `kt_off_fn`/`v_off_fn` compute each head's own base offset into that operand.
#[allow(clippy::too_many_arguments)]
/// GQA-GROUP batching for a block's two matmul legs: one op per KV-head with `batch=gqa`, instead
/// of one op per QUERY head. `nqh` ops become `nkvh`.
///
/// This is the shape our own `matmul_opspec_off` documents as the **dxp-VALIDATED** one (`opspec.rs`,
/// l0_tethering MatMul_49): INPUT `[mb,in,y]`, OUTPUT `[mb,out,y]`, and a **BARE 2-D SHARED KERNEL**
/// `[in,out]` — with the warning that a 3-D per-batch kernel "makes dxp treat the weight as per-batch
/// → garbage". GQA is exactly the shared-kernel case: the `gqa` query heads of one group read the
/// SAME K/V slice, so the kernel genuinely is shared and the validated form applies verbatim.
///
/// It also needs no stride trickery. `in`/`out` are the 64-element stick, so `y` indexes successive
/// sticks (y-stride = 64 = hd) — precisely how `qs` (head-major, head h at `h·mq·hd`) and `sc`
/// (row h at `h·mq·width`) are already laid out. And each block's kernel slice is CONTIGUOUS: `kct`
/// is stick-major on cap, so a 64-slot block is one whole `hd·64` plane at `b·hd·stick` (see the
/// prefix-block comment below). There is no window and no gap to express.
/// The nests this block addresses through. Every extent an offset below is allowed to use comes from
/// here, so a stride cannot be written at a call site — the caller names a coordinate and the nest
/// supplies the arithmetic. `slabs` answers through the one `hd / lanes` law (`Shape::slabs_of`).
struct BlockNests {
    /// The head count this stream is WIDE in — nqh for the query side, nkvh for the kv side. Kept
    /// explicitly rather than derived from `rows/mq`, because the two are independent quantities.
    heads: u32,
    hd: u32,
    /// The chunk's REAL query rows. Typed apart from `rows`, which it equals at nqh=1.
    mq: crate::sdsc_abstract::QueryRowCount,
    /// The ROW REGIME these nests frame the shared buffers with, carrying the extent that regime
    /// means — see [`BlockRows`]: the whole-batch `nqh*mq` and the per-request `nqh` are one value
    /// each, so a per-request extent cannot be paired with the whole-batch row law or vice versa.
    rows: BlockRows,
    /// WHICH request this op set names, when the regime is per-request. Ignored otherwise.
    ///
    /// The prefix fold leaves it 0: its ops are emitted once and the RUNTIME rebases each pass onto
    /// its own request's block, so naming a request at emit would double-count. The new-token block
    /// is the opposite — it is not folded, nothing rebases it, so it is emitted once PER request and
    /// each copy names its own here. Same row law, two ways of supplying the request term, which is
    /// why it is a field and not a parameter of `head_row`.
    req: u32,
}

/// The row REGIME one attention block's ops sweep, PAIRED with the extent that regime frames — one
/// value, one type, so the mismatched pairing (a per-request `nqh` in a whole-batch slot, or the
/// whole-batch `nqh*mq` on a per-request pass) does not compile. The two extents coincide exactly at
/// `mq == 1` — every bundle the fold has ever run per-request — which is what let them share a `u32`.
#[derive(Clone, Copy)]
enum BlockRows {
    /// The whole-batch shared-buffer framing: heads stacked on the row axis, `nqh*mq` rows, head
    /// `h` starting at row `h*mq` (`HeadRequestRow`). Every bundle baked so far.
    WholeBatch(MaskRows),
    /// ONE request's framing: `nqh` head rows, head `h` at row `req*nqh + h` (`RequestHeadRow`).
    /// The runtime rebases each fold pass onto its own request's block (`fold_plan`'s
    /// intermediate-segment shift); the new-token block instead names its request via `req`.
    PerRequest(PerRequestRows),
}

impl BlockRows {
    /// The row extent this regime frames the shared buffers with.
    fn extent(self) -> u32 {
        match self {
            BlockRows::WholeBatch(rows) => rows.get(),
            BlockRows::PerRequest(rows) => rows.get(),
        }
    }
    /// The rows every reduce/pointwise op of this block sweeps, as the typed row count.
    fn swept(self) -> RowCount {
        match self {
            BlockRows::WholeBatch(rows) => RowCount::of_mask_rows(rows),
            BlockRows::PerRequest(rows) => RowCount::of_per_request_rows(rows),
        }
    }
    /// TRUE when these ops are baked for ONE request's `nqh` rows. Decides `head_row` and the
    /// matmul legs' per-request forms; the two regimes coincide at `mq == 1` (`h*1 == 0*nqh + h`),
    /// which is why the per-request form cannot move the one-request decode bundle.
    fn per_request(self) -> bool {
        matches!(self, BlockRows::PerRequest(_))
    }
    /// The regime as the bundle-level declaration the manifest carries — the ONE source the
    /// worker's intermediate-segment rebase stride derives from
    /// (`FoldRowRegime::int_rep_stride_bytes`). Read off the value every fold block was assembled
    /// with, so the manifest cannot claim rows the kernels do not sweep.
    fn regime(self) -> crate::sdsc_abstract::FoldRowRegime {
        match self {
            BlockRows::WholeBatch(_) => crate::sdsc_abstract::FoldRowRegime::WholeBatch,
            BlockRows::PerRequest(_) => crate::sdsc_abstract::FoldRowRegime::PerRequest,
        }
    }
}
impl BlockNests {
    /// The ROW head `h` starts at, for the head-major buffer offsets — decided by the named row
    /// laws (`RequestHeadRow` / `HeadRequestRow`), the same laws `score_rows` mints its placements
    /// from, so neither order has a second spelling anywhere.
    ///
    /// Whole-batch: `h*mq`, heads stacked on the row axis with a request minor within each head.
    /// That order is what makes a kv-head group's `gqa*mq` rows CONTIGUOUS, which is why the
    /// full-batch matmul wants it.
    /// Per-request: `h`, because the pass has already been rebased onto its request's block. One
    /// request's `nqh` rows are contiguous there, which is what makes `mb = gqa` legal.
    /// ⭐⭐⭐⭐⭐ RETURNS A TYPED ROW INDEX, NOT A NUMBER. Both branches used to hand back a bare `u32`
    /// that each caller wrapped in `Idx::<Row>` itself, and the per-request branch spelled its order out
    /// inline as `self.req * self.heads + h` — the TRANSPOSE of the head-major law the other branch calls.
    /// Both reduce to `h` at `mq == 1, req == 0`, so bs=1 could never tell them apart. Each order is now a
    /// named law that states which buffer framing it belongs to, and the row leaves here already typed, so
    /// there is no u32 in between for the two to be confused through.
    fn head_row(&self, h: u32) -> Idx<crate::addr::Row> {
        let row = match self.rows {
            // Request-major: the pass is already rebased onto one request's block, so THAT request's
            // `nqh` rows are the contiguous ones — which is what makes `mb = gqa` legal here.
            BlockRows::PerRequest(_) => {
                crate::sdsc_abstract::RequestHeadRow::of(self.req, h, self.heads).get()
            }
            // THE ONE ROW LAW, shared with the mask that says which of these rows are valid. Restating
            // `h * self.mq` here is what let the two drift.
            BlockRows::WholeBatch(_) => {
                crate::sdsc_abstract::HeadRequestRow::of(h, 0, self.mq.get()).get()
            }
        };
        Idx::<crate::addr::Row>::n(row)
    }
    fn slabs(&self) -> u32 {
        crate::addr::Shape::<0, 0, 0, 0>::slabs_of(self.hd, crate::superdsc_opspec::Df::Fp16).get()
    }
    /// Head-major `[nqh*mq, hd]` — the online-softmax buffers. Head `h`, slab `s`.
    /// The online-softmax buffers are rank-2 `[rows, hd]` with the heads stacked ON THE ROW AXIS, so
    /// head `h` begins at ROW `h*mq`. Describing them as rank-3 `[nqh, mq, hd]` gives the head a stride
    /// of one stick instead of `mq` sticks — the law is right, the description was wrong. Naming the
    /// row is a coordinate; the stick plane and the row stride both come from the nest.
    fn head_major(&self, h: u32, s: u32) -> crate::addr::DevOff {
        crate::addr::Nest::new(
            &["row", "feat"],
            &[self.rows.extent(), self.hd],
            crate::superdsc_opspec::Df::Fp16,
        )
        .view()
        .at(self.head_row(h))
        .slab(s)
        .dev()
    }
    /// A per-head row block of a `[rows, width]` score/prob buffer, as the buffer's own PLACEMENT:
    /// base offset and head pitch, both minted by the regime's row law
    /// ([`OperandPlacement::of_request_major_rows`] / [`OperandPlacement::of_head_major_rows`]).
    /// The score/probability buffer is rank-2 `[rows, width]` with heads on the row axis, so head `h`
    /// begins at ROW `h*mq` — the same framing as the online-softmax buffers it feeds.
    ///
    /// `width` is a PARAMETER, not a field: only a block that HAS a score buffer can answer this
    /// question, and it answers it with the same typed width its matmul legs declare. The finalize
    /// nests have no score block, so they simply cannot call this — no filler value exists.
    ///
    /// This CHANGES the emitted address when `width > 64`. The previous `h*mq*width` treats the buffer
    /// as row-major, but a stick-blocked `[rows, width]` steps a row by one stick, not by its width, so
    /// the two part exactly when a block stops being one stick wide — `mq_pad == 128`, the long prefill
    /// rungs. Same class as the V cache block stride: correct for every shape that had ever run.
    fn score_rows(&self, h: u32, width: BlockCols) -> crate::sdsc_abstract::OperandPlacement {
        match self.rows {
            BlockRows::PerRequest(rows) => {
                crate::sdsc_abstract::OperandPlacement::of_request_major_rows(
                    self.req, h, rows, width,
                )
            }
            BlockRows::WholeBatch(rows) => {
                crate::sdsc_abstract::OperandPlacement::of_head_major_rows(h, self.mq, rows, width)
            }
        }
    }
    /// The token stream `[mq, nqh*hd]` — head `h`, slab `s`, as the stream's own PLACEMENT: the law
    /// ([`OperandPlacement::of_token_stream`]) owns the named-axis column arithmetic and mints the
    /// plane pitch (`mq` rows) with the offset. Head `h`'s slab `s` is the COLUMN `h*hd + s*stick`,
    /// and the stick plane it lands in is the law's business, not the caller's.
    fn token_stream(&self, h: u32, s: u32) -> crate::sdsc_abstract::OperandPlacement {
        crate::sdsc_abstract::OperandPlacement::of_token_stream(self.mq, self.heads, self.hd, h, s)
    }
    /// The same stream, for a walk that contracts ONE STICK per op — see
    /// [`OperandPlacement::of_token_stream_by_slab`]. The pitch differs; the offset and the head
    /// stride do not.
    fn token_stream_by_slab(&self, h: u32, s: u32) -> crate::sdsc_abstract::OperandPlacement {
        crate::sdsc_abstract::OperandPlacement::of_token_stream_by_slab(
            self.mq,
            self.heads,
            self.hd,
            h,
            s,
            crate::superdsc_opspec::Df::Fp16,
        )
    }
}

#[derive(Clone, Copy)]
struct BatchedKv {
    nkvh: u32,
    /// The score leg batches only when its CONTRACTION is one stick — see [`OneStickContraction`].
    score: Option<OneStickContraction>,
    value: Option<()>,
}

/// ⭐⭐⭐ A WITNESS THAT A BATCHED LEG'S CONTRACTION IS ONE STICK — the shared-kernel batched form's
/// REAL precondition, as a value that only the contraction extent can produce.
///
/// ⛔ THIS WAS A `bool` COMPUTED AT THE CALL SITE, and a bool is exactly what this file keeps getting
/// wrong: `let one_stick_contraction = Shape::<HD>::slabs(..) == 1` reads as a head-dim test, invites
/// being re-spelled as `HD == 64` by the next person, and can be handed to the wrong leg. The two legs
/// are NOT alike — the VALUE leg contracts the kv window, one stick by construction, while the SCORE
/// leg contracts the HEAD DIM and is `hd/lanes` sticks — so "may this leg batch" is a property of
/// WHICH AXIS IT CONTRACTS, not of the model.
///
/// `matmul_opspec_off`'s own doc pins the dxp-validated shared-kernel shape as the one where `in`/`out`
/// ARE the stick, and states the consequence as `y-stride = 64 = hd` — an hd=64 identity written into
/// the justification of the form. This type is that identity made checkable: the only mint takes the
/// contraction's [`MatK`], so a leg cannot claim the property without naming the extent it contracts.
#[derive(Clone, Copy)]
struct OneStickContraction(());

impl OneStickContraction {
    /// The ONE door. `Some` exactly when the contraction is a single stick at this operand format.
    ///
    /// ⏭ The score leg's contraction becomes one stick at every head dim once it is SPLIT BY SLAB
    /// (`nslab` ops of one stick each, accumulating), at which point this witness is always available
    /// and the per-head score arm can go. It is not a head-dim gate: it is the precondition the split
    /// would satisfy.
    #[allow(dead_code)]
    fn of_contraction(k: MatK, df: Df) -> Option<OneStickContraction> {
        (k.get() == df.elems_per_stick()).then_some(OneStickContraction(()))
    }

    /// ⭐⭐⭐ THE SPLIT SATISFIES THE PRECONDITION BY CONSTRUCTION, AT EVERY HEAD DIM. A slab IS one
    /// stick ([`MatK::of_head_slab`]), so a leg that contracts one slab per op has a one-stick
    /// contraction whatever `hd` is — there is nothing left to test, which is why this door takes no
    /// arguments. This is what replaced the head-dim gate: not a widened test, an eliminated one.
    fn by_slab_split() -> OneStickContraction {
        OneStickContraction(())
    }
}

/// ⭐⭐⭐ WHICH SCORE-LEG ARM, DECIDED BY **OP COUNT** — a COST question, deliberately a different
/// type from [`OneStickContraction`], which is the LEGALITY question.
///
/// ⛔ THESE TWO WERE ABOUT TO SHARE ONE `Option`, and they are not the same fact. Since the
/// contraction is slab-split, BOTH arms are correct at every head dim; what differs is how many ops
/// they cost per fold block:
///
/// | hd  | nslab | per-head (`nqh`) | slab-split (`nkvh*nslab`) |
/// |-----|-------|------------------|---------------------------|
/// | 64  | 1     | 32               | **8**                     |
/// | 128 | 2     | 32               | **16**                    |
/// | 256 | 4     | 32               | 32 (tie)                  |
/// | 512 | 8     | 32               | 64 — a **LOSS**           |
///
/// So making the split unconditional would REGRESS gemma-4's full-attention layers (hd=512), which
/// is the head dim this whole branch exists to make compilable. Choosing by cost keeps the win at
/// hd≤128 without paying for it at 512.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ScoreArm {
    /// One op per (kv head, slab) — the `y`-batched shared-kernel form.
    SlabSplitBatched,
    /// One op per query head, contracting the WHOLE head dim. Cheaper once `nkvh*nslab > nqh`.
    PerHead,
}

impl ScoreArm {
    /// The cheaper arm at these extents. Ties go to the batched form: same op count, but fewer
    /// distinct kernels for the scheduler to place.
    ///
    /// ⛔⛔⛔⭐⭐⭐⭐⭐ A PROOF THAT USED TO LIVE HERE WAS **WRONG**, AND IT IS WORTH KEEPING THE
    /// CORRECTION VISIBLE BECAUSE IT COST A DAY.
    ///
    /// The claim was: `nslab > 1` is a LEGALITY limit. A `y`-batched op gets one stride per operand
    /// from its declared order, so after a slab split — where BOTH operands' last axis is one stick —
    /// the two are assigned the SAME y-stride, while their real head strides differ by `nslab`
    /// (`qs` token stream heads `mq*hd` apart, `sc` head-major heads `mq*stick` apart). One `mb_dev`
    /// cannot be both `mq*nslab` and `mq`, therefore CONTRADICTION.
    ///
    /// ⭐ THE ERROR: "one `mb_dev`" was a property of the BUILDER, not of the machine. It read
    /// `phys_mb.unwrap_or(m)` once and used it for both operand checks. Each operand's placement
    /// already carried its own pitch; the pitch simply never travelled with the stride. With
    /// [`crate::sdsc_abstract::BatchStrides`] carrying `a_pitch`/`o_pitch` — and, the half that was
    /// missing entirely, with the OUTPUT declaring its own `device_extent` instead of merely being
    /// checked against one — both relations hold at once and the form is legal at every head dim.
    ///
    /// ⛔ WHAT IS STILL TRUE, MEASURED TWICE: the contraction must be ONE STICK. An UNSPLIT batched
    /// score passes both stride checks at mq=1/hd=128 and is incoherent inside dxp — the original note
    /// recorded it, and I reproduced it (degenerate "The history" output at a FASTER ITL, 78.1 vs 83.8,
    /// which is the tell: less work, done wrong). So the slab split is REQUIRED, not optional.
    ///
    /// Which leaves `nslab > 1` as a pure COST question: `nkvh*nslab` against `nqh`.
    fn choose(nqh: u32, nkvh: u32, nslab: u32) -> ScoreArm {
        // The slab split satisfies dxp's one-stick contraction limit and, now that each operand
        // declares its own pitch, both stride relations too — at every head dim. So this is cost only.
        if nkvh.saturating_mul(nslab) <= nqh {
            ScoreArm::SlabSplitBatched
        } else {
            ScoreArm::PerHead
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// ⭐⭐⭐⭐⭐ WHETHER THIS ATTENTION BLOCK **SEEDS** THE ONLINE SOFTMAX OR **FOLDS ONTO** AN EXISTING SEED.
///
/// ⛔ THIS WAS A `bool` NAMED `first`, eleven arguments into a twenty-argument call. The two states are not
/// interchangeable and the difference is not a fault: a SEED writes the block's own max/sum/output straight
/// into `run_m`/`run_l`/`run_o`, so a block that seeds when it should fold DISCARDS every page folded before
/// it, and a block that folds when it should seed folds onto `-inf` (`-inf - -inf` = NaN).
///
/// ⭐ AND IT IS THE INVARIANT THE FOLD PATH ONLY CLAIMED IN PROSE. This file says the prefix passes fold
/// "on top of the new-block seed above (never `first` now)" — a comment, with nothing enforcing it. With a
/// role type the claim is carried by the call: the fold site passes [`Self::FoldsOntoSeed`] and there is no
/// way for it to seed by getting a boolean backwards. That matters most for the DEEPEST row, which has the
/// most passes to lose, and losing them looks exactly like the card's remaining failure.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockSeed {
    /// This block writes `run_m`/`run_l`/`run_o` directly — the batched-decode new-token block, whose
    /// diagonal mask gives every row exactly one valid column and therefore a FINITE seed max.
    SeedsState,
    /// This block combines into an already-seeded running state. Every prefix/paged fold pass.
    FoldsOntoSeed,
}

impl BlockSeed {
    /// The one predicate the emission body asks. Named so `if seed.seeds()` reads as the question it is.
    pub const fn seeds(self) -> bool {
        matches!(self, BlockSeed::SeedsState)
    }
}

/// The BYTE INTERVAL one op touches in each operand, as `(buffer, is_input, lo, hi)`.
///
/// `startAddressCoreCorelet_.data_` is the emitter's own output — the byte address core `c` begins at
/// — so this reads the DESCRIPTOR, not a model of it. The value is TAGGED (`slot << 34 | offset`), and
/// the tag is masked off here: comparing raw values compares the operand SLOT, which differs between a
/// producer and a consumer by definition.
///
/// ⛔ A BASE ALONE IS NOT A FOOTPRINT. An op with FEWER, LARGER per-core blocks has lower bases than
/// one with many small blocks over the same bytes, so comparing bases makes a legitimate consumer look
/// out of range. MEASURED: a base-only version of this check false-fired on `expb` at hd=64 by 384
/// bytes — the SAME margin as the real defect it was written to catch, so bases alone cannot tell them
/// apart at all.
///
/// ⛔ AND THE SPAN IS NOT RE-DERIVED HERE EITHER. A second version folded the per-core
/// `dataStageParam_["0"].el_` over the node's `layoutDimOrder_`, which cannot see `Scale` — so a
/// BROADCAST operand was counted at the output's full width and this lock refused a CORRECT op
/// (`ocorr` reading one-stick `corr` across an `hd`-wide output) for one stick of overrun.
/// [`materialized_bytes`] is the one place that knows a collapsed axis occupies a single stick, and
/// `ArgBinding::footprint_bytes` carries its answer. The interval is returned WHOLE rather than as
/// `(bases, span)` so no caller can pair the operand-wide footprint with each per-core base and
/// reach past the operand — which the first version of this very refactor did.
fn per_core_blocks(e: &EmittedOp) -> Vec<(String, bool, u64, u64)> {
    const TAG_SHIFT: u32 = 34;
    let mut out = Vec::new();
    for dsc in e.dsc().dscs_.iter().flat_map(|m| m.values()) {
        for (slot, node) in dsc.scheduleTree_.iter().enumerate() {
            let Some(b) = e.arg_bindings.get(slot) else {
                continue;
            };
            let addrs: Vec<u64> = node
                .startAddressCoreCorelet_
                .data_
                .values()
                .filter_map(|s| s.trim().parse::<u64>().ok())
                .map(|raw| raw & ((1u64 << TAG_SHIFT) - 1))
                .collect();
            if let Some(&base) = addrs.iter().min() {
                // ⭐ THE SPAN IS THE EMITTER'S OWN `footprint_bytes`, NOT A RE-DERIVATION. This loop
                // used to fold the per-core `el_` over the node's `layoutDimOrder_`, which cannot see
                // `Scale` — so a BROADCAST operand (a one-stick `corr` read across an `hd`-wide
                // output) was counted at the output's full width and this lock refused a correct op
                // for reading one stick past the buffer. `materialized_bytes` is the one place that
                // knows a collapsed axis occupies one stick, and the binding carries its answer.
                //
                // The interval is [min(core address), + footprint): per-core bases are a partition of
                // the operand's own footprint, so their union is exactly that, and the coverage
                // question this lock asks is about the union — never about one core's slice.
                // ⭐⭐⭐ ONE INTERVAL PER BLOCK THE OPERAND REALLY TOUCHES. A full-head-width write on a
                // stick-blocked buffer is `nslab` blocks a whole `rows*stick` plane apart, so crediting
                // `[base, base + footprint)` claimed a contiguous region the op does not write — and the
                // lock then refused the correct collapsed value leg for a gap covered by that op's own
                // later block. See `TouchedBlocks` (and ⛔ NOT a bounding span, which would overstate
                // the write and let real gaps through).
                for off in b.touched.offsets() {
                    out.push((
                        b.buffer.clone(),
                        b.is_input,
                        base + off,
                        base + off + b.touched.run_bytes,
                    ));
                }
            }
        }
    }
    out
}

/// ⛔ THE BUILD-TIME PRODUCER/CONSUMER LOCK over this node's own scratch buffers. See the call site for
/// why it reads addresses rather than op names.
fn check_internal_buffers_are_read_where_written(
    ops: &[EmittedOp],
    bufs: &BlockBufs,
) -> Result<(), SuperDscError> {
    // The buffers this node both WRITES and READS. `out` is the hand-off (written, never read here) and
    // `qs`/`kct`/`vc`/the masks are inputs (read, never written), so neither has a pairing to check.
    let internal: [&str; 13] = [
        &bufs.sc,
        &bufs.bmax,
        &bufs.newm,
        &bufs.corr,
        &bufs.corrsubt,
        &bufs.expb,
        &bufs.esubt,
        &bufs.bsum,
        &bufs.otmp,
        &bufs.ltmp,
        &bufs.run_m,
        &bufs.run_l,
        &bufs.run_o,
    ];
    // Per buffer: the byte intervals WRITTEN, and each READ interval with the op that made it.
    let mut written: std::collections::BTreeMap<&str, Vec<(u64, u64)>> = Default::default();
    let mut reads: Vec<(&str, u64, u64, String)> = Vec::new();
    for e in ops {
        for (buf, is_input, lo, hi) in per_core_blocks(e) {
            let Some(&name) = internal.iter().find(|n| **n == buf) else {
                continue;
            };
            if is_input {
                reads.push((name, lo, hi, e.op_name.clone()));
            } else {
                written.entry(name).or_default().push((lo, hi));
            }
        }
    }
    // Coalesce each buffer's written intervals into maximal runs, so a read is checked against real
    // COVERAGE rather than against a min/max range that says nothing about the gaps.
    let covered: std::collections::BTreeMap<&str, Vec<(u64, u64)>> = written
        .into_iter()
        .map(|(name, mut iv)| {
            iv.sort_unstable();
            let mut merged: Vec<(u64, u64)> = Vec::with_capacity(iv.len());
            for (lo, hi) in iv {
                match merged.last_mut() {
                    Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
                    _ => merged.push((lo, hi)),
                }
            }
            (name, merged)
        })
        .collect();
    for (name, rlo, rhi, reader) in reads {
        // A buffer read but never written by this node is an ordering bug, not a span question.
        let Some(iv) = covered.get(name) else {
            return Err(SuperDscError(format!(
                "attention buffer '{name}' is READ (by {reader}) but never WRITTEN by this node — the \
                 consumer reads bytes no producer here produced."
            )));
        };
        if !iv.iter().any(|&(wlo, whi)| rlo >= wlo && rhi <= whi) {
            return Err(SuperDscError(format!(
                "attention buffer '{name}': {reader} reads bytes [{rlo}, {rhi}) which no producer in \
                 this node wrote — written coverage is {iv:?}. That is the two-framings defect: one \
                 side placing a head at `h*mq*hd` (token stream) and the other at `h*mq*stick` \
                 (head-major), which agree ONLY when a head is one stick. Both sides must take their \
                 offset from the SAME placement law."
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn assemble_attn_block(
    ops: &mut Vec<EmittedOp>,
    t: u32,
    tag: &str,
    nqh: u32,
    hd: u32,
    // The chunk's REAL query rows and the shared-buffer row extent, typed apart: `mq` and `rows`
    // sat side by side as two `u32`s while `rows == nqh*mq` equals `mq_pad` exactly up to mq=2.
    // `rows` also carries the ROW REGIME (see `BlockRows`): the whole-batch and per-request forms
    // demand their own extents, so this one parameter replaces the extent + `per_request` pair that
    // could be mismatched.
    mq: crate::sdsc_abstract::QueryRowCount,
    rows: BlockRows,
    qs: &str,
    // ⭐ THE SCORE LEG'S OWN FORM: its slab-split contraction needs the head-OUTERMOST order, the only
    // one whose `y` stride uses the operand's declared pitch. See `of_score_leg`.
    score_form: crate::ir::bridge::tiled_op_sdsc_op::matmul::SharedKernelBmmForm,
    kt_kernel: &str,
    // ⭐ WHICH WAY ROUND `kt_kernel` IS DECLARED, because this body's two callers disagree: the RESIDENT
    // prefix leg reads the natural-K plane (`[out,in]` stick `in` — contraction on the sticked axis, no
    // transpose anywhere), while the NEW-BLOCK leg reads a genuine Kᵗ scratch that `assemble_restickify_kt_2d`
    // transposed out of the activation (`[in,out]` stick `out`). The two are the SAME EXTENTS and, at one
    // stick, the same BYTES, so nothing in the shapes distinguishes them — see `sdsc_abstract::KernelOrient`.
    k_orient: crate::sdsc_abstract::KernelOrient,
    // The score kernel's declared physical column count. `KtKernelPitch` has exactly two doors — the
    // new block's slot extent and the pool page's slots — so a row count cannot reach this slot.
    kt_stride: crate::sdsc_abstract::KtKernelPitch,
    // Takes (query head, head-dim SLAB), symmetric with `v_off_fn` below and for the same reason:
    // the score leg splits its CONTRACTION by slab so that it is one stick at every head dim, and the
    // slab is a coordinate on the Kᵀ operand's own nest — applied inside the law, never added to a
    // finished address by the caller.
    kt_off_fn: impl Fn(crate::sdsc_abstract::QueryHead, u32) -> crate::addr::DevOff,
    v_kernel: &str,
    v_stride: u32,
    // Takes (query head, head-dim slab): the slab is a coordinate on the V operand's own nest,
    // so it is applied INSIDE the law rather than added to a finished address by the caller. That
    // addition was the last place in this file where a stride was composed by hand.
    v_off_fn: impl Fn(crate::sdsc_abstract::QueryHead, u32) -> crate::addr::DevOff,
    // This block's score column count, with its door naming WHICH width (one stick for every live
    // caller; `mq_pad` and the head dim are the other 64s it must not silently become).
    width: crate::sdsc_abstract::BlockCols,
    mask: &str,
    mask_off: crate::addr::DevOff,
    mask_bcast: bool,
    // ⛔⛔⛔ A ROLE, NOT A `bool`. See `BlockSeed`: the two states are "this block SEEDS the online
    // softmax" and "this block FOLDS onto an existing seed", and they were `true`/`false` on a parameter
    // eleven positions into a twenty-argument call. Inverting it does not fail to compile as a bool, and
    // the failure it produces is not a fault: a fold pass that seeds RESETS `run_m`/`run_l`/`run_o`
    // mid-accumulation, discarding every page folded before it — so the row with the MOST passes loses the
    // most, which is the deepest-row signature the card shows.
    seed: BlockSeed,
    batched: Option<BatchedKv>,
    // WHICH shared-kernel bmm form this bundle's batched score/value matmuls declare — named ONCE
    // at `assemble_attn`'s boundary (`SharedKernelBmmForm::of_attn_rows`) and threaded here, so a
    // block cannot pick a walk law its own bundle did not.
    bmm_form: SharedKernelBmmForm,
    // WHICH request, when the row regime is per-request. 0 for the fold — the runtime rebases each
    // pass, so naming one here would count it twice. The new-token block passes its own `r`.
    req: u32,
    // ONE PASS FOR THE WHOLE BATCH: give the score and value KERNELS a request axis, so row `h*mq+r`
    // reads request `r`'s own page instead of request 0's. `Some(mq)` only for the prefix fold of a
    // batched decode; `None` everywhere else, which emits exactly what shipped.
    //
    // This is what turns the fold's `pages × requests` launches into `pages` — 8 of the 12 a bs=8
    // layer paid, at the ~93 µs a launch costs whatever it carries. The runtime half is gated on
    // `OpKv::batched_requests`, and the mask's shape (one block per PAGE, every row valid on its own
    // request's history) is the same decision seen from the host side. All three move together.
    _request_axis: Option<u32>,
    bufs: &BlockBufs,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<(), SuperDscError> {
    // THE MODEL'S QUERY HEADS AS TYPES. `QueryHead::all` is the only door, so a head handed to an
    // off-closure is in range by construction — not by a bounds check, and not by an `expect`.
    let nqh_nz = std::num::NonZeroU32::new(nqh)
        .ok_or_else(|| SuperDscError("attention needs at least one query head".into()))?;

    let nests = BlockNests {
        heads: nqh,
        hd,
        mq,
        rows,
        req,
    };
    // The one predicate the two matmul legs branch on — derived FROM the row regime, so the
    // per-request forms and the per-request extent cannot disagree.
    let per_request = rows.per_request();
    // Bare extents for the `rb` shape slots only — `rb` is a name-wrapper whose dims stay `u32`
    // (outside this lock). The matmul dim run and the reduce/pointwise row-and-width run below take
    // the TYPED values.
    let (mq_n, rows_n, width_n) = (mq.get(), rows.extent(), width.get());
    let BlockBufs {
        sc,
        bmax,
        newm,
        corr,
        corrsubt,
        expb,
        esubt,
        bsum,
        otmp,
        ltmp,
        run_m,
        run_l,
        run_o,
    } = bufs;

    // Per-head score + value matmuls — unavoidably separate (each head reads its own K/V slice) —
    // each writing its `[mq,*]` block into ROWS `h*mq..h*mq+mq` of the shared buffers.
    //
    // `qs_off` FIX (2026-07-28): `qs` is declared `[mq, nqh·hd]` and WRITTEN stick-major (nqh·hd=2048
    // for granite, >64 ⇒ real stick-blocking, not degenerate): element (r, h·hd+d) lives at device
    // position h·mq·hd + r·hd + d (dev_off_stk: (c/64)·(mq·64)+r·64+(c%64), c=h·hd+d, c/64=h since
    // h·hd is stick-aligned). The READ here wraps `qs` as `rb(qs,mq,hd)` (cols=hd=64=lanes ⇒
    // DEGENERATE/flat regardless of declared rows) using a flat base offset `qs_off` — which only
    // equals the true write position when mq==1 (decode: h·hd == h·1·hd trivially). For mq>1
    // (prefill), every head but h=0 read from the wrong location, off by a factor of mq — found by
    // deriving the write/read addresses independently and confirmed by the K/V reads in this SAME
    // function, which already correctly include the width factor for their own per-head offsets
    // (e.g. the new-block K read: gqa_kv_head(h,gqa)·hd·mq_pad). Q's old `h·hd` was the decode-only
    // (mq=1) special case, carried over unchanged into the unified mq>1 path without generalizing.
    // KV-head-BATCHED score: ONE bmm over y=nkvh instead of nqh one-core ops. Every operand
    // offset below is the h=0 case of the per-head form, and each operand's `y` stride is
    // exactly the head-group stride the per-head loop walks by hand:
    //   input  qs  y-stride = mb·in  = gqa·mq·hd    == head h's `h·mq·hd` at h = kvh·gqa ✓
    //   output sc  y-stride = mb·out = gqa·mq·width == head h's `row_off·width` ✓
    //   kernel     y-stride = the DECLARED physical extent (b.kt_dev_ext), not in·out ✓
    // REQUEST-AXIS SCORE: one op per QUERY head, `y` = request, `mb` = 1. The kv-head grouping goes
    // away here — a GQA group shares a kv head's Kᵀ, but the requests inside that head do NOT share a
    // block, and `y` can only carry one of the two. Query-head-major is the one that pays: it trades
    // `nkvh` ops per pass for `nqh`, against `requests`× fewer PASSES, and a trip is ~3.4 µs where a
    // pass is a ~93 µs launch (measured: 8 ops × 8 passes → 32 ops × 1).
    //
    // `device_extent("out", PAGE_SLOTS)` because the op sweeps a 64-slot window of a whole-page
    // kernel; undeclared, `y` steps `in*64` and request 1 reads 64 slots into request 0's keys —
    // pinned by `fold_request_axis_strides.rs`, off the emitted per-core addresses.
    // ⭐ THE STRIDE GUARD BOTH FOLD-COLLAPSE ATTEMPTS NEEDED. A batched matmul lands its `(y, mb)` grid
    // on rows, and whether those are the rows `HeadRequestRow` names is arithmetic nobody was checking:
    // get it wrong and one request's scores are written onto another request's row, which is fluent,
    // wrong and silent. Stated here as a build error instead of discovered on the card.
    //
    // The request-axis form carries the REQUEST on `y` at `mb = 1`, so `y` steps one row and the HEAD is
    // carried by the op's own base offset (one op per head) — which the law accepts unconditionally.
    // The `phys_m` form claims the head on `y` with `mb = 1` swept, and its `y` really steps ONE row
    // where the law needs `mq`; that is why it garbled, and it now cannot be emitted.
    // ⭐⭐ ONE PLAIN 2-D MATMUL PER (REQUEST, QUERY HEAD) — the simplest form that can express this, and
    // the one with NOTHING TO DECLARE. `batch = 1` means there is no `y` axis, so there is no y-stride to
    // get right, no `device_extent` override, and no 3-D per-batch kernel: every address is a base
    // offset, and the pool composes the request term. The four mechanisms that broke the previous two
    // attempts are not guarded here, they are ABSENT.
    //
    // AND IT COSTS NOTHING. Today the fold runs `nkvh` ops in each of `requests` launches; this runs
    // `nqh * requests` ops in ONE. At nb=1, nkvh=8, nqh=32, mq=8 that is 512 op-executions either way —
    // the same work, no longer split across launches, and a launch is ~93 us against a trip's ~3.4.
    // The mask add FUSES directly into the score matmul's own launch in BOTH cases now:
    //  - `mask_bcast=false` (cmask): THE ONE ROW LAW comment above establishes cmask shares `sc`'s
    //    own row addressing 1:1, so `nests.score_rows(h, width)` — the SAME call `sc`'s own per-head
    //    offset already uses — gives the mask's per-head/group slice too, via `DevOff::Add`'s
    //    documented "compose a corner with a within-view step". No broadcast dim needed.
    //  - `mask_bcast=true` (pmask, head/query-row INDEPENDENT): the fused operand's `mb` dim is
    //    marked `Scale::RedNonStick` — the SAME marker `EwOperand::scale()` already gives a
    //    STANDALONE pointwise op's `In::mb_at` operand (verified: a matmul's own output layout is
    //    `["mb","out","y"]`, per `sdsc_bmm_lxopt.json`, the IDENTICAL names/order the standalone
    //    mechanism uses — "mb" means the same axis in both places). Since the value doesn't depend
    //    on which head/group is running, its offset stays `mask_off` UNCHANGED — no per-head shift
    //    (shifting a collapsed dim's address would read past the one row/block that exists).
    //
    //    The BATCHED score-matmul path below also batches `y = gqa` (MatY::of_gqa_group) — pmask is
    //    head-independent WITHIN a gqa group too (every one of that group's query heads shares the
    //    same kv head and therefore the same prefix validity), so the batch axis must ALSO be marked
    //    broadcast there, or the address derivation advances it per-y-step as if it were a real
    //    (Active) batch axis, reading past the mask's actual (unbatched) allocation.
    //
    //    ⚠️ NOT a plain `("y", RedNonStick)` entry in `broadcast_dims`: `matmul_dims` OMITS the `y`
    //    `ItDim` entirely when `batch == 1` (not merely sizes it 1 — see its own comment: "Plain 2-D
    //    matmul (batch==1): dims {mb,out,in} — NO y"), which is exactly what the unbatched per-head
    //    path below has, and what the gqa==1 degenerate case of the "batched" path collapses to.
    //    Naming "y" unconditionally shipped once already and PANICKED on that exact case (correctly
    //    — a silent skip there would have hidden it and produced garbled generation on the card
    //    instead of a local build failure). `attach_fused_epilogue`'s `broadcast_batch` flag (passed
    //    below) asks `OpSpec::batch_dim_name()` — the SAME `self.iter` `matmul_dims` built — whether
    //    "y" exists at all, so this file no longer re-derives "is this op batched" a second time in
    //    a way that could disagree with `matmul_dims`'s own answer.
    let mask_h = hm(mask);
    // `sc` as an epilogue operand: the slab partials accumulate INTO the score buffer itself, so from
    // slab 1 on this handle is both the op's output and its aux input.
    let sc_h = hm(sc);
    let nslab = nests.slabs();
    let broadcast_dims: &[(&str, crate::superdsc_opspec::Scale)] = if mask_bcast {
        &[("mb", crate::superdsc_opspec::Scale::RedNonStick)]
    } else {
        &[]
    };
    // TWO SEPARATE QUESTIONS, ASKED SEPARATELY: `b.score` is the LEGALITY witness (now unconditional —
    // the contraction is slab-split), and `ScoreArm::cheaper` is the COST choice on top of it. Folding
    // the cost into the witness would have made a hd=512 op-count REGRESSION look like a legality fact.
    if let Some(nkvh_b) = batched
        .and_then(|b| b.score.map(|_| b.nkvh))
        .filter(|&nkvh_b| ScoreArm::choose(nqh, nkvh_b, nslab) == ScoreArm::SlabSplitBatched)
    {
        let gqa = nqh / nkvh_b.max(1);
        let nkvh_nz = std::num::NonZeroU32::new(nkvh_b)
            .ok_or_else(|| SuperDscError("a batched score needs at least one kv head".into()))?;
        for kvh in crate::sdsc_abstract::KvHead::all(nkvh_nz) {
            // THE GROUP'S FIRST QUERY HEAD, FROM THE TYPE — not `kvh * gqa` spelled here.
            //
            // `KvHead::group_first_query` is the exact inverse of `KvHead::of_query` and TOTAL by the same
            // argument (`kvh < nkvh` for every constructible `KvHead`, `gqa == nqh/nkvh`, so
            // `kvh*gqa < nqh`). It replaced `QueryHead::all(nqh).nth(kvh*gqa).unwrap_or_default()`, which
            // asked for an index it had already proven good and then discharged the `Option` by falling
            // back to HEAD 0 — scoring a whole GQA group against the wrong head's keys, no fault, no shape
            // error, fluent wrong output.
            let qh0 = kvh.group_first_query(crate::addr::Gqa::new(gqa));
            let h0 = qh0.get();
            // ONE REQUEST'S ROW PER HEAD when this block is per-request: `m = 1` is the work; the
            // activation's placement carries its OWN plane packing as the pitch, declared as the
            // input's `mb` device extent (it sets the operand's declared walk and arrangement —
            // see `PhysM`). Passing `m = mq` here is what makes a whole-batch pass compute `mq`
            // rows to keep one.
            // `nests.score_rows(h0, width)` mints BOTH `sc`'s own per-group offset (unchanged below)
            // AND, added onto the block's `mask_off`, this group's mask slice — the SAME within-view
            // step, composed via `DevOff::Add` exactly as its doc says. `mask_bcast` mask reads the
            // SAME row regardless of head/group, so its address must NOT shift per group.
            let group_mask_off = if mask_bcast {
                mask_off
            } else {
                mask_off + nests.score_rows(h0, width).off()
            };
            // THE CONTRACTION IS SPLIT BY SLAB, so it is ONE STICK at every head dim — which is the
            // `y`-batched form's REAL precondition (`OneStickContraction`), the thing the head-dim gate
            // used to stand in for. Slab `s` contracts `qs`'s columns `[h*hd + s*stick, +stick)` against
            // the Kt kernel's feature rows `[s*stick, +stick)`, and the slab partials sum in `sc` itself.
            //
            // THE MASK IS ADDED ON SLAB 0 ONLY. `sc = sum_s (Q_s . Kt_s) + mask` — putting the mask on
            // every slab would add it `nslab` times, which is INERT at hd=64 (one slab) and wrong above
            // it, the exact shape of every other bug in this file.
            //
            // The kernel keeps its FULL declared `hd` rows and the slab is selected purely by OFFSET —
            // the same discipline the value leg uses for its output slabs. Declaring `stick` rows instead
            // would re-derive the stick-GROUP stride as `stick*stk` where the tensor's is `hd*stk`.
            // ⭐⭐⭐⭐⭐ ONE OP PER (KV HEAD, SLAB) — NO PER-HEAD LOOP, AT EVERY HEAD DIM.
            //
            // `nqh` score ops per fold block become `nkvh*nslab`: 32 -> 16 at granite hd=128. This is
            // the configuration I previously PROVED impossible, and the proof was wrong for a reason
            // worth stating exactly:
            //
            //   The contraction must be ONE STICK — dxp is measured-twice incoherent with a multi-stick
            //   contraction under a `y`-batch, and that limit is real.
            //   With `in` = one stick, the two operands need DIFFERENT row pitches:
            //     qs (token stream, heads `mq*hd` apart)   => pitch `mq*nslab`
            //     sc (head-major,   heads `mq*stick` apart) => pitch `mq`
            //   The builder used ONE `mb_dev` for both checks, so satisfying either broke the other. I
            //   read that as a contradiction in the FORM. It was a missing per-operand DECLARATION,
            //   which `BatchStrides::{a_pitch,o_pitch}` now carries and `batched_walk_args` now states.
            //
            // The slab partials accumulate in `sc` itself: slab 0 fuses the MASK add, each later slab
            // fuses `sc` at the same offset. `sc = sum_s (Q_s . Kt_s) + mask`, so the mask is added ONCE
            // — adding it per slab is inert at hd=64 and wrong above it.
            for s in 0..nslab {
                let (epi_h, epi_off, epi_bd, epi_bc): (
                    &Stk<FlatTag>,
                    crate::addr::DevOff,
                    &[(&str, crate::superdsc_opspec::Scale)],
                    bool,
                ) = if s == 0 {
                    (&mask_h, group_mask_off, broadcast_dims, mask_bcast)
                } else {
                    // ACCUMULATE IN PLACE: the aux operand IS `sc` at this group's own offset, so the
                    // op computes `sc += Q_s . Kt_s`. The DDL epilogue is SFP-resident with ONE store at
                    // the end, so the read of the prior partial precedes the store of the new sum.
                    (&sc_h, nests.score_rows(h0, width).off(), &[], false)
                };
                // nslab == 1 keeps the op's NAME, so granite-3.1-2b's emission does not move.
                let name = if nslab == 1 {
                    format!("attn_{tag}sc_g{}_o{t}", kvh.get())
                } else {
                    format!("attn_{tag}sc_g{}s{s}_o{t}", kvh.get())
                };
                ops.push(if per_request {
                    assemble_matmul_off_phys_m_with_epilogue(
                        &name,
                        MatM::single_row(),
                        MatN::of_kv_window(width),
                        MatK::of_head_dim(hd).with_orient(k_orient),
                        // The score leg reads the TOKEN STREAM (heads on columns, `mq*hd` apart) and writes
                        // the HEAD-MAJOR score buffer (heads on rows, `mq*stick` apart). Two different laws,
                        // both stated, so the builder can refuse a walk that matches neither.
                        MatY::of_gqa_group(
                            gqa,
                            nests.token_stream(h0, 0),
                            nests.score_rows(h0, width),
                        ),
                        bmm_form,
                        &rb(qs, mq_n, hd),
                        nests.token_stream_by_slab(h0, s),
                        &Stk::<KernelTag>::kernel(hd as usize, kt_stride.n_out_cols(), kt_kernel),
                        kt_off_fn(qh0, s),
                        &rb(sc, rows_n, width_n),
                        nests.score_rows(h0, width).off(),
                        epi_h,
                        epi_off,
                        crate::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                        epi_bd,
                        epi_bc,
                        sym_id_base,
                        layout,
                    )
                } else {
                    // ⛔ THE PITCH-DECLARING FORM, NOT THE PLAIN ONE. A one-stick `in` derives its
                    // head step as `pitch * in`, so the stream's own `mq` pitch would stride 64 where
                    // the heads are `mq*hd` apart — the build-time refusal this arm first hit. The
                    // slab law mints the pitch that makes the derived step the head stride.
                    assemble_matmul_off_phys_m_with_epilogue(
                        &name,
                        MatM::of_query_rows(mq),
                        MatN::of_kv_window(width),
                        MatK::of_head_slab(FeatIdx::SLAB_FEATS).with_orient(k_orient),
                        MatY::of_gqa_group(
                            gqa,
                            nests.token_stream_by_slab(h0, s),
                            nests.score_rows(h0, width),
                        ),
                        score_form,
                        &rb(qs, mq_n, hd),
                        nests.token_stream_by_slab(h0, s),
                        &Stk::<KernelTag>::kernel(hd as usize, kt_stride.n_out_cols(), kt_kernel),
                        kt_off_fn(qh0, s),
                        &rb(sc, rows_n, width_n),
                        nests.score_rows(h0, width).off(),
                        epi_h,
                        epi_off,
                        crate::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                        epi_bd,
                        epi_bc,
                        sym_id_base,
                        layout,
                    )
                });
            }
        }
    } else {
        for h in crate::sdsc_abstract::QueryHead::all(nqh_nz) {
            let hi = h.get();
            let qs_off = nests.token_stream(hi, 0).off();
            let head_mask_off = if mask_bcast {
                mask_off
            } else {
                mask_off + nests.score_rows(hi, width).off()
            };
            ops.push(assemble_matmul_off_with_epilogue(
                &format!("attn_{tag}sc_h{hi}_o{t}"),
                MatM::of_query_rows(mq),
                MatN::of_kv_window(width),
                MatK::of_head_dim(hd).with_orient(k_orient),
                MatY::unbatched(),
                bmm_form,
                &rb(qs, mq_n, hd),
                qs_off,
                // Slab 0: this arm contracts the WHOLE head dim in one op (`MatY::unbatched()`), so
                // there is no split and the kernel's base is the head's own feature origin.
                &Stk::<KernelTag>::kernel(hd as usize, kt_stride.n_out_cols(), kt_kernel),
                kt_off_fn(h, 0),
                &rb(sc, rows_n, width_n),
                nests.score_rows(hi, width).off(),
                &mask_h,
                head_mask_off,
                crate::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                broadcast_dims,
                false,
                sym_id_base,
                layout,
            ));
        }
    }
    // FIRST block: write the reduce-max straight into run_m (no bmax/seedm copy needed — there's no
    // prior state to combine against, so the block's own max IS the seed). Saves an "identity" op
    // per attention node (the only ever-"first" block).
    //
    // REVERTED (2026-07-28): a per-row split was tried here on the theory that reduce-MAX is broken at
    // rows>1 (per reduce_opspec_off's own doc). Direct comparison against the OLD, proven, hardware-
    // tested flash-decode (worktree superdsc-batch-perf, `attn_bmax{b}_o{t}`) shows it uses this EXACT
    // SAME pattern -- one reduce-MAX call at `rows=nqh` (>1), width=stick -- and that code is real,
    // hardware-verified, 31 tok/s. So rows>1 alone is NOT broken; the documented defect is about
    // reduce-MAX over MORE THAN ONE STICK OF COLUMNS (width>64), not row count. This per-row split was
    // based on a misapplied reading of that doc and is not needed -- confirmed harmless-but-pointless
    // (real hardware test showed zero behavioral change either way). Restored to the plain batched form.
    let (run_m_h, bmax_h) = (hm(run_m), hm(bmax));
    ops.push(assemble_reduce_off(
        &format!("attn_{tag}bmax_o{t}"),
        "max",
        rows.swept(),
        width,
        &hm(sc),
        crate::addr::DevOff::ZERO,
        if seed.seeds() { &run_m_h } else { &bmax_h },
        crate::addr::DevOff::ZERO,
        sym_id_base,
        layout,
    ));
    if seed.seeds() {
        // Seed already written directly above — nothing more to do here.
    } else {
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_{tag}newm_o{t}"),
            "maximum",
            rows.swept(),
            BlockCols::of_one_stick(Lanes::FP16),
            &[In::full(&hm(run_m)).ew(), In::full(&hm(bmax)).ew()],
            &hm(newm),
            crate::addr::DevOff::ZERO,
            sym_id_base,
            layout,
        ));
        // Separate subtract-output (corrsubt) from exp-output (corr) — an exp op must never read
        // and write the SAME buffer (a real hardware in-place-aliasing hazard, not a style choice;
        // see the old proven flash decode's fcsub->fcorr, always two distinct buffers).
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_{tag}corrsub_o{t}"),
            "subtract",
            rows.swept(),
            BlockCols::of_one_stick(Lanes::FP16),
            &[In::full(&hm(run_m)).ew(), In::full(&hm(newm)).ew()],
            &hm(corrsubt),
            crate::addr::DevOff::ZERO,
            sym_id_base,
            layout,
        ));
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_{tag}corre_o{t}"),
            "exp",
            rows.swept(),
            BlockCols::of_one_stick(Lanes::FP16),
            &[In::full(&hm(corrsubt)).ew()],
            &hm(corr),
            crate::addr::DevOff::ZERO,
            sym_id_base,
            layout,
        ));
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_{tag}mset_o{t}"),
            "identity",
            rows.swept(),
            BlockCols::of_one_stick(Lanes::FP16),
            &[In::full(&hm(newm)).ew()],
            &hm(run_m),
            crate::addr::DevOff::ZERO,
            sym_id_base,
            layout,
        ));
    }
    // expb = exp(sc - run_m) (post-update run_m). `esubt`/`expb` are distinct buffers — same
    // in-place-aliasing rule as above.
    ops.push(assemble_pointwise_broadcast_off(
        &format!("attn_{tag}esub_o{t}"),
        "subtract",
        rows.swept(),
        width,
        &[In::full(&hm(sc)).ew(), In::col(&hm(run_m)).ew()],
        &hm(esubt),
        crate::addr::DevOff::ZERO,
        sym_id_base,
        layout,
    ));
    ops.push(assemble_pointwise_broadcast_off(
        &format!("attn_{tag}ee_o{t}"),
        "exp",
        rows.swept(),
        width,
        &[In::full(&hm(esubt)).ew()],
        &hm(expb),
        crate::addr::DevOff::ZERO,
        sym_id_base,
        layout,
    ));
    // FIRST block: write bsum/ov straight into run_l/run_o (no lset/oset copy needed — same reasoning
    // as run_m above). Saves 2 more "identity" ops per attention node.
    let (run_l_h, bsum_h) = (hm(run_l), hm(bsum));
    ops.push(assemble_reduce_off(
        &format!("attn_{tag}bsum_o{t}"),
        "sum",
        rows.swept(),
        width,
        &hm(expb),
        crate::addr::DevOff::ZERO,
        if seed.seeds() { &run_l_h } else { &bsum_h },
        crate::addr::DevOff::ZERO,
        sym_id_base,
        layout,
    ));
    // ⭐⭐⭐ THE `oadd` IS THE VALUE MATMUL'S OWN EPILOGUE — `run_o = (expb @ V) + otmp` in ONE launch.
    //
    // `bmm.ddl`'s `stridedadd` adds a full-output-shaped operand to the PE result while it is still in
    // the SFP-LRF register, with one store at the end (the golden `MatMul_122` shows the 2-element
    // `computeOp_`). The online-softmax update `run_o = run_o*corr + ov` is exactly that shape once
    // `run_o*corr` is in hand: the matmul writes `run_o` DIRECTLY and adds `otmp` on the way out.
    //
    // So `ocorr` moves BEFORE the value leg (it reads the pre-update `run_o` and writes `otmp`), the
    // matmul's destination becomes `run_o` instead of `ov`, and `oadd` disappears — along with the `ov`
    // buffer and its store. `nslab` ops per fold block become zero: 8 at hd=128, 16 at 256, 32 at 512.
    //
    // ⛔ ORDER IS LOAD-BEARING. `ocorr` reads `run_o` and the fused matmul writes it, so every `ocorr`
    // must precede every value matmul of this block — which is why they are emitted here rather than
    // left where `oadd` used to live. The read is of the OLD value by construction, the same dependency
    // the separate `oadd` already had.
    // ⭐⭐⭐ AND IT IS **ONE** OP AT EVERY HEAD DIM — no slab loop, because there is no permutation to
    // undo here. `ocorr` reads head-major `run_o` and writes head-major `otmp`: the SAME shape, so the
    // whole `[nqh*mq, hd]` buffer is one contiguous rectangle and `hd` columns wide is just its width.
    //
    // ⛔ THE SLAB LOOP WAS COPIED FROM THE VALUE LEG, WHERE IT IS REQUIRED FOR A DIFFERENT REASON. A
    // BATCHED MATMUL reaches head `h` by striding `y` and derives that stride from `out`, so its `out`
    // must be one stick; a POINTWISE strides nothing — it sweeps a rectangle. `corr` rides in as
    // `In::col`, one stick broadcast across the output's `out` axis with the rows aligning (head-major
    // puts head `h` row `r` at row `h*mq+r` in BOTH buffers), which is the same pairing the finalize's
    // batched branch and `attn_esub` already ship — `esub` at a multi-stick `width`, so a wide
    // col-broadcast output is proven emission, not a new shape.
    //
    // The name is unchanged at `nslab == 1`, so granite-3.1-2b's emit does not move.
    if !seed.seeds() {
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_{tag}ocorr_o{t}"),
            "multiply",
            rows.swept(),
            BlockCols::of_head_dim(hd),
            &[
                In::sliced(&hm(run_o), nests.head_major(0, 0)).ew(),
                In::col(&hm(corr)).ew(),
            ],
            &hm(otmp),
            nests.head_major(0, 0),
            sym_id_base,
            layout,
        ));
    }
    // `Some(otmp)` = fold this block in via `stridedadd`; `None` = the seed block writes its own
    // result. The HANDLE is minted once here so the three value-matmul arms below cannot each decide
    // separately whether this block folds — they only choose WHERE in `otmp` their own slab sits.
    let fold_addend = (!seed.seeds()).then_some(otmp).map(|b| hm(b));
    // KV-head-BATCHED value bmm — the mirror of the score leg above (same y-stride argument:
    // expb's is mb·in = gqa·mq·width, ov's is mb·out = gqa·mq·hd, kernel's is declared).
    // REQUEST-AXIS VALUE — the mirror of the score leg. V is `[PAGE_SLOTS, hd]` per request and this
    // op sweeps 64 of its ROWS, so the dim to declare is `in`, not `out`.
    // ⭐ THE SAME RELATION ON THE VALUE LEG, which the score-side guard did not cover. V is
    // `[PAGE_SLOTS, hd]` per request and this op sweeps 64 of its ROWS, so the dim declared physical is
    // `in`; the step is `declared in * out` and it must equal the pool's request stride exactly as the
    // score kernel's does. Guarding one leg and not the other leaves half the fold unstated.
    {
        // ⛔ NO REQUEST-STRIDE GUARD. It compared the value kernel's per-step extent against the pool's
        // `request_stride` — "bytes between two requests' KV inside a page" — which no longer exists: a
        // page holds slots and a request is reached by its PAGE. There is nothing left to disagree.
    }
    // The value leg's mirror: plain 2-D, one per (request, query head), nothing declared.
    if let Some(nkvh_b) = batched.and_then(|b| b.value.map(|_| b.nkvh)) {
        let gqa = nqh / nkvh_b.max(1);
        let nkvh_nz = std::num::NonZeroU32::new(nkvh_b).ok_or_else(|| {
            SuperDscError("a batched value matmul needs at least one kv head".into())
        })?;
        for kvh in crate::sdsc_abstract::KvHead::all(nkvh_nz) {
            // Same total conversion as the score side above — the two must agree about which query head
            // owns a kv head's kernel, and they agree because there is only one function that says so.
            let qh0 = kvh.group_first_query(crate::addr::Gqa::new(gqa));
            let h0 = qh0.get();
            // ⭐⭐⭐ ONE STICK OF `out` PER OP, AND THAT IS WHAT MAKES THE GQA BATCHING HEAD-DIM-FREE.
            //
            // A batched matmul reaches head `h` by striding `y`, and it derives that stride as
            // `mb*out`. The accumulators are head-major `[nqh*mq, hd]`, where head `h` starts at ROW
            // `h*mq` and a row step is ONE STICK — so the pitch the buffer HAS is `mq*stick`. With
            // `out = hd` the op derives `mq*hd`, which equals it only at `hd == one stick`: that, and
            // nothing else, is what the `head_major_collapse_valid` gate on this arm was protecting.
            //
            // With `out = ONE STICK` the derived stride IS `mq*stick` — the buffer's own pitch, at
            // EVERY head dim. So the slab loop that the per-head arm below already runs is exactly
            // what this arm needed to be legal, and the gate has nothing left to protect. Dropping it
            // turns `nqh*nslab` value ops into `nkvh*nslab` — the GQA group is what `y` now carries.
            //
            // ⛔ AND NOTHING ELSE MOVES. The walk form stays whatever `of_attn_rows` already chose:
            // this is a change of the `out` EXTENT, not of the declared axis order or the split
            // policy. An earlier attempt changed the walk in the same commit and produced fluent
            // garbage — the two are independent, and only this one is required.
            let n_slab = MatN::of_head_slab(FeatIdx::SLAB_FEATS);
            for s in 0..nests.slabs() {
                let gs = |kv: u32| {
                    if nests.slabs() == 1 {
                        format!("g{kv}")
                    } else {
                        format!("g{kv}s{s}")
                    }
                };
                // Same split as the score leg: one request's row per head under `per_request`.
                ops.push(if per_request {
                    assemble_matmul_off_phys_m_maybe_epilogue(
                        &format!("attn_{tag}ov_{}_o{t}", gs(kvh.get())),
                        MatM::single_row(),
                        n_slab,
                        MatK::of_kv_window(width),
                        // The value leg reads the HEAD-MAJOR probability buffer and writes the HEAD-MAJOR
                        // accumulator. Both are `mq*stick` apart — the accumulator despite being `hd` WIDE,
                        // because its heads are separated along the ROW axis and a row step is one stick.
                        MatY::of_gqa_group(
                            gqa,
                            nests.score_rows(h0, width),
                            crate::sdsc_abstract::OperandPlacement::of_head_major_accum(
                                rows.swept(),
                                hd,
                                mq,
                                h0,
                                s,
                            ),
                        ),
                        bmm_form,
                        // `expb`'s OWN packing, not `qs`'s. The activation here is placed by
                        // `score_rows` under the per-request row law (`RequestHeadRow`), which puts
                        // adjacent heads ONE row apart — so the `y` step is a single row block of the
                        // one-stick-wide buffer, whatever the chunk's `mq`. The two quantities agree
                        // only at mq == 1; `qs`'s `mq` here put every head past the first `mq`x too far
                        // into `expb` — a pairing the placement leaves nowhere to write, since the
                        // pitch rides the same law that mints the offset. Pinned by
                        // `fold_request_axis_strides.rs` off the emitted per-core addresses.
                        &rb(expb, rows_n, width_n),
                        nests.score_rows(h0, width),
                        &Stk::<KernelTag>::kernel(v_stride as usize, hd as usize, v_kernel),
                        v_off_fn(qh0, s),
                        // ⛔ THE SAME PLACEMENT THE PER-HEAD ARM BELOW WRITES. Both arms produce this
                        // buffer, and they used to answer "where does head `h` live" differently — this
                        // one through `token_stream` (`h*mq*hd`), the other through `head_major` (row
                        // `h*mq` of a stick-blocked `[nqh*mq, hd]`, i.e. `h*mq*stick`). Equal iff `hd`
                        // is one stick, which is the only geometry that had ever taken both.
                        &rb(run_o, rows_n, hd),
                        nests.head_major(h0, s),
                        // The addend is THIS op's own output slice of `otmp` — same buffer geometry, same
                        // offset, so `attach_fused_epilogue` clones the output's shape onto it verbatim.
                        fold_addend.as_ref().map(|a| (a, nests.head_major(h0, s))),
                        crate::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                        &[],
                        false,
                        sym_id_base,
                        layout,
                    )
                } else {
                    assemble_matmul_off_maybe_epilogue(
                        &format!("attn_{tag}ov_{}_o{t}", gs(kvh.get())),
                        MatM::of_query_rows(mq),
                        n_slab,
                        MatK::of_kv_window(width),
                        // Same two laws as the per-request arm above: a head-major probability buffer read
                        // and a head-major accumulator written, both `mq*stick` apart.
                        MatY::of_gqa_group(
                            gqa,
                            nests.score_rows(h0, width),
                            crate::sdsc_abstract::OperandPlacement::of_head_major_accum(
                                rows.swept(),
                                hd,
                                mq,
                                h0,
                                s,
                            ),
                        ),
                        bmm_form,
                        &rb(expb, rows_n, width_n),
                        nests.score_rows(h0, width).off(),
                        &Stk::<KernelTag>::kernel(v_stride as usize, hd as usize, v_kernel),
                        v_off_fn(qh0, s),
                        &rb(run_o, rows_n, hd),
                        nests.head_major(h0, s),
                        fold_addend.as_ref().map(|a| (a, nests.head_major(h0, s))),
                        crate::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                        &[],
                        false,
                        sym_id_base,
                        layout,
                    )
                });
            }
        }
    } else {
        // ⭐⭐⭐⭐⭐ ONE OP PER HEAD, COVERING **EVERY SLAB**, AT EVERY HEAD DIM — no slab loop and no
        // slab axis. `nqh*nslab` value ops become `nqh`: 32 instead of 256 at hd=512.
        //
        // THE SLAB WAS NEVER AN AXIS. It is the STICK-GROUP COORDINATE of `out`. torch-spyre builds a
        // tensor's `layoutDimOrder_` from `arg.device_coordinates` — one symbolic EXPRESSION per device
        // dim — and a stick-blocked `[rows, hd]` tensor's are `(out//lanes, mb, out%lanes)`. The stick
        // group is a FUNCTION of `out`, so it iterates by itself once `out` spans the whole head dim.
        // `dev_off_stk` says the same thing from our side: `(j/stk)*(rows*stk) + i*stk + (j%stk)` walks
        // every stick group of a rank-2 view with no help.
        //
        // ⛔ WHAT MADE THE OLD SLAB LOOP NECESSARY WAS THE PITCH, NOT THE WIDTH. The comment here used
        // to say "a single `[mq, hd]` op writes elsewhere as soon as a head spans two stick groups" —
        // true of a view that declares `mq` rows, because the stick-GROUP stride it derives is then
        // `mq*stick` while the buffer's is `rows*stick`. `of_head_major_head_block` declares the
        // TENSOR's rows as the pitch, so the group stride is the buffer's own and the full-width write
        // lands. That is the same statement `phys_mb` already makes for the fold.
        //
        // The V kernel needs no slab offset either: `out` now sweeps its whole `hd`, so its own
        // out-stick-groups are walked by the same coordinate.
        // ⛔⛔⛔ ONE STICK OF `out` PER OP, SO ONE OP PER (HEAD, SLAB) — CARD-MEASURED, hd=128.
        //
        // The collapsed form this arm carried — ONE op per head with `out = hd`, the slab demoted to
        // `out`'s stick-GROUP coordinate — is incoherent on hardware above one stick. MEASURED on
        // granite-3.1-8b fp8 (hd=128, nslab=2), a 2084-token prompt whose answer is `Paris`:
        //
        //     '.\n\n.\n\n.\n'   and at other lengths  '. B. B. B W.' / 'Granite'
        //
        // rc=0, no RAS line, no abort — fluent nonsense. Bisected on card over the 37-commit range:
        // `db55367a` good, `6ba67f44` (the collapse) is the first bad, and `035b8df2` after it is only
        // the touched-bytes MODEL, so it cannot be the cause. granite-3.1-2b (hd=64) is unaffected at
        // every prompt length, because at ONE slab the two forms emit the same op.
        //
        // ⭐ THE ARGUMENT FOR COLLAPSING IS KEPT HERE BECAUSE IT IS SOUND AS FAR AS IT GOES, and
        // whoever retries this needs it: the slab really is a FUNCTION of `out`
        // (`layoutDimOrder_` for a stick-blocked `[rows, hd]` tensor is `(out//lanes, mb, out%lanes)`,
        // and `dev_off_stk` = `(j/stk)*(rows*stk) + i*stk + (j%stk)` walks every stick group by
        // itself), and `of_head_major_head_block` does declare the TENSOR's rows as the pitch, which is
        // what the old "writes elsewhere as soon as a head spans two stick groups" objection was about.
        // Both of those are true and neither is sufficient: the note on the batched arm above says the
        // same thing from the other side — "with `out = hd` the op derives `mq*hd`, which equals the
        // buffer's `mq*stick` pitch only at `hd == one stick`".
        //
        // ⏭ SO WHAT IS LEFT TO TRY is what that note already lists for its sibling: the declared axis
        // ORDER and the split policy, which `of_attn_rows` chooses and this change deliberately did not
        // touch. Retest the collapse WITH head-outermost order before concluding the form cannot serve
        // hd>64 — and on the 8b, since the 2b cannot see it.
        for h in crate::sdsc_abstract::QueryHead::all(nqh_nz) {
            let hi = h.get();
            for s in 0..nests.slabs() {
                ops.push(assemble_matmul_off_maybe_epilogue(
                    &if nests.slabs() == 1 {
                        format!("attn_{tag}ov_h{hi}_o{t}")
                    } else {
                        format!("attn_{tag}ov_h{hi}s{s}_o{t}")
                    },
                    MatM::of_query_rows(mq),
                    MatN::of_head_slab(FeatIdx::SLAB_FEATS),
                    MatK::of_kv_window(width),
                    MatY::unbatched(),
                    bmm_form,
                    &rb(expb, rows_n, width_n),
                    nests.score_rows(hi, width).off(),
                    &Stk::<KernelTag>::kernel(v_stride as usize, hd as usize, v_kernel),
                    v_off_fn(h, s),
                    &rb(run_o, rows_n, hd),
                    nests.head_major(hi, s),
                    fold_addend.as_ref().map(|a| (a, nests.head_major(hi, s))),
                    crate::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                    &[],
                    false,
                    sym_id_base,
                    layout,
                ));
            }
        }
    }
    if seed.seeds() {
        // Seed already written directly above — nothing more to do here.
    } else {
        // run_l = run_l*corr + bsum ; run_o = run_o*corr + ov. `ltmp`/`otmp` break the in-place
        // read/write alias in the multiply (mirrors the old proven flash decode's `fdtmp`).
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_{tag}lcorr_o{t}"),
            "multiply",
            rows.swept(),
            BlockCols::of_one_stick(Lanes::FP16),
            &[In::full(&hm(run_l)).ew(), In::full(&hm(corr)).ew()],
            &hm(ltmp),
            crate::addr::DevOff::ZERO,
            sym_id_base,
            layout,
        ));
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_{tag}ladd_o{t}"),
            "add",
            rows.swept(),
            BlockCols::of_one_stick(Lanes::FP16),
            &[In::full(&hm(ltmp)).ew(), In::full(&hm(bsum)).ew()],
            &hm(run_l),
            crate::addr::DevOff::ZERO,
            sym_id_base,
            layout,
        ));
        // ⭐ THE OUTPUT ACCUMULATOR IS DONE ALREADY. `ocorr` ran BEFORE the value leg (it needs the
        // pre-update `run_o`) and `oadd` is now that matmul's own `stridedadd` epilogue, so the
        // `nslab` add ops that used to live here — and the `ov` buffer they read — are gone.
    }
    Ok(())
}

/// Assemble the full AttnDecode node: ONE algorithm for any `mq`. `qs` is Q ALREADY multiplied by
/// sqrt_scale (the caller resolves that constant from the BundleLayout registry before calling
/// this). `new_k_scaled` is this step's roped new-K, ALSO already ·sqrt_scale (torch-spyre's
/// split-sqrt-scale form: `(Q*scale) @ (K*scale)^T`). `new_v` is this step's roped new-V, unscaled,
/// natural `[mq_pad, nqh·hd]` layout (already GQA-replicated to nqh columns by the caller's RoPE
/// stage, matching the resident cache's replication). `kct`/`vc` are the resident, nqh-replicated
/// caches (torch-spyre's literal GQA expand, not deduped — see this module's doc). `pmask`
/// (prefix validity) and `cmask` (new-block causal) are the worker-staged masks (see the module doc's
/// STAGE 1/2 note — `cmask` is only numerically exact at `mq==1`, STAGE 2 closes the mq>1 gap).
/// Returns the ops in emission order; the caller still runs the cache-write + post-restickify steps.
#[allow(clippy::too_many_arguments)]
pub fn assemble_attn<const NQH: u32, const NKVH: u32, const HD: u32>(
    t: u32,
    // ⛔⛔⛔ THE HEAD GEOMETRY IS NOT THREE BARE `u32` PARAMETERS ANY MORE. `nqh`/`nkvh`/`hd` arrive
    // as ONE `AttnGeometry`, whose consts the model's parsed `config.json` reached through
    // `model_geometry::with_config_attn_geometry` — so an nqh/nkvh transposition, or a head dim paired
    // with another model's head counts, is a type that does not exist rather than a signature slot
    // swap. The GQA divisibility proof rides on the type: a non-dividing pair fails the BUILD.
    geom: crate::sdsc_abstract::AttnGeometry<NQH, NKVH, HD>,
    // ⛔⛔⛔ NEITHER `mq` NOR `mq_pad` IS A BARE `u32` PARAMETER ANY MORE — and neither is the
    // shared-buffer row count derived HERE any more. The width, its pad AND the row laws arrive
    // FUSED in one `AttnBundleRows<NQH>`, minted at the bundle's parse boundary
    // (`attn_bundle_rows`): a decode width is a ladder CONST there (`Rung<MQ>`), so its pad and
    // every row extent — `MaskRows = NQH*MQ`, the mask shape's row count — are the COMPILER's
    // arithmetic (`RungRowLaws`); a prefill width is the runtime law through the same doors. So
    // this signature has no pair of row-count integers left to swap — the arrangement the `mq_pad`
    // parameter swap lived in — no way to pair a width with a pad computed from some other width,
    // and no second `nqh * mq` multiplication in this function for the mask staging to drift from.
    // The `NQH` on the carrier ties it to THIS geometry: a frame minted for another model's head
    // count is a type error at this slot. (Historically: `mq_pad` was a second `u32` two positions
    // from `mq` in a run of seven, reconciled only by a `debug_assert_eq!` the release bake
    // compiles out; `mq_pad` and `nqh*mq` are EQUAL at nqh=32/mq=2 and diverge from mq=4 up —
    // exactly the width boundary where the card's batch decode started failing.)
    bundle_rows: crate::sdsc_abstract::AttnBundleRows<NQH>,
    cap: u32,
    active_cap: u32,
    qs: &str,
    new_k_scaled: &str,
    new_v: &str,
    kct: &str,
    // The NATURAL-K plane. The streamed prefix form transposes a window of it into the one-block
    // `SynthRole::PrefixKt` scratch instead of reading the resident `kct` plane — see that role.
    knat: &str,
    vc: &str,
    pmask: &str,
    cmask: &str,
    out_id: crate::place::PlaceId,
    // TRUE when these rows are separate requests. Only the prefix mask cares: a prompt's rows share
    // one resident history, requests each have their own.
    rows_are_requests: bool,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    // Destructured ONCE, at the door. The boundary is where the swap happened (two `u32`s in a row of
    // seven); the body below is unchanged arithmetic on the same numbers.
    let (nqh, nkvh, hd) = (geom.nqh(), geom.nkvh(), geom.hd());
    let width = bundle_rows.width();

    // This function's own typed head count — same reason as the block assembler's: `QueryHead::all` is
    // the only door, so a head reaching an offset is in range by construction. Nonzero is the
    // geometry's own compile-time fact, so there is no runtime refusal to write here.
    let nqh_nz = geom.nqh_nz();
    // NEW-TOKEN block SUB-BLOCKING: the block is `mq_pad` columns wide, but every reduce here works on
    // ONE stick, so a chunk wider than 64 rows is folded as `nsub` stick-wide sub-blocks — structurally
    // the SAME loop the resident prefix already runs over its `nb` blocks, just with the causal mask
    // instead of the prefix-validity mask. `mq_pad` is `mq.div_ceil(stick)*stick` by construction, so
    // this divides exactly; at `mq_pad == stick` (every chunk ≤64 rows, and all of decode) `nsub == 1`
    // and the emit is byte-identical to the single-block form.
    //
    // Seeding is safe from sub-block 0 for EVERY row: row r attends columns 0..r, so every row sees at
    // least column 0 and no row's seed max is all-masked. (Rows in a later sub-block see sub-block 0 in
    // FULL; rows before it see nothing there — the mask already encodes both, so no per-sub-block mask
    // special-casing is needed.)
    // ⭐ THE DIVISION, NAMED. `mq_pad / stick` is a padded ROW count over a LANE count, and both are 64 at every
    // rung baked so far — so substituting one for the other, or for `hd` (128, the same as the fp8 stick on this
    // model), type-checks and yields a plausible 1. `MqPad::sub_blocks()` states the units once.
    // THE ONE SOURCE. The pad rides in `width`, fused to the `mq` it was computed from at the bundle's
    // parse boundary — a decode bundle's by `Rung<MQ>`'s compile-time arithmetic, a prefill chunk's by
    // the runtime law; there is no second derivation here to disagree with it. And there is no bare
    // `mq_pad` local either: every use below draws the ROLE it needs (`slots`/`cols`/`sub_blocks`) from this
    // one value — this function never asks the ROW question at all; the sites that used to consume `rows()`
    // were slot questions wearing the row count.
    let mq_pad_t = width.pad();
    let mq = width.mq().get();
    let nsub = mq_pad_t.sub_blocks();
    let mut ops: Vec<EmittedOp> = Vec::new();

    // The group size comes from the geometry's proof-carrying const — there is no `.max(1)` fudge
    // left, because a pair it would have fudged cannot instantiate this function.
    let gqa = geom.gqa();

    // Per KV-head (dedup, K-side — see lower_attn_node's attn_krep/cachewr_k comment): transpose this
    // step's new-K block [mq_pad,hd] -> Kᵀ_new [hd,mq_pad]. new_k_scaled is nkvh-wide now (deduped, not
    // GQA-replicated), matching kct's own resident convention — one transpose per DISTINCT kv-head,
    // not one per query head (cuts this loop nqh(32)->nkvh(8) too). The new-block score matmul below
    // maps query head h to its kv-head via the SAME gqa-dedup mapping `kct_base` already uses.
    use crate::place::SynthRole as R;
    // The op's operand SPELLING, rendered from the identity the layout is keyed by.
    let out_s = out_id.to_string();
    let out = out_s.as_str();
    let new_kt = crate::placement::syn(layout, out_id.synth(R::NewKt));
    if let Some(l) = layout {
        // `slots()`, not the bare number: this axis is the block's SLOT extent, which happens to equal its
        // padded row count because each new token occupies exactly one slot.
        l.synth(
            out_id.synth(R::NewKt),
            &[nkvh, hd, mq_pad_t.slots().slot_axis_extent()],
        );
    }
    // ONE restickify PER SUB-BLOCK, each producing a SINGLE-stick `[hd, 64]` Kᵀ written into its own
    // plane of the `[nkvh, hd, mq_pad]` scratch. Deliberately NOT one `[hd, mq_pad]` restickify: a
    // multi-stick Kᵀ RESTICKIFY is the known-bad shape — the on-card ReStickify wrote the 2nd Kᵀ stick
    // wrong, which is what garbled decode past 64 tokens, and the fix was to stop asking it to. The
    // resident `kct` is likewise multi-stick but only ever WRITTEN one stick at a time. Reading a
    // 64-column window OUT of a multi-stick Kᵀ is fine and already proven — that is what every prefix
    // block does. The offsets match that exactly: plane `j` of a `[·, hd, mq_pad]` stick-major tensor is
    // `j*hd*stick`, and slot rows `[j*stick, (j+1)*stick)` of the head-major producer are `j*stick*hd`.
    // SLAB-SPLIT (step 4b): the restickify's `[y, out=hd]` view spans the whole head-dim axis, so at
    // hd > stick it walks two stick groups of tensors whose groups are `mq*stick` (source) and
    // `mq_pad*stick` (destination) apart. Per slab the width is one stick. nslab == 1 keeps the op
    // count, the names and the offsets exactly as they are.
    // THE SLAB COUNT IS THE ONLY QUESTION THIS LOOP ASKS, so it is answered by the `hd / lanes` law
    // directly — there is no kv-side `BlockNests` to hold row/width slots the kv stream cannot
    // truthfully fill (its rows are the chunk's `mq`, not the shared-buffer extent a nest frames).
    let nslab =
        crate::addr::Shape::<0, 0, 0, 0>::slabs_of(hd, crate::superdsc_opspec::Df::Fp16).get();
    for kvh in 0..nkvh {
        for rw in mq_pad_t.row_windows() {
            let j = rw.index();
            for sl in 0..nslab {
                ops.push(assemble_restickify_kt_2d(
                    // Keep the pre-sub-blocking name at nsub==1 so those bundles stay BYTE-IDENTICAL (op
                    // names are bundle content): decode and every <=64-row chunk need no re-validation.
                    &if nsub.is_single() && nslab == 1 {
                        format!("attn_newkt{kvh}_o{t}")
                    } else if nslab == 1 {
                        format!("attn_newkt{kvh}s{j}_o{t}")
                    } else {
                        format!("attn_newkt{kvh}s{j}l{sl}_o{t}")
                    },
                    // The tile is one ROW WINDOW of new tokens by one FEATURE SLAB — quantities (3) and (4),
                    // not two lane counts: the window's rows are the tile's K rows, the slab its features.
                    KtTileSlots::of_row_window(RowWindow::ROWS),
                    KtTileFeats::of_head_slab(FeatIdx::SLAB_FEATS),
                    new_k_scaled,
                    // PER-KV-HEAD BASE = `kvh * mq * hd` (2026-07-28, CORRECTED from a briefly-committed
                    // `mq_pad` version that regressed decode). `new_k_scaled` inherits its kv-head stride from
                    // RoPE, this tensor family's producer, which writes head h row r at
                    // `rope_prefill_block_offset(r,h,mq,hd) = h*mq*hd + r*hd` -- stride `mq` (the chunk's REAL
                    // query rows), never the stick-padded `mq_pad`. At decode (mq=1) this is `kvh*hd`, exactly
                    // the long-standing original offset (correct all along); at prefill (mq=31) it is
                    // `kvh*31*hd`, where the original bare `kvh*hd` genuinely was wrong for kvh>=1. Same form
                    // as `qs_off`'s `h*mq*hd` (2aeb691a), independently confirmed decode-neutral.
                    // ... plus this sub-block's own row window, rows `[j*stick, (j+1)*stick)`. RoPE packs a
                    // (kv-head, slab) as `mq` rows of ONE stick, so a row window contributes `j*stick*stick`.
                    // The previous `j*stick*hd` is the same number only at hd == stick; above that it
                    // over-steps by `j*stick*(hd-stick)` and reads another slab's rows. It takes BOTH
                    // hd > 64 and mq > 64 to fire (below that `nsub` is 1 and `j` is only ever 0), so no rung
                    // baked so far reaches it -- but the longer prefill rungs do.
                    // SOURCE: the kv stream, packed by the REAL row count — kv head `kvh`, slab `sl`, plus
                    // this sub-block's own row window, which is `j` sticks of rows.
                    crate::addr::Nest::new(&["row", "head", "feat"], &[mq, nkvh, hd], Df::Fp16)
                        .view()
                        .at(rw.first_row().row_idx())
                        .at(Idx::<Head>::n(kvh))
                        .slab(sl)
                        .dev(),
                    &new_kt,
                    // DEST: `[nkvh, hd, mq_pad]` with the stick on the SLOT axis, so kv head `kvh` and
                    // feature `sl*stick` are coordinates and the `j`-th sub-block is slot window `j*stick`.
                    // The law places the plane; the previous `sl*mq_pad*stick` guessed it and was wrong above
                    // one stick per head.
                    // `new_kt` is `nkvh` SEPARATE `[hd, mq_pad]` blocks laid end to end, not one interleaved
                    // rank-3 tensor — a rank-3 nest would give the head a stride of one stick. The head names
                    // the BLOCK; the feature and slot window inside it are coordinates.
                    crate::addr::Nest::new(
                        &["feat", "slot"],
                        &[hd, mq_pad_t.slots().slot_axis_extent()],
                        Df::Fp16,
                    )
                    .block(kvh)
                    .at(Idx::<crate::addr::Feat>::n(FeatIdx::of_slab(sl).get()))
                    .slab(rw.index())
                    .dev(),
                    sym_id_base,
                    layout,
                ));
            }
        }
    }

    // ⭐ THE PREFIX MASK'S SHAPE, built ONCE and shared with the worker that stages it (the type lives
    // in `sdsc_abstract` precisely so both sides read one law). Blocks = one per fold pass: a pass is a
    // page when the fold spans the batch, so the block count is the pages a request can hold.
    //
    // ⛔ AND THE SWEEP MUST FIT A BLOCK. A pass sweeps `active_cap` columns; if that exceeds one
    // block, it reads into the NEXT block — another pass's validity rows — and the symptom is fluent
    // wrong output, never a fault. `Err` here makes it a `cargo build` failure instead.
    // The mask shape arrives FROM THE BOUNDARY, on the same carrier as the width: for a decode
    // rung its row count is `RungRowLaws`' const (`NQH*MQ` by the compiler), for a prefill chunk
    // the runtime law — this function derives no row extent of its own either way. COLS is a
    // const generic, so a block that is not whole sticks cannot be NAMED; the zero-head and
    // zero-row refusals live at the boundary's mint, not here.
    let mask_shape = {
        use crate::sdsc_abstract::SlotCount;
        let shape = bundle_rows.mask_shape();
        let per_page = shape.cols().get();
        if !shape.sweep_fits(SlotCount::new(active_cap)) {
            return Err(SuperDscError(format!(
                "prefix mask: one fold pass sweeps active_cap={active_cap} columns but a mask block \
                 holds only {per_page} — the pass would read the NEXT block's validity rows, i.e. \
                 another pass's history. Block the mask by active_cap, or cap the sweep at a page."
            )));
        }
        shape
    };
    let pool = crate::sdsc_abstract::PagedKvPool::new(nkvh as usize, hd as usize);
    // ⭐ ONE LAW FOR THE PRODUCT, AND ONE EXIT FOR ITS VALUE. The shape's `MaskRows` is what the
    // HOST stages the mask with, what `PrefixMaskShape` blocks by, and what every reduce and
    // pointwise op here sweeps — read off the boundary's one carrier, so this function holds no
    // `nqh * mq` multiplication at all: a decode rung's product was the COMPILER's
    // (`RungRowLaws::MASK_ROWS`), a prefill chunk's was the runtime law's, both upstream.
    // TYPED from here down: the value stays `MaskRows` and only the synth extents (a bare-`u32` API)
    // unwrap it.
    let rows = mask_shape.rows();

    // Every buffer's spelling is a rendering of `out_id.synth(role)` — the SAME value the layout
    // allocator is keyed by, so an op's operand and its reservation have one preimage.
    let syn = |r: R| crate::placement::syn(layout, out_id.synth(r));
    // ⛔⛔⛔ EVERY FIELD OF `BlockBufs` MUST BE DECLARED TO THE LAYOUT BELOW. A buffer that is
    // NAMED here but never passed to `synth` is not merely undeclared — `resolve_seg_base`
    // bump-allocates it on first reference at THAT ACCESS's size, so a per-head window reserves one
    // head where `heads` are written and the tensor aliases its neighbour. `RunM`/`RunL` were
    // dropped from that list once, and the symptom was a device that stopped responding.
    let bufs = BlockBufs {
        sc: syn(R::Sc),
        bmax: syn(R::BMax),
        newm: syn(R::NewM),
        corr: syn(R::Corr),
        corrsubt: syn(R::CorrSubT),
        expb: syn(R::ExpB),
        esubt: syn(R::ESubT),
        bsum: syn(R::BSum),
        otmp: syn(R::OTmp),
        ltmp: syn(R::LTmp),
        run_m: syn(R::RunM),
        run_l: syn(R::RunL),
        run_o: syn(R::RunO),
    };
    if let Some(l) = layout {
        // The shared buffers' extents: MaskRows rows throughout; the state buffers are ONE STICK
        // wide and the output accumulators HEAD-DIM wide — the same value only at hd == 64.
        for tnm in [
            &bufs.run_m,
            &bufs.run_l,
            &bufs.bmax,
            &bufs.newm,
            &bufs.corr,
            &bufs.corrsubt,
            &bufs.bsum,
            &bufs.ltmp,
            &bufs.sc,
            &bufs.expb,
            &bufs.esubt,
        ] {
            let _ = tnm;
        }
        // ⛔⛔⛔ ORDER AND MEMBERSHIP ARE BOTH LOAD-BEARING. `BundleLayout::synth` is a BUMP
        // allocator: the sequence of declarations IS the address assignment, and a tensor left out
        // of it is not declared at all — it falls to `resolve_seg_base`'s lazy bump, which sizes it
        // from its FIRST per-op access instead of its full footprint. That is the exact
        // under-reservation `synth` exists to kill (one head reserved where `heads` are written),
        // and it aliases whatever the allocator handed out next.
        //
        // `RunM`/`RunL` — the online-softmax running max and running sum — were dropped from this
        // list when it was rewritten from name strings to roles, and every buffer after them moved.
        // Eleven entries, in this order.
        for r in STICK_WIDE_BUFS {
            l.synth(
                out_id.synth(r),
                &[rows.get(), BlockCols::of_one_stick(Lanes::FP16).get()],
            );
        }
        for r in HEAD_WIDE_BUFS {
            l.synth(out_id.synth(r), &[rows.get(), hd]);
        }
    }

    // ── KV-head BATCHING (always on at decode) ──────────────────────────────────────────────
    // DECODE-ONLY (mq==1) by construction: at mq>1 the prefill path is untouched and the emit is
    // byte-identical, so this cannot collide with prefill work. Measured motivation: 320 of the
    // 527 decode ops/layer are these per-head score/value bmms and every one lands on
    // `numCoresUsed_ = 1` of 32 (matmul/dims.rs — out_sticks==1 and mb==1 ⇒ no dim splits),
    // doing 4,096 MACs each. The batched form takes matmul/dims.rs's y>1 arm, so one op covers
    // all nqh heads across many cores.
    //
    // ⭐ THE BATCHED WALK IS FORM-THREADED, NOT ONE ORDER (`SharedKernelBmmForm`, named ONCE below
    // from the row kind + count and threaded down — no env var, no ambient state):
    //   * mq==1 (and every non-request bundle): the batch-inner `[mb, y, *]` walk and the joint
    //     cost split — BYTE-IDENTICAL to the on-hardware-proven decode emit, kept structurally
    //     (the proven path emits through the same constructors it always did), not by coincidence.
    //   * rows-are-requests at mq>1 (the batch-decode rungs): HEAD-OUTERMOST `[y, mb, in]` /
    //     `[y, mb, out]`, y-stride `mb·in` (= mq·hd, the head plane at ANY mq) and mb-stride `in`
    //     (one row), with the splitter dividing `y` at its full core_split before `mb`/`out` take
    //     the remainder.
    //
    // What the corrected order is FOR, measured 2026-07-30 by baking rung 23 with the batch-inner
    // `[mb,y,in]` order and reading the descriptors:
    //   mq==1 (proven):  numWkSlicesPerDim_ {mb:1, y:4}   maxDimSizes_ [-1,-1,-1]  4 cores
    //   mq==23 (broken): numWkSlicesPerDim_ {mb:23, y:1}  maxDimSizes_ [23,4,64]  23 cores
    // A pinned (`maxDimSizes_` = actual) row-major walk of `[mb, y, in]` encodes mb-stride `y·in`
    // = 256 and y-stride `in` = 64, but `qs` really lives at `h·mq·hd + r·hd + d` — mb-stride 64,
    // y-stride mq·hd = 1472. The two are SWAPPED, coincident only at `mb == 1` — and the pinning
    // does NOT need `mb` to saturate the cores: every mq>=2 rank-3 view classifies Flat and pins
    // (dumped at mq 2/4/8: `maxDimSizes_ [mq,4,64]` with y split 4), which is the defect class of
    // the mq=4 bundle being wrong on hardware while mq<=2 survives. This is the same "dim ORDER sets
    // the stride" class that cost five on-card failures; note the per-core START addresses do NOT
    // catch it (they were a strict subset of the per-head set, with nothing extra) — the defect is
    // in the declared WALK, so any attempt here must diff `layoutDimOrder_`/`maxDimSizes_`/
    // `numWkSlicesPerDim_` too, not just addresses. The head-outermost order states the tensor's
    // true strides at every mq, so what the card reconstructs and what it pins are the same bytes.
    // Not yet proven on hardware at mq>1 — a previous batched-emission change was locally clean and
    // still garbled on the pod, so the card remains the only closing gate, and that is exactly why
    // the proven mq==1 form is not touched by this fix.
    //
    // Per-kernel physical extents (`phys > iteration`, the direction `with_device_extent` is for):
    //   kct    [nkvh, hd, cap], swept `out` = one 64-slot block ⇒ physical `out` extent is cap.
    //   vc     [nqh, cap, hd] with only every gqa-th head slot live, swept `in` = one 64-slot
    //          block ⇒ the live-head stride is gqa·cap rows of `in`.
    //   new_kt [nkvh, hd, mq_pad], swept whole ⇒ y-stride already IS in·out, no override.
    //   new_v  keeps RoPE's `mq` kv-head stride, which at mq==1 is SMALLER than the swept mq_pad
    //          extent — inexpressible, so the new block's VALUE leg stays per-head.
    // ONE STICK PER HEAD, and this is empirical, not theoretical. I removed this gate on the argument
    // that the batched score's `y` stride is `mb*in = mq*hd` — the head stride at any head_dim — once
    // `qs` was declared over `mq` rather than `rows`. Both models baked byte-identically and the op
    // count halved, and granite-3.1-8b's output went from partially-coherent to complete noise on
    // hardware. So something ELSE in the batched form still requires a head to be one stick; the `qs`
    // extent was necessary but not sufficient. Restored, with the evidence, rather than left off on an
    // argument the device disagreed with.
    // ROWS-ARE-REQUESTS IS ALLOWED IN, mq>1 PROMPTS ARE NOT. A prompt chunk runs at mq 31..96 and
    // keeps the per-head path that is proven on hardware; a decode BATCH runs the head-outermost
    // batched walk, whose declared strides are the tensor's own at every mq (the note above). The
    // gate is the blast radius of the unproven widths, not a stride law: the mq>=2 emit differs
    // from the mq==1-proven bytes by design and only the card can accept it.
    //
    // 🛑 THE WIDE RUNGS ARE STILL THE ONES TO DISTRUST FIRST if a batch garbles: at 16 and 32
    // requests the per-core row slices are prompt-sized even though `y` now divides first. Verify by
    // DIFFING `layoutDimOrder_` / `maxDimSizes_` / `numWkSlicesPerDim_` per rung, not addresses —
    // the previous attempt compared only start addresses, found them a strict subset, and shipped
    // complete noise on granite-8b.
    // THE ONE PARSE BOUNDARY for the shared-kernel bmm form (see the note above): request rows at
    // mq>1 take the corrected head-outermost walk + batch-first split; everything else — the
    // hardware-proven mq==1 bundle and every prefill chunk — keeps the proven batch-inner form.
    let score_form = SharedKernelBmmForm::of_score_leg(
        rows_are_requests,
        width.mq(),
        crate::addr::Shape::<0, 0, 0, 0>::slabs_of(hd, Df::Fp16),
    );
    let bmm_form = SharedKernelBmmForm::of_attn_rows(rows_are_requests, width.mq());
    // ⭐⭐⭐ NO HEAD-DIM GATE. This arm required `hd == one stick`, and the reason was ONE stride: the
    // value leg declared `out = hd`, so it derived a head stride of `mq*hd` against a head-major
    // buffer whose pitch is `mq*stick`. Both legs now declare ONE STICK of `out` — the value leg
    // through the slab loop it shares with the per-head arm, the score leg because its `out` IS the kv
    // window and a resident block is one stick by construction — and at one stick the derived stride
    // `mb*out` IS the buffer's pitch at every head dim. The precondition is met rather than assumed,
    // so there is nothing left to gate on.
    //
    // ⛔ THE WALK IS NOT PART OF THIS. `of_attn_rows` keeps choosing the declared axis order and the
    // split policy exactly as before; an earlier attempt changed both in the same commit and produced
    // fluent garbage on granite-3.1-8b. The stride and the order are independent facts and only the
    // stride was ever wrong here.
    let batched = (mq == 1 || rows_are_requests).then_some(());
    // ⛔⛔⛔ THE BATCHED FORM NEEDS A ONE-STICK CONTRACTION, AND ONLY THE VALUE LEG HAS ONE.
    //
    // `matmul_opspec_off`'s own doc pins the dxp-VALIDATED shared-kernel batched shape as the one
    // where `in`/`out` ARE the 64-element stick — it even states the consequence as an identity,
    // "`y` indexes successive sticks (y-stride = 64 = hd)". That sentence is only true at
    // `hd == one stick`; it is the hd=64 assumption, written into the justification of the form.
    //
    // The two legs are NOT alike in this respect:
    //   * VALUE — contracts the kv window (`in = width`, one stick by construction: a resident block
    //     is a stick, the new-token block is sub-blocked into stick-wide pieces) and now emits one
    //     stick of `out` per op. One stick on both axes ⇒ the validated shape verbatim, at any head dim.
    //   * SCORE — contracts the HEAD DIM (`in = hd`), which is `hd/lanes` sticks. At head_dim 128 that
    //     is a TWO-stick contraction under a `y` batch: a shape the validated form does not cover and
    //     that no bundle has ever run.
    //
    // MEASURED on granite-3.1-8b (hd=128): with both legs batched, every emitted address is correct —
    // `qs` at `h*mq*hd`, `sc` at `h*mq*stick`, `expb` at `h*mq*stick`, `ov` at `s*nqh*mq*stick +
    // h*mq*stick`, `vc` at its kv block + `s*cap*stick`, `kct` at `kvh*hd*cap` — and decode is STILL
    // incoherent from the second token. Addresses being right is what leaves the contraction as the
    // remaining difference from the per-head arm, which runs the same `in = 128` at `y = 1` correctly.
    //
    // ⏭ THE FIX IS TO SPLIT THE SCORE LEG'S CONTRACTION BY SLAB (nslab ops of one stick each,
    // accumulating), which restores the validated shape without a head-dim branch. Until that is
    // written, the score leg takes the per-head arm — a gate on "is the contraction one stick", which
    // is the form's REAL precondition, not on the head dim.
    let batched_kv = batched.map(|()| BatchedKv {
        nkvh,
        // ⭐ UNCONDITIONAL NOW: the score leg contracts ONE SLAB per op, so its contraction is one
        // stick at every head dim and the witness needs no extent to inspect. The head-dim gate this
        // replaces is gone, not widened — `of_contraction` remains for legs that have not been split.
        score: Some(OneStickContraction::by_slab_split()),
        // The value leg contracts the kv WINDOW, one stick by construction, so it always batches.
        value: Some(()),
    });
    let (batched_new, batched_prefix) = (batched_kv, batched_kv);

    // ── NEW block FIRST (2026-07-28, matches the OLD proven flash-decode's actual sequencing —
    //    worktree superdsc-batch-perf seeds fM/fdenom/fout from the new token's OWN self-attention
    //    score BEFORE folding in any prefix block; this file previously seeded from prefix block b==0
    //    instead and combined the new token last). Online-softmax is associative in EXACT arithmetic,
    //    so both orderings are mathematically valid, but they are NOT numerically identical in fp16:
    //    the new token's self-score is typically close to the eventual true max (self-attention is
    //    usually among the largest scores), so seeding from it needs small corrections; seeding from a
    //    possibly much-smaller first-block max (heavily pmask'd for short contexts) means every
    //    subsequent block-combine does a LARGER rescale, chaining more correction/exp/multiply steps
    //    with more accumulated rounding — and that chain grows with `nb` (i.e. with context length).
    //    Restored to match the proven ordering exactly: new-block seeds, prefix blocks fold on top. ──
    // ── ROW-BATCHED FOLD ── A fold pass is one (request, page): it reads THAT request's page, so
    //    every row belonging to another request is masked off for the whole pass. Emitting the block
    //    over all `nqh*mq` rows therefore computes `mq` requests' worth of attention to keep one,
    //    `pages*requests` times over. Measured: 137-140 ms of a bs=8 decode step at three pages
    //    against 39.3 ms at one, i.e. 145 tok/s down to 87 — and invisible at --output-len 64, where
    //    the context fits one page and the fold runs once per request.
    //
    //    Per-request, a pass carries `nqh` rows and the runtime rebases it onto its own request's
    //    block (`fold_plan`'s intermediate shift), so the work is `pages*requests*nqh`.
    //
    //    BOTH consumers of the row order move together. Request-major is what makes one request's
    //    rows contiguous; head-major is what makes the whole-batch `mb = gqa*mq` contiguous. The
    //    new-token block is not folded, so nothing rebases it — it is emitted once PER REQUEST and
    //    each copy names its own `req`. Flipping one and not the other leaves it reading a stride the
    //    other no longer writes, which is fluent wrong output, not a crash.
    //
    //    At mq==1 the two orders coincide (`h*1 == 0*nqh + h`), so the one-request decode bundle and
    //    every prefill rung emit exactly as before — which the fingerprint gate checks.
    // WHOLE-BATCH FOLD ROWS. A pass carries every row and the mask silences the ones that are not
    // its request — the design that predates the row-batched experiment. That experiment (rows =
    // nqh + a seg0 rebase per pass) was written while the fold was RUNNING ONCE (`reps` collapsed
    // to 1 because the fused body drops `PageFold`), so it could never have been validated against
    // a fold that actually re-launches. With `reps` fixed, the mask shape has to match the rows the
    // pass declares, and the whole-batch pairing is the one that was designed together.
    let per_req = false;
    // The ROW REGIME every block of this node is emitted under, extent and row law paired in one
    // value: per-request passes carry ONE request's `nqh` rows (the frame the runtime rebases, and
    // the frame each per-request new-block copy names its `req` into); whole-batch passes carry the
    // shared-buffer `nqh*mq`. The block assembler takes the regime itself, so a per-request extent
    // cannot ride into a whole-batch pass or vice versa.
    let block_rows = if per_req {
        // The geometry's own const: one request's rows are its `NQH` head rows, arithmetic the
        // compiler already did — not a runtime `nqh` fetched back out of the destructuring.
        BlockRows::PerRequest(geom.per_request_rows())
    } else {
        BlockRows::WholeBatch(rows)
    };
    // ⭐ ONE FOLD PASS FOR THE WHOLE BATCH. A pass used to read ONE request's page and mask off every
    // row that was not that request's, so it computed `mq` requests' worth of attention to keep one,
    // `pages × requests` times over — and a pass is a LAUNCH, which costs ~93 µs whatever it carries.
    // 8 of the 12 launches a bs=8 layer paid for were this factor.
    //
    // Giving the score and value kernels a request axis makes row `h*mq+r` read request `r`'s own
    // page, so one pass per PAGE serves everybody. Three things have to agree and this is one of them:
    // the axis here, `OpKv::batched_requests` below (which lets the runtime drop the factor from
    // `reps`), and the mask's shape on the host (one block per page, every row valid on its own
    // request's history — `decode_batch_prefix_mask_f16`). Any one alone is a wrong-history bug.
    //
    // GATED ON `hd == stick`: a request is one ROW of the stick-blocked `qs`, which is one stick. At
    // hd > 64 a head spans several sticks and the request step is no longer `hd`, so there is no single
    // `y` stride — `head_major_collapse_valid` already refuses the batched form there, and this rides
    // on the same gate rather than adding a second opinion.
    // ⛔ OFF. Three attempts to give the fold a request axis all produced wrong tokens on the card, and
    // enabling it here is what left this branch INCOHERENT while every host gate was green.
    //
    // The shape was never the problem — a strided batched attention, batch dimension = the sequence,
    // each element reading its own KV at a constant stride, is what every inference engine does for
    // batch decode, and the emitted addresses were MEASURED correct on all three operands
    // (`shipped_fold_op_strides.rs`). The bug is in how the KV BASE is composed: the runtime applies a
    // per-launch segment shift (`fold_plan::fold_delta`) AND the op carries a baked per-request stride,
    // and with `reps` collapsed those two have never been read against each other. That is one thing to
    // find, not a reason for a fourth design.
    //
    // Until it is found, the fold takes the launch-per-request path — which is the ~51 ms bs=8 step that
    // WORKS, rather than a 2x that does not.
    let fold_request_axis: Option<u32> = None;
    let _ = (rows_are_requests, batched.is_some());
    for r in 0..if per_req { mq } else { 1 } {
        for rw in mq_pad_t.row_windows() {
            let j = rw.index();
            assemble_attn_block(
                &mut ops,
                t,
                &{
                    let b = if nsub.is_single() {
                        "n".to_string()
                    } else {
                        format!("n{j}")
                    };
                    if per_req { format!("{b}_r{r}") } else { b }
                },
                nqh,
                hd,
                width.mq(),
                block_rows,
                qs,
                score_form,
                // Sub-block `j` is the 64-column window at plane `j*hd*stick` — the same read the prefix blocks
                // do into `kct` (`b*hd*stick`), with the Kᵗ scratch's SLOT extent as the full kernel width: this
                // slot is the kernel's physical pitch (the resident block's value here is `cap`), a SLOT question
                // — the row count answered it only through the new-block identity `rows == slots`.
                &new_kt,
                // A GENUINE Kᵗ: `assemble_restickify_kt_2d` transposed it out of the activation into this
                // scratch, so its contraction is the ROW axis, exactly as every other matmul's kernel is.
                // Only the RESIDENT leg below reads an untransposed plane.
                crate::sdsc_abstract::KernelOrient::ContractionOnRows,
                mq_pad_t.slots().kernel_row_pitch(),
                |h, sl| {
                    let hi = h.get();
                    // `new_kt` is nkvh separate `[hd, mq_pad]` blocks — a block index over the block's own
                    // footprint, plus this row window's slot-stick plane inside it (the MqPad rows/slots
                    // identity: the window's rows ARE the block's slots). The slot axis's extent is the SLOT
                    // answer, same as the scratch's own `synth` above — not the padded row count.
                    let blk = crate::addr::Nest::new(
                        &["feat", "slot"],
                        &[hd, mq_pad_t.slots().slot_axis_extent()],
                        Df::Fp16,
                    );
                    let kvh = crate::sdsc_abstract::gqa_kv_head(hi as usize, gqa as usize) as u32;
                    // TWO ORTHOGONAL SLAB COORDINATES on one nest: `slab` is the SLOT window (the stick
                    // axis), `slab_of::<Feat>` is the head-dim slab of the contraction (a non-stick axis).
                    // Both multipliers live in `addr`; neither is spelled here.
                    blk.block(kvh)
                        .slab(rw.index())
                        .slab_of::<crate::addr::Feat>(sl)
                        .dev()
                },
                // new_v is nkvh-wide now too (dedup, symmetric to new_kt above) — same gqa_kv_head mapping.
                // PER-KV-HEAD BASE = `kvh * mq * hd` (2026-07-28, CORRECTED from a briefly-committed `mq_pad`
                // version that regressed decode). Unlike `new_kt` (restickified into a COMPACT
                // `[nkvh,hd,mq_pad]` layout, where `kvh*hd*mq_pad` IS the right compact per-head base),
                // `new_v_rep` is never restickified: it keeps the wide `[*, nkvh*hd]` shape whose kv-head
                // stride RoPE sets to `mq` (`rope_prefill_block_offset`), not `mq_pad`. Decode (mq=1) reduces
                // to the original `kvh*hd`; prefill (mq=31) gets the `kvh*31*hd` the original bare `kvh*hd`
                // was missing for kvh>=1.
                // V rows `[j*stick, (j+1)*stick)` on top of the kv-head base — the prefix side's `b*stick*hd`.
                new_v,
                hd,
                |h, sl| {
                    let hi = h.get();
                    // `new_v_rep` keeps the wide `[mq, nkvh*hd]` shape RoPE packed, so the kv head is a
                    // COLUMN there, not a block; the row window `j` rides on top.
                    // The kv head is a COLUMN of the wide packed stream (not a block — this tensor is never
                    // restickified), and the row window `j` is `j*stick` rows into it.
                    let kvh = crate::sdsc_abstract::gqa_kv_head(hi as usize, gqa as usize) as u32;
                    crate::addr::Nest::new(&["row", "head", "feat"], &[mq, nkvh, hd], Df::Fp16)
                        .view()
                        .at(rw.first_row().row_idx())
                        .at(Idx::<Head>::n(kvh))
                        .slab(sl)
                        .dev()
                },
                // This block's score WIDTH: one row window's columns — quantity (3) as columns (each padded
                // row is one score column), NOT the lane count it shares a value with. The full `mq_pad`
                // width lives on the kernel's stride, never in this slot.
                BlockCols::of_row_window(RowWindow::ROWS),
                // cmask is staged BLOCK-MAJOR: `nsub` blocks of `[rows, stick]`, block `j` holding columns
                // `[j*stick, (j+1)*stick)`. Required because the mask is read through a FLAT view whose row
                // stride IS `width` (`In::sliced`), so a `[rows, mq_pad]` row-major staging would step 64 where
                // the row pitch is `mq_pad`. At nsub==1 this offset is 0 and the staging is unchanged.
                // cmask is staged BLOCK-MAJOR (see above): `nsub` blocks of `[rows, stick]`. The outer axis
                // of this nest IS that block index; `feat == stick`, so the corner is `j*rows*stick`.
                cmask,
                // The staging's row axis is the shared-buffer extent — the same MaskRows the ops sweep; its
                // block width is one row window's SCORE columns, the same quantity the width slot carries.
                Nest::new(
                    &["row", "head", "feat"],
                    &[
                        rows.get(),
                        j + 1,
                        BlockCols::of_row_window(RowWindow::ROWS).get(),
                    ],
                    <Fp16 as DataFormat>::DF,
                )
                .view()
                .at(Idx::<Head>::n(j))
                .dev(),
                false, // cmask: varies per query row, needs the real per-row read (worker pre-tiles it)
                // Sub-block 0 SEEDS the online softmax; later sub-blocks fold, exactly like the prefix blocks.
                // The new-token block SEEDS, but only on its first sub-block: later sub-blocks of the same block
                // fold onto that seed. `j == 0` was the boolean; the role says which of the two it means.
                if j == 0 {
                    BlockSeed::SeedsState
                } else {
                    BlockSeed::FoldsOntoSeed
                },
                batched_new,
                bmm_form,
                r,
                // THE NEW-TOKEN BLOCK IS NOT FOLDED, so there is nothing to collapse: it runs once inside the
                // body's own launch and its K/V come from this step's `new_kt`/`new_v`, which are indexed by
                // row, not by pool block. A request axis here would step a page stride through a buffer that
                // has none.
                None,
                &bufs,
                sym_id_base,
                layout,
            )?;
        }
    }
    // ── PREFIX blocks: nb stick-wide blocks over the resident cache, pmask sliced per block, folded
    //    on top of the new-block seed above (never `first` now). `vc` read maps query head h to its
    //    REPRESENTATIVE query head qh=kvh*gqa (V dedup — `vc` stays nqh-SIZED, unlike `kct`'s compact
    //    nkvh storage, so this is qh*cap*hd, NOT the kct-style compact kvh*hd*cap
    //    `gqa_dedup_kv_kernel_base` gives). Matches cachewr_v's own now-representative-slot-only
    //    writes (see lower_attn_node) — same discipline as kct_base, applied to a differently-shaped
    //    (still nqh-sized) tensor. ──
    // The prefix fold: ONE variant, re-launched once per page of resident prefix with that page's
    // own validity row. Emitted by POSITION so a rename cannot drop an op out of the fold.
    //
    // A zero-masked second variant for full pages was tried, to avoid shifting the mask per page.
    // It broke multi-page coherence and it was chasing a cost that is not there: the mask's segment
    // is ~1.6 MB in the DECODE bundle (the ~150 MB figure is the PREFILL bundle, where mq=96 scales
    // every intermediate, and prefill runs once per request).
    // ⭐⭐⭐⭐⭐ THE PREFIX BLOCK'S Kᵀ AS TRANSIENT SCRATCH — `[nkvh, hd, STK]`, ONE stick-block, rewritten
    // by every window of the sweep. This is the other side of the resident-`kct` trade: IBM's paged
    // attention keeps ONE natural-K cache and materialises the transpose PER TILE as the KV axis streams
    // (`spyre-inference/.../paged_vector_add_target.py:64`), where we cache it in a third KV plane. Declared
    // here so the two forms can be measured against each other rather than argued about; `SCRATCHY_PREFIX_KT_
    // STREAM` is NOT a gate — the choice is made below at emit, per bundle, from the SWEEP WIDTH, which is
    // the quantity the trade actually turns on.
    let pfx_kt = crate::placement::syn(layout, out_id.synth(R::PrefixKt));
    if let Some(l) = layout {
        l.synth(
            out_id.synth(R::PrefixKt),
            &[nkvh, hd, crate::sdsc_abstract::POOL_STICK],
        );
    }
    // ⛔ AND THE CHOICE IS THE SWEEP WIDTH, NOT A FLAG. Streaming re-transposes every window of the sweep,
    // so it costs `nb` restickifies per step against the resident plane's ONE at write time — a win only
    // while `nb == 1`, i.e. while a launch's sweep is a single stick-block. At `active_cap == POOL_STICK`
    // the two forms do identical work and streaming needs no third plane; above it, streaming re-reads the
    // whole prefix every step (at 576 tokens: 9 blocks x nkvh x 4096 elems x 40 layers ≈ 236 MB/step
    // against a 24 ms budget) and the resident plane buys that back.
    // MEASURING IT UNCONDITIONALLY FIRST. The width test below is what the arithmetic argues for, but that
    // arithmetic has been wrong about this device repeatedly this session, so the wide-sweep cost gets
    // measured on the card before it is believed:
    //   let stream_prefix_kt = active_cap <= crate::sdsc_abstract::POOL_STICK;
    let stream_prefix_kt = true;
    let fold_from = ops.len();
    // Windows ONE launch folds. `active_cap` is the sk_bucket rung as before, and the page equals the
    // pre-paged capacity, so this is bit-for-bit the baseline's fold; contexts past one page cost
    // additional LAUNCHES of this same group, never a wider sweep. The `active_cap / 64` lives in
    // `SlotWindow::count_in`, where the 64 is the window's SLOT count — quantity (2), the reduce's
    // one-stick column budget — not the lane count it used to be spelled as.
    for w in SlotWindow::sweep(crate::sdsc_abstract::SlotCount::new(active_cap)) {
        let b = w.index();
        // ⭐ STREAMED FORM: transpose THIS window's natural-K block into the one-block scratch, right before
        // the score ops that read it. Ordered restickify→score per window inside one launch, so a single
        // scratch block serves every window — which is what makes the resident Kᵀ plane unnecessary.
        if stream_prefix_kt {
            for kvh in 0..nkvh {
                let head = crate::sdsc_abstract::KvHead::new(
                    kvh,
                    std::num::NonZeroU32::new(nkvh).expect("nkvh > 0"),
                )
                .expect("kv head in range");
                ops.push(crate::emit::assemble_restickify_kt_2d(
                    &format!("attn_pfxkt{kvh}b{b}_o{t}"),
                    crate::sdsc_abstract::KtTileSlots::of_write_slab(),
                    crate::sdsc_abstract::KtTileFeats::of_head_dim(hd),
                    knat,
                    // This window's slots of the NATURAL-K plane — the window carries its own first slot,
                    // so no block stride is spelled here.
                    crate::addr::DevOff::from_view_step(pool.addr(
                        crate::sdsc_abstract::KvCoord::block(
                            crate::sdsc_abstract::KvPlane::Knat,
                            head,
                        )
                        .at_slot(w.first_slot()),
                    )),
                    &pfx_kt,
                    // The scratch's own `[nkvh, hd, STK]` block for this kv head — no window term, because
                    // the scratch holds exactly ONE window at a time. That is the whole point of it.
                    crate::addr::DevOff::from_view_step(kvh * hd * crate::sdsc_abstract::POOL_STICK),
                    sym_id_base,
                    layout,
                ));
            }
        }
        // PER-BLOCK OFFSETS, PRINTED (`SCRATCHY_SDSC_FOLD_TRACE`). Blocks 0-1 of a 4-block sweep are correct
        // and blocks 2-3 are not, and every candidate cause has been checked correct by reading — so the
        // next step is to make the blocks observable rather than to argue about them. The `[body-choice]`
        // trace killed two wrong theories in single runs; this is the same move one level down.
        if std::env::var_os("SCRATCHY_SDSC_FOLD_TRACE").is_some() {
            // The Kᵀ block coordinate this sweep reads. Its pool address at the block CORNER (slot 0)
            // is the sweep's zero point — the first window starts there by the sweep's own law — so
            // the printed delta is this window's slot displacement, with no window named by hand.
            let kt_block = crate::sdsc_abstract::KvCoord::block(
                crate::sdsc_abstract::KvPlane::Kt,
                crate::sdsc_abstract::KvHead::of_query(
                    crate::sdsc_abstract::QueryHead::FIRST,
                    crate::addr::Gqa::new(gqa),
                ),
            );
            let at = |win: SlotWindow| -> u32 { pool.addr(kt_block.at_slot(win.first_slot())) };
            let nb = SlotWindow::count_in(crate::sdsc_abstract::SlotCount::new(active_cap)).get();
            eprintln!(
                "[fold-block] b={b}/{nb} cap={active_cap} stick={} hd={hd} mq={mq} rows={} | kt_off={} delta_b0={} | mask_slab={} slab_step={}",
                BlockCols::of_one_stick(Lanes::FP16).get(),
                block_rows.extent(),
                at(w),
                at(w).wrapping_sub(pool.addr(kt_block)),
                mask_shape.slab_elems(0, w.mask_slab()).elems(),
                mask_shape.slab_elems(0, 1).elems(),
            );
        }
        assemble_attn_block(
            &mut ops,
            t,
            &format!("p{b}"),
            nqh,
            hd,
            width.mq(),
            block_rows,
            qs,
            score_form,
            // BLOCK STRIDE = `b * hd * stick`, NOT `b * stick` (2026-07-28). `kct` is `[hd, cap]` per
            // kv-head, STICK-MAJOR on `cap` (the last dim), so slot `s` sits at
            // `(s/64)*(hd*64) + d*64 + (s%64)` -- one whole `hd*64` PLANE per 64-slot block, not 64
            // elements. The score matmul reads a `[k=hd, n=stick]` kernel window whose own internal
            // addressing is `off + d*64 + j`, so `off` must be the block's plane base `b*hd*stick`.
            // With the old `b*stick`, block 0 was correct (offset 0 either way) but every later block
            // read `hd`x too low -- i.e. attention was exact for the first 64 KV slots and read garbage
            // from slot 64 onward. That is precisely the observed decode cliff (coherent to ~64 tokens,
            // then progressive degradation as more blocks fill) and prefill's immediate garbage (a
            // multi-block prompt is wrong from the first token). Matches the old proven flash decode
            // verbatim: `let k_blk = b * hd * stick;` (worktree superdsc-batch-perf). The V side below
            // is `b*stick*hd`, which is already correct and matches that same reference (`vblk`).
            // ⭐ THE SCRATCH WHEN STREAMING, THE RESIDENT PLANE OTHERWISE. In the streamed form the ops just
            // above transposed THIS window into `pfx_kt`, so the kernel is that one block and the resident
            // `kct` plane has no reader at all.
            if stream_prefix_kt { &pfx_kt } else { kct },
            crate::sdsc_abstract::KernelOrient::ContractionOnRows,
            // The scratch's pitch is ONE stick-block — it holds a single window — where the resident plane's
            // is a whole page. Getting this wrong reads the next kv head's block as this one's later slots.
            if stream_prefix_kt {
                crate::sdsc_abstract::SlotExtent::of_one_stick().kernel_row_pitch()
            } else {
                crate::sdsc_abstract::PagedKvPool::KT_KERNEL_PITCH
            },
            // THE POOL COMPOSES THE REQUEST TERM, not this call site. `kt_block_base_of` is the same
            // `block_index(kvh) + r` the cache write bakes, so the fold's kernel and the write that
            // fills it cannot disagree about where request `r` lives — and no stride is spelled here.
            |h, sl| {
                use crate::sdsc_abstract::{KvCoord, KvPlane};
                let kvh = crate::sdsc_abstract::KvHead::of_query(h, crate::addr::Gqa::new(gqa));
                if stream_prefix_kt {
                    // The scratch is `[nkvh, hd, STK]` and holds THIS window only, so there is no slot term
                    // — the window is already spent, in the restickify that filled it. Just the kv head's
                    // block and the contraction's feature slab.
                    return crate::addr::DevOff::from_view_step(
                        kvh.get() * hd * crate::sdsc_abstract::POOL_STICK
                            + FeatIdx::of_slab(sl).get() * crate::sdsc_abstract::POOL_STICK,
                    );
                }
                crate::addr::DevOff::from_view_step(
                    pool.addr(
                        KvCoord::block(KvPlane::Kt, kvh)
                            .at_slot(w.first_slot())
                            // The contraction's head-dim slab, through the pool's OWN model — the same
                            // `at_feat` door the V plane below uses, so the two planes cannot disagree about
                            // where a feature slab is.
                            .at_feat(FeatIdx::of_slab(sl)),
                    ),
                )
            },
            // The closure takes the head-dim SLAB as well as the head, so the slab rides inside the
            // pool's own model instead of being added to a finished address. The V operand's row count
            // is the PAGE, not `cap`, and it reaches the address only through that model — there is no
            // separate row-count argument to disagree with the kernel's declared `in` extent.
            vc,
            // ⭐ `PAGE_SLOTS`, NOT `hd` — the V plane's ROW axis is SLOTS, symmetric with `kt_stride` above.
            //
            // `Stk::kernel(k_in, n_out)` becomes `StickLayout { rows: k_in, cols: n_out }`, and a stick-blocked
            // kernel addresses `(r, c)` as `(c/stk)*(rows*stk) + r*stk + c%stk`. With `k_in = hd` the
            // stick-GROUP stride was declared `hd*stk` where this plane's is `PAGE_SLOTS*stk`.
            //
            // ⚠️ INERT AT head_dim 64 AND ONLY THERE: one feature stick means `c/stk == 0` always, so the
            // wrong `rows` factor multiplies zero and NO EXISTING TEST CAN SEE THIS CHANGE. At hd=128 there
            // are two sticks and the upper one landed at `hd*stk` instead of `PAGE_SLOTS*stk`, i.e. every
            // feature above 63 read the wrong slot. That is why this is verifiable ONLY on a real hd=128
            // model through `scr batch` — the emission at hd=64 must be byte-identical, and that is the
            // safety check for this edit.
            crate::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32,
            // NAMED, NOT COMPOSED: `w` is a slot window and `sl` a feature slab, so both are
            // COORDINATES. The hand-added `sl * PAGE_SLOTS * stick` was the V plane's own feature-stick
            // stride restated at the call site.
            |h, sl| {
                use crate::sdsc_abstract::{KvCoord, KvPlane};
                let kvh = crate::sdsc_abstract::KvHead::of_query(h, crate::addr::Gqa::new(gqa));
                crate::addr::DevOff::from_view_step(
                    pool.addr(
                        KvCoord::block(KvPlane::V, kvh)
                            .at_slot(w.first_slot())
                            .at_feat(FeatIdx::of_slab(sl)),
                    ),
                )
            },
            // This block's score WIDTH: the slots one fold window holds — quantity (2), the reduce's
            // one-stick column budget, NOT the lane count it shares a value with.
            BlockCols::of_slot_window(SlotWindow::SLOTS),
            // pmask is read mb-BROADCAST, so it materialises ONE row: block b is the slab-b corner
            // of a `[1, cap]` nest. Stating the MATERIALISED extent is what keeps the stick-group
            // term honest for a broadcast operand.
            pmask,
            // ONE VALIDITY ROW, OR ONE PER QUERY ROW.
            //
            // Prefix validity is head-independent, so a single broadcast row is exact while every row
            // of a bundle shares a resident length — true for a prompt chunk, whose rows are
            // consecutive positions of one sequence. It is false for a decode batch, where each row
            // is a different request with its own history and a broadcast row would give them all the
            // first one's.
            //
            // GATED, not made unconditional, because the broadcast is LOAD-BEARING FOR MEMORY. The
            // mask spans the whole context (MAX_PAGES_PER_REQUEST * PAGE_SLOTS), so one row per op
            // row is `nqh*mq` of them: about 50 MB at prefill's mq=96, re-uploaded every forward.
            // At a decode batch of 8 it is ~4 MB, which is affordable — the blow-up is prefill's
            // problem and prefill does not need this.
            if rows_are_requests {
                // ⭐ FROM THE SHARED SHAPE, not from a nest built here out of `rows` and `cap`. The
                // worker stages this buffer from the SAME value, so the read and the write cannot
                // describe different bytes — which they could when each side derived its own extents,
                // and an additive mask fails silently in the dangerous direction (unstaged bytes read
                // as 0, and 0 is VALID).
                // The corner crosses as the typed `MaskCorner` it is, through its own `DevOff`
                // door — a slab corner is `slab * rows * 64`, a plane term, not the single-stick
                // view step `from_view_step` is contracted for.
                crate::addr::DevOff::of_mask_corner(mask_shape.slab_elems(0, w.mask_slab()))
            } else {
                Nest::new(&["row", "feat"], &[1, cap], <Fp16 as DataFormat>::DF)
                    .view()
                    .slab(w.mask_slab())
                    .dev()
            },
            !rows_are_requests, // broadcast for a prompt; per-row when the rows are requests
            // ⭐ THE FOLD NEVER SEEDS — the claim this file made in prose ("never `first` now"), now carried
            // by the call. A seeding fold pass would reset the running state mid-accumulation and cost the
            // deepest row every page it had already folded.
            BlockSeed::FoldsOntoSeed,
            batched_prefix,
            bmm_form,
            0,
            // ⭐ THE PREFIX FOLD TAKES THE REQUEST AXIS when the rows ARE requests. That is what makes
            // one pass serve the whole batch, so the runtime can drop the `× requests` factor from
            // `reps` — see `fold_plan::reps`, gated on the `batched_requests` this sets below.
            fold_request_axis,
            &bufs,
            sym_id_base,
            layout,
        )?;
    }

    for op in ops[fold_from..].iter_mut() {
        op.kv_page_fold = true;
        // DECLARED, NOT INFERRED: the runtime cannot see what axes an op was baked with, and claiming
        // a request axis the kernel does not have turns `pages × requests` passes into `pages` while
        // every row but one reads the wrong history. It is only half the precondition — the other half
        // is a pool layout that admits one stride, which `fold_plan::LaunchPages` checks at launch.
        op.kv_batched_requests = fold_request_axis.is_some();
        // THE FOLD-ROW REGIME, declared from the value every fold block above was assembled with.
        // This is what the worker's intermediate-segment rebase stride derives from: whole-batch
        // passes sweep every shared-buffer row (stride 0, nothing to rebase), per-request passes
        // carry one request's rows and must be rebased onto their own block. Same discipline as
        // `kv_batched_requests` above — a bake fact the session cannot see, so it rides the manifest.
        op.kv_fold_rows = block_rows.regime();
    }
    // Finalize: out = run_o / run_l via ONE native `realdiv` op — no separate reciprocal step at all.
    // Earlier this session a single realdiv was reverted for reading-and-WRITING the SAME buffer
    // (run_o) in one op, a real in-place-aliasing hazard; that bug was the aliasing, not `realdiv`
    // itself. Writing to a genuinely DIFFERENT buffer (`out`, never run_o/run_l) is not in-place, so
    // this is safe and cuts reciprocal+multiply (2 ops batched / 1+nqh ops per-head) down to 1 op /
    // nqh ops — no `rden` buffer needed at all.
    //
    // The mq==1 vs mq>1 SHAPE split is unrelated to this and stays: `out`'s real physical layout is
    // `[mq, nqh·hd]` (multi-stick-wide once `nqh·hd>64`) — a GENUINELY different byte ordering than
    // this function's internal `[nqh·mq, hd]` head-major storage ONCE mq>1 (confirmed by a real
    // build-time arrangement-conflict error at mq=31: writing `out` as one `[nqh·mq,hd]` op declared a
    // DIFFERENT StickLayout than the downstream o_proj matmul's own read of the same tensor — a real
    // reshape, not a false-positive strictness check). At mq==1 there is no such conflict: with only
    // one query row, "`nqh` head-major rows" and "`nqh·hd` interleaved columns of one row" are the
    // SAME bytes. So decode keeps the single, cheap batched op; only mq>1 pays for the per-head reshape.
    // STEP 7 — the boundary un-permute, and the only place a conversion genuinely exists.
    //
    // Head-major `[nqh*mq, hd]` IS the token stream `[mq, nqh*hd]` under the feature permutation
    // pi(h,d) = (d/stk)*nqh*stk + h*stk + d%stk. The block is internally consistent in pi-order; the
    // one place pi must be undone is the hand-off to o_proj, which reads `out` in token order.
    //   run_o: slab s, head h, row r at  s*(nqh*mq*stk) + (h*mq+r)*stk
    //   out  : the same element at       h*mq*hd + s*mq*stk + r*stk
    // At nslab == 1 those are equal and pi is the identity — which is why the two forms below are the
    // ones that already ship, kept verbatim so granite-3.1-2b does not move.
    // The finalize nests address `token_stream`/`head_major` only. There is no score block here, so
    // there is no width to state: `score_rows` takes its width as a parameter, and a nest without a
    // score buffer simply cannot be asked that question.
    let nests = BlockNests {
        heads: nqh,
        hd,
        mq: width.mq(),
        rows: BlockRows::WholeBatch(rows),
        req: 0,
    };
    let nslab = nests.slabs();
    if nslab == 1 {
        // ONE OP AT ANY ROW COUNT. The note above works the permutation out: at `nslab == 1`, `hd`
        // IS the stick, so out's `h*mq*hd + s*mq*stk + r*stk` and run_o's `(h*mq+r)*stk` are the same
        // element — pi is the identity, and that does not depend on `mq`. The `mq == 1` guard was
        // costing `nqh` ops a layer (32 at granite) for a permutation that is not there.
        //
        // Scoped to a decode batch; prefill's bundles are proven on hardware and stay byte-identical.
        if mq == 1 || rows_are_requests {
            ops.push(assemble_pointwise_broadcast_off(
                &format!("attn_o_o{t}"),
                "realdiv",
                // The batched finalize: ALL `nqh*mq` head-major rows in one op, head-dim wide.
                RowCount::of_mask_rows(rows),
                BlockCols::of_head_dim(hd),
                &[
                    In::full(&hm(&bufs.run_o)).ew(),
                    In::col(&hm(&bufs.run_l)).ew(),
                ],
                &hm(out),
                crate::addr::DevOff::ZERO,
                sym_id_base,
                layout,
            ));
        } else {
            for h in crate::sdsc_abstract::QueryHead::all(nqh_nz) {
                let hi = h.get();
                ops.push(assemble_pointwise_broadcast_off(
                    &format!("attn_o_h{hi}_o{t}"),
                    "realdiv",
                    // The per-head finalize: ONE head's `mq` rows per op — the rows slot genuinely
                    // carries the chunk's rows here, not the shared-buffer extent.
                    RowCount::of_query_rows(width.mq()),
                    BlockCols::of_head_dim(hd),
                    &[
                        In::sliced(&hm(&bufs.run_o), nests.token_stream(hi, 0).off()).ew(),
                        In::col_at(&hm(&bufs.run_l), nests.head_major(hi, 0)).ew(),
                    ],
                    &rb(out, mq, nqh * hd),
                    nests.token_stream(hi, 0).off(),
                    sym_id_base,
                    layout,
                ));
            }
        }
    } else {
        // ⛔⛔⛔⭐⭐⭐ THIS PER-HEAD LOOP CANNOT GO VIA A POINTWISE OP, AND HERE IS THE PROOF — so the next
        // attempt starts from the right place instead of re-deriving strides (2026-08-16).
        //
        // The finalize is a RELAYOUT, not just a divide, and its two operands want OPPOSITE outermost
        // axes:
        //   run_o (head-major)   (h,r,d) at `dgroup*(nqh*mq*64) + (h*mq+r)*64 + d%64`  — dgroup OUTER
        //   out   (token stream)        at `(h*nslab+dgroup)*(mq*64) + r*64 + d%64`    — h      OUTER
        // One declared order cannot produce both. That is exactly why the `nslab == 1` branch above is
        // ONE op: `dgroup` has extent 1 there, so its stride is irrelevant and the two framings
        // coincide — the hd=64 coincidence, again.
        //
        // Dropping only the HEAD loop (keeping `nslab` ops) would need the OUTPUT declared as `nqh`
        // blocks strided `nslab*mq*64` apart. `assemble_pointwise_broadcast_off_from_tile` takes
        // `rows: u32, cols: u32` and ONE `out_offset` — rank-2, contiguous. A pointwise op has no axis
        // to hang that stride on, and no `y` either.
        //
        // ⏭ THE TWO ROUTES, both real work, neither a stride fix:
        //   1. `SELT_HEADMAJOR_TID` — the `nqh` one-hot `[hd, nqh*hd]` selectors that already scatter
        //      head-major back to token-stream as a MATMUL (`out += out_h · SelT_h`, prefill-only
        //      today). A matmul can carry `y`; ⛔ but check the ARRANGEMENT first — a `y`-batched
        //      rank-3 view makes `y` the leading axis, which is what killed RoPE's rotate batching.
        //   2. Permute o_proj's weight ROW-BLOCKS by the `nqh × nslab` perfect shuffle at staging, so
        //      o_proj reads head-major `run_o` directly and the finalize becomes the one-op form above.
        //      Host-side, once, identity at `nslab == 1`. `stage_weight_tiled` is rank-generic and a
        //      rank-4 descriptor already ships, so the affine reindex is expressible.
        //      ⛔⛔ AND IT DOES NOT WORK AS WRITTEN — two reasons, worked out 2026-08-17 so route 2 is not
        //      re-attempted from this note. (a) AT `mq > 1` NO WEIGHT PERMUTATION CAN FIX IT: `run_o`
        //      carries the head on its ROW axis (`h*mq + r`), so the contraction elements for ONE output
        //      row live in `nqh` DIFFERENT ROWS, and a matmul contracts along its activation's COLUMN
        //      axis only. The permutation story is only true at `mq == 1`, where a row IS a head.
        //      (b) AND THE WEIGHT IS STAGED ONCE FOR BOTH BUNDLES — prefill's o_proj reads the
        //      token-stream layout, so permuting the shared rows for decode breaks prefill; a second
        //      permuted copy costs 16.8 MB × 40 layers on granite-8b.
        //
        //   ⛔ A THIRD ROUTE, AND WHY IT IS ALSO CLOSED (checked before writing code): at `mq == 1` each
        //      `dgroup` plane of `run_o` IS contiguous (`dg*(nqh*64) + h*64`), so per-operand pitches —
        //      the machinery that collapsed the cache write — would need the OUTPUT to step `nslab`
        //      sticks per row (`out` block `h*nslab+dg` sits `hd` apart for fixed `dg`). A stick-blocked
        //      walk's `mb` stride IS one stick by construction, and `StickLayout::group_stride`'s fp16
        //      case is pinned to `lanes()` — the only value proven representable on-card (Kani
        //      `fp16_group_stride_always_lanes`). So a 2-stick row step is not a stride this backend can
        //      declare, in either direction (swapping which operand is strided just moves the same
        //      2048-element row step onto the input).
        for h in crate::sdsc_abstract::QueryHead::all(nqh_nz) {
            let hi = h.get();
            for s in 0..nslab {
                ops.push(assemble_pointwise_broadcast_off(
                    &format!("attn_o_h{hi}s{s}_o{t}"),
                    "realdiv",
                    // Per (head, slab): one head's `mq` rows, one head-dim SLAB of features.
                    RowCount::of_query_rows(width.mq()),
                    BlockCols::of_head_slab(FeatIdx::SLAB_FEATS),
                    &[
                        In::sliced(&hm(&bufs.run_o), nests.head_major(hi, s)).ew(),
                        In::col_at(&hm(&bufs.run_l), nests.head_major(hi, 0)).ew(),
                    ],
                    &rb(out, mq, nqh * hd),
                    nests.token_stream(hi, s).off(),
                    sym_id_base,
                    layout,
                ));
            }
        }
    }
    // ⛔⛔⛔ EVERY INTERNAL BUFFER MUST BE READ WHERE IT WAS WRITTEN — CHECKED AT BUILD, FROM THE
    // EMITTED ADDRESSES.
    //
    // This replaces `tests/emitted_attn_agreement.rs`, which keyed on OP NAMES: `attn_p0ov_h3s1`
    // carried "head 3, slab 1" and two ops naming the same block had to agree. Batching the value leg
    // over a GQA group renamed those ops `_g{kv}`, the per-head identity left the names, and the
    // comparison went DEGENERATE — 64 multi-use keys where the file's own doc records 128, so its "0
    // disagreements" meant nothing. A check that cannot fail is worse than none, and that one had gone
    // quiet exactly when the emission it guards started changing.
    //
    // So the identity comes from the DESCRIPTOR instead: the per-core start addresses the emitter
    // already baked. Nothing here names a head, a slab or an op, so no renaming can blind it again.
    //
    // WHAT IT REFUSES: a read of an internal buffer that falls outside the span some op WROTE. That is
    // the shape of every defect in this file — two arms disagreeing about where a head lives
    // (`token_stream` at `h*mq*hd` vs `head_major` at `h*mq*stick`, equal only at one stick), a
    // producer writing slab 1 where the consumer reads slab 0. Deliberately a SPAN test and not
    // set-equality: producer and consumer legitimately split their work across cores differently, so
    // their per-core bases need not coincide — but a read outside everything written cannot be right.
    check_internal_buffers_are_read_where_written(&ops, &bufs)?;
    Ok(ops)
}

#[cfg(test)]
mod buf_decl_tests {
    use super::{HEAD_WIDE_BUFS, STICK_WIDE_BUFS};
    use crate::place::SynthRole as R;

    /// ⭐⭐⭐ THE DECLARED SET IS EXACTLY THE ONLINE-SOFTMAX BUFFERS, IN THE ORIGINAL ORDER.
    ///
    /// `synth` is a bump allocator, so this list IS the address assignment. Dropping an entry does
    /// TWO things: that tensor falls to `resolve_seg_base`'s lazy per-access bump (reserving one
    /// head's worth where all heads are written, so it aliases its neighbour), and every tensor
    /// after it moves.
    ///
    /// This list was rewritten from name strings to roles and silently lost `RunM` and `RunL` —
    /// the running max and running sum the flash fold accumulates into. Nothing failed to compile,
    /// no test moved, and the only visible trace was the emitted body bundle's fingerprint.
    #[test]
    fn the_stick_wide_buffers_are_declared_in_the_original_order() {
        assert_eq!(
            STICK_WIDE_BUFS,
            [
                R::RunM,
                R::RunL,
                R::BMax,
                R::NewM,
                R::Corr,
                R::CorrSubT,
                R::BSum,
                R::LTmp,
                R::Sc,
                R::ExpB,
                R::ESubT,
            ]
        );
        assert_eq!(HEAD_WIDE_BUFS, [R::RunO, R::OTmp]);
    }

    /// Every role `BlockBufs` names is declared by one of the two lists — none may be minted and
    /// then left to the lazy allocator.
    #[test]
    fn every_block_buffer_role_is_declared_to_the_layout() {
        let declared: std::collections::BTreeSet<R> = STICK_WIDE_BUFS
            .iter()
            .chain(HEAD_WIDE_BUFS.iter())
            .copied()
            .collect();
        for r in [
            R::Sc,
            R::BMax,
            R::NewM,
            R::Corr,
            R::CorrSubT,
            R::ExpB,
            R::ESubT,
            R::BSum,
            R::OTmp,
            R::LTmp,
            R::RunM,
            R::RunL,
            R::RunO,
        ] {
            assert!(declared.contains(&r), "{r} is named but never declared");
        }
        assert_eq!(declared.len(), 13, "a declared role has no buffer");
    }
}
