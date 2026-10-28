//! Bridge 3 (`TiledOp -> SdscOp`), AttnDecode composite decomposition — ONE algorithm for any
//! query-row count `mq` (mq=1 decode, mq>1 prefill/chunked-prefill), no separate decode/prefill
//! implementation. Ported from torch-spyre's `spyre__sdpa_overrideable` (`decompositions.py:527`):
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
//! reduce — not the untiled "whole active_cap in one reduce" joint form: the blocked recurrence
//! is this emitter's card-proven form and matches the vendor's own tiling. (The old text
//! justified the blocking with "a documented, real hardware defect where reduce-MAX over more
//! than one stick of columns mis-combines partial per-tile maxima" — an attribution the
//! template's own DDL refutes: the one measured `mxp = 0` cell carried this crate's own
//! row-mixing and unmasked-padding defects, both since fixed. See
//! [`crate::emit::lower_ktir_to_superdsc::reduce`]'s law doc — the wide multi-row max now
//! lowers through the KTIR door and is card-gated. The blocking here is unchanged.)
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
    SharedKernelBmmForm, assemble_matmul_off, assemble_matmul_off_maybe_epilogue,
    assemble_matmul_off_phys_m_maybe_epilogue, assemble_matmul_off_phys_m_with_epilogue,
    assemble_matmul_off_with_epilogue,
};
use super::reduce::{
    ReduceDivided, assemble_reduce_off, assemble_reduce_off_divided, win_reduce_accum,
    win_reduce_data,
};
use crate::addr::{Head, Idx, Nest};
use crate::emit::{
    EmittedOp, In, PointwiseDivided, assemble_pointwise_broadcast_divided,
    assemble_pointwise_broadcast_off, assemble_restickify_kt_2d, fl, rb,
};
use crate::placement::BundleLayout;
use crate::sdsc_abstract::{
    BlockCols, DxGroupCeiling, FeatIdx, FlatTag, KernelTag, KtTileFeats, KtTileSlots, Lanes,
    MaskRows, MatK, MatM, MatN, MatY, PerRequestRows, RowCount, RowWindow, SlotWindow, Stk,
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

/// `hm` at the buffer's TRUE row count — the handle a reduce's accumulator mints through
/// ([`win_reduce_accum`]). The mint is the first reader of a handle's layout as a BOUND, so the
/// one-stick-wide buffers declare their real `[rows, 64]` shape there instead of `hm`'s `1×64`
/// annotation (the synth declarations are `[rows, 64]`, so the mint's shape door and footprint
/// door agree exactly). Same kind (`FlatTag` — `head_major` unchanged) and annotation-only dims:
/// the emit is byte-identical.
fn hm_rows(name: &str, rows: u32) -> Stk<FlatTag> {
    fl(name, rows, BlockCols::of_one_stick(Lanes::FP16).get())
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

/// ⭐ THE ONLINE-SOFTMAX LADDER'S OWN OP COUNT PER FOLD WINDOW — `bmax`, `newm`, `corrsub`, `corre`,
/// `mset`, `esub`, `ee`, `bsum`, `ocorr`, `lcorr`, `ladd`.
///
/// This is the SAME inventory the window-chunk split's projected group size multiplies by `nb`, so
/// it is stated ONCE here — beside the emission it counts — rather than re-derived at the
/// projection site, where a second count could drift from the ladder the sweep actually emits.
/// The seed block (`BlockSeed::SeedsState`) runs a SHORTER ladder (no `newm`/`corrsub`/`corre`/
/// `mset`/`ocorr`), but it is never a fold window: only the prefix fold trips are chunked, and
/// every one of them is `FoldsOntoSeed` with the full ladder below.
pub(crate) const LADDER_OPS_PER_WINDOW: usize = 11;

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
#[derive(Clone, Copy)]
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
    /// WHICH request this op set names — the MINOR coordinate of both row laws
    /// ([`crate::sdsc_abstract::HeadRequestRow`] / [`crate::sdsc_abstract::RequestHeadRow`]).
    ///
    /// An op that sweeps a head's whole `mq` rows leaves it 0 and the request rides inside its block:
    /// the shipped prefix fold does that, and the RUNTIME rebases each pass onto its own request. An op
    /// that computes ONE row names which — the new-token block (emitted once per request, since nothing
    /// rebases it) and the COLLAPSED fold's per-request score/value legs, which reach it through
    /// [`Self::at_request`]. Same row law for all three, which is why it is a field and not a parameter
    /// of `head_row`.
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
            //
            // ⭐ AND THE REQUEST IS THIS BLOCK'S OWN, exactly as the per-request branch above reads it —
            // a hardcoded `0` here was right only while every op swept a head's whole `mq` rows. It is
            // still 0 for every op that does; the collapsed fold's per-request legs re-base these nests
            // onto their own request, which is what lets their kernel be a plain shared 2-D weight.
            BlockRows::WholeBatch(_) => {
                crate::sdsc_abstract::HeadRequestRow::of(h, self.req, self.mq.get()).get()
            }
        };
        Idx::<crate::addr::Row>::n(row)
    }
    fn slabs(&self) -> u32 {
        crate::addr::Shape::<0, 0, 0, 0>::slabs_of(self.hd, crate::superdsc_opspec::Df::Fp16).get()
    }
    /// ⭐⭐⭐⭐⭐ THESE NESTS RE-BASED ONTO **ONE REQUEST** — the collapsed fold's per-request legs, in one
    /// coordinate change.
    ///
    /// Every buffer this block addresses is framed with the request as the row law's MINOR coordinate,
    /// so "request `r`'s slice of the block" is not four offsets composed at a call site: it is the same
    /// four laws asked for a different row. That is what makes a per-request OFFSET a substitute for a
    /// per-request kernel DIM — the shape the card refused — and what keeps the emitter's rows the same
    /// rows the mask's validity is staged into.
    fn at_request(&self, r: u32) -> BlockNests {
        BlockNests { req: r, ..*self }
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
                crate::sdsc_abstract::OperandPlacement::of_head_major_rows(
                    h, self.req, self.mq, rows, width,
                )
            }
        }
    }
    /// The token stream `[mq, nqh*hd]` — head `h`, slab `s`, as the stream's own PLACEMENT: the law
    /// ([`OperandPlacement::of_token_stream`]) owns the named-axis column arithmetic and mints the
    /// plane pitch (`mq` rows) with the offset. Head `h`'s slab `s` is the COLUMN `h*hd + s*stick`,
    /// and the stick plane it lands in is the law's business, not the caller's.
    fn token_stream(&self, h: u32, s: u32) -> crate::sdsc_abstract::OperandPlacement {
        crate::sdsc_abstract::OperandPlacement::of_token_stream(
            self.mq, self.heads, self.hd, h, self.req, s,
        )
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
            self.req,
            s,
            crate::superdsc_opspec::Df::Fp16,
        )
    }
}

/// ⭐⭐⭐⭐⭐ THE COLLAPSED FOLD'S PER-REQUEST KERNEL PLACEMENT — the gathered scratch this block's ops
/// read, and WHICH 64-slot window of it they are.
///
/// ⛔ ONE VALUE BECAUSE BOTH LEGS MOVE TOGETHER. `reps` is a property of the GROUP: the runtime launches
/// the whole fold once per pass, so a pass serves the batch only if EVERY leg in it does. A score leg
/// emitted per request beside a value leg emitted once computes every request's scores and then applies
/// request 0's values to all of them — fluent, wrong, and it would look like a working collapse in the
/// launch count. `Some` therefore means "this block's score AND value legs are per-request".
///
/// ⛔ AND IT CARRIES NO STRIDE, WHICH IS THE WHOLE FIX. It used to carry two `RequestAxis`es — the three
/// `y` steps a per-batch-KERNEL matmul would take. That form is refuted on card (`numCoresUsed_` collapses
/// onto the request count and the launch faults at `job_bin_ptr + cores*128`), so nothing strides across
/// requests any more: each request gets its OWN op, whose three bases are the block's nests re-based by
/// [`BlockNests::at_request`] and whose kernel base is
/// [`crate::sdsc_abstract::GatherScratch::kernel_row_off`].
#[derive(Clone, Copy)]
struct GatheredFold {
    scratch: crate::sdsc_abstract::PageScratch,
    /// The pool, because the gathered kernel base IS the pool's own address plus a request row — see
    /// [`crate::sdsc_abstract::PageScratch::coord_off`]. Carrying it means this type composes no address
    /// arithmetic of its own.
    pool: crate::sdsc_abstract::PagedKvPool,
    /// WHICH 64-slot window of the PAGE these ops read — the same `SlotWindow` the block's own name and
    /// mask slab come from, so the kernel column block and the mask column block cannot describe
    /// different slots.
    ///
    /// ⭐ THE WINDOW SURVIVES THE GRANULARITY CHANGE, AND ONLY THE GATHER'S GRANULARITY CHANGED. The
    /// scratch now holds a whole PAGE per request, so the fold's tile stays a 64-slot window read OUT of
    /// that page — exactly as the UNGATHERED legs already read one out of the pool. That is why the sweep,
    /// the mask blocking and the online-softmax state are untouched by this change.
    window: SlotWindow,
}

impl GatheredFold {
    /// Requests this pass serves — the scratch's own extent, which is also what the host's index table
    /// was staged for.
    fn requests(self) -> u32 {
        self.scratch.mq()
    }
    /// ⭐⭐⭐⭐⭐ THE KERNEL BASE for (this window, kv head `kvh`, request `r`) — **the POOL'S OWN ADDRESS
    /// plus request `r`'s row**, with no arithmetic spelled here.
    ///
    /// ⛔ WHAT THIS REPLACES, AND WHY THE OLD FORM WAS THE BUG. It was `GatherScratch::kernel_row_off`,
    /// which re-derived the window term against a scratch whose rows were 64-slot blocks — a SECOND
    /// derivation of where a window lives, beside [`crate::sdsc_abstract::PagedKvPool::addr`]'s. At page
    /// granularity the scratch row is a byte-for-byte copy of the page plane, so the base is the same
    /// `KvCoord` the ungathered closures build, offset by `r * cols`. One law, both paths.
    ///
    /// `plane` is the operand's own plane: the score leg reads Kᵗ, the value leg V. Passing it means the
    /// two legs cannot silently read one plane's arrangement at the other's offset.
    ///
    /// ⭐⭐⭐⭐⭐ AND IT TAKES THE HEAD-DIM **SLAB**, WHICH IS THE WHOLE hd=128 FIX ON THIS SIDE. Both
    /// gathered legs are one op per (kv head, request, SLAB) — the score leg because a `y`-batched
    /// contraction must be ONE STICK, the value leg because a `y`-batched `out` must be one stick — and
    /// the slab reaches the address the same way the ungathered closures send it: through
    /// [`crate::sdsc_abstract::PagedKvPool::addr`]'s own `at_feat`. A slab added onto a finished offset
    /// here would be a SECOND derivation of where a feature slab lives, which is the exact class of
    /// defect `coord_off` exists to remove.
    ///
    /// ⛔ IT WAS HARDCODED TO SLAB 0 — not as an argument, as an ABSENCE: the two legs never mentioned a
    /// slab, so at hd=128 the score leg contracted two sticks in one op and the value leg wrote only the
    /// lower half of every output head. Both bake clean. See [`crate::sdsc_abstract::PageScratch::of_pass`].
    fn kernel_off(
        self,
        kvh: crate::sdsc_abstract::KvHead,
        r: u32,
        plane: crate::sdsc_abstract::KvPlane,
        slab: u32,
    ) -> Result<crate::addr::DevOff, SuperDscError> {
        let coord = crate::sdsc_abstract::KvCoord::block(plane, kvh)
            .at_slot(self.window.first_slot())
            .at_feat(FeatIdx::of_slab(slab));
        let off = self
            .scratch
            .coord_off(r, &self.pool, coord)
            .ok_or_else(|| {
                SuperDscError(format!(
                    "the collapsed fold's kernel for ({plane:?}, kv head {}, window {}, slab {slab}, \
                     request {r}) is outside the gathered scratch: the scratch holds {} request row(s) \
                     of {} element(s). A base past a row reads into the NEXT request's page — fluent, \
                     wrong, no fault.",
                    kvh.get(),
                    self.window.index(),
                    self.scratch.rows(),
                    self.scratch.cols(),
                ))
            })?;
        u32::try_from(off)
            .map(crate::addr::DevOff::from_view_step)
            .map_err(|_| {
                SuperDscError(format!(
                    "the gathered kernel base for ({plane:?}, kv head {}, request {r}) is {off} \
                     elements, which does not fit the descriptor's offset width",
                    kvh.get(),
                ))
            })
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
    // ⭐⭐⭐⭐⭐ ONE PASS FOR THE WHOLE BATCH: give the score and value legs a REQUEST axis over the
    // GATHERED SCRATCH, so row `h*mq+r` reads request `r`'s own page in the same launch. `Some` only for
    // the prefix fold of a gathered batched decode; `None` everywhere else, which emits exactly what
    // shipped.
    //
    // This is what turns the fold's `pages × requests` launches into `pages`. The runtime half is gated
    // on `OpKv::batched_requests`, and the mask's shape (`MaskBlockForm::PerPage` — one block per page,
    // every row valid on its own request's history) is the same decision seen from the host side. All
    // three move together, and the gather is the precondition for any of them: without it the axis walks
    // an operand with no uniform per-request pitch, which is the fluent-garbage mechanism three earlier
    // attempts hit.
    //
    // ⛔ THE VALUE CARRIES THE STRIDES, NOT A COUNT. It used to be `Option<u32>` — the request count —
    // and the strides were left to be derived from whatever extents were in scope. `RequestAxis` carries
    // each of the three `y` steps differenced out of the buffer that has it, and the builder refuses a
    // walk that would step differently.
    request_axis: Option<GatheredFold>,
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
    // ⭐⭐⭐⭐⭐ THE COLLAPSED-FOLD ARM — **`mq` COPIES OF THE SHIPPED OP, ONE PER REQUEST, IN ONE PASS.**
    //
    // ⛔⛔⛔ THE `y = REQUEST` FORM IS REFUTED ON CARD AND MUST NOT COME BACK. Carrying the requests on
    // `y` needs a per-batch 3-D `[y,in,out]` KERNEL — the one operand in the bundle whose per-core start
    // count goes from ONE to `mq` — and with `mb` pinned to 1 and a one-stick `out` that `y` split IS
    // `numCoresUsed_`. Every rung faulted at `job_bin_ptr + numCoresUsed_*128`, one flit past the
    // program's per-core patch table (`init_binary.bin = 1920 + 128*cores`): rung 2 at +0x100, rung 4 at
    // +0x200, rung 8 at +0x400, syndrome `0xc00` / locator `0x2` / cases `[PrepZeroFlitCnt,PrepSwVer]`
    // bit-identical, only the index moving. Rung 4 is the discriminator that closed it: `{mb:1, y:4}` is
    // `matmul/dims.rs`'s on-hardware-proven solo-decode split, so the `y`-split VALUE is exonerated and
    // the 3-D kernel is what the address is made of.
    //
    // ⭐⭐⭐ AND THE GATHER IS WHAT MAKES THE DIM UNNECESSARY, which is the whole reason the scratch
    // exists. The scratch destination is CONTIGUOUS and request-MINOR: the index already put request
    // `r`'s KV block where `r` needs it, so `r`'s kernel is reached by a baked OFFSET — exactly as a GQA
    // group's shared kv head is reached by `kt_off_fn`'s. So this arm emits the SHIPPED shape: `y` back on
    // the GQA group with a bare 2-D `[in,out]` kernel (`MatY::of_gqa_group` +
    // `assemble_matmul_off_phys_m_with_epilogue`), `mb` = ONE ROW, and the request supplied as a
    // coordinate of all four laws (`BlockNests::at_request`). At `mq == 1` that is BYTE-FOR-BYTE the solo
    // decode op that runs at 41 tok/s today; at `mq > 1` it is that op `mq` times, and `numCoresUsed_` is
    // `{y:4, mb:1}` = 4 at EVERY rung.
    //
    // ⭐ THE OP COUNT IS THE TRADE, AND IT IS THE ONE TO WANT. `nkvh*mq` ops in ONE pass against the
    // shipped `nkvh` ops in each of `mq` passes — the SAME op count, `mq`× fewer launches — and each op
    // now computes ONE row where the shipped one computes `mq` and masks `mq-1` of them away. Measured:
    // 1.42 µs per op against ~28 µs per fold pass.
    //
    // ⛔ AND THIS IS NOT THE PAIRING `attn.rs` FORBIDS. That one PACKS `(gqa, request)` onto a single `y`
    // whose two components share one differenced step — garbage on the 8b at hd=128, 10/10 runs. Here `y`
    // carries the group ALONE, exactly as it ships, and the request is not on an axis at all.
    if let Some(gf) = request_axis {
        // THE KV HEADS FROM THE SCRATCH ITSELF — the same count the index table was staged for, so the
        // kernel row an op reads and the entry the host filled cannot come from two numbers.
        let nkvh_nz = std::num::NonZeroU32::new(gf.scratch.nkvh())
            .ok_or_else(|| SuperDscError("a gathered fold needs at least one kv head".into()))?;
        let gqa = nqh / nkvh_nz.get();
        // ⭐⭐⭐⭐⭐ THE CONTRACTION IS SPLIT BY SLAB — `nslab` ops of ONE STICK each, accumulating in `sc`.
        //
        // ⛔⛔⛔ IT WAS `MatK::of_head_dim(hd)` IN ONE OP, JUSTIFIED BY A COMMENT THAT SAID "the gather's
        // own precondition is `hd <= POOL_STICK`, so there is exactly one slab and no partial sums to
        // accumulate". The precondition was `PageScratch::of_pass`'s `slabs() != 1` refusal, so the prose
        // was true only while the door was shut — and the whole point of the door coming off is that at
        // hd=128 there ARE two slabs. An UNSPLIT `y`-batched contraction of two sticks is the shape
        // `ScoreArm::choose` records as MEASURED-TWICE incoherent inside dxp ("degenerate output at a
        // FASTER ITL, which is the tell: less work, done wrong"), and it is what the 8b's
        // wrong-from-the-first-token runs were emitting.
        //
        // ⭐ AND `OneStickContraction::by_slab_split` IS THE WITNESS, which is why it takes no arguments:
        // a slab IS one stick at every head dim, so the split satisfies the precondition by construction
        // rather than by a head-dim test. The batched form's two stride relations hold because each
        // operand declares its OWN pitch (`BatchStrides::{a_pitch,o_pitch}`).
        //
        // ⛔⛔⛔ AND THE KERNEL **VIEW** IS NOT WHAT WAS WRONG, WHICH IS WHY READING THE TEMPLATE WOULD
        // NOT HAVE FOUND THIS. The fp16 score kernel's slice layout is ONE-DIM ON `%out`
        // (`deeptools/share/ddc/ddl_templates/bmm.ddl:23`:
        // `%slice_layout_kernel_16bit = ddl.layout(%out) {is_order_fixed=true}`), i.e. sticked on SLOTS,
        // so `hd` is a non-stick reduction extent and a two-stick contraction is perfectly expressible in
        // the view. It is nonetheless incoherent on the CARD under a `y`-batch — measured twice, and the
        // template cannot say so. The lesson is the one this file keeps relearning: the declaration being
        // legal is not the machine agreeing.
        let (k, n) = (
            MatK::of_head_slab(FeatIdx::SLAB_FEATS),
            MatN::of_kv_window(width),
        );
        for r in 0..gf.requests() {
            let rn = nests.at_request(r);
            for kvh in crate::sdsc_abstract::KvHead::all(nkvh_nz) {
                let qh0 = kvh.group_first_query(crate::addr::Gqa::new(gqa));
                let h0 = qh0.get();
                // The mask under a collapsed pass is `MaskBlockForm::PerPage` and read PER ROW: this
                // op's row block is the same `score_rows` slice `sc` itself writes, so the validity it
                // adds is request `r`'s own. `mask_bcast` cannot be true here — a broadcast row would
                // give every request row 0's validity — so there is no arm for it.
                let row_mask_off = mask_off + rn.score_rows(h0, width).off();
                for s in 0..nslab {
                    // ⭐ THE MASK IS ADDED ON SLAB 0 ONLY — `sc = sum_s (Q_s . Kt_s) + mask`. Adding it
                    // per slab adds it `nslab` times, which is INERT at hd=64 and wrong above it: the
                    // same shape as every other defect this file records. Later slabs fuse `sc` itself
                    // at this group's own offset, so the op computes `sc += Q_s . Kt_s` (the DDL epilogue
                    // is SFP-resident with one store, so the read of the prior partial precedes it).
                    let (epi_h, epi_off): (&Stk<FlatTag>, crate::addr::DevOff) = if s == 0 {
                        (&mask_h, row_mask_off)
                    } else {
                        (&sc_h, rn.score_rows(h0, width).off())
                    };
                    // nslab == 1 keeps the op's NAME, so granite-3.1-2b's emission does not move.
                    let name = if nslab == 1 {
                        format!("attn_{tag}sc_g{}_r{r}_o{t}", kvh.get())
                    } else {
                        format!("attn_{tag}sc_g{}s{s}_r{r}_o{t}", kvh.get())
                    };
                    ops.push(assemble_matmul_off_phys_m_with_epilogue(
                        // THE KV HEAD AND THE REQUEST AS SEPARATE NAME SEGMENTS, so the descriptor
                        // projections that collapse `g{n}` / `r{n}` to one reported row keep working.
                        &name,
                        // ⛔ ONE ROW OF WORK. This op computes request `r`'s single score row for each
                        // head of the group; declaring `mq` rows is what makes a pass compute `mq` rows
                        // to keep one, which is the arithmetic the collapse is removing.
                        MatM::single_row(),
                        n,
                        k,
                        MatY::of_gqa_group(
                            gqa,
                            // ⛔ THE BY-SLAB STREAM PLACEMENT, NOT THE PLAIN ONE. A one-stick `in`
                            // derives its head step as `pitch * in`, so the stream's own `mq` pitch
                            // would stride 64 where the heads are `mq*hd` apart. Same law the
                            // ungathered slab-split arm takes.
                            rn.token_stream_by_slab(h0, s),
                            rn.score_rows(h0, width),
                        ),
                        score_form,
                        &rb(qs, mq_n, hd),
                        rn.token_stream_by_slab(h0, s),
                        // ⭐ THE GATHERED SCRATCH AS THE KERNEL, AT ITS FULL DECLARED `hd` ROWS with the
                        // slab selected purely by OFFSET — the same discipline the ungathered arm uses.
                        // Declaring `stick` rows instead would re-derive the stick-GROUP stride as
                        // `stick*stk` where the tensor's is `hd*stk`.
                        &Stk::<KernelTag>::kernel(hd as usize, kt_stride.n_out_cols(), kt_kernel),
                        gf.kernel_off(kvh, r, crate::sdsc_abstract::KvPlane::Kt, s)?,
                        &rb(sc, rows_n, width_n),
                        rn.score_rows(h0, width).off(),
                        epi_h,
                        epi_off,
                        crate::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                        &[],
                        false,
                        sym_id_base,
                        layout,
                    ));
                }
            }
        }
    }
    // TWO SEPARATE QUESTIONS, ASKED SEPARATELY: `b.score` is the LEGALITY witness (now unconditional —
    // the contraction is slab-split), and `ScoreArm::cheaper` is the COST choice on top of it. Folding
    // the cost into the witness would have made a hd=512 op-count REGRESSION look like a legality fact.
    else if let Some(nkvh_b) = batched
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
                        MatK::of_head_dim(hd),
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
                    //
                    // ⭐⭐⭐ THIS IS THE ARM THE SHIPPED PREFIX SCORE LEG TAKES, so it is the arm the
                    // gather has to be on. `per_req` is a hardcoded `false` (see its own note: the
                    // row-batched experiment was written against a fold that had collapsed to one
                    // pass), which means the `per_request` arm above is DEAD for the prefix block —
                    // a gather placed only there would appear in no bundle at all, which is the
                    // "a port with no caller is dead code" trap and was the plan of record until this
                    // file's own descriptor test looked at the emitted JSON instead of the call site.
                    assemble_matmul_off_phys_m_with_epilogue(
                        &name,
                        MatM::of_query_rows(mq),
                        MatN::of_kv_window(width),
                        MatK::of_head_slab(FeatIdx::SLAB_FEATS),
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
                MatK::of_head_dim(hd),
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
    let bmax_op = format!("attn_{tag}bmax_o{t}");
    ops.push(assemble_reduce_off(
        &bmax_op,
        "max",
        rows.swept(),
        width,
        win_reduce_data(
            layout,
            // The score buffer's true `[rows_n, width_n]` — the same dims the score matmul's own
            // output handle declares. `hm(sc)`'s `1×64` annotation was never the buffer's shape.
            &fl(sc, rows_n, width_n),
            crate::addr::DevOff::ZERO,
            rows.swept(),
            width,
            &format!("{bmax_op}: scores"),
        )
        .map_err(SuperDscError)?,
        win_reduce_accum(
            layout,
            &hm_rows(if seed.seeds() { run_m } else { bmax }, rows_n),
            crate::addr::DevOff::ZERO,
            rows.swept(),
            &format!("{bmax_op}: accum"),
        )
        .map_err(SuperDscError)?,
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
            &[
                In::full(&hm_rows(run_m, rows_n)).ew(),
                In::full(&hm_rows(bmax, rows_n)).ew(),
            ],
            &hm_rows(newm, rows_n),
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
            &[
                In::full(&hm_rows(run_m, rows_n)).ew(),
                In::full(&hm_rows(newm, rows_n)).ew(),
            ],
            &hm_rows(corrsubt, rows_n),
            crate::addr::DevOff::ZERO,
            sym_id_base,
            layout,
        ));
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_{tag}corre_o{t}"),
            "exp",
            rows.swept(),
            BlockCols::of_one_stick(Lanes::FP16),
            &[In::full(&hm_rows(corrsubt, rows_n)).ew()],
            &hm_rows(corr, rows_n),
            crate::addr::DevOff::ZERO,
            sym_id_base,
            layout,
        ));
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_{tag}mset_o{t}"),
            "identity",
            rows.swept(),
            BlockCols::of_one_stick(Lanes::FP16),
            &[In::full(&hm_rows(newm, rows_n)).ew()],
            &hm_rows(run_m, rows_n),
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
        &[
            // `sc`/`esubt` at the swept `[rows_n, width_n]` — the same dims the score matmul's
            // own output handle declares, not `hm`'s `1×64` annotation.
            In::full(&fl(sc, rows_n, width_n)).ew(),
            In::col(&hm_rows(run_m, rows_n)).ew(),
        ],
        &fl(esubt, rows_n, width_n),
        crate::addr::DevOff::ZERO,
        sym_id_base,
        layout,
    ));
    ops.push(assemble_pointwise_broadcast_off(
        &format!("attn_{tag}ee_o{t}"),
        "exp",
        rows.swept(),
        width,
        &[In::full(&fl(esubt, rows_n, width_n)).ew()],
        &fl(expb, rows_n, width_n),
        crate::addr::DevOff::ZERO,
        sym_id_base,
        layout,
    ));
    // FIRST block: write bsum/ov straight into run_l/run_o (no lset/oset copy needed — same reasoning
    // as run_m above). Saves 2 more "identity" ops per attention node.
    let bsum_op = format!("attn_{tag}bsum_o{t}");
    ops.push(assemble_reduce_off(
        &bsum_op,
        "sum",
        rows.swept(),
        width,
        win_reduce_data(
            layout,
            // `expb` at the same `[rows_n, width_n]` the `exp` pointwise above sweeps it at.
            &fl(expb, rows_n, width_n),
            crate::addr::DevOff::ZERO,
            rows.swept(),
            width,
            &format!("{bsum_op}: probabilities"),
        )
        .map_err(SuperDscError)?,
        win_reduce_accum(
            layout,
            &hm_rows(if seed.seeds() { run_l } else { bsum }, rows_n),
            crate::addr::DevOff::ZERO,
            rows.swept(),
            &format!("{bsum_op}: accum"),
        )
        .map_err(SuperDscError)?,
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
                // `run_o`/`otmp` at their true head-wide `[rows_n, hd]` — the HEAD_WIDE_BUFS synth
                // dimensions, not `hm`'s `1×64` annotation.
                In::sliced(&fl(run_o, rows_n, hd), nests.head_major(0, 0)).ew(),
                In::col(&hm_rows(corr, rows_n)).ew(),
            ],
            &fl(otmp, rows_n, hd),
            nests.head_major(0, 0),
            sym_id_base,
            layout,
        ));
    }
    // `Some(otmp)` = fold this block in via `stridedadd`; `None` = the seed block writes its own
    // result. The HANDLE is minted once here so the three value-matmul arms below cannot each decide
    // separately whether this block folds — they only choose WHERE in `otmp` their own slab sits.
    let fold_addend = (!seed.seeds()).then_some(otmp).map(|b| fl(b, rows_n, hd));
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
    // ⭐⭐⭐⭐⭐ THE COLLAPSED FOLD'S VALUE LEG — the score leg's mirror: `mq` copies of the SHIPPED
    // per-kv-head op, one per request, `y` on the GQA group, the V scratch row reached by a baked OFFSET.
    // See the score arm above for why the per-`y` kernel DIM is refuted and what the gather buys instead.
    if let Some(gf) = request_axis {
        let nkvh_nz = std::num::NonZeroU32::new(gf.scratch.nkvh())
            .ok_or_else(|| SuperDscError("a gathered fold needs at least one kv head".into()))?;
        let gqa = nqh / nkvh_nz.get();
        // ⭐⭐⭐⭐⭐ ONE STICK OF `out` PER OP, SO ONE OP PER (KV HEAD, REQUEST, **SLAB**) — the same cut
        // the ungathered batched arm below already makes, for the same reason.
        //
        // ⛔⛔⛔ THIS LEG HAD `n = MatN::of_head_slab(SLAB_FEATS)` — ONE STICK — AND **NO SLAB LOOP**,
        // under a comment saying "ONE slab, because the gather's own precondition is `hd <= POOL_STICK`".
        // At hd=64 one stick IS the whole head dim, so the absence was invisible. At hd=128 it means the
        // gathered fold wrote only feature slab 0 of `run_o`: the upper 64 features of EVERY head's
        // attention output received no prefix contribution at all, keeping only the new-token block's
        // seed. Clean bake, no fault, every row wrong from its first generated token — which is exactly
        // what `PageScratch::of_pass`'s refused hd=128 measurement recorded.
        //
        // ⛔ AND `out` MUST STAY ONE STICK, which is why the fix is a LOOP and not a wider `n`. A batched
        // matmul reaches head `h` by striding `y` and derives that stride as `mb*out`; the accumulators
        // are head-major `[rows, hd]` whose real pitch is `mq*stick`, so `out = hd` derives `mq*hd` and
        // agrees only at one stick. The slab is selected by the output's own OFFSET
        // (`of_head_major_accum(.., s)`) and the kernel's (`at_feat`), never by a fourth walk axis.
        let (k, n) = (
            MatK::of_kv_window(width),
            MatN::of_head_slab(FeatIdx::SLAB_FEATS),
        );
        for r in 0..gf.requests() {
            let rn = nests.at_request(r);
            for kvh in crate::sdsc_abstract::KvHead::all(nkvh_nz) {
                let qh0 = kvh.group_first_query(crate::addr::Gqa::new(gqa));
                let h0 = qh0.get();
                for s in 0..nests.slabs() {
                    // nslab == 1 keeps the op's NAME, so granite-3.1-2b's emission does not move.
                    let name = if nests.slabs() == 1 {
                        format!("attn_{tag}ov_g{}_r{r}_o{t}", kvh.get())
                    } else {
                        format!("attn_{tag}ov_g{}s{s}_r{r}_o{t}", kvh.get())
                    };
                    ops.push(assemble_matmul_off_phys_m_maybe_epilogue(
                        &name,
                        MatM::single_row(),
                        n,
                        k,
                        // The same two laws the shipped batched value arm declares — a head-major
                        // probability buffer read and a head-major accumulator written, both `mq*stick`
                        // apart — asked at THIS request's row and THIS slab.
                        MatY::of_gqa_group(
                            gqa,
                            rn.score_rows(h0, width),
                            crate::sdsc_abstract::OperandPlacement::of_head_major_accum(
                                rows.swept(),
                                hd,
                                mq,
                                h0,
                                r,
                                s,
                            ),
                        ),
                        bmm_form,
                        &rb(expb, rows_n, width_n),
                        rn.score_rows(h0, width),
                        // ⭐ THE V SCRATCH AS THE KERNEL, AND ITS ROW COUNT IS NOW THE PAGE — the scratch
                        // row is a byte-for-byte copy of the V page plane, so `v_stride` is the pool's own
                        // `PAGE_SLOTS` on both the gathered and ungathered paths. It used to be ONE
                        // WINDOW, which is what made the gathered geometry differ from the pool's at all.
                        &Stk::<KernelTag>::kernel(v_stride as usize, hd as usize, v_kernel),
                        gf.kernel_off(kvh, r, crate::sdsc_abstract::KvPlane::V, s)?,
                        &rb(run_o, rows_n, hd),
                        rn.head_major(h0, s),
                        // The addend is THIS op's own output slice of `otmp` — same buffer geometry, same
                        // offset, so `attach_fused_epilogue` clones the output's shape onto it verbatim.
                        fold_addend.as_ref().map(|a| (a, rn.head_major(h0, s))),
                        crate::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                        &[],
                        false,
                        sym_id_base,
                        layout,
                    ));
                }
            }
        }
    }
    // The value leg's mirror: plain 2-D, one per (request, query head), nothing declared.
    else if let Some(nkvh_b) = batched.and_then(|b| b.value.map(|_| b.nkvh)) {
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
                                nests.req,
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
                                nests.req,
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
            &[
                In::full(&hm_rows(run_l, rows_n)).ew(),
                In::full(&hm_rows(corr, rows_n)).ew(),
            ],
            &hm_rows(ltmp, rows_n),
            crate::addr::DevOff::ZERO,
            sym_id_base,
            layout,
        ));
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_{tag}ladd_o{t}"),
            "add",
            rows.swept(),
            BlockCols::of_one_stick(Lanes::FP16),
            &[
                In::full(&hm_rows(ltmp, rows_n)).ew(),
                In::full(&hm_rows(bsum, rows_n)).ew(),
            ],
            &hm_rows(run_l, rows_n),
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

/// The split-KV chunk's own buffer NAMES — the gather scratches and the per-kv-head wide blocks.
/// The LX-resident gather's tensors: the law, and each leg's core-local tensor names — V's by
/// [`crate::sdsc_abstract::ChunkLx::lx_tensor`], Kᵗ's by [`crate::sdsc_abstract::ChunkKeysLx::tensor`].
/// Kᵗ's exist only with its witness ([`crate::sdsc_abstract::ChunkLx::keys`], a chunk with no Kᵗ tail).
struct ChunkLxBufs {
    form: crate::sdsc_abstract::ChunkLx,
    keys: Option<(crate::sdsc_abstract::ChunkKeysLx, Vec<String>)>,
    v: Vec<String>,
}

impl ChunkLxBufs {
    /// Where one V copy lands: its core-local tensor, or V's tail scratch.
    fn dest<'a>(names: &'a [String], tail: &'a str, d: crate::sdsc_abstract::LxDest) -> &'a str {
        match d {
            crate::sdsc_abstract::LxDest::Lx(i) => names[i as usize].as_str(),
            crate::sdsc_abstract::LxDest::Tail => tail,
        }
    }
}

struct ChunkBufs {
    gkt: String,
    gv: String,
    // ⭐ THE GATHER IN LX, where the chunk's geometry lets the copies' division be the score's and PV's
    // (`ChunkScratch::lx_form`): one core-local tensor per (part, kv head) for Kᵗ and per (part, kv head,
    // slab) for V. `gkt` and `gv` then hold only what stays in HBM: Kᵗ whole when it has a tail, V's
    // tail pages. `None` keeps the HBM scratches for the whole chunk.
    lx: Option<ChunkLxBufs>,
    csc: String,
    cesub: String,
    cexp: String,
    /// ⭐ HOW `esub` AND `exp` DIVIDE THEIR WORK, and so how the core-local `cesub` between them is laid
    /// out. The caller sizes the pin under this same value ([`ChunkBufs::sm_division_of`]), and the body
    /// assembles both ops under it.
    sm_division: crate::work::EwDivision,
    /// ⭐ HOW THE ROW MAX AND THE ROW SUM SPEND THEIR CORES ([`ChunkBufs::red_division_of`]): the reduced
    /// axis across the cores for a decode batch, rows first for every other chunk.
    red_division: super::reduce::ReduceDivision,
    /// One-stick partial reductions: three for a row max / denominator cut into parts, one per probability
    /// part plus two accumulators for a prompt chunk's denominator. Declared only when used.
    red: Vec<String>,
    /// ⭐ THE RE-LAID QUERY, where the row blocks are query groups ([`ChunkRowForm::QueryGroup`]): the
    /// group-major block's name and the identity table its copies read. `None` for every other form.
    ///
    /// [`ChunkRowForm::QueryGroup`]: crate::sdsc_abstract::ChunkRowForm::QueryGroup
    qgroup: Option<QGroupBufs>,
}

impl ChunkBufs {
    /// ⭐⭐⭐ A DECODE BATCH's PROBABILITIES ARE DIVIDED STICKS FIRST — every core takes ALL of a row block's
    /// rows over a run of whole sticks. Every other chunk keeps the law, which gives rows first.
    ///
    /// A row block's `[rows, part]` score block is stick-major (`[part/64][rows][64]`). Rows first, each core
    /// reads its rows as `part/64` strided 128-byte bursts. Sticks first, each core reads one contiguous run
    /// of whole sticks. MEASURED (granite-3.1-2b fp16, conc-8, ~1k-token prompts: a probe that cut the batch
    /// chunk's softmax into launch groups of its own): at 8 requests the 8 `esub`+`exp` pairs, two HBM passes
    /// over each kv head's `[32, 10240]` block, took 12.38 ms/step, and the row sum, one pass, took 6.70 ms.
    /// That is 31–34 GB/s per pass: the strided rate, with `exp`'s transcendental costing nothing past its
    /// pass. The same law on the stick-major MLP ops measured 50.4 → 103.3 GB/s (`work::EwDivision`).
    ///
    /// ⛔ WHY ONLY THE BATCH. The law already gives sticks first to a stick-major op, but not to one touching
    /// a core-local pin: such an op keeps the division its pin was laid out under, and the pin was sized rows
    /// first. So the batch states its division outright, and this one value sizes the pin and divides both
    /// ops. The other chunks stay as they are:
    ///   * the one-request decode's softmax shares a launch group with its score, so it was not measured
    ///     apart, and its forms are card-proven as they stand. This leaves its bundles byte-identical;
    ///   * a prompt chunk keeps `cexp` core-local through its row sum and PV (`prompt_probabilities_in_lx`),
    ///     whose PV divides its ROWS. A sticks-first `cexp` would hand the PV's cores slices they never
    ///     wrote, and the read law would refuse it.
    fn sm_division_of(rows_are_requests: bool) -> crate::work::EwDivision {
        if rows_are_requests {
            crate::work::EwDivision::SticksFirst
        } else {
            crate::work::EwDivision::Law
        }
    }

    /// ⭐⭐⭐ A DECODE BATCH's ROW MAX AND ROW SUM SPLIT THE AXIS THEY REDUCE ACROSS THE CORES — every core
    /// reduces a contiguous run of whole sticks of every row of the `[rows, slots]` block, and the reduce
    /// template's own `genericpartialreduction` ring merges the 32 partials into the row's one result
    /// (`ReduceDivision::ReducedAxis`). Every other chunk keeps rows first.
    ///
    /// MEASURED (granite-3.1-2b fp16, conc-8, ~1k-token prompts, each in a launch group of its own): at 8
    /// requests the 8 row maxes took 6.72 ms/step and the 8 row sums 6.72 ms/step, 21 µs per kv head over
    /// `[32, 10240]`. That is 31 GB/s: rows first, each core reads its one row as 160 strided 128-byte
    /// bursts, the same rate `esub`/`exp` ran at before they went sticks first (31 → 127 GB/s).
    ///
    /// ⛔ WHY ONLY THE BATCH: the same reasons as [`Self::sm_division_of`]. The one-request decode's
    /// reductions share a launch group with its score and are card-proven as they stand, and a prompt
    /// chunk's row sum reads a core-local `cexp` laid out for its row-divided PV. Its row max reads the
    /// HBM score block, but it is left as it is, so every non-batch bundle stays byte-identical.
    fn red_division_of(rows_are_requests: bool) -> super::reduce::ReduceDivision {
        if rows_are_requests {
            super::reduce::ReduceDivision::ReducedAxis
        } else {
            super::reduce::ReduceDivision::Rows
        }
    }
}

/// The re-laid query block and the `[hd, hd]` identity its stick copies read.
struct QGroupBufs {
    block: String,
    ident: String,
}

/// ⭐⭐⭐ WHOSE HISTORY A CHUNK's ROWS ATTEND — the one fact that separates the batch chunk from the solo
/// and prompt chunks, and it decides exactly two things: how the prefix validity is read, and whether
/// the value leg's contraction is divided across the cores.
#[derive(Clone, Copy)]
enum ChunkHistory {
    /// Every row shares ONE resident history — the one-request decode (one row per head) and a prompt
    /// chunk's consecutive positions. The validity is one broadcast row over the chunk's logical slots.
    Shared,
    /// Every row is a separate REQUEST, over its own run of the chunk's pages ([`BatchChunk`]). The
    /// validity is per row: ONE `[rows, slots]` block, read by every row block at offset 0, in the score
    /// block's own arrangement.
    ///
    /// [`BatchChunk`]: crate::sdsc_abstract::BatchChunk
    Requests(crate::sdsc_abstract::BatchChunk),
}

/// What one layer's split-KV chunk attends with: its geometry, the pool planes and index it gathers
/// through, the query and mask it scores, and the buffers it writes.
struct ChunkAttn<'a> {
    t: u32,
    nqh: u32,
    hd: u32,
    gqa: u32,
    // Query rows per head: 1 at decode, the prompt chunk's width at prefill, the batch width for a
    // batched decode.
    mq: u32,
    chunk: crate::sdsc_abstract::ChunkScratch,
    history: ChunkHistory,
    index: &'a str,
    kct: &'a str,
    vc: &'a str,
    qs: &'a str,
    pmask: &'a str,
    bmm_form: SharedKernelBmmForm,
    bufs: &'a BlockBufs,
    cb: &'a ChunkBufs,
}

/// ⭐ ROW BLOCK `b`'s QUERY OPERAND for a score matmul — one law for the split-KV chunk's score and the
/// chunk-formed causal block's. At one slab the block is a GQA group and Q is its `[gqa·mq, hd]` rows of
/// `qs` read flat. Above it the block is one query head and Q is that head's own `[mq, hd]` block of the
/// token stream — the fold's per-head score form (`ScoreArm::PerHead`): the WHOLE head dim contracted in
/// one UNBATCHED op. (A two-stick contraction under a `y`-batch is the measured-incoherent shape;
/// unbatched it is gemma-4's full-attention score.)
fn row_block_query(
    blocks: crate::sdsc_abstract::ChunkRowBlocks,
    b: u32,
    mq: u32,
    nqh: u32,
    hd: u32,
    qs: &str,
) -> Result<
    (
        MatM,
        Stk<crate::sdsc_abstract::RowBlockedTag>,
        crate::addr::DevOff,
    ),
    SuperDscError,
> {
    use crate::sdsc_abstract::{ChunkRowForm, OperandPlacement, QueryRowCount};
    Ok(match blocks.form() {
        ChunkRowForm::OneSlabGroup => (
            MatM::of_head_major_rows(blocks.rows()),
            rb(qs, nqh * mq, hd),
            crate::addr::DevOff::from_view_step(
                blocks.first_head(b) * mq * BlockCols::of_one_stick(Lanes::FP16).get(),
            ),
        ),
        ChunkRowForm::QueryHead => {
            let rows = QueryRowCount::of_mq(mq);
            (
                MatM::of_query_rows(rows),
                rb(qs, mq, hd),
                OperandPlacement::of_token_stream(rows, nqh, hd, blocks.first_head(b), 0, 0).off(),
            )
        }
        // ⭐ THE GROUP's WHOLE `[gqa, hd]` BLOCK of the re-laid query — a standalone RowBlocked block, so
        // the score reads it as a tensor of exactly its own rows (never a row slice of a taller one).
        // `qs` here is the re-laid buffer's name (the caller hands it in place of the stream's).
        ChunkRowForm::QueryGroup(q) => (
            MatM::of_head_major_rows(q.rows()),
            rb(qs, q.rows(), hd),
            q.block_off(blocks.kv_head(b)).ok_or_else(|| {
                SuperDscError(format!(
                    "row block {b}: kv head {} is past the re-laid query's {} groups",
                    blocks.kv_head(b),
                    q.nkvh()
                ))
            })?,
        ),
    })
}

/// ⭐ ROW BLOCK `b`'s VALUE LEG at feature slab `s` — one law for the split-KV chunk's PV and the
/// chunk-formed causal block's. At one slab: ONE op per block, `out = hd`, landing at the block's rows of
/// the `[q_rows, hd]` output. Above it: ONE op per (query head, slab), one stick of `out` each, landing in
/// slab `s`'s plane at the head's rows — the fold's own card-proven hd=128 form. ONE op per head with
/// `out = hd` writes `run_o` incoherently above one stick (MEASURED on granite-3.1-8b: fluent nonsense,
/// bisected to exactly that collapse — see the fold's value leg).
#[derive(Clone, Copy)]
struct RowBlockValue {
    m: MatM,
    n: MatN,
    /// The V kernel's columns: the whole head dim at one slab, one stick above it.
    v_cols: u32,
    /// The block's rows of the output, in slab `s`'s plane above one slab.
    out_off: crate::addr::DevOff,
}

impl RowBlockValue {
    /// The value ops per row block: one at one slab, one per slab above it.
    fn slabs(blocks: crate::sdsc_abstract::ChunkRowBlocks, hd: u32) -> u32 {
        use crate::sdsc_abstract::ChunkRowForm;
        match blocks.form() {
            ChunkRowForm::OneSlabGroup => 1,
            ChunkRowForm::QueryHead | ChunkRowForm::QueryGroup(_) => {
                hd / BlockCols::of_one_stick(Lanes::FP16).get()
            }
        }
    }

    fn of(
        blocks: crate::sdsc_abstract::ChunkRowBlocks,
        b: u32,
        s: u32,
        mq: u32,
        q_rows: u32,
        hd: u32,
    ) -> RowBlockValue {
        use crate::sdsc_abstract::ChunkRowForm;
        let stick = BlockCols::of_one_stick(Lanes::FP16).get();
        let h0 = blocks.first_head(b) * mq;
        // Slab `s`'s plane of the `[q_rows, hd]` output at the block's first row — every row of a
        // one-stick output step is one stick, so a block's `rows` consecutive rows are one contiguous
        // run of that plane.
        let slab_rows = Nest::new(&["row", "feat"], &[q_rows, hd], Df::Fp16)
            .view()
            .at(Idx::<crate::addr::Row>::n(h0))
            .slab(s)
            .dev();
        match blocks.form() {
            ChunkRowForm::OneSlabGroup => RowBlockValue {
                m: MatM::of_head_major_rows(blocks.rows()),
                n: MatN::of_head_dim(hd),
                v_cols: hd,
                out_off: crate::addr::DevOff::from_view_step(h0 * stick),
            },
            ChunkRowForm::QueryHead => RowBlockValue {
                m: MatM::of_query_rows(crate::sdsc_abstract::QueryRowCount::of_mq(mq)),
                n: MatN::of_head_slab(FeatIdx::SLAB_FEATS),
                v_cols: stick,
                out_off: slab_rows,
            },
            // ⭐ THE GROUP's `gqa` ROWS of slab `s` in ONE op — one stick of `out`, the fold's card-proven
            // hd=128 width; only the row count grows, and the group's rows are contiguous in the plane.
            ChunkRowForm::QueryGroup(q) => RowBlockValue {
                m: MatM::of_head_major_rows(q.rows()),
                n: MatN::of_head_slab(FeatIdx::SLAB_FEATS),
                v_cols: stick,
                out_off: slab_rows,
            },
        }
    }
}

/// Row block `b`'s op-name stem: its kv head when a block is a GQA group, its query head otherwise.
fn row_block_name(blocks: crate::sdsc_abstract::ChunkRowBlocks, b: u32) -> String {
    use crate::sdsc_abstract::ChunkRowForm;
    match blocks.form() {
        ChunkRowForm::OneSlabGroup | ChunkRowForm::QueryGroup(_) => {
            format!("g{}", blocks.kv_head(b))
        }
        ChunkRowForm::QueryHead => format!("h{}", blocks.first_head(b)),
    }
}

/// The chunk-formed causal new-token block's operands ([`NewBlockForm::Chunk`]) — see
/// [`assemble_new_block_chunk`].
///
/// [`NewBlockForm::Chunk`]: crate::sdsc_abstract::NewBlockForm::Chunk
struct NewChunkAttn<'a> {
    t: u32,
    nqh: u32,
    nkvh: u32,
    hd: u32,
    /// The row blocks the block is cut into ([`ChunkRowBlocks`]): a kv head at one slab, a query head above.
    ///
    /// [`ChunkRowBlocks`]: crate::sdsc_abstract::ChunkRowBlocks
    blocks: crate::sdsc_abstract::ChunkRowBlocks,
    /// The chunk's rows per head — equal to `mq_pad` by [`NewBlockForm::of`]'s own condition.
    ///
    /// [`NewBlockForm::of`]: crate::sdsc_abstract::NewBlockForm::of
    mq: u32,
    cols: BlockCols,
    qs: &'a str,
    new_kt: &'a str,
    new_v: &'a str,
    cmask: &'a str,
    bmm_form: crate::ir::bridge::tiled_op_sdsc_op::matmul::SharedKernelBmmForm,
    bufs: &'a BlockBufs,
    sc: &'a str,
    esub: &'a str,
    exp: &'a str,
}

/// ⭐⭐⭐ THE CAUSAL NEW-TOKEN BLOCK AS ONE BLOCK PER ROW BLOCK — the split-KV chunk's own form, over the
/// chunk's own `mq_pad` keys, SEEDING the online softmax that the resident prefix then folds onto.
///
/// The sub-blocked form sweeps the block one 64-key window at a time, per QUERY head, so its op count
/// grows with the chunk's width — at 256 prompt rows it costs about what halving the chunk count saves in
/// weight streaming (see [`NewBlockForm`]). Here each row block (a kv head's GQA group at one feature slab,
/// one query head above it — the chunk's [`ChunkRowBlocks`]) runs a fixed set over its whole
/// `[rows, mq_pad]` block, at any width:
///   * score `Q_b · Kᵗ_kvh + cmask_b` — `new_kt`'s kv-head block is already the `[hd, mq_pad]` kernel (the
///     restickifies above fill it), and the causal mask rides in the matmul's epilogue as the full
///     per-row addend it is (staged by row block, [`NewBlockForm::mask_heads_per_block`]);
///   * each row's max STRAIGHT into `run_m` — the seed has no prior state to merge with;
///   * `esub` into a CORE-LOCAL LX scratch and `exp` out of it (the chunk's own residency, #501);
///   * each row's sum straight into `run_l`, and `P_b · V_kvh` straight into `run_o` — one op per slab
///     above one slab ([`RowBlockValue`]). `new_v`'s kv-head block is exactly `mq` = `mq_pad` rows (RoPE
///     packs it `[nkvh][mq][hd]`), so the PV contracts it whole.
///
/// Every wide buffer is a WHOLE `[rows, mq_pad]` block per row block (`[blocks][rows][mq_pad]`), the
/// chunk's law: dxp rebuilds a stick-blocked operand's group stride from the op's own row count, so a row
/// slice of a taller multi-stick buffer is read wrong on card ([`ChunkScratch::rows_corner`]).
///
/// [`ChunkRowBlocks`]: crate::sdsc_abstract::ChunkRowBlocks
///
/// [`NewBlockForm`]: crate::sdsc_abstract::NewBlockForm
/// [`NewBlockForm::mask_heads_per_block`]: crate::sdsc_abstract::NewBlockForm::mask_heads_per_block
/// [`ChunkScratch::rows_corner`]: crate::sdsc_abstract::ChunkScratch::rows_corner
fn assemble_new_block_chunk(
    ops: &mut Vec<EmittedOp>,
    a: NewChunkAttn<'_>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<(), SuperDscError> {
    let NewChunkAttn {
        t,
        nqh,
        nkvh,
        hd,
        blocks,
        mq,
        cols,
        qs,
        new_kt,
        new_v,
        cmask,
        bmm_form,
        bufs,
        sc,
        esub,
        exp,
    } = a;
    use crate::addr::DevOff;
    use crate::superdsc_opspec::EpilogueOpFunc;
    let stick = BlockCols::of_one_stick(Lanes::FP16);
    let slots = cols.get();
    let (g_rows, q_rows) = (blocks.rows(), nqh * mq);
    let grp_rows = RowCount::of_head_major_rows(g_rows);
    // A row block's query rows are the contiguous `[first_head·mq, …)` of the head-major token stream; in
    // the one-stick state buffers that is a single-stick view step.
    let row_step = |h: u32| DevOff::from_view_step(h * stick.get());
    // The whole-block laws, each from its nest's own footprint.
    let wide = crate::addr::Nest::new(&["row", "slot"], &[g_rows, slots], Df::Fp16);
    let kt = crate::addr::Nest::new(&["feat", "slot"], &[hd, slots], Df::Fp16);
    let vv = crate::addr::Nest::new(&["row", "head", "feat"], &[mq, nkvh, hd], Df::Fp16);
    let first_op = ops.len();
    for b in 0..blocks.count() {
        let (kvh, h0, bn) = (
            blocks.kv_head(b),
            blocks.first_head(b) * mq,
            row_block_name(blocks, b),
        );
        // ⭐ THE SCORE AND THE PROBABILITIES ARE CORE-LOCAL LX, ONE ROW BLOCK AT A TIME. Each is one core's
        // slice of ONE row block (`sc`, `esub`, `exp`, declared by the caller), reused block after block,
        // so every op addresses them at the pin — offset 0, no block corner. Every op here divides the
        // block's ROWS alone, the score too (`MatN::of_new_block_scores`), so core `c` reads exactly the
        // rows core `c` wrote; `emit`'s core-local read law compares each op's slice at build.
        let at_pin = DevOff::ZERO;
        let (m, q, q_off) = row_block_query(blocks, b, mq, nqh, hd, qs)?;
        ops.push(assemble_matmul_off_with_epilogue(
            &format!("attn_ncsc_{bn}_o{t}"),
            m,
            MatN::of_new_block_scores(cols),
            MatK::of_head_dim(hd),
            MatY::unbatched(),
            bmm_form,
            &q,
            q_off,
            &Stk::<KernelTag>::kernel(hd as usize, slots as usize, new_kt),
            kt.block(kvh).dev(),
            &rb(sc, g_rows, slots),
            at_pin,
            &rb(cmask, g_rows, slots),
            wide.block(b).dev(),
            EpilogueOpFunc::StridedAdd,
            &[],
            false,
            sym_id_base,
            layout,
        ));
        let ncbmax_op = format!("attn_ncbmax_{bn}_o{t}");
        ops.push(assemble_reduce_off(
            &ncbmax_op,
            "max",
            grp_rows,
            cols,
            win_reduce_data(
                layout,
                &rb(sc, g_rows, slots),
                at_pin,
                grp_rows,
                cols,
                &format!("{ncbmax_op}: scores"),
            )
            .map_err(SuperDscError)?,
            win_reduce_accum(
                layout,
                &hm_rows(&bufs.run_m, q_rows),
                row_step(h0),
                grp_rows,
                &format!("{ncbmax_op}: accum"),
            )
            .map_err(SuperDscError)?,
            sym_id_base,
            layout,
        ));
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_ncesub_{bn}_o{t}"),
            "subtract",
            grp_rows,
            cols,
            &[
                In::sliced(&rb(sc, g_rows, slots), at_pin).ew(),
                In::col_at(&hm_rows(&bufs.run_m, q_rows), row_step(h0)).ew(),
            ],
            &rb(esub, g_rows, slots),
            at_pin,
            sym_id_base,
            layout,
        ));
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_ncexp_{bn}_o{t}"),
            "exp",
            grp_rows,
            cols,
            &[In::sliced(&rb(esub, g_rows, slots), at_pin).ew()],
            &rb(exp, g_rows, slots),
            at_pin,
            sym_id_base,
            layout,
        ));
        let ncbsum_op = format!("attn_ncbsum_{bn}_o{t}");
        ops.push(assemble_reduce_off(
            &ncbsum_op,
            "sum",
            grp_rows,
            cols,
            win_reduce_data(
                layout,
                &rb(exp, g_rows, slots),
                at_pin,
                grp_rows,
                cols,
                &format!("{ncbsum_op}: probabilities"),
            )
            .map_err(SuperDscError)?,
            win_reduce_accum(
                layout,
                &hm_rows(&bufs.run_l, q_rows),
                row_step(h0),
                grp_rows,
                &format!("{ncbsum_op}: accum"),
            )
            .map_err(SuperDscError)?,
            sym_id_base,
            layout,
        ));
        // The seed: no prior output to fold, so no epilogue. Above one slab, one op per slab of V.
        let n_slabs = RowBlockValue::slabs(blocks, hd);
        for s in 0..n_slabs {
            let v = RowBlockValue::of(blocks, b, s, mq, q_rows, hd);
            let v_off = if n_slabs == 1 {
                vv.view().at(Idx::<Head>::n(kvh)).dev()
            } else {
                vv.view().at(Idx::<Head>::n(kvh)).slab(s).dev()
            };
            let slab = if n_slabs == 1 {
                String::new()
            } else {
                format!("s{s}")
            };
            ops.push(assemble_matmul_off(
                &format!("attn_ncov_{bn}{slab}_o{t}"),
                v.m,
                v.n,
                MatK::of_kv_window(cols),
                MatY::unbatched(),
                bmm_form,
                &rb(exp, g_rows, slots),
                at_pin,
                &Stk::<KernelTag>::kernel(slots as usize, v.v_cols as usize, new_v),
                v_off,
                &rb(&bufs.run_o, q_rows, hd),
                v.out_off,
                sym_id_base,
                layout,
            ));
        }
    }
    no_chunk_op_is_time_tiled(&ops[first_op..], t)
}

/// The per-core LX a prompt chunk's two core-local probability scratches (`cesub`, `cexp`) may pin together:
/// a third of the 2 MiB scratchpad, leaving the rest to the landings of the ops reading and writing HBM
/// around them (the score part `esub` reads, the V part the PV reads).
const PROMPT_LX_PIN_BUDGET: u64 = 768 * 1024;

/// Pages per part of a PROMPT chunk's probabilities ([`assemble_attn_chunk`]'s LX loop): the fewest EQUAL
/// whole-page parts where, at `g_rows` rows per kv head, `esub` and the row sum each run in one trip, the PV
/// part contracts at most [`ChunkScratch::VALUE_PART_PAGES`] (the K dxp verifies), and one core's slices of
/// `cesub` and `cexp` fit [`PROMPT_LX_PIN_BUDGET`]. One page at worst. ONE place, so the caller that pins the
/// scratches and the body that cuts the parts read the same number.
///
/// [`ChunkScratch::VALUE_PART_PAGES`]: crate::sdsc_abstract::ChunkScratch::VALUE_PART_PAGES
fn prompt_lx_part_pages(chunk: crate::sdsc_abstract::ChunkScratch, g_rows: u32) -> u32 {
    let page_slots = crate::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32;
    (1..=chunk.pages())
        .filter(|n| chunk.pages().is_multiple_of(*n))
        .map(|n| chunk.pages() / n)
        .find(|&pp| {
            let cols = pp * page_slots;
            pp <= crate::sdsc_abstract::ChunkScratch::VALUE_PART_PAGES
                && crate::emit::reduce_fits_untiled(g_rows, cols)
                && crate::emit::pointwise_untiled_slice_bytes(g_rows, cols, 2)
                    .is_some_and(|b| 2 * b <= PROMPT_LX_PIN_BUDGET)
        })
        .unwrap_or(1)
}

/// Pages per part of a chunk's ROW REDUCTIONS (each kv head's max and denominator): the fewest EQUAL
/// whole-page parts whose `[g_rows, part]` reduction runs in one trip (`emit::reduce_fits_untiled`), one
/// page at worst. One part — the whole chunk — wherever that fits: every width at decode and at 128 prompt
/// rows. Read by the caller (which declares the partial buffers only when there is more than one part) and
/// by the body, so the two cannot disagree. See `no_chunk_op_is_time_tiled` for why a trip-sliced chunk
/// reduction is refused rather than emitted.
fn reduction_part_pages(chunk: crate::sdsc_abstract::ChunkScratch, g_rows: u32) -> u32 {
    let page_slots = crate::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32;
    (1..=chunk.pages())
        .filter(|n| chunk.pages().is_multiple_of(*n))
        .map(|n| chunk.pages() / n)
        .find(|&pp| crate::emit::reduce_fits_untiled(g_rows, pp * page_slots))
        .unwrap_or(1)
}

/// Pages per probability part of a chunk whose row blocks carry `b_rows` query rows each: the fewest
/// EQUAL whole-page parts whose `[b_rows, part]` pointwise op runs untiled under `division` (asked of the
/// builder's own tiler), one page at worst. ONE place, so the caller that sizes the core-local `cesub` pin
/// and the body that cuts the parts read the same number. `division` is the chunk's own
/// ([`ChunkBufs::sm_division_of`]).
fn softmax_part_pages(
    chunk: crate::sdsc_abstract::ChunkScratch,
    b_rows: u32,
    division: crate::work::EwDivision,
) -> u32 {
    let page_slots = crate::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32;
    (1..=chunk.pages())
        .filter(|n| chunk.pages().is_multiple_of(*n))
        .map(|n| chunk.pages() / n)
        .find(|&pp| {
            crate::emit::pointwise_untiled_slice_bytes_divided(b_rows, pp * page_slots, 2, division)
                .is_some()
        })
        .unwrap_or(1)
}

/// One core-local pin the chunk form mints beside its gathers: its synth role, its producer's
/// input count, and one core's slice of it in bytes.
pub type ChunkProbPin = (crate::place::SynthRole, u32, u64);

/// ⭐⭐ THE CHUNK FORM'S CORE-LOCAL PINS BEYOND ITS GATHERS — the probability-part scratches:
/// `cesub` always, and a prompt chunk's `cexp` beside it, each one row block's per-core slice
/// of one probability part under its own producer's division. Returns `(part pages, one row
/// per pin)`; the pins are `None` when a probability part is time-tiled, so no untiled
/// per-core slice exists to pin.
///
/// ⭐ ONE ENUMERATION FOR BOTH SIDES OF THE WINDOW BUDGET (#645): the chunk's `lx_form`
/// admission sums these bytes into its core-local term — beside the gather pins
/// `ChunkLx::pinned_bytes_per_core` counts — and the mint in `assemble_attn` declares exactly
/// these rows, so the witness and the pin allocator read ONE footprint. Before this, the
/// witness's term counted the gathers only while the emission also pinned the probability
/// parts, so a band of op-local class sizes was admitted by the witness and then PANICKED the
/// allocator where the contract wants the graceful HBM fall. A NEW core-local pin the chunk
/// form mints belongs HERE, or the gap reopens.
///
/// ⭐ AND #692's DIVISION RIDES THE SAME ENUMERATION: `sm_division` is the chunk's own
/// (`ChunkBufs::sm_division_of`), and it sizes a decode part's pages and slice exactly as it
/// divides the ops that produce them — the sticks-first batch law included — while a prompt
/// chunk's parts keep the law its row-divided PV demands. ONE value, so the pin the witness
/// budgets, the pin the mint declares, and the ops' own division can never disagree.
pub fn chunk_prob_pins(
    chunk: crate::sdsc_abstract::ChunkScratch,
    b_rows: u32,
    prompt: bool,
    sm_division: crate::work::EwDivision,
) -> (u32, Option<Vec<ChunkProbPin>>) {
    let page_slots = crate::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32;
    let part_pages = if prompt {
        prompt_lx_part_pages(chunk, b_rows)
    } else {
        softmax_part_pages(chunk, b_rows, sm_division)
    };
    let cols = part_pages * page_slots;
    let division = if prompt {
        crate::work::EwDivision::Law
    } else {
        sm_division
    };
    let roles: &[(crate::place::SynthRole, u32)] = if prompt {
        &[
            (crate::place::SynthRole::ChunkESub, 2),
            (crate::place::SynthRole::ChunkExp, 1),
        ]
    } else {
        &[(crate::place::SynthRole::ChunkESub, 2)]
    };
    let pins = roles
        .iter()
        .map(|&(r, inputs)| {
            Some((
                r,
                inputs,
                crate::emit::pointwise_untiled_slice_bytes_divided(b_rows, cols, inputs, division)?,
            ))
        })
        .collect::<Option<Vec<_>>>();
    (part_pages, pins)
}

/// ⭐⭐⭐⭐⭐ THE SPLIT-KV CHUNK — EVERY resident page of ONE request attended in ONE pass, folded onto the
/// new-token seed: the reference's decode structure (`spyre_inference`'s
/// `batched_decode_head_major_kernel`: gather the chunk's pages, ONE score and ONE value matmul over the
/// whole chunk, then reduce the row max and the denominators across the pages).
///
/// ⛔ WHAT IT REPLACES: the per-page fold, `SlotWindow::sweep(active_cap)` re-launched once per resident
/// page, each launch 4 windows × (8 per-kv-head score matmuls + 11 softmax ops + 8 per-kv-head value
/// matmuls) = 108 ops. Per resident page per layer that is ~80 µs of serialized op glue on granite-2b
/// (the decode slope this exists to remove). Here a chunk of up to [`ChunkScratch::PAGES_MAX`] pages costs:
///
/// * gather: one copy per index stick, for K and for V (the card-proven copy op, [`ChunkScratch::copies`]
///   and [`ChunkScratch::value_copies`]);
/// * per ROW BLOCK ([`ChunkRowBlocks`]: a kv head at one feature slab, a query head above it): score,
///   row max, sub, exp, row sum, and the value leg — one op, or one per slab above one slab;
/// * shared over every head: the fold block's own merge ops (`newm`, `corrsub`, `corre`, `mset`, `ocorr`,
///   `lcorr`, `ladd`).
///
/// The same count at every chunk width — the slope is in the gather's copies and the ops' own widths.
///
/// ⛔ EACH ROW BLOCK'S WIDE DATA IS A WHOLE `[rows, slots]` BLOCK ([`ChunkRowBlocks::corner`]), never a
/// row slice of a taller multi-stick buffer — dxp rebuilds a stick-blocked operand's group stride from the
/// op's own `N_`. The narrow running stats stay in the shared `[nqh, stick]` buffers, whose row slices are
/// flat at one stick and therefore exact.
///
/// [`ChunkRowBlocks`]: crate::sdsc_abstract::ChunkRowBlocks
/// [`ChunkRowBlocks::corner`]: crate::sdsc_abstract::ChunkRowBlocks::corner
///
/// The row max is the stick-major `[gqa, slots]` reduce, the same form as the row sum (see the
/// `attn_cbmax_g*` ops for why it must be per row and why not a page-axis reduce).
fn assemble_attn_chunk(
    ops: &mut Vec<EmittedOp>,
    a: ChunkAttn<'_>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<(), SuperDscError> {
    let ChunkAttn {
        t,
        nqh,
        hd,
        gqa,
        mq,
        chunk,
        history,
        index,
        kct,
        vc,
        qs,
        pmask,
        bmm_form,
        bufs,
        cb,
    } = a;
    use crate::addr::DevOff;
    use crate::superdsc_opspec::{EpilogueOpFunc, Scale};
    let first_op = ops.len();
    let stick = BlockCols::of_one_stick(Lanes::FP16);
    let width = BlockCols::of_chunk(chunk);
    let slots = width.get();
    // ⭐ ROWS, NOT HEADS. A head's rows are `mq` consecutive positions of the head-major token stream
    // (row `h·mq + r`): one at decode, the prompt chunk's width at prefill. So kv head `kvh`'s query rows
    // are the contiguous `[kvh·gqa·mq, (kvh+1)·gqa·mq)`, and every per-head block, base and extent below
    // is in those rows; at `mq == 1` each is the decode body's, byte for byte.
    let q_rows = nqh * mq;
    let all_rows = RowCount::of_head_major_rows(q_rows);
    let nkvh = chunk.nkvh();
    if nkvh * gqa != nqh {
        return Err(SuperDscError(format!(
            "attention chunk o{t}: {nkvh} kv heads × {gqa} query heads per group is not the {nqh} \
             query heads the bundle carries"
        )));
    }
    // ⭐ THE ROW BLOCKS every wide op below is cut into: one per KV HEAD at one feature slab (a GQA
    // group's `gqa·mq` rows are contiguous in `qs`), one per QUERY HEAD above it (only a head's own rows
    // are) — see `ChunkRowBlocks`. At one slab every op, name and offset is the per-kv-head form.
    // ⭐ OR, FOR A ONE-REQUEST DECODE WHOSE QUERY IS RE-LAID (`cb.qgroup`), one per KV HEAD of its whole
    // group — the caller decided, so the body only reads its decision back off the buffers it was handed.
    let blocks = match &cb.qgroup {
        Some(_) => match history {
            ChunkHistory::Requests(_) => {
                crate::sdsc_abstract::QGroupBlock::of_batch(chunk.nkvh(), gqa, hd, mq)
            }
            ChunkHistory::Shared => {
                crate::sdsc_abstract::QGroupBlock::of_decode(chunk.nkvh(), gqa, hd, mq)
            }
        }
        .and_then(|q| chunk.row_blocks_of_query_groups(q))
        .ok_or_else(|| {
            SuperDscError(format!(
                "attention chunk o{t}: a re-laid query was handed to a chunk whose rows are not a \
                 decode above one slab ({mq} row(s) per head at head dim {hd})"
            ))
        })?,
        None => chunk.row_blocks(gqa, mq),
    };
    // ⛔ A BATCH CHUNK's LAW MUST DESCRIBE THIS VERY CHUNK AT THIS VERY WIDTH: the host stages the mask
    // block and the page list from it, so a law for another chunk or another request count is a mask
    // whose rows and runs are some other body's — fluent, wrong, and silent.
    if let ChunkHistory::Requests(bc) = history
        && (bc.chunk() != chunk || bc.requests() != mq || bc.mask_rows() != blocks.rows())
    {
        return Err(SuperDscError(format!(
            "attention chunk o{t}: the batch chunk law describes a {}-page chunk over {} request(s) with \
             {}-row mask blocks, but this body attends a {}-page chunk at {mq} row(s) per head with \
             {}-row blocks",
            bc.chunk().pages(),
            bc.requests(),
            bc.mask_rows(),
            chunk.pages(),
            blocks.rows(),
        )));
    }
    let b_rows = blocks.rows();
    let blk_rows = RowCount::of_head_major_rows(b_rows);
    // Block `b`'s ops are named for its kv head at one slab, its query head above it.
    let blk_name = |b: u32| row_block_name(blocks, b);
    // A one-stick row offset in a `[rows, stick]` buffer (or, at one slab, in `qs`) is a single-stick
    // view step.
    let row_step = |h: u32| DevOff::from_view_step(h * stick.get());
    let corner = |c: Option<crate::sdsc_abstract::ChunkCorner>, what: &str| {
        c.map(DevOff::of_chunk_corner).ok_or_else(|| {
            SuperDscError(format!(
                "attention chunk o{t}: {what} lies outside the {}-page chunk scratch",
                chunk.pages()
            ))
        })
    };

    // ── (1) THE GATHER: every page's Kᵗ and V blocks, kv-head-major, into the two scratches. V's copies
    //    read V's own slab-major index run above one slab (`ChunkScratch::value_copies`). Where the gather
    //    stays in LX (`cb.lx`), each leg reads its part-major run (ONE shared run at one slab): a whole
    //    part's copy lands in its own core-local tensor, and a V tail copy in V's tail HBM scratch. ──
    type Copies<'c> = Vec<(&'c str, crate::sdsc_abstract::GatherCopy)>;
    let (k_copies, v_copies): (Copies<'_>, Copies<'_>) = match &cb.lx {
        None => (
            chunk.copies().map(|cp| (cb.gkt.as_str(), cp)).collect(),
            chunk
                .value_copies()
                .map(|cp| (cb.gv.as_str(), cp))
                .collect(),
        ),
        Some(lx) => (
            // Kᵗ stays in LX only with its witness (`ChunkLx::keys`, a chunk with no tail); otherwise
            // it is the whole-chunk HBM leg, read through its own head-major run.
            match &lx.keys {
                Some((keys, kt)) => keys
                    .copies()
                    .map(|(i, cp)| (kt[i as usize].as_str(), cp))
                    .collect(),
                None => chunk.copies().map(|cp| (cb.gkt.as_str(), cp)).collect(),
            },
            lx.form
                .copies()
                .map(|(d, cp)| (ChunkLxBufs::dest(&lx.v, &cb.gv, d), cp))
                .collect(),
        ),
    };
    for (name, src, copies) in [
        (format!("attn_cgkt_o{t}"), kct, k_copies),
        (format!("attn_cgv_o{t}"), vc, v_copies),
    ] {
        for (dst, cp) in copies {
            let mut op = crate::ir::bridge::tiled_op_sdsc_op::assemble_gather_copy(
                &format!("{name}_s{}", cp.stick()),
                src,
                dst,
                cp,
                index,
                sym_id_base,
                layout,
            );
            // ⭐ DECLARED ON THE COPY ITSELF: it groups as `GroupKind::Gathered`, whose launch takes no
            // KV page shift (the entries are absolute) and whose body the host stages the index for.
            op.kv_gathered = true;
            ops.push(op);
        }
    }

    // ── (1′) THE QUERY, RE-LAID GROUP-MAJOR (`QGroupBlock`) where the row blocks are query groups: one
    //    identity stick copy per (kv head, slab) — the KV cache write's own descriptor shape, `y` walking
    //    the group's heads `nslab` stick planes apart in the stream and one row apart in the block. ──
    let q_name: &str = match (&cb.qgroup, blocks.form()) {
        (Some(qb), crate::sdsc_abstract::ChunkRowForm::QueryGroup(q)) => {
            for kvh in 0..q.nkvh() {
                for s in 0..q.slabs() {
                    let missing = || {
                        SuperDscError(format!(
                            "attention chunk o{t}: no re-layout copy for kv head {kvh} slab {s}"
                        ))
                    };
                    let src = q.copy_src(kvh, s).ok_or_else(missing)?;
                    let dst = q.copy_dest(kvh, s).ok_or_else(missing)?;
                    ops.push(super::matmul::assemble_matmul_placed(
                        &format!("attn_cqg_g{kvh}s{s}_o{t}"),
                        MatM::of_query_rows(crate::sdsc_abstract::QueryRowCount::of_mq(mq)),
                        MatN::one_stick(Lanes::FP16),
                        MatK::one_stick(Lanes::FP16),
                        MatY::of_gqa_group(q.gqa(), src, dst),
                        SharedKernelBmmForm::of_query_group_copy(),
                        &rb(qs, mq, nqh * hd),
                        src,
                        // I[64,64] — the class identity's first diagonal block, for every slab: an
                        // identity copy needs exactly the shared 2-D kernel a `y` batch demands.
                        &Stk::<KernelTag>::kernel(
                            stick.get() as usize,
                            stick.get() as usize,
                            &qb.ident,
                        ),
                        DevOff::ZERO,
                        &rb(&qb.block, q.nkvh() * q.rows(), hd),
                        dst,
                        sym_id_base,
                        layout,
                    ));
                }
            }
            qb.block.as_str()
        }
        (None, crate::sdsc_abstract::ChunkRowForm::QueryGroup(_)) | (Some(_), _) => {
            return Err(SuperDscError(format!(
                "attention chunk o{t}: the row blocks and the re-laid query disagree about whether the \
                 query is re-laid"
            )));
        }
        (None, _) => qs,
    };

    // ── (2) SCORE per row block, then each row's max over its whole chunk ──
    let mask_h = hm(pmask);
    // ⭐ HOW THE PREFIX VALIDITY IS READ. A shared history is one broadcast row (`mb` collapsed); a batch
    // chunk's rows are separate requests, so the epilogue reads a FULL `[rows, slots]` block — the score
    // block's own operand cloned, per row, at offset 0 in every row block (`BatchChunk::mask_elem` is
    // the arrangement the host stages it in).
    let mask_dims: &[(&str, Scale)] = match history {
        ChunkHistory::Shared => &[("mb", Scale::RedNonStick)],
        ChunkHistory::Requests(_) => &[],
    };
    for b in 0..blocks.count() {
        let kvh = blocks.kv_head(b);
        let kc = corner(chunk.kernel_corner(kvh), "a kv head's kernel block")?;
        let rc = corner(blocks.corner(b), "a row block's score block")?;
        // `sc_b = Q_b · Kᵗ_kvh + mask` — the mask is the broadcast prefix-validity row over the chunk's
        // logical slots (head-independent, so `mb` collapses), added in the matmul's own epilogue; for a
        // batch chunk, the per-row block (`mask_dims` above). Q is the row block's own operand
        // (`row_block_query`) — the re-laid group block where the query was re-laid above.
        let (m, a, a_off) = row_block_query(blocks, b, mq, nqh, hd, q_name)?;
        let Some((keys, kt)) = cb.lx.as_ref().and_then(|lx| lx.keys.as_ref()) else {
            ops.push(assemble_matmul_off_with_epilogue(
                &format!("attn_csc_{}_o{t}", blk_name(b)),
                m,
                MatN::of_kv_window(width),
                MatK::of_head_dim(hd),
                MatY::unbatched(),
                bmm_form,
                &a,
                a_off,
                &Stk::<KernelTag>::kernel(hd as usize, slots as usize, &cb.gkt),
                kc,
                &rb(&cb.csc, b_rows, slots),
                rc,
                &mask_h,
                DevOff::ZERO,
                EpilogueOpFunc::StridedAdd,
                mask_dims,
                true,
                sym_id_base,
                layout,
            ));
            continue;
        };
        // ⭐ WHERE Kᵗ STAYS IN LX, ONE SCORE PER (row block, Kᵗ part). Each fills its part's columns of
        // the block's score row block and adds its part's columns of the mask, reading its (part, kv head)
        // core-local Kᵗ at the pin. Each part's slots are divided across the cores one stick each,
        // exactly as its copy delivered them (`emit`'s core-local read law proves it per op). Kᵗ stays
        // in LX only with its witness (`ChunkLx::keys`, a chunk with no Kᵗ tail), so every part is whole.
        for (p, part) in chunk.slot_parts(keys.part_pages()).enumerate() {
            let k_name = kt[keys.tensor(p as u32, kvh) as usize].as_str();
            let pc = corner(part.probs_corner(blocks, b), "a row block's score part")?;
            // The part's columns of the mask. A shared history is one broadcast row, so they start
            // `first_slot` in. A batch's per-row block is RowBlocked `[rows, slots]` like the score block it
            // clones, so they start `first_slot · rows` in: row block 0's own part corner.
            let mask_off = match history {
                ChunkHistory::Shared => DevOff::of_chunk_corner(part.row_corner()),
                ChunkHistory::Requests(_) => {
                    corner(part.probs_corner(blocks, 0), "the batch mask's score part")?
                }
            };
            ops.push(assemble_matmul_off_with_epilogue(
                &format!("attn_csc_{}_k{p}_o{t}", blk_name(b)),
                m,
                // Its COLUMNS divided across the cores, rows whole (`OutputSplit` says why): core `c`
                // holds only slot stick `c` of the part's Kᵗ.
                MatN::of_chunk_decode_scores(BlockCols::of_chunk_part(part)),
                MatK::of_head_dim(hd),
                MatY::unbatched(),
                bmm_form,
                &a,
                a_off,
                &Stk::<KernelTag>::kernel(hd as usize, part.slots() as usize, k_name),
                DevOff::ZERO,
                &rb(&cb.csc, b_rows, slots),
                pc,
                &mask_h,
                mask_off,
                EpilogueOpFunc::StridedAdd,
                mask_dims,
                true,
                sym_id_base,
                layout,
            ));
        }
    }
    // ⭐ THE STABILIZER IS EACH ROW's OWN MAX, ONE OP PER KV HEAD — the stick-major `[gqa, slots]` max,
    // the same form as the denominators' sum below (both card-verified per row by reading the chunk's
    // own buffers back after layer 0 and recomputing each op on the host).
    //
    // ⛔⛔⛔ IT MUST BE THE ROW's MAX, NOT THE HEAD GROUP's. A group max is a valid upper bound, and at
    // layer 0 it gave the per-page fold's attention output to the 4th digit (granite-3.1-2b fp8, A/B'd
    // against the fold binary) — but it scales a row's `o` and `l` by `exp(row_max - group_max)`, which is
    // `0.0117` for row 31 already at layer 0; once a later layer's gap passes fp16's range both underflow,
    // `o / l` is NaN, and the residual stream is `inf` by the last layer (measured). The softmax is
    // invariant to the stabilizer only in exact arithmetic.
    //
    // ⛔ AND NOT A PAGE-AXIS REDUCE: every per-lane page-axis form tried over these HBM blocks —
    // `maxnonstick`, and an elementwise slab fold — walked only part of each core's rows on card. Those used
    // row-major operands with a declared walk; this op's operands are stick-blocked (dxp reconstructs).
    //
    // ⛔⛔ AND IN ONE TRIP. Where a row block's whole-chunk reduction would be time-tiled (256 prompt rows
    // past 51 pages), it is cut into equal page parts that each run untiled, each part reduced into a
    // one-stick partial and the partials combined elementwise (`maximum` / `add`, the online merge's own
    // ops) — never in place, alternating between the partial buffers into the destination. A time-tiled
    // chunk reduction answered garbage on card (`no_chunk_op_is_time_tiled`).
    let red_parts: Vec<crate::sdsc_abstract::ChunkPart> = chunk
        .slot_parts(reduction_part_pages(chunk, b_rows))
        .collect();
    let reduce_in_parts = |ops: &mut Vec<EmittedOp>,
                           sym_id_base: &mut i64,
                           stem: &str,
                           func: &'static str,
                           combine: &'static str,
                           src: &str,
                           dst: &str|
     -> Result<(), SuperDscError> {
        let temps = [cb.red[0].as_str(), cb.red[1].as_str(), cb.red[2].as_str()];
        let mut acc = temps[0];
        for (p, part) in red_parts.iter().enumerate() {
            let into = if p == 0 { temps[0] } else { temps[1] };
            for b in 0..blocks.count() {
                let pc = corner(part.probs_corner(blocks, b), "a row block's reduction part")?;
                let part_op = format!("attn_c{stem}_{}_s{p}_o{t}", blk_name(b));
                ops.push(assemble_reduce_off(
                    &part_op,
                    func,
                    blk_rows,
                    BlockCols::of_chunk_part(*part),
                    win_reduce_data(
                        layout,
                        // The FULL `[q_rows, slots]` score buffer — `pc` is the part's corner in
                        // the whole buffer (ChunkPart::probs_corner's law), and the block's own
                        // `b_rows` rows are a slice of it, so the part-sized and block-sized
                        // annotations the raw form carried were never the window's bound.
                        &rb(src, q_rows, slots),
                        pc,
                        blk_rows,
                        BlockCols::of_chunk_part(*part),
                        &format!("{part_op}: probabilities"),
                    )
                    .map_err(SuperDscError)?,
                    win_reduce_accum(
                        layout,
                        &hm_rows(into, q_rows),
                        row_step(blocks.first_head(b) * mq),
                        blk_rows,
                        &format!("{part_op}: partial"),
                    )
                    .map_err(SuperDscError)?,
                    sym_id_base,
                    layout,
                ));
            }
            if p > 0 {
                let out = if p + 1 == red_parts.len() {
                    dst
                } else if acc == temps[2] {
                    temps[0]
                } else {
                    temps[2]
                };
                ops.push(assemble_pointwise_broadcast_off(
                    &format!("attn_c{stem}_k{p}_o{t}"),
                    combine,
                    all_rows,
                    stick,
                    &[
                        In::full(&hm_rows(acc, q_rows)).ew(),
                        In::full(&hm_rows(temps[1], q_rows)).ew(),
                    ],
                    &hm_rows(out, q_rows),
                    DevOff::ZERO,
                    sym_id_base,
                    layout,
                ));
                acc = out;
            }
        }
        Ok(())
    };
    if red_parts.len() == 1 {
        for b in 0..blocks.count() {
            let h0 = blocks.first_head(b) * mq;
            let rc = corner(blocks.corner(b), "a row block's score block")?;
            // The chunk's own division (`ChunkBufs::red_division_of`): the reduced axis across the
            // cores for a decode batch, rows first for every other chunk, byte for byte.
            ops.push(assemble_reduce_off_divided(
                ReduceDivided {
                    op_name: &format!("attn_cbmax_{}_o{t}", blk_name(b)),
                    op_func: "max",
                    rows: blk_rows,
                    cols: width,
                    data: win_reduce_data(
                        layout,
                        // The FULL `[q_rows, slots]` chunk score buffer — the block's `rc` is a
                        // corner in it, so the block-sized `b_rows` annotation the raw form
                        // carried was never the window's bound.
                        &rb(&cb.csc, q_rows, slots),
                        rc,
                        blk_rows,
                        width,
                        &format!("attn_cbmax_{}_o{t}: scores", blk_name(b)),
                    )
                    .map_err(SuperDscError)?,
                    accum: win_reduce_accum(
                        layout,
                        &hm_rows(&bufs.bmax, q_rows),
                        row_step(h0),
                        blk_rows,
                        &format!("attn_cbmax_{}_o{t}: accum", blk_name(b)),
                    )
                    .map_err(SuperDscError)?,
                    division: cb.red_division,
                },
                sym_id_base,
                layout,
            ));
        }
    } else {
        reduce_in_parts(
            ops,
            sym_id_base,
            "bmax",
            "max",
            "maximum",
            &cb.csc,
            &bufs.bmax,
        )?;
    }

    // ── (3) THE ONLINE MERGE ONTO THE SEED — the fold block's own four ops, in its order ──
    for (op_name, func, ins, out) in [
        ("newm", "maximum", [&bufs.run_m, &bufs.bmax], &bufs.newm),
        (
            "corrsub",
            "subtract",
            [&bufs.run_m, &bufs.newm],
            &bufs.corrsubt,
        ),
    ] {
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_c{op_name}_o{t}"),
            func,
            all_rows,
            stick,
            &[
                In::full(&hm_rows(ins[0], q_rows)).ew(),
                In::full(&hm_rows(ins[1], q_rows)).ew(),
            ],
            &hm_rows(out, q_rows),
            DevOff::ZERO,
            sym_id_base,
            layout,
        ));
    }
    for (op_name, func, inp, out) in [
        ("corre", "exp", &bufs.corrsubt, &bufs.corr),
        ("mset", "identity", &bufs.newm, &bufs.run_m),
    ] {
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_c{op_name}_o{t}"),
            func,
            all_rows,
            stick,
            &[In::full(&hm_rows(inp, q_rows)).ew()],
            &hm_rows(out, q_rows),
            DevOff::ZERO,
            sym_id_base,
            layout,
        ));
    }

    // A prompt chunk keeps its probabilities in LX (the caller pins `cexp` by the same condition); decode —
    // one request, or a batch chunk whose `mq` rows are separate requests — keeps the form below.
    if mq > 1 && matches!(history, ChunkHistory::Shared) {
        prompt_probabilities_in_lx(
            ops,
            PromptProbs {
                t,
                hd,
                mq,
                blocks,
                q_rows,
                chunk,
                bmm_form,
                bufs,
                cb,
            },
            sym_id_base,
            layout,
        )?;
        return no_chunk_op_is_time_tiled(&ops[first_op..], t);
    }
    // ── (4) PROBABILITIES and each row's DENOMINATOR, per kv head ──
    //
    // ⭐ THE PROBABILITIES IN SLOT PARTS — the fewest EQUAL whole-page parts whose `[gqa·mq, part]`
    // sub-block runs untiled, asked of the builder's own tiler (`pointwise_fits_untiled`). A time-tiled
    // op with a broadcast operand — `esub`'s row max — is refused at build, and at 1024 rows a 64-page
    // chunk is time-tiled whole (`attn_cesub_g0`, granite-2b at 256 prompt rows). Equal parts keep every
    // part's base a whole multiple of its own block, the stick-major condition. ONE part, the whole
    // chunk, wherever that already fits — every width at decode and at 96 prompt rows, byte for byte. A
    // geometry where even one page is tiled keeps one-page parts and meets the builder's refusal.
    //
    // ⭐ AND `esub`'s OUTPUT NEVER LEAVES LX: `cb.cesub` is a CORE-LOCAL scratch (declared by the caller,
    // `BundleLayout::synth_lx_core_local`) holding one core's slice of ONE part of ONE kv head, reused
    // part after part — each `exp` reads it straight after its `esub` wrote it, under the same division
    // (both are the tiler's own `[gqa·mq, part]` pointwise op; `emit`'s core-local read law proves it at
    // build). That is the reference's form for every softmax intermediate, and it removes the `cesub`
    // write and read from HBM: two of the chunk's eight passes over its score block.
    //
    // ⭐ BOTH OPS UNDER THE CHUNK's OWN DIVISION (`ChunkBufs::sm_division_of`): sticks first for a decode
    // batch, the law for every other chunk. The caller sized the `cesub` pin and these parts under the same
    // value, and the read law holds `exp` to `esub`'s slices at build.
    let sm_pages = softmax_part_pages(chunk, b_rows, cb.sm_division);
    let sm_parts: Vec<crate::sdsc_abstract::ChunkPart> = chunk.slot_parts(sm_pages).collect();
    for b in 0..blocks.count() {
        let (h0, bn) = (blocks.first_head(b) * mq, blk_name(b));
        let rc = corner(blocks.corner(b), "a row block's score block")?;
        for (p, part) in sm_parts.iter().enumerate() {
            let pc = corner(part.probs_corner(blocks, b), "a row block's score part")?;
            let (pw, ps) = (BlockCols::of_chunk_part(*part), part.slots());
            let suffix = if sm_parts.len() == 1 {
                String::new()
            } else {
                format!("_s{p}")
            };
            // `esub` and `exp` write DIFFERENT buffers — an exp must never read and write one buffer.
            // `cesub` is core-local LX, so both address it at its pin (no corner). The score and
            // probability blocks are read/written at CHUNK-FRAME corners (`pc`), so their handles
            // declare the FULL `[q_rows, slots]` buffers — the block-shaped `b_rows` annotation the
            // raw form carried was never the window's bound.
            ops.push(assemble_pointwise_broadcast_divided(
                PointwiseDivided {
                    op_name: &format!("attn_cesub_{bn}{suffix}_o{t}"),
                    op_func: "subtract",
                    rows: blk_rows,
                    cols: pw,
                    inputs: &[
                        In::sliced(&rb(&cb.csc, q_rows, slots), pc).ew(),
                        In::col_at(&hm_rows(&bufs.run_m, q_rows), row_step(h0)).ew(),
                    ],
                    o: &rb(&cb.cesub, b_rows, ps),
                    out_offset: DevOff::ZERO.into(),
                    division: cb.sm_division,
                },
                sym_id_base,
                layout,
            ));
            ops.push(assemble_pointwise_broadcast_divided(
                PointwiseDivided {
                    op_name: &format!("attn_cexp_{bn}{suffix}_o{t}"),
                    op_func: "exp",
                    rows: blk_rows,
                    cols: pw,
                    inputs: &[In::sliced(&rb(&cb.cesub, b_rows, ps), DevOff::ZERO).ew()],
                    o: &rb(&cb.cexp, q_rows, slots),
                    out_offset: pc.into(),
                    division: cb.sm_division,
                },
                sym_id_base,
                layout,
            ));
        }
        // ⭐ EACH ROW's SUM OVER THE WHOLE CHUNK, straight into `bsum` — the stick-major `sum` over
        // `rows > 1 × cols > one stick`, the cell `Sum` combines correctly (prefill's RMS sums run it).
        // In parts below instead, after every block's probabilities, when one trip cannot hold it.
        if red_parts.len() == 1 {
            ops.push(assemble_reduce_off_divided(
                ReduceDivided {
                    op_name: &format!("attn_cbsum_{bn}_o{t}"),
                    op_func: "sum",
                    rows: blk_rows,
                    cols: width,
                    data: win_reduce_data(
                        layout,
                        // The FULL `[q_rows, slots]` probability buffer (the bsum twin above).
                        &rb(&cb.cexp, q_rows, slots),
                        rc,
                        blk_rows,
                        width,
                        &format!("attn_cbsum_{bn}_o{t}: probabilities"),
                    )
                    .map_err(SuperDscError)?,
                    accum: win_reduce_accum(
                        layout,
                        &hm_rows(&bufs.bsum, q_rows),
                        row_step(h0),
                        blk_rows,
                        &format!("attn_cbsum_{bn}_o{t}: accum"),
                    )
                    .map_err(SuperDscError)?,
                    division: cb.red_division,
                },
                sym_id_base,
                layout,
            ));
        }
    }
    if red_parts.len() > 1 {
        reduce_in_parts(ops, sym_id_base, "bsum", "sum", "add", &cb.cexp, &bufs.bsum)?;
    }

    // ── (5) RESCALE THE SEED'S OUTPUT, THEN THE VALUE LEG FOLDS ONTO IT ──
    // `ocorr` reads the PRE-update `run_o`, so it precedes every value matmul (the fold block's order).
    ops.push(assemble_pointwise_broadcast_off(
        &format!("attn_cocorr_o{t}"),
        "multiply",
        all_rows,
        BlockCols::of_head_dim(hd),
        &[
            In::full(&fl(&bufs.run_o, q_rows, hd)).ew(),
            In::col(&hm_rows(&bufs.corr, q_rows)).ew(),
        ],
        &fl(&bufs.otmp, q_rows, hd),
        DevOff::ZERO,
        sym_id_base,
        layout,
    ));
    let (otmp_h, run_o_h) = (fl(&bufs.otmp, q_rows, hd), fl(&bufs.run_o, q_rows, hd));
    // ⭐ THE VALUE LEG, IN K-PARTS (`ChunkScratch::value_parts`): ONE part at decode, 16 pages a part
    // for a prompt chunk, whose whole-chunk PV dxp refuses to verify. Each part folds onto the last
    // through the same `stridedadd` epilogue, PING-PONGING between `run_o` and `otmp` so no matmul reads
    // the buffer it writes: part `p` writes `run_o` when `p` is even. An even part count leaves the sum
    // in `otmp`, and one copy brings it home below.
    // ⭐ A DECODE's VALUE LEG — one request row per head, or a batch's `B` — divides its contraction
    // across the cores (`ContractionSplit` below), so no core's V slice nears LX and it stays ONE part;
    // only a prompt chunk's rows-divided PV is cut.
    let decode_values = mq == 1 || matches!(history, ChunkHistory::Requests(_));
    // ⭐ AN LX-RESIDENT V IS CUT INTO ITS OWN PARTS (`ChunkLx::part_pages`), one copy per (part, kv
    // head, slab): each whole part's PV reads that part's core-local tensor for its slab, and the tail
    // reads the HBM scratch. The parts fold through the same ping-pong.
    let parts: Vec<crate::sdsc_abstract::ChunkPart> = match &cb.lx {
        None => chunk.value_parts(!decode_values).collect(),
        Some(lx) => chunk.slot_parts(lx.form.part_pages()).collect(),
    };
    // ⛔ The LX form is minted only for the one-request decode, whose PV divides its contraction across
    // the cores. That division is the one each part's tensor is laid out for.
    if cb.lx.is_some() && !decode_values {
        return Err(SuperDscError(format!(
            "attention chunk o{t}: an LX-resident value scratch needs the decode value leg, whose \
             contraction is divided across the cores"
        )));
    }
    // ⭐ ABOVE ONE SLAB THE VALUE LEG IS ONE OP PER (QUERY HEAD, SLAB) (`RowBlockValue`): each op contracts
    // its head's probabilities against slab `s` of the slab-major V scratch and lands in slab `s`'s plane of
    // `run_o` at the head's rows. At one slab the op is the per-kv-head one, unchanged.
    let out_slabs = RowBlockValue::slabs(blocks, hd);
    for b in 0..blocks.count() {
        let (kvh, bn) = (blocks.kv_head(b), blk_name(b));
        for s in 0..out_slabs {
            let v = RowBlockValue::of(blocks, b, s, mq, q_rows, hd);
            let o_off = v.out_off;
            for (p, part) in parts.iter().enumerate() {
                // An LX-resident part is its (part, kv head, slab) core-local tensor, read at its pin (no
                // corner). The tail is kv head `kvh`'s slab `s` of the tail's HBM scratch.
                let (v_name, kc) = match &cb.lx {
                    None => (
                        cb.gv.as_str(),
                        corner(part.value_corner(kvh, s), "a kv head's value block")?,
                    ),
                    Some(lx) if (p as u32) < lx.form.lx_parts() => (
                        lx.v[lx.form.lx_tensor(p as u32, kvh, s) as usize].as_str(),
                        DevOff::ZERO,
                    ),
                    Some(lx) => (
                        cb.gv.as_str(),
                        corner(lx.form.tail_corner(kvh, s), "a kv head's tail value block")?,
                    ),
                };
                let rc = corner(
                    part.probs_corner(blocks, b),
                    "a row block's probability block",
                )?;
                let (out, addend) = if p % 2 == 0 {
                    (&bufs.run_o, &otmp_h)
                } else {
                    (&bufs.otmp, &run_o_h)
                };
                let slab = if out_slabs == 1 {
                    String::new()
                } else {
                    format!("s{s}")
                };
                let name = if parts.len() == 1 {
                    format!("attn_cov_{bn}{slab}_o{t}")
                } else {
                    format!("attn_cov_{bn}{slab}_k{p}_o{t}")
                };
                let RowBlockValue { m, n, v_cols, .. } = v;
                // ⭐ AT DECODE THE CONTRACTION IS DIVIDED ACROSS THE CORES (`ContractionSplit`): the output
                // is one stick of `gqa` rows (one row per head above one slab), so the planner would give
                // it only its rows and every core would read this kv head's WHOLE V. A prompt chunk has
                // `gqa·mq` rows to divide and keeps the planned split. A BATCH chunk's `gqa·B` rows are
                // still a few dozen, against a contraction `B` requests deep — the decode form.
                let k_part = BlockCols::of_chunk_part(*part);
                let k = if decode_values {
                    MatK::of_chunk_decode_values(k_part)
                } else {
                    MatK::of_kv_window(k_part)
                };
                // `run_o_b = P_b · V_kvh + otmp_b` — the `stridedadd` epilogue is the online-softmax update.
                ops.push(assemble_matmul_off_maybe_epilogue(
                    &name,
                    m,
                    n,
                    k,
                    MatY::unbatched(),
                    bmm_form,
                    &rb(&cb.cexp, b_rows, slots),
                    rc,
                    &Stk::<KernelTag>::kernel(part.slots() as usize, v_cols as usize, v_name),
                    kc,
                    &rb(out, q_rows, hd),
                    o_off,
                    Some((addend, o_off)),
                    EpilogueOpFunc::StridedAdd,
                    &[],
                    false,
                    sym_id_base,
                    layout,
                ));
            }
        }
    }
    if parts.len().is_multiple_of(2) {
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_covsum_o{t}"),
            "identity",
            all_rows,
            BlockCols::of_head_dim(hd),
            &[In::full(&otmp_h).ew()],
            &run_o_h,
            DevOff::ZERO,
            sym_id_base,
            layout,
        ));
    }
    // run_l = run_l*corr + bsum — the fold block's own pair.
    ops.push(assemble_pointwise_broadcast_off(
        &format!("attn_clcorr_o{t}"),
        "multiply",
        all_rows,
        stick,
        &[
            In::full(&hm_rows(&bufs.run_l, q_rows)).ew(),
            In::full(&hm_rows(&bufs.corr, q_rows)).ew(),
        ],
        &hm_rows(&bufs.ltmp, q_rows),
        DevOff::ZERO,
        sym_id_base,
        layout,
    ));
    ops.push(assemble_pointwise_broadcast_off(
        &format!("attn_cladd_o{t}"),
        "add",
        all_rows,
        stick,
        &[
            In::full(&hm_rows(&bufs.ltmp, q_rows)).ew(),
            In::full(&hm_rows(&bufs.bsum, q_rows)).ew(),
        ],
        &hm_rows(&bufs.run_l, q_rows),
        DevOff::ZERO,
        sym_id_base,
        layout,
    ));
    no_chunk_op_is_time_tiled(&ops[first_op..], t)
}

/// A prompt chunk's probability phase's operands — see [`prompt_probabilities_in_lx`].
struct PromptProbs<'a> {
    t: u32,
    hd: u32,
    mq: u32,
    /// The chunk's row blocks: a kv head at one slab, a query head above it.
    blocks: crate::sdsc_abstract::ChunkRowBlocks,
    q_rows: u32,
    chunk: crate::sdsc_abstract::ChunkScratch,
    bmm_form: crate::ir::bridge::tiled_op_sdsc_op::matmul::SharedKernelBmmForm,
    bufs: &'a BlockBufs,
    cb: &'a ChunkBufs,
}

/// ⭐⭐⭐ A PROMPT CHUNK'S PROBABILITIES NEVER LEAVE LX — THREE HBM PASSES OVER THE SCORE BLOCK, NOT SIX.
///
/// After the score (written to HBM), its row max (read once) and the online merge, each row block's chunk
/// (a kv head's at one slab, a query head's above it) is walked in [`prompt_lx_part_pages`] parts, and in
/// each part every op divides the work the same way — each core its own `rows / 32` rows and all of the
/// part's columns (`{mb:32}`, the PV's own split too):
///   * `esub` reads the score part from HBM into a core-local `cesub`;
///   * `exp` reads `cesub` into a core-local `cexp`;
///   * the row sum reduces `cexp` into the part's one-stick partial;
///   * the PV contracts `cexp` with the part's V block — one op per feature slab above one slab
///     ([`RowBlockValue`]) — and folds onto the output through the same `stridedadd` ping-pong the value
///     parts always used.
///
/// The probabilities are never written to or read back from HBM. That is the reference's own form, and the
/// emitter's core-local read law proves at build that each core reads exactly the slice it wrote. The
/// partials then add into the denominator. MEASURED before this (granite-2b fp16, 256 rows, ~10k tokens): the
/// chunk's attention group was 59% of prefill, HBM-bound at ~135 GB/s over six passes.
fn prompt_probabilities_in_lx(
    ops: &mut Vec<EmittedOp>,
    a: PromptProbs<'_>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<(), SuperDscError> {
    let PromptProbs {
        t,
        hd,
        mq,
        blocks,
        q_rows,
        chunk,
        bmm_form,
        bufs,
        cb,
    } = a;
    use crate::addr::DevOff;
    use crate::superdsc_opspec::EpilogueOpFunc;
    // A row block's rows: a kv head's `gqa·mq` at one slab, a query head's `mq` above it.
    let g_rows = blocks.rows();
    let stick = BlockCols::of_one_stick(Lanes::FP16);
    let all_rows = RowCount::of_head_major_rows(q_rows);
    let grp_rows = RowCount::of_head_major_rows(g_rows);
    let row_step = |h: u32| DevOff::from_view_step(h * stick.get());
    let corner = |c: Option<crate::sdsc_abstract::ChunkCorner>, what: &str| {
        c.map(DevOff::of_chunk_corner).ok_or_else(|| {
            SuperDscError(format!(
                "attention chunk o{t}: {what} lies outside the {}-page chunk scratch",
                chunk.pages()
            ))
        })
    };
    // `ocorr` reads the PRE-update `run_o`, so it precedes every value matmul (the fold block's order).
    ops.push(assemble_pointwise_broadcast_off(
        &format!("attn_cocorr_o{t}"),
        "multiply",
        all_rows,
        BlockCols::of_head_dim(hd),
        &[
            In::full(&fl(&bufs.run_o, q_rows, hd)).ew(),
            In::col(&hm_rows(&bufs.corr, q_rows)).ew(),
        ],
        &fl(&bufs.otmp, q_rows, hd),
        DevOff::ZERO,
        sym_id_base,
        layout,
    ));
    let (otmp_h, run_o_h) = (fl(&bufs.otmp, q_rows, hd), fl(&bufs.run_o, q_rows, hd));
    let parts: Vec<crate::sdsc_abstract::ChunkPart> = chunk
        .slot_parts(prompt_lx_part_pages(chunk, g_rows))
        .collect();
    let n = parts.len();
    let n_slabs = RowBlockValue::slabs(blocks, hd);
    for b in 0..blocks.count() {
        // Block `b`'s rows are the `first_head(b)·mq`-th on of the head-major stream.
        let (kvh, h0, bn) = (
            blocks.kv_head(b),
            blocks.first_head(b) * mq,
            row_block_name(blocks, b),
        );
        for (p, part) in parts.iter().enumerate() {
            let pc = corner(part.probs_corner(blocks, b), "a row block's score part")?;
            let (pw, ps) = (BlockCols::of_chunk_part(*part), part.slots());
            ops.push(assemble_pointwise_broadcast_off(
                &format!("attn_cesub_{bn}_s{p}_o{t}"),
                "subtract",
                grp_rows,
                pw,
                &[
                    // The score block at its CHUNK-FRAME part corner — the FULL `[q_rows, slots]`
                    // buffer, not the block-shaped `g_rows` annotation. `cesub` below is core-local
                    // LX, addressed at its pin (offset 0), so ITS handle stays block-shaped.
                    In::sliced(&rb(&cb.csc, q_rows, chunk.slots()), pc).ew(),
                    In::col_at(&hm_rows(&bufs.run_m, q_rows), row_step(h0)).ew(),
                ],
                &rb(&cb.cesub, g_rows, ps),
                DevOff::ZERO,
                sym_id_base,
                layout,
            ));
            ops.push(assemble_pointwise_broadcast_off(
                &format!("attn_cexp_{bn}_s{p}_o{t}"),
                "exp",
                grp_rows,
                pw,
                &[In::sliced(&rb(&cb.cesub, g_rows, ps), DevOff::ZERO).ew()],
                &rb(&cb.cexp, g_rows, ps),
                DevOff::ZERO,
                sym_id_base,
                layout,
            ));
            // One part: the whole chunk's sum, straight into the denominator.
            let sum_into = if n == 1 { &bufs.bsum } else { &cb.red[p] };
            let cbsum_op = format!("attn_cbsum_{bn}_s{p}_o{t}");
            ops.push(assemble_reduce_off(
                &cbsum_op,
                "sum",
                grp_rows,
                pw,
                win_reduce_data(
                    layout,
                    &rb(&cb.cexp, g_rows, ps),
                    DevOff::ZERO,
                    grp_rows,
                    pw,
                    &format!("{cbsum_op}: probabilities"),
                )
                .map_err(SuperDscError)?,
                win_reduce_accum(
                    layout,
                    &hm_rows(sum_into, q_rows),
                    row_step(h0),
                    grp_rows,
                    &format!("{cbsum_op}: denominator"),
                )
                .map_err(SuperDscError)?,
                sym_id_base,
                layout,
            ));
            let (out, addend) = if p % 2 == 0 {
                (&bufs.run_o, &otmp_h)
            } else {
                (&bufs.otmp, &run_o_h)
            };
            // `run_o_b = P_b · V_kvh + otmp_b` — the `stridedadd` epilogue is the online-softmax update. One
            // op per slab above one slab, each reading the same core-local `cexp`.
            for s in 0..n_slabs {
                let v = RowBlockValue::of(blocks, b, s, mq, q_rows, hd);
                let kc = corner(part.value_corner(kvh, s), "a kv head's value part")?;
                let slab = if n_slabs == 1 {
                    String::new()
                } else {
                    format!("s{s}")
                };
                ops.push(assemble_matmul_off_maybe_epilogue(
                    &format!("attn_cov_{bn}{slab}_k{p}_o{t}"),
                    v.m,
                    v.n,
                    MatK::of_kv_window(pw),
                    MatY::unbatched(),
                    bmm_form,
                    &rb(&cb.cexp, g_rows, ps),
                    DevOff::ZERO,
                    &Stk::<KernelTag>::kernel(ps as usize, v.v_cols as usize, &cb.gv),
                    kc,
                    &rb(out, q_rows, hd),
                    v.out_off,
                    Some((addend, v.out_off)),
                    EpilogueOpFunc::StridedAdd,
                    &[],
                    false,
                    sym_id_base,
                    layout,
                ));
            }
        }
    }
    if n.is_multiple_of(2) {
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_covsum_o{t}"),
            "identity",
            all_rows,
            BlockCols::of_head_dim(hd),
            &[In::full(&otmp_h).ew()],
            &run_o_h,
            DevOff::ZERO,
            sym_id_base,
            layout,
        ));
    }
    // The partials add into the denominator, never in place: alternating between the two accumulators
    // past the part buffers, the last add landing in `bsum`.
    let mut acc = cb.red[0].as_str();
    for p in 1..n {
        let out = if p + 1 == n {
            bufs.bsum.as_str()
        } else if p % 2 == 1 {
            cb.red[n].as_str()
        } else {
            cb.red[n + 1].as_str()
        };
        ops.push(assemble_pointwise_broadcast_off(
            &format!("attn_cbsum_k{p}_o{t}"),
            "add",
            all_rows,
            stick,
            &[
                In::full(&hm_rows(acc, q_rows)).ew(),
                In::full(&hm_rows(&cb.red[p], q_rows)).ew(),
            ],
            &hm_rows(out, q_rows),
            DevOff::ZERO,
            sym_id_base,
            layout,
        ));
        acc = out;
    }
    // run_l = run_l*corr + bsum — the fold block's own pair.
    ops.push(assemble_pointwise_broadcast_off(
        &format!("attn_clcorr_o{t}"),
        "multiply",
        all_rows,
        stick,
        &[
            In::full(&hm_rows(&bufs.run_l, q_rows)).ew(),
            In::full(&hm_rows(&bufs.corr, q_rows)).ew(),
        ],
        &hm_rows(&bufs.ltmp, q_rows),
        DevOff::ZERO,
        sym_id_base,
        layout,
    ));
    ops.push(assemble_pointwise_broadcast_off(
        &format!("attn_cladd_o{t}"),
        "add",
        all_rows,
        stick,
        &[
            In::full(&hm_rows(&bufs.ltmp, q_rows)).ew(),
            In::full(&hm_rows(&bufs.bsum, q_rows)).ew(),
        ],
        &hm_rows(&bufs.run_l, q_rows),
        DevOff::ZERO,
        sym_id_base,
        layout,
    ));
    Ok(())
}

/// ⛔⛔⛔ NO SPLIT-KV CHUNK OP IS TIME-TILED — a bake refusal naming each op that is, never a card run.
///
/// MEASURED, granite-3.1-2b fp16 at 256 prompt rows: a ~16k-token prompt — whose last chunks are 52-64
/// pages, the widths where the tiler first cuts a kv head's `[1024, J·256]` row max and denominator into
/// two time trips — answered `" signature signature signature…"` on both runs, where the same prompt at
/// 128 rows (nothing tiled) and every ~10k prompt at 256 (chunks ≤ 40 pages, nothing tiled) are coherent.
/// The trips of a reduction over its own reduced axis do not combine into the whole-chunk value on card,
/// so a chunk op must run in one trip; one that cannot fit is cut into parts instead (the probabilities'
/// slot parts, the value leg's K-parts).
fn no_chunk_op_is_time_tiled(ops: &[EmittedOp], t: u32) -> Result<(), SuperDscError> {
    let tiled: Vec<String> = ops
        .iter()
        .filter(|e| e.time > 1)
        .map(|e| format!("{} ({} trips)", e.op_name, e.time))
        .collect();
    if tiled.is_empty() {
        Ok(())
    } else {
        Err(SuperDscError(format!(
            "attention o{t}: split-KV chunk op(s) {} are TIME-TILED — a chunk reduction's trips do not \
             combine on card (granite-2b at 256 prompt rows answered garbage past 52 pages). Cut the op \
             into parts that each run in one trip.",
            tiled.join(", ")
        )))
    }
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
    vc: &str,
    pmask: &str,
    cmask: &str,
    // ⭐⭐⭐⭐⭐ THE KV BLOCK INDEX TENSOR'S NAME — `Some` makes the PREFIX score leg's KV read a
    // HARDWARE GATHER; `None` emits exactly what this bundle emitted before the gather existed.
    //
    // ⛔ THE `Option` IS THE COUPLING, NOT A SWITCH. A gathered read takes its base from this tensor's
    // entries, so an emission that gathers without a host staging them addresses block 0 for every
    // row — every request reading row 0's keys, fluent and wrong. Making the NAME the condition means
    // the emitter cannot gather unless a caller has a tensor to name, which is the same caller that
    // must fill it. There is no arrangement in which one is present and the other is not.
    kv_block_index: Option<&str>,
    // ⭐⭐⭐⭐⭐ THE SPLIT-KV CHUNK — `Some` attends every resident page in ONE pass through a kv-head-major
    // gather ([`assemble_attn_chunk`]) instead of the per-page fold; the index tensor above is then the
    // CHUNK's table. `None` emits exactly the per-page fold this bundle emitted before.
    chunk: Option<crate::sdsc_abstract::ChunkScratch>,
    out_id: crate::place::PlaceId,
    // TRUE when these rows are separate requests. Only the prefix mask cares: a prompt's rows share
    // one resident history, requests each have their own.
    rows_are_requests: bool,
    // ⭐⭐⭐ WHETHER THE FINALIZE HANDS `out` TO o_proj IN THE SLAB-MAJOR ORDER — the caller-side
    // statement of the head-major restructure, minted where the bundle's facts are (the spyre door's
    // `BundleAttnParams::headmajor_handoff`; a triton producer's caller states its own). TRUE only
    // for the single-row (mq == 1) body of a model whose head dim spans more than one feature slab
    // and whose o matmul is fp8 W8A8 — the conditions the permuted o_proj copy is staged for. The
    // matmul half of the contract (the B-operand swap to that copy) keys on the SAME fact at the
    // matmul door, so neither half can be emitted without the other.
    headmajor_handoff: bool,
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

    // Per KV-head (dedup, K-side — see main's `lower_attn_node` attn_krep/cachewr_k comment): transpose this
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
                    // ⭐ THE INPUT HANDLE, AT THE BUFFER'S OWN TRUTH: new_k_scaled is allocated
                    // `[mq_pad, nkvh·hd]` (the padded rows — the scale multiply and RoPE's producer both
                    // state it; the pad rows past `mq` hold the next head's masked lanes by the zero-fill
                    // law, still INSIDE this tensor). The mint bounds every (kvh, slab, sub-block) tile's
                    // flat run against THAT, so a corner arithmetic that walks past the last head's last
                    // slab is a build failure here rather than a clean bake reading the next tensor.
                    &rb(new_k_scaled, mq_pad_t.rows().row_axis_extent(), nkvh * hd),
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
                    // ⭐ THE OUTPUT HANDLE, likewise: `new_kt` is `nkvh` SEPARATE `[hd, mq_pad]` blocks
                    // laid end to end (the synth above reserves exactly `[nkvh, hd, mq_pad]`), so the
                    // handle states the whole flat run. The LAST sub-block's tile ends exactly at the
                    // scratch's last element — fits with equality, which is what the mint checks.
                    &rb(&new_kt, 1, nkvh * hd * mq_pad_t.slots().slot_axis_extent()),
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
    //
    // ── MEASURED ON THE CARD, 2026-09-17, granite-3.1-2b fp8, so the next attempt argues from numbers ──
    //
    // ⭐ THE COST MODEL. Launches and compute per DECODE STEP, `SCRATCHY_SDSC_SUBMIT_TIME`, this bundle:
    //     bs   launches  barriered  compute   step    per-token
    //      1        122        122   23.4 ms  24.1      24.1 ms
    //      2        362        322   30.5 ms  31.3      15.7
    //      4        602        482   40.3 ms  41.0      10.3
    //      8       1082        802   51.0 ms  52.2       6.5
    //     16       2040       1600   ~54  ms  55.0       3.44
    //   Launches per layer are EXACTLY `3 + 3*requests` in the batch bundle (fits all four rungs), and
    //   `pages` adds another `requests` per layer: 1082 → 1402 → 1722 at one, two and three resident
    //   pages, i.e. `+320 = 8 requests × 40 layers` per page. So the `pages × requests` factor is real
    //   and it is exactly what this gate turns off.
    //
    // ⛔ BUT THE FOLD IS ONE OF **THREE** PER-REQUEST LAUNCH FACTORS, NOT THE FACTOR. Of the 3 launches
    //   per request per layer, one is this fold's extra rep and the other two are the KV CACHE WRITE and
    //   the Kᵗ RE-TRANSPOSE (`lower_ktir_to_superdsc.rs`'s `per_request` loops, which tag `kv_request`
    //   and so break the launch group at `Trip::fusable_with`). Collapsing the fold alone therefore
    //   removes 1/3 of the per-request LAUNCHES; the other two need the same per-request page base and
    //   are not touched by this axis.
    //
    // ⛔ AND A LAUNCH DOES **NOT** COST ~93 µs AT THE WIDTHS THAT MATTER — the sentence at the top of
    //   this note is a bs≤4 number generalised. bs=8 → bs=16 adds 958 launches for +2.8 ms, i.e. ~2.9 µs
    //   of marginal launch cost, because the host submits ahead (submit is 4-5 µs/launch and 5-11% of
    //   compute) and the device pipelines. The honest marginal figures are ~28 µs per fold pass at
    //   bs=8 — and most of that is the pass's WORK, since a pass computes `nqh*mq` rows to keep `nqh`.
    //
    // ⭐ SO WHAT THE COLLAPSE IS WORTH, from the per-page measurement (the fold is ~9.15 ms of a bs=8
    //   step per resident page, launches and redundant rows together): 51.2 → ~43 ms at ONE page
    //   (**1.19x**) and 69.5 → ~45 ms at THREE (**1.53x**), growing with pages and with requests. It is
    //   a LONG-CONTEXT win, not a short-context one — at one page it is 19%, and `scr batch` at
    //   bs=8/48 tokens never leaves one page. Measure the multi-page case or the win is invisible.
    //
    // ✅ AND THE OP-COUNT OBJECTION TO THE COLLAPSE IS DEAD, MEASURED. The collapsed fold emits one op
    //   per (request, kv head) — `nkvh*mq` per pass against the shipped `nkvh` per pass — which is the
    //   SAME op count over the step, since the shipped fold runs `mq` passes. Each op also computes ONE
    //   row where the shipped one computes `mq` and masks `mq-1` away. The per-op price is measured:
    //   forcing `ScoreArm::PerHead` emits 4x the ops (32 of `y=1` instead of 8 of `y=gqa`) at IDENTICAL
    //   launch count (1082), trips and rows, and bs=8 compute went 51.0 → 63.3 ms — +12.3 ms over +8640
    //   op executions = **1.42 µs per op**, against ~28 µs for a fold pass. Op count is ~20x cheaper
    //   than a pass. (Corollary, unrelated but measured here: `ScoreArm::choose` picks correctly at
    //   hd=64 — the batched arm really is faster on the card, 51.0 vs 63.3, so that cost model needs no
    //   revisit.)
    //
    // ⛔⛔⛔ AND THE REQUEST DOES **NOT** GO ON `y`. That form — a per-batch 3-D `[y,in,out]` kernel, one
    //   distinct weight start per core — is REFUTED ON CARD: it faulted at
    //   `job_bin_ptr + numCoresUsed_*128` at rungs 2, 4 and 8 alike (syndrome/locator/case set
    //   bit-identical, only the index moving), one flit past the program's per-core patch table, and rung
    //   4's `{mb:1, y:4}` is the on-hardware-proven solo-decode split, so the `y`-split value is
    //   exonerated and the KERNEL RANK is what the address is made of. What the gather bought is that the
    //   dim is unnecessary: the scratch is contiguous and request-MINOR, so request `r`'s block is a baked
    //   OFFSET (`GatherScratch::kernel_row_off`) exactly as a GQA group's kv head is, and `y` stays on the
    //   group with the bare 2-D kernel that ships. See the collapsed-fold arm in `assemble_attn_block`.
    // ⭐⭐⭐⭐⭐ THE COLLAPSE, ON — and every half of it comes from ONE `Option`.
    //
    // ⛔ THE COUPLING IS THE POINT, AND IT IS WHY THIS IS ONE EXPRESSION. Four things have to be true
    // together or the fold reads another request's keys with a clean bake: (a) the caller has an index
    // tensor it will STAGE (`kv_block_index` — a name no placement matches is a bind the launcher skips
    // in silence, and a skipped index reads as entry 0, which is a REAL address), (b) the geometry admits
    // a flat block copy (`GatherScratch::of_fold_pass`), (c) the score and value legs both take the
    // request axis, and (d) the runtime drops the per-pass KV shift and blocks the mask per PAGE. (a) and
    // (b) are decided here; (c) rides on this value into every fold block; (d) is the manifest flags set
    // below, derived from this same value.
    //
    // `None` — a caller with no index, or a head dim above one stick — emits EXACTLY the bundle that
    // shipped, byte for byte.
    // ⭐ `zip`, not `and_then` + `map`: both halves are wanted TOGETHER or not at all, and stating it as
    // one combinator is what clippy's `manual_option_zip` asks for (`-D warnings` is the CI gate, and
    // `#[allow]` is not available). `of_fold_pass` is pure arithmetic over the pool's two extents, so
    // evaluating it for a caller with no index costs nothing and decides nothing.
    // ⭐⭐⭐⭐⭐ PAGE GRANULARITY: THE WINDOW COUNT IS NOT AN ARGUMENT ANY MORE, AND THAT IS THE FIX.
    //
    // `of_fold_pass` took `SlotWindow::count_in(active_cap)` because a scratch ROW was a 64-slot window, so
    // the row count carried `nb` — and the emitter's `nb` (its body's ladder rung) and the host's (the
    // CEILING rung) are DIFFERENT NUMBERS, which is the recorded defect that made every kv head above the
    // first read another head's page block. `PageScratch` holds ONE WHOLE PAGE PER REQUEST, so its extent
    // is `mq` alone: a quantity both sides read off the same rung, with no window in it to disagree about.
    //
    // A head dim above one stick is no longer a refusal either — a whole page plane is contiguous at every
    // stick-multiple `hd` (`zz_a_whole_page_plane_is_contiguous_at_every_head_dim`), so hd=128 gathers.
    // A chunked body takes its own gather (`ChunkScratch`), never the per-page one.
    let gather = if chunk.is_some() {
        None
    } else {
        kv_block_index.zip(crate::sdsc_abstract::PageScratch::of_pass(pool, width.mq()))
    };
    // The two scratches' spellings, and their FULL footprints declared up front — the same discipline as
    // every other synthetic here: a name that reaches its first access undeclared is a build panic, and a
    // shape declared smaller than the ops write aliases whatever the allocator handed out next.
    let (gkt, gv) = (syn(R::GatherKt), syn(R::GatherV));
    if let Some((_, scratch)) = gather
        && let Some(l) = layout
    {
        for r in [R::GatherKt, R::GatherV] {
            // ⛔ ONE VALUE, NOT A SLICE THE CALLER ASSEMBLES. `&[scratch.rows(), scratch.cols()]` is two
            // same-typed numbers in a literal, so the wrong pair or the wrong order compiled and
            // reserved the wrong footprint for a tensor whose overrun is the next intermediate's bytes
            // read as block numbers. See `GatherScratch::footprint_dims`.
            l.synth(out_id.synth(r), &scratch.footprint_dims());
        }
    }
    let _ = (rows_are_requests, batched.is_some());
    // ⭐⭐⭐ A WHOLE-STICK PROMPT CHUNK'S NEW BLOCK IS ONE BLOCK PER KV HEAD (`assemble_new_block_chunk`); every
    // other launch keeps the sub-blocks below, byte for byte. The same law (`NewBlockForm::of`) tells the
    // host how to lay out the causal mask this reads.
    let new_form = crate::sdsc_abstract::NewBlockForm::of(mq, hd, rows_are_requests);
    if new_form == crate::sdsc_abstract::NewBlockForm::Chunk {
        let (sc, esub, exp) = (
            syn(R::NewChunkSc),
            syn(R::NewChunkESub),
            syn(R::NewChunkExp),
        );
        let cols = BlockCols::of_new_block(mq_pad_t.cols());
        // The chunk's own row blocks over the new block's `mq_pad` columns: a kv head at one slab, a query
        // head above it.
        let blocks = crate::sdsc_abstract::ChunkRowBlocks::of_group(
            nkvh,
            gqa,
            mq,
            hd / BlockCols::of_one_stick(Lanes::FP16).get(),
            cols.get(),
        );
        let g_rows = blocks.rows();
        if let Some(l) = layout {
            // ⭐⭐ THE WHOLE BLOCK NEVER LEAVES LX — score, `esub` and `exp` are each one core's slice of one
            // row block, reused block after block, exactly as the chunk's `cesub`/`cexp`. Every op over them
            // divides the block's rows alone, so one rows-first slice sizes all three; the read law holds each
            // op to it at build. MEASURED before this (granite-2b fp16, 512 rows, RB_TIME): the score and
            // `exp` blocks made six HBM passes per kv head — 12 MB at ~100 GB/s, the whole of the block's
            // 119 µs — and the block was 18% of a ~1k prompt's device time.
            //
            // ⛔ THE PINS COME OUT OF #675's ONE WINDOW BUDGET (`lx_window_fits`): `synth_lx_core_local`
            // panics at expansion when the core-local pins and the op-local slots would meet, so a geometry
            // whose block does not fit is a build failure, never an on-card overlap.
            for (r, inputs) in [
                (R::NewChunkSc, 2u32),
                (R::NewChunkESub, 2),
                (R::NewChunkExp, 1),
            ] {
                let slice = crate::emit::pointwise_untiled_slice_bytes(g_rows, cols.get(), inputs)
                    .ok_or_else(|| {
                        SuperDscError(format!(
                            "attention o{t}: the {g_rows}-row causal new block is time-tiled at {} \
                             columns, so its {r} has no core-local slice to pin in LX",
                            cols.get()
                        ))
                    })?;
                l.synth_lx_core_local(out_id.synth(r), slice);
            }
        }
        assemble_new_block_chunk(
            &mut ops,
            NewChunkAttn {
                t,
                nqh,
                nkvh,
                hd,
                blocks,
                mq,
                cols,
                qs,
                new_kt: &new_kt,
                new_v,
                cmask,
                bmm_form,
                bufs: &bufs,
                sc: &sc,
                esub: &esub,
                exp: &exp,
            },
            sym_id_base,
            layout,
        )?;
    }
    // The sub-blocked passes: none when the chunk form above emitted the block.
    let sub_block_passes = if new_form == crate::sdsc_abstract::NewBlockForm::Chunk {
        0
    } else if per_req {
        mq
    } else {
        1
    };
    for r in 0..sub_block_passes {
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
    //    writes (see main's `lower_attn_node`) — same discipline as kct_base, applied to a differently-shaped
    //    (still nqh-sized) tensor. ──
    // The prefix fold: ONE variant, re-launched once per page of resident prefix with that page's
    // own validity row. Emitted by POSITION so a rename cannot drop an op out of the fold.
    //
    // A zero-masked second variant for full pages was tried, to avoid shifting the mask per page.
    // It broke multi-page coherence and it was chasing a cost that is not there: the mask's segment
    // is ~1.6 MB in the DECODE bundle (the ~150 MB figure is the PREFILL bundle, where mq=96 scales
    // every intermediate, and prefill runs once per request).
    //
    // ⭐⭐⭐⭐⭐ UNLESS THE BODY IS CHUNKED: then every resident page is attended HERE, in this launch, and
    // the per-page fold below sweeps nothing (`gather` is `None` and the sweep extent is zero), so it emits
    // no op and marks no fold. The chunk ops are NOT fold ops — the body launches once.
    if let Some(chunk) = chunk {
        let index = kv_block_index.ok_or_else(|| {
            SuperDscError(format!(
                "attention o{t}: a chunked body gathers its pages, but no index tensor was named — a \
                 gather with no staged index reads block 0 for every page"
            ))
        })?;
        // ⭐ THE ONE-REQUEST DECODE ABOVE ONE SLAB RE-LAYS ITS QUERY GROUP-MAJOR (`QGroupBlock`), so each kv
        // head attends its whole GQA group in one score and one value op per slab — the reference's form
        // (one `QKᵀ` per kv head's group). Minted only for a one-request decode above one slab; every other
        // chunk keeps the stream's own row blocks. The copy reads the class's identity table, which a
        // bundle without a layout (an emission fixture) does not have, and so keeps the per-head form.
        // ⭐ AND A DECODE BATCH TOO: its rows are head-major (`h·B + r`), so the same one-stick copy at
        // `M = B` re-lays each kv head's `gqa·B` rows into one block, rows `g·B + r` — the batch mask's own
        // `row % B` law. A prompt chunk keeps the stream's row blocks.
        let qgroup = if rows_are_requests {
            crate::sdsc_abstract::QGroupBlock::of_batch(nkvh, nqh / nkvh, hd, mq)
        } else {
            crate::sdsc_abstract::QGroupBlock::of_decode(nkvh, nqh / nkvh, hd, mq)
        }
        .zip(layout.and_then(|l| l.identity_tid_of_class(hd)));
        // The row blocks the body cuts every wide op into (`ChunkRowBlocks`): a kv head's `gqa·mq` rows at
        // one slab, a query head's `mq` above it, a kv head's re-laid group where the query is re-laid.
        let blocks = match qgroup {
            Some((q, _)) => chunk.row_blocks_of_query_groups(q).ok_or_else(|| {
                SuperDscError(format!(
                    "attention o{t}: the re-laid query group block does not describe this chunk's {} kv \
                     heads at head dim {hd}",
                    chunk.nkvh()
                ))
            })?,
            None => chunk.row_blocks(nqh / nkvh, mq),
        };
        let b_rows = blocks.rows();
        // A PROMPT chunk keeps its probabilities in LX from `esub` through the PV (the body's step 4′);
        // decode keeps the form it has. ⛔ A batch chunk's `mq` is its request count, not a prompt's width:
        // its rows are separate requests and it keeps the decode form (`ChunkHistory::Requests`).
        let prompt = mq > 1 && !rows_are_requests;
        let lx_parts = if prompt {
            chunk.pages() / prompt_lx_part_pages(chunk, b_rows)
        } else {
            0
        };
        let red_cut = reduction_part_pages(chunk, b_rows) < chunk.pages();
        let n_red = if prompt { lx_parts + 2 } else { 0 }.max(if red_cut { 3 } else { 0 });
        // ⭐ THE GATHER STAYS IN LX for the ONE-REQUEST decode at the geometry `lx_form` admits — the
        // decode PV divides its contraction across the cores (`ContractionSplit`), which is the division the
        // copy already has there. A prompt chunk's PV divides its rows, and a batch chunk's rows are other
        // requests: both keep the HBM scratch. The host stages the table from the same `(chunk, rows)`.
        let chunk_rows = if rows_are_requests {
            crate::sdsc_abstract::ChunkRowKind::Batch
        } else if mq == 1 {
            crate::sdsc_abstract::ChunkRowKind::OneRequest
        } else {
            crate::sdsc_abstract::ChunkRowKind::Prompt
        };
        // The op-local term of the window budget: the rung's pin slots, read off the layout. The declare
        // pass has already walked prefix, every body and the suffix against THIS layout before any real
        // pass ran, so the slot map is complete — the FINAL sum, never a mid-walk projection. `0` when
        // no layout is in scope (a layout-less emission mints no pins at all).
        let op_local_bytes = layout.map(|l| l.lx_op_local_slot_bytes()).unwrap_or(0);
        // ⭐ AND THE PROBABILITY-PART TERM, from the ONE enumeration the mint below also declares
        // (`chunk_prob_pins`): the witness's core-local sum is the gathers PLUS these pins — the sum
        // the allocator bumps — so a geometry the window cannot hold falls to HBM HERE instead of
        // panicking the mint. `0` when the part is time-tiled: that emission refuses at the mint
        // regardless, so the admission's sum there is moot. Recorded on the layout so the artifact
        // carries the term the host's own budget read has to match (the bake/host agreement law).
        // The chunk's own division (`ChunkBufs::sm_division_of` — #692's sticks-first batch law)
        // sizes the parts HERE too, so the witness's term and the ops' own division read one value.
        let sm_division = ChunkBufs::sm_division_of(rows_are_requests);
        let (prob_part_pages, prob_pins) = chunk_prob_pins(chunk, b_rows, prompt, sm_division);
        let prob_pin_bytes = prob_pins
            .as_ref()
            .map(|pins| pins.iter().map(|&(_, _, bytes)| bytes).sum())
            .unwrap_or(0);
        if let Some(l) = layout {
            l.note_chunk_prob_pin_bytes(prob_pin_bytes);
        }
        let lx = chunk
            .lx_form(chunk_rows, op_local_bytes, prob_pin_bytes)
            .map(|form| ChunkLxBufs {
                form,
                keys: form.keys().map(|keys| {
                    let names = (0..keys.tensors()).map(|i| syn(R::GatherKtLx(i))).collect();
                    (keys, names)
                }),
                v: (0..form.lx_tensors())
                    .map(|i| syn(R::GatherVLx(i)))
                    .collect(),
            });
        let cb = ChunkBufs {
            gkt: syn(R::GatherKt),
            gv: syn(R::GatherV),
            lx,
            csc: syn(R::ChunkSc),
            cesub: syn(R::ChunkESub),
            cexp: syn(R::ChunkExp),
            sm_division,
            red_division: ChunkBufs::red_division_of(rows_are_requests),
            red: (0..n_red).map(|i| syn(R::ChunkRed(i))).collect(),
            qgroup: qgroup.map(|(_, ident_tid)| QGroupBufs {
                block: syn(R::ChunkQGroup),
                ident: crate::place::act_name(ident_tid),
            }),
        };
        if let (Some(l), Some((q, _))) = (layout, qgroup) {
            l.synth(out_id.synth(R::ChunkQGroup), &q.footprint_dims());
        }
        if let Some(l) = layout {
            match &cb.lx {
                None => {
                    for r in [R::GatherKt, R::GatherV] {
                        l.synth(out_id.synth(r), &chunk.footprint_dims());
                    }
                }
                Some(lx) => {
                    // One core-local pin per LX tensor, each one entry per core: V's per (part, kv head,
                    // slab), Kᵗ's per (part, kv head). `ChunkLx::pinned_bytes_per_core` in all, which
                    // `lx_form` already held to the frontend's LX window: 1 MiB at granite-2b's 64-page
                    // rung, 1.125 MiB at granite-8b's 40-page rung. A chunk with a Kᵗ tail keeps Kᵗ
                    // whole in HBM (`ChunkLx::keys`), and V's tail stays in HBM.
                    //
                    // Declared interleaved, V's tensor `i` then Kᵗ's: the pins are bump-allocated in
                    // declaration order, and at one slab this is the order (and so the addresses) the
                    // form has always had.
                    let k_tensors = lx.keys.as_ref().map_or(0, |(keys, _)| keys.tensors());
                    for i in 0..lx.form.lx_tensors().max(k_tensors) {
                        if i < lx.form.lx_tensors() {
                            l.synth_lx_core_local(
                                out_id.synth(R::GatherVLx(i)),
                                lx.form.per_core_bytes(),
                            );
                        }
                        if i < k_tensors {
                            l.synth_lx_core_local(
                                out_id.synth(R::GatherKtLx(i)),
                                lx.form.per_core_bytes(),
                            );
                        }
                    }
                    if lx.keys.is_none() {
                        l.synth(out_id.synth(R::GatherKt), &chunk.footprint_dims());
                    }
                    if lx.form.tail_pages() > 0 {
                        l.synth(out_id.synth(R::GatherV), &lx.form.tail_footprint_dims());
                    }
                }
            }
            // Every query ROW's scores over the chunk — `nqh·mq` rows, the token stream's own; a decode
            // chunk's probabilities too.
            l.synth(out_id.synth(R::ChunkSc), &[nqh * mq, chunk.slots()]);
            if !prompt {
                l.synth(out_id.synth(R::ChunkExp), &[nqh * mq, chunk.slots()]);
            }
            // The partial reductions exist only where they are used, so every bundle that needs none is
            // laid out exactly as before.
            for i in 0..n_red {
                l.synth(
                    out_id.synth(R::ChunkRed(i)),
                    &[nqh * mq, BlockCols::of_one_stick(Lanes::FP16).get()],
                );
            }
            // ⭐ CORE-LOCAL LX, each pin one core's slice of one row block's probability part under its own
            // producer's division — `esub`'s always, and a one-slab prompt chunk's `exp` too. See the body's
            // step 4. The rows are `chunk_prob_pins`' — the SAME enumeration the admission summed into
            // the window budget above, each pin already sized under the chunk's own division (#692's
            // sticks-first batch law included) — so what the witness budgets is exactly what this mint
            // declares.
            match &prob_pins {
                None => {
                    return Err(SuperDscError(format!(
                        "attention o{t}: a {prob_part_pages}-page probability part of {b_rows} rows is \
                         time-tiled, so it has no core-local slice to pin in LX"
                    )));
                }
                Some(pins) => {
                    for &(r, _inputs, slice) in pins {
                        l.synth_lx_core_local(out_id.synth(r), slice);
                    }
                }
            }
        }
        // ⭐ WHOSE HISTORY THE ROWS ATTEND — from the same `rows_are_requests` the per-page fold reads its
        // per-row mask under. A batch's law is minted HERE, from this chunk and this width, so the host's
        // staging (which reads the same law off the selected body's swept extent) cannot describe a
        // different layout; a chunk that is not a whole number of pages per request is a build failure.
        let history = if rows_are_requests {
            // The law of THIS body's own row blocks: the query-group blocks where the query is re-laid,
            // the stream's per-head blocks where it is not (a registry-less fixture).
            let bc = crate::sdsc_abstract::BatchChunk::of_blocks(chunk, mq, blocks).ok_or_else(|| {
                SuperDscError(format!(
                    "attention o{t}: a {}-page batch chunk is not a whole number of pages for each of \
                     its {mq} request row(s)",
                    chunk.pages()
                ))
            })?;
            // ⛔ AND THE MASK BLOCK MUST LIE INSIDE THE MASK's OWN PLACEMENT. Every row block reads it at
            // offset 0, so a block past the placement reads the next activation's bytes as validity — an
            // additive mask, where an unstaged zero is VALID. The placement is the bundle's, so this is a
            // bake fact; refused here it is a build failure, never a row attending another's keys.
            if let Some(l) = layout
                && let Some(crate::place::PlaceId::Act(tid)) = l.id_of(pmask)
                && let Some(p) = l.placements.get(&tid)
                && p.size < bc.mask_elems() as u64 * 2
            {
                return Err(SuperDscError(format!(
                    "attention o{t}: the batch chunk's mask block ({} x {} fp16) is past the mask's {} B \
                     placement",
                    bc.mask_rows(),
                    bc.mask_cols(),
                    p.size
                )));
            }
            ChunkHistory::Requests(bc)
        } else {
            ChunkHistory::Shared
        };
        assemble_attn_chunk(
            &mut ops,
            ChunkAttn {
                t,
                nqh,
                hd,
                gqa: nqh / nkvh,
                mq,
                chunk,
                history,
                index,
                kct,
                vc,
                qs,
                pmask,
                bmm_form,
                bufs: &bufs,
                cb: &cb,
            },
            sym_id_base,
            layout,
        )?;
    }
    let sweep_cap = if chunk.is_some() { 0 } else { active_cap };
    let fold_from = ops.len();
    // ⭐⭐⭐⭐⭐ THE WINDOW-CHUNK SPLIT — the gathered fold's own answer to the dxp group ceiling.
    //
    // A gathered PageFold run is UNCHUNKABLE by the group walk (`GroupKind::run_may_be_chunked ==
    // false`): the loop is GROUP-MAJOR (`for op { for rep }`), so cutting the run mid-fold leaves
    // group B's passes reading the scratch page group A's copy left. Left alone, that exemption
    // makes one group per fold whose descriptor count grows with `mq · nkvh · nslab · nb` — at
    // gemma-4-12b's class-0 geometry (mq=32, nkvh=8, nslab=4, nb=4) it is 8,300 descriptors, and a
    // `dxp_standalone` child of that group was MEASURED at 2.5+ hours of compile against the
    // 30-minute whole-bake ceiling. The bake-side guard (`oversized_group`) turns that into a build
    // error naming the group; THIS is the emission-side half that keeps gemma buildable at all.
    //
    // ⭐ THE CUT IS PER **WINDOW CHUNK**, AND EACH CHUNK CARRIES ITS OWN GATHER COPIES.
    //
    // A chunk `[copies][windows of the chunk]` is SELF-CONTAINED under the group-major law: its
    // copy runs at the head of its OWN group, so pass `p` of chunk `c` re-fills the scratch with
    // page `p` before chunk `c`'s legs read it — the same reason the copies sit inside the fold
    // group at all. The online-softmax state is ACCUMULATE-ONLY across (window, pass), so which
    // group a window's contribution lands in cannot change the answer — the same argument the
    // codebase records for the ungathered fold's lawful chunking. What the split must NOT do is
    // break a pass's recurrence (copies → legs → ladder per page), and it cannot: every op of one
    // chunk's window run is inside one group, and the chunk boundary is the only cut.
    //
    // ⛔ THE BOUNDARY MECHANISM IS THE EXISTING REQUEST TAG, AND IT IS ADDRESSING-INERT HERE.
    // `Trip::fusable_with` breaks any run where the request changes, so tagging chunk `c`'s fold
    // ops with `kv_request = c` breaks the groups BETWEEN chunks (and only between them — the tag
    // is per-op uniform within a chunk). For a gathered fold op the tag reaches no address: the
    // KV base comes from the fold pass (zero, gathered — absolute entries), the write cursor and
    // slab shifts are cache-write/restickify concerns a fold op has neither of
    // (`fold_plan::nonfold_page_delta` short-circuits on `is_paged_fold`), and the intermediate
    // rebase multiplies the PASS's request under a per-request regime — the gathered fold is
    // whole-batch, stride 0. Pinned by `fold_plan::tests::a_gathered_fold_ops_request_tag_moves_no_address`.
    //
    // ⛔ CHUNK 0 KEEPS REQUEST 0, so every under-target bundle — granite-8b's own bake included —
    // emits the SAME trips, the SAME partition and the SAME fingerprint as before this existed:
    // `k == 1` tags nothing and reorders nothing. The gate is the PROJECTED size in closed form
    // at the emission site — copies `2 · mq · ops_per_row` (one per plane per request-run) plus
    // `nb · (2·mq·nkvh·nslab legs + 11 ladder)` per window — against `DxGroupCeiling::SPLIT_TARGET`
    // (the compile-cost target, ≤ the bake-side guard's `MAX_DESCRIPTORS` ceiling, so a split that
    // fires here always lands under what the guard would refuse): the emitter aims BELOW what the
    // guard forbids, and the two consts cannot disagree about legality.
    let nslab =
        crate::addr::Shape::<0, 0, 0, 0>::slabs_of(hd, crate::superdsc_opspec::Df::Fp16).get();
    let nb = SlotWindow::count_in(crate::sdsc_abstract::SlotCount::new(active_cap)).get();
    let chunking = gather.map(|(_, scratch)| {
        // ONE CHUNK'S FIXED COST (its own copies) and PER-WINDOW COST (both legs + the 11-op
        // ladder), straight off the scratch's own extents — the quantities `copies()` and the two
        // gathered leg loops below actually emit, so the projection cannot drift from the emission.
        let per_chunk_fixed = 2usize * scratch.mq() as usize * scratch.ops_per_row() as usize;
        let per_window = 2usize * scratch.mq() as usize * scratch.nkvh() as usize * nslab as usize
            + LADDER_OPS_PER_WINDOW;
        let projected = per_chunk_fixed + nb as usize * per_window;
        // ⭐ `k = 1` WHENEVER ONE CHUNK FITS — the byte-identity case. The ceil-division only ever
        // ADDS chunks, and the fixed copies ride every chunk, so `k` is the least count whose every
        // chunk is at or under the ceiling; a fold that cannot fit even alone (window count 0, or a
        // single window past the ceiling) stays `k = 1` and lets the bake-side guard refuse it by
        // name rather than looping forever here.
        let k = if projected <= DxGroupCeiling::SPLIT_TARGET || nb == 0 {
            1
        } else {
            // Least k with per_chunk_fixed + ceil(nb/k)·per_window ≤ target: the windows are
            // distributed as evenly as the per-window cost allows, remainder to the LAST chunks
            // (chunk 0 keeps the floor, so the first group stays the SMALLEST — the one whose
            // byte-identity matters most is the one least likely to move).
            let mut k = 1usize;
            while k < nb as usize {
                let windows = (nb as usize).div_ceil(k);
                if per_chunk_fixed + windows * per_window <= DxGroupCeiling::SPLIT_TARGET {
                    break;
                }
                k += 1;
            }
            k
        };
        (k, per_chunk_fixed, per_window)
    });
    // The chunk a window index belongs to: consecutive windows grouped `ceil(nb/k)` per chunk, the
    // short remainder falling to the last chunk. `k == 1` puts every window in chunk 0, which is
    // the unsplit sweep.
    let k_chunks = chunking.map_or(1, |(k, _, _)| k);
    let windows_per_chunk = (nb as usize).div_ceil(k_chunks);
    let chunk_of = |w: u32| -> u32 { (w as usize / windows_per_chunk.max(1)) as u32 };

    // ⭐⭐⭐⭐⭐ THE GATHER ITSELF — TWO KERNEL-LESS COPIES, FIRST IN THE FOLD GROUP (FIRST IN **EACH
    // CHUNK'S** GROUP WHEN THE FOLD SPLITS).
    //
    // ⛔ THEY MUST BE **IN** THE GROUP, WHICH IS WHY THEY SIT AFTER `fold_from`. `reps` is per-group: the
    // runtime relaunches everything from here once per PAGE, and pass `p` needs pass `p`'s blocks in the
    // scratch. A copy outside the group would run once and every pass but the first would score against
    // page 0's keys — and since the mask is now blocked per page, nothing would mask that away.
    //
    // ⛔ AND ONE COPY PER **INDEX STICK** OF THE PASS, NOT PER WINDOW AND NOT ONE FOR THE WHOLE PASS.
    // The per-head and per-window terms ride in the INDEX ENTRY (`GatherScratch::block_in_page`) rather
    // than in a base offset, because an `EwOperand` carries one `col_offset` and there is nowhere for a
    // second base to go — which is exactly why the entries are GLOBAL stick-block numbers. But one op
    // for the whole pass declares `nkvh * nb * mq` entries, and dxp loads a gather's index ONE STICK at
    // a time: above 32 entries the cores past the wrap read another core's page addresses, with a clean
    // bake and no fault. `GatherScratch::copies` cuts the pass into contiguous one-stick RUNS of rows —
    // the only cut a `[mb, out]` operand can name under a window-major row law — and each run carries
    // its own index base and destination base as one value.
    //
    // ⛔ AND THE SOURCE OFFSET IS STILL ZERO FOR EVERY RUN, WHICH IS LOAD-BEARING.
    // `addr = idx * skip_addr + base_addr` adds the index to the operand's OWN declared start, so the
    // plane the copy reads is the one `kct`/`vc` name and the entry supplies EVERYTHING inside the page.
    // A per-run source base would be added twice.
    //
    // ⭐ THE COPIES ARE EMITTED PER CHUNK, IMMEDIATELY BEFORE THAT CHUNK'S FIRST WINDOW — not all up
    // front. A chunk's copy must be in THE CHUNK'S OWN GROUP (see the split note above), and the window
    // sweep is the only thing that knows where a chunk begins, so the emission interleaves
    // `chunk 0 copies, chunk 0 windows, chunk 1 copies, chunk 1 windows, …`. At `k == 1` this is
    // exactly the old order — every copy, then every window — because there is one chunk and it
    // starts at window 0.
    //
    // ⛔ CHUNK 0's TAG STAYS 0 — `kv_request` is only stamped on chunks ABOVE the first, so the tag
    // the group walk sees is 0 everywhere an under-ceiling fold emits. The tag itself is stamped in
    // the fold-tag loop below, from the chunk each op's window belongs to.
    // (op index from `fold_from`, chunk) pairs — where each chunk's run of fold ops begins. The
    // tag loop below stamps `kv_request` from these, so the boundary the group walk sees is
    // exactly the boundary the emission cut at.
    let mut chunk_starts: Vec<(usize, u32)> = Vec::new();
    let mut emitted_chunk_copies: Option<u32> = None;
    // ⛔ A SWEEP WITH NO WINDOWS STILL EMITS THE COPIES — the shipped emission pushed them
    // unconditionally, before any window existed to interleave them with, so an `active_cap` under
    // one window (nb == 0) keeps its (dead but shipped) copies and the bundle does not move.
    if nb == 0 && gather.is_some() {
        chunk_starts.push((ops.len() - fold_from, 0));
        if let Some((idx, scratch)) = gather {
            for (name, src, dst) in [
                (format!("attn_gkt_o{t}"), kct, gkt.as_str()),
                (format!("attn_gv_o{t}"), vc, gv.as_str()),
            ] {
                for cp in scratch.copies() {
                    ops.push(crate::ir::bridge::tiled_op_sdsc_op::assemble_gather_copy(
                        &format!("{name}_s{}", cp.stick()),
                        src,
                        dst,
                        cp,
                        idx,
                        sym_id_base,
                        layout,
                    ));
                }
            }
        }
        emitted_chunk_copies = Some(0);
    }
    // ⭐ WHERE THE FOLD'S KV COMES FROM: the gathered scratch when this bundle gathers, the paged pool
    // otherwise. Named once here so the four things that must agree — the operand SPELLING, the declared
    // kernel PITCH, the per-head base OFFSET and the `y` step — are one decision per plane instead of four
    // at the call site. The scratch's pitch is ONE WINDOW on both planes (`[hd,64]` for Kᵗ, `[64,hd]` for
    // V), where the pool's are a whole page.
    // ⭐⭐⭐⭐⭐ THE PITCHES ARE NOW THE POOL'S ON BOTH PATHS, AND ONLY THE TENSOR NAME DIFFERS.
    //
    // A page-granular scratch row is a byte-for-byte copy of the page plane, so the gathered operand's
    // GEOMETRY IS the pool's — same declared pitch, same internal arrangement, and (via
    // `PageScratch::coord_off`) the same address law offset by one request row. There is nothing left for
    // a `Some` arm to declare differently.
    //
    // ⛔ WHAT THIS REPLACES WAS THE WHOLE BUG SURFACE. The gathered arms used to declare ONE WINDOW
    // (`[hd,64]` for Kᵗ, `[64,hd]` for V) where the pool declares a whole page — a second geometry, whose
    // window term was re-derived in `kernel_row_off` rather than taken from `PagedKvPool::addr`. Two
    // geometries for one read is what let the emitter's `nb` and the host's disagree.
    let kt_src = match gather {
        Some(_) => gkt.as_str(),
        None => kct,
    };
    let kt_pitch = crate::sdsc_abstract::PagedKvPool::KT_KERNEL_PITCH;
    let v_src = match gather {
        Some(_) => gv.as_str(),
        None => vc,
    };
    let v_pitch = crate::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32;
    // Windows ONE launch folds. `active_cap` is the sk_bucket rung as before, and the page equals the
    // pre-paged capacity, so this is bit-for-bit the baseline's fold; contexts past one page cost
    // additional LAUNCHES of this same group, never a wider sweep. The `active_cap / 64` lives in
    // `SlotWindow::count_in`, where the 64 is the window's SLOT count — quantity (2), the reduce's
    // one-stick column budget — not the lane count it used to be spelled as.
    for w in SlotWindow::sweep(crate::sdsc_abstract::SlotCount::new(sweep_cap)) {
        let b = w.index();
        // ⭐ A NEW CHUNK STARTS HERE — its own copies first (the self-containment the group-major
        // law demands; see the split note above). `emitted_chunk_copies` is `None` before the first
        // window, so chunk 0's copies land exactly where the single block of copies used to —
        // before the first `assemble_attn_block` of the sweep — and an unsplit fold (`k == 1`)
        // emits the old sequence byte for byte.
        let chunk = chunk_of(b);
        if emitted_chunk_copies != Some(chunk)
            && let Some((idx, scratch)) = gather
        {
            chunk_starts.push((ops.len() - fold_from, chunk));
            for (name, src, dst) in [
                (format!("attn_gkt_o{t}"), kct, gkt.as_str()),
                (format!("attn_gv_o{t}"), vc, gv.as_str()),
            ] {
                // ⛔ THE STICK IS IN THE NAME AT EVERY WIDTH, including the single-stick case. Two ops
                // that differ only in a baked offset and share a name are indistinguishable in every
                // descriptor diff, every `[fold-block]` trace and every launch table — and the
                // one-stick case is precisely the shape that is already proven on card, so it is the
                // one whose identity must stay legible.
                //
                // ⭐ AND THE **CHUNK** IS IN THE NAME OF EVERY CHUNK ABOVE THE FIRST, for the same
                // reason: a split fold's chunk-1 copies are descriptor-identical to chunk 0's (they
                // copy the same plane through the same index), so without the segment the two chunks
                // of a split fold could not be told apart in any dump. Chunk 0 keeps the shipped name.
                for cp in scratch.copies() {
                    let op_name = if chunk == 0 {
                        format!("{name}_s{}", cp.stick())
                    } else {
                        format!("{name}_c{chunk}_s{}", cp.stick())
                    };
                    ops.push(crate::ir::bridge::tiled_op_sdsc_op::assemble_gather_copy(
                        &op_name,
                        src,
                        dst,
                        cp,
                        idx,
                        sym_id_base,
                        layout,
                    ));
                }
            }
            emitted_chunk_copies = Some(chunk);
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
            kt_src,
            kt_pitch,
            // THE POOL COMPOSES THE REQUEST TERM, not this call site. `kt_block_base_of` is the same
            // `block_index(kvh) + r` the cache write bakes, so the fold's kernel and the write that
            // fills it cannot disagree about where request `r` lives — and no stride is spelled here.
            //
            // ⛔ AND THIS IS THE UNGATHERED READ ONLY. A gathered bundle's score and value legs take the
            // collapsed per-request arm, whose kernel base is the SCRATCH's own row
            // (`GatherScratch::kernel_row_off`, reached through `GatheredFold`) — a request coordinate this
            // closure has no parameter for. A `gather` arm here would be a base no op reads.
            |h, sl| {
                use crate::sdsc_abstract::{KvCoord, KvPlane};
                let kvh = crate::sdsc_abstract::KvHead::of_query(h, crate::addr::Gqa::new(gqa));
                crate::addr::DevOff::from_view_step(
                    pool.addr(
                        KvCoord::block(KvPlane::Kt, kvh)
                            .at_slot(w.first_slot())
                            // The contraction's head-dim slab, through the pool's OWN model — the same
                            // `at_feat` door the V plane below uses, so the two planes cannot disagree
                            // about where a feature slab is.
                            .at_feat(FeatIdx::of_slab(sl)),
                    ),
                )
            },
            // The closure takes the head-dim SLAB as well as the head, so the slab rides inside the
            // pool's own model instead of being added to a finished address. The V operand's row count
            // is the PAGE, not `cap`, and it reaches the address only through that model — there is no
            // separate row-count argument to disagree with the kernel's declared `in` extent.
            v_src,
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
            v_pitch,
            // NAMED, NOT COMPOSED: `w` is a slot window and `sl` a feature slab, so both are
            // COORDINATES. The hand-added `sl * PAGE_SLOTS * stick` was the V plane's own feature-stick
            // stride restated at the call site.
            //
            // ⛔ UNGATHERED ONLY, for the same reason as the Kᵗ closure above: a gathered bundle's value
            // leg reads the V scratch through `GatheredFold::kernel_off`, which needs a request.
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
            // mask spans the whole context (the derived page ceiling × PAGE_SLOTS), so one row per
            // op row is `nqh*mq` of them: tens of MB at prefill's mq=96, re-uploaded every forward.
            // At a decode batch of 8 it is a few MB, which is affordable — the blow-up is prefill's
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
            // ⭐ THE PREFIX FOLD GOES PER-REQUEST INSIDE ONE PASS when the bundle gathers. That is what
            // makes one pass serve the whole batch, so the runtime can drop the `× requests` factor from
            // `reps` — see `fold_plan::reps`, gated on the `batched_requests` this sets below.
            //
            // ⛔ THE **WINDOW** IS PART OF IT, and it is why this value is built here rather than once
            // above the loop: the kernel base an op reads is the scratch row for (kv head, THIS window,
            // request), the same window whose mask slab and block name this call already carries. A value
            // hoisted out of the loop would give every window window 0's kernel rows.
            gather.map(|(_, scratch)| GatheredFold {
                scratch,
                pool,
                window: w,
            }),
            &bufs,
            sym_id_base,
            layout,
        )?;
    }

    for (i, op) in ops[fold_from..].iter_mut().enumerate() {
        op.kv_page_fold = true;
        // ⭐⭐⭐⭐⭐ THE CHUNK TAG — the fold's run-break between window chunks, from the chunk_starts
        // the sweep itself recorded. `kv_request` is the ONE field `Trip::fusable_with` already breaks
        // any run on, so this is the whole mechanism that turns k chunks into k launch groups; and it
        // is addressing-inert for a gathered fold op (see the split note above, pinned by
        // `fold_plan::tests::a_gathered_fold_ops_request_tag_moves_no_address`). Chunk 0's ops and every
        // op of an unsplit fold keep 0, so the group walk's view of an under-ceiling bundle does not
        // move.
        if let Some(&(_, c)) = chunk_starts.iter().rev().find(|(s, _)| *s <= i) {
            op.kv_request = c;
        }
        // DECLARED, NOT INFERRED: the runtime cannot see what rows an op was baked for, and claiming a
        // whole-batch pass the kernels do not serve turns `pages × requests` passes into `pages` while
        // every row but one reads the wrong history. It is the SAME `gather` the per-window
        // `GatheredFold` above is built from, so the flag and the ops cannot disagree.
        op.kv_batched_requests = gather.is_some();
        // ⭐⭐⭐ AND THAT THE AXIS IS OVER A **GATHERED** OPERAND, which is what makes `LaunchPages::Affine`
        // stop being a precondition. `Affine` says the HOST can reach every row's page by one stride — a
        // launch-time fact that hands the device no axis, and false for any free-list allocation (rows at
        // pages 0,1,3,7 have no single stride, which is precisely the case the gather exists for). A
        // gathered fold resolves each row's page in the INDEX instead, so it collapses over any pool.
        //
        // ⛔ IT IS A SEPARATE FLAG AND NOT IMPLIED BY THE ONE ABOVE. They happen to move together in this
        // emitter today, but `collapsed` reads `batched_requests && (gathered || Affine)`: the axis alone
        // over an ungathered pool still needs the stride proof, and folding the two into one bool would
        // discard that proof for a future bundle that has the axis without the gather.
        op.kv_gathered = gather.is_some();
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
    // ⛔ THE HANDOFF FACT AND THE PROGRAM'S OWN WIDTH MUST AGREE. The fact is minted from the
    // bundle's attention-output row count — the same count this emission's `width` states — so a
    // handoff at any other width, or at a single slab (where there is nothing to restructure), is
    // a caller/program desync in which each side alone emits HALF a restructure: a slab-major
    // `out` read by an unswapped o_proj weight, or the reverse. Fluent, wrong tokens, nothing
    // else — refused here so the bundle never bakes.
    if headmajor_handoff && (mq != 1 || nslab == 1) {
        return Err(SuperDscError(format!(
            "AttnDecode t{t}: the bundle states the head-major o_proj handoff, but this program's \
             width is mq={mq} over {nslab} feature slab(s) — the handoff is minted for the \
             single-row body of a multi-slab head dim (mq == 1, nslab > 1). Half a restructure is \
             a slab-major `out` read by an unswapped o_proj weight: fix the caller's statement, \
             not the emission.",
        )));
    }
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
                    // The accumulators at their true extents — `run_o` head-wide `[rows, hd]`,
                    // `run_l` one-stick `[rows, 64]` — and `out` at the flat-equal `[rows, hd]`
                    // view of its `[mq, nqh·hd]` token-stream buffer (same element count; the
                    // mint's bound is flat).
                    In::full(&fl(&bufs.run_o, rows.get(), hd)).ew(),
                    In::col(&hm_rows(&bufs.run_l, rows.get())).ew(),
                ],
                &fl(out, rows.get(), hd),
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
                        In::sliced(
                            &fl(&bufs.run_o, rows.get(), hd),
                            nests.token_stream(hi, 0).off(),
                        )
                        .ew(),
                        In::col_at(&hm_rows(&bufs.run_l, rows.get()), nests.head_major(hi, 0)).ew(),
                    ],
                    &rb(out, mq, nqh * hd),
                    nests.token_stream(hi, 0).off(),
                    sym_id_base,
                    layout,
                ));
            }
        }
    } else if headmajor_handoff {
        // ⭐⭐⭐ THE HEAD-MAJOR HANDOFF — the single-row body's slab-major finalize, the other half
        // of the contract whose weight-copy price the route-2 note above works out. At `mq == 1` a
        // row IS a head, so o_proj's contraction axis can be permuted to meet `out` where the
        // accumulator already holds it: slab-outer, `s·nqh·64 + h·64 + d` — `run_o`'s own
        // arrangement. Each op below is ONE STICK WIDE (`nqh` rows × 64 cols), so the stick-blocked
        // write law (`(c/64)·(R·64) + r·64 + c%64` over one stick of columns) degenerates to a
        // CONTIGUOUS `nqh·64`-element block at the offset — one op per feature slab, ALL heads at
        // once, where the per-head loop below pays `nqh·nslab` single-core ops to relayout what it
        // divides. On granite-3.1-8b that is 64 ops → 2 per layer (2,480 of ~15,400 descriptor
        // executions per decode step; the baked mq=1 body loses exactly 62 descriptors).
        //
        // The read and the write share ONE offset: `head_major(0, s)` is `s·nqh·64` at `mq == 1`,
        // which is BOTH `run_o`'s slab-s plane base and `out`'s slab-s block base — the handoff
        // copies each accumulator plane to `out` VERBATIM, no relayout at all. The o matmul reads
        // the result flat (`[1, nqh·hd]`, one row) against the SECOND, k-shuffled o_proj copy the
        // same bundle fact swaps in at the matmul door: the copy's staging walk enumerates its
        // k-axis in exactly this order (`s`, then `h`, then `d`).
        for s in 0..nslab {
            ops.push(assemble_pointwise_broadcast_off(
                &format!("attn_o_s{s}_o{t}"),
                "realdiv",
                // ALL `nqh` head-major rows at once — at `mq == 1` the mask rows ARE the heads.
                RowCount::of_mask_rows(rows),
                BlockCols::of_head_slab(FeatIdx::SLAB_FEATS),
                &[
                    In::sliced(&fl(&bufs.run_o, rows.get(), hd), nests.head_major(0, s)).ew(),
                    In::col(&hm_rows(&bufs.run_l, rows.get())).ew(),
                ],
                &fl(out, rows.get(), hd),
                nests.head_major(0, s),
                sym_id_base,
                layout,
            ));
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
                        In::sliced(&fl(&bufs.run_o, rows.get(), hd), nests.head_major(hi, s)).ew(),
                        In::col_at(&hm_rows(&bufs.run_l, rows.get()), nests.head_major(hi, 0)).ew(),
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
