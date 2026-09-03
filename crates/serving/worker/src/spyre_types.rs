// SPDX-License-Identifier: Apache-2.0
//! THE NOUNS — what a loaded Spyre model IS.
//!
//! `BundleMeta` (one baked bundle: its wiring, its `BakeFacts`, its capacity), `Loaded` (the
//! sessions and the rung ladders), `ReqState` (one request's KV pages and history), and the
//! smaller value types the launch path passes around.
//!
//! ⛔ EVERY MODEL FACT IN HERE ARRIVED GENERATED. `BundleMeta` carries the `&'static Wiring` the
//! `#[forward]` macro emitted; nothing in this file inspects a checkpoint, a config, or a manifest.

use std::path::PathBuf;

use scratchy_core_model::weight::HfModelConfig;
use scratchy_subtile::sdsc_abstract::{
    BlockTable, PagedKvPool, PoolRows, PoolSplit, RowPages, SlotCount,
};
// ⭐ THE CARD PATH NO LONGER PARSES A MANIFEST. `Manifest` survives only for
// the KTIR-emulator session, whose `new_multi` takes `&Manifest` to thread its
// HBM buffers. Under `sendnn` every fact it carried comes from the GENERATED
// `SUPERDSC_WIRINGS` static instead, so the type is not even in scope.
#[cfg(not(feature = "sendnn"))]
use scratchy_target_spyre::manifest::Manifest;
// `--target sendnn` swaps the KTIR emulator runner for the on-silicon sendnn
// runner; the bundle type + session type are cfg-selected, everything else
// (weight load, dynamic sources, KV loop, sampling) is shared.
#[cfg(not(feature = "sendnn"))]
use scratchy_target_spyre::manifest::KtirBundle;
#[cfg(not(feature = "sendnn"))]
use scratchy_target_spyre::runner::SpyreSession;
#[cfg(feature = "sendnn")]
use scratchy_target_spyre::sdsc_runner::SuperDscSession;

use crate::spyre_pool::*;

/// Per-layer wiring the decode loop needs: the prefix-KV cache source tensors
/// it fills, and the op-output tensors holding THIS position's roped-K / V to
/// lift back into the cache.
pub(crate) struct LayerWiring {
    /// Source tensor id of `prefix_k[layer]` (filled with the KV cache).
    pub(crate) prefix_k_src: usize,
    #[cfg(not(feature = "sendnn"))]
    pub(crate) prefix_v_src: usize,
    /// Op-output tensor id of the new token's roped K (AttnDecode's seg-1 K).
    pub(crate) new_k_id: usize,
    /// Op-output tensor id of the new token's V (AttnDecode's seg-1 V).
    pub(crate) new_v_id: usize,
}

/// Per-phase metadata for one program (decode m=1 or prefill m=M) inside the
/// shared [`SpyreSession`]. Decode and prefill share weight tensor ids (same
/// graph, different m), so the weights live ONCE in the session; this carries
/// only the per-phase wiring + the program index to address `run_step`.
pub(crate) struct BundleMeta {
    /// Program index in the shared session (see [`SpyreSession::run_step`]).
    #[cfg(not(feature = "sendnn"))]
    pub(crate) prog: usize,
    /// Prefix capacity (mask cols) — max context length this bundle serves.
    pub(crate) capacity: usize,
    /// Baked query-row count: 1 (decode) or M (batched prefill).
    pub(crate) m_cap: usize,
    /// Runtime source ids (shared id space across phases).
    pub(crate) embed_src: usize,
    /// cos / sin sources as `(id, full width)` — GQA gives >1 width.
    pub(crate) cos_srcs: Vec<(usize, usize)>,
    pub(crate) sin_srcs: Vec<(usize, usize)>,
    pub(crate) result_id: usize,
    /// One per layer, in layer order.
    pub(crate) layers: Vec<LayerWiring>,
    /// Tensors read back from each forward, in a stable order: `result_id`
    /// then every layer's `new_k_id`, `new_v_id`.
    #[cfg(not(feature = "sendnn"))]
    pub(crate) output_ids: Vec<usize>,
    /// What the bake placed — asked ONCE, in the target crate, off the generated layout.
    pub(crate) facts: scratchy_target_spyre::wiring::BakeFacts,
    /// ⭐ THE GENERATED scratchy_target_spyre::wiring::IRING THIS BUNDLE scratchy_target_spyre::wiring::AS BAKED scratchy_target_spyre::wiring::ITH. Everything above it is a copy taken
    /// out of this static on the way through `Parsed`; the forward tape is read straight off it
    /// (`Wiring::forward_shape`) so the shape a forward plays cannot be a second opinion about
    /// what the bake decided.
    pub(crate) wiring: &'static scratchy_target_spyre::wiring::Wiring,
}

/// The BAKED logits tensor of a bundle: `[rows, width]`, STICK-BLOCKED on the vocab axis.
///
/// Two things about it are not what a reader assumes, and both are silent when `rows == 1`:
///
///  * `width` is the PADDED placement width, not the model's vocab. The emitter pads the vocab axis
///    to a whole number of sticks per core — granite's 49155 is placed as 51200 — so a row spans
///    `width` elements of which only the leading `vocab` are real.
///  * A row is NOT contiguous. [`scratchy_subtile::sdsc_abstract::StickLayout::dev_off`] places
///    element `(r, c)` at `(c/64)·(rows·64) + r·64 + (c%64)`: the vocab axis is cut into 64-wide
///    blocks and all `rows` rows of a block sit together.
///
/// At `rows == 1` that formula collapses to `c`, which is why the one-request path can slice
/// `..vocab` straight off the buffer — and why the same slice at `r * vocab` for `r > 0` reads
/// mostly request 0's logits instead of request `r`'s. Fluent, wrong, and per-request.
pub(crate) struct LogitsGeom {
    pub(crate) rows: usize,
    pub(crate) width: usize,
}

impl LogitsGeom {
    /// From the result tensor's baked `bundle_layout.json` placement (BYTES) and the rung's row
    /// count — the BAKED width, because the tail runs unfolded at the rung's `m` whatever is live.
    /// `None` when the placement is not a whole number of fp16 rows, or is narrower than the
    /// vocab it must hold — either means this is not the tensor we think it is, and gathering from
    /// it would read neighbouring placements.
    pub(crate) fn from_placement(
        size_bytes: u64,
        rung: scratchy_subtile::sdsc_abstract::RungWidth,
        vocab: usize,
    ) -> Option<LogitsGeom> {
        let elems = (size_bytes / 2) as usize;
        let rows = rung.count();
        if size_bytes % 2 != 0 || elems % rows != 0 {
            return None;
        }
        let width = elems / rows;
        (width >= vocab).then_some(LogitsGeom { rows, width })
    }

    /// Element count of the whole placement — what to ask [`SuperDscSession::read_tensor`] for.
    pub(crate) fn len(&self) -> usize {
        self.rows * self.width
    }

    /// De-interleave `flat` (the whole `[rows, width]` placement) into `rows` CONTIGUOUS `vocab`-wide
    /// rows, so a caller can index request `r` at `r * vocab`. The gather runs through the SSOT
    /// device-layout formula, never a copy of it, so it cannot drift from the addresses the emitter
    /// baked.
    pub(crate) fn rows_contiguous(&self, flat: &[f32], vocab: usize) -> Option<Vec<f32>> {
        let lay = scratchy_subtile::sdsc_abstract::StickLayout::row_blocked(self.rows, self.width);
        let mut out = Vec::with_capacity(self.rows * vocab);
        for r in 0..self.rows {
            for c in 0..vocab {
                out.push(*flat.get(lay.dev_off(r, c))?);
            }
        }
        Some(out)
    }
}

/// What a forward should hand back from its lm-head tail. An enum rather than a `bool` because the
/// per-request case CANNOT be served without the tensor's geometry — see [`LogitsGeom`] — and a bare
/// `want_logits: true` gave no place to put it, which is exactly how the batched path came to slice
/// one row's worth and hand it to eight requests.
/// One rung of the batched-decode ladder: a bundle baked for exactly `seqs` requests, aliasing the
/// one-request decode session's weights and page pool.
// ⛔ `sendnn`-ONLY, restored from 8bd5c755b. My branch was cut BEFORE that commit added
// these gates, and resolving the rebase conflict in favour of the split took the whole
// file — which discarded them. They are what makes the KTIR-only build compile.
#[cfg(feature = "sendnn")]
pub(crate) struct DecodeRung {
    /// The row count this rung's bundle was BAKED at — every launch on it binds exactly this many
    /// rows, live or padding, and every mask/fold/logits extent below derives from it.
    pub(crate) seqs: scratchy_subtile::sdsc_abstract::RungWidth,
    /// ⭐ THE COLUMNS ONE FOLD PASS OF **THIS RUNG'S BODY** SWEEPS (`active_cap`), plumbed from the
    /// manifest. The mask-coverage guard below used `PAGE_SLOTS` because this was not available, which
    /// made it the LOOSE bound: the fold covers `swept * pages`, not `PAGE_SLOTS * pages`, and a row
    /// deeper than the covered window answers from its prompt and its newest token alone.
    pub(crate) swept: scratchy_target_spyre::manifest::SweptCols,
    /// ⭐ THE BLOCKS THIS RUNG'S OWN PMASK BAKED ROOM FOR, read from its `bundle_layout.json`. `None` when
    /// the layout could not be read — the launch then keeps the pre-existing behaviour rather than
    /// refusing on a number it does not have. Compared against `MaskBlocks::of(grid)` at launch: the
    /// capacity used to be computed for a `[mask-cap]` eprintln and thrown away, so overrunning it was
    /// silent, and an unstaged additive-mask byte reads as ZERO = VALID.
    pub(crate) mask_cap: Option<scratchy_subtile::sdsc_abstract::BakedMaskBlocks>,
    /// ⭐ THE FOLD-ROW REGIME THIS RUNG'S BUNDLE WAS BAKED WITH, read from its own `op_manifest.json`
    /// (`manifest_fold_row_regime`). The launch's intermediate-segment rebase stride
    /// (`set_int_stride`) DERIVES from this: a whole-batch bundle gets stride 0 because its every
    /// fold pass already sweeps every shared-buffer row, a per-request bundle gets one request's
    /// row-block distance. It used to be a worker-side `const` hand-mirroring the emitter's choice
    /// across two crates with no tie.
    pub(crate) fold_rows: scratchy_subtile::sdsc_abstract::FoldRowRegime,
    pub(crate) sess: SuperDscSession,
    /// THIS rung's logits geometry. It is per-rung, not per-model: the lm-head tail runs unfolded at
    /// `m = seqs`, so the placement's row count — and therefore every gather address — differs
    /// between rungs.
    pub(crate) logits: LogitsGeom,
}

pub(crate) enum LogitsWanted<'a> {
    /// Nothing — every prompt chunk but the last, which skips the whole-segment D2H.
    None,
    /// The FOLDED `[1, vocab]` tail: one row at placement offset 0, whatever `m` the bundle ran at.
    /// This is a prompt's tail, where only the last row's logits are ever sampled.
    LastRow,
    /// One row PER REQUEST: the tail ran unfolded at `m = geom.rows`, so every row is a different
    /// request's next-token distribution and all of them are sampled.
    PerRequest(&'a LogitsGeom),
}

/// The sendnn run backend selected from the baked bundle shape. DEFAULT-mode
/// (`SCRATCHY_SENDNN_MODE=default`) bakes ONE group whose `prefill` and `decode`
/// slots hold the SAME single static-SDPA graph (host-grown KV, no resident paged
/// The sendnn session type. ONE variant: the dxp-compiled SuperDSC bundle is the only thing that
/// runs on silicon. The sengraph families (offline_decoder `GroupedSession`, the static-paged and
/// default-mode single-graph sessions) are GONE — they were the pre-SuperDSC bring-up path and the
/// router sent nothing to them.
#[cfg(feature = "sendnn")]
pub(crate) enum SendnnSession {
    /// SUPERDSC (the typed-Rust SuperDSC OpSpec emitter path). The lowering owns the
    /// 32-core work-division and emits a dxp bundle (a DIRECTORY of per-supernode dxp
    /// blobs + plan), NOT a single sengraph JSON the DeepTools toolchain compiles. The
    /// codegen stamps the bundle's decode-slot `graph_json` with a `SUPERDSC_BUNDLE:<fp>`
    /// sentinel (NOT a real graph); [`is_superdsc_bundle`] detects it and `load_inner`
    /// recovers the compiled device code via `bundle_code::bundle(fp)` — the bytes are
    /// in this binary; nothing is read from a build-time cache.
    /// ONE session drives every forward (single-stream, chunk size 1): the KV is
    /// DEVICE-RESIDENT (written + attended in the dxp run-path), so the host does NOT
    /// thread grown-KV back. `run_step(n_new, start, &acts)` mirrors the incremental
    /// contract: write `n_new` tokens at absolute slots `[start..start+n_new)` and
    /// return the last new position's logits.
    ///
    /// BATCHED PREFILL (prefill-all-but-last): the bundle carries the m=1 `decode` session
    /// AND an optional m=N `prefill` session (a SEPARATE SuperDscSession baked at
    /// KTIR_PREFILL_LEN, lm_head skipped). The prompt runs through `prefill` in ONE forward
    /// (populating its resident seg2 KV on-card); the worker then copies that seg2 KV into
    /// `decode`'s seg2 (`read_seg`/`write_seg`, byte-identical layout) and decodes the last
    /// prompt token via `decode` → first-gen logits. `prefill` is `None` when the prefill
    /// bundle didn't bake (its slot fell back to the decode fp) → sequential mq=1 prefill.
    SuperDsc(SuperDscBundle),
}

/// The SUPERDSC session pair: the m=1 `decode` reroll program + an optional m=N `prefill`
/// reroll program (prefill-all-but-last). They are SEPARATE sessions (the prefill's seg0
/// activations / seg3 intermediates are m-larger, so they cannot share all segments); their
/// seg2 KV layout is byte-identical, so the worker hands the prompt's KV from `prefill` to
/// `decode` via a raw seg2 copy after the batched-prefill forward.
#[cfg(feature = "sendnn")]
pub(crate) struct SuperDscBundle {
    pub(crate) decode: SuperDscSession,
    /// PREFILL WIDTH LADDER — ASCENDING by baked query-row count `mq`; the last entry is the widest.
    /// EMPTY ⇒ no batched-prefill bundle at all (multi-token prompts fall back to the sequential
    /// mq=1 decode path).
    ///
    /// Each rung is baked for EXACTLY its `mq` query rows (the tensor shapes are concrete), so
    /// `run_step` must be called with exactly that m: the real prompt tokens fill rows `[0..real)`
    /// and rows `[real..mq)` are PAD — future positions the causal mask never lets a real row attend,
    /// whose KV slots sit beyond `seq_pos` so decode never reads them. Picking the SMALLEST rung that
    /// still holds the chunk is what keeps that padding cheap: per-chunk cost is `fixed + slope·mq`
    /// with the fixed term dominant, so one width is wrong both ways (too narrow re-pays the fixed
    /// term per extra chunk, too wide computes empty rows).
    ///
    /// A rung needs its OWN session because `mq` sizes the prefix (embed rows), the suffix and every
    /// seg0/seg3 activation. Rungs DO share the resident weights (seg1) and KV (seg2) by alias, so the
    /// ladder costs program + activation memory, not another copy of the model. All rungs carry the
    /// SAME tensor ids (verified: identical 1467-key placement set across widths 15/31/47/63 — only
    /// the 611 activation SIZES scale with `mq`), so ONE prefill `BundleMeta` binds any of them, and
    /// `run_prefill_batch` rebinds every const on each call — so switching rungs between chunks is free.
    pub(crate) prefill_rungs: Vec<(usize, SuperDscSession)>,
    /// DECODE BATCH LADDER `(seqs, session)`, ascending — one session per baked request count.
    ///
    /// A decode step runs one token of each RUNNING request. Without this the worker runs one
    /// forward per request and streams the whole weight set once per token per request, which is the
    /// entire cost of a decode step paid again for each. A step picks the SMALLEST rung that holds
    /// the live count and pads the rest.
    ///
    /// Rungs alias the decode session's weights (seg1) and page pool (seg2), so the ladder costs
    /// programs and activations, not another copy of the model — exactly as the prefill ladder does.
    /// EMPTY ⇒ no ladder; every request runs its own forward, as before.
    pub(crate) decode_rungs: Vec<DecodeRung>,
    /// HOW MANY STRIPES THE KV POOL IS CUT INTO — decided once when the pool is sized, and the same
    /// value the pool was sized BY. Stored rather than derived so that the number multiplied into the
    /// pool's byte budget and the number divided into its pages cannot be two different numbers; see
    /// [`PoolRows`] for the pair they used to be.
    pub(crate) pool_rows: PoolRows,
    // ⛔⛔⛔ THERE IS NO `prefill_pending`, AND ITS ABSENCE IS THE POINT.
    //
    // It held "ladder rungs the bake produced that have NOT been built yet", so a rung could be built the
    // first time a chunk actually needed it — its doc justified that with a measurement: "building all 21
    // costs 6.05s of startup (302 ms each)". The field was constructed `Vec::new()` at the ONE site that
    // makes this struct and nothing ever pushed to it, so the lazy build could not happen and the 6.05 s
    // was being paid anyway: `prefill ladder: 20 rung(s) ready in 6.19s wall — built in parallel`.
    //
    // ⭐ A DEAD PATH IS A LIE ABOUT WHAT THE PROGRAM CAN DO, and this one cost real time: creating a session
    // mid-run is exactly what `fxa_stream_create` aborts on under concurrency, so when that abort appeared
    // the dead build was the first thing I blamed — for an hour, before the log's own zero
    // `PREFILL_PATH` lines showed the abort was at LOAD. Deleting it removes a state the program cannot
    // reach and a wrong answer a reader can reach.
    //
    // If deferred rung building is ever wanted again, it must be built where the OTHER sessions are (at
    // load, serialized — see `sdk_abi::Stream::create`), not from inside a forward.
    /// `(mq, session)` for the ONE prefill bundle baked with a full resident-prefix sweep — the bundle
    /// a CONTINUATION chunk (`start > 0`) must run on, because it has resident KV to attend.
    ///
    /// Every entry in [`prefill_rungs`](Self::prefill_rungs) is baked PREFIX-FREE: at `start == 0`
    /// there is nothing resident, so those 4 prefix blocks would be masked out in full and their 308
    /// ops/layer (43% of the body, ~12,320 launches per forward) would compute nothing.
    ///
    /// `None` ⇒ no prefix-capable bundle baked, so a continuation chunk cannot run and is refused
    /// rather than silently attending an unswept cache.
    pub(crate) prefill_prefix: Option<(usize, SuperDscSession)>,
    /// PAGED: how many pages the pool has. The pool is SHARED — every session above addresses the
    /// same seg2 — so a page belongs to at most one request at a time. This is what replaces "one
    /// cache, one sequence": concurrent requests get DISJOINT pages instead of both writing from
    /// slot 0 and silently overwriting each other.
    ///
    /// Which pages are FREE is derived from the live requests, never stored: see `alloc_pages`.
    pub(crate) pool_pages: usize,
}

#[cfg(feature = "sendnn")]
impl SuperDscBundle {
    /// The widest baked prefill width = the largest chunk ONE batched forward can take. 0 = no ladder.
    pub(crate) fn prefill_top_m(&self) -> usize {
        self.prefill_rungs.last().map(|(m, _)| *m).unwrap_or(0)
    }

    /// Index of the SMALLEST decode rung that holds `live` requests, or `None` when the ladder is
    /// empty or `live` exceeds the widest rung — the caller then falls back to one forward per
    /// request, which is slower and never wrong.
    ///
    /// 🛑 THE LIVE COUNT IS THE ONLY INPUT, and the launch must therefore be laid out from the live
    /// count too. `SlotMap` hands out slots densely for exactly this reason: any layout that reads a
    /// request's KV ROW instead makes the width needed depend on which rows are held, and then this
    /// function — which cannot see them — picks a rung too narrow to express the batch. Widening the
    /// choice to cover the highest row held would work and is worse: one request on row 31 would drag
    /// the whole batch onto the 32-row rung and pay 32 rows of launch for it.
    pub(crate) fn decode_rung_for(
        &self,
        live: scratchy_subtile::sdsc_abstract::LiveRows,
    ) -> Option<usize> {
        self.decode_rungs.iter().position(|r| r.seqs.holds(live))
    }

    /// HOW THE POOL IS DIVIDED: one equal, consecutive run of pages per KV row.
    ///
    /// Cut for the WIDEST baked decode rung, not for the requests currently live, so a run does not
    /// move when the batch narrows or widens. That stability is the whole property being bought — see
    /// [`ReqState::row`].
    ///
    /// `None` when nothing is paged or the pool cannot give every row a page. The latter is a real
    /// condition to handle (admit fewer, or raise `SUPERDSC_POOL_PAGES`), never something to round
    /// down: a zero-page run aliases row 0's KV.
    ///
    /// 🛑 THE STRIPE COUNT IS READ, NOT RECOMPUTED. This used to be
    /// `decode_rungs.iter().map(|r| r.seqs).max()` — the widest rung that survived baking — while the
    /// pool's SIZE was computed from the const `WIDEST_BATCH_RUNG`. Two integers of the same type, from
    /// different sources, one multiplied into the pool and the other divided into it. They happened to
    /// agree whenever every rung baked, so the disagreement was invisible exactly until a rung dropped
    /// out. [`PoolRows`] is now the single value both sides use, decided once at load.
    pub(crate) fn kv_split(&self) -> Option<PoolSplit> {
        if !self.decode.is_paged() {
            return None;
        }
        PagedKvPool::split_pool(self.pool_pages as u32, self.pool_rows)
    }
}

// ⛔ AND NO `rows_held`: with no rows to assign there is no held-set to derive. It collected every KV row a
// live request held, as the input to `free_row`.

// ⛔⛔⛔ THERE IS NO `pages_held`, AND ITS ABSENCE IS THE LOCK.
//
// It collected "every pool page a live request holds" so that `ensure_pages` could search for one no
// live request held and push it onto the request's map. That made the WORKER a second allocator of the
// pool, alongside the scheduler — which owns the same pool, refcounts its blocks, and keeps the prefix
// cache alive BY those refcounts. The two agreed only because the worker ignored the ids the scheduler
// handed it, and that is the whole reason prefix caching could not be turned on: a cache HIT is the
// scheduler saying "these blocks already hold that prefix", and a worker holding pages of its own
// choosing holds the prefix nowhere.
//
// The pages a request holds now come from `NewRequestData::block_ids` / `CachedRequestData::new_block_ids`
// through [`BlockTable`], whose only constructor takes host ids. With no free-list search left there is
// nothing to derive a held-set FOR, and a page two requests share stops being unrepresentable and starts
// being the ordinary case — which is what a block table has always been on metal and cuda.

// ⛔⛔⛔ THERE IS NO `free_row` EITHER, AND ITS ABSENCE IS THE SAME LOCK ONE LEVEL UP.
//
// It handed out the lowest KV row no live request held. A row was a per-request reservation of the pages
// that backed a batched write past the request's own keys — the LAST request concept below the host, and the
// reason this file had a notion of request identity at all. The scheduler allocates that page now (`own pages
// + 1`, `KvSlotSpan::blocks_this_step`), so there is nothing to reserve, nothing to hand out, nothing to
// release, and no "every row is held" refusal capping concurrency. Verified before deleting: 0 hole pages
// across 30 gate runs on granite-3.1-2b (hd=64) and granite-3.1-8b (hd=128).

/// Model-level state shared by both bundles, borrowed alongside a `&mut Bundle`.
pub(crate) struct Shared<'a> {
    pub(crate) embed_tokens: &'a [f32],
    pub(crate) hidden: usize,
    pub(crate) head_dim: usize,
    pub(crate) kv_dim: usize,
    pub(crate) vocab: usize,
    pub(crate) rope_theta: f32,
}

/// Everything [`Worker::load_model`] resolves: model-level shared state + the
/// per-phase bundles. Generation (one token/step) runs through the cheap m=1
/// `decode` bundle; the prompt runs through the batched m=M `prefill` bundle.
pub(crate) struct Loaded {
    /// On-disk model dir (weights + `config.json`); surfaced to `create_worker`.
    pub(crate) model_dir: PathBuf,
    /// Parsed HF `config.json` — the engine needs it for scheduling/cache sizing.
    pub(crate) hf_config: HfModelConfig,
    /// `embed_tokens.weight` `[vocab, hidden]` kept for the per-token gather.
    pub(crate) embed_tokens: Vec<f32>,
    pub(crate) hidden: usize,
    pub(crate) head_dim: usize,
    pub(crate) kv_dim: usize,
    pub(crate) vocab: usize,
    pub(crate) rope_theta: f32,
    /// KTIR path: ONE resident session holding both programs (weights once).
    #[cfg(not(feature = "sendnn"))]
    pub(crate) session: SpyreSession,
    /// sendnn path: the SuperDSC dxp-bundle session or the offline_decoder
    /// layer-group split — selected from the baked bundle shape.
    #[cfg(feature = "sendnn")]
    pub(crate) session: SendnnSession,
    /// m=1 decode program metas, one per static cap bucket, ASCENDING by
    /// capacity. The generation loop runs the smallest bucket whose capacity
    /// covers the current position (decode is O(capacity)); never empty. The
    /// sendnn path bakes a single bucket, so its ladder is the full-cap entry.
    pub(crate) decode: Vec<BundleMeta>,
    /// m=M batched-prefill program meta, if emitted; else prefill falls back to
    /// the decode program (sequential, one token per forward).
    pub(crate) prefill: Option<BundleMeta>,
    /// The prefill program meta PRESERVED for the batched-prefill path. `model.prefill` is nulled for
    /// SuperDsc so the PER-FORWARD binding uses the decode graph (the prefill manifest has different
    /// tensor ids); but `run_prefill_batch` binds this meta to the DISTINCT prefill SESSION (where the ids
    /// match), so it is kept here. Some only for SuperDsc with a baked prefill bundle; None otherwise.
    pub(crate) batched_prefill: Option<BundleMeta>,
    /// THE KV BYTE BUDGET THE POOL WAS ACTUALLY SIZED AGAINST — the card's real capacity ×
    /// `--gpu-memory-utilization` − what this bundle's non-KV segments already reserve.
    ///
    /// ⛔⛔ ONE QUANTITY, ONE PLACE, BECAUSE TWO SPELLINGS OF IT DISAGREED BY 8 GiB. The pool is sized from
    /// this while sizing happens; the ENGINE separately sizes the scheduler's block allocator from
    /// [`Worker::determine_available_memory`], which returned a flat `8 GiB` constant. The engine takes
    /// `num_gpu_blocks = min(computed_from_that, kv_cache_num_blocks_override)`, so a stale 8 GiB would cap
    /// a larger real pool at ~273 pages and silently shrink every request's context. Computed once here,
    /// reported from here.
    ///
    /// `None` on the paths that never size a paged pool (KTIR emulator, non-paged sessions).
    pub(crate) kv_budget_bytes: Option<u64>,
}

/// Per-request generation state: token history + the growing host KV cache.
pub(crate) struct ReqState {
    // ⛔⛔⛔ THE TOKENS ARE NOT HERE, AND NEITHER IS THE PROMPT LENGTH.
    //
    // They were `tokens: Vec<u32>` (prompt ++ generated) and `prompt_len: usize`, and they now live in
    // `InputBatch` — the engine's backend-neutral per-request store, the same place this worker's block
    // table went. `InputBatch::prompt_len` is DERIVED from the prompt it stores, so the pair that could
    // disagree is one value now. The metal worker keeps the identical pair privately
    // (`token_buffers` + `prompt_lengths`, the second assigned `token_buffers[id].len()` at its only
    // insert); it can drop both onto the same accessors.
    //
    // ⭐ WHAT IS LEFT HERE IS WHAT NO OTHER BACKEND HAS: a KV history of RUNS rather than a length,
    // because a batched write appends every row of a batch at ONE slot and a shorter row carries a
    // masked hole. That is the only reason this struct still exists.
    // ⛔ AND `n_computed` IS NOT HERE EITHER — it is `InputBatch::tokens_in_pool`.
    //
    // "Positions already forwarded" is this request's TOKEN count, hence its RoPE position, and the engine's
    // store has held exactly that number for every backend since before this worker used it — seeded at
    // admission from the scheduler's `num_computed_tokens` and advanced by `commit_step` on cuda/metal. This
    // path does not call `commit_step` (it chunks its own prefill and knows how far it got only after the
    // loop), so it writes through `set_tokens_in_pool` instead. One store, one writer per step.
    //
    // ⛔ IT IS NOT WHERE THE NEXT KEY IS WRITTEN. Those were the same number for as long as every request
    // appended at its own position, which is exactly the arrangement that costs one launch per request.
    // `kv_hist` below is the SLOTS, and it stays here because no other backend has one.
    /// WHICH KV SLOTS THIS REQUEST'S KEYS OCCUPY. Its own `[0, prompt)` until it shares a step with a
    /// longer request, after which it holds a masked hole and its slot count exceeds `n_computed`.
    pub(crate) kv_hist: scratchy_subtile::sdsc_abstract::KvHistory,
    /// ⛔ A HOST MIRROR OF THE KV — and it does not exist on the sendnn path at all.
    ///
    /// Per-layer, row-major `[n_computed, kv_dim]`, grown one row per step by the KTIR EMULATOR's
    /// `forward_chunk` (`#[cfg(not(feature = "sendnn"))]`), which threads the prefix KV from the host. On
    /// the on-silicon paged path the KV is resident on card and addressed by [`PagedKv`] + the host block
    /// table, so this was allocated per request (`vec![Vec::new(); nlayers]`) and never read — a dead
    /// mirror of the thing `d6906140` removed. `cfg`-ed out rather than documented as unused: an empty
    /// mirror is one `extend_from_slice` away from being a real one again.
    #[cfg(not(feature = "sendnn"))]
    pub(crate) kv_k: Vec<Vec<f32>>,
    #[cfg(not(feature = "sendnn"))]
    pub(crate) kv_v: Vec<Vec<f32>>,
    // ⛔⛔⛔ THERE IS NO `kv: Option<PagedKv>` ANY MORE, AND THAT IS THE END OF REQUEST IDENTITY BELOW THE
    // HOST. It held the request's POOL ROW — the pages that backed its batched write past its own keys. The
    // scheduler allocates that page now, so a request's on-card KV is fully described by two things this
    // struct already has: `kv_hist` (which slots its keys occupy) and the host's block list in `InputBatch`.
    // The map one launch binds is derived from those on every bind and stored nowhere.
}

#[allow(dead_code)]
impl ReqState {
    /// ⭐ THE MAP ONE LAUNCH BINDS, derived here and stored nowhere: the host's blocks (from
    /// [`InputBatch`], via [`PageMapCtx`]) placed at the logical pages this request holds keys in, and the
    /// fully-masked pages aliased to the pool's ONE scratch page.
    ///
    /// ⛔ IT TOOK A `KvRow` TOO, and that argument was the last per-request address below the host: the write
    /// page came from a run of pool pages reserved for that row. The scheduler allocates the write page now,
    /// so this is a function of the request's HISTORY and the HOST'S LIST — nothing else.
    #[cfg(feature = "sendnn")]
    pub(crate) fn page_map(&self, ctx: PageMapCtx<'_>, want: RowPages) -> Option<BlockTable> {
        BlockTable::map_row(&self.kv_hist, ctx.host, want, ctx.part)
    }

    /// ⭐ WHAT THE SCHEDULER MUST KNOW ABOUT THIS REQUEST'S KV, both halves at once — the slots to
    /// allocate blocks for, and the leading tokens a token-indexed prefix cache may claim.
    ///
    /// Reported every step (`ModelRunnerOutput::kv_extent`) because both halves are facts about the
    /// history this worker just wrote, and neither is derivable from the token count:
    ///
    /// * the SPAN exceeds the token count for every request that shared a batched step, because a batch
    ///   appends all its rows at ONE slot and the shorter rows carry a masked hole;
    /// * the CACHEABLE prefix stops at the first hole, because past it token `t` is no longer at slot
    ///   `t`, and a host block hashed over it would hand a later request the keys of the wrong tokens.
    ///
    /// ⛔ THE SCHEDULER MUST NOT DERIVE EITHER ONE ITSELF. It has the batch composition and could run
    /// the same recurrence, and that is exactly the two-sources-for-one-quantity shape that has produced
    /// every defect in this path. The history is here; the report is a read of it.
    ///
    /// 🛑 THE SPAN IS THE BATCH'S, NOT THIS REQUEST'S OWN END, and passing it in is what says so. A
    /// batched step writes EVERY row at [`BatchSlot::of`] over the LIVE set, so the slot a short request
    /// is next written at is the batch's maximum and not its own — report its own end and the scheduler
    /// allocates blocks that stop short of the page that write lands in, on exactly the step the request
    /// joins a deeper batch. That is a per-POOL fact reported per request because allocation is per
    /// request; every live request gets the same span, and each keeps its own cacheable prefix.
    ///
    /// ⛔⛔ AND THE BATCH'S MAXIMUM MUST BE ITS FUTURE ONE, NOT THE ONE THIS STEP ENDED WITH. MEASURED
    /// 2026-08-13, ragged 6-probe batch, caching ON *and* OFF:
    /// `request needs 885 slot(s) = 4 page(s), but the scheduler has allocated only 1 block(s)`.
    /// A report is read by the scheduler BEFORE the next step and acted on for it, so anything that can
    /// move the maximum WITHIN a step outruns it — and a prefill chunk does exactly that: the prefills of
    /// a step record before its decode batch computes `BatchSlot::of`, so a request that was 116 slots
    /// deep at the report and 884 after its chunk drags every decoding row's write slot with it, and the
    /// rows that were allocated for 116 are one page short mid-forward.
    /// ⇒ THE BOUND IS `max(end, prompt_len)` OVER THE LIVE SET ([`Self::kv_reach`]): a prefilling
    /// request's end can only walk up to its own prompt, which is known at ADMISSION and cannot move. No
    /// step lag, no arbitrary headroom, and no dependence on the chunk schedule — which is the same
    /// property `PoolRows` and `ChunkKvExtent` exist for.
    pub(crate) fn kv_extent(
        &self,
        pool_reach: SlotCount,
        prompt: SlotCount,
    ) -> scratchy_core_common::KvExtent {
        scratchy_core_common::KvExtent::new(
            scratchy_core_common::KvSlotSpan::new(
                pool_reach.get().max(self.kv_reach(prompt).get()),
            ),
            scratchy_core_common::CacheableTokens::new(self.kv_hist.cacheable_prefix().get()),
        )
    }

    /// ⭐ THE DEEPEST SLOT THIS REQUEST WILL EVER OCCUPY ON ITS OWN ACCOUNT — its history's end, or its
    /// PROMPT, whichever is further.
    ///
    /// The prompt is the part that cannot move: it is fixed at admission, and a prefilling request's end
    /// walks up to exactly it however the chunks fall. Taking the maximum over the live set therefore
    /// gives a batch write-slot bound that no in-step prefill can outrun — see [`Self::kv_extent`] for
    /// the measurement that made this necessary.
    /// ⛔ THE PROMPT IS PASSED IN, NOT HELD. It lives in `InputBatch` (one store, and the length derived
    /// from the tokens), so the caller reads it there and hands it over as a [`SlotCount`] — the same
    /// type the history's end has, because the maximum of the two is the whole point of this function.
    pub(crate) fn kv_reach(&self, prompt: SlotCount) -> SlotCount {
        SlotCount::new(self.kv_hist.end().get().max(prompt.get()))
    }

    /// ⭐ A PREEMPTED REQUEST HAS NO KV LEFT — reset it to what the scheduler says it now holds.
    ///
    /// The scheduler frees a preempted request's blocks and re-schedules it with `resumed_req_ids` and a
    /// fresh `num_computed_tokens` (0, or a prefix-cache hit). Its old pages are another request's keys
    /// by then, so keeping the old history would have it attend them: `contains` would answer true on
    /// slots whose page it no longer owns. Nothing in this worker used to notice a resume at all — the
    /// history simply carried on — which was survivable only because a pool with no prefix caching and a
    /// worker-side free list rarely produced the block pressure that preempts.
    /// ⛔ THE TOKEN COUNT IS THE CALLER'S TO RESET, in `InputBatch` — this resets only the SLOTS, which
    /// is the half no other backend has. Both must move together on a resume, and the call site does them
    /// adjacently for that reason.
    pub(crate) fn reset_for_resume(&mut self, computed: usize) {
        self.kv_hist = scratchy_subtile::sdsc_abstract::KvHistory::contiguous(computed);
    }

    /// HOW MANY KV SLOTS THIS REQUEST SPANS — one past its last key, which is what the pool has to hold
    /// pages for. `>= n_computed`, and strictly greater once it has shared a step with a longer request.
    pub(crate) fn kv_span(&self) -> usize {
        // The one place a slot becomes a plain count — for messages and for the token arithmetic the
        // scheduler speaks in. Everything that DIVIDES it by a page takes [`Self::kv_span_slots`].
        self.kv_hist.end().get() as usize
    }

    /// The same span as a [`SlotCount`], which is what [`install_host_blocks`] takes: a page count may
    /// only be derived from this, and only by `RowPages::holding`.
    pub(crate) fn kv_span_slots(&self, writing: u32) -> SlotCount {
        SlotCount::new(self.kv_hist.end().get().saturating_add(writing))
    }

    /// Record the `n` slots a launch just wrote for this request, at the slot the launch used.
    ///
    /// The ONLY way the history moves, and it takes a [`BatchSlot`] rather than a number so the slot it
    /// records is provably the slot the forward was bound at. The mask of the NEXT step is built from
    /// this, so a recorded slot the card did not write, or a written slot not recorded, is a request
    /// attending keys that are not its own.
    pub(crate) fn record_kv(
        &mut self,
        at: scratchy_subtile::sdsc_abstract::BatchSlot,
        n: scratchy_subtile::sdsc_abstract::SlotCount,
    ) {
        self.kv_hist.record(at, n);
    }

    // ⛔⛔⛔ THERE IS NO `record_kv_solo(n)`, AND ITS ABSENCE IS THE LOCK.
    //
    // It took a COUNT and DERIVED the slot from this request's own end. That is exactly the shape the
    // 2026-08-12 defect needed: a prefill chunk that had STEPPED BACK to keep its padded window inside
    // one page wrote at `placed`, and this function recorded the same count at the UN-stepped-back end,
    // marking `back` slots valid that no chunk wrote (450-token prompt: history 514 vs position 450).
    //
    // With it gone, no API takes a count and guesses where it goes. A prefill chunk records through
    // `record_chunk`, whose `ChunkWrite` can only be minted by `ChunkStart::wrote` — the start the
    // placement function actually chose. A decode step must name its `BatchSlot` at the call site, so
    // "the slot came from somewhere other than the write" is visible in the caller rather than hidden
    // behind a convenience.

    /// Record the run a prefill chunk WROTE, at the slot [`PagedKvPool::chunk_write_start`] placed it.
    pub(crate) fn record_chunk(&mut self, w: scratchy_subtile::sdsc_abstract::ChunkWrite) {
        self.kv_hist.record_chunk(w);
    }
}
