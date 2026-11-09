// SPDX-License-Identifier: Apache-2.0
//! THE PAGED STEP'S HOST STAGING — spyre-inference's own attention operands one forward of a paged
//! bundle reads besides the pools ([`PagedStep`]), built from a request's KV blocks and the step's
//! positions as their builder builds them (`SpyreAttentionMetadataBuilder.build`,
//! `_build_query_row_tables`, `SpyreHeadMajorAttentionImpl.kv_write_index` / `_kv_row_pool`,
//! spyre-inference 297fc48):
//!
//! * the PAGE INDEX TABLE `[walk, INT32_ELEMS_PER_STICK]` int32 — `page_index_tables_cpu`:
//!   `table[:, 0] = block_table[s, :walk]`, the rest of each row zero. Blocks past the request's own
//!   are the block table's own zero padding; their tiles are fully masked;
//! * the MASK STACK `[walk, rows, page_slots]` — `attention_mask_stacks`: 0 where a query row may
//!   attend the key, `finfo(f16).min` where it may not — past the sequence (`kv_pos < seq_len`) and,
//!   for a prompt chunk, past the row's causal limit `context_len + q_pos`, a padded row clamped to
//!   the last real one;
//! * the KV WRITE INDEX — `kv_write_index`: one `[rows]` index per kv head, one after another, head
//!   `h` of the token at `block · page_slots + offset` at row
//!   `(block · nkvh + h) · page_slots + offset`;
//! * the QUERY ROW INDEX `[rows]` int32 — `_build_query_row_tables`: the sequence's rows of the
//!   query. A forward here is one sequence whose padded rows already repeat its last real row (the
//!   worker's pad law), so its rows are `0..rows` in order;
//! * `kv_row_pool` `[pages, nkvh, 1]` int32 — `_kv_row_pool`: every page's folded-cache rows
//!   `page · nkvh + kv`, read by the decode kernel alone.
//!
//! A DECODE BATCH — one token of each of several sequences, `batched_decode_head_major` — reads its
//! own operands instead, built as the builder's decode-batch precomputes (`spyre_attn.py:892-967`)
//! in the tiled walk's layout (`_mirror_batched_decode_indices`): [`batch_sources`].
//!
//! ONE statement with two readers — the worker every step, and the bake's numeric gate once per
//! gated bundle — so the gate checks exactly the bytes the worker stages.

use crate::wiring::PagedStep;

/// The page walk's laws, where the bake and the worker that state a bundle's walk reach them: the
/// page size, a rung's walk ([`walk_pages`]) and the rungs a pool is walked by ([`walk_buckets`]).
pub use scratchy_triton_splice::{
    BatchChunking, PagedRows, SchedulerBlockTokens, walk_buckets, walk_pages,
};

/// `finfo(float16).min` — spyre-inference's masked additive value: finite, so a fully-masked tile's
/// row maximum stays finite and its weights vanish under the running maximum.
pub const MASKED: f32 = -65504.0;

/// spyre-inference's `_NULL_SLOT` (`attn_layer.py`): where a padded token's KV write lands — slot 0
/// of the null page, which no request holds (vLLM's block pool keeps block 0 out of every
/// allocation; [`page_of_block`] keeps it out of the scheduler's here).
pub const NULL_SLOT: usize = 0;

/// The pool page a scheduler block id names: block `b` is page `b + 1`. Page 0 is the null page
/// ([`NULL_SLOT`]) — vLLM's block pool reserves its block 0 and this scheduler's does not, so the
/// worker offers it [`PagedStep::host_blocks`] blocks and keeps page 0 back.
pub const fn page_of_block(b: u32) -> u32 {
    b + 1
}

/// ⭐⭐ EVERY POSITION A PAGED BUNDLE ADMITS STAGES A WALK — the bake's proof, over the ladder the
/// card executor reads and the selection it runs ([`RerollMeta::ladder`],
/// [`Ladder::rung_for`](crate::bundle_code::Ladder::rung_for)): for every `valid_len` up to the
/// ladder's widest sweep, the rung selected covers it, and its sweep is a walk of `step`'s pool that
/// the worker stages ([`PagedStep::on_body`], at the step's own rows). Panicked on by the bake, so a
/// ladder the executor would read as no walk — or a rung it would select that stages none — fails
/// `cargo build` rather than a card's first forward.
pub fn prove_ladder_walks(
    meta: &crate::bundle_code::RerollMeta<'_>,
    step: PagedStep,
) -> Result<(), String> {
    let ladder = meta.ladder()?;
    let top = ladder
        .rungs()
        .iter()
        .map(|r| r.active_cap.get())
        .max()
        .unwrap_or(0);
    // ⭐ ONE POSITION PER INTERVAL, NOT EVERY POSITION (#1037). Which rung `rung_for` selects, and so
    // both properties checked below, depends only on which rungs' sweeps reach `p`, and that changes
    // only where `p` crosses a sweep. So the positions `1`, every sweep `c` and `c + 1` up to the top
    // land in every interval the selection is constant on, and proving them proves `1..=top` — a few
    // positions a rung instead of granite-3.1-8b's ~105k per bundle at expansion.
    let top = u64::from(top);
    let mut probes: Vec<u64> = std::iter::once(1)
        .chain(ladder.rungs().iter().flat_map(|r| {
            let c = u64::from(r.active_cap.get());
            [c, c + 1]
        }))
        .filter(|&p| (1..=top).contains(&p))
        .collect();
    probes.sort_unstable();
    probes.dedup();
    for p in probes {
        let (at, rung) = ladder.rung_for(p);
        let swept = rung.active_cap.get();
        if !rung.active_cap.covers(p) {
            return Err(format!(
                "a step reaching {p} position(s) selects rung {at}, whose {swept}-slot sweep does not \
                 reach it"
            ));
        }
        if step.on_body(step.rows, step.paged_rows, swept).is_none() {
            return Err(format!(
                "a step reaching {p} position(s) selects rung {at}, whose {swept}-slot sweep is no \
                 walk of the {}-page pool ({} slots a page) — the worker has no walk to stage",
                step.pages, step.page_slots
            ));
        }
    }
    Ok(())
}

/// ⭐ THE POOL ONE DEVICE REGION HOLDS, in token slots. Every layer's K and V pools are one segment,
/// one allocation of at most [`scratchy_spyre_bundle::MAX_SEGMENT_BYTES`], at `kv_slot_bytes` a slot
/// (every layer's K and V row of one token). So the pool is the model's own context,
/// `context_slots`, where that fits, and otherwise the most whole pages that do — vLLM's `num_blocks`
/// bounded by memory, with `--max-model-len` checked against the pool it got
/// ([`PagedStep::capacity`]). granite-3.1-8b's 131072 slots at 163840 B are 20 GiB; its pool is
/// 104832 positions, 819 pages. `None` when not one page fits.
pub fn pool_slots(context_slots: u32, kv_slot_bytes: u64) -> Option<std::num::NonZeroU32> {
    let fit = scratchy_spyre_bundle::MAX_SEGMENT_BYTES / kv_slot_bytes.max(1);
    let fit_pages =
        u32::try_from(fit / u64::from(SchedulerBlockTokens::REFERENCE.get())).unwrap_or(u32::MAX);
    let slots = context_slots.min(fit_pages.saturating_mul(SchedulerBlockTokens::REFERENCE.get()));
    std::num::NonZeroU32::new(slots)
}

/// The rows of one forward: `real` tokens at positions `start..start + real`, in a bundle
/// [`PagedStep::rows`] wide. A row past the real ones repeats the last real token
/// ([`scratchy_subtile::sdsc_abstract::ChunkRows::row_logical_pos`]'s law), so its forward, its
/// mask and its KV write are the last real token's.
///
/// ⭐ `real` IS A COUNT, ONE OR MORE — a step of zero real rows is no step, and every reader that
/// clamped it (`pos`'s own `.max(1)`, the worker's `live_of`) was defending against a value that
/// names a broken staging, not tolerating a real one. The mint refuses it, so the count the type
/// carries is the count the staging meant.
#[derive(Clone, Copy, Debug)]
pub struct StepRows {
    pub start: usize,
    pub real: std::num::NonZeroUsize,
}

impl StepRows {
    /// Declare `real` rows from `start`. `None` for zero real rows — a staging that asks for a
    /// step of nothing, which the caller names at its own door.
    pub fn new(start: usize, real: usize) -> Option<StepRows> {
        Some(StepRows {
            start,
            real: std::num::NonZeroUsize::new(real)?,
        })
    }

    /// One row at `start` — the literal-row mint, `const` so no `expect` stands where the value
    /// is a literal. `NonZeroUsize::MIN` is 1, the count one row is.
    pub const fn one_at(start: usize) -> StepRows {
        StepRows {
            start,
            real: std::num::NonZeroUsize::MIN,
        }
    }

    /// The sequence position row `r` carries.
    pub fn pos(&self, r: usize) -> usize {
        self.start + r.min(self.real.get() - 1)
    }

    /// One past the last position this step reaches — the KV length after its write.
    pub fn end(&self) -> usize {
        self.start + self.real.get()
    }
}

/// Why a step cannot be staged against a pool.
#[derive(Debug)]
pub struct PagedStepError(pub String);

impl std::fmt::Display for PagedStepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The blocks a step reads and writes must exist: every position below `rows.end()` has a block,
/// and every block is a page of the pool. `bound` is the row count the rows are checked against —
/// the graph's rows for one sequence's whole forward, the extent's per-request width for one
/// request of a packed rung's (the two are the same number at one request).
fn check(
    step: &PagedStep,
    bound: usize,
    blocks: &[u32],
    rows: &StepRows,
) -> Result<(), PagedStepError> {
    let slots = step.page_slots as usize;
    let need = rows.end().div_ceil(slots);
    if rows.real.get() > bound {
        return Err(PagedStepError(format!(
            "a step of {} real row(s) in a {bound}-row window",
            rows.real
        )));
    }
    if blocks.len() < need {
        return Err(PagedStepError(format!(
            "positions up to {} need {need} block(s) of {slots} slots; the request holds {}",
            rows.end(),
            blocks.len()
        )));
    }
    if need > step.pages as usize {
        return Err(PagedStepError(format!(
            "positions up to {} need {need} block(s); the pool holds {} page(s)",
            rows.end(),
            step.pages
        )));
    }
    // The attention reads only the blocks its walk covers, so a step that ends past the bundle's
    // reach would attend without its last keys.
    if rows.end() > step.reach() {
        return Err(PagedStepError(format!(
            "positions up to {} are past this bundle's reach — its attention walks {} block(s), \
             {} positions",
            rows.end(),
            step.walk,
            step.reach()
        )));
    }
    if let Some(&p) = blocks.iter().find(|&&p| p >= step.pages) {
        return Err(PagedStepError(format!(
            "block page {p} is past the pool's {} page(s)",
            step.pages
        )));
    }
    Ok(())
}

/// `INT32_ELEMS_PER_STICK` — a page index table row's width.
pub const INT32_ELEMS_PER_STICK: usize =
    <ktir_superdsc::superdsc_opspec::SenUint32 as ktir_superdsc::superdsc_opspec::DataFormat>::ELEMS_PER_STICK
        as usize;

/// The page index table: `[walk, INT32_ELEMS_PER_STICK]` int32, row `i` holding block `i`'s page at
/// column 0 — `table[:, 0] = block_table[s, :walk]`, the block table's own zeros past the request's
/// blocks.
pub fn page_table(
    step: &PagedStep,
    blocks: &[u32],
    rows: &StepRows,
) -> Result<Vec<i32>, PagedStepError> {
    check(step, step.rows as usize, blocks, rows)?;
    let mut t = vec![0i32; step.walk as usize * INT32_ELEMS_PER_STICK];
    for (i, &page) in blocks.iter().take(step.walk as usize).enumerate() {
        t[i * INT32_ELEMS_PER_STICK] = page as i32;
    }
    Ok(t)
}

/// The mask stack: `[walk, q, page_slots]`, tile `i` column `c` keyed by position
/// `i · page_slots + c` — 0 at or before query row `m`'s position, [`MASKED`] after it. `q` is
/// the row count one request's stack serves — the graph's rows for one sequence's whole forward
/// ([`step_sources`]'s spelling), the extent's per-request width for one request of a packed
/// rung's ([`card_step_sources`]'s); the two are the same number at one request.
pub fn mask_stack(step: &PagedStep, rows: &StepRows) -> Vec<f32> {
    mask_stack_of(step, step.rows as usize, rows)
}

fn mask_stack_of(step: &PagedStep, q: usize, rows: &StepRows) -> Vec<f32> {
    let slots = step.page_slots as usize;
    let mut m = vec![MASKED; step.walk as usize * q * slots];
    for i in 0..step.walk as usize {
        for r in 0..q {
            let pos = rows.pos(r);
            let base = (i * q + r) * slots;
            for c in 0..slots {
                if i * slots + c <= pos {
                    m[base + c] = 0.0;
                }
            }
        }
    }
    m
}

/// The KV write index: one `[rows]` index per kv head, one after another — head `h` of row `r`'s
/// token at cache row `(page · nkvh + h) · page_slots + slot`, at the row's position.
pub fn write_rows(
    step: &PagedStep,
    blocks: &[u32],
    rows: &StepRows,
) -> Result<Vec<i32>, PagedStepError> {
    check(step, step.rows as usize, blocks, rows)?;
    let (slots, nkvh, q) = (
        step.page_slots as usize,
        step.nkvh as usize,
        step.rows as usize,
    );
    let mut w = Vec::with_capacity(nkvh * q);
    for h in 0..nkvh {
        for r in 0..q {
            let pos = rows.pos(r);
            let page = blocks[pos / slots] as usize;
            w.push(((page * nkvh + h) * slots + pos % slots) as i32);
        }
    }
    Ok(w)
}

/// ⭐ A QUERY ROW INDEX THAT IS CONSECUTIVE BY CONSTRUCTION. The prefill kernel under Triton's
/// 8-row scatter floor reads and writes its rows as the one tile at `query_row_index[0]`
/// (`page_attn_head_major_prefill.py`, `out_tile.store([first, 0], attn)`), which is the rows the
/// index names only when they are consecutive. The only ways to make one are [`Self::of`] — the
/// forward's rows in order from 0 — and [`Self::of_request`] — one request's window of a packed
/// rung, in order from its first row — so the kernel's assumption is a property of the type, not
/// a comment (#1038).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryRowIndex {
    first: u32,
    rows: u32,
}

impl QueryRowIndex {
    /// The forward's `rows` query rows, `0..rows`, in order.
    pub const fn of(step: &PagedStep) -> Self {
        Self {
            first: 0,
            rows: step.rows,
        }
    }

    /// ⭐ REQUEST `r`'s WINDOW of a packed rung — `r·w .. r·w + w` in order, `w` the extent's
    /// per-request width: the rung's graph rows ARE its requests' windows in order (request `r`
    /// owns rows `r·w..r·w+w`, the row its answer comes back from), so the kernel's one-tile
    /// scatter serves each request's rows exactly as it served one sequence's. `r` comes from the
    /// stager's own `0..requests` walk over the capacity's slots, so the window is in range by
    /// construction; at one request (`r = 0`, `w = rows`) this is [`Self::of`].
    pub const fn of_request(step: &PagedStep, r: u32) -> Self {
        let w = step.extent.per_request().get();
        Self {
            first: r * w,
            rows: w,
        }
    }

    /// The staged words: `first, first + 1, …, first + rows - 1`.
    pub fn words(self) -> Vec<i32> {
        (self.first..self.first + self.rows)
            .map(|v| v as i32)
            .collect()
    }
}

/// The query row index: the forward's `rows` query rows, in order ([`QueryRowIndex`]).
pub fn query_rows(step: &PagedStep) -> Vec<i32> {
    QueryRowIndex::of(step).words()
}

/// `kv_row_pool`: `[pages, nkvh, 1]`, page `p`'s kv head `kv` at folded-cache row `p · nkvh + kv`.
pub fn kv_row_pool(step: &PagedStep) -> Vec<i32> {
    (0..(step.pages * step.nkvh) as i32).collect()
}

/// One step's sources, each at its tid and its shape — the ONE spelling both stagers bind: the mask
/// stack as f16-channel values, and the int32 tables as little-endian words (their views are
/// int32, so they ride the typed channel). `kv_row_pool` rides only where the decode kernel reads it
/// ([`PagedStep::reads_kv_row_pool`]).
pub struct StepSources {
    pub mask: (u32, Vec<f32>, Vec<usize>),
    pub int32: Vec<(u32, Vec<u8>, Vec<usize>)>,
}

pub fn step_sources(
    step: &PagedStep,
    blocks: &[u32],
    rows: &StepRows,
) -> Result<StepSources, PagedStepError> {
    let words = |w: Vec<i32>| -> Vec<u8> { w.into_iter().flat_map(i32::to_le_bytes).collect() };
    let (q, nkvh) = (step.rows as usize, step.nkvh as usize);
    let mut int32 = vec![
        (
            step.page_table,
            words(page_table(step, blocks, rows)?),
            vec![step.walk as usize, INT32_ELEMS_PER_STICK],
        ),
        (
            step.write_rows,
            words(write_rows(step, blocks, rows)?),
            vec![nkvh * q],
        ),
        (step.query_rows, words(query_rows(step)), vec![q]),
    ];
    if step.reads_kv_row_pool() {
        int32.push((
            step.kv_row_pool,
            words(kv_row_pool(step)),
            vec![step.pages as usize, nkvh],
        ));
        // The add identity the chained index's run copy reads on the card.
        let run = ktir_superdsc::emit::lower_ktir_to_superdsc::INDEX_RUN_WORDS as usize;
        int32.push((step.int32_zero, words(vec![0; run]), vec![1, run]));
    }
    Ok(StepSources {
        mask: (step.mask, mask_stack(step, rows), mask_shape(step)),
        int32,
    })
}

/// One sequence of a decode batch: its KV blocks, and the position of the one token this step
/// decodes (its context before the step).
#[derive(Clone, Copy, Debug)]
pub struct BatchSeq<'a> {
    pub blocks: &'a [u32],
    pub pos: usize,
}

/// A decode batch's sources: `rep_row_ids` `[E · P]`, `chunk_page_ids` `[C, E · P]` and
/// `mask_by_chunk` `[C, J, B, 1, 1, page_slots]` (`E = J · B`, `P` the chunking's
/// `rows_per_page`), and the KV write index of each sequence's token —
/// `SpyreAttentionMetadataBuilder.build`'s decode-batch precomputes:
///
/// * the query rows are the sequences' own, `0..n`, a padded sequence's row 0, repeated once per block
///   slot (`query_row_ids.repeat(blocks_per_chunk)`);
/// * the page ids are each sequence's blocks down the block-slot axis, zero past them;
/// * both index tables as the kernel gathers: entry `e`'s row `a` at `e · P + a` holding
///   `x · P + a` — at `P = 1` theirs as they are; at `P = nkvh` (a chunk under Triton's gather
///   floor in pages) folded by kv head, the per-sequence decode kernel's `kv_row_pool` form, which
///   their batched builder never stages ([`BatchChunking`]);
/// * the mask is 0 where a sequence's key is at or before its position and `finfo(f16).min` past
///   it, `-inf` on the block slots past its blocks, and a padded sequence's block 0 `finfo.min` (so
///   its softmax stays finite); one row per (block slot, sequence), which every kv head reads
///   through the kernel's broadcast, as their builder's `expand` over the kv axis does.
///
/// A padded row's KV write lands on [`NULL_SLOT`], as spyre-inference's padded tokens' do: its
/// attention reads page 0 under the padded sequence's mask and answers nothing anyone reads, so what
/// it writes must reach no request's page.
pub fn batch_sources(
    step: &PagedStep,
    seqs: &[BatchSeq<'_>],
) -> Result<StepSources, PagedStepError> {
    let chunking = step.batch().ok_or_else(|| {
        PagedStepError("batch_sources: this bundle's rows are one sequence's, not a batch".into())
    })?;
    // ⛔ THE LIVE SEQUENCES ARE ROWS `0..n`, THE PADDING `n..b`. The kernel stores its output as the
    // first `b` rows of `out`, row `s` sequence `s`'s (`out[:num_seqs].copy_(attn)`), and the
    // worker reads sequence `s`'s token from launch slot `s`; this staging is what makes the two
    // the same rows — `rep` below names sequence `s` at row `s` and a pad row as row 0 (#1038).
    let (b, n) = (step.rows as usize, seqs.len());
    if n == 0 || n > b {
        return Err(PagedStepError(format!(
            "a decode batch of {n} sequence(s) in a {b}-sequence bundle"
        )));
    }
    let (slots, nkvh) = (step.page_slots as usize, step.nkvh as usize);
    let (j, c) = (
        chunking.blocks_per_chunk as usize,
        chunking.num_chunks as usize,
    );
    let padded = c * j;
    for (s, q) in seqs.iter().enumerate() {
        let need = (q.pos + 1).div_ceil(slots);
        if need > step.walk as usize || q.blocks.len() < need {
            return Err(PagedStepError(format!(
                "sequence {s} at position {} needs {need} block(s); it holds {} and the bundle \
                 walks {}",
                q.pos,
                q.blocks.len(),
                step.walk
            )));
        }
        if let Some(&p) = q.blocks[..need].iter().find(|&&p| p >= step.pages) {
            return Err(PagedStepError(format!(
                "sequence {s}: block page {p} is past the pool's {} page(s)",
                step.pages
            )));
        }
    }
    let words = |w: Vec<i32>| -> Vec<u8> { w.into_iter().flat_map(i32::to_le_bytes).collect() };
    // Each entry as the `rows_per_page` rows `x · rows_per_page + a` the kernel gathers by.
    let per = chunking.rows_per_page as i32;
    let fold = |x: &[i32]| -> Vec<i32> {
        x.iter()
            .flat_map(|&v| (0..per).map(move |a| v * per + a))
            .collect()
    };
    // query_row_ids, repeated once per block slot: entry `j · B + s`.
    let rep: Vec<i32> = (0..j)
        .flat_map(|_| (0..b).map(|s| if s < n { s as i32 } else { 0 }))
        .collect();
    // block_ids_padded `[C·J, B]`: row `i`, column `s` = sequence `s`'s block `i`.
    let mut pages = vec![0i32; padded * b];
    // mask_bs_bb `[B, C·J, slots]`.
    let mut mask_bs = vec![f32::NEG_INFINITY; b * padded * slots];
    for (s, q) in seqs.iter().enumerate() {
        let need = (q.pos + 1).div_ceil(slots);
        for i in 0..need {
            pages[i * b + s] = q.blocks[i] as i32;
            for col in 0..slots {
                mask_bs[(s * padded + i) * slots + col] = if i * slots + col <= q.pos {
                    0.0
                } else {
                    MASKED
                };
            }
        }
    }
    for s in n..b {
        for col in 0..slots {
            mask_bs[(s * padded) * slots + col] = MASKED;
        }
    }
    // `[C·J, B, slots]`: row `i`, sequence `s` is mask_bs_bb[s, i]. Every kv head reads that one row
    // through the kernel's broadcast, so it is staged once, not once per kv head.
    let mut chunk_masks = Vec::with_capacity(padded * b * slots);
    for i in 0..padded {
        for s in 0..b {
            chunk_masks.extend_from_slice(
                &mask_bs[(s * padded + i) * slots..(s * padded + i + 1) * slots],
            );
        }
    }
    // One token per real row, at its position; a padded row at the null slot.
    let mut write = Vec::with_capacity(nkvh * b);
    for h in 0..nkvh {
        for r in 0..b {
            let (page, slot) = match seqs.get(r) {
                Some(q) => (q.blocks[q.pos / slots] as usize, q.pos % slots),
                None => (NULL_SLOT / slots, NULL_SLOT % slots),
            };
            write.push(((page * nkvh + h) * slots + slot) as i32);
        }
    }
    let e = j * b;
    Ok(StepSources {
        mask: (step.chunk_masks, chunk_masks, vec![c * e, slots]),
        int32: vec![
            (step.rep_rows, words(fold(&rep)), vec![e * per as usize]),
            (
                step.chunk_pages,
                words(fold(&pages)),
                vec![c * e * per as usize],
            ),
            (step.write_rows, words(write), vec![nkvh * b]),
        ],
    })
}

/// ⭐ ONE STEP ON THE CARD — the step's own operands ([`step_sources`], [`batch_sources`]) as the
/// card's placements hold them ([`PagedStep::card_placements`]): the mask as values the session
/// narrows to the card's fp16 at bind, laid stick-major ([`card_mask`]), and the int32 tables as
/// little-endian words, the index table a walk reads one row per trip laid at the walk's card step
/// ([`PagedStep::card_walk_step`], `loop_fold`'s pitched index table). `kv_row_pool` and the int32
/// zero are no step's: they are the same for every step of a pool, staged once at load
/// ([`card_statics`]).
/// ⭐ THE MASK STACK AS THE CARD READS IT — stick-major over the whole `[rows, page_slots]` stack
/// ([`crate::forward_tape::stick_scatter`]): row `r`'s stick `s` at stick `s · rows + r`, the
/// layout the kernels address the stack in (a tile's second stick one plane of `rows` further on),
/// and the reason a trip's tile is [`PagedStep::card_walk_step`] further along every plane. One
/// stick wide, at 64-slot pages, it is the row-major stack byte for byte; at 128 slots the row-major
/// stack read stick-major handed every odd query row the next row's masked half — MEASURED on card,
/// granite 2b and 8b prompt-blind, the P·V rows of odd prompt positions all zero.
fn card_mask(mask: &[f32], shape: &[usize]) -> Vec<f32> {
    let (width, rows) = match shape.split_last() {
        Some((&w, lead)) => (w, lead.iter().product()),
        None => (0, 0),
    };
    if width <= ktir_superdsc::superdsc_opspec::Df::Fp16.elems_per_stick() as usize {
        return mask.to_vec();
    }
    crate::forward_tape::stick_scatter(rows, width, mask)
}

/// The single-sequence mask stack's `[walk · rows, page_slots]` shape — the one statement the host
/// stack ([`step_sources`]) and the card's ([`card_step_sources`]) are laid by.
fn mask_shape(step: &PagedStep) -> Vec<usize> {
    vec![(step.walk * step.rows) as usize, step.page_slots as usize]
}

pub struct CardSources {
    pub mask: (u32, Vec<f32>),
    pub int32: Vec<(u32, Vec<u8>)>,
}

impl CardSources {
    /// The int32 table staged at `tid`.
    pub fn words(&self, tid: u32) -> Option<&[u8]> {
        self.int32
            .iter()
            .find(|(t, _)| *t == tid)
            .map(|(_, w)| w.as_slice())
    }
}

/// `logical`'s rows of `row` bytes with row `i` at byte `i · step` — an index table one row per walk
/// trip, at the card's walk step.
fn pitched(logical: &[u8], row: usize, step: usize) -> Vec<u8> {
    let mut out = vec![0u8; logical.len().div_ceil(row) * step];
    for (i, r) in logical.chunks(row).enumerate() {
        out[i * step..i * step + r.len()].copy_from_slice(r);
    }
    out
}

fn le_words(w: Vec<i32>) -> Vec<u8> {
    w.into_iter().flat_map(i32::to_le_bytes).collect()
}

/// ⭐ ONE REQUEST'S STEP — its KV blocks (as pool pages) and its rows of the forward. The card
/// stager ([`card_step_sources`]) takes the launch's requests as this list, dense from slot 0:
/// request `r`'s tables stage at `r · `[`PagedStep::card_request_pitch`] within their
/// placements, and the slots past the live ones (the rung's capacity is the count, the launch's
/// live requests the fill) stage the empty-slot laws — an empty slot's writes reach no request's
/// cache.
#[derive(Clone, Copy, Debug)]
pub struct RequestStep<'a> {
    pub blocks: &'a [u32],
    pub rows: StepRows,
}

/// [`step_sources`] on the card: the launch's requests' step, each request's page table one block
/// per trip, its block at its own offset ([`PagedStep::card_request_pitch`]).
///
/// ⭐ THE EMPTY-SLOT LAW — a launch with fewer live requests than the rung's capacity stages the
/// slots past them as NO request's: page table rows name page 0 (the null page,
/// [`page_of_block`]'s own reserve), the mask stack is fully [`MASKED`] (a finite all-masked tile
/// whose weights vanish under the running maximum), and the KV write index lands on
/// [`NULL_SLOT`] — the empty slot's forward runs, but nothing it writes reaches a request's cache
/// and nothing it answers is read back. At one request nothing moves: the buffers are exactly
/// one sequence's own ([`step_sources`]'s laws), byte for byte.
pub fn card_step_sources(
    step: &PagedStep,
    reqs: &[RequestStep<'_>],
) -> Result<CardSources, PagedStepError> {
    if step.batch().is_some() {
        return Err(PagedStepError(
            "card_step_sources: this bundle's rows are a decode batch's — `card_batch_sources`"
                .into(),
        ));
    }
    let (w, cap) = (
        step.extent.per_request().get() as usize,
        step.extent.requests().get() as usize,
    );
    if reqs.is_empty() || reqs.len() > cap {
        return Err(PagedStepError(format!(
            "a step of {} live request(s) in a rung that packs {cap}",
            reqs.len()
        )));
    }
    for (r, req) in reqs.iter().enumerate() {
        check(step, w, req.blocks, &req.rows)
            .map_err(|e| PagedStepError(format!("request {r}: {}", e.0)))?;
    }
    let (walk, slots, nkvh) = (
        step.walk as usize,
        step.page_slots as usize,
        step.nkvh as usize,
    );
    let (pitch, trip) = (
        step.card_request_pitch() as usize,
        step.card_walk_step() as usize,
    );
    // ⭐ THE PAGE INDEX TABLE — request `r`'s `[walk, INT32_ELEMS_PER_STICK]` rows at
    // `r·P + i·trip`: the walk's own trip pitch within the block, the request's block pitch
    // between blocks. An empty slot's rows stay zero — page 0, the null page.
    let mut table = vec![0u8; (cap - 1) * pitch + walk * trip];
    for (r, req) in reqs.iter().enumerate() {
        for (i, &page) in req.blocks.iter().take(walk).enumerate() {
            let at = r * pitch + i * trip;
            table[at..at + 4].copy_from_slice(&(page as i32).to_le_bytes());
        }
    }
    // ⭐ THE MASK STACK — request `r`'s `[walk·w, page_slots]` stack, stick-major ([`card_mask`],
    // the planes the walk's per-trip shift reads), at value offset `r·P/2`. An empty slot's stack
    // stays fully [`MASKED`] — the zero of its own fill.
    let mut mask = vec![MASKED; (cap - 1) * pitch / 2 + walk * w * slots];
    for (r, req) in reqs.iter().enumerate() {
        let scattered = card_mask(&mask_stack_of(step, w, &req.rows), &[walk * w, slots]);
        let at = r * pitch / 2;
        mask[at..at + scattered.len()].copy_from_slice(&scattered);
    }
    // ⭐ THE KV WRITE INDEX — packed over the graph's rows, head-major: row `r·w + j` (request
    // `r`'s row `j`) names its OWN request's cache slot at its own position, a pad row clamped to
    // its request's last real one (the host law, an idempotent rewrite); an EMPTY slot's rows name
    // [`NULL_SLOT`], the null page's own slot. Not pitched: the KV writers run once per step over
    // the whole graph — they are not the walk the per-request shift serves.
    let mut write: Vec<i32> = Vec::with_capacity(nkvh * cap * w);
    for h in 0..nkvh {
        for r in 0..cap {
            match reqs.get(r) {
                Some(req) => {
                    for j in 0..w {
                        let pos = req.rows.pos(j);
                        let page = req.blocks[pos / slots] as usize;
                        write.push(((page * nkvh + h) * slots + pos % slots) as i32);
                    }
                }
                None => {
                    let null = ((NULL_SLOT / slots) * nkvh + h) * slots + NULL_SLOT % slots;
                    write.extend(std::iter::repeat_n(null as i32, w));
                }
            }
        }
    }
    // ⭐ THE QUERY ROW INDEX — request `r`'s consecutive window `r·w .. r·w+w`
    // ([`QueryRowIndex::of_request`]) at word offset `r·P/4`.
    let mut query = vec![0i32; (cap - 1) * pitch / 4 + w];
    for r in 0..cap {
        let at = r * pitch / 4;
        query[at..at + w].copy_from_slice(&QueryRowIndex::of_request(step, r as u32).words());
    }
    Ok(CardSources {
        mask: (step.mask, mask),
        int32: vec![
            (step.page_table, table),
            (step.write_rows, le_words(write)),
            (step.query_rows, le_words(query)),
        ],
    })
}

/// [`batch_sources`] on the card: a decode batch's step, its chunk page ids one chunk per trip.
pub fn card_batch_sources(
    step: &PagedStep,
    seqs: &[BatchSeq<'_>],
) -> Result<CardSources, PagedStepError> {
    let logical = batch_sources(step, seqs)?;
    let rows = step
        .batch()
        .ok_or_else(|| PagedStepError("card_batch_sources: not a decode batch".into()))?
        .index_rows() as usize;
    let int32 = logical
        .int32
        .into_iter()
        .map(|(tid, bytes, _)| {
            if tid == step.chunk_pages {
                (
                    tid,
                    pitched(&bytes, rows * 4, step.card_walk_step() as usize),
                )
            } else {
                (tid, bytes)
            }
        })
        .collect();
    let (mask_tid, mask, shape) = logical.mask;
    Ok(CardSources {
        mask: (mask_tid, card_mask(&mask, &shape)),
        int32,
    })
}

/// ⭐ ONE REQUEST'S STEP ON EITHER KIND OF BUNDLE — its prompt rows on a sequence bundle
/// ([`step_sources`]), or its one decode row on a decode bundle, which spyre-inference decodes with
/// the batched kernel at one sequence ([`batch_sources`] of the request alone: its sequence buckets
/// start at 1). The bundle states which it is ([`PagedStep::paged_rows`]); a decode bundle takes
/// one row of a request, its token at `rows.start`.
pub fn request_sources(
    step: &PagedStep,
    blocks: &[u32],
    rows: &StepRows,
) -> Result<StepSources, PagedStepError> {
    match step.paged_rows {
        PagedRows::Sequence => step_sources(step, blocks, rows),
        PagedRows::Batch if rows.real.get() == 1 => batch_sources(
            step,
            &[BatchSeq {
                blocks,
                pos: rows.start,
            }],
        ),
        PagedRows::Batch => Err(PagedStepError(format!(
            "a decode bundle runs one row of a request, and this step asks for {} rows from {}",
            rows.real, rows.start
        ))),
    }
}

/// ⭐ THE POOL'S STATICS ON THE CARD — `kv_row_pool` at the card's run pitch (page `p`'s `nkvh` rows
/// in the first words of its [`INDEX_RUN_WORDS`] run: the run the decode walk's chained index copies
/// per page) and the int32 zero that copy adds to. The same for every step, so the session whose
/// bundles read them (the decode kernel's) stages them once, at load, where
/// [`PagedStep::card_placements`] puts them. Empty for a bundle whose attention reads no
/// `kv_row_pool`.
///
/// [`INDEX_RUN_WORDS`]: ktir_superdsc::emit::lower_ktir_to_superdsc::INDEX_RUN_WORDS
pub fn card_statics(step: &PagedStep) -> Result<Vec<(u32, Vec<u8>)>, PagedStepError> {
    if !step.reads_kv_row_pool() {
        return Ok(Vec::new());
    }
    let run = ktir_superdsc::emit::lower_ktir_to_superdsc::INDEX_RUN_WORDS as usize;
    let nkvh = step.nkvh as usize;
    if nkvh > run {
        return Err(PagedStepError(format!(
            "a page's {nkvh} kv-head rows do not fit its {run}-word index run"
        )));
    }
    let mut words = vec![0i32; step.pages as usize * run];
    for (p, page) in kv_row_pool(step).chunks(nkvh).enumerate() {
        words[p * run..p * run + nkvh].copy_from_slice(page);
    }
    Ok(vec![
        (step.kv_row_pool, le_words(words)),
        (step.int32_zero, vec![0u8; run * 4]),
    ])
}

/// A pool holding token-major cache rows: row `t` of `cache` (`[·, nkvh · hd]`) laid into its page —
/// head `h` at pool row `(blocks[t / page_slots] · nkvh + h) · page_slots + t % page_slots` — for
/// every `t < valid`; every other row zero. Returned as the pool's PREFIX through the last page a
/// row lands on (empty when `valid` is 0): a pool is the model's whole context, and the rows past
/// that page are the zeroed allocation the session already holds. The bake's numeric gate stages its
/// reference's cached keys this way, so the paged walk reads the very rows the reference attends.
pub fn pool_of_rows(
    step: &PagedStep,
    hd: usize,
    blocks: &[u32],
    cache: &[f32],
    valid: usize,
) -> Result<Vec<f32>, PagedStepError> {
    let (slots, nkvh) = (step.page_slots as usize, step.nkvh as usize);
    let width = nkvh * hd;
    if cache.len() < valid * width {
        return Err(PagedStepError(format!(
            "{valid} cached row(s) of {width} need {} values; the cache holds {}",
            valid * width,
            cache.len()
        )));
    }
    let need = valid.div_ceil(slots);
    if valid > step.capacity() || blocks.len() < need {
        return Err(PagedStepError(format!(
            "{valid} cached row(s) need {need} block(s) of a {}-slot pool; {} given",
            step.capacity(),
            blocks.len()
        )));
    }
    if let Some(&p) = blocks[..need].iter().find(|&&p| p >= step.pages) {
        return Err(PagedStepError(format!(
            "block page {p} is past the pool's {} page(s)",
            step.pages
        )));
    }
    let pages = blocks[..need]
        .iter()
        .map(|&p| p as usize + 1)
        .max()
        .unwrap_or(0);
    let mut pool = vec![0f32; pages * slots * width];
    for t in 0..valid {
        let page = blocks[t / slots] as usize;
        for h in 0..nkvh {
            let dst = ((page * nkvh + h) * slots + t % slots) * hd;
            let src = t * width + h * hd;
            pool[dst..dst + hd].copy_from_slice(&cache[src..src + hd]);
        }
    }
    Ok(pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(n: usize) -> std::num::NonZeroUsize {
        std::num::NonZeroUsize::new(n).unwrap()
    }

    /// The extent pair a fixture's rows state — the mint's own spelling, `fn` so `expect` reads.
    /// The pair's held product must equal the fixture's `rows` (the mint's agreement, which
    /// [`crate::lower_subtile_tape_to_ktir::paged_step_of_ir`] checks on every real step).
    fn pair(per_request: u32, requests: u32) -> crate::wiring::QueryExtent {
        crate::wiring::QueryExtent::of_baked_rung(
            crate::wiring::RungWidth::of_baked_rows(per_request)
                .expect("the fixture's row count is nonzero"),
            crate::wiring::RungCap::of_baked_seqs(requests)
                .expect("the fixture's request count is nonzero"),
        )
        .expect("the fixture's rows fit a u32")
    }

    fn step() -> PagedStep {
        PagedStep {
            mask: 10,
            page_table: 11,
            write_rows: 12,
            query_rows: 13,
            kv_row_pool: 14,
            rep_rows: 15,
            chunk_pages: 16,
            chunk_masks: 17,
            int32_zero: 18,
            pages: 4,
            page_slots: 4,
            rows: 3,
            nkvh: 2,
            // Two kv heads of one query head each, one stick a head: the fold's own floor law
            // (`BatchChunking::of`) leaves these fixtures at their kv-head cut.
            q_width: 128,
            head_size: 64,
            walk: 4,
            paged_rows: PagedRows::Sequence,
            extent: pair(3, 1),
        }
    }

    /// A 4-sequence batch rung over a 16-page pool of 4-slot pages, walking 2 blocks: two real
    /// sequences (positions 5 and 2) and two padded rows, at the chunking `of(4, 2)` = J 2, C 1.
    fn batch() -> PagedStep {
        PagedStep {
            pages: 16,
            rows: 4,
            walk: 2,
            paged_rows: PagedRows::Batch,
            extent: pair(1, 4),
            ..step()
        }
    }

    /// A ladder proves its walks: one rung (a prompt chunk's own pages) and several (a decode body's
    /// walk buckets) stage a walk at every position they admit; a sweep that is no whole number of
    /// pages, or reaches past the pool, does not.
    #[test]
    fn every_position_a_ladder_admits_stages_a_walk() {
        use crate::bundle_code::{LadderRung, RerollMeta, SiblingFp, SweptCols};
        use std::borrow::Cow;
        let meta = |caps: &[u32]| RerollMeta {
            prefix: SiblingFp::none(),
            suffix: SiblingFp::none(),
            bodies: Cow::Owned(vec![SiblingFp::none()]),
            bodies_fused: Cow::Owned(vec![SiblingFp::none()]),
            rungs: Cow::Owned(
                caps.iter()
                    .map(|&c| LadderRung {
                        active_cap: SweptCols::new(c),
                        bodies: Cow::Owned(vec![SiblingFp::none()]),
                        bodies_fused: Cow::Owned(vec![SiblingFp::none()]),
                    })
                    .collect(),
            ),
            fallback_rung: caps.len().saturating_sub(1) as u32,
            launches: Cow::Owned(Vec::new()),
            kv_page_stride: 0,
            prefix_weight_bank: 0,
            suffix_weight_bank: 0,
        };
        // `step()`: a 4-page pool of 4-slot pages.
        assert_eq!(prove_ladder_walks(&meta(&[8]), step()), Ok(()));
        assert_eq!(prove_ladder_walks(&meta(&[4, 8, 16]), step()), Ok(()));
        assert!(
            prove_ladder_walks(&meta(&[6]), step()).is_err(),
            "half a page"
        );
        assert!(
            prove_ladder_walks(&meta(&[4, 32]), step()).is_err(),
            "past the pool"
        );
        assert!(prove_ladder_walks(&meta(&[]), step()).is_err(), "no rung");
    }

    /// ⭐ THE INTERVAL PROOF IS THE EVERY-POSITION PROOF. `prove_ladder_walks` checks one position per
    /// interval of constant rung selection; this checks it against walking every position, over
    /// ladders that pass and ladders that fail each way, in any rung order.
    #[test]
    fn the_interval_proof_agrees_with_every_position() {
        use crate::bundle_code::{LadderRung, RerollMeta, SiblingFp, SweptCols};
        use std::borrow::Cow;
        let meta = |caps: &[u32]| RerollMeta {
            prefix: SiblingFp::none(),
            suffix: SiblingFp::none(),
            bodies: Cow::Owned(vec![SiblingFp::none()]),
            bodies_fused: Cow::Owned(vec![SiblingFp::none()]),
            rungs: Cow::Owned(
                caps.iter()
                    .map(|&c| LadderRung {
                        active_cap: SweptCols::new(c),
                        bodies: Cow::Owned(vec![SiblingFp::none()]),
                        bodies_fused: Cow::Owned(vec![SiblingFp::none()]),
                    })
                    .collect(),
            ),
            fallback_rung: caps.len().saturating_sub(1) as u32,
            launches: Cow::Owned(Vec::new()),
            kv_page_stride: 0,
            prefix_weight_bank: 0,
            suffix_weight_bank: 0,
        };
        let every = |m: &RerollMeta<'_>, s: PagedStep| -> Result<(), String> {
            let ladder = m.ladder()?;
            let top = ladder
                .rungs()
                .iter()
                .map(|r| r.active_cap.get())
                .max()
                .unwrap_or(0);
            for p in 1..=u64::from(top) {
                let (_, rung) = ladder.rung_for(p);
                if !rung.active_cap.covers(p)
                    || s.on_body(s.rows, s.paged_rows, rung.active_cap.get())
                        .is_none()
                {
                    return Err(format!("{p}"));
                }
            }
            Ok(())
        };
        // `step()`: a 4-page pool of 4-slot pages.
        for caps in [
            &[8][..],
            &[4, 8, 16],
            &[16, 4, 8],
            &[6],
            &[4, 6, 16],
            &[4, 32],
            &[16, 12],
            &[1],
            &[4, 4, 8],
            &[],
        ] {
            let m = meta(caps);
            assert_eq!(
                prove_ladder_walks(&m, step()).is_ok(),
                every(&m, step()).is_ok(),
                "ladder {caps:?}"
            );
        }
    }

    /// The pool is the model's context where every layer's K and V of it fit one device region,
    /// else the whole pages that do.
    #[test]
    fn the_pool_is_the_context_one_region_holds() {
        // K and V of every layer, at fp16, for one token.
        let slot = |layers: u64, nkvh: u64, hd: u64| layers * 2 * nkvh * hd * 2;
        let ctx = 131_072;
        // llama-3.2-1b: 16 layers × 8 × 64 — 4 GiB, the whole context.
        assert_eq!(pool_slots(ctx, slot(16, 8, 64)).map(|s| s.get()), Some(ctx));
        // granite-3.1-8b: 40 layers × 8 × 128 — 20 GiB, so the 104832 positions 16 GiB holds in
        // whole pages (819 of 128 slots).
        let p = pool_slots(ctx, slot(40, 8, 128)).expect("a page fits");
        assert_eq!(p.get(), 104_832);
        assert!(
            p.get()
                .is_multiple_of(SchedulerBlockTokens::REFERENCE.get()),
            "whole pages"
        );
        assert!(u64::from(p.get()) * slot(40, 8, 128) <= scratchy_spyre_bundle::MAX_SEGMENT_BYTES);
        assert!(
            u64::from(p.get() + SchedulerBlockTokens::REFERENCE.get()) * slot(40, 8, 128)
                > scratchy_spyre_bundle::MAX_SEGMENT_BYTES
        );
        // A context shorter than the pool would hold is the pool.
        assert_eq!(
            pool_slots(4096, slot(40, 8, 128)).map(|s| s.get()),
            Some(4096)
        );
        // Not one page fits.
        assert_eq!(
            pool_slots(ctx, scratchy_spyre_bundle::MAX_SEGMENT_BYTES / 32),
            None
        );
    }

    /// The batch's operands are spyre-inference's: query rows `0..n` (a pad sequence's 0) repeated
    /// per block slot, page ids down the block-slot axis, and the mask -inf past a sequence's blocks,
    /// finfo.min past its position, a pad sequence's block 0 finfo.min.
    #[test]
    fn a_decode_batch_stages_spyre_inferences_metadata() {
        let s = batch();
        assert_eq!(
            s.batch(),
            Some(BatchChunking {
                seqs: 4,
                blocks_per_chunk: 2,
                num_chunks: 1,
                rows_per_page: 1
            })
        );
        let (a, b) = ([7u32, 3], [9u32]);
        let src = batch_sources(
            &s,
            &[
                BatchSeq { blocks: &a, pos: 5 },
                BatchSeq { blocks: &b, pos: 2 },
            ],
        )
        .expect("stages");
        let words = |t: u32| -> Vec<i32> {
            let (_, w, _) = src
                .int32
                .iter()
                .find(|(tid, ..)| *tid == t)
                .expect("staged");
            w.as_chunks::<4>()
                .0
                .iter()
                .map(|x| i32::from_le_bytes(*x))
                .collect()
        };
        assert_eq!(words(s.rep_rows), vec![0, 1, 0, 0, 0, 1, 0, 0]);
        // `[C·J, B]`: slot 0 = (7, 9, 0, 0), slot 1 = (3, 0, 0, 0).
        assert_eq!(words(s.chunk_pages), vec![7, 9, 0, 0, 3, 0, 0, 0]);
        // Row r's token, head h at ((page · 2 + h) · 4 + slot); the two padded rows at the null
        // slot, page 0.
        let w = words(s.write_rows);
        assert_eq!(&w[..4], &[3 * 8 + 1, 9 * 8 + 2, 0, 0]);
        assert_eq!(&w[4..], &[3 * 8 + 4 + 1, 9 * 8 + 4 + 2, 4, 4]);
        let (tid, m, shape) = &src.mask;
        // One row per (slot, sequence): every kv head reads it through the kernel's broadcast.
        assert_eq!((*tid, shape.clone()), (s.chunk_masks, vec![2 * 4, 4]));
        // Element (slot i, sequence q, column c) at (i · 4 + q) · 4 + c.
        let at = |i: usize, q: usize| &m[(i * 4 + q) * 4..][..4];
        assert_eq!(at(0, 0), &[0.0; 4]);
        assert_eq!(at(1, 0), &[0.0, 0.0, MASKED, MASKED]);
        assert_eq!(at(0, 1), &[0.0, 0.0, 0.0, MASKED]);
        assert_eq!(at(1, 1), &[f32::NEG_INFINITY; 4]);
        assert_eq!(at(0, 2), &[MASKED; 4]);
        assert_eq!(at(1, 3), &[f32::NEG_INFINITY; 4]);
    }

    /// ⭐ A CHUNK UNDER TRITON'S GATHER FLOOR IN PAGES STAGES ITS TABLES FOLDED BY KV HEAD — two
    /// sequences of one block: spyre-inference's chunk is one block slot (2 pages), under the floor of
    /// 8, so each page is gathered as its 2 kv heads' rows, and only those 4 rows under the floor
    /// take a second, masked slot (`ceil(8 / (2 · 2))`). Entry `e`'s head `a` at `e · 2 + a` holds
    /// `x · 2 + a`, of its query row and of its page alike, on the host and in the card's pitched
    /// chunk row.
    #[test]
    fn a_chunk_under_the_gather_floor_stages_its_tables_by_kv_head() {
        let s = PagedStep {
            rows: 2,
            walk: 1,
            extent: pair(1, 2),
            ..batch()
        };
        let c = s.batch().unwrap();
        assert_eq!(
            (c.blocks_per_chunk, c.num_chunks, c.rows_per_page),
            (2, 1, 2)
        );
        let (a, b) = ([5u32], [11u32]);
        let seqs = [
            BatchSeq { blocks: &a, pos: 2 },
            BatchSeq { blocks: &b, pos: 0 },
        ];
        let src = batch_sources(&s, &seqs).expect("stages");
        let words = |t: u32| -> Vec<i32> {
            ints(
                &src.int32
                    .iter()
                    .find(|(tid, ..)| *tid == t)
                    .expect("staged")
                    .1,
            )
        };
        // Entries (slot 0: rows 0, 1; slot 1: rows 0, 1), each at `row · 2 + a`.
        assert_eq!(words(s.rep_rows), vec![0, 1, 2, 3, 0, 1, 2, 3]);
        // Slot 0 pages (5, 11), slot 1 the padding's page 0, each at `page · 2 + a`.
        assert_eq!(words(s.chunk_pages), vec![10, 11, 22, 23, 0, 1, 0, 1]);
        let card = card_batch_sources(&s, &seqs).unwrap();
        let got = ints(card.words(s.chunk_pages).unwrap());
        assert_eq!(got.len(), s.card_walk_step() as usize / 4);
        assert_eq!(&got[..8], &[10, 11, 22, 23, 0, 1, 0, 1]);
        assert!(got[8..].iter().all(|&v| v == 0));
        assert_eq!(
            s.card_placements()
                .iter()
                .find(|(t, ..)| *t == s.rep_rows)
                .map(|&(.., bytes)| bytes),
            Some(u64::from(scratchy_triton_splice::INT32_STICK) * 4),
            "the folded query rows fit their one stick"
        );
    }

    /// ⭐ ONE REQUEST'S DECODE IS A DECODE BATCH OF ONE — spyre-inference's sequence buckets start at
    /// 1, so a decode bundle's one row stages the batched kernel's tables (the request alone, at its
    /// token's position), exactly what a batch of that one sequence stages; a prompt bundle stages
    /// the sequence's tables; and a decode bundle given several rows of one request refuses them.
    #[test]
    fn one_requests_decode_is_a_decode_batch_of_one() {
        let decode = PagedStep {
            rows: 1,
            paged_rows: PagedRows::Batch,
            extent: pair(1, 1),
            ..step()
        };
        let blocks = [2u32, 3];
        let got = request_sources(
            &decode,
            &blocks,
            &StepRows {
                start: 5,
                real: nz(1),
            },
        )
        .unwrap();
        let want = batch_sources(
            &decode,
            &[BatchSeq {
                blocks: &blocks,
                pos: 5,
            }],
        )
        .unwrap();
        assert_eq!(got.int32, want.int32);
        assert_eq!(got.mask, want.mask);
        let tids: Vec<u32> = got.int32.iter().map(|(t, ..)| *t).collect();
        assert!(tids.contains(&decode.rep_rows) && tids.contains(&decode.chunk_pages));
        assert!(!tids.contains(&decode.page_table) && !tids.contains(&decode.kv_row_pool));
        let prompt = step();
        let rows = StepRows {
            start: 5,
            real: nz(2),
        };
        assert_eq!(
            request_sources(&prompt, &blocks, &rows).unwrap().int32,
            step_sources(&prompt, &blocks, &rows).unwrap().int32
        );
        assert!(request_sources(&decode, &blocks, &rows).is_err());
    }

    /// A sequence whose position is past the rung's walk cannot run on it.
    #[test]
    fn a_batch_sequence_past_the_walk_is_refused() {
        let s = batch();
        let a = [1u32, 2, 3];
        assert!(batch_sources(&s, &[BatchSeq { blocks: &a, pos: 8 }]).is_err());
    }

    /// A 3-row chunk at positions 5, 6 and a pad row repeating 6, over blocks `[2, 0]`.
    #[test]
    fn the_tables_follow_spyre_inference_s_laws() {
        let s = step();
        let rows = StepRows {
            start: 5,
            real: nz(2),
        };
        let blocks = [2u32, 0];
        let pt = page_table(&s, &blocks, &rows).unwrap();
        assert_eq!(
            pt.len(),
            4 * INT32_ELEMS_PER_STICK,
            "one int32 stick per walked block"
        );
        let col0: Vec<i32> = pt.iter().step_by(INT32_ELEMS_PER_STICK).copied().collect();
        assert_eq!(
            col0,
            vec![2, 0, 0, 0],
            "the blocks, then the block table's zeros"
        );
        assert!(
            pt.iter()
                .enumerate()
                .all(|(i, &v)| i % INT32_ELEMS_PER_STICK == 0 || v == 0),
            "only column 0 holds a page"
        );
        // Position 5 is block 1 (page 0) slot 1; position 6 slot 2; the pad row repeats 6. Cache row
        // (page·2 + h)·4 + slot, one index per kv head.
        assert_eq!(
            write_rows(&s, &blocks, &rows).unwrap(),
            vec![1, 2, 2, 5, 6, 6]
        );
        let m = mask_stack(&s, &rows);
        assert_eq!(m.len(), 4 * 3 * 4, "walk · rows tiles of page_slots");
        // Row 0 (position 5) attends positions 0..=5: block 0 whole, block 1 slots 0..=1.
        assert_eq!(&m[0..4], &[0.0; 4]);
        assert_eq!(&m[12..16], &[0.0, 0.0, MASKED, MASKED]);
        // The pad row (2) is position 6, the last real row's.
        assert_eq!(&m[20..24], &[0.0, 0.0, 0.0, MASKED]);
        // Blocks past the sequence are fully masked.
        assert!(m[24..].iter().all(|&v| v == MASKED));
        assert_eq!(query_rows(&s), vec![0, 1, 2]);
        assert_eq!(kv_row_pool(&s), (0..8).collect::<Vec<i32>>());
    }

    /// spyre-inference's dispatch: the decode kernel, at one query row, is the one that reads
    /// `kv_row_pool`, so a decode step stages it and a prompt chunk does not.
    #[test]
    fn only_a_decode_step_stages_kv_row_pool() {
        let blocks = [2u32, 0];
        let tids = |s: &PagedStep, rows: StepRows| -> Vec<u32> {
            step_sources(s, &blocks, &rows)
                .unwrap()
                .int32
                .iter()
                .map(|(t, ..)| *t)
                .collect()
        };
        let chunk = step();
        let staged = tids(
            &chunk,
            StepRows {
                start: 5,
                real: nz(2),
            },
        );
        assert!(!staged.contains(&chunk.kv_row_pool) && !staged.contains(&chunk.int32_zero));
        let decode = PagedStep {
            rows: 1,
            extent: pair(1, 1),
            ..step()
        };
        let srcs = step_sources(
            &decode,
            &blocks,
            &StepRows {
                start: 5,
                real: nz(1),
            },
        )
        .unwrap();
        assert!(srcs.int32.iter().any(|(t, ..)| *t == decode.kv_row_pool));
        let (_, zero, _) = srcs
            .int32
            .iter()
            .find(|(t, ..)| *t == decode.int32_zero)
            .expect("a decode step stages the int32 zero beside kv_row_pool");
        assert!(zero.iter().all(|&b| b == 0) && !zero.is_empty());
    }

    fn ints(b: &[u8]) -> Vec<i32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|x| i32::from_le_bytes(*x))
            .collect()
    }

    /// ⭐ THE CARD HOLDS THE PAGE TABLE ONE BLOCK PER TRIP: block `i`'s page at `i · rows · 128` B —
    /// the bytes one trip moves the mask tile — and nothing else of the step moves.
    #[test]
    fn the_card_holds_the_page_table_one_block_per_trip() {
        let s = step();
        let rows = StepRows {
            start: 5,
            real: nz(2),
        };
        let blocks = [2u32, 3];
        let card = card_step_sources(
            &s,
            &[RequestStep {
                blocks: &blocks,
                rows,
            }],
        )
        .unwrap();
        let pitch = s.card_walk_step() as usize;
        assert_eq!(pitch, 3 * 128, "one fp16 stick per score row, three rows");
        let table = ints(card.words(s.page_table).unwrap());
        assert_eq!(table.len(), s.walk as usize * pitch / 4);
        for (i, &v) in table.iter().enumerate() {
            let want = match i * 4 % pitch {
                0 => [2, 3, 0, 0][i * 4 / pitch],
                _ => 0,
            };
            assert_eq!(v, want, "word {i}");
        }
        assert_eq!(
            ints(card.words(s.write_rows).unwrap()),
            write_rows(&s, &blocks, &rows).unwrap()
        );
        assert_eq!(ints(card.words(s.query_rows).unwrap()), query_rows(&s));
        assert_eq!(card.mask, (s.mask, mask_stack(&s, &rows)));
        assert!(
            card.words(s.kv_row_pool).is_none() && card.words(s.int32_zero).is_none(),
            "the pool's statics are staged at load, never per step"
        );
    }

    /// ⭐ THE CARD HOLDS THE MASK STACK STICK-MAJOR — at 128-slot pages a row is two sticks, and the
    /// kernels read the stack's row `r` stick `s` at stick `s · rows + r`, the second half of every
    /// row one whole plane on. Staged row-major (the bytes the 64-slot stack happened to share with
    /// its stick-major form) every odd query row read the next row's masked half: measured on the
    /// card, the P·V rows of odd prompt positions all zero. A sequence's stack and a decode batch's.
    #[test]
    fn the_card_holds_the_mask_stack_stick_major() {
        let s = PagedStep {
            page_slots: 128,
            ..step()
        };
        let rows = StepRows {
            start: 5,
            real: nz(2),
        };
        let blocks = [2u32, 3];
        let logical = mask_stack(&s, &rows);
        let card = card_step_sources(
            &s,
            &[RequestStep {
                blocks: &blocks,
                rows,
            }],
        )
        .unwrap();
        let (n, w) = ((s.walk * s.rows) as usize, s.page_slots as usize);
        assert_eq!(card.mask.1.len(), logical.len());
        for r in 0..n {
            for c in 0..w {
                assert_eq!(
                    card.mask.1[(c / 64) * (n * 64) + r * 64 + c % 64],
                    logical[r * w + c],
                    "row {r} slot {c}"
                );
            }
        }
        assert_ne!(
            card.mask.1, logical,
            "two sticks a row: not the row-major bytes"
        );
        let b = PagedStep {
            page_slots: 128,
            ..batch()
        };
        let seqs = [BatchSeq {
            blocks: &[2, 3],
            pos: 133,
        }];
        let (_, want, shape) = batch_sources(&b, &seqs).unwrap().mask;
        let got = card_batch_sources(&b, &seqs).unwrap().mask.1;
        let n = shape[0];
        for r in 0..n {
            for c in 0..w {
                assert_eq!(
                    got[(c / 64) * (n * 64) + r * 64 + c % 64],
                    want[r * w + c],
                    "batch row {r} slot {c}"
                );
            }
        }
    }

    /// A decode batch's chunk page ids, one chunk per trip: chunk `c`'s `J · B` ids at
    /// `c · J · B · nkvh · 128` B.
    #[test]
    fn a_decode_batch_holds_its_chunk_pages_one_chunk_per_trip() {
        let s = PagedStep {
            walk: 16,
            ..batch()
        };
        let c = s.batch().unwrap();
        assert_eq!((c.blocks_per_chunk, c.num_chunks), (8, 2));
        let blocks: Vec<u32> = (0..16).rev().collect();
        let seqs = [BatchSeq {
            blocks: &blocks,
            pos: 61,
        }];
        let logical = batch_sources(&s, &seqs).unwrap();
        let card = card_batch_sources(&s, &seqs).unwrap();
        let (_, want, _) = logical
            .int32
            .iter()
            .find(|(t, ..)| *t == s.chunk_pages)
            .unwrap();
        let (row, pitch) = (c.index_rows() as usize, s.card_walk_step() as usize / 4);
        let got = ints(card.words(s.chunk_pages).unwrap());
        assert_eq!(got.len(), c.num_chunks as usize * pitch);
        for (k, chunk) in ints(want).chunks(row).enumerate() {
            assert_eq!(&got[k * pitch..k * pitch + row], chunk, "chunk {k}");
            assert!(
                got[k * pitch + row..(k + 1) * pitch]
                    .iter()
                    .all(|&v| v == 0)
            );
        }
        for t in [s.rep_rows, s.write_rows] {
            let (_, w, _) = logical.int32.iter().find(|(x, ..)| *x == t).unwrap();
            assert_eq!(card.words(t).unwrap(), w.as_slice());
        }
        assert_eq!(card.mask.1, logical.mask.1);
    }

    /// The pool's statics: page `p`'s `nkvh` folded-cache rows in the first words of its run, and
    /// one run of int32 zeros.
    #[test]
    fn the_pool_statics_hold_each_pages_rows_at_the_run_pitch() {
        let run = ktir_superdsc::emit::lower_ktir_to_superdsc::INDEX_RUN_WORDS as usize;
        let decode = PagedStep {
            rows: 1,
            extent: pair(1, 1),
            ..step()
        };
        let statics = card_statics(&decode).unwrap();
        let pool = ints(
            &statics
                .iter()
                .find(|(t, _)| *t == decode.kv_row_pool)
                .unwrap()
                .1,
        );
        assert_eq!(pool.len(), decode.pages as usize * run);
        for p in 0..decode.pages as usize {
            assert_eq!(
                &pool[p * run..p * run + 2],
                &[2 * p as i32, 2 * p as i32 + 1]
            );
            assert!(pool[p * run + 2..(p + 1) * run].iter().all(|&v| v == 0));
        }
        let zero = &statics
            .iter()
            .find(|(t, _)| *t == decode.int32_zero)
            .unwrap()
            .1;
        assert_eq!(zero.len(), run * 4);
        assert!(zero.iter().all(|&b| b == 0));
        assert!(
            card_statics(&step()).unwrap().is_empty(),
            "a prompt chunk reads none"
        );
        assert!(
            card_statics(&batch()).unwrap().is_empty(),
            "nor does a decode batch"
        );
    }

    /// ⭐ EVERY CARD SOURCE FITS ITS PLACEMENT — at the pool's whole walk the bake places by, and at a
    /// body walking fewer blocks, whose step stages a prefix of it.
    #[test]
    fn every_card_source_fits_its_placement() {
        let fits = |placed: PagedStep, card: &CardSources, statics: &[(u32, Vec<u8>)]| {
            let at = placed.card_placements();
            let size = |tid: u32, seat: crate::wiring::CardSeat| {
                at.iter()
                    .find(|(t, s, _)| *t == tid && *s == seat)
                    .map(|&(.., b)| b)
                    .unwrap_or_else(|| panic!("t{tid} is staged but not placed"))
            };
            let (mask_tid, mask) = &card.mask;
            assert!(mask.len() as u64 * 2 <= size(*mask_tid, crate::wiring::CardSeat::Step));
            for (tid, w) in &card.int32 {
                assert!(
                    w.len() as u64 <= size(*tid, crate::wiring::CardSeat::Step),
                    "t{tid}"
                );
            }
            for (tid, w) in statics {
                assert_eq!(
                    w.len() as u64,
                    size(*tid, crate::wiring::CardSeat::Static),
                    "t{tid}"
                );
            }
        };
        for (rows, real) in [(1u32, 1usize), (3, 2)] {
            let full = PagedStep {
                rows,
                extent: pair(rows, 1),
                ..step()
            };
            for walk in [1, 2, 4] {
                let body = full
                    .on_body(rows, PagedRows::Sequence, walk * full.page_slots)
                    .unwrap();
                let st = StepRows {
                    start: 0,
                    real: nz(real),
                };
                let blocks = [3u32, 2, 1, 0];
                let card = card_step_sources(
                    &body,
                    &[RequestStep {
                        blocks: &blocks,
                        rows: st,
                    }],
                )
                .unwrap();
                fits(full, &card, &card_statics(&body).unwrap());
            }
        }
        let full = PagedStep {
            walk: 16,
            ..batch()
        };
        for walk in [1, 2, 3, 8, 16] {
            let body = full
                .on_body(4, PagedRows::Batch, walk * full.page_slots)
                .unwrap();
            let blocks: Vec<u32> = (0..walk).collect();
            let seqs = [BatchSeq {
                blocks: &blocks,
                pos: walk as usize * 4 - 1,
            }];
            fits(full, &card_batch_sources(&body, &seqs).unwrap(), &[]);
        }
    }

    /// ⭐ A PACKED STEP STAGES EACH REQUEST AT ITS OWN PITCH — the S=2 shape the packed prefill rung
    /// bakes: request `r`'s page table, mask stack and query rows at `r · card_request_pitch()`
    /// within their own regions, each request's block by the ONE-REQUEST laws (its own blocks, its
    /// own positions, its own consecutive window `r·w..r·w+w`), and the write rows PACKED over the
    /// graph's rows — every row naming its OWN request's cache slot.
    #[test]
    fn a_packed_step_stages_each_request_at_its_own_pitch() {
        // Two requests of 3 rows over a 6-row graph, the pool and walk of `step()`.
        let s = PagedStep {
            rows: 6,
            extent: pair(3, 2),
            ..step()
        };
        // The pitch: the largest of the three per-request blocks, 128-aligned — the page table
        // (4 pages · 3 rows · 128 B) dwarfs the mask (96 B) and the query rows (one stick).
        let pitch = s.card_request_pitch() as usize;
        assert_eq!(pitch, 4 * 3 * 128, "the page table is the largest block");
        let (trip, slots, nkvh) = (s.card_walk_step() as usize, 4, 2);
        assert_eq!(trip, 3 * 128);
        let a = [2u32, 3];
        let b = [1u32, 0, 3];
        let ra = StepRows {
            start: 5,
            real: nz(2),
        };
        let rb = StepRows {
            start: 1,
            real: nz(3),
        };
        let card = card_step_sources(
            &s,
            &[
                RequestStep {
                    blocks: &a,
                    rows: ra,
                },
                RequestStep {
                    blocks: &b,
                    rows: rb,
                },
            ],
        )
        .unwrap();
        // THE PAGE TABLE — request `r`'s block `i` at `r·P + i·trip`, zero past its own blocks.
        let table = ints(card.words(s.page_table).unwrap());
        assert_eq!(table.len(), (pitch + 4 * trip) / 4);
        for (r, blocks) in [(0usize, &a[..]), (1, &b[..])] {
            for i in 0..4 {
                let want = blocks.get(i).copied().unwrap_or(0) as i32;
                assert_eq!(
                    table[(r * pitch + i * trip) / 4],
                    want,
                    "request {r} block {i}"
                );
            }
        }
        // THE MASK — request `r`'s stack is the one-request law over its own rows (4-slot pages:
        // the stick-major form is the logical one), at value offset `r·P/2`.
        let mask = &card.mask.1;
        assert_eq!(mask.len(), pitch / 2 + 4 * 3 * slots);
        for (r, rows) in [(0, ra), (1, rb)] {
            let want = mask_stack_of(&s, 3, &rows);
            let at = r * pitch / 2;
            assert_eq!(&mask[at..at + want.len()], &want[..], "request {r}");
        }
        // THE WRITE ROWS — packed `[nkvh · 6]`, head-major: request `r`'s row `j` from its own
        // blocks at its own position (a pad row repeats its request's last real one).
        let write = ints(card.words(s.write_rows).unwrap());
        assert_eq!(write.len(), nkvh * 6);
        // Request A: positions 5, 6 and a pad row repeating 6, over blocks `[2, 3]`.
        // Request B: positions 1, 2, 3, over blocks `[1, 0, 3]`.
        let slot_of =
            |page: usize, h: usize, pos: usize| ((page * nkvh + h) * slots + pos % slots) as i32;
        let want_a = [slot_of(3, 0, 1), slot_of(3, 0, 2), slot_of(3, 0, 2)];
        let want_b = [slot_of(1, 0, 1), slot_of(1, 0, 2), slot_of(1, 0, 3)];
        for h in 0..nkvh {
            let at = h * 6;
            assert_eq!(
                &write[at..at + 3],
                &want_a.map(|v| v + (h * slots) as i32)[..],
                "request A, head {h}"
            );
            assert_eq!(
                &write[at + 3..at + 6],
                &want_b.map(|v| v + (h * slots) as i32)[..],
                "request B, head {h}"
            );
        }
        // THE QUERY ROWS — request `r`'s consecutive window at word offset `r·P/4`.
        let query = ints(card.words(s.query_rows).unwrap());
        assert_eq!(query.len(), pitch / 4 + 3);
        assert_eq!(&query[..3], &[0, 1, 2], "request 0's window");
        assert_eq!(&query[pitch / 4..][..3], &[3, 4, 5], "request 1's window");
        // ⭐ AND EVERY SOURCE FITS ITS PLACEMENT — each table's region grew by the capacity's
        // `(S - 1) · P` spare over its per-request size.
        let size = |tid: u32| {
            s.card_placements()
                .iter()
                .find(|(t, ..)| *t == tid)
                .map(|&(.., b)| b)
                .unwrap()
        };
        assert_eq!(size(s.page_table), 4 * 3 * 128 + pitch as u64);
        assert_eq!(size(s.mask), (4 * 3 * 4 * 2) as u64 + pitch as u64);
        assert_eq!(size(s.query_rows), 128 + pitch as u64);
        assert_eq!(
            size(s.write_rows),
            128,
            "the write rows stay the graph's own rows, unpitched"
        );
        for (tid, words) in &card.int32 {
            assert!(words.len() as u64 <= size(*tid), "t{tid}");
        }
        assert!(card.mask.1.len() as u64 * 2 <= size(s.mask));
    }

    /// ⭐ AN EMPTY SLOT IS NO REQUEST'S — a launch seating fewer requests than the rung's capacity
    /// stages the slots past its live ones to the null page, a fully-`MASKED` stack and
    /// [`NULL_SLOT`] writes: the empty slot's forward runs, but nothing it writes reaches a
    /// request's cache and nothing it answers is read back. The live slot's own tables are exactly
    /// the one-request laws' — nothing moves at the front.
    #[test]
    fn an_empty_slot_is_no_request_s() {
        let s = PagedStep {
            rows: 6,
            extent: pair(3, 2),
            ..step()
        };
        let pitch = s.card_request_pitch() as usize;
        let trip = s.card_walk_step() as usize;
        let a = [2u32, 3];
        let ra = StepRows {
            start: 5,
            real: nz(2),
        };
        let card = card_step_sources(
            &s,
            &[RequestStep {
                blocks: &a,
                rows: ra,
            }],
        )
        .unwrap();
        // The empty slot's page table rows are zero — page 0, the null page.
        let table = ints(card.words(s.page_table).unwrap());
        for i in 0..4 {
            assert_eq!(table[(pitch + i * trip) / 4], 0, "empty slot block {i}");
        }
        // Its mask stack is fully `MASKED` — the finite all-masked tile.
        assert!(
            card.mask.1[pitch / 2..].iter().all(|&v| v == MASKED),
            "the empty slot's stack is fully masked"
        );
        // Its write rows name the null slot — page 0, slot 0, head-major.
        let write = ints(card.words(s.write_rows).unwrap());
        for h in 0..2 {
            let at = h * 6 + 3;
            assert_eq!(&write[at..at + 3], &[h as i32 * 4; 3], "head {h}");
        }
        // And the live slot's tables are the one-request staging's own bytes.
        let one = PagedStep {
            rows: 3,
            extent: pair(3, 1),
            ..step()
        };
        let solo = card_step_sources(
            &one,
            &[RequestStep {
                blocks: &a,
                rows: ra,
            }],
        )
        .unwrap();
        assert_eq!(
            &card.words(s.page_table).unwrap()[..4 * trip],
            solo.words(s.page_table).unwrap(),
            "the live slot's page table is the one-request law's"
        );
        assert_eq!(
            &card.mask.1[..4 * 3 * 4],
            &solo.mask.1[..],
            "the live slot's mask is the one-request law's"
        );
        assert_eq!(
            &card.words(s.query_rows).unwrap()[..3 * 4],
            solo.words(s.query_rows).unwrap(),
            "the live slot's query rows are the one-request law's"
        );
        // A launch wider than the capacity refuses; so does an empty one. (Two requests in a
        // two-request rung is the FULL launch, not a wide one.)
        let three: Vec<RequestStep<'_>> = (0..3)
            .map(|_| RequestStep {
                blocks: &a,
                rows: ra,
            })
            .collect();
        assert!(card_step_sources(&s, &three).is_err());
        assert!(card_step_sources(&s, &[]).is_err());
        // A request wider than the per-request window refuses by name.
        assert!(
            card_step_sources(
                &one,
                &[RequestStep {
                    blocks: &[0, 1, 2, 3],
                    rows: StepRows {
                        start: 0,
                        real: nz(4),
                    },
                }]
            )
            .is_err()
        );
    }

    /// ⭐ AT ONE REQUEST NOTHING MOVES — the capacity's spare is zero, every placement size is the
    /// per-request one, and the three tables stay contiguous and LAST in the activation segment
    /// (write rows first) — the layout every rung today bakes, which is why the gate's bundle
    /// pairing against main must come back identical.
    #[test]
    fn at_one_request_nothing_moves() {
        let one = step();
        let seats: Vec<(u32, crate::wiring::CardSeat)> = one
            .card_placements()
            .iter()
            .map(|&(t, s, _)| (t, s))
            .collect();
        assert_eq!(
            seats,
            vec![
                (one.write_rows, crate::wiring::CardSeat::Step),
                (one.page_table, crate::wiring::CardSeat::Step),
                (one.mask, crate::wiring::CardSeat::Step),
                (one.query_rows, crate::wiring::CardSeat::Step),
            ],
            "write rows first, then the three tables contiguous"
        );
        let packed = PagedStep {
            rows: 6,
            extent: pair(3, 2),
            ..step()
        };
        let spare = packed.card_request_pitch();
        let own = |s: &PagedStep, tid: u32| {
            s.card_placements()
                .iter()
                .find(|(t, ..)| *t == tid)
                .map(|&(.., b)| b)
                .unwrap()
        };
        for t in [packed.page_table, packed.mask, packed.query_rows] {
            assert_eq!(own(&packed, t), own(&one, t) + spare, "t{t}'s region");
        }
        assert_eq!(own(&packed, packed.write_rows), own(&one, one.write_rows));
    }

    /// A body's step walks the whole pages its sweep covers, and a sweep that is no walk of the pool
    /// names no step. ⭐ AND THE BODY'S EXTENT — a body's step states the LAUNCH's own extent
    /// pair, by the launch's kind: a `Sequence`
    /// launch is one request of `rows` rows — `(rows, 1)` — whatever width the wiring's own
    /// pair names (the ladders bake one tape at several widths under one wiring, and `rows` is
    /// the rung actually launching — the worker's chunk ladder stages a narrow chunk under the
    /// shared wiring every prompt), and a `Batch` launch is one row per request — `(1, rows)`.
    /// Both are the same law the bake's codegen minted each rung's own pair by, so a launch on
    /// the wiring's own width re-derives the wiring's pair exactly.
    #[test]
    fn a_body_steps_at_its_own_walk() {
        let s = step();
        let body = s.on_body(3, PagedRows::Sequence, 8).unwrap();
        assert_eq!((body.rows, body.walk, body.reach()), (3, 2, 8));
        assert_eq!(
            (
                body.extent.per_request().get(),
                body.extent.requests().get()
            ),
            (3, 1),
            "a sequence launch is one request of its own rows"
        );
        // ⭐ THE WORKER'S REAL SHAPE: a chunk narrower than the wiring's representative width —
        // the prefill ladder stages a 4-row chunk under a 3-per-request wiring (one tape, one
        // wiring, several baked widths). The launch's own pair, never a refusal.
        let wide = PagedStep {
            rows: 6,
            extent: pair(3, 2),
            ..step()
        };
        let body = wide.on_body(4, PagedRows::Sequence, 8).unwrap();
        assert_eq!(
            (
                body.extent.per_request().get(),
                body.extent.requests().get()
            ),
            (4, 1),
            "a chunk narrower than the wiring's width still states its own rows"
        );
        let decode = PagedStep {
            rows: 1,
            extent: pair(1, 1),
            ..step()
        };
        let body = decode.on_body(4, PagedRows::Batch, 8).unwrap();
        assert_eq!(
            (
                body.extent.per_request().get(),
                body.extent.requests().get()
            ),
            (1, 4),
            "a batch launch is one row per request"
        );
        let body = s.on_body(3, PagedRows::Sequence, 8).unwrap();
        assert_eq!(
            (
                body.extent.per_request().get(),
                body.extent.requests().get()
            ),
            (3, 1),
            "a launch on the wiring's own width re-derives its pair"
        );
        assert!(
            s.on_body(3, PagedRows::Sequence, 6).is_none(),
            "not whole pages"
        );
        assert!(s.on_body(3, PagedRows::Sequence, 0).is_none());
        assert!(
            s.on_body(3, PagedRows::Sequence, 20).is_none(),
            "past the pool"
        );
        assert!(s.on_body(0, PagedRows::Sequence, 8).is_none());
    }

    #[test]
    fn a_step_past_the_request_s_blocks_is_refused() {
        let s = step();
        assert!(
            page_table(
                &s,
                &[2],
                &StepRows {
                    start: 3,
                    real: nz(2)
                }
            )
            .is_err()
        );
        assert!(
            write_rows(
                &s,
                &[2, 9],
                &StepRows {
                    start: 0,
                    real: nz(1)
                }
            )
            .is_err()
        );
    }

    /// A bundle whose attention walks two of the pool's four blocks serves a chunk over a cached
    /// prefix while both sit inside those blocks, and refuses the step that runs past them.
    #[test]
    fn a_step_past_the_bundle_s_walk_is_refused() {
        let s = PagedStep { walk: 2, ..step() };
        assert_eq!(s.reach(), 8);
        let blocks = [3u32, 1, 0];
        assert!(
            page_table(
                &s,
                &blocks,
                &StepRows {
                    start: 5,
                    real: nz(3)
                }
            )
            .is_ok()
        );
        let past = StepRows {
            start: 6,
            real: nz(3),
        };
        assert!(page_table(&s, &blocks, &past).is_err());
        assert!(write_rows(&s, &blocks, &past).is_err());
    }

    #[test]
    fn cached_rows_land_on_their_pages() {
        let s = step();
        let hd = 2;
        // Five cached tokens, token t's values t·10 + (head·2 + d).
        let cache: Vec<f32> = (0..5)
            .flat_map(|t| (0..4).map(move |j| (t * 10 + j) as f32))
            .collect();
        let pool = pool_of_rows(&s, hd, &[3, 1], &cache, 5).unwrap();
        // Token 4 is block 1 (page 1) slot 0: head 1 at pool row (1·2 + 1)·4 + 0 = 12.
        assert_eq!(&pool[12 * hd..13 * hd], &[42.0, 43.0]);
        // Token 0 is page 3 slot 0, head 0 at pool row 24.
        assert_eq!(&pool[24 * hd..25 * hd], &[0.0, 1.0]);
    }
}
