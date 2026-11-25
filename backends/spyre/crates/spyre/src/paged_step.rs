// SPDX-License-Identifier: Apache-2.0
//! THE PAGED STEP'S HOST STAGING — spyre-inference's own attention operands one forward of a paged
//! bundle reads besides the pools ([`PagedStep`]), built from a request's KV blocks and the step's
//! positions as their builder builds them (`SpyreAttentionMetadataBuilder.build`,
//! `_build_query_row_tables`, `SpyreHeadMajorAttentionImpl.kv_write_index` / `_kv_row_pool`,
//! spyre-inference 297fc48):
//!
//! * the PAGE INDEX TABLE `[walk, INT32_ELEMS_PER_STICK]` int32 — `page_index_tables_cpu`:
//!   `table[:, 0] = block_table[s, anchor..anchor + walk]` (the walk's own blocks — see the anchor
//!   law below), the rest of each row zero. Blocks past the request's own are the block table's own
//!   zero padding; their tiles are fully masked;
//! * the MASK STACK `[walk, rows, page_slots]` — `attention_mask_stacks`: 0 where a query row may
//!   attend the key, `finfo(f16).min` where it may not — past the sequence (`kv_pos < seq_len`),
//!   past the row's causal limit `context_len + q_pos`, and, where the op's mask states one, before
//!   the row's window opens (`kv_pos >= pos - window + 1`, `_build_single_tile`'s per-query
//!   `window_start`), a padded row clamped to the last real one. A windowed walk ANCHORS at the
//!   earliest query's reach ([`first_active`] — the reference's active-block staging,
//!   `first_active = earliest_window_start // block_size`), so its trips read the window's blocks
//!   however long the context;
//! * the KV WRITE INDEX — `kv_write_index`: one `[rows]` index per kv head, one after another, head
//!   `h` of the token at [`PagedStep::cache_row`] — the folded-cache row law's one statement;
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

use crate::wiring::{BatchStep, PagedStep, SequenceStep, StepKind};

/// The page walk's laws, where the bake and the worker that state a bundle's walk reach them: the
/// page size, a rung's walk ([`walk_pages`]) and the rungs a pool is walked by ([`walk_buckets`]).
pub use scratchy_triton_splice::{
    BatchChunking, PagedRows, SchedulerBlockTokens, WalkBlocks, walk_buckets, walk_pages,
    window_walk_blocks,
};

/// `finfo(float16).min` — spyre-inference's masked additive value: finite, so a fully-masked tile's
/// row maximum stays finite and its weights vanish under the running maximum.
pub const MASKED: f32 = -65504.0;

/// spyre-inference's `_NULL_SLOT` (`attn_layer.py`): where a padded token's KV write lands — slot 0
/// of the null page, which no request holds (vLLM's block pool keeps block 0 out of every
/// allocation; [`page_of_block`] keeps it out of the scheduler's here).
pub const NULL_SLOT: usize = 0;

/// ⭐ A POOL PAGE — the page of the KV pool a request's block lives on. Minted only by
/// [`Self::of_block`] (a scheduler block id `b` is page `b + 1`) and as [`Self::NULL`], so a raw
/// block id can never stand where a page is read: page 0 is the null page ([`NULL_SLOT`]) — vLLM's
/// block pool reserves its block 0 and this scheduler's does not, so the worker offers it
/// [`PagedStep::host_blocks`] blocks and keeps page 0 back.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PoolPage(u32);

impl PoolPage {
    /// The null page — where a padded row's KV write lands, held by no request.
    pub const NULL: PoolPage = PoolPage(0);

    /// The page scheduler block `b` lives on.
    pub const fn of_block(b: u32) -> PoolPage {
        PoolPage(b + 1)
    }

    /// The page's index in the pool.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// ⭐ A REQUEST'S PAGES — its scheduler block table, every block as its [`PoolPage`], in order. THE
/// ONE mapping of a block table onto the pool: every front that stages a request's step (the card's
/// prompt chunks and decode steps, the emulator's) reads its pages through this.
pub fn request_pages(block_table: &[usize]) -> Vec<PoolPage> {
    block_table
        .iter()
        .map(|&b| PoolPage::of_block(b as u32))
        .collect()
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

/// The rows of one forward: `real` tokens at positions `start..start + real`, in a chunk
/// [`width`](Self::width) wide — the rung's own row count, which `real` never exceeds: the mint
/// refuses it. A row past the real ones repeats the last real token ([`Self::live`]'s law), so
/// its forward, its mask and its KV write are the last real token's.
///
/// ⭐ THE MINT REFUSES A BROKEN STAGING — a chunk of zero rows, a step of zero real rows, or
/// `real` rows in a narrower chunk. Each names a staging the runtime could only mis-stage, which
/// the caller names at its own door; the count the type carries is the count the staging meant,
/// and the width it carries is the width it is staged against — a fact of the type, stated once
/// at the mint.
///
/// ⭐ THE PAD LAW IS THE TYPE'S, ONCE. [`Self::live`] delegates to
/// [`ktir_superdsc::sdsc_abstract::ChunkRows::row_logical_pos`] — the ONE statement of the
/// chunk's pad-row map (Kani-pinned `prefill_pad_row_holds_last_real_token`) — and every front
/// (the card worker, the emu forward, the bake's gate) reads it here. A front that hand-rolls
/// `min(r, real − 1)` or a copy-from-last-row rewrite is re-deriving a law the type already
/// carries: that is a defect, not a staging.
#[derive(Clone, Copy, Debug)]
pub struct StepRows {
    start: usize,
    chunk: ktir_superdsc::sdsc_abstract::ChunkRows,
    real: std::num::NonZeroUsize,
}

impl StepRows {
    /// Declare `real` rows of a chunk `width` wide from `start`. `None` for a staging the type
    /// refuses: a chunk of zero rows, a step of zero real rows, or `real` rows in a narrower
    /// chunk — each a broken staging the caller names at its own door.
    pub fn new(width: usize, start: usize, real: usize) -> Option<StepRows> {
        let chunk = ktir_superdsc::sdsc_abstract::ChunkRows::chunk(
            u32::try_from(width)
                .ok()
                .and_then(std::num::NonZeroU32::new)?,
        );
        let real = std::num::NonZeroUsize::new(real)?;
        if real.get() > chunk.count().get() as usize {
            return None;
        }
        Some(StepRows { start, chunk, real })
    }

    /// One row at `start` — the literal-row mint, `const` so no `expect` stands where the value
    /// is a literal. `NonZeroUsize::MIN` is 1, the count one row is.
    pub const fn one_at(start: usize) -> StepRows {
        StepRows {
            start,
            chunk: ktir_superdsc::sdsc_abstract::ChunkRows::chunk(std::num::NonZeroU32::MIN),
            real: std::num::NonZeroUsize::MIN,
        }
    }

    /// ⭐ THE PAD LAW — the live row row `r` reads its token, its rope and its write from:
    /// [`ChunkRows::row_logical_pos`]'s, the one statement, never re-derived. A row past the
    /// real ones carries the last real row's values.
    pub fn live(&self, r: usize) -> usize {
        self.chunk.row_logical_pos(r, self.real.get())
    }

    /// The last real row — [`Self::live`]'s bound, the row every pad row reads.
    pub fn last_live(&self) -> usize {
        self.real.get() - 1
    }

    /// The sequence position row `r` carries.
    pub fn pos(&self, r: usize) -> usize {
        self.start + self.live(r)
    }

    /// The chunk's width — the rung's row count the rows are staged against.
    pub fn width(&self) -> usize {
        self.chunk.count().get() as usize
    }

    /// The real-row count — a count for arithmetic only (totals, seed widths), never a law to
    /// re-derive from: the law is [`Self::live`].
    pub fn real(&self) -> usize {
        self.real.get()
    }

    /// The step's first position.
    pub fn start(&self) -> usize {
        self.start
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
/// and every block is a page of the pool. The rows' own width — that they are no wider than the
/// window they are staged against — is a fact of [`StepRows`]'s mint, not a check here.
fn check(step: &PagedStep, blocks: &[PoolPage], rows: &StepRows) -> Result<(), PagedStepError> {
    let slots = step.page_slots as usize;
    let need = rows.end().div_ceil(slots);
    if blocks.len() < need {
        return Err(PagedStepError(format!(
            "positions up to {} need {need} block(s) of {slots} slots; the request holds {}",
            rows.end(),
            blocks.len()
        )));
    }
    if need > step.pages.get() as usize {
        return Err(PagedStepError(format!(
            "positions up to {} need {need} block(s); the pool holds {} page(s)",
            rows.end(),
            step.pages.get()
        )));
    }
    // The attention reads only the blocks its walk covers, so a step whose span exceeds the
    // walk's reach would attend without its keys. A causal step's span is its whole end — the
    // law it has always been; a windowed step's is its window's sweep, whatever its position.
    let span = step.span(rows);
    if span > step.reach() {
        return Err(PagedStepError(format!(
            "positions up to {} sweep a {}-position span — past this bundle's {}-position reach \
             (its attention walks {} block(s))",
            rows.end(),
            span,
            step.reach(),
            step.walk.get()
        )));
    }
    if let Some(p) = blocks
        .iter()
        .map(|p| p.get())
        .find(|&p| p >= step.pages.get())
    {
        return Err(PagedStepError(format!(
            "block page {p} is past the pool's {} page(s)",
            step.pages.get()
        )));
    }
    Ok(())
}

/// `INT32_ELEMS_PER_STICK` — a page index table row's width.
pub const INT32_ELEMS_PER_STICK: usize =
    <ktir_superdsc::superdsc_opspec::SenUint32 as ktir_superdsc::superdsc_opspec::DataFormat>::ELEMS_PER_STICK
        as usize;

/// ⭐ THE QUERY'S REACH — the earliest position the query at `pos` attends: every position for a
/// causal op; the last `window` positions (its own included) for one whose mask states a window.
/// The reference's per-query `window_start` (`(abs_q_pos - W + 1).clamp(min=0)`), the one law the
/// mask stack's cutoffs and the walk's anchor both read — never re-derived.
fn window_start(step: &PagedStep, pos: usize) -> usize {
    match step.window {
        None => 0,
        Some(w) => pos.saturating_sub(w.get() as usize - 1),
    }
}

/// ⭐ THE BLOCK A QUERY AT `pos` ANCHORS ITS WALK AT — the block holding its reach
/// ([`window_start`]): block 0 for a causal op, and for a window whose reach opens before the
/// sequence does. THE ONE DERIVATION of an anchor: a sequence step's ([`first_active`], at its
/// earliest query) and a decode batch's per-sequence ([`batch_sources`], at each sequence's one
/// query) both read this, never a second copy of the arithmetic.
fn anchor_of(step: &PagedStep, pos: usize) -> usize {
    window_start(step, pos) / step.page_slots as usize
}

/// ⭐ THE WALK'S ANCHOR — the block the walk's trip 0 reads: [`anchor_of`] the step's EARLIEST
/// query (`rows.start` — never the latest's, so a prompt chunk's early queries keep their
/// in-window blocks; the reference's `first_active`). The mask stack's frame, the page table's
/// rows and the walk check read this.
pub fn first_active(step: &PagedStep, rows: &StepRows) -> usize {
    anchor_of(step, rows.start)
}

/// ⭐ THE MASK CELL — the additive mask value of the query at `pos` against the key at absolute
/// position `kv`: 0 inside the query's reach (at or before `pos`, at or after [`window_start`] at
/// it), [`MASKED`] everywhere else. THE ONE statement of the mask's value law: the sequence stack
/// ([`mask_stack_of`]) and the decode batch's chunk masks ([`batch_sources`]) both read it.
fn mask_cell(step: &PagedStep, pos: usize, kv: usize) -> f32 {
    if kv <= pos && kv >= window_start(step, pos) {
        0.0
    } else {
        MASKED
    }
}

/// The page index table: `[walk, INT32_ELEMS_PER_STICK]` int32, row `i` holding the walk's block
/// `i`'s page at column 0 — `table[:, 0] = block_table[s, anchor..anchor + walk]`, the block
/// table's own zeros past the request's blocks. A causal walk anchors at block 0, so its rows
/// are the blocks from 0 as they always were.
pub fn page_table(
    step: &PagedStep,
    blocks: &[PoolPage],
    rows: &StepRows,
) -> Result<Vec<i32>, PagedStepError> {
    check(step, blocks, rows)?;
    let anchor = first_active(step, rows);
    let mut t = vec![0i32; step.walk.get() as usize * INT32_ELEMS_PER_STICK];
    for (i, &page) in blocks
        .iter()
        .skip(anchor)
        .take(step.walk.get() as usize)
        .enumerate()
    {
        t[i * INT32_ELEMS_PER_STICK] = page.get() as i32;
    }
    Ok(t)
}

/// The mask stack: `[walk, q, page_slots]`, tile `i` column `c` keyed by the ABSOLUTE position
/// `(anchor + i) · page_slots + c` of the block walk trip `i` reads — 0 where the query row's
/// reach holds it (at or before the row's position, at or after [`window_start`] at it),
/// [`MASKED`] everywhere else. `q` is the row count one request's stack serves — the graph's
/// rows for one sequence's whole forward ([`step_sources`]'s spelling), the extent's per-request
/// width for one request of a packed rung's ([`card_step_sources`]'s); the two are the same
/// number at one request. A causal step anchors at block 0 and reaches from position 0, so its
/// bytes are the law it has always staged.
pub fn mask_stack(step: &PagedStep, rows: &StepRows) -> Vec<f32> {
    mask_stack_of(step, step.rows() as usize, rows)
}

fn mask_stack_of(step: &PagedStep, q: usize, rows: &StepRows) -> Vec<f32> {
    let slots = step.page_slots as usize;
    let anchor = first_active(step, rows);
    let mut m = vec![MASKED; step.walk.get() as usize * q * slots];
    for i in 0..step.walk.get() as usize {
        for r in 0..q {
            let pos = rows.pos(r);
            let base = (i * q + r) * slots;
            for c in 0..slots {
                m[base + c] = mask_cell(step, pos, (anchor + i) * slots + c);
            }
        }
    }
    m
}

/// The KV write index: one `[rows]` index per kv head, one after another — head `h` of row `r`'s
/// token at [`PagedStep::cache_row`], at the row's position.
pub fn write_rows(
    step: &PagedStep,
    blocks: &[PoolPage],
    rows: &StepRows,
) -> Result<Vec<i32>, PagedStepError> {
    check(step, blocks, rows)?;
    let (nkvh, q) = (step.nkvh().get() as usize, step.rows() as usize);
    let mut w = Vec::with_capacity(nkvh * q);
    for h in 0..nkvh {
        for r in 0..q {
            w.push(step.cache_row(blocks, rows.pos(r), h) as i32);
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
            rows: step.rows(),
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
    (0..(step.pages.get() * step.nkvh().get()) as i32).collect()
}

/// One step's sources, each at its tid and its shape — the ONE spelling both stagers bind: the
/// mask stacks as f16-channel values (one per window, see [`StepSources::masks`]), and the
/// int32 tables as little-endian words (their views are int32, so they ride the typed channel).
/// `kv_row_pool` rides only where the decode kernel reads it
/// ([`PagedStep::reads_kv_row_pool`]).
pub struct StepSources {
    /// ⭐ EVERY MASK THE STEP STAGES, ONE PER WINDOW — a packed prefill stages one mask stack
    /// per request at its own window tid; every other step stages exactly one. The list is the
    /// one spelling (the card's [`CardSources::masks`] is its transform); a one-mask step is a
    /// one-entry list.
    pub masks: Vec<(u32, Vec<f32>, Vec<usize>)>,
    pub int32: Vec<(u32, Vec<u8>, Vec<usize>)>,
}

/// ⭐ A SEQUENCE STEP'S SOURCES, STAGED FOR ITS LAUNCH'S REQUESTS — the ONE logical law
/// ([`card_step_sources`] is its card transform; the bake's numeric gate binds it as the
/// emulator stages). Request `r`'s window stages AT ITS OWN WINDOW TIDS
/// ([`scratchy_triton_splice::paged_window_of_class`], window 0's being the step's own
/// sentinels): the page index table (`[walk, INT32_ELEMS_PER_STICK]`, request `r`'s page ids
/// one block per trip from its own anchor), the mask stack (`[walk·w, page_slots]`, `w` the
/// extent's per-request width — one request's rows, keyed by its own absolute positions), and
/// the query row index (the consecutive window `r·w .. r·w+w`, [`QueryRowIndex::of_request`]).
/// The KV write index is ONE table for the whole graph, head-major: row `r·w + j` (request
/// `r`'s row `j`) names its OWN request's cache slot at its own position, a pad row clamped to
/// its request's last real one — the KV writers run once per step over every row, they are not
/// per-window.
///
/// ⭐ THE EMPTY-SLOT LAW — a launch seating fewer requests than the rung packs stages the
/// windows past them as NO request's: page table rows name page 0 (the null page,
/// [`page_of_block`]'s own reserve), the mask stack is fully [`MASKED`] (a finite all-masked
/// tile whose weights vanish under the running maximum), and the KV write index lands on
/// [`NULL_SLOT`] — the empty slot's forward runs, but nothing it writes reaches a request's
/// cache and nothing it answers is read back. The requests are dense from slot 0; an empty or
/// over-capacity launch does not exist to stage.
///
/// At one request nothing moves: window 0's tids ARE the step's own sentinels, `w` is the
/// graph's rows, and every table is exactly the one-sequence law's bytes
/// ([`request_sources`]'s spelling), byte for byte, at the tids they have always staged at.
pub fn step_sources(
    step: &SequenceStep,
    reqs: &[RequestStep<'_>],
) -> Result<StepSources, PagedStepError> {
    let words = |w: Vec<i32>| -> Vec<u8> { w.into_iter().flat_map(i32::to_le_bytes).collect() };
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
        check(step, req.blocks, &req.rows)
            .map_err(|e| PagedStepError(format!("request {r}: {}", e.0)))?;
    }
    let (walk, slots, nkvh) = (
        step.walk.get() as usize,
        step.page_slots as usize,
        step.nkvh().get() as usize,
    );
    // ⭐ THE WINDOW FAMILY — one window per request the rung packs, window `r`'s three tables
    // at its own tids, each exactly the one-request size.
    let mut masks = Vec::with_capacity(cap);
    let mut int32 = Vec::with_capacity(cap * 2 + 1);
    for r in 0..cap {
        let tid = |table: scratchy_triton_splice::WindowTable| {
            scratchy_triton_splice::paged_window_of_class(step.class.index(), r as u32, table)
        };
        // THE PAGE INDEX TABLE — request `r`'s page ids one block per trip; an empty window's
        // rows stay zero — page 0, the null page.
        let table = match reqs.get(r) {
            Some(req) => page_table(step, req.blocks, &req.rows)?,
            None => vec![0i32; walk * INT32_ELEMS_PER_STICK],
        };
        // THE MASK STACK — request `r`'s stack over its own rows, or the fully-masked tile of
        // an empty window.
        let mask = match reqs.get(r) {
            Some(req) => mask_stack_of(step, w, &req.rows),
            None => vec![MASKED; walk * w * slots],
        };
        // THE QUERY ROW INDEX — window `r`'s consecutive rows, live or empty (the graph's rows
        // are real however many requests ride the launch, and the seat clamps a pad row to its
        // request's last real one).
        let query = QueryRowIndex::of_request(step, r as u32).words();
        use scratchy_triton_splice::WindowTable as W;
        int32.push((
            tid(W::PageTable),
            words(table),
            vec![walk, INT32_ELEMS_PER_STICK],
        ));
        int32.push((tid(W::QueryRows), words(query), vec![w]));
        masks.push((tid(W::Mask), mask, vec![walk * w, slots]));
    }
    // ⭐ THE KV WRITE INDEX — packed over the graph's rows, head-major (see this fn's doc).
    let mut write: Vec<i32> = Vec::with_capacity(nkvh * cap * w);
    for h in 0..nkvh {
        for r in 0..cap {
            match reqs.get(r) {
                Some(req) => {
                    for j in 0..w {
                        write.push(step.cache_row(req.blocks, req.rows.pos(j), h) as i32);
                    }
                }
                None => write.extend(std::iter::repeat_n(step.null_cache_row(h) as i32, w)),
            }
        }
    }
    int32.push((step.write_rows(), words(write), vec![nkvh * cap * w]));
    if step.reads_kv_row_pool() {
        int32.push((
            step.kv_row_pool(),
            words(kv_row_pool(step)),
            vec![step.pages.get() as usize, nkvh],
        ));
        // The add identity the chained index's run copy reads on the card.
        let run = ktir_superdsc::emit::lower_ktir_to_superdsc::INDEX_RUN_WORDS as usize;
        int32.push((step.int32_zero(), words(vec![0; run]), vec![1, run]));
    }
    Ok(StepSources { masks, int32 })
}

/// One sequence of a decode batch: its KV blocks, and the position of the one token this step
/// decodes (its context before the step).
#[derive(Clone, Copy, Debug)]
pub struct BatchSeq<'a> {
    pub blocks: &'a [PoolPage],
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
/// * the mask is 0 where a sequence's key is inside its one query's reach — at or before its
///   position, at or after its window opens — and `finfo(f16).min` past it, on the block slots
///   past its touched blocks, and a padded sequence's block 0 `finfo.min` (so its softmax stays
///   finite); one row per (block slot, sequence), which every kv head reads through the kernel's
///   broadcast, as their builder's `expand` over the kv axis does. A windowed sequence's chunk
///   pages stage ITS OWN blocks from its own anchor, so a batch's sequences anchor independently.
///
/// A padded row's KV write lands on [`NULL_SLOT`], as spyre-inference's padded tokens' do: its
/// attention reads page 0 under the padded sequence's mask and answers nothing anyone reads, so what
/// it writes must reach no request's page.
pub fn batch_sources(
    step: &BatchStep,
    seqs: &[BatchSeq<'_>],
) -> Result<StepSources, PagedStepError> {
    let chunking = step.chunking();
    // ⛔ THE LIVE SEQUENCES ARE ROWS `0..n`, THE PADDING `n..b`. The kernel stores its output as the
    // first `b` rows of `out`, row `s` sequence `s`'s (`out[:num_seqs].copy_(attn)`), and the
    // worker reads sequence `s`'s token from launch slot `s`; this staging is what makes the two
    // the same rows — `rep` below names sequence `s` at row `s` and a pad row as row 0 (#1038).
    let (b, n) = (step.rows() as usize, seqs.len());
    if n == 0 || n > b {
        return Err(PagedStepError(format!(
            "a decode batch of {n} sequence(s) in a {b}-sequence bundle"
        )));
    }
    let (slots, nkvh) = (step.page_slots as usize, step.nkvh().get() as usize);
    let (j, c) = (
        chunking.blocks_per_chunk as usize,
        chunking.num_chunks as usize,
    );
    let padded = c * j;
    // ⭐ EACH SEQUENCE'S OWN SPAN, in blocks, stated ONCE — from [`anchor_of`] its one query to its
    // position: every block it holds for a causal sequence, the window's sweep for a windowed one,
    // whatever the position. The check below and the staging after it both read this list.
    let spans: Vec<(usize, usize)> = seqs
        .iter()
        .map(|q| {
            let anchor = anchor_of(step, q.pos);
            (anchor, q.pos / slots - anchor + 1)
        })
        .collect();
    for (s, (q, &(anchor, touched))) in seqs.iter().zip(&spans).enumerate() {
        let need = (q.pos + 1).div_ceil(slots);
        if touched > step.walk.get() as usize || q.blocks.len() < need {
            return Err(PagedStepError(format!(
                "sequence {s} at position {} touches {touched} block(s) from its anchor \
                 {anchor}; it holds {} and the bundle walks {}",
                q.pos,
                q.blocks.len(),
                step.walk.get()
            )));
        }
        if let Some(p) = q.blocks[..need]
            .iter()
            .map(|p| p.get())
            .find(|&p| p >= step.pages.get())
        {
            return Err(PagedStepError(format!(
                "sequence {s}: block page {p} is past the pool's {} page(s)",
                step.pages.get()
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
    // mask_bs_bb `[B, C·J, slots]`, filled `-inf` as theirs is (`spyre_attn.py:1123`): a block slot
    // past every touched one contributes zero; the touched tiles take [`mask_cell`]'s values.
    let mut mask_bs = vec![f32::NEG_INFINITY; b * padded * slots];
    for (s, (q, &(anchor, touched))) in seqs.iter().zip(&spans).enumerate() {
        // The sequence's walk anchors at its own query's reach and stages the blocks it touches
        // from there; the block slots past them stay the null page under fully-masked rows.
        for i in 0..touched {
            pages[i * b + s] = q.blocks[anchor + i].get() as i32;
            for col in 0..slots {
                mask_bs[(s * padded + i) * slots + col] =
                    mask_cell(step, q.pos, (anchor + i) * slots + col);
            }
        }
    }
    // A row past the batch is -inf in every block slot, so its softmax would be NaN: its block 0
    // is [`MASKED`] (theirs, `mask_bs_bb[n:, 0] = finfo.min`, `spyre_attn.py:1138`).
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
            let row = match seqs.get(r) {
                Some(q) => step.cache_row(q.blocks, q.pos, h),
                None => step.null_cache_row(h),
            };
            write.push(row as i32);
        }
    }
    let e = j * b;
    Ok(StepSources {
        masks: vec![(step.chunk_masks(), chunk_masks, vec![c * e, slots])],
        int32: vec![
            (step.rep_rows(), words(fold(&rep)), vec![e * per as usize]),
            (
                step.chunk_pages(),
                words(fold(&pages)),
                vec![c * e * per as usize],
            ),
            (step.write_rows(), words(write), vec![nkvh * b]),
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

pub struct CardSources {
    /// ⭐ EVERY MASK THE STEP STAGES, ONE PER TABLE — a packed prefill stages one mask stack per
    /// window at its own window tid; every other step stages exactly one. A single `(tid, values)`
    /// pair would be a second `mask` field the packed path grows around, so the list is the one
    /// spelling and a one-mask step is a one-entry list.
    pub masks: Vec<(u32, Vec<f32>)>,
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
/// request `r`'s tables stage at window `r`'s own tids
/// ([`ktir_superdsc::reserved_tids::paged_window_tid`]), and the slots past the live ones (the
/// rung's capacity is the count, the launch's live requests the fill) stage the empty-slot laws —
/// an empty slot's writes reach no request's cache.
#[derive(Clone, Copy, Debug)]
pub struct RequestStep<'a> {
    pub blocks: &'a [PoolPage],
    pub rows: StepRows,
}

/// [`step_sources`] on the card: the logical tables' card transforms — every window's page
/// index table PITCHED ONE WORD PER TRIP at the card's walk step (the bytes one trip moves the
/// mask tile; the logical row's remaining zero words ride only the host view), every mask stack
/// stick-major via [`card_mask`] and padded to the card window's elements, the word tables
/// (the query rows, the write index) verbatim. The pool's STATICS — `kv_row_pool` and the
/// int32 zero the logical stager binds where the decode kernel reads them — are DROPPED here:
/// the card stages them once at load ([`card_statics`]), never per step.
pub fn card_step_sources(
    step: &SequenceStep,
    reqs: &[RequestStep<'_>],
) -> Result<CardSources, PagedStepError> {
    let logical = step_sources(step, reqs)?;
    let (walk, cap) = (
        step.walk.get() as usize,
        step.extent.requests().get() as usize,
    );
    let trip = step.card_walk_step() as usize;
    // The window page-table tids — the one family the card pitches.
    let page_tables: std::collections::HashSet<u32> = (0..cap as u32)
        .map(|r| {
            scratchy_triton_splice::paged_window_of_class(
                step.class.index(),
                r,
                scratchy_triton_splice::WindowTable::PageTable,
            )
        })
        .collect();
    let int32 = logical
        .int32
        .into_iter()
        .filter(|(tid, ..)| *tid != step.kv_row_pool() && *tid != step.int32_zero())
        .map(|(tid, bytes, _)| {
            if !page_tables.contains(&tid) {
                return (tid, bytes);
            }
            // ⭐ ONE WORD PER TRIP — the logical row's first word (its page id, little-endian)
            // at trip `i`'s stride; the card holds the table one block per trip.
            let mut card = vec![0u8; walk * trip];
            for i in 0..walk {
                let src = i * INT32_ELEMS_PER_STICK * 4;
                if src + 4 <= bytes.len() {
                    card[i * trip..i * trip + 4].copy_from_slice(&bytes[src..src + 4]);
                }
            }
            (tid, card)
        })
        .collect();
    let masks = logical
        .masks
        .into_iter()
        .map(|(tid, mask, shape)| {
            let mut card = card_mask(&mask, &shape);
            card.resize(step.card_window_mask_elems(), MASKED);
            (tid, card)
        })
        .collect();
    Ok(CardSources { masks, int32 })
}

/// [`batch_sources`] on the card: a decode batch's step, its chunk page ids one chunk per trip.
pub fn card_batch_sources(
    step: &BatchStep,
    seqs: &[BatchSeq<'_>],
) -> Result<CardSources, PagedStepError> {
    let logical = batch_sources(step, seqs)?;
    let rows = step.chunking().index_rows() as usize;
    let int32 = logical
        .int32
        .into_iter()
        .map(|(tid, bytes, _)| {
            if tid == step.chunk_pages() {
                (
                    tid,
                    pitched(&bytes, rows * 4, step.card_walk_step() as usize),
                )
            } else {
                (tid, bytes)
            }
        })
        .collect();
    let masks = logical
        .masks
        .into_iter()
        .map(|(tid, mask, shape)| (tid, card_mask(&mask, &shape)))
        .collect();
    Ok(CardSources { masks, int32 })
}

/// ⭐ ONE REQUEST'S STEP ON EITHER KIND OF BUNDLE — its prompt rows on a sequence bundle
/// ([`step_sources`]), or its one decode row on a decode bundle, which spyre-inference decodes with
/// the batched kernel at one sequence ([`batch_sources`] of the request alone: its sequence buckets
/// start at 1). The bundle states which it is ([`PagedStep::paged_rows`]); a decode bundle takes
/// one row of a request, its token at `rows.start`.
pub fn request_sources(
    step: &PagedStep,
    blocks: &[PoolPage],
    rows: &StepRows,
) -> Result<StepSources, PagedStepError> {
    match step.kind() {
        StepKind::Sequence(step) => step_sources(
            &step,
            &[RequestStep {
                blocks,
                rows: *rows,
            }],
        ),
        StepKind::Batch(step) if rows.real.get() == 1 => batch_sources(
            &step,
            &[BatchSeq {
                blocks,
                pos: rows.start,
            }],
        ),
        StepKind::Batch(_) => Err(PagedStepError(format!(
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
    let nkvh = step.nkvh().get() as usize;
    if nkvh > run {
        return Err(PagedStepError(format!(
            "a page's {nkvh} kv-head rows do not fit its {run}-word index run"
        )));
    }
    let mut words = vec![0i32; step.pages.get() as usize * run];
    for (p, page) in kv_row_pool(step).chunks(nkvh).enumerate() {
        words[p * run..p * run + nkvh].copy_from_slice(page);
    }
    Ok(vec![
        (step.kv_row_pool(), le_words(words)),
        (step.int32_zero(), vec![0u8; run * 4]),
    ])
}

/// A pool holding token-major cache rows: row `t` of `cache` (`[·, nkvh · hd]`) laid into its page —
/// head `h` at pool row [`PagedStep::cache_row`] — for every `t < valid`; every other row zero. Returned as the pool's PREFIX through the last page a
/// row lands on (empty when `valid` is 0): a pool is the model's whole context, and the rows past
/// that page are the zeroed allocation the session already holds. The bake's numeric gate stages its
/// reference's cached keys this way, so the paged walk reads the very rows the reference attends.
pub fn pool_of_rows(
    step: &PagedStep,
    hd: usize,
    blocks: &[PoolPage],
    cache: &[f32],
    valid: usize,
) -> Result<Vec<f32>, PagedStepError> {
    let (slots, nkvh) = (step.page_slots as usize, step.nkvh().get() as usize);
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
    if let Some(p) = blocks[..need]
        .iter()
        .map(|p| p.get())
        .find(|&p| p >= step.pages.get())
    {
        return Err(PagedStepError(format!(
            "block page {p} is past the pool's {} page(s)",
            step.pages.get()
        )));
    }
    let pages = blocks[..need]
        .iter()
        .map(|p| p.get() as usize + 1)
        .max()
        .unwrap_or(0);
    let mut pool = vec![0f32; pages * slots * width];
    for t in 0..valid {
        for h in 0..nkvh {
            let dst = step.cache_row(blocks, t, h) * hd;
            let src = t * width + h * hd;
            pool[dst..dst + hd].copy_from_slice(&cache[src..src + hd]);
        }
    }
    Ok(pool)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scratchy_triton_splice::WindowTable as W;

    /// A fixture's walk of `n` blocks.
    fn wb(n: u32) -> WalkBlocks {
        WalkBlocks::new(nz32(n))
    }

    /// A fixture's pool of `n` pages.
    fn pool(n: u32) -> crate::pool_pages::PoolPages {
        crate::pool_pages::PoolPages::of_nonzero(nz32(n))
    }

    /// A sequence fixture's step, through the one mint ([`PagedStep::kind`]).
    fn seq(s: PagedStep) -> SequenceStep {
        match s.kind() {
            StepKind::Sequence(q) => q,
            StepKind::Batch(_) => panic!("the fixture is a sequence's step"),
        }
    }

    /// A decode-batch fixture's step, through the one mint ([`PagedStep::kind`]).
    fn bat(s: PagedStep) -> BatchStep {
        match s.kind() {
            StepKind::Batch(b) => b,
            StepKind::Sequence(_) => panic!("the fixture is a decode batch's step"),
        }
    }

    /// A fixture's pool pages, by page number — each minted through the type's own constructors:
    /// page 0 is [`PoolPage::NULL`], page `p` is the page of scheduler block `p − 1`.
    fn pages(numbers: &[u32]) -> Vec<PoolPage> {
        numbers
            .iter()
            .map(|&p| match p.checked_sub(1) {
                Some(b) => PoolPage::of_block(b),
                None => PoolPage::NULL,
            })
            .collect()
    }

    fn nz32(n: u32) -> std::num::NonZeroU32 {
        std::num::NonZeroU32::new(n).unwrap()
    }

    /// One request of a fixture's launch — `blocks` shared, `rows` its own.
    fn req(blocks: &[PoolPage], rows: StepRows) -> RequestStep<'_> {
        RequestStep { blocks, rows }
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
        // Class 0, so the reserved sentinels (`PagedStep::tids`, `PagedStepTids::RESERVED`) — window
        // 0's tids in the card stagers and the placements ARE these, which is what keeps a packed
        // rung byte-identical at one request.
        PagedStep {
            view: crate::wiring::ContextPages::new(nz32(4)),
            pages: pool(4),
            page_slots: 4,
            // Two kv heads of one query head each, one stick a head: the fold's own floor law
            // (`BatchChunking::of`) leaves these fixtures at their kv-head cut.
            heads: crate::wiring::HeadGeometry::new(
                crate::wiring::QueryHeads::new(2),
                crate::wiring::KvHeads::new(2),
                crate::wiring::HeadDim::new(64),
                crate::wiring::RotDim::new(64),
                crate::wiring::BlockSize::new(16),
            ),
            walk: wb(4),
            paged_rows: PagedRows::Sequence,
            extent: pair(3, 1),
            // The fixture's op is causal — its mask states no window — and it is the tape's
            // only class, so its tids are the sentinels.
            window: None,
            class: crate::wiring::AttnClass::FIRST,
        }
    }

    /// A 4-sequence batch rung over a 16-page pool of 4-slot pages, walking 2 blocks: two real
    /// sequences (positions 5 and 2) and two padded rows, at the chunking `of(4, 2)` = J 2, C 1.
    fn batch() -> PagedStep {
        PagedStep {
            view: crate::wiring::ContextPages::new(nz32(16)),
            pages: pool(16),
            walk: wb(2),
            paged_rows: PagedRows::Batch,
            extent: pair(1, 4),
            ..step()
        }
    }

    /// ⭐ A RUNG'S SWEEP STAGES A WALK ONLY IN WHOLE PAGES OF THE VIEW — the law the bake mints
    /// every rung's per-class walk by (`PagedStep::staged_walk`): one page, several, the whole view,
    /// and each sweep's walk the pages it reaches; half a page, a zero sweep, or a sweep past the
    /// view is no walk, and no rung carries one.
    #[test]
    fn a_rung_sweep_stages_a_walk_in_whole_pages_of_the_view() {
        // `step()`: a 4-page view of 4-slot pages.
        let s = step();
        for (swept, walk) in [(4, 1), (8, 2), (12, 3), (16, 4)] {
            assert_eq!(s.staged_walk(swept), Some(wb(walk)), "sweep {swept}");
        }
        for swept in [0, 1, 6, 20, 32] {
            assert_eq!(s.staged_walk(swept), None, "sweep {swept}");
        }
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
        let (a, b) = (pages(&[7, 3]), pages(&[9]));
        let src = batch_sources(
            &bat(s),
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
        assert_eq!(words(s.rep_rows()), vec![0, 1, 0, 0, 0, 1, 0, 0]);
        // `[C·J, B]`: slot 0 = (7, 9, 0, 0), slot 1 = (3, 0, 0, 0).
        assert_eq!(words(s.chunk_pages()), vec![7, 9, 0, 0, 3, 0, 0, 0]);
        // Row r's token, head h at ((page · 2 + h) · 4 + slot); the two padded rows at the null
        // slot, page 0.
        let w = words(s.write_rows());
        assert_eq!(&w[..4], &[3 * 8 + 1, 9 * 8 + 2, 0, 0]);
        assert_eq!(&w[4..], &[3 * 8 + 4 + 1, 9 * 8 + 4 + 2, 4, 4]);
        let (tid, m, shape) = &src.masks[0];
        // One row per (slot, sequence): every kv head reads it through the kernel's broadcast.
        assert_eq!((*tid, shape.clone()), (s.chunk_masks(), vec![2 * 4, 4]));
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
            walk: wb(1),
            extent: pair(1, 2),
            ..batch()
        };
        let c = s.batch().unwrap();
        assert_eq!(
            (c.blocks_per_chunk, c.num_chunks, c.rows_per_page),
            (2, 1, 2)
        );
        let (a, b) = (pages(&[5]), pages(&[11]));
        let seqs = [
            BatchSeq { blocks: &a, pos: 2 },
            BatchSeq { blocks: &b, pos: 0 },
        ];
        let src = batch_sources(&bat(s), &seqs).expect("stages");
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
        assert_eq!(words(s.rep_rows()), vec![0, 1, 2, 3, 0, 1, 2, 3]);
        // Slot 0 pages (5, 11), slot 1 the padding's page 0, each at `page · 2 + a`.
        assert_eq!(words(s.chunk_pages()), vec![10, 11, 22, 23, 0, 1, 0, 1]);
        let card = card_batch_sources(&bat(s), &seqs).unwrap();
        let got = ints(card.words(s.chunk_pages()).unwrap());
        assert_eq!(got.len(), s.card_walk_step() as usize / 4);
        assert_eq!(&got[..8], &[10, 11, 22, 23, 0, 1, 0, 1]);
        assert!(got[8..].iter().all(|&v| v == 0));
        assert_eq!(
            s.card_placements()
                .iter()
                .find(|(t, ..)| *t == s.rep_rows())
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
            paged_rows: PagedRows::Batch,
            extent: pair(1, 1),
            ..step()
        };
        let blocks = pages(&[2, 3]);
        let got = request_sources(&decode, &blocks, &StepRows::new(1, 5, 1).unwrap()).unwrap();
        let want = batch_sources(
            &bat(decode),
            &[BatchSeq {
                blocks: &blocks,
                pos: 5,
            }],
        )
        .unwrap();
        assert_eq!(got.int32, want.int32);
        assert_eq!(got.masks, want.masks);
        let tids: Vec<u32> = got.int32.iter().map(|(t, ..)| *t).collect();
        assert!(tids.contains(&decode.rep_rows()) && tids.contains(&decode.chunk_pages()));
        assert!(!tids.contains(&decode.page_table()) && !tids.contains(&decode.kv_row_pool()));
        let prompt = step();
        let rows = StepRows::new(3, 5, 2).unwrap();
        assert_eq!(
            request_sources(&prompt, &blocks, &rows).unwrap().int32,
            step_sources(
                &seq(prompt),
                &[RequestStep {
                    blocks: &blocks,
                    rows
                }]
            )
            .unwrap()
            .int32
        );
        assert!(request_sources(&decode, &blocks, &rows).is_err());
    }

    /// A sequence whose position is past the rung's walk cannot run on it.
    #[test]
    fn a_batch_sequence_past_the_walk_is_refused() {
        let s = batch();
        let a = pages(&[1, 2, 3]);
        assert!(batch_sources(&bat(s), &[BatchSeq { blocks: &a, pos: 8 }]).is_err());
    }

    /// A 3-row chunk at positions 5, 6 and a pad row repeating 6, over blocks `[2, 0]`.
    #[test]
    fn the_tables_follow_spyre_inference_s_laws() {
        let s = step();
        let rows = StepRows::new(3, 5, 2).unwrap();
        let blocks = pages(&[2, 0]);
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

    /// ⭐ A WINDOWED STEP STAGES THE REFERENCE'S ACTIVE WINDOW — the walk anchors at the block
    /// the earliest query's reach opens at (the reference's `first_active`), the page table names
    /// the anchored blocks, and the mask stack cuts every row at its own `pos − window + 1` —
    /// `_build_single_tile`'s per-query `window_start` over `_build_active_tiles_with_skip`'s
    /// active blocks. Here: W 6 over 4-slot pages, 3 rows at positions 9, 10, 11, so the earliest
    /// query (position 9) reaches from 4 — block 1 — and every trip's tile is keyed by the
    /// ABSOLUTE positions of the blocks the anchored table names.
    #[test]
    fn a_windowed_step_stages_the_reference_s_active_window() {
        let s = PagedStep {
            window: Some(nz32(6)),
            ..step()
        };
        let rows = StepRows::new(3, 9, 3).unwrap();
        let blocks = pages(&[0, 1, 2, 3]);
        assert_eq!(
            first_active(&s, &rows),
            1,
            "the earliest query reaches from 4"
        );
        assert_eq!(s.span(&rows), 8, "end 12 less the anchor's 4 positions");
        // The table names the walk's own blocks — 1, 2, 3 — then the block table's zero past
        // the request's four.
        let pt = page_table(&s, &blocks, &rows).unwrap();
        let col0: Vec<i32> = pt.iter().step_by(INT32_ELEMS_PER_STICK).copied().collect();
        assert_eq!(col0, vec![1, 2, 3, 0]);
        // The stack: trip i covers the absolute positions (1 + i)·4 + c. Row 0 (position 9)
        // attends [4, 9], row 1 (position 10) [5, 10], row 2 (position 11) [6, 11].
        let m = mask_stack(&s, &rows);
        assert_eq!(&m[0..4], &[0.0; 4], "trip 0 row 0: positions 4..=7");
        assert_eq!(&m[4..8], &[MASKED, 0.0, 0.0, 0.0], "trip 0 row 1: from 5");
        assert_eq!(
            &m[8..12],
            &[MASKED, MASKED, 0.0, 0.0],
            "trip 0 row 2: from 6"
        );
        assert_eq!(
            &m[12..16],
            &[0.0, 0.0, MASKED, MASKED],
            "trip 1 row 0: to 9"
        );
        assert_eq!(&m[16..20], &[0.0, 0.0, 0.0, MASKED], "trip 1 row 1: to 10");
        assert_eq!(&m[20..24], &[0.0; 4], "trip 1 row 2: to 11");
        assert!(
            m[24..].iter().all(|&v| v == MASKED),
            "trips past the window are fully masked"
        );
    }

    /// A WINDOW THAT COVERS THE SEQUENCE IS CAUSAL — the earliest query's reach opens at or
    /// before position 0, so the anchor is block 0 and every row's reach is the whole sequence:
    /// the anchored table and the windowed stack are the causal ones, byte for byte.
    #[test]
    fn a_window_past_the_sequence_s_start_is_the_causal_stack() {
        let rows = StepRows::new(3, 3, 2).unwrap();
        let blocks = pages(&[2, 0]);
        let w = PagedStep {
            window: Some(nz32(10)),
            ..step()
        };
        assert_eq!(first_active(&w, &rows), 0);
        assert_eq!(w.span(&rows), rows.end());
        assert_eq!(
            page_table(&w, &blocks, &rows).unwrap(),
            page_table(&step(), &blocks, &rows).unwrap()
        );
        assert_eq!(mask_stack(&w, &rows), mask_stack(&step(), &rows));
    }

    /// ⭐ A PACKED RUNG STAGES ONE WINDOW PER REQUEST — window `r`'s page table, query rows
    /// and mask stack at ITS OWN tids ([`scratchy_triton_splice::paged_window_of_class`]),
    /// each request's own bytes (its blocks, its rows); the write index is packed head-major
    /// over the graph's rows — row `r·w + j` at request `r`'s cache row — so request `r`'s
    /// answer comes back from its own window's rows. Window 0's tids are the step's
    /// sentinels, and its tables are the SOLO rung's own bytes, which is what keeps a packed
    /// rung byte-identical at one request.
    #[test]
    fn a_packed_rung_stages_one_window_per_request() {
        let s = PagedStep {
            view: crate::wiring::ContextPages::new(nz32(8)),
            pages: pool(8),
            extent: pair(3, 2),
            ..step()
        };
        let (a, b) = (pages(&[2, 3]), pages(&[5, 6]));
        let (ra, rb) = (
            StepRows::new(3, 5, 2).unwrap(),
            StepRows::new(3, 4, 3).unwrap(),
        );
        let src = step_sources(&seq(s), &[req(&a, ra), req(&b, rb)]).expect("stages");
        let wt = |r: u32, table: W| {
            scratchy_triton_splice::paged_window_of_class(
                crate::wiring::AttnClass::FIRST.index(),
                r,
                table,
            )
        };
        let tbl = |src: &StepSources, tid: u32| -> Vec<i32> {
            ints(
                &src.int32
                    .iter()
                    .find(|(t, ..)| *t == tid)
                    .expect("staged")
                    .1,
            )
        };
        fn mask_at(src: &StepSources, tid: u32) -> (&[f32], &[usize]) {
            let (.., m, shape) = src.masks.iter().find(|(t, ..)| *t == tid).expect("staged");
            (m, shape)
        }
        // Each request's window stages ITS OWN page table, query rows and mask stack, at its
        // own tids — request 1's query rows are its window's rows `3..6` of the graph.
        assert_eq!(
            tbl(&src, wt(0, W::PageTable)),
            page_table(&s, &a, &ra).unwrap()
        );
        assert_eq!(
            tbl(&src, wt(1, W::PageTable)),
            page_table(&s, &b, &rb).unwrap()
        );
        assert_eq!(tbl(&src, wt(0, W::QueryRows)), (0..3).collect::<Vec<i32>>());
        assert_eq!(tbl(&src, wt(1, W::QueryRows)), (3..6).collect::<Vec<i32>>());
        let (ma, sa) = mask_at(&src, wt(0, W::Mask));
        assert_eq!(sa, &[4 * 3, 4], "one window's rows, the walk's trips");
        assert_eq!(ma, mask_stack_of(&s, 3, &ra).as_slice());
        assert_eq!(
            mask_at(&src, wt(1, W::Mask)).0,
            mask_stack_of(&s, 3, &rb).as_slice()
        );
        // Window 0's three tables are the SOLO rung's own bytes — the identity at one
        // request, on the same request's launch.
        let alone = step_sources(&seq(step()), &[req(&a, ra)]).expect("stages");
        assert_eq!(
            tbl(&src, wt(0, W::PageTable)),
            tbl(&alone, wt(0, W::PageTable))
        );
        assert_eq!(
            tbl(&src, wt(0, W::QueryRows)),
            tbl(&alone, wt(0, W::QueryRows))
        );
        assert_eq!(
            mask_at(&src, wt(0, W::Mask)).0,
            mask_at(&alone, wt(0, W::Mask)).0
        );
        // The write index: head-major, each head's rows the requests' in order —
        // `nkvh · requests · width` entries, row `r·w + j` at request `r`'s cache row.
        let w = tbl(&src, s.write_rows());
        assert_eq!(w.len(), 2 * 2 * 3, "nkvh · requests · width");
        let mut want: Vec<i32> = Vec::with_capacity(w.len());
        for h in 0..2 {
            for j in 0..3 {
                want.push(s.cache_row(&a, ra.pos(j), h) as i32);
            }
            for j in 0..3 {
                want.push(s.cache_row(&b, rb.pos(j), h) as i32);
            }
        }
        assert_eq!(w, want);
    }

    /// ⭐ AN EMPTY SLOT STAGES THE EMPTY-WINDOW LAWS — window `r` past the live requests
    /// stages a zero page table (page 0, the null page), a fully-[`MASKED`] stack and its
    /// window's own query rows (the graph's rows are real however many requests ride the
    /// launch), and the write index sends its rows to the null slot, so an empty window's
    /// writes reach no request's cache. A launch of no requests, or more than the rung
    /// packs, is refused by name.
    #[test]
    fn an_empty_slot_stages_the_empty_window_laws() {
        let s = PagedStep {
            view: crate::wiring::ContextPages::new(nz32(8)),
            pages: pool(8),
            extent: pair(3, 2),
            ..step()
        };
        let a = pages(&[2, 3]);
        let ra = StepRows::new(3, 5, 2).unwrap();
        let src = step_sources(&seq(s), &[req(&a, ra)]).expect("stages");
        let wt = |r: u32, table: W| {
            scratchy_triton_splice::paged_window_of_class(
                crate::wiring::AttnClass::FIRST.index(),
                r,
                table,
            )
        };
        let tbl = |tid: u32| -> Vec<i32> {
            ints(
                &src.int32
                    .iter()
                    .find(|(t, ..)| *t == tid)
                    .expect("staged")
                    .1,
            )
        };
        assert_eq!(tbl(wt(1, W::PageTable)), vec![0; 4 * INT32_ELEMS_PER_STICK]);
        assert_eq!(tbl(wt(1, W::QueryRows)), (3..6).collect::<Vec<i32>>());
        let (.., m, shape) = src
            .masks
            .iter()
            .find(|(t, ..)| *t == wt(1, W::Mask))
            .expect("staged");
        assert_eq!(shape, &[4 * 3, 4]);
        assert!(
            m.iter().all(|&v| v == MASKED),
            "the empty window is fully masked"
        );
        // The write index: the live request's rows at its cache rows, the empty slot's at
        // the null slot's own.
        let w = tbl(s.write_rows());
        for h in 0..2 {
            for j in 0..3 {
                assert_eq!(w[h * 6 + j], s.cache_row(&a, ra.pos(j), h) as i32);
                assert_eq!(w[h * 6 + 3 + j], s.null_cache_row(h) as i32);
            }
        }
        // A launch the rung cannot pack is refused by name.
        assert!(step_sources(&seq(s), &[]).is_err());
        let (b, c, d) = (pages(&[1]), pages(&[2]), pages(&[3]));
        let rb = StepRows::new(3, 0, 1).unwrap();
        assert!(
            step_sources(
                &seq(s),
                &[req(&a, ra), req(&b, rb), req(&c, rb), req(&d, rb)]
            )
            .is_err()
        );
    }

    /// ⭐ A WINDOWED STEP'S ADMISSION IS ITS SPAN, NOT ITS END — a step whose end is far past a
    /// causal bundle's reach runs on it when its window's span fits, because the walk anchors at
    /// the earliest query's reach however long the context is. The causal refusal (`end` past the
    /// reach) is the same law at `window: None`.
    #[test]
    fn a_windowed_step_runs_on_a_rung_by_its_span_not_its_end() {
        // An 8-page pool of 4-slot pages, a 4-block walk (16 positions of reach).
        let base = PagedStep {
            view: crate::wiring::ContextPages::new(nz32(8)),
            pages: pool(8),
            window: Some(nz32(6)),
            ..step()
        };
        let blocks = pages(&[1, 2, 3, 4, 5]);
        // End 17 (3 rows from 14): the causal step is past the reach and refuses; the W-6 step's
        // span is 17 − 8 = 9 and runs.
        let rows = StepRows::new(3, 14, 3).unwrap();
        let causal = PagedStep {
            view: crate::wiring::ContextPages::new(nz32(8)),
            pages: pool(8),
            ..step()
        };
        assert!(step_sources(&seq(causal), &[req(&blocks, rows)]).is_err());
        assert!(step_sources(&seq(base), &[req(&blocks, rows)]).is_ok());
        // The same windowed step on a 2-block walk (8 positions) refuses: its span does not fit.
        let narrow = PagedStep {
            walk: wb(2),
            ..base
        };
        assert!(step_sources(&seq(narrow), &[req(&blocks, rows)]).is_err());
    }

    /// ⭐ A WINDOWED BODY WALKS ITS OWN BOUND, NOT THE RUNG'S SWEEP — `staged_walk` names the same walk
    /// its body baked, however wide the ladder swept to reach the context: the whole pool's sweep
    /// caps at the window's bound, a rung narrower than the bound keeps its own sweep, and a
    /// causal body walks everything it sweeps. The bound reads the launch's own per-request
    /// width — one row for a decode or batch launch, the chunk's width packed — and the step's
    /// OWN page size, the units its walk is counted in.
    #[test]
    fn a_windowed_body_walks_its_bound_not_the_rung_s_sweep() {
        // An 8-page pool of 4-slot pages, W 6: the bound is ceil(6/4) + 1 = 3 blocks at one
        // query row, the same 3 at a batch (one row per request), and ceil((6 + 3 - 1)/4) + 1 = 3
        // at a 3-row packed chunk.
        let windowed = PagedStep {
            view: crate::wiring::ContextPages::new(nz32(8)),
            pages: pool(8),
            window: Some(nz32(6)),
            ..step()
        };
        let causal = PagedStep {
            view: crate::wiring::ContextPages::new(nz32(8)),
            pages: pool(8),
            ..step()
        };
        // One row of one request — the step at the pair a one-row launch is.
        let one_row = |s: PagedStep| PagedStep {
            extent: pair(1, 1),
            ..s
        };
        // The whole pool swept (32 slots): the causal body walks all 8 blocks, the windowed
        // body its 3.
        assert_eq!(one_row(causal).staged_walk(32), Some(wb(8)));
        assert_eq!(one_row(windowed).staged_walk(32), Some(wb(3)));
        // A batch: one row per request, so the bound is the same 3 blocks.
        let batch = PagedStep {
            extent: pair(1, 4),
            paged_rows: PagedRows::Batch,
            ..windowed
        };
        assert_eq!(batch.staged_walk(32), Some(wb(3)));
        // A rung narrower than the bound keeps its own sweep: one block swept, one walked.
        assert_eq!(one_row(windowed).staged_walk(4), Some(wb(1)));
        // A packed chunk of 3 rows: the wider query pushes the earliest window start back, so
        // the bound is ceil((6 + 3 - 1)/4) + 1 = 3 blocks.
        let packed = PagedStep {
            extent: pair(3, 2),
            ..windowed
        };
        assert_eq!(packed.staged_walk(32), Some(wb(3)));
    }

    /// ⭐ A WINDOWED DECODE BATCH ANCHORS EACH SEQUENCE AT ITS OWN REACH — every sequence's chunk
    /// pages and mask rows are keyed by its own window's blocks, so one sequence's long context
    /// never widens another's staging; a sequence whose span is past the walk refuses.
    #[test]
    fn a_windowed_decode_batch_anchors_each_sequence_at_its_own_reach() {
        // `batch()`: a 16-page pool of 4-slot pages, a 2-block walk, J 2 · C 1 at B 4.
        let s = PagedStep {
            window: Some(nz32(6)),
            ..batch()
        };
        let a = pages(&[1, 2, 3]);
        let b = pages(&[4]);
        // Position 9 touches 2 blocks from anchor 1 (it held 3 under the causal law, past this
        // walk); position 2 touches its one block from anchor 0.
        let seqs = [
            BatchSeq { blocks: &a, pos: 9 },
            BatchSeq { blocks: &b, pos: 2 },
        ];
        let src = batch_sources(&bat(s), &seqs).unwrap();
        let words = |t: u32| -> Vec<i32> {
            ints(
                &src.int32
                    .iter()
                    .find(|(tid, ..)| *tid == t)
                    .expect("staged")
                    .1,
            )
        };
        // Block slot i, sequence s: A's are its blocks 1 and 2 (anchored at 1), B's its block 0
        // then the null page past its one touched block.
        assert_eq!(
            words(s.chunk_pages()),
            vec![2, 4, 0, 0, 3, 0, 0, 0],
            "each sequence's own anchored blocks down the block-slot axis"
        );
        let (mask, shape) = (&src.masks[0].1, &src.masks[0].2);
        assert_eq!(shape, &vec![8usize, 4]);
        let tile = |i: usize, s: usize| -> &[f32] { &mask[(i * 4 + s) * 4..(i * 4 + s + 1) * 4] };
        // A (position 9, reach [4, 9]): slot 0 covers 4..=7 whole, slot 1 (8..=11) to 9.
        assert_eq!(tile(0, 0), &[0.0; 4]);
        assert_eq!(tile(1, 0), &[0.0, 0.0, MASKED, MASKED]);
        // B (position 2, reach [0, 2]): slot 0 to position 2; slot 1 untouched — the stack's own
        // initial -inf, the padding value a block slot past every touched one carries.
        assert_eq!(tile(0, 1), &[0.0, 0.0, 0.0, MASKED]);
        assert_eq!(tile(1, 1), &[f32::NEG_INFINITY; 4]);
        // A sequence whose span is past the walk refuses: position 12 touches 3 blocks.
        assert!(
            batch_sources(
                &bat(s),
                &[BatchSeq {
                    blocks: &a,
                    pos: 12
                }]
            )
            .is_err()
        );
    }

    /// spyre-inference's dispatch: the decode kernel, at one query row, is the one that reads
    /// `kv_row_pool`, so a decode step stages it and a prompt chunk does not.
    #[test]
    fn only_a_decode_step_stages_kv_row_pool() {
        let blocks = pages(&[2, 0]);
        let tids = |s: &PagedStep, rows: StepRows| -> Vec<u32> {
            step_sources(&seq(*s), &[req(&blocks, rows)])
                .unwrap()
                .int32
                .iter()
                .map(|(t, ..)| *t)
                .collect()
        };
        let chunk = step();
        let staged = tids(&chunk, StepRows::new(3, 5, 2).unwrap());
        assert!(!staged.contains(&chunk.kv_row_pool()) && !staged.contains(&chunk.int32_zero()));
        let decode = PagedStep {
            extent: pair(1, 1),
            ..step()
        };
        let srcs = step_sources(
            &seq(decode),
            &[req(&blocks, StepRows::new(1, 5, 1).unwrap())],
        )
        .unwrap();
        assert!(srcs.int32.iter().any(|(t, ..)| *t == decode.kv_row_pool()));
        let (_, zero, _) = srcs
            .int32
            .iter()
            .find(|(t, ..)| *t == decode.int32_zero())
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
        let rows = StepRows::new(3, 5, 2).unwrap();
        let blocks = pages(&[2, 3]);
        let card = card_step_sources(
            &seq(s),
            &[RequestStep {
                blocks: &blocks,
                rows,
            }],
        )
        .unwrap();
        let pitch = s.card_walk_step() as usize;
        assert_eq!(pitch, 3 * 128, "one fp16 stick per score row, three rows");
        let table = ints(card.words(s.page_table()).unwrap());
        assert_eq!(table.len(), s.walk.get() as usize * pitch / 4);
        for (i, &v) in table.iter().enumerate() {
            let want = match i * 4 % pitch {
                0 => [2, 3, 0, 0][i * 4 / pitch],
                _ => 0,
            };
            assert_eq!(v, want, "word {i}");
        }
        assert_eq!(
            ints(card.words(s.write_rows()).unwrap()),
            write_rows(&s, &blocks, &rows).unwrap()
        );
        assert_eq!(ints(card.words(s.query_rows()).unwrap()), query_rows(&s));
        assert_eq!(card.masks, vec![(s.mask(), mask_stack(&s, &rows))]);
        assert!(
            card.words(s.kv_row_pool()).is_none() && card.words(s.int32_zero()).is_none(),
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
        let rows = StepRows::new(3, 5, 2).unwrap();
        let blocks = pages(&[2, 3]);
        let logical = mask_stack(&s, &rows);
        let card = card_step_sources(
            &seq(s),
            &[RequestStep {
                blocks: &blocks,
                rows,
            }],
        )
        .unwrap();
        let (n, w) = ((s.walk.get() * s.rows()) as usize, s.page_slots as usize);
        let mask = &card.masks[0].1;
        assert_eq!(mask.len(), logical.len());
        for r in 0..n {
            for c in 0..w {
                assert_eq!(
                    mask[(c / 64) * (n * 64) + r * 64 + c % 64],
                    logical[r * w + c],
                    "row {r} slot {c}"
                );
            }
        }
        assert_ne!(mask, &logical, "two sticks a row: not the row-major bytes");
        let b = PagedStep {
            page_slots: 128,
            ..batch()
        };
        let seqs = [BatchSeq {
            blocks: &pages(&[2, 3]),
            pos: 133,
        }];
        let (_, want, shape) = &batch_sources(&bat(b), &seqs).unwrap().masks[0];
        let got = &card_batch_sources(&bat(b), &seqs).unwrap().masks[0].1;
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
            walk: wb(16),
            ..batch()
        };
        let c = s.batch().unwrap();
        assert_eq!((c.blocks_per_chunk, c.num_chunks), (8, 2));
        let blocks = pages(&(0..16).rev().collect::<Vec<u32>>());
        let seqs = [BatchSeq {
            blocks: &blocks,
            pos: 61,
        }];
        let logical = batch_sources(&bat(s), &seqs).unwrap();
        let card = card_batch_sources(&bat(s), &seqs).unwrap();
        let (_, want, _) = logical
            .int32
            .iter()
            .find(|(t, ..)| *t == s.chunk_pages())
            .unwrap();
        let (row, pitch) = (c.index_rows() as usize, s.card_walk_step() as usize / 4);
        let got = ints(card.words(s.chunk_pages()).unwrap());
        assert_eq!(got.len(), c.num_chunks as usize * pitch);
        for (k, chunk) in ints(want).chunks(row).enumerate() {
            assert_eq!(&got[k * pitch..k * pitch + row], chunk, "chunk {k}");
            assert!(
                got[k * pitch + row..(k + 1) * pitch]
                    .iter()
                    .all(|&v| v == 0)
            );
        }
        for t in [s.rep_rows(), s.write_rows()] {
            let (_, w, _) = logical.int32.iter().find(|(x, ..)| *x == t).unwrap();
            assert_eq!(card.words(t).unwrap(), w.as_slice());
        }
        assert_eq!(card.masks[0].1, logical.masks[0].1);
    }

    /// The pool's statics: page `p`'s `nkvh` folded-cache rows in the first words of its run, and
    /// one run of int32 zeros.
    #[test]
    fn the_pool_statics_hold_each_pages_rows_at_the_run_pitch() {
        let run = ktir_superdsc::emit::lower_ktir_to_superdsc::INDEX_RUN_WORDS as usize;
        let decode = PagedStep {
            extent: pair(1, 1),
            ..step()
        };
        let statics = card_statics(&decode).unwrap();
        let pool = ints(
            &statics
                .iter()
                .find(|(t, _)| *t == decode.kv_row_pool())
                .unwrap()
                .1,
        );
        assert_eq!(pool.len(), decode.pages.get() as usize * run);
        for p in 0..decode.pages.get() as usize {
            assert_eq!(
                &pool[p * run..p * run + 2],
                &[2 * p as i32, 2 * p as i32 + 1]
            );
            assert!(pool[p * run + 2..(p + 1) * run].iter().all(|&v| v == 0));
        }
        let zero = &statics
            .iter()
            .find(|(t, _)| *t == decode.int32_zero())
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
            for (mask_tid, mask) in &card.masks {
                assert!(
                    mask.len() as u64 * 2 <= size(*mask_tid, crate::wiring::CardSeat::Step),
                    "t{mask_tid}"
                );
            }
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
                extent: pair(rows, 1),
                ..step()
            };
            for walk in [1, 2, 4] {
                let body = full.at_walk(full.staged_walk(walk * full.page_slots).unwrap());
                let st = StepRows::new(rows as usize, 0, real).unwrap();
                let blocks = pages(&[3, 2, 1, 0]);
                let card = card_step_sources(
                    &seq(body),
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
            walk: wb(16),
            ..batch()
        };
        for walk in [1, 2, 3, 8, 16] {
            let body = full.at_walk(full.staged_walk(walk * full.page_slots).unwrap());
            let blocks = pages(&(0..walk).collect::<Vec<u32>>());
            let seqs = [BatchSeq {
                blocks: &blocks,
                pos: walk as usize * 4 - 1,
            }];
            fits(full, &card_batch_sources(&bat(body), &seqs).unwrap(), &[]);
        }
    }

    /// ⭐ THE WALK COVERS THE CONTEXT, NEVER THE POOL — a pool of 16 pages over a 4-page context
    /// (a 16-position model on 4-slot pages): the walk is the context's 4 pages, so the
    /// per-request page table and mask stack regions come out 4 pages wide, while `kv_row_pool`
    /// alone stays pool-sized — it is indexed by page, never by walked block.
    #[test]
    fn the_walk_covers_the_context_not_the_pool() {
        let s = PagedStep {
            pages: pool(16),
            walk: wb(4),
            extent: pair(1, 1),
            ..step()
        };
        // The walk's one law, stated by [`walk_pages`] at the scheduler's own page: the context's
        // 4 pages, whatever the pool holds.
        assert_eq!(
            walk_pages(4, 1, 1, ktir_superdsc::ktir_node::ActiveCap::FULL, None),
            4
        );
        let placed = |tid: u32| {
            s.card_placements()
                .iter()
                .find(|(t, ..)| *t == tid)
                .map(|&(.., bytes)| bytes)
                .unwrap()
        };
        assert_eq!(
            placed(s.page_table()),
            4 * s.card_walk_step(),
            "the page table is the walk's 4 pages, one trip each — not the pool's 16"
        );
        assert_eq!(
            placed(s.mask()),
            u64::from(s.walk.get())
                * u64::from(s.extent.per_request().get())
                * u64::from(s.page_slots)
                * 2,
            "the mask stack is the walk's 4 pages of one request's row — not the pool's 16"
        );
        assert_eq!(
            placed(s.kv_row_pool()),
            16 * ktir_superdsc::emit::lower_ktir_to_superdsc::INDEX_RUN_WORDS as u64 * 4,
            "kv_row_pool is the pool's 16 pages, not the walk's 4"
        );
    }

    /// ⭐ A PACKED STEP STAGES EACH REQUEST AT ITS OWN WINDOW — the S=2 shape the packed prefill
    /// rung bakes: request `r`'s page table, mask stack and query rows at window `r`'s own tids
    /// ([`ktir_superdsc::reserved_tids::paged_window_tid`], window 0's being the step's own
    /// sentinels), each request's block by the ONE-REQUEST laws (its own blocks, its own
    /// positions, its own consecutive window `r·w..r·w+w`), and the write rows PACKED over the
    /// graph's rows — every row naming its OWN request's cache slot.
    #[test]
    fn a_packed_step_stages_each_request_at_its_own_window() {
        // Two requests of 3 rows over a 6-row graph, the pool and walk of `step()`.
        let s = PagedStep {
            extent: pair(3, 2),
            ..step()
        };
        let (trip, slots, nkvh) = (s.card_walk_step() as usize, 4, 2);
        assert_eq!(trip, 3 * 128);
        let wtid = |r: u32, table: W| ktir_superdsc::reserved_tids::paged_window_tid(r, table);
        let a = pages(&[2, 3]);
        let b = pages(&[1, 0, 3]);
        let ra = StepRows::new(3, 5, 2).unwrap();
        let rb = StepRows::new(3, 1, 3).unwrap();
        let card = card_step_sources(
            &seq(s),
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
        // THE PAGE TABLES — one per window, request `r`'s block `i` at `i·trip`, zero past its
        // own blocks.
        for (r, blocks) in [(0u32, &a[..]), (1, &b[..])] {
            let table = ints(card.words(wtid(r, W::PageTable)).unwrap());
            assert_eq!(table.len(), 4 * trip / 4, "window {r}'s table");
            for i in 0..4 {
                let want = blocks.get(i).copied().unwrap_or(PoolPage::NULL).get() as i32;
                assert_eq!(table[i * trip / 4], want, "request {r} block {i}");
            }
        }
        // THE MASKS — one per window, request `r`'s stack the one-request law over its own rows
        // (4-slot pages: the stick-major form is the logical one).
        for (r, rows) in [(0u32, ra), (1, rb)] {
            let want = mask_stack_of(&s, 3, &rows);
            let got = card
                .masks
                .iter()
                .find(|(t, _)| *t == wtid(r, W::Mask))
                .map(|(_, m)| m.clone())
                .unwrap();
            assert_eq!(got.len(), 4 * 3 * slots, "window {r}'s stack");
            assert_eq!(&got[..], &want[..], "request {r}");
        }
        // THE WRITE ROWS — packed `[nkvh · 6]`, head-major: request `r`'s row `j` from its own
        // blocks at its own position (a pad row repeats its request's last real one).
        let write = ints(card.words(s.write_rows()).unwrap());
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
        // THE QUERY ROWS — one per window, request `r`'s consecutive window `r·w..r·w+w`.
        assert_eq!(ints(card.words(wtid(0, W::QueryRows)).unwrap()), &[0, 1, 2]);
        assert_eq!(ints(card.words(wtid(1, W::QueryRows)).unwrap()), &[3, 4, 5]);
        // ⭐ AND EVERY SOURCE FITS ITS PLACEMENT — one region per window, each the ONE-REQUEST
        // size (the pool's whole walk), the write rows the graph's own rows.
        let size = |tid: u32| {
            s.card_placements()
                .iter()
                .find(|(t, ..)| *t == tid)
                .map(|&(.., b)| b)
                .unwrap()
        };
        for r in 0..2u32 {
            assert_eq!(
                size(wtid(r, W::PageTable)),
                4 * 3 * 128,
                "window {r}'s page table"
            );
            assert_eq!(
                size(wtid(r, W::Mask)),
                (4 * 3 * 4 * 2) as u64,
                "window {r}'s mask"
            );
            assert_eq!(size(wtid(r, W::QueryRows)), 128, "window {r}'s query rows");
        }
        assert_eq!(
            size(s.write_rows()),
            128,
            "the write rows stay the graph's own rows, one region for the whole graph"
        );
        for (tid, words) in &card.int32 {
            assert!(words.len() as u64 <= size(*tid), "t{tid}");
        }
        for (tid, mask) in &card.masks {
            assert!(mask.len() as u64 * 2 <= size(*tid), "t{tid}");
        }
    }

    /// ⭐ AN EMPTY SLOT IS NO REQUEST'S — a launch seating fewer requests than the rung's capacity
    /// stages the windows past its live ones to the null page, a fully-`MASKED` stack and
    /// [`NULL_SLOT`] writes: the empty window's forward runs, but nothing it writes reaches a
    /// request's cache and nothing it answers is read back. The live window's own tables are
    /// exactly the one-request laws' — nothing moves at the front.
    #[test]
    fn an_empty_slot_is_no_request_s() {
        let s = PagedStep {
            extent: pair(3, 2),
            ..step()
        };
        let trip = s.card_walk_step() as usize;
        let wtid = |r: u32, table: W| ktir_superdsc::reserved_tids::paged_window_tid(r, table);
        let a = pages(&[2, 3]);
        let ra = StepRows::new(3, 5, 2).unwrap();
        let card = card_step_sources(
            &seq(s),
            &[RequestStep {
                blocks: &a,
                rows: ra,
            }],
        )
        .unwrap();
        // The empty window's page table rows are zero — page 0, the null page.
        let table = ints(card.words(wtid(1, W::PageTable)).unwrap());
        assert_eq!(table.len(), 4 * trip / 4);
        for i in 0..4 {
            assert_eq!(table[i * trip / 4], 0, "empty window block {i}");
        }
        // Its mask stack is fully `MASKED` — the finite all-masked tile.
        let empty_mask = &card
            .masks
            .iter()
            .find(|(t, _)| *t == wtid(1, W::Mask))
            .map(|(_, m)| m.clone())
            .unwrap();
        assert_eq!(empty_mask.len(), 4 * 3 * 4);
        assert!(
            empty_mask.iter().all(|&v| v == MASKED),
            "the empty window's stack is fully masked"
        );
        // Its query rows are its window's own — the graph's rows are real however many requests
        // ride the launch.
        assert_eq!(ints(card.words(wtid(1, W::QueryRows)).unwrap()), &[3, 4, 5]);
        // Its write rows name the null slot — page 0, slot 0, head-major.
        let write = ints(card.words(s.write_rows()).unwrap());
        for h in 0..2 {
            let at = h * 6 + 3;
            assert_eq!(&write[at..at + 3], &[h as i32 * 4; 3], "head {h}");
        }
        // And the live window's tables are the one-request staging's own bytes, at the tids they
        // have always staged at (window 0's ARE the step's sentinels).
        let one = PagedStep {
            extent: pair(3, 1),
            ..step()
        };
        let solo = card_step_sources(
            &seq(one),
            &[RequestStep {
                blocks: &a,
                rows: ra,
            }],
        )
        .unwrap();
        assert_eq!(
            card.words(s.page_table()).unwrap(),
            solo.words(one.page_table()).unwrap(),
            "the live window's page table is the one-request law's"
        );
        assert_eq!(
            &card.masks[0].1, &solo.masks[0].1,
            "the live window's mask is the one-request law's"
        );
        assert_eq!(
            card.words(s.query_rows()).unwrap(),
            solo.words(one.query_rows()).unwrap(),
            "the live window's query rows are the one-request law's"
        );
        // A launch wider than the capacity refuses; so does an empty one. (Two requests in a
        // two-request rung is the FULL launch, not a wide one.)
        let three: Vec<RequestStep<'_>> = (0..3)
            .map(|_| RequestStep {
                blocks: &a,
                rows: ra,
            })
            .collect();
        assert!(card_step_sources(&seq(s), &three).is_err());
        assert!(card_step_sources(&seq(s), &[]).is_err());
        // A request wider than the per-request window DOES NOT EXIST — the mint refuses it, so
        // no staging can name one (the runtime check `check` used to make, now the type's fact).
        assert!(
            StepRows::new(one.extent.per_request().get() as usize, 0, 4).is_none(),
            "a step wider than its chunk does not exist"
        );
    }

    /// ⭐ AT ONE REQUEST NOTHING MOVES — window 0's tids ARE the step's own sentinels, so every
    /// placement is the per-request one at the tid it has always had, and the three tables stay
    /// contiguous and LAST in the activation segment (write rows first) — the layout every rung
    /// today bakes, which is why the gate's bundle pairing against main must come back identical.
    /// A packed rung's placements are the same rows one per window, in window order.
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
                (one.write_rows(), crate::wiring::CardSeat::Step),
                (one.page_table(), crate::wiring::CardSeat::Step),
                (one.mask(), crate::wiring::CardSeat::Step),
                (one.query_rows(), crate::wiring::CardSeat::Step),
            ],
            "write rows first, then the three tables contiguous"
        );
        let packed = PagedStep {
            extent: pair(3, 2),
            ..step()
        };
        let wtid = |r: u32, table: W| ktir_superdsc::reserved_tids::paged_window_tid(r, table);
        let own = |s: &PagedStep, tid: u32| {
            s.card_placements()
                .iter()
                .find(|(t, ..)| *t == tid)
                .map(|&(.., b)| b)
                .unwrap()
        };
        // The packed rung's rows: window 0's three, then window 1's three, each exactly the
        // one-request size — no spare, no pitch.
        assert_eq!(
            packed
                .card_placements()
                .iter()
                .map(|&(t, _, _)| t)
                .collect::<Vec<_>>(),
            vec![
                packed.write_rows(),
                wtid(0, W::PageTable),
                wtid(0, W::Mask),
                wtid(0, W::QueryRows),
                wtid(1, W::PageTable),
                wtid(1, W::Mask),
                wtid(1, W::QueryRows),
            ],
            "write rows first, then the windows in order"
        );
        for r in 0..2u32 {
            for table in [W::PageTable, W::Mask, W::QueryRows] {
                assert_eq!(
                    own(&packed, wtid(r, table)),
                    own(&one, wtid(0, table)),
                    "window {r} table {table:?}'s region"
                );
            }
        }
        assert_eq!(
            own(&packed, packed.write_rows()),
            own(&one, one.write_rows())
        );
    }

    /// ⭐ A LAUNCH'S STEP STATES THE LAUNCH'S OWN EXTENT, AT ITS BODY'S BAKED WALK — a `Sequence`
    /// launch is one request of its rung's `width` rows — `(width, 1)` — whatever width the
    /// wiring's own pair names (the ladders bake one tape at several widths under one wiring, and
    /// the worker's chunk ladder stages a narrow chunk under the shared wiring every prompt), and
    /// a `Batch` launch is one row per request — `(1, width)`. Total: the walk is the rung's
    /// record (`RungBody::walk`) and the pair is the width's.
    #[test]
    fn a_launch_steps_at_its_own_extent_and_its_bodys_walk() {
        let w = |n: u32| crate::wiring::RungWidth::of_baked_rows(n).expect("a nonzero width");
        let pair_of = |s: PagedStep| (s.extent.per_request().get(), s.extent.requests().get());
        let body = step().on_sequence_body(w(3), wb(2));
        assert_eq!((body.rows(), body.walk.get(), body.reach()), (3, 2, 8));
        assert_eq!(
            pair_of(*body),
            (3, 1),
            "a launch on the wiring's own width re-derives its pair"
        );
        // ⭐ THE WORKER'S REAL SHAPE: a chunk narrower than the wiring's representative width —
        // the prefill ladder stages a 4-row chunk under a 3-per-request wiring.
        let wide = PagedStep {
            extent: pair(3, 2),
            ..step()
        };
        assert_eq!(
            pair_of(*wide.on_sequence_body(w(4), wb(2))),
            (4, 1),
            "a chunk narrower than the wiring's width still states its own rows"
        );
        let decode = PagedStep {
            extent: pair(1, 1),
            ..step()
        };
        let body = decode.on_batch_body(w(4), wb(2));
        assert_eq!(
            pair_of(*body),
            (1, 4),
            "a batch launch is one row per request"
        );
        assert_eq!(body.paged_rows, PagedRows::Batch);
    }

    /// ⭐ THE PACKED LAUNCH'S EXTENT — the wiring's OWN stated pair, kept verbatim: a packed
    /// rung's wiring is not a representative of a tape baked at several widths (the bake
    /// minted `(w, s)` for that rung), and a wave launch IS that pair, so nothing but the walk —
    /// the body's own record — is the launch's.
    #[test]
    fn a_packed_launch_keeps_the_wirings_own_pair() {
        let packed = PagedStep {
            extent: pair(3, 2),
            ..step()
        };
        let body = packed.on_body_packed(wb(2));
        assert_eq!((body.rows(), body.walk.get(), body.reach()), (6, 2, 8));
        assert_eq!(
            (
                body.extent.per_request().get(),
                body.extent.requests().get()
            ),
            (3, 2),
            "the wiring's own pair, verbatim — the launch IS the pair"
        );
    }

    #[test]
    fn a_step_past_the_request_s_blocks_is_refused() {
        let s = step();
        assert!(page_table(&s, &pages(&[2]), &StepRows::new(3, 3, 2).unwrap()).is_err());
        assert!(write_rows(&s, &pages(&[2, 9]), &StepRows::new(3, 0, 1).unwrap()).is_err());
    }

    /// A bundle whose attention walks two of the pool's four blocks serves a chunk over a cached
    /// prefix while both sit inside those blocks, and refuses the step that runs past them.
    #[test]
    fn a_step_past_the_bundle_s_walk_is_refused() {
        let s = PagedStep {
            walk: wb(2),
            ..step()
        };
        assert_eq!(s.reach(), 8);
        let blocks = pages(&[3, 1, 0]);
        assert!(page_table(&s, &blocks, &StepRows::new(3, 5, 3).unwrap()).is_ok());
        let past = StepRows::new(3, 6, 3).unwrap();
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
        let pool = pool_of_rows(&s, hd, &pages(&[3, 1]), &cache, 5).unwrap();
        // Token 4 is block 1 (page 1) slot 0: head 1 at pool row (1·2 + 1)·4 + 0 = 12.
        assert_eq!(&pool[12 * hd..13 * hd], &[42.0, 43.0]);
        // Token 0 is page 3 slot 0, head 0 at pool row 24.
        assert_eq!(&pool[24 * hd..25 * hd], &[0.0, 1.0]);
    }
}
