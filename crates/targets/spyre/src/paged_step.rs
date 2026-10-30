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
//! ONE statement with two readers — the worker every step, and the bake's numeric gate once per
//! gated bundle — so the gate checks exactly the bytes the worker stages.

use crate::wiring::PagedStep;

/// The page walk's laws, where the bake and the worker that state a bundle's walk reach them: the
/// page size, a rung's walk ([`walk_pages`]) and the rungs a pool is walked by ([`walk_buckets`]).
pub use scratchy_triton_splice::{PAGE_SLOTS, walk_buckets, walk_pages};

/// `finfo(float16).min` — spyre-inference's masked additive value: finite, so a fully-masked tile's
/// row maximum stays finite and its weights vanish under the running maximum.
pub const MASKED: f32 = -65504.0;

/// The rows of one forward: `real` tokens at positions `start..start + real`, in a bundle
/// [`PagedStep::rows`] wide. A row past the real ones repeats the last real token
/// ([`scratchy_subtile::sdsc_abstract::ChunkRows::row_logical_pos`]'s law), so its forward, its
/// mask and its KV write are the last real token's.
#[derive(Clone, Copy, Debug)]
pub struct StepRows {
    pub start: usize,
    pub real: usize,
}

impl StepRows {
    /// The sequence position row `r` carries.
    pub fn pos(&self, r: usize) -> usize {
        self.start + r.min(self.real.max(1) - 1)
    }

    /// One past the last position this step reaches — the KV length after its write.
    pub fn end(&self) -> usize {
        self.start + self.real
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
/// and every block is a page of the pool.
fn check(step: &PagedStep, blocks: &[u32], rows: &StepRows) -> Result<(), PagedStepError> {
    let slots = step.page_slots as usize;
    let need = rows.end().div_ceil(slots);
    if rows.real == 0 || rows.real > step.rows as usize {
        return Err(PagedStepError(format!(
            "a step of {} real row(s) in a {}-row bundle",
            rows.real, step.rows
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
pub const INT32_ELEMS_PER_STICK: usize = 32;

/// The page index table: `[walk, INT32_ELEMS_PER_STICK]` int32, row `i` holding block `i`'s page at
/// column 0 — `table[:, 0] = block_table[s, :walk]`, the block table's own zeros past the request's
/// blocks.
pub fn page_table(
    step: &PagedStep,
    blocks: &[u32],
    rows: &StepRows,
) -> Result<Vec<i32>, PagedStepError> {
    check(step, blocks, rows)?;
    let mut t = vec![0i32; step.walk as usize * INT32_ELEMS_PER_STICK];
    for (i, &page) in blocks.iter().take(step.walk as usize).enumerate() {
        t[i * INT32_ELEMS_PER_STICK] = page as i32;
    }
    Ok(t)
}

/// The mask stack: `[walk, rows, page_slots]`, tile `i` column `c` keyed by position
/// `i · page_slots + c` — 0 at or before query row `m`'s position, [`MASKED`] after it.
pub fn mask_stack(step: &PagedStep, rows: &StepRows) -> Vec<f32> {
    let (slots, q) = (step.page_slots as usize, step.rows as usize);
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
    check(step, blocks, rows)?;
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

/// The query row index: the forward's `rows` query rows, in order.
pub fn query_rows(step: &PagedStep) -> Vec<i32> {
    (0..step.rows as i32).collect()
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
    }
    Ok(StepSources {
        mask: (
            step.mask,
            mask_stack(step, rows),
            vec![(step.walk * step.rows) as usize, step.page_slots as usize],
        ),
        int32,
    })
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

    fn step() -> PagedStep {
        PagedStep {
            mask: 10,
            page_table: 11,
            write_rows: 12,
            query_rows: 13,
            kv_row_pool: 14,
            pages: 4,
            page_slots: 4,
            rows: 3,
            nkvh: 2,
            walk: 4,
        }
    }

    /// A 3-row chunk at positions 5, 6 and a pad row repeating 6, over blocks `[2, 0]`.
    #[test]
    fn the_tables_follow_spyre_inference_s_laws() {
        let s = step();
        let rows = StepRows { start: 5, real: 2 };
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
        assert!(!tids(&chunk, StepRows { start: 5, real: 2 }).contains(&chunk.kv_row_pool));
        let decode = PagedStep { rows: 1, ..step() };
        assert!(tids(&decode, StepRows { start: 5, real: 1 }).contains(&decode.kv_row_pool));
    }

    #[test]
    fn a_step_past_the_request_s_blocks_is_refused() {
        let s = step();
        assert!(page_table(&s, &[2], &StepRows { start: 3, real: 2 }).is_err());
        assert!(write_rows(&s, &[2, 9], &StepRows { start: 0, real: 1 }).is_err());
    }

    /// A bundle whose attention walks two of the pool's four blocks serves a chunk over a cached
    /// prefix while both sit inside those blocks, and refuses the step that runs past them.
    #[test]
    fn a_step_past_the_bundle_s_walk_is_refused() {
        let s = PagedStep { walk: 2, ..step() };
        assert_eq!(s.reach(), 8);
        let blocks = [3u32, 1, 0];
        assert!(page_table(&s, &blocks, &StepRows { start: 5, real: 3 }).is_ok());
        let past = StepRows { start: 6, real: 3 };
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
